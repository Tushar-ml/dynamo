"""Dynamo auth proxy with an optional wallet-balance gate.

Request lifecycle (proxy is active whenever DYN_AUTH_ENABLED=true):

  1. Auth  — validate ``Authorization: Bearer <jwt>``, decode ``org_uuid``
             from claims, enforce DYN_AUTH_VALID_ORGS allowlist if set.
             Return 401 on any failure.

  2. Wallet balance gate (only when DYN_FLEXPRICE_ENABLED=true) — block
             prepaid orgs below the minimum balance (402) before proxying at
             all. See ``balance.py``.

  3. Forward to the internal Dynamo Rust HTTP service, unchanged — including
             the original Authorization header.

Usage billing is deliberately NOT done here. The Rust service on the other
end of step 3 receives the same Authorization header and independently runs
its own native FlexPrice billing (lib/llm/src/http/service/flexprice/), so
this proxy emitting its own event as well would bill every request twice.
The Rust service is the sole biller; this proxy's job is auth + the fast-fail
balance check in front of it.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from .auth import AuthError, authenticate
from .balance import BalanceChecker, BalanceStatus
from .client import FlexPriceClient
from .config import FlexPriceConfig

logger = logging.getLogger(__name__)

# Endpoints where token usage should be captured and metered
_BILLED_PATHS = frozenset(
    ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
)

# System endpoints that must stay reachable without a JWT — kube-probe,
# Prometheus/Alloy scrapers, and `GET /v1/models` clients don't carry one.
# Mirrors the Rust service's system_router (health, live, metrics, models),
# which is unauthenticated by design; this proxy sits in front of it and must
# not re-impose auth on the same paths.
_UNAUTHENTICATED_PATHS = frozenset(["/health", "/live", "/metrics", "/v1/models"])

# Hop-by-hop headers that must not be forwarded
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)

_JSON_CT = "application/json"

# Mirrors the Rust service's header priority (see `get_or_create_request_id`
# in openai.rs) so a request_id captured here lines up with the one the
# backend records for the same request, whenever the client supplies one.
_DYNAMO_REQUEST_ID_HEADER = "x-dynamo-request-id"
_X_REQUEST_ID_HEADER = "x-request-id"


def _get_or_create_request_id(request: web.Request) -> str:
    return (
        request.headers.get(_DYNAMO_REQUEST_ID_HEADER)
        or request.headers.get(_X_REQUEST_ID_HEADER)
        or str(uuid.uuid4())
    )


def _json_error(status: int, message: str) -> web.Response:
    return web.Response(
        status=status,
        content_type=_JSON_CT,
        text=json.dumps({"statusCode": status, "message": message}),
    )


class DynamoProxy:
    """Lightweight aiohttp reverse proxy providing JWT auth and optional FlexPrice metering.

    FlexPrice usage events are enqueued fire-and-forget after the response is
    written — billing adds zero latency to the request path.
    """

    def __init__(
        self,
        backend_url: str,
        config: FlexPriceConfig,
        flexprice_client: Optional[FlexPriceClient] = None,
        balance_checker: Optional[BalanceChecker] = None,
        model_name: str = "",
    ) -> None:
        self._backend = backend_url.rstrip("/")
        self._config = config
        self._client = flexprice_client  # None when DYN_FLEXPRICE_ENABLED=false
        self._balance_checker = balance_checker  # None when DYN_FLEXPRICE_ENABLED=false
        self._model_name = model_name
        self._session: Optional[ClientSession] = None

    async def start(self) -> None:
        self._session = ClientSession(
            connector=TCPConnector(ssl=False, limit=1000),
            # No total timeout — requests may stream for an extended period.
            timeout=ClientTimeout(total=None, connect=10),
        )
        if self._balance_checker:
            await self._balance_checker.start()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._balance_checker:
            await self._balance_checker.stop()

    # Main handler

    async def handle(self, request: web.Request) -> web.StreamResponse:
        # ---- 1. Authentication (skipped for system routes) -------------
        org_id = ""
        user_id = ""
        if request.path not in _UNAUTHENTICATED_PATHS:
            auth_header = request.headers.get("Authorization", "")
            try:
                auth_ctx = authenticate(
                    auth_header,
                    self._config.auth_secret_keys,
                    self._config.auth_valid_orgs or None,
                )
            except AuthError as exc:
                logger.warning("Auth failed: %s", exc)
                return _json_error(exc.status, str(exc))
            org_id = auth_ctx.org_uuid
            user_id = auth_ctx.user_uuid

            # Wallet balance gate: prepaid orgs below the configured minimum
            # balance (negative by default) are blocked; postpaid orgs bypass
            # this entirely. Only runs when FlexPrice billing is enabled.
            # check() never awaits a live FlexPrice call (see balance.py) —
            # it's cache-only-or-fail-open, so this never adds network
            # latency before the request is forwarded.
            if self._balance_checker:
                status = await self._balance_checker.check(org_id)
                if status is BalanceStatus.SUSPENDED:
                    return _json_error(
                        402,
                        "Your account has been suspended. Please contact "
                        "support to re-activate your account.",
                    )
                if status is BalanceStatus.INSUFFICIENT_BALANCE:
                    return _json_error(
                        402,
                        "You have exhausted your wallet balance. Please add "
                        "credits to resume using.",
                    )

        # ---- 2. Forward to Dynamo Rust service ------------------------
        path = request.path
        qs = request.query_string
        url = f"{self._backend}{path}{'?' + qs if qs else ''}"
        request_id = _get_or_create_request_id(request)

        # Only capture usage when FlexPrice metering is enabled
        is_metered = self._client is not None and path in _BILLED_PATHS
        body = await request.read()

        model_name = self._model_name
        is_streaming_req = False
        if is_metered and body:
            try:
                req_json = json.loads(body)
                model_name = req_json.get("model") or model_name
                is_streaming_req = bool(req_json.get("stream", False))
            except Exception:
                pass

        fwd_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }

        start = time.monotonic()

        try:
            async with self._session.request(  # type: ignore[union-attr]
                method=request.method,
                url=url,
                headers=fwd_headers,
                data=body,
                allow_redirects=False,
            ) as backend_resp:
                resp_headers = {
                    k: v
                    for k, v in backend_resp.headers.items()
                    if k.lower()
                    not in (_HOP_BY_HOP | {"content-encoding", "content-length"})
                }
                is_sse = "text/event-stream" in backend_resp.headers.get(
                    "content-type", ""
                )

                if is_sse or is_streaming_req:
                    return await self._handle_stream(
                        request,
                        backend_resp,
                        resp_headers,
                        is_metered=is_metered,
                        org_id=org_id,
                        user_id=user_id,
                        request_id=request_id,
                        model_name=model_name,
                        start=start,
                    )
                else:
                    return await self._handle_buffered(
                        backend_resp,
                        resp_headers,
                        is_metered=is_metered,
                        org_id=org_id,
                        user_id=user_id,
                        request_id=request_id,
                        model_name=model_name,
                        start=start,
                    )
        except Exception as exc:
            logger.warning("Proxy error on %s: %s", path, exc)
            return _json_error(502, "Bad Gateway")

    # Streaming response

    async def _handle_stream(
        self,
        request: web.Request,
        backend_resp: Any,
        resp_headers: Dict[str, str],
        *,
        is_metered: bool,
        org_id: str,
        user_id: str,
        request_id: str,
        model_name: str,
        start: float,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=backend_resp.status, headers=resp_headers
        )
        await response.prepare(request)

        usage: Optional[Dict[str, Any]] = None
        buf = b""

        async for chunk in backend_resp.content.iter_any():
            await response.write(chunk)
            if is_metered:
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="ignore").rstrip("\r")
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            usage = _parse_usage_from_sse(data_str, usage)

        await response.write_eof()

        # Fire-and-forget after response is fully written to the client
        if is_metered and usage:
            self._emit_usage(
                org_id=org_id,
                user_id=user_id,
                request_id=request_id,
                model_name=model_name,
                usage=usage,
                elapsed=time.monotonic() - start,
                streaming=True,
            )

        return response

    # Buffered (non-streaming) response

    async def _handle_buffered(
        self,
        backend_resp: Any,
        resp_headers: Dict[str, str],
        *,
        is_metered: bool,
        org_id: str,
        user_id: str,
        request_id: str,
        model_name: str,
        start: float,
    ) -> web.Response:
        body_bytes = await backend_resp.read()

        response = web.Response(
            status=backend_resp.status,
            headers=resp_headers,
            body=body_bytes,
        )

        # Fire-and-forget after response body is ready
        if is_metered:
            usage = _extract_usage_from_json(body_bytes)
            if usage:
                self._emit_usage(
                    org_id=org_id,
                    user_id=user_id,
                    request_id=request_id,
                    model_name=model_name,
                    usage=usage,
                    elapsed=time.monotonic() - start,
                    streaming=False,
                )

        return response

    # Usage emission (enqueued — never blocks the request path)

    def _emit_usage(
        self,
        *,
        org_id: str,
        user_id: str,
        request_id: str,
        model_name: str,
        usage: Dict[str, Any],
        elapsed: float,
        streaming: bool,
    ) -> None:
        assert self._client is not None
        event_name = self._config.resolve_event_name(model_name)
        source = self._config.resolve_source_name()

        properties: Dict[str, Any] = {
            "model_id": model_name,
            "user_id": user_id,
            "request_id": request_id,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "time_taken": round(elapsed, 4),
            "streaming": streaming,
            "status": "success",
        }

        self._client.enqueue(
            event_name=event_name,
            external_customer_id=org_id,
            properties=properties,
            source=source,
        )
        logger.debug(
            "FlexPrice usage enqueued: org=%s user=%s model=%s in=%s out=%s",
            org_id,
            user_id,
            model_name,
            properties["input_tokens"],
            properties["output_tokens"],
        )


# Utility functions


def _extract_usage_from_json(body: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(body).get("usage")
    except Exception:
        return None


def _parse_usage_from_sse(
    data_str: str, current: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge ``usage`` from an SSE data payload into *current*."""
    try:
        obj = json.loads(data_str)
        usage = obj.get("usage")
        if usage and isinstance(usage, dict):
            if current:
                merged = dict(current)
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        merged[k] = merged.get(k, 0) + v
                    else:
                        merged[k] = v
                return merged
            return usage
    except Exception:
        pass
    return current


# Server entrypoint


async def run_proxy(
    host: str,
    port: int,
    backend_url: str,
    config: FlexPriceConfig,
    model_name: str = "",
) -> None:
    """Start the Dynamo auth proxy and block until cancelled.

    A FlexPriceClient (and wallet BalanceChecker) is created only when
    DYN_FLEXPRICE_ENABLED=true so that auth-only mode has zero FlexPrice
    overhead.

    This proxy forwards the client's Authorization header unchanged to the
    Dynamo Rust service, which independently re-validates it and runs its own
    native FlexPrice billing (see lib/llm/src/http/service/flexprice/). If
    this proxy *also* emitted a usage event for the same request, every
    request would be billed twice — so `flexprice_client` is deliberately
    never constructed here; the Rust service is the sole biller. The wallet
    BalanceChecker has no such double-count risk (it's a read-only gate, not
    an event emission), so it stays active here as a fast-fail check before
    proxying to the backend at all.
    """
    flexprice_client: Optional[FlexPriceClient] = None
    balance_checker: Optional[BalanceChecker] = None
    if config.enabled:
        balance_checker = BalanceChecker(
            api_host=config.api_host,
            api_key=config.api_key,
            minimum_balance=config.minimum_balance,
        )
        await balance_checker.start()

    proxy = DynamoProxy(
        backend_url=backend_url,
        config=config,
        flexprice_client=flexprice_client,
        balance_checker=balance_checker,
        model_name=model_name,
    )
    await proxy.start()

    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", proxy.handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info(
        "Dynamo proxy listening on %s:%d → %s  (auth=%s flexprice=%s)",
        host,
        port,
        backend_url,
        config.auth_enabled,
        config.enabled,
    )

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        await proxy.stop()
        if flexprice_client:
            await flexprice_client.stop()

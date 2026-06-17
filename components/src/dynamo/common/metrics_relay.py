# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async HTTP client for sending LLM metrics to the ss-agent metrics relay service.

Reads METRICS_RELAY_ADDR env var (default: disabled). When set, emits TTFT,
input throughput, and output throughput with a streaming label via fire-and-forget
background tasks so the hot path is never blocked.

Retry behaviour: up to _MAX_RETRIES additional attempts on connection errors or
5xx responses, with exponential back-off (_RETRY_DELAYS). 4xx responses are not
retried (client error — retrying would just fail again).
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_RELAY_ADDR_ENV = "METRICS_RELAY_ADDR"
_RELAY_SKIP_TLS_ENV = "METRICS_RELAY_SKIP_TLS_VERIFY"  # set to "1" to disable SSL verification
_DEPLOYMENT_SLUG_ENV = "METRICS_DEPLOYMENT_SLUG"  # explicit deployment label override
_MAX_RETRIES = 3
_RETRY_DELAYS = (0.1, 0.3, 0.9)  # seconds before each successive retry

_client: Optional["MetricsRelayClient"] = None


class MetricsRelayClient:
    def __init__(self, relay_addr: str, verify_ssl: bool = True) -> None:
        self._relay_addr = relay_addr.rstrip("/")
        self._verify_ssl = verify_ssl
        # Lazily created and reused; recreated if closed.
        self._session: Optional[object] = None  # aiohttp.ClientSession
        logger.info(
            "MetricsRelayClient created: relay_addr=%s verify_ssl=%s",
            self._relay_addr,
            self._verify_ssl,
        )

    def _get_session(self) -> object:
        import aiohttp

        if self._session is None or self._session.closed:  # type: ignore[union-attr]
            connector = aiohttp.TCPConnector(ssl=False if not self._verify_ssl else None)
            logger.debug(
                "metrics relay: creating new aiohttp.ClientSession verify_ssl=%s",
                self._verify_ssl,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _post_with_retry(self, path: str, payload: dict) -> None:
        import aiohttp

        url = f"{self._relay_addr}{path}"
        logger.info("metrics relay: posting to %s payload=%s", url, payload)

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.debug(
                    "metrics relay: retry %d/%d in %.1fs for %s",
                    attempt,
                    _MAX_RETRIES,
                    delay,
                    path,
                )
                await asyncio.sleep(delay)

            try:
                session = self._get_session()
                async with session.post(  # type: ignore[union-attr]
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ) as resp:
                    if resp.status < 500:
                        if resp.status >= 400:
                            logger.warning(
                                "metrics relay client error %d for %s (payload=%s)",
                                resp.status,
                                path,
                                payload,
                            )
                        else:
                            logger.info(
                                "metrics relay: posted %s → HTTP %d",
                                payload.get("metric_type"),
                                resp.status,
                            )
                        return
                    # 5xx server error — fall through to retry
                    logger.warning(
                        "metrics relay server error %d (attempt %d/%d) for %s",
                        resp.status,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        path,
                    )
            except Exception as exc:
                logger.warning(
                    "metrics relay post failed (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )

        logger.warning("metrics relay gave up after %d attempts for %s", _MAX_RETRIES + 1, path)

    def capture_generic_metric(
        self,
        metric_type: str,
        deployment: str,
        value: int,
        streaming: bool,
    ) -> None:
        """Schedule a background task to send one metric sample; never blocks."""
        payload = {
            "metric_type": metric_type,
            "deployment": deployment,
            "key": f"dynamo_frontend_{metric_type}",
            "value": value,
            "metadata": {"streaming": streaming},
        }
        logger.debug(
            "metrics relay: scheduling metric metric_type=%s deployment=%s value=%d streaming=%s",
            metric_type,
            deployment,
            value,
            streaming,
        )
        try:
            asyncio.get_running_loop().create_task(
                self._post_with_retry("/custom-metric", payload)
            )
        except RuntimeError:
            logger.warning("metrics relay: no running event loop, metric dropped: %s", payload)


def resolve_deployment(request_model: Optional[str]) -> str:
    """Return the deployment label for metrics.

    Priority:
      1. model field from the request (set by the caller)
      2. METRICS_DEPLOYMENT_SLUG env var (explicit Helm/env override)
      3. NAMESPACE env var (Kubernetes downward API)
      4. "unknown" as last resort
    """
    if request_model:
        return request_model
    slug = os.environ.get(_DEPLOYMENT_SLUG_ENV, "").strip()
    if slug:
        return slug
    ns = os.environ.get("NAMESPACE", "").strip()
    if ns:
        return ns
    return "unknown"


def get_metrics_relay_client() -> Optional[MetricsRelayClient]:
    """Return the singleton client, or None if METRICS_RELAY_ADDR is not set."""
    global _client
    if _client is not None:
        return _client
    addr = os.environ.get(_RELAY_ADDR_ENV)
    if not addr:
        logger.info(
            "metrics relay disabled: %s env var not set, no metrics will be emitted",
            _RELAY_ADDR_ENV,
        )
        return None
    verify_ssl = os.environ.get(_RELAY_SKIP_TLS_ENV, "").strip() not in ("1", "true", "yes")
    _client = MetricsRelayClient(addr, verify_ssl=verify_ssl)
    return _client

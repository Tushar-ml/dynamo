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
_MAX_RETRIES = 3
_RETRY_DELAYS = (0.1, 0.3, 0.9)  # seconds before each successive retry

_client: Optional["MetricsRelayClient"] = None


class MetricsRelayClient:
    def __init__(self, relay_addr: str) -> None:
        self._relay_addr = relay_addr.rstrip("/")
        # Lazily created and reused; recreated if closed.
        self._session: Optional[object] = None  # aiohttp.ClientSession
        logger.info("MetricsRelayClient created: relay_addr=%s", self._relay_addr)

    def _get_session(self) -> object:
        import aiohttp

        if self._session is None or self._session.closed:  # type: ignore[union-attr]
            logger.debug("metrics relay: creating new aiohttp.ClientSession")
            self._session = aiohttp.ClientSession()
        return self._session

    async def _post_with_retry(self, path: str, payload: dict) -> None:
        import aiohttp

        url = f"{self._relay_addr}{path}"
        logger.debug("metrics relay: posting to %s payload=%s", url, payload)

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
                            logger.debug(
                                "metrics relay: success %d for %s",
                                resp.status,
                                path,
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
            "key": f"dynamo_frontend_{metric_type}_{deployment}",
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


def get_metrics_relay_client() -> Optional[MetricsRelayClient]:
    """Return the singleton client, or None if METRICS_RELAY_ADDR is not set."""
    global _client
    if _client is not None:
        return _client
    addr = os.environ.get(_RELAY_ADDR_ENV)
    if not addr:
        logger.debug(
            "metrics relay disabled: %s env var not set", _RELAY_ADDR_ENV
        )
        return None
    _client = MetricsRelayClient(addr)
    return _client

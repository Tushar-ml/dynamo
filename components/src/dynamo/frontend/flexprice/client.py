import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

_EVENTS_PATH = "/events"
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)
# Bound on the pending-events queue (see class docstring); overridable via
# env for deployments that see burstier traffic than the default anticipates.
_QUEUE_SIZE = int(os.environ.get("DYN_FLEXPRICE_QUEUE_SIZE", "4500"))
# Cap on in-flight POSTs so a burst of billed requests drains faster than
# one-at-a-time (which would otherwise cap throughput at 1/RTT), without
# spawning unbounded concurrent tasks.
_MAX_CONCURRENT_SENDS = 20
# Total attempts (including the first) before giving up on a single event.
_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECS = 0.2


class FlexPriceClient:
    """Async client that emits LLM usage events to FlexPrice in the background.

    Enqueue is non-blocking — the caller returns immediately and the background
    worker drains the queue independently, so billing never adds latency to the
    request path. The queue is bounded (``_QUEUE_SIZE``); an event is dropped
    only when either:
      - the queue is full, i.e. events are arriving faster than
        ``_MAX_CONCURRENT_SENDS`` in-flight POSTs can drain them, or
      - a single event's POST still fails after ``_MAX_ATTEMPTS`` retries with
        backoff (persistent failure — the closest local proxy for "FlexPrice
        is down" without a live health check).
    A lone transient error (timeout, connection reset, one 5xx) does *not*
    drop an event — it's retried — and draining up to ``_MAX_CONCURRENT_SENDS``
    events concurrently means throughput isn't capped at one request-per-RTT.
    """

    def __init__(self, api_host: str, api_key: str) -> None:
        self._events_url = f"https://{api_host}{_EVENTS_PATH}"
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(
            maxsize=_QUEUE_SIZE
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SENDS)
        self._inflight: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=aiohttp.TCPConnector(ssl=True),
        )
        self._worker_task = asyncio.create_task(
            self._worker(), name="flexprice-event-worker"
        )

    async def stop(self) -> None:
        await self._queue.put(None)  # sentinel — drain then exit
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
        if self._inflight:
            # Best-effort: give any sends still retrying a moment to finish
            # rather than silently dropping them on shutdown.
            await asyncio.wait(self._inflight, timeout=5.0)
        if self._session:
            await self._session.close()

    def enqueue(
        self,
        event_name: str,
        external_customer_id: str,
        properties: Dict[str, Any],
        source: str = "",
        event_id: Optional[str] = None,
    ) -> None:
        """Non-blocking enqueue. Drops silently when the queue is full."""
        event: Dict[str, Any] = {
            "event_name": event_name,
            "external_customer_id": external_customer_id,
            "properties": {k: str(v) for k, v in properties.items()},
            "source": source,
            "event_id": event_id or str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "FlexPrice event queue full; dropping event for customer=%s",
                external_customer_id,
            )

    async def _worker(self) -> None:
        """Drains the queue, dispatching up to ``_MAX_CONCURRENT_SENDS`` POSTs
        at once so throughput isn't serialized behind one request-per-RTT.
        Acquiring the semaphore (inside ``_send_with_retry``) means
        backpressure happens off the request path — never a dropped event on
        its own.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is not None:
                        self._dispatch(item)
                if self._inflight:
                    await asyncio.gather(*self._inflight, return_exceptions=True)
                break
            self._dispatch(event)

    def _dispatch(self, event: Dict[str, Any]) -> None:
        task = asyncio.create_task(self._send_with_retry(event))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _send_with_retry(self, event: Dict[str, Any]) -> None:
        """Sends one event, retrying transient failures (network errors, 5xx)
        with backoff. Only gives up — dropping the event — after
        ``_MAX_ATTEMPTS`` consecutive failures.
        """
        async with self._semaphore:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                ok, detail = await self._send_once(event)
                if ok:
                    return
                if attempt == _MAX_ATTEMPTS:
                    logger.warning(
                        "FlexPrice event %s dropped after %d attempts: %s",
                        event.get("event_name"), attempt, detail,
                    )
                    return
                logger.debug(
                    "FlexPrice event %s send failed (attempt %d/%d): %s; retrying",
                    event.get("event_name"), attempt, _MAX_ATTEMPTS, detail,
                )
                await asyncio.sleep(_RETRY_BASE_DELAY_SECS * attempt)

    async def _send_once(self, event: Dict[str, Any]) -> "tuple[bool, str]":
        if not self._session:
            return False, "no session"
        payload = {
            "event_name": event["event_name"],
            "external_customer_id": event["external_customer_id"],
            "properties": event["properties"],
            "source": event.get("source", ""),
            "event_id": event.get("event_id", ""),
            "timestamp": event.get("timestamp", ""),
        }
        try:
            async with self._session.post(
                self._events_url, json=payload, timeout=_REQUEST_TIMEOUT
            ) as resp:
                if 200 <= resp.status < 300:
                    return True, ""
                return False, f"status={resp.status}"
        except Exception as exc:
            return False, str(exc)

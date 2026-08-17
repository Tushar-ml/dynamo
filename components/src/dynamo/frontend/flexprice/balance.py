"""Wallet-balance gate for prepaid orgs, mirroring go-proxy's
``pkg/flexprice.CheckBalance`` and the Rust ``BalanceChecker``.

Postpaid orgs (FlexPrice ``metadata.isKYC == "true"``) are always allowed
through regardless of balance. Suspended orgs are always blocked. Prepaid
orgs are blocked once their wallet balance drops below
``DYN_FLEXPRICE_MINIMUM_BALANCE`` (default ``0.0``, i.e. block once negative).

Results are cached in-process for ``_CACHE_TTL_SECS`` so a burst of requests
from the same org doesn't hit FlexPrice's wallet API once per request —
go-proxy uses a shared Redis cache for the same purpose across replicas; this
proxy has no Redis dependency, so this is a per-process equivalent. Any
FlexPrice API error fails open (allows the request) — a billing provider
outage must never itself drop inference traffic.
"""

import logging
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

_CACHE_TTL_SECS = 60.0
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=5)


class BalanceStatus(Enum):
    OK = "ok"
    SUSPENDED = "suspended"
    INSUFFICIENT_BALANCE = "insufficient_balance"


class BalanceChecker:
    def __init__(self, api_host: str, api_key: str, minimum_balance: float) -> None:
        self._wallets_url = f"https://{api_host}/customers/wallets"
        self._headers = {"x-api-key": api_key}
        self._minimum_balance = minimum_balance
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[BalanceStatus, float]] = {}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    async def check(self, org_uuid: str) -> BalanceStatus:
        """Whether `org_uuid` may proceed. Never blocks on a FlexPrice API
        failure — only a confirmed suspended/insufficient-balance result does.
        """
        cached = self._cache.get(org_uuid)
        if cached is not None:
            status, expires_at = cached
            if time.monotonic() < expires_at:
                return status

        status = await self._fetch(org_uuid)
        self._cache[org_uuid] = (status, time.monotonic() + _CACHE_TTL_SECS)
        return status

    async def _fetch(self, org_uuid: str) -> BalanceStatus:
        if not self._session:
            return BalanceStatus.OK
        try:
            async with self._session.get(
                self._wallets_url,
                params={
                    "lookup_key": org_uuid,
                    "include_real_time_balance": "true",
                },
                timeout=_REQUEST_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "FlexPrice wallet lookup returned status=%s for org=%s; allowing request",
                        resp.status,
                        org_uuid,
                    )
                    return BalanceStatus.OK
                wallets = await resp.json()
        except Exception as exc:
            logger.warning(
                "FlexPrice wallet lookup failed for org=%s: %s; allowing request",
                org_uuid,
                exc,
            )
            return BalanceStatus.OK

        if not wallets:
            logger.warning("No wallet found for org=%s; allowing request", org_uuid)
            return BalanceStatus.OK

        metadata = wallets[0].get("metadata") or {}

        if str(metadata.get("isSuspended", "")).lower() == "true":
            return BalanceStatus.SUSPENDED
        if str(metadata.get("isKYC", "")).lower() == "true":
            return BalanceStatus.OK

        balance_str = wallets[0].get("real_time_balance") or wallets[0].get("balance")
        if not balance_str:
            logger.warning(
                "Wallet response missing balance for org=%s; allowing request",
                org_uuid,
            )
            return BalanceStatus.OK

        try:
            balance = float(balance_str)
        except (TypeError, ValueError):
            logger.warning(
                "Could not parse wallet balance %r for org=%s; allowing request",
                balance_str,
                org_uuid,
            )
            return BalanceStatus.OK

        if balance >= self._minimum_balance:
            return BalanceStatus.OK
        return BalanceStatus.INSUFFICIENT_BALANCE

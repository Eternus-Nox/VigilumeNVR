"""Web Push (pywebpush + VAPID).

pywebpush is synchronous (requests) — every send runs in a worker thread.
Subscriptions that come back 404/410 (expired/unsubscribed) are pruned.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pywebpush import WebPushException, webpush

from ..db import Database

log = logging.getLogger(__name__)

_VAPID_CLAIMS_SUB = "mailto:admin@vigilume-nvr.local"


@dataclass
class PushSendResult:
    """Outcome of a send_to_all fan-out."""

    attempted: int = 0  # number of stored subscriptions targeted
    sent: int = 0  # successful deliveries
    errors: list[str] = field(default_factory=list)  # one message per failed send


class PushService:
    def __init__(self, db: Database, vapid_private_key: str, vapid_public_key: str):
        self._db = db
        self._private_key = vapid_private_key
        self.public_key = vapid_public_key

    async def send_to_all(self, payload: dict[str, Any]) -> PushSendResult:
        """Send `payload` to every stored subscription. Never raises."""
        subscriptions = await self._db.list_subscriptions()
        result = PushSendResult(attempted=len(subscriptions))
        if not subscriptions:
            return result
        data = json.dumps(payload)
        outcomes = await asyncio.gather(
            *(self._send_one(sub, data) for sub in subscriptions), return_exceptions=True
        )
        for outcome in outcomes:
            if outcome is None:
                result.sent += 1
            elif isinstance(outcome, BaseException):
                log.warning("push send raised unexpectedly: %s", outcome)
                result.errors.append(str(outcome))
            else:
                result.errors.append(outcome)
        return result

    async def _send_one(self, subscription: dict[str, Any], data: str) -> Optional[str]:
        """Send to one subscription. Returns None on success, else an error string."""
        endpoint: Optional[str] = subscription.get("endpoint")
        if not endpoint:
            return "stored subscription has no endpoint"
        try:
            await asyncio.to_thread(self._push_blocking, subscription, data)
            return None
        except WebPushException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 410):
                log.info("pruning expired push subscription (%s): %s...", status, endpoint[:60])
                await self._db.delete_subscription(endpoint)
                return f"push subscription expired (HTTP {status}); it has been removed"
            log.warning("push send failed (%s): %s", status, exc)
            return f"push send failed (HTTP {status}): {exc}" if status else f"push send failed: {exc}"
        except Exception as exc:  # noqa: BLE001 — network layer; never crash callers
            log.warning("push send failed: %s", exc)
            return f"push send failed: {exc}"

    def _push_blocking(self, subscription: dict[str, Any], data: str) -> None:
        webpush(
            subscription_info=subscription,
            data=data,
            vapid_private_key=self._private_key,
            vapid_claims={"sub": _VAPID_CLAIMS_SUB},
            ttl=120,
            # requests' default is no timeout at all — a hung push endpoint
            # would pin the worker thread and stall the notify caller (the
            # doorbell watcher awaits sends inline). APNs parity: 10s cap.
            timeout=10.0,
        )

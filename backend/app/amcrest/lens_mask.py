"""One-way migration off the RETIRED hardware lens mask.

Vigilume's original "privacy mode" drove the camera's own ``LeLensMask`` — a
full-frame device-side blackout. That control is GONE, replaced by software
Privacy Mode (``app/privacy.py``), which stops Vigilume's capture without
reconfiguring anything on the camera.

Removing a control does NOT clear the state it left behind. A camera masked
before the upgrade would stay blind forever, and with the control deleted
nothing in Vigilume could turn it back off — the operator would have to find the
camera's own web UI. On a security system that is a silently dead camera, so
every camera gets ``LeLensMask`` cleared once.

Driven off the SAME on-connect hook as :class:`~app.amcrest.time_sync.TimeSyncManager`
and :class:`~app.amcrest.speaker_probe.SpeakerProbeManager`: the camera prober
calls :meth:`notify_reachable` on every online transition. That is deliberately
NOT a boot sweep — at boot the fleet may still be coming up (PoE, switch, DHCP),
so a sweep would quietly skip exactly the cameras it needs to fix and would not
retry until the next restart. Clearing on the reachable transition instead means
a camera is fixed as soon as it appears, however late that is.

Everything is best-effort: idempotent (once per connection identity per
process), and NON-FATAL — an offline camera or a model without the CGI leaves
the camera unmarked so a later reachable transition retries.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from .client import AmcrestClient

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger(__name__)

# Per-camera ceiling for the clear. Shorter than the client's 8s default: this
# is a cleanup, so a slow camera is better retried on its next online
# transition than left holding a socket.
CLEAR_TIMEOUT_S = 5.0

# Connection identity. Model is NOT included: the mask is a device-side flag
# that a reclassification cannot change, so a model correction should not cost
# a redundant write.
ConnKey = tuple


def _conn_key(cam: dict[str, Any]) -> ConnKey:
    return (cam.get("ip"), cam.get("username"), cam.get("password"))


class LensMaskCleaner:
    def __init__(self, client_factory=None):
        # Injectable for tests; defaults to a real Amcrest client.
        self._client_factory = client_factory or (
            lambda cam: AmcrestClient(cam["ip"], cam["username"], cam["password"])
        )
        self._done: dict[str, ConnKey] = {}
        self._inflight: dict[str, ConnKey] = {}
        self._tasks: set[asyncio.Task] = set()

    async def notify_reachable(self, cam: dict[str, Any]) -> None:
        """Called by the prober when a camera becomes reachable. Clears the
        retired lens mask once, in the background. Never raises."""
        if not (cam.get("username") and cam.get("password")):
            return
        name = cam["name"]
        key = _conn_key(cam)
        if self._done.get(name) == key or self._inflight.get(name) == key:
            return
        self._inflight[name] = key
        task = asyncio.create_task(self._run(dict(cam), key), name=f"lens-mask:{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def forget(self, names: set[str]) -> None:
        """Drop bookkeeping for cameras that no longer exist."""
        for name in list(self._done):
            if name not in names:
                del self._done[name]

    async def _run(self, cam: dict[str, Any], key: ConnKey) -> None:
        name = cam["name"]
        client = self._client_factory(cam)
        try:
            await asyncio.wait_for(client.clear_lens_mask(), timeout=CLEAR_TIMEOUT_S)
            # Only mark done on success: a failure must retry on the next
            # online transition rather than strand a masked camera.
            self._done[name] = key
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — offline / no CGI / auth: retry later
            log.debug("lens-mask %s: clear failed (%s); will retry", name, exc)
        finally:
            if self._inflight.get(name) == key:
                del self._inflight[name]
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass

    async def stop_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._inflight.clear()

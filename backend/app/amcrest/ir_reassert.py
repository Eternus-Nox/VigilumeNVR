"""Doorbell IR re-assert (AD410 reverts IR Mode to Auto on RTSP connect).

The AD410 (and other Amcrest Smart-Home doorbells) reset their IR illuminator
Mode back to Auto whenever an RTSP client connects to the stream — so every
time the recorder / go2rtc (re)connects the doorbell, a user who chose a fixed
"IR Manual @ 60%" silently loses it. This module re-applies the camera's
STORED desired IR mode+brightness after each (re)connect, plus a slow periodic
sweep as a backstop.

Gating: only cameras that actually exhibit the revert are touched — the AD410
model or any camera whose capabilities advertise ``doorbell``. The IR-only
turrets (IP5M-T1277EW-AI / IP8M-2779EW-AI) keep their IR across streaming, so
they are never re-asserted (avoids pointless CGI traffic on every segment).

The desired state lives in the camera row's ``ir_state``
({"mode","brightness","night_vision_mode"}), written by PUT
/api/cameras/{name}/settings. An empty ir_state means the user never pinned
anything, so there is nothing to re-assert.

NIGHT VISION reverts the same way. The AD410 also resets its day/night mode to
Auto on RTSP (re)connect, so a doorbell pinned to FULL COLOUR silently fell back
to Auto every time the recorder/go2rtc reconnected — and, before the companion
fix in routers/cameras.py, the chosen mode was not even persisted, so there was
nothing to restore. Both are handled now; see `desired_night_vision_from`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from .client import AmcrestClient, AmcrestError

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Database

log = logging.getLogger(__name__)

# Delay after a stream (re)connect before re-asserting: the doorbell flips IR to
# Auto as the RTSP session comes up, so wait until that has settled.
REASSERT_DELAY_S = 5.0
# Backstop sweep interval — re-assert every doorbell's desired IR periodically
# in case a reconnect slipped past the hook (recording disabled, etc.).
REASSERT_INTERVAL_S = 300.0


def model_reverts_ir(model: str, capabilities: Optional[dict[str, Any]] = None) -> bool:
    """True when this camera resets IR Mode on RTSP connect and therefore needs
    re-asserting: the AD410 doorbell, or any camera whose capabilities advertise
    ``doorbell``. Turrets return False."""
    if "AD410" in (model or "").upper():
        return True
    return bool((capabilities or {}).get("doorbell"))


def desired_night_vision_from(ir_state: Optional[dict[str, Any]]) -> Optional[str]:
    """Stored night-vision mode ("auto"|"color"|"bw") to re-assert, or None.

    The AD410 resets its day/night mode to Auto on RTSP (re)connect exactly like
    it does the IR illuminator, so a user who picked FULL COLOUR lost it every
    time the recorder/go2rtc reconnected. `day_night` is deliberately NOT
    re-asserted here — that is the IR-cut filter, which streaming does not
    disturb."""
    if not isinstance(ir_state, dict):
        return None
    mode = ir_state.get("night_vision_mode")
    return mode if mode in ("auto", "color", "bw") else None


def desired_ir_from(ir_state: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Extract the re-assertable {mode?, brightness?} from a stored ir_state,
    or None when nothing was pinned. Ignores unrelated keys (e.g. day_night —
    the IR-cut filter is not reset by streaming). Night vision is handled
    separately by `desired_night_vision_from`."""
    if not isinstance(ir_state, dict):
        return None
    out: dict[str, Any] = {}
    mode = ir_state.get("mode")
    if mode in ("auto", "on", "off"):
        out["mode"] = mode
    brightness = ir_state.get("brightness")
    if isinstance(brightness, (int, float)) and not isinstance(brightness, bool):
        out["brightness"] = max(0, min(100, int(brightness)))
    return out or None


async def apply_ir_state(client: AmcrestClient, ir_state: Optional[dict[str, Any]]) -> bool:
    """Re-apply the stored desired IR mode+brightness through ``client``.
    Returns True when a re-assert was actually issued (there was a desired
    state), False when there was nothing to do. Raises AmcrestError on a device
    failure (callers log-and-swallow)."""
    desired = desired_ir_from(ir_state)
    night = desired_night_vision_from(ir_state)
    if desired is None and night is None:
        return False
    # NIGHT VISION FIRST, deliberately: set_night_vision_mode couples the IR
    # illuminator back to Auto by design, so a separately pinned IR mode has to
    # be applied AFTER it to survive.
    if night is not None:
        await client.set_night_vision_mode(night)
    if desired is not None:
        await client.set_ir(mode=desired.get("mode"), brightness=desired.get("brightness"))
    return True


# Factory: build an AmcrestClient for a camera row (injected so tests / the app
# can swap it; defaults to the standard client with per-camera creds + model).
ClientFactory = Callable[[dict[str, Any]], AmcrestClient]


def _default_client_factory(cam: dict[str, Any]) -> AmcrestClient:
    return AmcrestClient(cam["ip"], cam["username"], cam["password"], model=cam.get("model", ""))


class IrReasserter:
    """Re-applies stored desired IR to doorbell cameras after each stream
    (re)connect and on a slow periodic sweep. All device work is best-effort —
    a down camera or missing CGI never propagates out."""

    def __init__(
        self,
        db: "Database",
        client_factory: Optional[ClientFactory] = None,
        delay_s: float = REASSERT_DELAY_S,
        interval_s: float = REASSERT_INTERVAL_S,
    ):
        self._db = db
        self._client_factory = client_factory or _default_client_factory
        self._delay_s = delay_s
        self._interval_s = interval_s
        self._pending: dict[str, asyncio.Task] = {}
        self._sweep_task: Optional[asyncio.Task] = None
        self._running = False

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._sweep_task = asyncio.create_task(self._sweep_loop(), name="ir-reassert-sweep")

    async def stop(self) -> None:
        self._running = False
        tasks = [*self._pending.values()]
        if self._sweep_task is not None:
            tasks.append(self._sweep_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._sweep_task = None

    # ---------- hook: recorder/go2rtc (re)connected a camera's stream ----------

    def reassert_soon(self, camera_name: str) -> None:
        """Schedule a re-assert for ``camera_name`` after the settle delay.
        Safe to call from a sync context (the recorder's connect hook). A
        pending re-assert for the same camera is coalesced (latest wins)."""
        if not self._running:
            return
        existing = self._pending.get(camera_name)
        if existing is not None and not existing.done():
            existing.cancel()
        self._pending[camera_name] = asyncio.create_task(
            self._delayed_reassert(camera_name), name=f"ir-reassert:{camera_name}"
        )

    async def _delayed_reassert(self, camera_name: str) -> None:
        try:
            await asyncio.sleep(self._delay_s)
            await self._reassert_camera(camera_name, reason="stream (re)connect")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the hook must never surface an error
            log.exception("ir-reassert[%s]: unexpected error", camera_name)
        finally:
            if self._pending.get(camera_name) is asyncio.current_task():
                self._pending.pop(camera_name, None)

    # ---------- periodic backstop sweep ----------

    async def _sweep_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                raise
            try:
                for cam in await self._db.list_cameras():
                    if model_reverts_ir(cam.get("model", ""), cam.get("capabilities")):
                        await self._reassert_camera(cam["name"], reason="periodic sweep", cam=cam)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the sweep must outlive any cycle
                log.exception("ir-reassert: sweep cycle failed")

    # ---------- shared apply ----------

    async def _reassert_camera(
        self, camera_name: str, *, reason: str, cam: Optional[dict[str, Any]] = None
    ) -> bool:
        """Re-apply the camera's stored desired IR. No-op (returns False) for a
        missing camera, a non-doorbell model, missing creds, or an empty
        ir_state. Returns True when a re-assert CGI was issued."""
        if cam is None:
            cam = await self._db.get_camera(camera_name)
        if cam is None:
            return False
        if not model_reverts_ir(cam.get("model", ""), cam.get("capabilities")):
            return False
        if not cam.get("username") or not cam.get("password"):
            return False
        desired = desired_ir_from(cam.get("ir_state"))
        if desired is None:
            return False
        client = self._client_factory(cam)
        try:
            await apply_ir_state(client, cam.get("ir_state"))
        except AmcrestError as exc:
            log.info("ir-reassert[%s]: could not re-apply IR (%s): %s", camera_name, reason, exc)
            return False
        except Exception:  # noqa: BLE001
            log.exception("ir-reassert[%s]: unexpected device error", camera_name)
            return False
        finally:
            await client.aclose()
        log.info(
            "ir-reassert[%s]: re-applied desired IR %s after %s",
            camera_name, desired, reason,
        )
        return True

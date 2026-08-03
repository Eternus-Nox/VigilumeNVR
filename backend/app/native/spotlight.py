"""Per-camera "Smart spotlight" controller.

Feature (per-camera ``smart_spotlight`` flag, schema v14): when ON, a PERSON
detected at NIGHT (local sunset..sunrise for the configured location) on a
camera that has a white-light spotlight (``capabilities.white_light`` — the
on-demand illuminator driven by :meth:`AmcrestClient.set_white_light`) turns
that camera's spotlight ON, and keeps it on until the camera's per-camera hold
(``spotlight_hold_seconds``, default 60, valid 5..600) has elapsed AFTER the
LAST person detection, then turns it OFF. Each new person detection resets the
trailing hold.

Driven live off the stored flag by the detection engine: engine.process()
calls :meth:`notify_person` once per frame that carries a confirmed person
(the controller debounces — it never re-sends "on" while already on, and one
call per person-frame is fine). Structure mirrors TimeSyncManager /
DoorbellManager: a ``client_factory``, a cameras roster provider, and a
background task set.

Everything is best-effort: a device error (offline camera, rejected CGI, a
mock without the method) only logs — it MUST NOT crash the app or the detection
worker. :meth:`notify_person` is synchronous and returns immediately; the
device calls and the trailing off-timer run as background asyncio tasks.

Gates (all must hold for a person-frame to arm the light):
  * ``cam['smart_spotlight']`` is truthy (the per-camera toggle), AND
  * the camera has the ``white_light`` capability (else the toggle is
    meaningless — hidden/ignored in the UI), AND
  * ``is_night(now, lat, lon)`` — night for the configured lat/lon.

Off-safety: the controller only turns a spotlight OFF that IT turned on (it
tracks a per-camera on/off flag), so a light the operator switched on manually
is never extinguished by an expiring hold.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from ..amcrest.client import AmcrestClient
from ..amcrest.features import static_capabilities
from .sun import is_night as _default_is_night

log = logging.getLogger(__name__)

# Trailing hold: turn the spotlight OFF this long after the LAST person
# detection at night. Each new person detection restarts it. This is the
# fallback default when a camera row omits ``spotlight_hold_seconds`` (or it is
# None); the effective hold is per-camera (``cam['spotlight_hold_seconds']``),
# read live per detection and clamped to [MIN_HOLD_S, MAX_HOLD_S].
DEFAULT_HOLD_S = 60.0
# Defensive clamp bounds for the per-camera stored hold (the API validates the
# same 5..600 range; the controller re-clamps in case a stale/out-of-range
# value ever reaches it).
MIN_HOLD_S = 5.0
MAX_HOLD_S = 600.0


def camera_has_white_light(cam: dict[str, Any]) -> bool:
    """Whether ``cam`` has the on-demand white-light spotlight. Reads it the
    same way the camera routes do: the stored capability snapshot if present,
    else the static per-model map (``white_light`` is a static-only capability,
    so the model map is authoritative)."""
    caps = cam.get("capabilities") or {}
    if isinstance(caps.get("white_light"), bool):
        return caps["white_light"]
    return bool(static_capabilities(cam.get("model", "")).get("white_light"))


class SpotlightController:
    def __init__(
        self,
        config: Any,
        client_factory: Optional[Callable[[dict[str, Any]], AmcrestClient]] = None,
        cameras_provider: Optional[Callable[[], Awaitable[list[dict[str, Any]]]]] = None,
        *,
        hold_s: float = DEFAULT_HOLD_S,
        now: Callable[[], float] = time.time,
        is_night: Callable[[float, float, float], bool] = _default_is_night,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        # Async () -> camera rows (typically db.list_cameras); used by sync/
        # reload to prune per-camera state for removed cameras. Optional.
        self._cameras_provider = cameras_provider
        # Fallback trailing hold when a camera row omits spotlight_hold_seconds
        # (or it is None). The effective hold is per-camera; see _resolve_hold.
        self._default_hold_s = hold_s
        self._now = now
        self._is_night = is_night
        # The coroutine used to wait out the trailing hold. Injectable so tests
        # can compress the real-time wait (the tiny-interval approach); defaults
        # to asyncio.sleep in production.
        self._sleep = sleep
        # name -> True while the controller currently has the spotlight on (so
        # off->on is sent ONCE, and only a controller-lit spotlight is turned
        # off when its hold expires).
        self._on: dict[str, bool] = {}
        # name -> the pending trailing "off" timer task (restarted on each
        # person; its hold is the camera's spotlight_hold_seconds).
        self._timers: dict[str, asyncio.Task] = {}
        # In-flight device-call tasks (the off->on set_white_light), tracked so
        # stop_all can drain them.
        self._tasks: set[asyncio.Task] = set()

    def _latlon(self) -> tuple[float, float]:
        return (
            float(getattr(self._config, "latitude", 0.0)),
            float(getattr(self._config, "longitude", 0.0)),
        )

    # ---------- the engine hook ----------

    def notify_person(self, cam: dict[str, Any]) -> None:
        """Called by the detection engine for each frame that carries a confirmed
        person on ``cam`` (a DB camera row dict). Synchronous + best-effort;
        never raises. Arms/holds the spotlight when all gates hold."""
        try:
            name = cam.get("name")
            if not name:
                return
            if not cam.get("smart_spotlight"):
                return
            if not camera_has_white_light(cam):
                return
            lat, lon = self._latlon()
            if not self._is_night(self._now(), lat, lon):
                return
            # Person, at night, on an enabled white-light camera.
            if not self._on.get(name):
                # off -> on transition: send set_white_light("on") ONCE.
                self._on[name] = True
                self._spawn_set(cam, "on")
            # (Re)start the trailing off-timer on EVERY person detection. The
            # hold is read per-notify from the camera's spotlight_hold_seconds.
            self._restart_timer(cam)
        except Exception:  # noqa: BLE001 — must never crash the detection worker
            log.exception("smart-spotlight notify_person failed for %s", cam.get("name"))

    # ---------- timer + device calls ----------

    def _resolve_hold(self, cam: dict[str, Any]) -> float:
        """The per-camera trailing hold (seconds) for ``cam``: its stored
        ``spotlight_hold_seconds`` clamped defensively to [MIN_HOLD_S,
        MAX_HOLD_S]. A missing/None/non-numeric value falls back to the default
        (``self._default_hold_s``, normally 60)."""
        raw = cam.get("spotlight_hold_seconds")
        if raw is None:
            return self._default_hold_s
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return self._default_hold_s
        return max(MIN_HOLD_S, min(MAX_HOLD_S, val))

    def _restart_timer(self, cam: dict[str, Any]) -> None:
        name = cam["name"]
        hold = self._resolve_hold(cam)
        old = self._timers.pop(name, None)
        if old is not None:
            old.cancel()
        task = asyncio.create_task(
            self._hold_then_off(dict(cam), hold), name=f"spotlight-off:{name}"
        )
        self._timers[name] = task

    async def _hold_then_off(self, cam: dict[str, Any], hold: float) -> None:
        name = cam["name"]
        try:
            await self._sleep(hold)
            # Held the full window with no new person: turn the light off — but
            # ONLY if WE lit it (never extinguish an operator-lit spotlight).
            if self._on.get(name):
                self._on[name] = False
                await self._set_white_light(cam, "off")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the off-timer must never crash the app
            log.exception("smart-spotlight off-timer failed for %s", name)
        finally:
            # Only clear our own registration (a newer person may have replaced
            # this timer with a fresh one already).
            if self._timers.get(name) is asyncio.current_task():
                del self._timers[name]

    def _spawn_set(self, cam: dict[str, Any], mode: str) -> None:
        task = asyncio.create_task(
            self._set_white_light(dict(cam), mode), name=f"spotlight-{mode}:{cam.get('name')}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _set_white_light(self, cam: dict[str, Any], mode: str) -> None:
        """Drive the device spotlight to ``mode`` ("on" | "off"). Best-effort:
        an offline camera / rejected CGI / mock without the method only logs."""
        name = cam.get("name")
        client = self._client_factory(cam)
        try:
            await client.set_white_light(mode)
            log.info("smart-spotlight %s: white light -> %s", name, mode)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — device errors are non-fatal
            log.info("smart-spotlight %s: set_white_light(%s) failed (%s)", name, mode, exc)
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    # ---------- lifecycle (mirrors the other background managers) ----------

    async def start(self) -> None:
        """No periodic loop — the controller is driven entirely by the engine's
        per-frame notify_person. Provided for symmetry with the other managers."""
        return None

    async def sync(self, cameras: list[dict[str, Any]]) -> None:
        """Reconcile against the camera rows: forget per-camera state (and cancel
        any pending off-timer) for cameras that no longer exist. Never raises."""
        names = {cam["name"] for cam in cameras}
        for name in list(self._on):
            if name not in names:
                del self._on[name]
        for name in list(self._timers):
            if name not in names:
                self._timers.pop(name).cancel()

    async def reload(self) -> None:
        """Re-read the roster (if a provider was supplied) and prune removed
        cameras. Convenience for camera-CRUD; never raises."""
        if self._cameras_provider is None:
            return
        try:
            cameras = await self._cameras_provider()
            await self.sync(cameras)
        except Exception:  # noqa: BLE001 — reload must never crash CRUD
            log.exception("smart-spotlight reload failed")

    async def stop_all(self) -> None:
        for task in list(self._timers.values()):
            task.cancel()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(
            *self._timers.values(), *self._tasks, return_exceptions=True
        )
        self._timers.clear()
        self._tasks.clear()
        self._on.clear()

    # Alias so a caller using the start()/stop() vocabulary works too.
    async def stop(self) -> None:
        await self.stop_all()


def _default_client_factory(cam: dict[str, Any]) -> AmcrestClient:
    return AmcrestClient(
        cam["ip"], cam["username"], cam["password"], model=cam.get("model", "")
    )

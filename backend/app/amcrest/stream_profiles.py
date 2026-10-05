"""Keeps each camera's MAIN stream at its encode profile (amcrest/encode.py).

Every camera follows ``settings.streams.main`` unless it pins its own profile
(``cameras.main_stream``) — so one change sets every camera, and a camera that
needs something different (the one covering the far end of the drive,
say) keeps it. A profile is applied:

  * when it is saved (the API waits for the result, so the screen can say what
    each camera actually did);
  * when a camera comes back online (the prober's on-connect hook), and
  * every ``interval_s`` (30 min) — a factory reset, a firmware update, or a
    change made in the camera's own web page drifts back.

Applying is idempotent (only differing values are written) and read back
(encode results report what the camera kept, not what was asked). A camera
with nothing to change costs one getConfig per cycle.

A value the camera answered OK to and did not keep is NOT written again by
the background passes (reconnect, every 30 min) while the profile is
unchanged — only a save retries it. Every write can restart the camera's
encoder, which drops go2rtc and every live viewer for a few seconds; a value
that never sticks was being rewritten on every pass, and a reconnect-triggered
pass could follow the very drop it caused. Reconnect passes are also limited
to one per camera per ``_RECONNECT_COOLDOWN_S``. A resolution or codec
change restarts the camera's encoder; go2rtc and the recorder reconnect on
their own, and ``on_stream_changed`` lets the recording reader forget the old
frame size.

Best-effort throughout: an offline camera, a rejected value or a camera
without credentials only logs and is reported in ``status()``.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from . import encode
from .client import AmcrestClient, AmcrestError

log = logging.getLogger(__name__)

_INTERVAL_S = 30 * 60.0
#: One camera's apply: a size may take several key combinations, each read
#: back after the camera restarts its encoder.
_APPLY_TIMEOUT_S = 60.0
#: Cameras configured at once by an "apply to all".
_CONCURRENCY = 4
#: A reconnect re-applies at most this often per camera.
_RECONNECT_COOLDOWN_S = 15 * 60.0


class StreamProfileManager:
    def __init__(
        self,
        settings: Any,
        cameras_provider: Optional[Callable[[], Awaitable[list[dict[str, Any]]]]] = None,
        client_factory: Optional[Callable[[dict[str, Any]], AmcrestClient]] = None,
        on_stream_changed: Optional[Callable[[str], None]] = None,
        interval_s: float = _INTERVAL_S,
    ):
        self._settings = settings
        self._cameras_provider = cameras_provider
        self._client_factory = client_factory or _default_client_factory
        self._on_stream_changed = on_stream_changed
        self._interval = interval_s
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, dict[str, Any]] = {}
        self._tasks: set[asyncio.Task] = set()
        # name -> (profile it was applied with, keys the camera would not keep)
        self._stuck: dict[str, tuple[dict[str, Any], set[str]]] = {}
        self._reconnect_at: dict[str, float] = {}

    # ---------- profiles ----------

    def global_profile(self) -> dict[str, Any]:
        streams = getattr(self._settings, "streams", None) or {}
        try:
            return encode.normalize_profile(streams.get("main"))
        except ValueError:
            return encode.empty_profile()

    def effective(self, cam: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """(profile, inherited) for one camera."""
        own = cam.get("main_stream")
        if own is None:
            return self.global_profile(), True
        try:
            return encode.normalize_profile(own), False
        except ValueError:
            return self.global_profile(), True

    # ---------- applying ----------

    async def apply(self, cam: dict[str, Any], *, background: bool = False) -> dict[str, Any]:
        """Apply this camera's effective profile now; returns the outcome
        ({ok, changed, rejected, not_applied, notes, before, after} or
        {ok: False, error}). Never raises.

        `background` (reconnect / periodic): skip what this camera would not
        keep last time with the same profile. A save passes False and retries
        everything."""
        name = cam["name"]
        profile, inherited = self.effective(cam)
        out: dict[str, Any] = {"camera": name, "inherited": inherited, "at": time.time()}
        if not (cam.get("username") and cam.get("password")):
            out.update(ok=False, error="no camera credentials stored")
            self._last[name] = out
            return out
        if encode.is_noop(profile):
            out.update(ok=True, changed=[], rejected=[], not_applied=[], notes=[], skipped=True)
            self._last[name] = out
            return out
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            client = self._client_factory(cam)
            skip: Optional[set[str]] = None
            if background:
                prev = self._stuck.get(name)
                if prev is not None and prev[0] == profile:
                    skip = prev[1]
            try:
                result = await asyncio.wait_for(
                    client.apply_main_stream(profile, skip=skip) if skip
                    else client.apply_main_stream(profile),
                    timeout=_APPLY_TIMEOUT_S,
                )
                self._stuck[name] = (profile, set(result.get("stuck_keys") or ()))
                out.update(ok=not result["rejected"] and not result["not_applied"], **result)
                if result["changed"]:
                    log.info("main-stream %s: %s", name, "; ".join(result["changed"]))
                for item in result["rejected"] + result["not_applied"]:
                    log.warning("main-stream %s: not applied — %s", name, item)
                before, after = result.get("before") or {}, result.get("after") or {}
                moved = any(before.get(k) != after.get(k) for k in ("width", "height", "codec"))
                if moved and self._on_stream_changed is not None:
                    with contextlib.suppress(Exception):
                        self._on_stream_changed(name)
            except asyncio.TimeoutError:
                out.update(ok=False, error="the camera did not answer in time")
            except AmcrestError as exc:
                out.update(ok=False, error=str(exc))
            except ValueError as exc:
                out.update(ok=False, error=f"invalid profile: {exc}")
            except Exception as exc:  # noqa: BLE001 — never crash over provisioning
                log.exception("main-stream %s: unexpected error", name)
                out.update(ok=False, error=f"unexpected error: {exc.__class__.__name__}")
            finally:
                with contextlib.suppress(Exception):
                    await client.aclose()
        self._last[name] = out
        return out

    async def apply_many(
        self, cameras: list[dict[str, Any]], *, background: bool = False,
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(_CONCURRENCY)

        async def one(cam: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await self.apply(cam, background=background)

        return list(await asyncio.gather(*(one(c) for c in cameras)))

    def apply_in_background(
        self, cameras: list[dict[str, Any]], *, background: bool = False,
    ) -> None:
        """Apply without waiting. `background` as for `apply` — False for a
        settings save, which should retry everything."""
        task = asyncio.create_task(self.apply_many(cameras, background=background),
                                   name="main-stream-apply")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def notify_reachable(self, cam: dict[str, Any]) -> None:
        """Prober on-connect hook: re-assert in the background, at most once
        per _RECONNECT_COOLDOWN_S per camera."""
        profile, _ = self.effective(cam)
        if encode.is_noop(profile):
            return
        now = time.monotonic()
        last = self._reconnect_at.get(cam["name"])
        if last is not None and now - last < _RECONNECT_COOLDOWN_S:
            return
        self._reconnect_at[cam["name"]] = now
        self.apply_in_background([cam], background=True)

    async def run(self) -> None:
        """Periodic re-assert loop. Never raises out of the loop."""
        if self._cameras_provider is None:
            return
        while True:
            await asyncio.sleep(self._interval)
            try:
                cams = await self._cameras_provider()
                await self.apply_many([c for c in cams
                                       if not encode.is_noop(self.effective(c)[0])],
                                      background=True)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the loop must never die
                log.exception("main-stream periodic re-apply failed")

    def last(self, name: str) -> Optional[dict[str, Any]]:
        return self._last.get(name)

    def forget(self, name: str) -> None:
        self._last.pop(name, None)
        self._locks.pop(name, None)
        self._stuck.pop(name, None)
        self._reconnect_at.pop(name, None)

    async def stop_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


def _default_client_factory(cam: dict[str, Any]) -> AmcrestClient:
    return AmcrestClient(cam["ip"], cam["username"], cam["password"], model=cam.get("model", ""))

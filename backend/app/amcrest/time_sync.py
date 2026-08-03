"""Automatic camera clock correction (set local time + disable NTP).

Amcrest/Dahua doorbell/camera clocks drift — one AD410 shipped stuck at
year-2000 on a factory Beijing timezone. NTP proved unreliable on these units
(a forced sync did not land within 150s) AND the Dahua ``NTP.TimeZone`` index ->
UTC-offset mapping can not be trusted (a wrong index drifts the wall-clock hours
off). So this manager does NOT touch NTP-for-correctness or the device timezone
index. Instead it provisions each reachable Dahua/Amcrest camera via
:meth:`AmcrestClient.provision_time`, which pushes the correct LOCAL wall-clock
time (computed for a configurable IANA timezone via zoneinfo, independent of the
container's own — usually UTC — clock) and DISABLES the device NTP client so a
wrong index can never re-drift it.

Design (mirrors DoorbellManager / AiEventListener):
  * :meth:`sync` reconciles against the camera rows (called at boot + after
    camera CRUD): it provisions any camera not yet successfully provisioned for
    its current connection identity and forgets removed cameras.
  * :meth:`notify_reachable` is called by the camera prober the moment a camera
    transitions to online (first contact / reconnect), so an offline-at-boot
    camera gets provisioned the instant it becomes reachable.
  * :meth:`run` is a periodic background loop (every ``interval_s``, default
    30 min) that re-pushes the current time to every camera UNCONDITIONALLY
    (clocks drift, so this is NOT gated on the per-identity done-set) — a
    self-healing backstop on top of the connect hook.

Everything is best-effort and MUST NOT crash the app or block startup / a
request: each provision runs in its own background task; a camera that is
offline or rejects the CGI only logs. On the connect path a camera is marked
provisioned only on SUCCESS, so a transient failure is retried on the next
reachable transition. Only Dahua/Amcrest-type cameras (those with device
credentials — required for any CGI) are attempted; the auto_sync setting gates
the whole thing.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Optional

from ..config import DEFAULT_CAMERA_TIMEZONE
from ..settings_store import SettingsStore
from .client import AmcrestClient, AmcrestError


def _is_doorbell(cam: dict[str, Any]) -> bool:
    """Local mirror of native.streams.is_doorbell / doorbell.py's own test.
    Inlined rather than imported: this module lives under `amcrest`, and
    reaching into `native.streams` from here would add a package dependency
    purely for a two-line predicate."""
    if (cam.get("capabilities") or {}).get("doorbell"):
        return True
    return (cam.get("model") or "").strip().upper() == "AD410"

log = logging.getLogger(__name__)

# Periodic re-push cadence: clocks drift, so the correct time is re-pushed to
# every camera on this interval (independent of the connect hook).
_RESYNC_INTERVAL_S = 30 * 60.0

# Connection identity: a change here (ip / creds) means a different target that
# must be (re)provisioned.
ConnKey = tuple


def _conn_key(cam: dict[str, Any]) -> ConnKey:
    return (cam.get("ip"), cam.get("username"), cam.get("password"))


def is_amcrest_camera(cam: dict[str, Any]) -> bool:
    """Whether NTP provisioning should be attempted for ``cam``. Every camera in
    this system is Amcrest/Dahua, but a camera with no stored credentials can not
    be driven over CGI at all — so require both, which is the same practical gate
    the control routes use (``needs_credentials``). Provisioning is non-fatal, so
    a non-Amcrest device that happens to carry credentials merely logs a rejected
    CGI and is otherwise harmless."""
    return bool(cam.get("username") and cam.get("password"))


class TimeSyncManager:
    def __init__(
        self,
        settings: SettingsStore,
        client_factory: Optional[Callable[[dict[str, Any]], AmcrestClient]] = None,
        cameras_provider: Optional[Callable[[], Awaitable[list[dict[str, Any]]]]] = None,
        interval_s: float = _RESYNC_INTERVAL_S,
    ):
        self._settings = settings
        self._client_factory = client_factory or _default_client_factory
        # Async () -> camera rows, used by the periodic loop (typically
        # db.list_cameras). None disables the loop (the connect hook still runs).
        self._cameras_provider = cameras_provider
        self._interval = interval_s
        # name -> conn key successfully provisioned (so the CONNECT hook never
        # re-hits a camera whose clock is already set for its current identity;
        # the periodic loop re-pushes regardless — see _maybe_provision(force)).
        self._done: dict[str, ConnKey] = {}
        # name -> conn key currently being provisioned (dedupes the boot sweep
        # racing the prober's online transition for the same camera).
        self._inflight: dict[str, ConnKey] = {}
        self._tasks: set[asyncio.Task] = set()

    def _config(self) -> tuple[bool, str]:
        ts = self._settings.time_sync
        enabled = bool(ts.get("auto_sync", True))
        tz_name = (ts.get("timezone") or DEFAULT_CAMERA_TIMEZONE).strip() or DEFAULT_CAMERA_TIMEZONE
        return enabled, tz_name

    async def sync(self, cameras: list[dict[str, Any]]) -> None:
        """Reconcile against the camera rows: forget removed cameras and
        provision any Amcrest camera not yet done for its current identity.
        Never raises."""
        names = {cam["name"] for cam in cameras}
        for name in list(self._done):
            if name not in names:
                del self._done[name]
        enabled, tz_name = self._config()
        if not enabled:
            return
        for cam in cameras:
            self._maybe_provision(cam, tz_name)

    async def notify_reachable(self, cam: dict[str, Any]) -> None:
        """Called by the prober when a camera becomes reachable (first contact /
        reconnect). Provisions it once, in the background. Never raises."""
        enabled, tz_name = self._config()
        if not enabled:
            return
        self._maybe_provision(cam, tz_name)

    async def resync_all(self, cameras: list[dict[str, Any]]) -> None:
        """Force a clock re-push on every credentialled camera regardless of the
        per-identity done-set (clocks drift, so this is unconditional). Used by
        the periodic :meth:`run` loop. Never raises."""
        enabled, tz_name = self._config()
        if not enabled:
            return
        for cam in cameras:
            self._maybe_provision(cam, tz_name, force=True)

    async def run(self) -> None:
        """Periodic re-push loop: every ``interval_s`` re-set the clock on every
        Dahua/Amcrest camera (offline ones just log + retry next cycle). No-op
        when no cameras_provider was supplied. Never raises out of the loop."""
        if self._cameras_provider is None:
            return
        while True:
            await asyncio.sleep(self._interval)
            try:
                cameras = await self._cameras_provider()
                await self.resync_all(cameras)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the loop must never die
                log.exception("time-sync periodic re-push cycle failed")

    def _maybe_provision(
        self, cam: dict[str, Any], tz_name: str, *, force: bool = False
    ) -> None:
        if not is_amcrest_camera(cam):
            return
        name = cam["name"]
        key = _conn_key(cam)
        # Always dedupe a concurrent in-flight provision for the same identity.
        # The connect hook additionally skips an already-done camera; the
        # periodic loop (force=True) re-pushes it (clocks drift).
        if self._inflight.get(name) == key:
            return
        if not force and self._done.get(name) == key:
            return
        self._inflight[name] = key
        task = asyncio.create_task(
            self._provision(dict(cam), key, tz_name), name=f"time-sync:{name}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _provision(self, cam: dict[str, Any], key: ConnKey, tz_name: str) -> None:
        name = cam["name"]
        client = self._client_factory(cam)
        try:
            summary = await client.provision_time(tz_name)
            self._done[name] = key
            # "pushed", not "local_time": current_time is the stamp WE sent, not
            # anything the camera confirmed — the old "clock set ... local_time=X"
            # line read like an observation and was echoing our own input, which
            # is how 11 cameras drifted 1-20 min behind a perfect log.
            # measured_offset IS the observation (provision_time reads the clock
            # back); "unverified" means the device wouldn't tell us.
            delta = summary.get("clock_delta_s")
            log.info(
                "time-sync %s: clock pushed (tz=%s pushed=%s ntp_disabled=%s "
                "measured_offset=%s)",
                name, summary.get("timezone"), summary.get("current_time"),
                summary.get("ntp_disabled"),
                "unverified" if delta is None else f"{delta:+.1f}s",
            )
            # Also force the audio encoder to the camera's stored preference:
            # "g711a" (default) keeps live-view (WebRTC) audio working (go2rtc
            # passes G.711/PCMA through but not AAC, so an AAC camera plays silent
            # in live view); "aac" trades live-view audio for higher recording
            # quality. Idempotent + best-effort — a getConfig/setConfig failure
            # (or a mock without the method) never fails the time provision or the
            # app. Re-checked on the periodic cycle, so a factory-reset camera
            # self-heals back to the chosen codec.
            provision_audio = getattr(client, "provision_audio", None)
            if provision_audio is not None:
                codec = "AAC" if (cam.get("audio_codec") == "aac") else "G.711A"
                try:
                    audio = await provision_audio(codec)
                    if audio.get("changed"):
                        log.info("audio-codec %s: set %s on %d format(s)",
                                 name, codec, len(audio["changed"]))
                except AmcrestError as exc:
                    log.info("audio-codec %s: not set yet (%s)", name, exc)
                except Exception:  # noqa: BLE001 — never crash over audio provisioning
                    log.exception("audio-codec %s: unexpected error", name)

            # Shorten the SUBSTREAM keyframe interval to ~1 s so live view paints
            # sooner (a consumer can only start decoding on an I-frame, and
            # go2rtc caches no GOP for a new consumer). ExtraFormat only — the
            # main stream feeds the 24/7 recorder and more keyframes there would
            # inflate every recording. Only ever SHORTENS, idempotent, and
            # best-effort like the audio provision above.
            #
            # The AD410 is skipped on purpose: its go2rtc `_sub` resolves to the
            # main source, so changing its ExtraFormat would buy live view
            # nothing — and this is the camera whose encoder/session handling has
            # already cost us the talk backchannel once. Not worth the risk.
            provision_gop = getattr(client, "provision_substream_gop", None)
            if provision_gop is not None and not _is_doorbell(cam):
                try:
                    gop = await provision_gop()
                    if gop.get("changed"):
                        log.info("substream-gop %s: %s", name, ", ".join(gop["changed"]))
                except AmcrestError as exc:
                    log.info("substream-gop %s: not set yet (%s)", name, exc)
                except Exception:  # noqa: BLE001 — never crash over GOP provisioning
                    log.exception("substream-gop %s: unexpected error", name)
        except AmcrestError as exc:
            # Offline / auth-fail / CGI rejected: expected on a flaky or
            # non-Dahua camera. Left unmarked so a later reachable transition
            # (or the next periodic cycle) retries. Never fatal.
            log.info("time-sync %s: not provisioned yet (%s)", name, exc)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — provisioning must never crash the app
            log.exception("time-sync %s: unexpected provisioning error", name)
        finally:
            if self._inflight.get(name) == key:
                del self._inflight[name]
            with contextlib.suppress(Exception):
                await client.aclose()

    async def stop_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._inflight.clear()


def _default_client_factory(cam: dict[str, Any]) -> AmcrestClient:
    return AmcrestClient(
        cam["ip"], cam["username"], cam["password"], model=cam.get("model", "")
    )

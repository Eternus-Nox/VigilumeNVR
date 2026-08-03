"""On-connect talk-speaker detection for the two-way TALK button.

The TALK button must appear ONLY on cameras that actually have an audio-output
(talk) speaker. Model families are mixed and the per-model static map (see
``features.STATIC_CAPABILITIES``) can not cover every future/unlisted model, so
this manager PROBES each reachable camera and pins its ``speaker`` capability
from a conclusive runtime signal:

  * ONVIF ``GetAudioOutputs()`` — the authoritative "does this device have an
    audio output" query (confirmed live: AD410 doorbell => 1, EW turrets => 0).
    The devAudioOutput.cgi ``getCollect`` used by the generic capability probe
    is unreliable on the AD410 (it answers 0 despite a real speaker), so ONVIF
    is used here instead.
  * ``backchannel`` implies ``speaker``: a camera whose talk runs over the RTSP
    audio backchannel (the AD410) necessarily has a speaker, so it is pinned
    True WITHOUT an ONVIF round-trip — this also keeps the two flags consistent.

The AD410's clock drifts (it shipped stuck in year 2000), which makes ONVIF
WS-Security timestamps fail; onvif-zeep's ``ONVIFCamera(..., adjust_time=True)``
reads the device clock and offsets the request timestamps, so the probe retries
with that on any first-attempt failure.

Design mirrors :class:`~app.amcrest.time_sync.TimeSyncManager` and reuses the
exact same connect hook: the camera prober calls :meth:`notify_reachable` on
every online transition, a boot sweep + camera-CRUD resync call :meth:`sync`.
Everything is best-effort:

  * idempotent — a camera is probed once per connection identity (ip/creds/model);
  * NON-FATAL — an offline camera, a non-ONVIF device, or any probe error leaves
    the prior/static ``speaker`` value untouched and never crashes startup, and
    the camera is left unmarked so a later reachable transition retries;
  * a change is persisted with a targeted ``json_set`` (never clobbers other
    capabilities) and broadcast as ``cameras_changed`` so clients re-fetch and
    hide/show the Talk button live.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

from .features import CAPABILITY_KEYS, static_capabilities

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Database
    from ..ws import WSManager

log = logging.getLogger(__name__)

# ONVIF host port = the camera HTTP port (validated at 80 on the Amcrest units,
# same as the AI-event PullPoint path).
ONVIF_PORT = 80

# Connection identity: model is included (unlike time-sync) because the
# speaker verdict depends on it — a camera reclassified unknown -> AD410 gains
# backchannel/speaker and must be re-probed.
ConnKey = tuple

# probe(cam) -> Optional[bool]: True/False when conclusively determined, None
# when inconclusive (offline / not ONVIF-capable / error) so the caller retries.
ProbeFn = Callable[[dict[str, Any]], Awaitable[Optional[bool]]]


def _conn_key(cam: dict[str, Any]) -> ConnKey:
    return (cam.get("ip"), cam.get("username"), cam.get("password"), cam.get("model"))


def has_credentials(cam: dict[str, Any]) -> bool:
    """ONVIF (like every CGI control path) needs device credentials; a camera
    with none can not be probed at all."""
    return bool(cam.get("username") and cam.get("password"))


def effective_capabilities(cam: dict[str, Any]) -> dict[str, bool]:
    """The camera's current capabilities: the per-model static map with any
    conclusively-stored values merged over it (same rule the camera response
    uses). Read here to check ``backchannel`` and to skip a no-op write."""
    caps = static_capabilities(cam.get("model") or "")
    stored = cam.get("capabilities") or {}
    for key in CAPABILITY_KEYS:
        val = stored.get(key)
        if isinstance(val, bool):
            caps[key] = val
    return caps


class SpeakerProbeError(Exception):
    """ONVIF GetAudioOutputs could not be read from the device."""


# onvif-zeep's ONVIFCamera.__init__ -> update_xaddrs() calls
# CreatePullPointSubscription on the device, so MERELY CONSTRUCTING a camera
# object registers an event subscription — even here, where we only want
# GetAudioOutputs and never touch events. Nothing released it, and the
# adjust_time retry ladder below builds a SECOND camera, so a probe could strand
# two. Probes fire on every offline->online transition, so a flapping camera
# churned them.
#
# Dahua devices cap concurrent subscriptions. These do expire on the device's
# own lease (update_xaddrs sends no InitialTerminationTime), so this was bounded
# rather than a permanent leak — but transient exhaustion is exactly the kind of
# thing that makes an unrelated ONVIF call fail intermittently, which is a
# miserable bug to chase. Release it explicitly instead.
_PULLPOINT_NS = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"


def _release_pullpoint(cam: Any) -> None:
    """Best-effort Unsubscribe of the subscription ONVIFCamera.__init__ created.

    Entirely non-fatal: this runs in a finally during a probe whose ONLY job is
    to report whether a speaker exists. A device that never registered one, or
    that refuses Unsubscribe, must not turn a successful probe into a failure.
    """
    try:
        if _PULLPOINT_NS not in getattr(cam, "xaddrs", {}):
            return  # nothing was ever created (the common failure path)
        cam.create_pullpoint_service().Unsubscribe()
    except Exception:  # noqa: BLE001 — hygiene only; never fail a probe over it
        log.debug("speaker-probe: PullPoint Unsubscribe failed", exc_info=True)


def _get_audio_outputs(
    ip: str, port: int, username: str, password: str, *, adjust_time: bool
) -> int:
    """BLOCKING ONVIF ``GetAudioOutputs`` -> count of audio outputs on the device.

    ``GetAudioOutputs`` lives on the DeviceIO service on most firmwares and on
    Media on others, so both are tried. Raises :class:`SpeakerProbeError` when
    neither service answers (offline / not ONVIF / auth). Lazy-imports
    onvif-zeep so web-only / test import paths don't need it. Overridable in
    tests by monkeypatching this name."""
    from onvif import ONVIFCamera  # noqa: PLC0415 — deliberate lazy import

    cam = ONVIFCamera(ip, port, username, password, adjust_time=adjust_time)
    try:
        errors: list[str] = []
        for create in ("create_deviceio_service", "create_media_service"):
            try:
                service = getattr(cam, create)()
                outputs = service.GetAudioOutputs()
            except Exception as exc:  # noqa: BLE001 — try the next service transport
                errors.append(f"{create}={exc.__class__.__name__}")
                continue
            if outputs is None:
                return 0
            try:
                return len(outputs)
            except TypeError:
                return 0
        raise SpeakerProbeError("; ".join(errors) or "no ONVIF audio service")
    finally:
        _release_pullpoint(cam)


async def probe_speaker_onvif(cam: dict[str, Any]) -> Optional[bool]:
    """Default probe: ONVIF ``GetAudioOutputs() > 0`` for ``cam``.

    Runs the blocking ONVIF calls in a thread. Retries once with
    ``adjust_time=True`` (handles the AD410's drifted clock breaking WS-Security
    timestamps). Returns True/False on a conclusive read, or None when the
    device could not be reached / is not ONVIF-capable (so the manager leaves
    the prior value and retries later)."""
    ip = cam["ip"]
    username = cam["username"]
    password = cam["password"]
    last: Optional[Exception] = None
    for adjust_time in (False, True):
        try:
            count = await asyncio.to_thread(
                _get_audio_outputs, ip, ONVIF_PORT, username, password,
                adjust_time=adjust_time,
            )
            return count > 0
        except Exception as exc:  # noqa: BLE001 — non-fatal; fall through to retry/None
            last = exc
    log.info(
        "speaker-probe %s: ONVIF GetAudioOutputs unreadable (%s: %s)",
        cam.get("name"), last.__class__.__name__ if last else "?", last,
    )
    return None


class SpeakerProbeManager:
    def __init__(
        self,
        db: "Database",
        ws: Optional["WSManager"] = None,
        probe: Optional[ProbeFn] = None,
    ):
        self._db = db
        self._ws = ws
        self._probe = probe or probe_speaker_onvif
        # name -> conn key already resolved (so a probed camera is not re-hit
        # for its current identity).
        self._done: dict[str, ConnKey] = {}
        # name -> conn key currently probing (dedupes the boot sweep racing the
        # prober's online transition for the same camera).
        self._inflight: dict[str, ConnKey] = {}
        self._tasks: set[asyncio.Task] = set()

    async def sync(self, cameras: list[dict[str, Any]]) -> None:
        """Reconcile against the camera rows: forget removed cameras and probe
        any credentialled camera not yet resolved for its current identity.
        Never raises."""
        names = {cam["name"] for cam in cameras}
        for name in list(self._done):
            if name not in names:
                del self._done[name]
        for cam in cameras:
            self._maybe_probe(cam)

    async def notify_reachable(self, cam: dict[str, Any]) -> None:
        """Called by the prober when a camera becomes reachable (first contact /
        reconnect). Probes it once, in the background. Never raises."""
        self._maybe_probe(cam)

    def _maybe_probe(self, cam: dict[str, Any]) -> None:
        if not has_credentials(cam):
            return
        name = cam["name"]
        key = _conn_key(cam)
        if self._done.get(name) == key or self._inflight.get(name) == key:
            return
        self._inflight[name] = key
        task = asyncio.create_task(
            self._run(dict(cam), key), name=f"speaker-probe:{name}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, cam: dict[str, Any], key: ConnKey) -> None:
        name = cam["name"]
        try:
            caps = effective_capabilities(cam)
            # backchannel implies speaker: a backchannel-talk camera (AD410) has
            # a speaker by definition — pin it True without an ONVIF round-trip
            # and keep the two flags consistent.
            if caps.get("backchannel"):
                desired: Optional[bool] = True
            else:
                desired = await self._probe(cam)
            if desired is None:
                # Inconclusive (offline / not ONVIF / error): leave the prior
                # value and DO NOT mark done, so a later reachable transition
                # retries. Non-fatal.
                return
            if desired != caps.get("speaker"):
                updated = await self._db.set_camera_capability(name, "speaker", desired)
                if updated:
                    log.info(
                        "speaker-probe %s [%s]: speaker capability set to %s",
                        name, cam.get("model") or "?", desired,
                    )
                    if self._ws is not None:
                        with contextlib.suppress(Exception):
                            await self._ws.broadcast({"type": "cameras_changed"})
            self._done[name] = key
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — probing must never crash the app
            log.exception("speaker-probe %s: unexpected error", name)
        finally:
            if self._inflight.get(name) == key:
                del self._inflight[name]

    async def stop_all(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._inflight.clear()

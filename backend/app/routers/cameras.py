"""Camera CRUD + Amcrest device control routes (docs/CONTRACTS.md)."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from ..amcrest.client import (
    AmcrestAuthError,
    AmcrestClient,
    AmcrestError,
    AmcrestUnsupportedError,
    white_light_control_for_model,
)
from ..amcrest.features import (
    CAPABILITY_KEYS,
    PROBE_KEYS,
    merge_capabilities,
    probe_capabilities,
    static_capabilities,
)
from ..auth import require_admin, require_auth, require_media_auth, role_from_claims
from .. import privacy
from ..config import (
    DEFAULT_DETECT_FPS,
    DEFAULT_DETECT_OBJECTS,
    KNOWN_MODELS,
    MODEL_DETECT_DEFAULTS,
    VALID_DETECT_MODES,
    effective_detect_mode,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cameras", tags=["cameras"], dependencies=[Depends(require_auth)])

# The camera name doubles as the go2rtc stream slug and appears in RTSP
# restream URLs — enforce a strict lowercase slug.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_PROBE_TIMEOUT_S = 12.0
# POST /{name}/probe (on-demand "Save & Test") — 8 s total per contract.
_ONDEMAND_PROBE_CAP_S = 8.0

UNKNOWN_MODEL = "unknown"

# Device strings that name a known model but neither equal nor prefix it, so
# no amount of string logic could resolve them — the firmware's own id simply
# differs from the name the model is sold and stored under.
#
# Only add an entry VERIFIED against real hardware (fleet audit, not a guess):
# a wrong alias silently mis-gates features with no way to notice.
DEVICE_TYPE_ALIASES = {
    # Verified 2026-07-16 on 192.168.1.87 + 192.168.1.88: the IP8M-2779EW-AI
    # reports "IP8M-2779E-AI" — no W (that suffix denotes the white housing and
    # isn't part of the firmware id). Without this both cameras auto-detect to
    # "unknown" and lose their spotlight/AI gating.
    "IP8M-2779E-AI": "IP8M-2779EW-AI",
}


def match_known_model(device_type: Optional[str]) -> Optional[str]:
    """Match a getDeviceType response against KNOWN_MODELS. An exact match wins
    outright; then a verified alias (DEVICE_TYPE_ALIASES); otherwise a TRUNCATED
    device string ('IP3M-941' -> 'IP3M-941B') resolves only when exactly one
    known model extends it. None when inconclusive OR ambiguous.

    Fails closed on purpose: a wrong-but-confident model is worse than
    "unknown", because the keys it corrupts (white_light, ptz, ai_on_camera)
    are static-only — not in features.PROBE_KEYS — so no later probe can undo
    it. A longer, genuinely different device string must NOT adopt a shorter
    known model: 'IP4M-1056EW-AI' is not an 'IP4M-1056E' (which would pin
    mic/ai_on_camera False forever), and a bare 'IP5M-' is ambiguous across
    the IP5M family rather than a match for the first one listed."""
    dt = (device_type or "").strip().upper()
    if len(dt) < 5:  # shortest known model is 'AD410'; reject junk
        return None
    for model in KNOWN_MODELS:
        if model.upper() == dt:
            return model
    aliased = DEVICE_TYPE_ALIASES.get(dt)
    if aliased:
        return aliased
    matches = [m for m in KNOWN_MODELS if m.upper().startswith(dt)]
    return matches[0] if len(matches) == 1 else None


class ExemptZone(BaseModel):
    """One privacy/ignore polygon: normalized (0..1) points + an optional name.

    Points are ``[x, y]`` pairs in resolution-independent 0..1 coords (origin
    top-left). Coords are clamped to [0, 1]; malformed points are dropped by the
    CameraInput validator. Anything whose box foot-center falls inside is
    suppressed by the detection engine."""

    name: str = Field(default="", max_length=64)
    points: list[tuple[float, float]] = Field(default_factory=list)


class IncludeZone(BaseModel):
    """One "only alert here" polygon — the allow-list counterpart of
    :class:`ExemptZone`, same normalized 0..1 point format.

    The two compose: include zones decide what is watched at all, exempt zones
    then punch holes in it. Configure none and the whole frame is watched, which
    is what every existing install does."""

    name: str = Field(default="", max_length=64)
    points: list[tuple[float, float]] = Field(default_factory=list)


class CrossLine(BaseModel):
    """A boundary whose crossings are counted (sv.LineZone).

    ``start``/``end`` are normalized 0..1 ``[x, y]``. Direction matters: a
    crossing to the LEFT of the start->end arrow counts as "in", the other way
    as "out", so drawing the line the other way round swaps the two. The web UI
    draws the arrow, so this is a visible property rather than a hidden one."""

    name: str = Field(default="", max_length=64)
    start: tuple[float, float]
    end: tuple[float, float]


class CameraInput(BaseModel):
    name: str
    friendly_name: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=64)
    ip: str = Field(min_length=1, max_length=253)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(max_length=128)
    detect_objects: Optional[list[str]] = None
    # Per-camera exempt (privacy/ignore) detection zones. None (field omitted)
    # = defaults ([]) on create / keep stored on update; an explicit [] clears
    # all zones. Each zone is normalized-coord polygon (see ExemptZone).
    exempt_zones: Optional[list[ExemptZone]] = None
    # Per-camera INCLUDE zones and crossing lines. Same None/[] contract as
    # exempt_zones: omitted = defaults on create / keep stored on update; an
    # explicit [] clears them.
    include_zones: Optional[list[IncludeZone]] = None
    cross_lines: Optional[list[CrossLine]] = None
    # "Only alert me when something crosses a line on this camera." None
    # (omitted) = keep stored on update / off on create. Gates the NOTIFICATION
    # only — the event, its clip and its snapshot are recorded either way — and
    # the pipeline ignores it entirely on a camera with no crossing lines.
    notify_on_cross: Optional[bool] = None
    # Optional per-camera engine toggles. None means "default true" on
    # create and "keep the stored value" on update, so clients that don't
    # send them never flip a camera's state.
    detect_enabled: Optional[bool] = None
    record_enabled: Optional[bool] = None
    # Detection frame rate (native engine samples the substream at this fps).
    # None = default 5 on create / keep stored on update.
    detect_fps: Optional[int] = Field(default=None, ge=1, le=10)
    # Per-camera server-detection mode (see config.VALID_DETECT_MODES). None
    # (omitted) = keep stored on update / inherit settings.default_mode on
    # create; "" clears it back to inherit; a valid mode is stored verbatim.
    detect_mode: Optional[str] = Field(default=None, max_length=32)
    # Optional RTSP URL overrides for non-Amcrest cameras / odd ports.
    # None = keep stored (update) or empty (create); "" = Amcrest default
    # derived from ip + credentials.
    main_url: Optional[str] = Field(default=None, max_length=512)
    sub_url: Optional[str] = Field(default=None, max_length=512)

    @field_validator("name")
    @classmethod
    def _name_slug(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                "name must be a lowercase slug matching [a-z][a-z0-9_]{0,31} "
                "(it becomes the stream name)"
            )
        return v

    @field_validator("main_url", "sub_url")
    @classmethod
    def _rtsp_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if v and not v.startswith("rtsp://"):
            raise ValueError("stream URL overrides must start with rtsp:// (blank = Amcrest default)")
        return v

    @field_validator("detect_objects")
    @classmethod
    def _objects_slugs(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        # None (field omitted) is preserved as None so create/update can tell
        # "not sent" (-> defaults / keep stored) apart from an explicit empty
        # list. An explicit [] survives as [] (record-only, detect nothing).
        if v is None:
            return None
        cleaned = []
        for item in v:
            item = item.strip().lower()
            if item and re.match(r"^[a-z][a-z0-9_]{0,31}$", item):
                cleaned.append(item)
        return cleaned

    @field_validator("exempt_zones")
    @classmethod
    def _clean_zones(cls, v: Optional[list[ExemptZone]]) -> Optional[list[ExemptZone]]:
        # None (omitted) stays None so create/update can tell "not sent" apart
        # from an explicit [] (clear all zones). Each zone's points are clamped
        # to [0, 1]; zones with fewer than 3 points are dropped (a polygon needs
        # at least 3 vertices — the engine ignores them anyway).
        if v is None:
            return None
        cleaned: list[ExemptZone] = []
        for zone in v:
            pts = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in zone.points]
            if len(pts) >= 3:
                cleaned.append(ExemptZone(name=zone.name.strip(), points=pts))
        return cleaned

    @field_validator("include_zones")
    @classmethod
    def _clean_include(cls, v: Optional[list[IncludeZone]]) -> Optional[list[IncludeZone]]:
        # Same contract as _clean_zones: None (omitted) survives, points clamp to
        # [0, 1], and anything under 3 points is dropped rather than stored. A
        # degenerate include zone is more dangerous than a degenerate exempt one
        # — it would silently include NOTHING and blind the camera — so it must
        # never reach the engine.
        if v is None:
            return None
        cleaned: list[IncludeZone] = []
        for zone in v:
            pts = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in zone.points]
            if len(pts) >= 3:
                cleaned.append(IncludeZone(name=zone.name.strip(), points=pts))
        return cleaned

    @field_validator("cross_lines")
    @classmethod
    def _clean_lines(cls, v: Optional[list[CrossLine]]) -> Optional[list[CrossLine]]:
        # Endpoints clamp to [0, 1]; a line whose ends coincide is dropped —
        # sv.LineZone raises on a zero-magnitude vector, so storing one would
        # turn a mis-click into a camera that logs a warning every reload.
        if v is None:
            return None
        cleaned: list[CrossLine] = []
        for line in v:
            sx, sy = (min(1.0, max(0.0, c)) for c in line.start)
            ex, ey = (min(1.0, max(0.0, c)) for c in line.end)
            if (sx, sy) == (ex, ey):
                continue
            cleaned.append(
                CrossLine(name=line.name.strip(), start=(sx, sy), end=(ex, ey))
            )
        return cleaned

    @field_validator("detect_mode")
    @classmethod
    def _detect_mode_clean(cls, v: Optional[str]) -> Optional[str]:
        # None (omitted) is preserved so create/update can tell "not sent" apart
        # from an explicit clear. "" or a valid mode survive; an UNKNOWN mode is
        # rejected (a typo must never silently disable/gate detection).
        if v is None:
            return None
        v = v.strip().lower()
        if v == "":
            return ""
        if v not in VALID_DETECT_MODES:
            raise ValueError(f"detect_mode must be one of {VALID_DETECT_MODES} or blank")
        return v

    @field_validator("ip")
    @classmethod
    def _ip_hostish(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9.\-]+$", v):
            raise ValueError("ip must be a bare IP or hostname (no scheme/port)")
        return v


class CameraUpdate(CameraInput):
    """PUT /api/cameras/{name} body.

    A blank username — exactly like a blank password — means "keep the stored
    value", so the edit form doesn't force re-entering credentials, and a
    settings-only change (e.g. flipping detect_mode) that never carries the
    password can omit it entirely instead of 422-ing "password field required".
    """

    username: str = Field(default="", max_length=64)
    password: str = Field(default="", max_length=128)
    # Per-camera live-view audio codec preference: "g711a" (default; native
    # G.711/PCMA is WebRTC-legal, so live-view audio WORKS) | "aac" (higher
    # recording quality, but NO live-view audio — WebRTC can't carry AAC). None
    # (omitted) keeps the stored value. Validated in the route handler (an
    # invalid value -> 400) rather than by a field validator, so a bad codec is
    # a clean 400 per the contract instead of a 422.
    audio_codec: Optional[str] = Field(default=None, max_length=8)
    # Per-camera "Smart spotlight" toggle. None (omitted) keeps the stored value;
    # true/false persists. Only meaningful on capabilities.white_light cameras
    # (ignored by the controller otherwise). Persist-only — no device call on the
    # PUT; SpotlightController reads the stored flag live.
    smart_spotlight: Optional[bool] = None
    # Per-camera smart-spotlight trailing hold (seconds): how long the spotlight
    # stays on after the LAST person detection before it turns off. None (omitted)
    # keeps the stored value; a value must be an int in [5, 600] (out-of-range ->
    # a clean 400 in the route handler, not a 422). Only meaningful alongside
    # smart_spotlight on a white_light camera. Persist-only — the controller reads
    # the stored value live.
    spotlight_hold_seconds: Optional[int] = None


class VolumePatch(BaseModel):
    mic: Optional[int] = Field(default=None, ge=0, le=100)
    speaker: Optional[int] = Field(default=None, ge=0, le=100)


class WhiteLightPatch(BaseModel):
    mode: Optional[Literal["off", "on", "auto"]] = None
    brightness: Optional[int] = Field(default=None, ge=0, le=100)


class DeviceSettingsPatch(BaseModel):
    # IR is mode-only (Auto / On / Off). "On" = Manual at the camera's stored
    # illuminator strength — there is no brightness control (removed by request).
    ir_mode: Optional[Literal["auto", "on", "off"]] = None
    # Day/Night IR-cut mode (VideoInDayNight[0][0].Mode). Optional, additive.
    day_night: Optional[Literal["color", "black_white", "brightness"]] = None
    # Full-color night-vision mode (VideoInDayNight[0][N].Mode on ALL profiles).
    # Capability-gated on night_vision (the IP4M-1056E). Additive, distinct from
    # ir_mode (which drives the IR illuminator, not the day/night colour mode).
    night_vision_mode: Optional[Literal["auto", "color", "bw"]] = None
    white_light: Optional[WhiteLightPatch] = None
    flip: Optional[bool] = None
    osd_name: Optional[str] = Field(default=None, max_length=64)
    motion_detect: Optional[bool] = None
    volume: Optional[VolumePatch] = None


class LightRequest(BaseModel):
    """POST /{name}/light body (Camera controls v2 addendum)."""

    mode: Literal["off", "on", "auto"]
    brightness: Optional[int] = Field(default=None, ge=0, le=100)


class SirenRequest(BaseModel):
    duration_s: int = Field(default=10, ge=1, le=30)


class PtzRequest(BaseModel):
    """POST /{name}/ptz body (capability-gated on `ptz` — the IP3M-941B dome).

    - action "move": continuous pan/tilt in `direction` at `speed` (1-8).
    - action "step": ONE small bounded nudge in `direction` (a single tap =
      "a hair"); the server issues start -> brief dwell -> stop itself.
    - action "stop": halt the continuous move for `direction`.
    - action "preset_set"/"preset_goto"/"preset_clear": operate on preset
      `index` (1-3).
    """

    action: Literal["move", "step", "stop", "preset_set", "preset_goto", "preset_clear"]
    direction: Optional[
        Literal["up", "down", "left", "right", "upleft", "upright", "downleft", "downright"]
    ] = None
    speed: int = Field(default=4, ge=1, le=8)
    index: Optional[int] = Field(default=None, ge=1, le=3)


# ---------- helpers ----------


def _zones_to_stored(zones: Optional[list[ExemptZone]]) -> list[dict[str, Any]]:
    """ExemptZone models -> plain JSON-serializable dicts for the DB / response
    ({"name": str, "points": [[x, y], ...]})."""
    out: list[dict[str, Any]] = []
    for z in zones or []:
        out.append({"name": z.name, "points": [[x, y] for x, y in z.points]})
    return out


def _include_to_stored(zones: Optional[list[IncludeZone]]) -> list[dict[str, Any]]:
    """IncludeZone models -> plain dicts for the DB / response."""
    return [{"name": z.name, "points": [[x, y] for x, y in z.points]} for z in zones or []]


def _lines_to_stored(lines: Optional[list[CrossLine]]) -> list[dict[str, Any]]:
    """CrossLine models -> plain dicts for the DB / response."""
    return [
        {"name": ln.name, "start": list(ln.start), "end": list(ln.end)}
        for ln in lines or []
    ]


def _caps_of(cam: dict[str, Any]) -> dict[str, bool]:
    stored = cam.get("capabilities") or {}
    caps = static_capabilities(cam["model"])
    for key in CAPABILITY_KEYS:
        # Only PROBE-determined keys may be overridden by the stored snapshot;
        # static-only keys (ptz, night_vision, white_light, ai_on_camera) always
        # come from the current model map so a STATIC_CAPABILITIES update is never
        # masked by a stale stored value from an earlier registration.
        if key in PROBE_KEYS and key in stored and isinstance(stored[key], bool):
            caps[key] = stored[key]
    # A backchannel two-way camera (AD410, IP3M-941B) always has a speaker to
    # play talk audio; its devAudioOutput CGI probe is unreliable and may have
    # stored speaker=False, so never let that hide two-way talk.
    if caps.get("backchannel"):
        caps["speaker"] = True
    return caps


def _camera_response(
    cam: dict[str, Any],
    online: bool,
    *,
    default_mode: str = "always",
    ai_active: bool = False,
    private: bool = False,
    is_admin: bool = False,
) -> dict[str, Any]:
    return {
        "name": cam["name"],
        "friendly_name": cam["friendly_name"],
        "model": cam["model"],
        "ip": cam["ip"],
        "online": online,
        # Software Privacy Mode (app/privacy.py): this camera is currently
        # capturing NOTHING. Carried on the camera row so a tile can render the
        # "Privacy Mode" overlay straight from the list it already has, instead
        # of every client fetching /api/privacy separately and drifting out of
        # sync with the camera list.
        "private": private,
        # Whether Amcrest control still needs device credentials.
        "source": cam.get("source") or "manual",
        "needs_credentials": not cam.get("username") or not cam.get("password"),
        "capabilities": _caps_of(cam),
        # White-light/spotlight control contract so the UI renders the right
        # control: the EW turrets are coaxialControlIO on/off ONLY (no
        # brightness slider, no 'auto'); Lighting_V2 models get off/on/auto +
        # brightness. Only meaningful when capabilities.white_light is true.
        "white_light_control": white_light_control_for_model(cam["model"]),
        # The STORED tracked-object list, verbatim: an empty list stays empty
        # (record-only) so the object picker round-trips exactly what is
        # stored and the 4 defaults never silently reappear.
        "detect_objects": list(cam.get("detect_objects") or []),
        # Stored exempt (privacy/ignore) zones, verbatim: [{name, points:[[x,y]]}].
        "exempt_zones": list(cam.get("exempt_zones") or []),
        # Stored INCLUDE zones ([{name, points}]) and crossing lines
        # ([{name, start, end}]), verbatim — the editor round-trips exactly what
        # is stored, same as exempt_zones.
        "include_zones": list(cam.get("include_zones") or []),
        "cross_lines": list(cam.get("cross_lines") or []),
        "notify_on_cross": bool(cam.get("notify_on_cross") or False),
        "detect": {"enabled": bool(cam.get("detect_enabled", True))},
        "record": {"enabled": bool(cam.get("record_enabled", True))},
        "detect_fps": int(cam.get("detect_fps") or DEFAULT_DETECT_FPS),
        # Effective server-detection mode (stored value, or the settings default
        # when unset/NULL). The raw stored value (may be null) is under
        # detect_mode_stored so the edit UI can show "inherit default".
        "detect_mode": effective_detect_mode(
            cam.get("detect_mode"),
            default_mode,
            ai_on_camera=bool((cam.get("capabilities") or {}).get("ai_on_camera")),
        ),
        "detect_mode_stored": cam.get("detect_mode"),
        # Live camera-AI indicator: true while the camera's on-board AI is firing
        # (only meaningful in camera_ai / camera_ai_only modes). Always present.
        "ai_active": bool(ai_active),
        # RTSP overrides; empty string = Amcrest default from ip+creds.
        #
        # ADMIN-ONLY VALUES. For an override to work against an Amcrest/Dahua
        # unit it has to carry credentials — rtsp://admin:<password>@… — so
        # these fields are the camera admin password in plaintext. This router
        # is require_auth, not require_admin, so without this gate any VIEWER
        # account could read them from GET /api/cameras and log straight into
        # the camera's own web UI (PTZ, firmware, factory reset), entirely
        # outside this system's RBAC and Privacy Mode.
        #
        # `*_url_set` keeps the UI able to say "an override is configured"
        # without handing over the secret. Defaults to redacted: a new call site
        # that forgets to pass is_admin leaks nothing.
        "main_url": (cam.get("main_url") or "") if is_admin else "",
        "sub_url": (cam.get("sub_url") or "") if is_admin else "",
        "main_url_set": bool(cam.get("main_url")),
        "sub_url_set": bool(cam.get("sub_url")),
        # Live-view audio codec preference: "g711a" (default; WebRTC-legal, so
        # live-view audio works) | "aac" (higher recording quality, no live-view
        # audio). The backend re-provisions the device encoder on a change.
        "audio_codec": cam.get("audio_codec") or "g711a",
        # Per-camera "Smart spotlight": when true, a person detected at night on
        # a white_light camera auto-turns the spotlight on (held past the last
        # person). Only meaningful when capabilities.white_light is true.
        "smart_spotlight": bool(cam.get("smart_spotlight")),
        # Per-camera smart-spotlight trailing hold (seconds): how long the
        # spotlight stays on after the LAST person detection (default 60, 5..600).
        "spotlight_hold_seconds": int(cam.get("spotlight_hold_seconds") or 60),
    }


def _client_for(cam: dict[str, Any]) -> AmcrestClient:
    # model is passed so the client can pick the right per-model control CGI
    # (e.g. coaxialControlIO white light for the EW turrets).
    return AmcrestClient(cam["ip"], cam["username"], cam["password"], model=cam.get("model", ""))


async def _get_cam_or_404(request: Request, name: str) -> dict[str, Any]:
    cam = await request.app.state.db.get_camera(name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{name}' not found")
    return cam


async def _probe_caps(cam: dict[str, Any]) -> dict[str, Any]:
    """Best-effort runtime capability probe; {} when the device is offline."""
    client = _client_for(cam)
    try:
        return await asyncio.wait_for(probe_capabilities(client), timeout=_PROBE_TIMEOUT_S)
    except (asyncio.TimeoutError, AmcrestError):
        return {}
    except Exception:  # noqa: BLE001
        log.exception("capability probe failed for %s", cam["name"])
        return {}
    finally:
        await client.aclose()


async def _apply_camera_change(request: Request) -> None:
    """Propagate a camera CRUD change to every native subsystem: go2rtc
    config regen + sync, engine reload (ingest/detect set), recorder reload
    (segment ffmpeg set), doorbell watcher resync, status probe. None of
    these raise — a down go2rtc or skeleton engine must not fail CRUD."""
    state = request.app.state
    # Refresh the resolved private set FIRST: a camera deleted/recreated or moved
    # between groups changes who is effectively private, and every reconciler
    # below reads state.private_cameras. Must precede the reloads.
    await privacy.refresh(state)
    await state.go2rtc.apply()
    await state.engine.reload()
    await state.recorder.reload()
    cameras = await state.db.list_cameras()
    await state.doorbells.sync(cameras)
    # Resync camera-AI watchers: a mode/capability/credentials change may start,
    # stop, or reconnect a camera's AI event stream. None-safe + never raises.
    ai_events = getattr(state, "ai_events", None)
    if ai_events is not None:
        default_mode = state.settings.detection.get("default_mode", "always")
        await ai_events.sync(cameras, default_mode=default_mode)
    # Provision NTP + timezone on any newly-added / changed-identity camera.
    # Idempotent (only unprovisioned identities are attempted) + non-fatal.
    time_sync = getattr(state, "time_sync", None)
    if time_sync is not None:
        await time_sync.sync(cameras)
    # Prune smart-spotlight state for removed cameras (the per-frame
    # notify_person reads the live camera row, so no other resync is needed).
    # None-safe + never raises.
    spotlight = getattr(state, "spotlight", None)
    if spotlight is not None:
        await spotlight.sync(cameras)
    # Re-probe the talk-speaker on any newly-added / changed-identity camera so
    # the Talk button reflects real hardware. Idempotent + non-fatal.
    speaker_probe = getattr(state, "speaker_probe", None)
    if speaker_probe is not None:
        await speaker_probe.sync(cameras)
    state.prober.probe_soon()


# ---------- CRUD ----------


def _default_mode_of(state) -> str:
    return state.settings.detection.get("default_mode", "always")


def _ai_active_of(state, name: str) -> bool:
    ai_events = getattr(state, "ai_events", None)
    return bool(ai_events is not None and ai_events.is_active(name))


async def _cameras_payload(request: Request, *, is_admin: bool) -> list[dict[str, Any]]:
    """The camera list, with RTSP overrides redacted unless `is_admin`.

    Separate from the route so INTERNAL callers (e.g. set_camera_order, which
    returns the reordered list) state their own privilege explicitly. Calling
    the route function directly does not run FastAPI's dependency injection, so
    a `claims` parameter would arrive as the raw Depends object.
    """
    state = request.app.state
    cams = await state.db.list_cameras()
    default_mode = _default_mode_of(state)
    return [
        _camera_response(
            cam,
            state.prober.is_online(cam["name"]),
            default_mode=default_mode,
            ai_active=_ai_active_of(state, cam["name"]),
            private=privacy.is_private(state, cam["name"]),
            is_admin=is_admin,
        )
        for cam in cams
    ]


@router.get("")
async def list_cameras(
    request: Request, claims: dict[str, Any] = Depends(require_auth)
) -> list[dict[str, Any]]:
    # The only VIEWER-reachable path that renders a camera. The admin settings
    # page edits the RTSP overrides through this same list, so an admin still
    # gets the real values; a viewer gets them redacted.
    return await _cameras_payload(request, is_admin=role_from_claims(claims) == "admin")


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
async def add_camera(body: CameraInput, request: Request) -> dict[str, Any]:
    state = request.app.state
    if await state.db.get_camera(body.name) is not None:
        raise HTTPException(status_code=409, detail=f"Camera '{body.name}' already exists")
    width, height = MODEL_DETECT_DEFAULTS.get(body.model, (704, 480))
    cam = {
        "name": body.name,
        "friendly_name": body.friendly_name,
        "model": body.model,
        "ip": body.ip,
        "username": body.username,
        "password": body.password,
        # None (caller omitted the field) -> the defaults; [] (caller sent an
        # empty list) -> record-only; otherwise the cleaned list verbatim.
        "detect_objects": (
            list(DEFAULT_DETECT_OBJECTS) if body.detect_objects is None else body.detect_objects
        ),
        # None (omitted) -> no zones; otherwise the cleaned polygons verbatim.
        "exempt_zones": _zones_to_stored(body.exempt_zones),
        "include_zones": _include_to_stored(body.include_zones),
        "cross_lines": _lines_to_stored(body.cross_lines),
        "notify_on_cross": bool(body.notify_on_cross),
        "detect_width": width,
        "detect_height": height,
        "detect_fps": body.detect_fps or DEFAULT_DETECT_FPS,
        "detect_enabled": True if body.detect_enabled is None else body.detect_enabled,
        "record_enabled": True if body.record_enabled is None else body.record_enabled,
        # None (omitted) / "" -> NULL (inherit settings.default_mode); a valid
        # mode is stored verbatim.
        "detect_mode": body.detect_mode or None,
        # smart_spotlight is not set on create (like audio_codec) — it defaults
        # off via _camera_params; it is toggled later via PUT (CameraUpdate).
        "main_url": body.main_url or "",
        "sub_url": body.sub_url or "",
        "created_at": time.time(),
    }
    probed = await _probe_caps(cam)
    # The device already told us what it is (_probe_caps -> probe_capabilities
    # -> getDeviceType), so adopt that rather than trusting the caller's claim
    # — but ONLY when the caller asked for auto-detect (""/"unknown"). A
    # concrete model is a deliberate override and is never second-guessed.
    # Offline / auth-failed / unlisted model -> matched is None -> we write
    # nothing and the camera stays "unknown". Never write on a None: a network
    # blip must not cost a camera its model.
    if cam["model"].strip().lower() in ("", UNKNOWN_MODEL):
        matched = match_known_model(probed.get("device_type"))
        if matched:
            log.info(
                "model-detect %s: device reported %r -> adopted %s",
                cam["name"], probed.get("device_type"), matched,
            )
            cam["model"] = matched
            width, height = MODEL_DETECT_DEFAULTS.get(matched, (704, 480))
            cam["detect_width"], cam["detect_height"] = width, height
    # cam["model"] (not body.model): keeps this consistent with the PUT and
    # /probe paths, and pairs the capability snapshot with the model actually
    # stored. The two must never move independently.
    cam["capabilities"] = merge_capabilities(cam["model"], probed)
    await state.db.upsert_camera(cam)
    await _apply_camera_change(request)
    return _camera_response(
        cam,
        state.prober.is_online(cam["name"]),
        default_mode=_default_mode_of(state),
        ai_active=_ai_active_of(state, cam["name"]),
        private=privacy.is_private(state, cam["name"]),
        is_admin=True,   # route is admin-gated
    )


class CameraOrder(BaseModel):
    """PUT /api/cameras/order body: full or partial dashboard order."""

    names: list[str] = Field(max_length=256)


# NOTE: registered before PUT /{name} so "order" is never captured as a
# camera name by the dynamic route.
@router.put("/order", dependencies=[Depends(require_admin)])
async def set_camera_order(body: CameraOrder, request: Request) -> list[dict[str, Any]]:
    """Assign display positions in the given order; names not listed keep
    their relative order after the listed ones (unknown names ignored).
    Purely cosmetic — touches nothing but the position column. Returns the
    reordered camera list (same shape as GET /api/cameras)."""
    state = request.app.state
    await state.db.set_camera_order(body.names)
    # Admin-gated route (reordering cameras), so full visibility.
    return await _cameras_payload(request, is_admin=True)


# The hardware lens-mask routes (GET/POST /api/cameras/privacy) were REMOVED.
# They drove the camera's own LeLensMask blackout, which meant reconfiguring the
# device and left the camera itself in a modified state. Software Privacy Mode
# (app/privacy.py, /api/privacy) replaces them: it stops all capture — recording,
# detection, events, live view, audio, on-camera AI — while touching nothing on
# the camera. See amcrest/lens_mask.py for the one-way migration that makes sure
# no camera is left masked by the old feature.


@router.put("/{name}", dependencies=[Depends(require_admin)])
async def update_camera(name: str, body: CameraUpdate, request: Request) -> dict[str, Any]:
    state = request.app.state
    existing = await _get_cam_or_404(request, name)
    if body.name != name:
        raise HTTPException(status_code=400, detail="Camera rename is not supported")
    cam = dict(existing)
    cam.update(
        {
            "friendly_name": body.friendly_name,
            "model": body.model,
            "ip": body.ip,
            # Empty username/password in an edit mean "keep the stored ones".
            "username": body.username or existing["username"],
            "password": body.password or existing["password"],
        }
    )
    if body.detect_objects is not None:
        cam["detect_objects"] = body.detect_objects
    if body.exempt_zones is not None:
        # Omitted keeps stored zones; an explicit [] clears them.
        cam["exempt_zones"] = _zones_to_stored(body.exempt_zones)
    if body.include_zones is not None:
        # Omitted keeps stored zones; an explicit [] clears them (back to
        # watching the whole frame).
        cam["include_zones"] = _include_to_stored(body.include_zones)
    if body.cross_lines is not None:
        cam["cross_lines"] = _lines_to_stored(body.cross_lines)
    if body.notify_on_cross is not None:
        cam["notify_on_cross"] = body.notify_on_cross
    if body.detect_enabled is not None:
        cam["detect_enabled"] = body.detect_enabled
    if body.record_enabled is not None:
        cam["record_enabled"] = body.record_enabled
    if body.detect_fps is not None:
        cam["detect_fps"] = body.detect_fps
    if body.detect_mode is not None:
        # "" clears back to inherit (NULL); a valid mode is stored verbatim.
        cam["detect_mode"] = body.detect_mode or None
    # Live-view audio codec preference. None (omitted) keeps the stored value; a
    # value must be exactly "g711a" or "aac" (invalid -> a clean 400, not 422).
    audio_codec_changed = False
    if body.audio_codec is not None:
        codec = body.audio_codec.strip().lower()
        if codec not in ("g711a", "aac"):
            raise HTTPException(
                status_code=400, detail='audio_codec must be "g711a" or "aac"'
            )
        audio_codec_changed = codec != (existing.get("audio_codec") or "g711a")
        cam["audio_codec"] = codec
    # Smart-spotlight toggle. Persist-only — no device call here; the controller
    # reads the stored flag live on the next person-at-night detection.
    if body.smart_spotlight is not None:
        cam["smart_spotlight"] = bool(body.smart_spotlight)
    # Smart-spotlight trailing hold (seconds). None (omitted) keeps the stored
    # value; a value must be an int in [5, 600] (out-of-range -> a clean 400, not
    # 422). Persist-only — the controller reads (and re-clamps) the stored value.
    if body.spotlight_hold_seconds is not None:
        hold = body.spotlight_hold_seconds
        if isinstance(hold, bool) or not isinstance(hold, int) or not (5 <= hold <= 600):
            raise HTTPException(
                status_code=400,
                detail="spotlight_hold_seconds must be an integer between 5 and 600",
            )
        cam["spotlight_hold_seconds"] = hold
    if body.main_url is not None:
        cam["main_url"] = body.main_url
    if body.sub_url is not None:
        cam["sub_url"] = body.sub_url
    if body.model != existing["model"]:
        width, height = MODEL_DETECT_DEFAULTS.get(body.model, (704, 480))
        cam["detect_width"], cam["detect_height"] = width, height
    probed = await _probe_caps(cam)
    if cam["model"].strip().lower() in ("", UNKNOWN_MODEL):
        # Cameras can start as "unknown"; once an edit supplies working
        # credentials the probe's getDeviceType names the device — adopt the
        # matching known model (best-effort: any failure keeps "unknown").
        matched = match_known_model(probed.get("device_type"))
        if matched:
            cam["model"] = matched
            width, height = MODEL_DETECT_DEFAULTS.get(matched, (704, 480))
            cam["detect_width"], cam["detect_height"] = width, height
    cam["capabilities"] = merge_capabilities(cam["model"], probed)
    await state.db.upsert_camera(cam)
    await _apply_camera_change(request)
    # A changed live-view audio codec is re-provisioned onto the device encoder
    # (best-effort, NEVER fatal): "g711a" -> G.711A so the native RTSP audio is
    # WebRTC-legal and go2rtc/WebRTC live-view audio works; "aac" -> AAC for
    # higher recording quality (no live-view audio). An offline device / rejected
    # CGI only logs — the periodic time-sync loop re-applies it later. The
    # preference is already persisted above regardless of the device call.
    if audio_codec_changed:
        client = _client_for(cam)
        try:
            await client.provision_audio(
                "AAC" if cam["audio_codec"] == "aac" else "G.711A"
            )
        except AmcrestError as exc:
            log.info("audio-codec %s: not re-provisioned now (%s)", name, exc)
        finally:
            await client.aclose()
    return _camera_response(
        cam,
        state.prober.is_online(name),
        default_mode=_default_mode_of(state),
        ai_active=_ai_active_of(state, name),
        private=privacy.is_private(state, name),
        is_admin=True,   # route is admin-gated
    )


@router.delete("/{name}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_camera(name: str, request: Request) -> Response:
    state = request.app.state
    await _get_cam_or_404(request, name)
    await state.db.delete_camera(name)
    await _apply_camera_change(request)
    return Response(status_code=204)


# ---------- on-demand probe ("Save & Test") ----------


def _probe_failure(cam: dict[str, Any], detail: str) -> dict[str, Any]:
    # device_type mirrors the success shape (always present) so clients can
    # read it unconditionally; None here — the device never told us.
    return {
        "ok": False,
        "model": None,
        "device_type": None,
        "capabilities": _caps_of(cam),
        "detail": detail,
    }


@router.post("/{name}/probe", dependencies=[Depends(require_admin)])
async def probe_camera(name: str, request: Request) -> dict[str, Any]:
    """Probe the device with the stored credentials right now:
    getDeviceType + capability probe, 8 s total cap. On success adopts a
    matched model when the stored one is "unknown"/empty (same logic as the
    PUT re-probe) and refreshes cached capabilities."""
    state = request.app.state
    existing = await _get_cam_or_404(request, name)
    cam = dict(existing)
    deadline = time.monotonic() + _ONDEMAND_PROBE_CAP_S
    client = _client_for(cam)
    try:
        # Direct getDeviceType (not the error-swallowing background probe) so
        # bad credentials and dead hosts produce distinct, human details.
        try:
            device_type = await asyncio.wait_for(
                client.fetch_device_type(), timeout=_ONDEMAND_PROBE_CAP_S
            )
        except asyncio.TimeoutError:
            return _probe_failure(cam, "camera unreachable")
        except AmcrestAuthError:
            return _probe_failure(cam, "authentication failed")
        except AmcrestUnsupportedError:
            device_type = None  # device answered; getDeviceType just missing
        except AmcrestError:
            return _probe_failure(cam, "camera unreachable")

        probed: dict[str, Any] = {}
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                probed = await asyncio.wait_for(probe_capabilities(client), timeout=remaining)
            except (asyncio.TimeoutError, AmcrestError):
                probed = {}
        if device_type and not probed.get("device_type"):
            probed["device_type"] = device_type
    finally:
        await client.aclose()

    model_adopted = False
    if cam["model"].strip().lower() in ("", UNKNOWN_MODEL):
        matched = match_known_model(device_type)
        if matched:
            cam["model"] = matched
            width, height = MODEL_DETECT_DEFAULTS.get(matched, (704, 480))
            cam["detect_width"], cam["detect_height"] = width, height
            model_adopted = True
    cam["capabilities"] = merge_capabilities(cam["model"], probed)

    if cam != existing:
        await state.db.upsert_camera(cam)
        if model_adopted:
            # Detect dimensions changed with the model — the native engines
            # and go2rtc config must pick that up.
            await _apply_camera_change(request)
        else:
            # Capabilities-only refresh: a newly-found doorbell/speaker may
            # need a watcher, but streams/engines are untouched.
            await state.doorbells.sync(await state.db.list_cameras())

    # `model` favors the adopted/stored model; falls back to the raw
    # getDeviceType string so the UI can still show what the device said.
    model = cam["model"]
    if model.strip().lower() in ("", UNKNOWN_MODEL):
        model = device_type or None
    else:
        # Stored model is a deliberate pick, so we don't touch it — but if the
        # device disagrees, say so. A mislabeled camera gets silently wrong
        # feature gating (missing spotlight/PTZ/talk, wrong detect dims) and
        # nothing else in the system would ever notice.
        reported = match_known_model(device_type)
        if reported and reported != cam["model"]:
            log.warning(
                "model-detect %s: stored model %s but device reports %r (%s) — "
                "feature gating may be wrong; set the model to Auto-detect and "
                "Save & Test to re-detect",
                name, cam["model"], device_type, reported,
            )
    # device_type is always returned (not just when the stored model is blank)
    # so the UI can surface a stored-vs-reported disagreement to a human.
    return {
        "ok": True,
        "model": model,
        "device_type": device_type,
        "capabilities": _caps_of(cam),
        "detail": None,
    }


# ---------- Amcrest device settings ----------


@router.get("/{name}/settings", dependencies=[Depends(require_admin)])
async def get_device_settings(name: str, request: Request) -> dict[str, Any]:
    cam = await _get_cam_or_404(request, name)
    caps = _caps_of(cam)
    client = _client_for(cam)
    result: dict[str, Any] = {}
    reachable = False
    try:
        async def grab(key: str, coro) -> None:
            nonlocal reachable
            try:
                value = await coro
            except AmcrestUnsupportedError:
                reachable = True  # device answered; feature just missing
                return
            except AmcrestError:
                return
            reachable = True
            if value is not None:
                result[key] = value

        items = [("flip", client.get_flip()),
                 ("osd_name", client.get_osd_name()),
                 ("motion_detect", client.get_motion_detect()),
]
        if caps.get("ir"):
            # get_ir returns {mode}; unpacked into the flat ir_mode below
            # (IR is mode-only — no brightness).
            items.append(("ir", client.get_ir()))
            items.append(("day_night", client.get_day_night()))
        if caps.get("night_vision"):
            # Flat "auto"|"color"|"bw" from the VideoInDayNight day/night table.
            items.append(("night_vision_mode", client.get_night_vision_mode()))
        if caps.get("white_light"):
            items.append(("white_light", client.get_white_light()))
        if caps.get("speaker"):
            items.append(("speaker_volume", client.get_speaker_volume()))
        # One request must pay the digest 401->200 challenge before the rest
        # can reuse the cached challenge (httpx.DigestAuth caches only AFTER a
        # response returns), so prime it with the first grab, then fan out the
        # remainder concurrently. Firing all grabs at once would make every
        # one pay the full challenge round trip.
        first_key, first_coro = items[0]
        await grab(first_key, first_coro)
        await asyncio.gather(*(grab(key, coro) for key, coro in items[1:]))
    finally:
        await client.aclose()

    if not reachable:
        raise HTTPException(status_code=502, detail=f"Camera '{name}' is unreachable")
    if "speaker_volume" in result:
        result["volume"] = {"speaker": result.pop("speaker_volume")}
    # Unpack the IR read into the flat contract: ir_mode only (mode-only IR).
    ir = result.pop("ir", None)
    if isinstance(ir, dict) and ir.get("mode") is not None:
        result["ir_mode"] = ir["mode"]
    return result


@router.put("/{name}/settings", dependencies=[Depends(require_admin)])
async def put_device_settings(name: str, body: DeviceSettingsPatch, request: Request) -> dict[str, Any]:
    cam = await _get_cam_or_404(request, name)
    caps = _caps_of(cam)
    client = _client_for(cam)
    failures: list[str] = []
    # Track the IR state that actually landed on the device so it can be
    # persisted (and later re-asserted on doorbell stream reconnect).
    ir_applied: dict[str, Any] = {}
    try:
        if body.ir_mode is not None:
            if not caps.get("ir"):
                raise HTTPException(status_code=400, detail="Camera has no IR control")
            try:
                # set_ir writes Mode to EVERY Lighting profile (the camera obeys
                # whichever day/night profile is active). "On" = Manual at the
                # camera's stored illuminator strength — no brightness control.
                await client.set_ir(mode=body.ir_mode)
                ir_applied["mode"] = body.ir_mode
            except AmcrestError as exc:
                failures.append(f"ir: {exc}")
        if body.day_night is not None:
            if not caps.get("ir"):
                raise HTTPException(status_code=400, detail="Camera has no IR control")
            try:
                await client.set_day_night(body.day_night)
                ir_applied["day_night"] = body.day_night
            except AmcrestError as exc:
                failures.append(f"day_night: {exc}")
        if body.night_vision_mode is not None:
            if not caps.get("night_vision"):
                raise HTTPException(status_code=400, detail="Camera has no night-vision mode")
            try:
                await client.set_night_vision_mode(body.night_vision_mode)
                # PERSIST it: without this the chosen mode lived only on the
                # camera, and the AD410 resets day/night to Auto on every RTSP
                # (re)connect — so "full colour" silently reverted and there was
                # nothing stored for the re-asserter to restore.
                ir_applied["night_vision_mode"] = body.night_vision_mode
            except AmcrestError as exc:
                failures.append(f"night_vision_mode: {exc}")
        if body.white_light is not None:
            if not caps.get("white_light"):
                raise HTTPException(status_code=400, detail="Camera has no white light")
            # Drop brightness on on/off-only illuminators (coax turrets); it has
            # no effect there and the UI is told so via white_light_control.
            wl_brightness = (
                body.white_light.brightness
                if white_light_control_for_model(cam["model"])["brightness"]
                else None
            )
            if body.white_light.mode is not None or wl_brightness is not None:
                try:
                    await client.set_white_light(
                        mode=body.white_light.mode, brightness=wl_brightness
                    )
                except AmcrestError as exc:
                    failures.append(f"white_light: {exc}")
        if body.flip is not None:
            try:
                await client.set_flip(body.flip)
            except AmcrestError as exc:
                failures.append(f"flip: {exc}")
        if body.osd_name is not None:
            try:
                await client.set_osd_name(body.osd_name)
            except AmcrestError as exc:
                failures.append(f"osd_name: {exc}")
        if body.motion_detect is not None:
            try:
                await client.set_motion_detect(body.motion_detect)
            except AmcrestError as exc:
                failures.append(f"motion_detect: {exc}")
        # (No privacy_mode here any more — the camera's LeLensMask blackout was
        # replaced by software Privacy Mode, which never touches the device.)
        if body.volume is not None and body.volume.speaker is not None:
            if not caps.get("speaker"):
                raise HTTPException(status_code=400, detail="Camera has no speaker")
            try:
                await client.set_speaker_volume(body.volume.speaker)
            except AmcrestError as exc:
                failures.append(f"volume.speaker: {exc}")
        # volume.mic: no verified public CGI on these models — ignored.
    finally:
        await client.aclose()

    # Persist the operator's desired IR (merged over any prior pin) so a
    # doorbell that reverts IR on RTSP reconnect can be re-asserted to it.
    if ir_applied:
        merged = {**(cam.get("ir_state") or {}), **ir_applied}
        await request.app.state.db.set_camera_ir_state(name, merged)

    if failures:
        raise HTTPException(status_code=502, detail="; ".join(failures))
    return await get_device_settings(name, request)


# ---------- actions ----------


@router.post("/{name}/light", status_code=204, dependencies=[Depends(require_admin)])
async def set_light(name: str, body: LightRequest, request: Request) -> Response:
    """White light / spotlight control.

    EW turrets drive the spotlight over coaxialControlIO, which is on/off only
    — brightness is DROPPED for those models (the coax CGI has no brightness
    parameter) so the UI's brightness slider, if any stale client still sends
    it, never silently pretends to work. Non-coax models still take the full
    Dahua Lighting_V2 white-LED brightness (config names verified against
    rroller/dahua — see amcrest/client.py). white_light_control_for_model()
    is the authoritative capability the camera response advertises."""
    cam = await _get_cam_or_404(request, name)
    if not _caps_of(cam).get("white_light"):
        raise HTTPException(status_code=400, detail="Camera has no white light / spotlight")
    control = white_light_control_for_model(cam["model"])
    # Drop brightness on on/off-only illuminators (coaxialControlIO turrets).
    brightness = body.brightness if control["brightness"] else None
    client = _client_for(cam)
    try:
        await client.set_white_light(mode=body.mode, brightness=brightness)
    except AmcrestUnsupportedError as exc:
        raise HTTPException(status_code=501, detail=f"No supported white-light CGI on this device: {exc}")
    except AmcrestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        await client.aclose()
    return Response(status_code=204)


@router.post("/{name}/siren", status_code=204, dependencies=[Depends(require_admin)])
async def sound_siren(name: str, request: Request, body: SirenRequest = SirenRequest()) -> Response:
    cam = await _get_cam_or_404(request, name)
    if not _caps_of(cam).get("siren"):
        raise HTTPException(status_code=400, detail="Camera has no siren")
    client = _client_for(cam)
    try:
        await client.play_tone(body.duration_s)
    except AmcrestUnsupportedError:
        # Verified against python-amcrest + community Amcrest bridge sources:
        # the AD410 exposes no public siren CGI; audio.cgi tone playback is
        # the only supported path, and this firmware rejected it.
        raise HTTPException(
            status_code=501,
            detail=(
                "This device has no public siren API; playing an alarm tone via "
                "audio.cgi was rejected by the firmware."
            ),
        )
    except AmcrestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        await client.aclose()
    return Response(status_code=204)


# Viewer-accessible (router-level require_auth still applies): aiming a camera
# is a live-viewing action, like watching the stream — not an admin
# configuration change. It mutates no stored state and is bounded by the
# camera's own PTZ limits. Deliberate product decision; see the RBAC matrix in
# docs/CONTRACTS.md and the viewer-allowed list in tests/rbac_smoke.py.
@router.post("/{name}/ptz", status_code=204)
async def ptz_control(name: str, body: PtzRequest, request: Request) -> Response:
    """Pan/tilt + preset control for PTZ domes (capability-gated on `ptz`).

    Drives the camera's Dahua ptz.cgi (continuous move/stop, a bounded "step"
    nudge, + 3 presets). A firmware without ptz.cgi -> 501; a transient device
    rejection -> 502.
    """
    cam = await _get_cam_or_404(request, name)
    if not _caps_of(cam).get("ptz"):
        raise HTTPException(status_code=400, detail="Camera has no PTZ")
    if body.action in ("move", "step", "stop") and body.direction is None:
        raise HTTPException(status_code=422, detail="direction is required for move/step/stop")
    if body.action.startswith("preset_") and body.index is None:
        raise HTTPException(status_code=422, detail="index (1-3) is required for preset actions")
    client = _client_for(cam)
    try:
        if body.action == "move":
            await client.ptz_move(body.direction, body.speed)
        elif body.action == "step":
            await client.ptz_step(body.direction)
        elif body.action == "stop":
            await client.ptz_stop(body.direction)
        else:
            await client.ptz_preset(body.action, body.index)
    except AmcrestUnsupportedError as exc:
        raise HTTPException(status_code=501, detail=f"No supported PTZ CGI on this device: {exc}")
    except AmcrestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        await client.aclose()
    return Response(status_code=204)


@router.post("/{name}/reboot", status_code=204, dependencies=[Depends(require_admin)])
async def reboot_camera(name: str, request: Request) -> Response:
    cam = await _get_cam_or_404(request, name)
    client = _client_for(cam)
    try:
        await client.reboot()
    except AmcrestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        await client.aclose()
    return Response(status_code=204)


# Media route: token query param allowed (see auth.require_media_auth).
snapshot_router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@snapshot_router.get("/{name}/snapshot.jpg", dependencies=[Depends(require_media_auth)])
async def camera_snapshot(name: str, request: Request) -> Response:
    cam = await _get_cam_or_404(request, name)
    state = request.app.state
    # PRIVACY MODE GATE (app/privacy.py) — before ANY frame is fetched. This
    # route is the one live-image path that bypasses go2rtc completely: the
    # fallback below hits the camera's CGI directly, and latest_jpg can return a
    # hot cached frame. Removing the go2rtc streams does NOT cover it.
    #
    # Placed before the media/CGI calls so a private camera is never even
    # contacted. Fires for MEDIA-TOKEN requests too (this route accepts the
    # long-lived tokens embedded in notifications/MQTT) — those must not become
    # a privacy bypass.
    if privacy.is_private(state, name):
        raise HTTPException(status_code=403, detail="Camera is in Privacy Mode")
    jpeg = await state.media.latest_jpg(name, height=720)
    if jpeg is None:
        # Ingest down (or detection disabled) — go straight to the camera.
        client = _client_for(cam)
        try:
            jpeg = await client.snapshot()
        except AmcrestError:
            jpeg = None
        finally:
            await client.aclose()
    if jpeg is None:
        raise HTTPException(status_code=502, detail="No live snapshot available (ingest and camera unreachable)")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

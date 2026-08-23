"""GET/PUT /api/settings — validated KV settings merged over defaults.

Side effects on PUT:
- system.webrtc_candidates change  -> regenerate + sync the go2rtc config
- detection.model/confidence change -> engine reload (detector reconfigure)
- recording changes                 -> picked up live by the pruner/recorder

Legacy blocks from removed features (notifications.ntfy, detection.audio_*)
are silently dropped by the Pydantic models here and by settings_store on
load — an old /data volume must never 500.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..auth import require_admin
from ..config import DEFAULT_CAMERA_TIMEZONE
from ..native.recorder import SEGMENT_SECONDS, max_clip_post_s
from ..native.streams import webrtc_status
from .detection import activate_model

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"], dependencies=[Depends(require_admin)])

_LABEL_MAX = 32


class ApnsSettings(BaseModel):
    """APNs (iOS) push per docs/push-architecture.md §4.

    **"direct" is GONE.** The relay is the only APNs transport: it holds the
    .p8, so a self-hoster needs no Apple developer account, and it is what
    delivers a real native notification plus the CallKit doorbell ring. (ntfy
    is a separate channel and stays — no Apple account at all — but its alerts
    land in the *ntfy app*: no ring, no native UI. That is exactly why the
    relay came back after being retired.)

    A stored ``mode="direct"`` is migrated to "off" in
    ``settings_store._strip_legacy`` BEFORE it can reach this Literal. That is
    load-bearing, not tidiness: an unmigrated blob 422s EVERY settings save and
    locks the admin out of the settings page — including out of changing the
    mode. And because pydantic defaults to ``extra="ignore"``, a `direct` block
    left in a stored document is silently dropped on the next save of ANY
    setting; the p8 it held is gone with it. (The owner's key was rescued to
    secrets/ before this landed.)
    """

    # Default "off", NOT "relay". mode="relay" with an empty relay_url would
    # error on every event; and defaulting to a baked-in owner URL would make a
    # fresh install silently push through someone else's Apple credentials the
    # moment a phone registers. The admin opts in.
    mode: Literal["relay", "off"] = "off"
    relay_url: str = Field(default="", max_length=256)

    @field_validator("relay_url")
    @classmethod
    def _relay_url_clean(cls, v: str) -> str:
        # MUST accept http://push-relay:8090 — that is the owner's own correct
        # value (the compose service name on the shared default network). A
        # validator demanding https or a public TLD would lock him out of the
        # one config that actually works.
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("relay_url must be an http(s) URL")
        return v


class NtfySettings(BaseModel):
    """ntfy (ntfy.sh or self-hosted) — push without Apple credentials.

    The whole point of this channel: a self-hoster installs the ntfy app,
    points this at a server + topic, and gets notifications. No Apple
    developer account, no .p8, no relay run by anyone else.

    SECURITY — the topic IS the password. ntfy's own docs say so: on a server
    with default-allow access (including ntfy.sh), ANYONE who knows the topic
    receives every message on it. Notification text alone maps when a house is
    empty, and `attach_snapshot` links a media URL. So `topic` DEFAULTS TO
    EMPTY and the UI generates an unguessable one — never a memorable name.
    Self-host with `auth-default-access: deny-all` and set `auth_token` for a
    real permission model.
    """

    enabled: bool = False
    server: str = Field(default="https://ntfy.sh", max_length=256)
    topic: str = Field(default="", max_length=64)
    # ntfy access token ("tk_..."). Sent as `Authorization: Bearer`. Required
    # by a deny-all self-hosted server; also unlocks ntfy.sh reserved topics.
    auth_token: str = Field(default="", max_length=128)
    priority: int = Field(default=4, ge=1, le=5)  # ntfy scale: 1 min .. 5 urgent
    # Link the event snapshot via ntfy's `Attach` header. The PHONE fetches it
    # straight from this NVR — the image never touches the ntfy server. Needs
    # system.public_url reachable from the phone. Off => text-only.
    attach_snapshot: bool = True

    @field_validator("server")
    @classmethod
    def _server_clean(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("ntfy server must be an http(s) URL")
        return v

    @field_validator("topic")
    @classmethod
    def _topic_clean(cls, v: str) -> str:
        v = v.strip().strip("/")
        # ntfy's own rule. Also keeps the topic a single path segment, so it
        # cannot smuggle a path/query into the publish URL.
        if v and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v):
            raise ValueError("ntfy topic may contain only A-Z a-z 0-9 _ - (max 64)")
        return v

    @field_validator("auth_token")
    @classmethod
    def _token_clean(cls, v: str) -> str:
        return v.strip()


class NotificationSettings(BaseModel):
    enabled: bool = True
    labels: list[str] = Field(default_factory=lambda: ["person", "dog", "cat", "car"])
    cooldown_seconds: int = Field(default=60, ge=0, le=86400)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    # Draw detection boxes/labels on event snapshots (banner always stays).
    draw_boxes: bool = True
    # Push a system alert when a camera stops responding. OFF by default: a
    # debounced down is still noisier than most detection alerts, and it fires
    # via the UNCOUPLED push primitive, so it is NOT filtered by the labels /
    # min_score / cooldown gates above. See camera_health.py.
    camera_down_alerts: bool = False
    apns: ApnsSettings = Field(default_factory=ApnsSettings)
    ntfy: NtfySettings = Field(default_factory=NtfySettings)

    @field_validator("labels")
    @classmethod
    def _labels_clean(cls, v: list[str]) -> list[str]:
        return [s.strip().lower()[:_LABEL_MAX] for s in v if s.strip()]


class RecordingSettings(BaseModel):
    continuous_days: int = Field(default=7, ge=0, le=365)
    event_days: int = Field(default=14, ge=0, le=365)
    snapshot_days: int = Field(default=14, ge=0, le=365)
    # Space-based rotation ("overwrite the oldest with the newest"), applied on
    # top of the day-based cutoffs above — whichever frees a recording first
    # wins. Only 24/7 continuous footage rotates; event clips are never deleted
    # for space, only by event_days.
    #
    # 0 disables the cap: recordings grow until min_free_gb stops them.
    # The cap is what you want when the recordings disk is shared with other
    # data (an Unraid array), since the free-space floor alone lets Vigilume
    # consume everything else's headroom before it reacts.
    max_storage_gb: int = Field(default=0, ge=0, le=1_000_000)
    # Never let the recordings filesystem fall below this much free space. The
    # backstop that keeps the box healthy even if something ELSE fills the disk.
    min_free_gb: int = Field(default=5, ge=1, le=10_000)
    # Event clip padding either side of the detected event (see DEFAULT_SETTINGS
    # for why pre-roll usually wants to be the larger of the two).
    clip_pre_s: int = Field(default=5, ge=0, le=120)
    # Post-roll is capped where pre-roll is not, and the cap is a fact about the
    # recorder rather than a matter of taste: extraction runs clip_delay_s after
    # the event ends, and a segment is only on disk once its SEGMENT_SECONDS are
    # up, so footage past that horizon has not been written yet. Asking for more
    # would not fail loudly — ffmpeg would just stop at the end of what exists
    # and hand back a clip quietly shorter than requested. The real bound is
    # therefore clip_delay_s-dependent and enforced in _check_clip_window below;
    # this `le` is only the absolute ceiling at the largest allowed delay.
    clip_post_s: int = Field(default=5, ge=0, le=290)
    # Waiting longer before cutting the clip is what buys post-roll past the
    # default. Floor of SEGMENT_SECONDS: below one segment length nothing has
    # been flushed and every clip would lose its tail entirely.
    clip_delay_s: int = Field(default=20, ge=SEGMENT_SECONDS, le=300)

    @model_validator(mode="after")
    def _check_clip_window(self) -> "RecordingSettings":
        """Reject post-roll the delay cannot deliver, instead of silently
        truncating it — a clip shorter than configured, with nothing logged, is
        the kind of thing an operator only discovers when they need the footage.
        """
        reachable = max_clip_post_s(self.clip_delay_s)
        if self.clip_post_s > reachable:
            raise ValueError(
                f"clip_post_s={self.clip_post_s}s needs clip_delay_s of at least "
                f"{self.clip_post_s + SEGMENT_SECONDS}s (a segment is only on disk "
                f"{SEGMENT_SECONDS}s after it opens); at clip_delay_s="
                f"{self.clip_delay_s}s the most that can be captured is {reachable}s"
            )
        return self


class DetectionSettings(BaseModel):
    # Model pins/hashes live in native/detector.py; a model change triggers
    # download (if absent) + engine reload.
    # Any model key the registry knows (dfine_n/s/m/l/x, dfine_l_obj365, future
    # detector models). Plain str, not a Literal, so adding a model never breaks
    # settings validation — the model store / detector is the source of truth for
    # valid keys, and activate_model handles an unknown one gracefully.
    model: str = Field(default="dfine_s", min_length=1, max_length=64)
    confidence: float = Field(default=0.5, ge=0.2, le=0.9)
    # Effective server-detection mode for cameras whose per-camera detect_mode is
    # unset/NULL (see config.VALID_DETECT_MODES). Default "always" (continuous
    # server inference — the safe default for a security system): detection can
    # never silently stop because of an unvalidated camera-AI gate. The camera-AI
    # modes stay available as an explicit per-camera opt-in (the toggle): set a
    # camera to "camera_ai" so the GPU only runs while its own AI fires — the load
    # win — or "camera_ai_only" for no server inference. A camera WITHOUT on-board
    # AI (capabilities.ai_on_camera=False) degrades to "always" automatically in
    # effective_detect_mode, so nothing silently stops detecting.
    default_mode: Literal["always", "camera_ai", "camera_ai_only"] = "always"
    # Which silicon runs inference. Applied at BOOT (build_detector), so a change
    # here needs a backend restart — unlike model/confidence, which reconfigure
    # the live detector. Default "gpu": Coral is never opted into uninvited, and
    # an install with no Edge TPU fitted must keep detecting.
    # "auto" is the default: use an Edge TPU when the box has one, else the
    # GPU. "gpu"/"coral" remain as explicit overrides for an operator who wants
    # to pin one regardless of what is fitted.
    backend: Literal["auto", "gpu", "coral"] = "auto"
    # Edge TPU model key. Separate from `model` because the GPU (D-FINE tiers)
    # and Coral lists are disjoint — one field would let an invalid pair be
    # stored the instant backend flips. Validated against the registry.
    coral_model: Literal[
        "ssd_mobilenet_v2", "ssdlite_mobiledet", "efficientdet_lite0",
        "efficientdet_lite1", "efficientdet_lite2", "efficientdet_lite3",
    ] = "ssdlite_mobiledet"
    # How long a label may go unseen before its event ends. Floor of 1 s so a
    # single dropped frame cannot end an event; ceiling of 300 s because the
    # clip is not cut until the event ends, and an event that never closes is
    # an event whose footage never arrives.
    absence_timeout_s: int = Field(default=5, ge=1, le=300)


class AutoRestartSettings(BaseModel):
    """Optional nightly restart of the backend process.

    The restart is a SIGTERM to ourselves: the app's shutdown hooks run (ffmpeg
    children terminated, engine stopped), the process exits, and the container's
    `restart: unless-stopped` policy brings it straight back — roughly a
    15-second gap in recording and live view. Nothing is deleted.

    Off by default. Only meaningful under a restart policy / supervisor: run
    bare (a dev shell) the process would exit and simply stay down, which is why
    the UI says so next to the toggle.
    """

    enabled: bool = False
    # Local wall-clock 24h "HH:MM" (the container's TZ), not UTC — an operator
    # picking "04:00" means 4am where the cameras are.
    time: str = Field(default="04:00", max_length=5)

    @field_validator("time")
    @classmethod
    def _time_valid(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("time must be 24-hour HH:MM (e.g. 04:00)")
        return v


class SystemSettings(BaseModel):
    public_url: str = Field(default="", max_length=256)
    auto_restart: AutoRestartSettings = Field(default_factory=AutoRestartSettings)
    # Extra WebRTC ICE host candidates for go2rtc ("ip:8555" entries — LAN
    # and/or Tailscale IPs). Empty = go2rtc defaults (STUN only).
    webrtc_candidates: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("public_url")
    @classmethod
    def _url_clean(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("public_url must be an http(s) URL")
        return v

    @field_validator("webrtc_candidates")
    @classmethod
    def _candidates_clean(cls, v: list[str]) -> list[str]:
        cleaned = []
        for item in v:
            item = item.strip()
            if not item:
                continue
            if len(item) > 64:
                raise ValueError("webrtc candidate entries must be at most 64 characters")
            cleaned.append(item)
        return cleaned


class MqttSettings(BaseModel):
    """Outbound MQTT + Home Assistant discovery publisher config. Publish-only;
    changing any field restarts the publisher (reconnect) with no app restart.
    Admin-only (the whole /api/settings router is require_admin gated)."""

    enabled: bool = False
    host: str = Field(default="", max_length=253)
    port: int = Field(default=1883, ge=1, le=65535)
    username: str = Field(default="", max_length=128)
    password: str = Field(default="", max_length=256)
    discovery_prefix: str = Field(default="homeassistant", min_length=1, max_length=64)
    base_topic: str = Field(default="vigilume", min_length=1, max_length=64)

    @field_validator("host")
    @classmethod
    def _host_clean(cls, v: str) -> str:
        return v.strip()

    @field_validator("discovery_prefix", "base_topic")
    @classmethod
    def _topic_clean(cls, v: str) -> str:
        # Topic prefixes must not contain MQTT wildcards, spaces, or a leading/
        # trailing slash — they are concatenated into concrete topic strings.
        v = v.strip().strip("/")
        if not v or any(ch in v for ch in ("+", "#", " ")):
            raise ValueError("must be a non-empty topic segment without + # or spaces")
        return v


class TimeSyncSettings(BaseModel):
    """Automatic camera time provisioning. When ``auto_sync`` is on, each
    reachable Dahua/Amcrest camera has its clock set to the current local time
    in ``timezone`` (an IANA zone name) and its on-device NTP client disabled —
    on connect and periodically. NTP + the device timezone index are NOT used
    (both are unreliable on these units). See amcrest/time_sync.py."""

    auto_sync: bool = True
    timezone: str = Field(default=DEFAULT_CAMERA_TIMEZONE, max_length=64)

    @field_validator("timezone")
    @classmethod
    def _tz_valid(cls, v: str) -> str:
        # Must be a real IANA zone name (e.g. America/New_York) — it is resolved
        # with zoneinfo at provisioning time. Empty falls back to the default.
        v = v.strip() or DEFAULT_CAMERA_TIMEZONE
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"timezone must be a valid IANA zone name (got {v!r})") from exc
        return v


class AppSettings(BaseModel):
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    mqtt: MqttSettings = Field(default_factory=MqttSettings)
    time_sync: TimeSyncSettings = Field(default_factory=TimeSyncSettings)


def _with_webrtc(settings: dict[str, Any]) -> dict[str, Any]:
    """Attach the read-only computed ``webrtc`` block (effective candidates +
    readiness + a detected-IP hint) to a settings document for the response. It
    is NOT part of the writable AppSettings model — an old field sent back on
    PUT is silently ignored — so the System tab can warn "live is on slow
    fallback" and pre-fill the server's detected IP with one click."""
    settings["webrtc"] = webrtc_status(settings)
    return settings


# Stand-in returned in place of a stored secret. A client that sends it back
# unchanged means "leave this alone" — see _carry_forward_secrets.
SECRET_MASK = "********"

# (path-into-the-settings-doc) for every value that is a credential rather than
# a setting. Both are write-only from the client's point of view.
_SECRET_PATHS: tuple[tuple[str, ...], ...] = (
    ("mqtt", "password"),
    ("notifications", "ntfy", "auth_token"),
)


def _dig(doc: dict[str, Any], path: tuple[str, ...]) -> Optional[dict[str, Any]]:
    """The parent dict holding path[-1], or None if the branch is absent."""
    node: Any = doc
    for key in path[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return None
    return node


def _mask_secrets(doc: dict[str, Any]) -> dict[str, Any]:
    """Replace stored credentials with SECRET_MASK for the response.

    The settings routes are admin-gated, so this is not an anonymous leak — but
    without it the MQTT password and ntfy token are pulled into browser memory
    and into the iOS client on EVERY settings load, and land in any devtools or
    HAR capture taken while debugging. A credential the client never needs to
    read should not be handed to it.
    """
    out = copy.deepcopy(doc)
    for path in _SECRET_PATHS:
        parent = _dig(out, path)
        if parent and parent.get(path[-1]):
            parent[path[-1]] = SECRET_MASK
    return out


def _carry_forward_secrets(incoming: dict[str, Any], previous: dict[str, Any]) -> None:
    """Restore any secret the client echoed back masked (or blank), in place.

    Without this, masking would DESTROY both credentials on the first save from
    any client that round-trips the document — exactly the full-replace footgun
    that PUT /api/settings already has a history of.

    BLANK IS TREATED AS "UNCHANGED", NOT AS "CLEAR". That is deliberate and it
    does cost something: there is no longer a way to empty these two fields by
    saving an empty string. The alternative is worse — every field on this model
    carries a default, so a client that simply OMITS the key sends "", and
    honouring that as "clear" is precisely how omitting the APNs p8 silently
    destroyed the signing key. Disabling the integration (`mqtt.enabled`,
    `notifications.ntfy.enabled`) is the real intent behind "clear" anyway, and
    it still works.
    """
    for path in _SECRET_PATHS:
        dst, src = _dig(incoming, path), _dig(previous, path)
        if dst is None or src is None:
            continue
        if dst.get(path[-1]) in ("", SECRET_MASK, None):
            dst[path[-1]] = src.get(path[-1], "")


@router.get("")
async def get_settings(request: Request) -> dict[str, Any]:
    return _mask_secrets(_with_webrtc(request.app.state.settings.get()))


@router.put("")
async def put_settings(body: AppSettings, request: Request) -> dict[str, Any]:
    state = request.app.state
    previous = state.settings.get()
    payload = body.model_dump()
    # BEFORE the write: a client that received the mask and sent it back must
    # not overwrite the real credential with asterisks.
    _carry_forward_secrets(payload, previous)
    updated = await state.settings.update(payload)

    if previous["detection"] != updated["detection"]:
        log.info("detection settings changed — reloading engine")
        # Route model changes through the SAME activate path as
        # POST /api/detection/models/{key}/activate: reconfigure the detector
        # (non-blocking) + start the store download of the (possibly new)
        # active model. Confidence-only changes are picked up by the reload.
        await activate_model(state, updated["detection"]["model"])
        # A default_mode change may start/stop camera-AI watchers for cameras
        # with an unset per-camera mode. Resync them (engine.reload, invoked by
        # activate_model, already re-read default_mode for the ingest gate).
        if previous["detection"].get("default_mode") != updated["detection"].get("default_mode"):
            ai_events = getattr(state, "ai_events", None)
            if ai_events is not None:
                await ai_events.sync(
                    await state.db.list_cameras(),
                    default_mode=updated["detection"]["default_mode"],
                )
    if previous["system"].get("webrtc_candidates") != updated["system"].get("webrtc_candidates"):
        log.info("webrtc_candidates changed — regenerating go2rtc config")
        await state.go2rtc.apply()
    if previous.get("mqtt") != updated.get("mqtt"):
        log.info("mqtt settings changed — restarting Home Assistant publisher")
        # restart() reloads config from settings and reconnects with the new
        # host/creds/topics (or stops the publisher when it was disabled). It
        # never raises — a down broker must not fail a settings save.
        mqtt = getattr(state, "mqtt", None)
        if mqtt is not None:
            await mqtt.restart()
    return _mask_secrets(_with_webrtc(updated))


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``patch`` onto ``base``. Only dict-vs-dict recurses;
    any other value (including a list) replaces wholesale."""
    out = dict(base)
    for key, value in patch.items():
        current = out.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            out[key] = _deep_merge(current, value)
        else:
            out[key] = value
    return out


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """JSON-safe pydantic errors for an HTTPException detail.

    `exc.errors()` is NOT JSON-serializable: for a validator that raises
    ValueError, pydantic v2 puts the raw exception OBJECT in each error's
    `ctx["error"]`. FastAPI then fails encoding the 422 body and the client gets
    a **500** instead — which is what happened to every custom-validator failure
    on PATCH (bad ntfy topic/server, public_url, mqtt topic segment, timezone).
    `include_context=False` drops that object; the human-readable text lives in
    `msg` ("Value error, ntfy topic may contain only …") and survives.
    `include_url=False` also strips the errors.pydantic.dev links, which are
    noise in an API response.

    PUT does not need this: its body is validated by FastAPI's own request
    parsing, which already serializes errors safely.
    """
    return exc.errors(include_url=False, include_context=False)


@router.patch("")
async def patch_settings(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Deep-merge a PARTIAL settings document over the stored one.

    Use this instead of PUT from any client that does not model the ENTIRE
    settings document. PUT is a full replace and every field carries a default,
    so an omitted key is not "left alone" — it is reset. Omitting
    ``notifications.apns.direct.p8`` on a PUT silently destroys the APNs signing
    key and breaks push. PATCH only touches the keys you send, e.g.
    ``{"detection": {"model": "dfine_s"}}``.

    Validation and every side-effect (detector reconfigure, go2rtc regen, MQTT
    restart) are identical to PUT — the merged document is handed straight to it.
    """
    merged = _deep_merge(request.app.state.settings.get(), body)
    try:
        validated = AppSettings(**merged)
    except ValidationError as exc:
        # Raised inside the handler (not by request parsing), so surface it as a
        # 422 rather than letting it become a bare 500.
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc
    return await put_settings(validated, request)

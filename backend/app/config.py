"""Environment configuration.

All paths are overridable via env so tests can point the app at temp dirs.
Container defaults match docker-compose.yml volume mounts.

Vigilume is standalone: its own detection (D-FINE ONNX), its own 24/7
recording (ffmpeg stream-copy) and its own go2rtc for live view. There are
no deployment modes and no legacy external-NVR/broker env vars.

Tunable env vars use the ``VIGILUME_*`` prefix. The legacy ``SENTINEL_*``
names are still read as a fallback (see ``env_dual``) so an existing .env
from before the rename keeps working untouched.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

APP_VERSION = "2.0.0"

# Inference backend selection (VIGILUME_DETECTOR). ``onnx`` (default) + the
# alternatives the detector factory (native.detector.build_detector) understands.
#   onnx       — D-FINE ONNX via onnxruntime; VIGILUME_REQUIRE_GPU gates CUDA/CPU
#   onnx_cpu   — D-FINE ONNX forced onto CPU (overrides VIGILUME_REQUIRE_GPU)
VALID_DETECTORS = ("onnx", "onnx_cpu", "coral", "auto")

# User-facing detection backends (settings.detection.backend) -> detector kind.
# Deliberately a SMALLER, friendlier set than VALID_DETECTORS: "onnx_cpu" is a
# debugging/fallback knob, not something to offer in a settings dropdown.
VALID_BACKENDS = ("auto", "gpu", "coral")
BACKEND_TO_DETECTOR = {"auto": "auto", "gpu": "onnx", "coral": "coral"}
DEFAULT_DETECTOR = "onnx"

# Camera models with a known static capability map (see amcrest/features.py).
KNOWN_MODELS = ("IP5M-T1277EW-AI", "IP8M-2779EW-AI", "AD410", "IP3M-941B", "IP4M-1041B", "IP4M-1056E")

# Detector model options (pins + hashes live in native/detector.py).
DETECTION_MODELS = ("dfine_n", "dfine_s", "dfine_m")

# Per-camera server-detection mode (cameras.detect_mode). Gates whether — and
# when — the GPU detector runs on a camera's frames, using the camera's OWN
# on-device AI (SMD human/vehicle, IVS tripwire/intrusion) as the trigger:
#   always         — continuous server inference (historical behavior).
#   camera_ai      — server inference runs ONLY while the camera's on-board AI
#                    is active (from an AI Start until Stop + a short cooldown);
#                    otherwise the detector idles that camera (the GPU win). The
#                    ffmpeg ingest still runs so live view / frame cache work.
#   camera_ai_only — NO server inference at all; events are created directly
#                    from the camera's AI event stream (label from the AI code,
#                    a snapshot grabbed from the stream), respecting the camera's
#                    detect_objects filter + notification cooldown.
# A camera's stored detect_mode may be NULL ("unset") — those inherit
# settings.detection.default_mode (itself defaulting to "always"), so the
# observable default across the system is "always".
VALID_DETECT_MODES = ("always", "camera_ai", "camera_ai_only")
DEFAULT_DETECT_MODE = "always"


def effective_detect_mode(
    mode: Optional[str], default: str = DEFAULT_DETECT_MODE, ai_on_camera: bool = True
) -> str:
    """Resolve a camera's effective detect mode. A stored mode that is one of
    VALID_DETECT_MODES wins; otherwise (NULL / '' / unknown) the supplied
    ``default`` (settings.detection.default_mode) applies, itself falling back to
    DEFAULT_DETECT_MODE when it too is unset/invalid. Never raises — an unknown
    value can never disable detection silently, it degrades to "always".

    A camera-AI mode requires on-camera AI: without it the AI gate never fires,
    so the camera would silently never detect. When ``ai_on_camera`` is False a
    camera_ai/camera_ai_only mode degrades to "always" (detect normally)."""
    m = (mode or "").strip()
    if m not in VALID_DETECT_MODES:
        d = (default or "").strip()
        m = d if d in VALID_DETECT_MODES else DEFAULT_DETECT_MODE
    if m in ("camera_ai", "camera_ai_only") and not ai_on_camera:
        return "always"
    return m


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw not in ("0", "false", "no", "")


# Legacy env prefix kept as a fallback so an existing .env from before the
# Sentinel -> Vigilume rename keeps working untouched.
_LEGACY_PREFIX = "SENTINEL_"
_CANON_PREFIX = "VIGILUME_"


def env_dual(suffix: str, default: str = "") -> str:
    """Read a tunable env var, preferring the canonical ``VIGILUME_<suffix>``
    name and falling back to the legacy ``SENTINEL_<suffix>`` name. An empty /
    whitespace-only value is treated as unset so the fallback still applies."""
    for prefix in (_CANON_PREFIX, _LEGACY_PREFIX):
        raw = os.environ.get(prefix + suffix)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return default


def env_dual_bool(suffix: str, default: bool) -> bool:
    raw = env_dual(suffix, "1" if default else "0").lower()
    return raw not in ("0", "false", "no", "")


def env_dual_float(suffix: str, default: float) -> float:
    """Read a numeric tunable via :func:`env_dual` (canonical VIGILUME_ then
    legacy SENTINEL_). An unset OR unparseable value falls back to ``default``
    with a warning rather than failing boot."""
    raw = env_dual(suffix, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "%s%s=%r is not a number — using default %s",
            _CANON_PREFIX, suffix, raw, default,
        )
        return default


def _env_detector() -> str:
    """Parse VIGILUME_DETECTOR (legacy: SENTINEL_DETECTOR) into one of
    VALID_DETECTORS (case-insensitive). An unknown/empty value falls back to the
    default ONNX backend with a loud warning rather than failing boot."""
    raw = env_dual("DETECTOR", DEFAULT_DETECTOR).lower()
    if raw not in VALID_DETECTORS:
        log.warning(
            "VIGILUME_DETECTOR=%r is not one of %s — falling back to %s",
            raw, VALID_DETECTORS, DEFAULT_DETECTOR,
        )
        return DEFAULT_DETECTOR
    return raw


@dataclass
class Config:
    admin_password: str = field(default_factory=lambda: _env("ADMIN_PASSWORD", "change-me"))
    public_url: str = field(default_factory=lambda: _env("PUBLIC_URL").rstrip("/"))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "/data")))
    media_dir: Path = field(default_factory=lambda: Path(_env("MEDIA_DIR", "/media")))
    go2rtc_config_dir: Path = field(
        default_factory=lambda: Path(_env("GO2RTC_CONFIG_DIR", "/go2rtc-config"))
    )

    # go2rtc endpoints as seen from the backend container.
    go2rtc_api_url: str = field(
        default_factory=lambda: _env("GO2RTC_URL", "http://go2rtc:1984").rstrip("/")
    )
    go2rtc_rtsp_url: str = field(
        default_factory=lambda: _env("GO2RTC_RTSP_URL", "rtsp://go2rtc:8554").rstrip("/")
    )

    # VIGILUME_REQUIRE_GPU=1 (compose default): a missing CUDA EP is a hard
    # detector failure (ready:false + loud log) instead of a silent CPU
    # fallback. Set 0 to consciously run detection on CPU (dev).
    require_gpu: bool = field(default_factory=lambda: env_dual_bool("REQUIRE_GPU", True))

    # Inference backend (VIGILUME_DETECTOR) — see VALID_DETECTORS above. The
    # detector factory (native.detector.build_detector) reads this to construct
    # the OnnxDetector. Default "onnx" keeps the D-FINE/GPU path.
    detector: str = field(default_factory=_env_detector)

    # Geographic location for the day/night (local sunset..sunrise) computation
    # that gates the per-camera "Smart spotlight" (native/sun.py +
    # native/spotlight.py). Defaults ~ New York City, matching the deploy's
    # default America/New_York timezone. Override with VIGILUME_LATITUDE /
    # VIGILUME_LONGITUDE (legacy SENTINEL_* still honored). Longitude is
    # east-positive (western hemisphere is negative).
    latitude: float = field(default_factory=lambda: env_dual_float("LATITUDE", 40.71))
    longitude: float = field(default_factory=lambda: env_dual_float("LONGITUDE", -74.01))

    # JWT session lifetime (days) and media-token lifetime (days; used in
    # push-notification image URLs which are fetched without headers).
    token_days: int = 30
    media_token_days: int = 7

    @property
    def db_path(self) -> Path:
        return self.data_dir / "nvr.db"

    @property
    def secrets_path(self) -> Path:
        return self.data_dir / "secrets.json"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def go2rtc_config_path(self) -> Path:
        return self.go2rtc_config_dir / "go2rtc.yaml"

    @property
    def recordings_dir(self) -> Path:
        return self.media_dir / "native" / "recordings"

    @property
    def clips_dir(self) -> Path:
        return self.media_dir / "native" / "clips"

    @property
    def suppression_thumbs_dir(self) -> Path:
        return self.data_dir / "suppression-thumbs"

    def seed_cameras(self) -> list[dict]:
        """Read CAM{1..3}_* env vars for first-boot camera seeding."""
        cams: list[dict] = []
        for i in (1, 2, 3):
            name = _env(f"CAM{i}_NAME")
            ip = _env(f"CAM{i}_IP")
            if not name or not ip:
                continue
            cams.append(
                {
                    "name": name,
                    "friendly_name": _env(f"CAM{i}_FRIENDLY") or name.replace("_", " ").title(),
                    "model": _env(f"CAM{i}_MODEL"),
                    "ip": ip,
                    "username": _env(f"CAM{i}_USER") or "admin",
                    "password": _env(f"CAM{i}_PASS"),
                }
            )
        return cams


# Default detect-stream dimensions per model (Amcrest substream, subtype=1).
# Operators can adjust via PUT /api/cameras/{name} if a device differs.
MODEL_DETECT_DEFAULTS = {
    "IP5M-T1277EW-AI": (704, 480),
    "IP8M-2779EW-AI": (704, 480),
    "AD410": (640, 480),
    "IP3M-941B": (640, 480),
    "IP4M-1041B": (640, 480),
    "IP4M-1056E": (704, 480),
}
DEFAULT_DETECT_FPS = 5

# Effective tracked-object list when a camera has no explicit detect_objects.
DEFAULT_DETECT_OBJECTS = ["person", "dog", "cat", "car"]

# Automatic camera time provisioning (settings.time_sync). Amcrest/Dahua
# doorbell/camera clocks drift (one AD410 shipped stuck at year-2000 on factory
# Beijing time). NTP proved unreliable on these units AND the Dahua NTP.TimeZone
# index -> offset mapping can not be trusted (a wrong index drifts the clock
# hours off), so when auto_sync is on we instead push the correct local
# wall-clock time straight to each reachable Dahua/Amcrest camera and DISABLE
# its NTP client (see amcrest/time_sync.py). The target zone is an IANA name
# (default the deploy's America/New_York); override with CAMERA_TIMEZONE (or the
# prefixed VIGILUME_CAMERA_TIMEZONE / legacy SENTINEL_CAMERA_TIMEZONE).
DEFAULT_CAMERA_TIMEZONE = (
    _env("CAMERA_TIMEZONE") or env_dual("CAMERA_TIMEZONE", "America/New_York")
)

# Defaults from docs/CONTRACTS.md — the settings KV store is merged over
# these. detection.audio_events/audio_labels were REMOVED with the legacy
# audio classifier (roadmap item); settings_store silently strips persisted
# copies from old /data volumes.
DEFAULT_SETTINGS: dict = {
    "notifications": {
        "enabled": True,
        "labels": ["person", "dog", "cat", "car"],
        "cooldown_seconds": 60,
        "min_score": 0.7,
        # Draw detection boxes + labels on event snapshots (the count banner
        # always stays). Legacy-safe: a stored blob missing this key means True.
        "draw_boxes": True,
        # APNs (iOS) push — docs/push-architecture.md. "relay" posts E2E
        # ciphertext to a push relay that holds the Apple .p8 (relay/), so no
        # Apple developer account is needed here; "off" disables APNs.
        # Default "off": never silently push through someone else's Apple
        # credentials. The "direct" mode (this server holding its own .p8) is
        # RETIRED — a stored "direct" is migrated in
        # settings_store._strip_legacy. NOTE: no `direct` key here on purpose —
        # settings_store.update() deep-merges over DEFAULT_SETTINGS, so leaving
        # it would stamp a dead, always-empty block into every saved document.
        "apns": {
            "mode": "off",
            "relay_url": "",
        },
        # ntfy (ntfy.sh or self-hosted) — the no-Apple-account push channel.
        # `topic` is deliberately EMPTY: it is a bearer secret (anyone who
        # knows it reads every message on a default-allow server), so the UI
        # generates a random one rather than defaulting to a guessable name.
        "ntfy": {
            "enabled": False,
            "server": "https://ntfy.sh",
            "topic": "",
            "auth_token": "",
            "priority": 4,
            "attach_snapshot": True,
        },
    },
    "recording": {"continuous_days": 7, "event_days": 14, "snapshot_days": 14},
    # Automatic time provisioning for Dahua/Amcrest cameras. When auto_sync is
    # on, each reachable camera has its clock set to the current local time in
    # `timezone` (an IANA name) and its on-device NTP client disabled — on
    # connect and every 30 min (see amcrest/time_sync.py).
    "time_sync": {"auto_sync": True, "timezone": DEFAULT_CAMERA_TIMEZONE},
    # default_mode is the effective detect mode for cameras whose per-camera
    # detect_mode is unset/NULL (see VALID_DETECT_MODES / effective_detect_mode).
    # backend: which silicon runs inference.
    #   "auto"  (default) — use an Edge TPU if the box has one, else the GPU.
    #                       Plug a Coral in, restart, and it is used; pull it
    #                       out and detection carries on unchanged. The Coral
    #                       bootstrap never raises, so "is one fitted?" is just
    #                       "did it come up ready?" — see AutoDetector.
    #   "gpu"             — force D-FINE ONNX on CUDA (highest accuracy).
    #   "coral"           — force the Edge TPU; detection is OFF if none binds.
    # Env VIGILUME_DETECTOR, when set, overrides this.
    "detection": {
        "model": "dfine_s",
        "confidence": 0.5,
        "default_mode": "always",
        "backend": "auto",
        # Edge TPU model, a SEPARATE key from `model` (the D-FINE tier): the two
        # lists share no entries, so one field would make an invalid pair
        # reachable the moment you flip backend. Ignored while backend="gpu".
        "coral_model": "ssdlite_mobiledet",
    },
    # auto_restart: optional nightly restart of the BACKEND process at a local
    # wall-clock time. Off by default — a restart costs a short recording gap,
    # so it is never something an install does uninvited.
    "system": {
        "public_url": "",
        "webrtc_candidates": [],
        "auto_restart": {"enabled": False, "time": "04:00"},
    },
    # Outbound MQTT + Home Assistant auto-discovery publisher (opt-in). This is
    # a PUBLISH-only integration (Vigilume -> the operator's broker); it is
    # unrelated to the removed inbound Frigate-MQTT. Changing this block
    # restarts the publisher live (no app restart). See docs/home-assistant.md.
    "mqtt": {
        "enabled": False,
        "host": "",
        "port": 1883,
        "username": "",
        "password": "",
        "discovery_prefix": "homeassistant",
        "base_topic": "vigilume",
    },
}

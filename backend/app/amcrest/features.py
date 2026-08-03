"""Per-model capability map (docs/CONTRACTS.md) + best-effort runtime probe.

The static map is authoritative for the supported models (config.KNOWN_MODELS);
the probe (run at camera registration) merges conclusive runtime findings over
it so unlisted/future models still get sensible capabilities.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .client import AmcrestClient

log = logging.getLogger(__name__)

CAPABILITY_KEYS = (
    "ir", "white_light", "siren", "mic", "speaker", "doorbell", "ai_on_camera",
    "backchannel", "ptz", "night_vision",
)

# Keys the runtime probe (probe_capabilities) can conclusively determine. ONLY
# these may be overridden by a camera's STORED capabilities snapshot. The rest
# (white_light, ai_on_camera, ptz, night_vision) are static-only and must always
# reflect the CURRENT model map — otherwise a stale stored snapshot from an
# earlier registration (e.g. IP3M-941B saved when night_vision was still False)
# permanently masks a later STATIC_CAPABILITIES update.
PROBE_KEYS = frozenset({"ir", "mic", "speaker", "doorbell", "siren", "backchannel"})

# From the CONTRACTS.md capability table. The two turrets are Amcrest "EW"
# dual-illuminator models (user-confirmed hardware): IR plus an on-demand
# white-LED spotlight driven via the Dahua Lighting_V2 CGI.
STATIC_CAPABILITIES: dict[str, dict[str, bool]] = {
    # night_vision is TRUE on every model: the night-vision Auto/Full-color/IR
    # control (VideoInDayNight day/night Mode) REPLACES the retired IR button on
    # all cameras, so every model must expose the night_vision_mode control.
    "IP5M-T1277EW-AI": {
        "ir": True, "white_light": True, "siren": False, "mic": True,
        "speaker": False, "doorbell": False, "ai_on_camera": True,
        "backchannel": False, "ptz": False, "night_vision": True,
    },
    "IP8M-2779EW-AI": {
        "ir": True, "white_light": True, "siren": False, "mic": True,
        "speaker": False, "doorbell": False, "ai_on_camera": True,
        "backchannel": False, "ptz": False, "night_vision": True,
    },
    "AD410": {
        "ir": True, "white_light": False, "siren": True, "mic": True,
        "speaker": True, "doorbell": True, "ai_on_camera": True,
        # The AD410 doorbell does NOT accept the HTTP CGI postAudio talk; its
        # two-way talk goes over the go2rtc RTSP/ONVIF audio backchannel
        # (WebRTC mic path). Clients gate the backchannel talk path on this.
        "backchannel": True, "ptz": False, "night_vision": True,
    },
    # IP3M-941B: pan/tilt PTZ dome (user-confirmed live). IR + mic + a working
    # RTSP two-way audio backchannel (codec pinned to G.711A), so talk auto-
    # routes to the backchannel path like the AD410. No on-camera AI, no white
    # light, no siren, no full-color night vision.
    "IP3M-941B": {
        "ir": True, "white_light": False, "siren": False, "mic": True,
        "speaker": True, "doorbell": False, "ai_on_camera": False,
        "backchannel": True, "ptz": True, "night_vision": True,
    },
    # IP4M-1041B: the 4 MP sibling of the IP3M-941B PTZ dome — identical
    # capability set (pan/tilt PTZ + presets, IR, mic/speaker with the RTSP
    # G.711A two-way backchannel, no AI/white-light/siren/doorbell). Added as a
    # clone per the operator's "just like the 941 variant"; retune only if the
    # real unit diverges (e.g. talk gain, see backchannel.TALK_GAIN_BY_MODEL).
    "IP4M-1041B": {
        "ir": True, "white_light": False, "siren": False, "mic": True,
        "speaker": True, "doorbell": False, "ai_on_camera": False,
        "backchannel": True, "ptz": True, "night_vision": True,
    },
    # IP4M-1056E: IR PLUS full-color night vision (the auto white LED driven by
    # the VideoInDayNight day/night mode — NOT an on-demand spotlight). No mic,
    # speaker, PTZ, siren, doorbell or on-camera AI (user-confirmed live).
    "IP4M-1056E": {
        "ir": True, "white_light": False, "siren": False, "mic": False,
        "speaker": False, "doorbell": False, "ai_on_camera": False,
        "backchannel": False, "ptz": False, "night_vision": True,
    },
    # Cameras stored with model "unknown" and no working credentials (e.g.
    # rows carried over from pre-standalone installs): nothing can be probed,
    # so promise nothing — the runtime probe merges real findings over this
    # once creds are supplied.
    "UNKNOWN": {
        "ir": False, "white_light": False, "siren": False, "mic": False,
        "speaker": False, "doorbell": False, "ai_on_camera": False,
        "backchannel": False, "ptz": False, "night_vision": True,
    },
}

# Conservative baseline for unknown models: most Amcrest cameras have IR and
# a microphone; everything else must be proven by the probe.
_UNKNOWN_MODEL_DEFAULTS: dict[str, bool] = {
    "ir": True, "white_light": False, "siren": False, "mic": True,
    "speaker": False, "doorbell": False, "ai_on_camera": False,
    "backchannel": False, "ptz": False, "night_vision": True,
}


def static_capabilities(model: str) -> dict[str, bool]:
    caps = STATIC_CAPABILITIES.get((model or "").strip().upper().replace(" ", ""))
    if caps is None:
        # try exact key as-is (map keys contain '-'), else fall back
        caps = STATIC_CAPABILITIES.get((model or "").strip(), _UNKNOWN_MODEL_DEFAULTS)
    return dict(caps)


async def probe_capabilities(client: "AmcrestClient") -> dict[str, Any]:
    """Best-effort device probe. Returns only conclusively-determined keys
    (plus 'device_type' metadata when readable). Network failures return {}."""
    probed: dict[str, Any] = {}

    device_type = await client.get_device_type()
    is_ad410 = False
    if device_type:
        probed["device_type"] = device_type
        if "AD410" in device_type.upper():
            is_ad410 = True
            probed["doorbell"] = True
            probed["siren"] = True
            # AD410's two-way talk runs over the go2rtc audio backchannel
            # (WebRTC mic path), NOT the HTTP CGI postAudio talk (confirmed
            # non-working). Advertise backchannel so clients pick that path.
            probed["backchannel"] = True
            # The AD410 is a two-way-talk doorbell: it ALWAYS has a mic and a
            # speaker. Some AD410 firmwares don't answer
            # devAudioOutput.cgi?action=getCollect (it returns 0 / unsupported),
            # which the merge below would otherwise treat as "no speaker" and
            # silently disable two-way talk (talk WS -> 4003). Pin both here so
            # a missing collect CGI can never mask the doorbell's audio.
            probed["speaker"] = True
            probed["mic"] = True

    # Lighting config presence -> IR control available.
    has_lighting = await client.has_lighting_config()
    if has_lighting is not None:
        probed["ir"] = has_lighting

    # For the AD410 these are pinned True above; a devAudioInput/Output collect
    # that reports 0 must not clobber that (the doorbell has both regardless).
    audio_in = await client.audio_input_channels()
    if audio_in is not None and not is_ad410:
        probed["mic"] = audio_in > 0

    audio_out = await client.audio_output_channels()
    if audio_out is not None and not is_ad410:
        probed["speaker"] = audio_out > 0

    # Night vision (the VideoInDayNight day/night Mode control) is the retired
    # IR button's replacement on EVERY camera, so default it True for any device
    # that actually answered a probe — every reachable camera then exposes the
    # night_vision_mode control. A fully offline device stays {} (the static map
    # keeps deciding) so an unprobed/unknown camera doesn't over-promise.
    if (device_type is not None or has_lighting is not None
            or audio_in is not None or audio_out is not None):
        probed.setdefault("night_vision", True)

    return probed


def merge_capabilities(model: str, probed: dict[str, Any]) -> dict[str, bool]:
    """Static map for `model` with conclusive probed values merged over it."""
    caps = static_capabilities(model)
    for key in CAPABILITY_KEYS:
        if key in probed and isinstance(probed[key], bool):
            caps[key] = probed[key]
    return caps

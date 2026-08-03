"""Async Amcrest (Dahua-compatible) CGI client over HTTP digest auth.

Endpoints follow the Dahua HTTP API as implemented by python-amcrest
(github.com/tchellomello/python-amcrest), which is sync — we call the same
CGIs directly with httpx.DigestAuth.

AD410 "siren": neither python-amcrest nor dchesterton's community Amcrest
bridge exposes a siren CGI (verified against their sources, 2026-07-05) and
no reliable public CGI
exists for the AD410 siren/tamper alarm. What IS public and verified
(python-amcrest audio.py) is `audio.cgi?action=postAudio`, which plays an
audio stream through the device speaker. play_tone() uses it to play a
generated two-tone alarm (G.711 A-law) — the closest supported behavior to
"sound the siren". If the device rejects it, callers surface a 501.
Two-way talk (talk_stream) rides the exact same postAudio path via the
shared _post_audio helper, transcoding browser PCM16 to A-law on the fly.

AD410 two-way talk — honest limitation (researched 2026-07-12 against
python-amcrest audio.py, AlexxIT/go2rtc PR #1795 "Dahua CGI source for
2-way audio", go2rtc issues #52/#141, and the Amcrest forum thread "AD410
does not honor POSTed audio"): the audio.cgi postAudio path we use is the
*correct and only public* HTTP CGI for pushing speaker audio, and its
parameters (httptype=singlepart, channel=1, Content-Type Audio/G.711A,
8 kHz A-law) already match every working reference. The remaining failure
is NOT a wrong parameter — it is the AD410 firmware itself:

  * The AD410 only *honors* POSTed talk audio when the device's own audio
    Encode format is G.711A/G.711Mu at an 8 kHz sample rate. The factory
    default is 16 kHz AAC, which silently swallows the POSTed stream (the
    doorbell briefly mutes its own mic but plays nothing) — talk_audio_
    diagnostic() reads that config and logs an actionable warning.
  * Even with the audio format corrected, multiple mature projects
    (go2rtc PR #1795 was closed after the author could "no longer make
    this work"; the AD110 issue #141 was closed "not planned") could not
    get reliable two-way talk over the CGI. Amcrest support has stated the
    Smart Home doorbells were not designed for third-party API audio — the
    in-app two-way talk rides a proprietary P2P/SIP path the HTTP CGI can't
    reach. So this path is a best-effort *attempt*, not a guarantee.

What DOES work on the AD410: one-way listen (RTSP audio from the camera),
doorbell button-press events (see doorbell.py), and the siren tone (same
postAudio path, but a complete fixed-length payload rather than an
open-ended stream). talk_stream keeps streaming the A-law regardless, so a
correctly-configured device still gets the best available shot at playing
it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, Union
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import numpy as np

# The talk stream deliberately under-runs its pinned dummy Content-Length
# (see _post_audio). h11 flags that at end-of-message, and httpcore does NOT
# map it to httpx.LocalProtocolError on the body-send path — the raw h11
# exception escapes, so it must be caught alongside the httpx one.
try:
    from h11 import LocalProtocolError as _H11LocalProtocolError  # httpcore dep
except Exception:  # pragma: no cover — h11 always ships with httpx
    class _H11LocalProtocolError(Exception):
        ...

_LOCAL_PROTOCOL_ERRORS = (httpx.LocalProtocolError, _H11LocalProtocolError)

log = logging.getLogger(__name__)

# How far a camera clock may read from the pushed time before provision_time
# calls the write a failure. The read-back happens milliseconds after the set,
# so a landed write measures well under a second; 10 s only absorbs a slow
# device answering its own CGI.
CLOCK_TOLERANCE_S = 10.0

# Dahua/Amcrest CGIs answer a bare "OK" on success and an "Error..." body on
# failure — both with HTTP 200, so the STATUS tells you nothing and the BODY is
# the only signal.
_CGI_OK_RE = re.compile(r"\bok\b", re.IGNORECASE)


def cgi_accepted(text: str) -> bool:
    """True when a Dahua CGI body means success.

    Matches "ok" as a WHOLE WORD, which is the only form that is safe in both
    directions:

    - A substring test (`"ok" in text.lower()`) accepts error bodies that merely
      CONTAIN the letters o-k: "Broken", "Error: Invalid token" (t-OK-en),
      "Set-Cookie" (co-OK-ie). That is a false SUCCESS — the caller reports a
      rejected command as done. This shipped in _ptz/reboot, and its twin in
      set_current_time helped hide the clock bug for months.
    - Case-sensitivity does not rescue it: "Error: TOKEN invalid" contains an
      uppercase OK.
    - An exact match (`text.strip().lower() == "ok"`) is safe against all of the
      above but would REJECT a decorated success like "Preset OK" — a false
      FAILURE on firmware we have not surveyed.

    A word-boundary match accepts "OK", "ok", "OK\\r\\n" and "Preset OK" while
    rejecting every error body above, so it needs no assumption about which
    shape a given firmware uses.
    """
    return _CGI_OK_RE.search(text) is not None

# Amcrest/Dahua "EW" dual-illuminator turrets whose on-demand white spotlight
# is driven over coaxialControlIO.cgi rather than the Dahua Lighting_V2
# white-LED CGI. The user verified on real IP5M-T1277EW-AI hardware that the
# Lighting_V2 / Lighting[0][0].Mode CGIs do NOTHING for the visible spotlight
# on this model — only coaxialControlIO toggles the illuminator. The IP8M-2779
# is the same EW illuminator family, so it takes the same path (best-effort).
COAX_WHITE_LIGHT_MODELS = frozenset({"IP5M-T1277EW-AI", "IP8M-2779EW-AI"})
_COAX_WHITE_LIGHT_MODELS_UPPER = frozenset(m.upper() for m in COAX_WHITE_LIGHT_MODELS)


def white_light_control_for_model(model: str) -> dict[str, Any]:
    """Describe the white-light / spotlight control contract for `model` so the
    UI can render the right control (and NOT show a dead brightness slider).

    The EW turrets drive their spotlight over coaxialControlIO, which is a bare
    on/off illuminator toggle: NO brightness and NO smart 'auto' (user-verified
    on real IP5M-T1277EW-AI hardware — see _set_white_light_coax). Everything
    else uses the Dahua Lighting_V2 white-LED slot, which does support off/on/
    auto plus 0-100 brightness.
    """
    if (model or "").strip().upper() in _COAX_WHITE_LIGHT_MODELS_UPPER:
        return {"brightness": False, "modes": ["off", "on"]}
    return {"brightness": True, "modes": ["off", "on", "auto"]}

# Read side of the postAudio timeout for STREAMING (two-way talk) bodies:
# once the browser stops talking the request body ends, and some firmwares
# only answer once the promised Content-Length arrives — this bounds how
# long a finished talk session waits for that response (tests shrink it).
TALK_READ_TIMEOUT_S = 10.0

# One PTZ "step" (a single client tap): a bounded start -> sleep -> stop nudge
# so a tap moves "a hair" instead of the continuous-move runaway ("one press
# goes all the way"). Small + tunable: STEP_SPEED is the ptz.cgi arg2 speed
# (1-8) and STEP_DURATION_S the dwell between start and stop.
STEP_SPEED = 2
STEP_DURATION_S = 0.25


class AmcrestError(Exception):
    """Device unreachable or CGI call failed."""


class AmcrestUnsupportedError(AmcrestError):
    """The device answered but does not support the requested CGI."""


class AmcrestAuthError(AmcrestError):
    """The device rejected the credentials (HTTP 401)."""


def _parse_kv(text: str) -> dict[str, str]:
    """Parse 'table.X=Y' CGI responses into {X: Y}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("table."):
            key = key[6:]
        out[key] = value.strip()
    return out


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _alaw_encode(samples: np.ndarray) -> bytes:
    """G.711 A-law encode float samples in [-1, 1] (ITU-T G.711, XOR 0x55).

    Implemented directly (audioop is removed in py3.12+); tonal accuracy is
    all that matters for an alarm tone.
    """
    a = 87.6
    ln_a = 1.0 + math.log(a)
    x = np.clip(samples, -1.0, 1.0)
    ax = np.abs(x)
    y = np.where(ax < (1.0 / a), (a * ax) / ln_a, (1.0 + np.log(np.maximum(a * ax, 1e-12))) / ln_a)
    magnitude = np.minimum((y * 128.0).astype(np.int32), 127).astype(np.uint8)
    sign = np.where(x >= 0, 0x80, 0x00).astype(np.uint8)
    return bytes(((sign | magnitude) ^ 0x55).tobytes())


def make_alarm_tone(duration_s: int, sample_rate: int = 8000) -> bytes:
    """Two-tone (800/1000 Hz) alarm, G.711 A-law mono @ 8 kHz."""
    duration_s = max(1, min(int(duration_s), 30))
    t = np.arange(duration_s * sample_rate, dtype=np.float64) / sample_rate
    # Alternate tone every 0.5 s.
    phase = (t * 2).astype(np.int64) % 2
    freq = np.where(phase == 0, 800.0, 1000.0)
    samples = 0.85 * np.sin(2.0 * np.pi * freq * t)
    return _alaw_encode(samples)


def pcm16le_to_alaw(pcm: bytes) -> bytes:
    """Transcode raw little-endian Int16 mono PCM to G.711 A-law bytes.

    Shares _alaw_encode with the siren tone generator (audioop is removed in
    py3.13+, so no stdlib option). Tolerates an odd trailing byte (dropped).
    """
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    return _alaw_encode(samples)


# ---------------------------------------------------------------------------
# Camera wall-clock time (timezone-index-INDEPENDENT)
# ---------------------------------------------------------------------------
#
# Dahua/Amcrest cameras CAN carry a timezone as an integer index into an
# opaque, firmware-specific table (``NTP.TimeZone`` / ``General.LocalNo``), but
# that index -> UTC-offset mapping can NOT be trusted on these units (the index
# assumed for US Eastern could not be confirmed live, and a wrong index silently
# drifts the wall-clock hours off). NTP on these cameras also proved unreliable
# (a forced sync did not land within 150s). So we never touch the device tz
# index or NTP client for correctness — instead we compute the correct LOCAL
# wall-clock time here and push it straight to the device via setCurrentTime.


def camera_local_now(tz_name: str) -> datetime:
    """Current wall-clock time in the IANA ``tz_name``, as a naive datetime.

    This is exactly the shape ``global.cgi?action=setCurrentTime`` wants:
    ``YYYY-MM-DD HH:MM:SS`` in the device's own local zone. It is computed via
    :mod:`zoneinfo` (Python 3.9+ stdlib) so it is INDEPENDENT of the container's
    own timezone — a backend container is usually UTC, and reading its local
    clock would set every camera to UTC. An unknown / malformed zone name falls
    back to UTC with a warning so provisioning never crashes."""
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        log.warning(
            "camera time-sync: unknown timezone %r (%s); falling back to UTC",
            tz_name, exc,
        )
        return datetime.now(timezone.utc).replace(tzinfo=None)
    return datetime.now(tz).replace(tzinfo=None)


class AmcrestClient:
    def __init__(
        self, ip: str, username: str, password: str, timeout: float = 8.0, model: str = ""
    ):
        self.ip = ip
        # Model drives per-model control gating (e.g. which white-light CGI to
        # use). Optional: callers that only need generic CGIs may omit it.
        self.model = (model or "").strip()
        self._client = httpx.AsyncClient(
            base_url=f"http://{ip}",
            auth=httpx.DigestAuth(username, password),
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- low-level ----------

    async def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> str:
        try:
            resp = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise AmcrestError(f"camera {self.ip} unreachable: {exc.__class__.__name__}") from exc
        if resp.status_code in (400, 404):
            raise AmcrestUnsupportedError(f"camera {self.ip} does not support {path}")
        if resp.status_code == 401:
            raise AmcrestAuthError(f"camera {self.ip}: authentication failed")
        if resp.status_code >= 400:
            raise AmcrestError(f"camera {self.ip}: HTTP {resp.status_code} for {path}")
        return resp.text

    async def _get_bytes(self, path: str, params: Optional[dict[str, Any]] = None) -> bytes:
        try:
            resp = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise AmcrestError(f"camera {self.ip} unreachable: {exc.__class__.__name__}") from exc
        if resp.status_code >= 400:
            raise AmcrestError(f"camera {self.ip}: HTTP {resp.status_code} for {path}")
        return resp.content

    async def get_config(self, name: str) -> dict[str, str]:
        text = await self._get("/cgi-bin/configManager.cgi", {"action": "getConfig", "name": name})
        if text.lstrip().startswith("Error"):
            raise AmcrestUnsupportedError(f"camera {self.ip}: config '{name}' not available")
        return _parse_kv(text)

    async def set_config(self, **params: Any) -> None:
        text = await self._get("/cgi-bin/configManager.cgi", {"action": "setConfig", **params})
        # Was `"OK" not in text` — case-sensitivity dodges the "Broken"/"token"
        # trap but NOT "Error: TOKEN invalid", which shouts an uppercase OK.
        if not cgi_accepted(text):
            raise AmcrestError(f"camera {self.ip}: setConfig rejected ({text.strip()[:120]})")

    # ---------- probes ----------

    async def fetch_device_type(self) -> Optional[str]:
        """getDeviceType, raising AmcrestAuthError/AmcrestError on failure —
        the on-demand probe route needs "authentication failed" vs
        "camera unreachable"; get_device_type keeps the old best-effort
        None-on-failure behavior for background probes."""
        text = await self._get("/cgi-bin/magicBox.cgi", {"action": "getDeviceType"})
        # response: "type=IP5M-T1277EW-AI"
        return _parse_kv(text).get("type")

    async def get_device_type(self) -> Optional[str]:
        try:
            return await self.fetch_device_type()
        except AmcrestError:
            return None

    async def has_lighting_config(self) -> Optional[bool]:
        """True/False when the device answered conclusively, None when offline."""
        try:
            await self.get_config("Lighting")
            return True
        except AmcrestUnsupportedError:
            return False
        except AmcrestError:
            return None

    async def audio_input_channels(self) -> Optional[int]:
        return await self._collect("/cgi-bin/devAudioInput.cgi")

    async def audio_output_channels(self) -> Optional[int]:
        return await self._collect("/cgi-bin/devAudioOutput.cgi")

    async def _collect(self, path: str) -> Optional[int]:
        try:
            text = await self._get(path, {"action": "getCollect"})
        except AmcrestUnsupportedError:
            return 0
        except AmcrestError:
            return None
        parsed = _parse_kv(text)
        try:
            return int(parsed.get("result", "0"))
        except ValueError:
            return None

    # ---------- IR (Lighting[0][0].Mode + MiddleLight brightness) ----------
    #
    # IR illumination is a configManager Lighting table property (NOT the
    # coaxialControlIO CGI, which only drives the visible white spotlight — see
    # the coaxial section below). Two knobs on all three supported models:
    #   Lighting[0][0].Mode                 = Auto | Manual | Off   (Manual="On")
    #   Lighting[0][0].MiddleLight[0].Light = 0..100                (IR strength)
    # Brightness is only meaningful in Manual mode, so set_ir() writes Mode
    # FIRST (its own setConfig) and the brightness SECOND, so a "make it Manual
    # at 60%" request lands the mode before the strength it depends on.

    # Incoming mode -> Dahua Lighting CGI value. Normalized case-insensitively
    # (see set_ir): "auto"->Auto (camera decides), "on"/"manual"->Manual (IR
    # forced ON), "off"->Off (IR forced OFF). Both "on" and "manual" are accepted
    # for the forced-on state so either the UI contract or a raw device term maps.
    _IR_TO_DEVICE = {"auto": "Auto", "on": "Manual", "manual": "Manual", "off": "Off"}
    _IR_FROM_DEVICE = {"auto": "auto", "manual": "on", "off": "off"}
    _IR_MODE_KEY = "Lighting[0][0].Mode"
    _IR_LIGHT_KEY = "Lighting[0][0].MiddleLight[0].Light"

    async def get_ir(self) -> Optional[dict[str, Any]]:
        """{mode: "auto"|"on"|"off", brightness: 0-100} from the IR illuminator,
        or None when the device has no readable Lighting Mode. brightness is
        omitted (key absent) when the table carries no MiddleLight value."""
        cfg = await self.get_config("Lighting")
        mode = cfg.get(self._IR_MODE_KEY)
        if mode is None:
            return None
        out: dict[str, Any] = {"mode": self._IR_FROM_DEVICE.get(mode.lower(), "auto")}
        raw = cfg.get(self._IR_LIGHT_KEY)
        if raw is not None:
            try:
                out["brightness"] = max(0, min(100, int(raw)))
            except ValueError:
                pass
        return out

    async def get_ir_mode(self) -> Optional[str]:
        """Back-compat: IR mode only ("auto"|"on"|"off"), None when unreadable."""
        ir = await self.get_ir()
        return ir["mode"] if ir else None

    @staticmethod
    def _lighting_indices(cfg: dict[str, Any]) -> list[int]:
        """Lighting PROFILE indices present in a getConfig (``Lighting[0][N].*``).

        Dahua/Amcrest expose one Lighting profile per day/night config, and the
        camera obeys whichever profile ``VideoInMode`` currently selects as
        active — which is frequently NOT profile 0 (verified: a T1277 with
        ``VideoInMode[0].Config[0]=2`` ignores ``Lighting[0][0]`` entirely). So
        IR writes must cover EVERY exposed profile. Falls back to ``[0]``."""
        idxs = {int(m.group(1)) for key in cfg
                if (m := re.match(r"Lighting\[0\]\[(\d+)\]\.", key))}
        return sorted(idxs) or [0]

    async def set_ir(
        self, mode: Optional[str] = None, brightness: Optional[int] = None
    ) -> None:
        """Apply IR mode and/or brightness to EVERY Lighting profile.

        The camera follows the ACTIVE day/night profile (often not [0][0]), so
        writing only [0][0] is silently ignored — the exact reason IR "didn't
        work". We read the exposed profiles and write them all. Mode is written
        before brightness (the Manual switch precedes the strength it governs);
        brightness (0..100) is clamped. A no-op when both are None.
        """
        if mode is None and brightness is None:
            return
        device_mode: Optional[str] = None
        if mode is not None:
            device_mode = self._IR_TO_DEVICE.get(mode.strip().lower())
            if device_mode is None:
                raise ValueError(f"invalid ir_mode {mode!r}")
        try:
            profiles = self._lighting_indices(await self.get_config("Lighting"))
        except AmcrestError:
            profiles = [0]
        if device_mode is not None:
            await self.set_config(
                **{f"Lighting[0][{p}].Mode": device_mode for p in profiles}
            )
        if brightness is not None:
            strength = str(max(0, min(100, int(brightness))))
            await self.set_config(
                **{f"Lighting[0][{p}].MiddleLight[0].Light": strength for p in profiles}
            )

    async def set_ir_mode(self, mode: str) -> None:
        """Back-compat: set only the IR mode (no brightness)."""
        await self.set_ir(mode=mode)

    # ---------- Day/Night IR-cut (VideoInDayNight[0][0].Mode) ----------
    #
    # The IR-cut filter / colour mode, distinct from the IR illuminator above:
    #   VideoInDayNight[0][0].Mode = Color | BlackWhite | Brightness
    # (Brightness = auto day/night by scene luminance). Exposed as an OPTIONAL
    # control so the UI can force colour or B/W independent of IR strength.

    _DAYNIGHT_TO_DEVICE = {"color": "Color", "black_white": "BlackWhite", "brightness": "Brightness"}
    _DAYNIGHT_FROM_DEVICE = {"color": "color", "blackwhite": "black_white", "brightness": "brightness"}
    _DAYNIGHT_KEY = "VideoInDayNight[0][0].Mode"

    async def get_day_night(self) -> Optional[str]:
        """"color"|"black_white"|"brightness" from VideoInDayNight, or None when
        the device reports no (recognised) mode."""
        cfg = await self.get_config("VideoInDayNight")
        mode = cfg.get(self._DAYNIGHT_KEY)
        if mode is None:
            return None
        return self._DAYNIGHT_FROM_DEVICE.get(mode.strip().lower())

    async def set_day_night(self, mode: str) -> None:
        device_mode = self._DAYNIGHT_TO_DEVICE.get(mode)
        if device_mode is None:
            raise ValueError(f"invalid day_night mode {mode!r}")
        await self.set_config(**{self._DAYNIGHT_KEY: device_mode})

    # ---------- Night-vision mode (VideoInDayNight[0][N].Mode, ALL profiles) ----------
    #
    # The night-vision control (ALL models — it REPLACES the retired IR button):
    # the "spotlight" is the auto white LED driven by the same VideoInDayNight
    # day/night table as the IR-cut above, exposed to clients as a simple
    # three-way night_vision_mode:
    #   auto  -> Mode=Brightness  (camera decides colour vs B/W by scene light)
    #   color -> Mode=Color       (force full-colour night vision / white LED)
    #   bw    -> Mode=BlackWhite  (force IR mono)
    # "auto" maps to Brightness (auto day/night by scene luminance) — NOT the
    # Dahua "Auto" enum, which the IP5M turret REJECTS. Color / BlackWhite /
    # Brightness are the only three Mode values accepted by every user model.
    # Like set_ir, the camera obeys whichever day/night PROFILE is active (often
    # not [0][0]), so the write must cover EVERY exposed VideoInDayNight profile
    # index — writing only [0][0] is silently ignored on multi-profile firmware.

    _NV_TO_DEVICE = {"auto": "Brightness", "color": "Color", "bw": "BlackWhite"}
    _NV_FROM_DEVICE = {"brightness": "auto", "color": "color", "blackwhite": "bw"}

    @staticmethod
    def _daynight_indices(cfg: dict[str, Any]) -> list[int]:
        """VideoInDayNight PROFILE indices present in a getConfig
        (``VideoInDayNight[0][N].*``). Falls back to ``[0]``."""
        idxs = {int(m.group(1)) for key in cfg
                if (m := re.match(r"VideoInDayNight\[0\]\[(\d+)\]\.", key))}
        return sorted(idxs) or [0]

    async def get_night_vision_mode(self) -> Optional[str]:
        """"auto"|"color"|"bw" from the first readable VideoInDayNight profile,
        or None when the device reports no recognised mode."""
        cfg = await self.get_config("VideoInDayNight")
        for p in self._daynight_indices(cfg):
            raw = cfg.get(f"VideoInDayNight[0][{p}].Mode")
            if raw is not None:
                mapped = self._NV_FROM_DEVICE.get(raw.strip().lower())
                if mapped is not None:
                    return mapped
        return None

    async def set_night_vision_mode(self, mode: str) -> None:
        """Set the night-vision control (all models). Writes the day/night Mode
        to EVERY exposed VideoInDayNight profile, then couples the IR illuminator
        to Auto so the retired IR button can't strand it.

        Because the user-facing IR Auto/On/Off control has been replaced by this
        night-vision control, a night-vision change ALSO forces the Lighting IR
        Mode to Auto (all profiles), so the IR LED auto-follows the sensor
        day/night instead of staying stuck at whatever the old IR button left
        it. The coupling is best-effort — a device without a writable Lighting
        table must not fail the night-vision change itself.
        """
        device_mode = self._NV_TO_DEVICE.get(mode.strip().lower())
        if device_mode is None:
            raise ValueError(f"invalid night_vision_mode {mode!r}")
        try:
            profiles = self._daynight_indices(await self.get_config("VideoInDayNight"))
        except AmcrestError:
            profiles = [0]
        await self.set_config(
            **{f"VideoInDayNight[0][{p}].Mode": device_mode for p in profiles}
        )
        # IR LED coupling: force the Lighting IR Mode to Auto (every profile) so
        # the illuminator auto-follows day/night now that the manual IR button is
        # gone. Best-effort — never fail the night-vision set over this.
        try:
            await self.set_ir(mode="auto")
        except AmcrestError as exc:
            log.info(
                "camera %s [%s]: night-vision IR Auto coupling skipped (%s)",
                self.ip, self.model or "?", exc,
            )

    # ---------- audio encoder (WebRTC-legal codec provisioning) ----------

    _ENCODE_AUDIO_COMPRESSION = re.compile(
        r"^Encode\[\d+\]\.(?:Main|Extra)Format\[\d+\]\.Audio\.Compression$"
    )

    async def provision_audio(self, codec: str) -> dict[str, Any]:
        """Force every stream's audio encoder to ``codec`` ("G.711A" or "AAC").

        Live-view audio rides go2rtc -> WebRTC, which passes native G.711 (PCMA)
        straight through to a live-view consumer but CANNOT carry AAC/MPEG4 — a
        camera whose Encode audio is AAC plays SILENT in live view (and go2rtc's
        ffmpeg transcode is unreliable on this image; advertising a codec it
        can't produce actually breaks the working cameras). So "G.711A" is the
        WebRTC-legal choice that makes live-view audio WORK (the default, and the
        historical behavior — the IP3M-941B already shipped as PCMA, which is why
        only it had sound); "AAC" trades live-view audio for higher recording
        quality.

        Reads the Encode config, rewrites ONLY the ``Audio.Compression`` values
        that aren't already ``codec``, and returns which formats changed.
        Idempotent (a no-op once every format is ``codec``). Raises AmcrestError
        on a getConfig/setConfig transport or rejection failure."""
        target = codec.strip().upper().replace(" ", "")
        # G.711A can be reported (and written) two ways; treat them as one codec.
        already = {"G.711A", "G711A"} if target in ("G.711A", "G711A") else {target}
        cfg = await self.get_config("Encode")
        keys = [k for k in cfg if self._ENCODE_AUDIO_COMPRESSION.match(k)]
        params: dict[str, str] = {}
        changed: list[str] = []
        for k in keys:
            cur = (cfg.get(k) or "").strip().upper().replace(" ", "")
            if cur in already:
                continue
            params[k] = codec
            changed.append(k)
        if params:
            await self.set_config(**params)
        return {"changed": changed, "formats": len(keys)}

    # ---------- substream keyframe interval (live-view first-frame latency) ----------

    _ENCODE_EXTRA_GOP = re.compile(
        r"^Encode\[(\d+)\]\.ExtraFormat\[(\d+)\]\.Video\.GOP$"
    )

    async def provision_substream_gop(self) -> dict[str, Any]:
        """Shorten the SUBSTREAM keyframe interval to ~1 second (GOP = FPS).

        WHY. A WebRTC/MSE/HLS consumer can only START decoding on an I-frame, and
        go2rtc caches no GOP for a newly attached consumer — so the first frame
        waits for the camera's next keyframe. At the common Dahua default of
        GOP = 2xFPS that is up to ~2 s (median ~1 s), and on a healthy LAN it is
        the single biggest component of live-view startup. GOP = FPS halves it,
        and it is paid again on every reconnect, scroll-back and app resume.

        EXTRAFORMAT ONLY — deliberately. More keyframes means more bits, and
        MainFormat is what the 24/7 recorder stores: shortening ITS GOP would
        inflate every recording for all 12 cameras and eat retention. The
        substream is what every live surface now opens on (and what the detector
        ingests), it is low-res, so the extra keyframes there cost very little.
        Promotion to the main stream happens make-before-break, with video
        already on screen, so main's keyframe wait is never seen.

        ONLY EVER SHORTENS. If a camera already keyframes more often than
        GOP = FPS we leave it alone, so this can never make a stream heavier than
        the operator configured. Cameras reporting FPS < 2 are skipped (GOP = 1
        would be all-intra). Idempotent: a second run is a no-op. Raises
        AmcrestError only on a getConfig/setConfig transport or rejection
        failure — callers treat that as "not provisioned yet" and retry later."""
        cfg = await self.get_config("Encode")
        params: dict[str, str] = {}
        changed: list[str] = []
        found = 0
        for key, raw in cfg.items():
            m = self._ENCODE_EXTRA_GOP.match(key)
            if not m:
                continue
            found += 1
            fps_key = f"Encode[{m.group(1)}].ExtraFormat[{m.group(2)}].Video.FPS"
            try:
                fps = int(float(cfg.get(fps_key)))
                current = int(float(raw))
            except (TypeError, ValueError):
                continue  # unreadable pair — leave the camera alone
            if fps < 2 or current <= 0:
                continue
            target = max(2, min(fps, current))
            if target >= current:
                continue  # already at or below one keyframe per second
            params[key] = str(target)
            changed.append(f"{key} {current}->{target}")
        if params:
            await self.set_config(**params)
        return {"changed": changed, "streams": found}

    # ---------- white light / spotlight (Lighting_V2) ----------
    #
    # Verified against the rroller/dahua Home Assistant integration
    # (custom_components/dahua/client.py @ main, fetched 2026-07-06):
    #   read:  configManager.cgi?action=getConfig&name=Lighting_V2
    #   write: Lighting_V2[{ch}][{profile}][{slot}].Mode={Off|Manual|Auto}
    #          Lighting_V2[{ch}][{profile}][{slot}].MiddleLight[0].Light={0-100}
    #   (channel 0-based; profile 0=day, 1=night, 2=scene; brightness 0-100)
    # Slot [0] is the IR illuminator and slot [1] the white-LED one on
    # dual-illuminator hardware: async_set_lighting_v2_for_flood_lights and
    # async_set_lighting_v2_for_amcrest_doorbells both address
    # Lighting_V2[ch][profile][1], while plain async_set_lighting_v2
    # (single-light devices) uses [0]. The EW turrets are IR+white
    # dual-illuminator models, so Vigilume drives slot [1] on channel 0,
    # profile 0. The HA integration writes Mode=Manual/Off for V2;
    # Mode=Auto is the same Dahua enum the V1 Lighting table uses
    # (smart-illumination) and Amcrest EW firmware accepts it.

    _WL_TO_DEVICE = {"off": "Off", "on": "Manual", "auto": "Auto"}
    _WL_FROM_DEVICE = {"off": "off", "manual": "on", "auto": "auto"}
    _WL_KEY = "Lighting_V2[0][0][1]"

    def _uses_coax_light(self) -> bool:
        """The EW turrets drive their white spotlight over coaxialControlIO,
        not Lighting_V2 (user-verified — see COAX_WHITE_LIGHT_MODELS)."""
        return self.model.upper() in _COAX_WHITE_LIGHT_MODELS_UPPER

    async def get_white_light(self) -> dict[str, Any]:
        """{mode: "off"|"on"|"auto", brightness: 0-100} from the white-LED
        slot. Raises AmcrestUnsupportedError when the device has no
        Lighting_V2 table or no white-light entry in it."""
        if self._uses_coax_light():
            return await self._get_white_light_coax()
        cfg = await self.get_config("Lighting_V2")
        mode = cfg.get(f"{self._WL_KEY}.Mode")
        if mode is None:
            raise AmcrestUnsupportedError(
                f"camera {self.ip}: no white-light entry in Lighting_V2"
            )
        try:
            brightness = max(0, min(100, int(cfg.get(f"{self._WL_KEY}.MiddleLight[0].Light", ""))))
        except ValueError:
            brightness = 100
        return {"mode": self._WL_FROM_DEVICE.get(mode.lower(), "off"), "brightness": brightness}

    async def set_white_light(
        self, mode: Optional[str] = None, brightness: Optional[int] = None
    ) -> None:
        """Apply spotlight mode and/or brightness (either may be omitted)."""
        if self._uses_coax_light():
            await self._set_white_light_coax(mode)
            return
        params: dict[str, str] = {}
        if mode is not None:
            device_mode = self._WL_TO_DEVICE.get(mode)
            if device_mode is None:
                raise ValueError(f"invalid white_light mode {mode!r}")
            params[f"{self._WL_KEY}.Mode"] = device_mode
        if brightness is not None:
            params[f"{self._WL_KEY}.MiddleLight[0].Light"] = str(
                max(0, min(100, int(brightness)))
            )
        if not params:
            return
        try:
            await self.set_config(**params)
        except AmcrestUnsupportedError:
            raise
        except AmcrestError as exc:
            # setConfig rejections all look alike ("Error"); classify by
            # reading the table back — a device without the white-LED slot
            # is unsupported (501), not a transient failure (502).
            try:
                await self.get_white_light()
            except AmcrestUnsupportedError:
                raise
            except AmcrestError:
                pass
            raise exc

    # ---------- coaxial illuminator / alarm control (EW turrets) ----------
    #
    # coaxialControlIO.cgi is the Dahua HDCVI "coax" control CGI, repurposed on
    # TiOC / EW IP turrets to drive the on-demand white spotlight (and siren /
    # warning light). User-verified working on a real IP5M-T1277EW-AI where the
    # Lighting_V2 white-LED CGI is inert:
    #   GET /cgi-bin/coaxialControlIO.cgi?action=control&channel=1
    #       &info[0].Type=<T>&info[0].IO=<1|0>
    #   IO=1 -> illuminator ON, IO=0 -> OFF (this firmware; the rroller/dahua
    #   HA integration sends IO=2 for OFF, but the T1277 accepts 0).
    #
    # Type mapping (confirmed against the rroller/dahua HA integration source +
    # the Dahua TiOC HTTP command reference, 2026-07-11):
    #   Type=1 -> white light (visible spotlight)   <- what we drive
    #   Type=2 -> siren / audible alarm
    #   Type=3 -> red/blue warning light (some models)
    # There is NO coaxialControlIO Type for the IR illuminator: IR / night
    # illumination is exclusively a configManager Lighting[0][0].Mode property
    # (see set_ir_mode), so IR control is NOT moved onto this CGI.
    #
    # The bracketed info[0] keys are percent-encoded by httpx exactly like the
    # Lighting_V2[..] setConfig keys the same firmware already accepts.
    COAX_WHITE_LIGHT_TYPE = 1
    COAX_SIREN_TYPE = 2

    async def coaxial_control(self, type_code: int, on: bool) -> None:
        """Toggle a coaxialControlIO illuminator/alarm channel on/off.

        Raises AmcrestUnsupportedError when the firmware lacks the CGI (HTTP
        400/404 via _get) and AmcrestError when the device rejects the call.
        """
        io = 1 if on else 0
        log.info(
            "camera %s: coaxialControlIO Type=%d IO=%d (%s)",
            self.ip, int(type_code), io, "on" if on else "off",
        )
        text = await self._get(
            "/cgi-bin/coaxialControlIO.cgi",
            {
                "action": "control",
                "channel": 1,
                "info[0].Type": int(type_code),
                "info[0].IO": io,
            },
        )
        # DELIBERATELY NOT cgi_accepted(): this one rejects on an explicit
        # "Error" prefix and accepts everything else, rather than requiring an
        # "ok" body. The EW-turret spotlight is user-verified working through
        # this path, and its real success body has never been surveyed — if the
        # firmware answers something other than "OK" (empty, or a result=...
        # line), demanding "ok" here would break the spotlight. Don't "unify"
        # this with the others without capturing what the turrets actually send.
        if text.lstrip().lower().startswith("error"):
            raise AmcrestError(
                f"camera {self.ip}: coaxialControlIO rejected ({text.strip()[:120]})"
            )

    async def _set_white_light_coax(self, mode: Optional[str]) -> None:
        """Spotlight on/off for EW turrets via coaxialControlIO Type=1.

        coaxialControlIO is on/off only — no brightness and no smart 'auto', so
        'on' and 'auto' both illuminate and 'off' extinguishes. A brightness-
        only patch (mode is None) is a no-op on this hardware.
        """
        if mode is None:
            return
        if mode not in ("off", "on", "auto"):
            raise ValueError(f"invalid white_light mode {mode!r}")
        await self.coaxial_control(self.COAX_WHITE_LIGHT_TYPE, mode != "off")

    async def _get_white_light_coax(self) -> dict[str, Any]:
        """Spotlight state for EW turrets via coaxialControlIO?action=getStatus.

        Firmware wording varies ('status.WhiteLight=On', 'Status[0].WhiteLight'
        …) — match any key containing 'WhiteLight'. Brightness is not reported
        over this CGI, so it is fixed at 100 to keep the {mode, brightness}
        response shape the UI expects.
        """
        text = await self._get(
            "/cgi-bin/coaxialControlIO.cgi", {"action": "getStatus", "channel": 1}
        )
        on = False
        for key, value in _parse_kv(text).items():
            if "whitelight" in key.lower():
                on = value.strip().lower() in ("on", "1", "true")
                break
        return {"mode": "on" if on else "off", "brightness": 100}

    # ---------- motion detect ----------

    async def get_motion_detect(self) -> Optional[bool]:
        cfg = await self.get_config("MotionDetect")
        value = cfg.get("MotionDetect[0].Enable")
        return _to_bool(value) if value is not None else None

    async def set_motion_detect(self, enabled: bool) -> None:
        await self.set_config(**{"MotionDetect[0].Enable": "true" if enabled else "false"})

    # ---------- hardware lens mask (RETIRED — migration use only) ----------
    #
    # The camera's own LeLensMask blackout was Vigilume's original "privacy
    # mode". It is GONE from the API and both UIs, replaced by software Privacy
    # Mode (app/privacy.py), which stops all capture without reconfiguring the
    # device.
    #
    # `clear_lens_mask` survives for exactly ONE reason: removing a control does
    # not un-set the state it left behind. A camera masked by the old feature
    # would otherwise stay blind forever with nothing in Vigilume able to clear
    # it — you'd have to go to the camera's own web UI. amcrest/lens_mask.py calls
    # this the first time each camera is reachable so that cannot happen.
    # Do NOT reintroduce a setter.

    async def clear_lens_mask(self) -> None:
        """Turn the camera's hardware lens mask OFF. Idempotent."""
        await self.set_config(**{"LeLensMask[0].Enable": "false"})

    # ---------- OSD channel title ----------

    async def get_osd_name(self) -> Optional[str]:
        cfg = await self.get_config("ChannelTitle")
        return cfg.get("ChannelTitle[0].Name")

    async def set_osd_name(self, name: str) -> None:
        await self.set_config(**{"ChannelTitle[0].Name": name})

    # ---------- flip ----------

    async def get_flip(self) -> Optional[bool]:
        cfg = await self.get_config("VideoInOptions")
        value = cfg.get("VideoInOptions[0].Flip")
        return _to_bool(value) if value is not None else None

    async def set_flip(self, enabled: bool) -> None:
        await self.set_config(**{"VideoInOptions[0].Flip": "true" if enabled else "false"})

    # ---------- speaker volume (best-effort; AD410) ----------

    async def get_speaker_volume(self) -> Optional[int]:
        try:
            cfg = await self.get_config("AudioOutputVolume")
        except AmcrestError:
            return None
        for value in cfg.values():
            try:
                return max(0, min(100, int(value)))
            except ValueError:
                continue
        return None

    async def set_speaker_volume(self, volume: int) -> None:
        await self.set_config(**{"AudioOutputVolume[0]": str(max(0, min(100, int(volume))))})

    # ---------- PTZ (ptz.cgi) ----------
    #
    # Continuous pan/tilt + presets over the Dahua ptz.cgi CGI (IP3M-941B dome):
    #   move:  ptz.cgi?action=start&channel=1&code=<Up|Down|Left|Right|LeftUp|
    #          RightUp|LeftDown|RightDown>&arg1=0&arg2=<speed 1-8>&arg3=0
    #   stop:  ptz.cgi?action=stop&...   (same code as the move being stopped)
    #   preset: ptz.cgi?action=start&channel=1&code=<SetPreset|GotoPreset|
    #          ClearPreset>&arg1=0&arg2=<index 1-3>&arg3=0
    # The diagonal directions map to Dahua's Left/Right-prefixed codes.

    _PTZ_DIRECTION_CODE = {
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "upleft": "LeftUp", "upright": "RightUp",
        "downleft": "LeftDown", "downright": "RightDown",
    }
    _PTZ_PRESET_CODE = {
        "preset_set": "SetPreset",
        "preset_goto": "GotoPreset",
        "preset_clear": "ClearPreset",
    }

    async def _ptz(
        self, action: str, code: str, *, arg1: int = 0, arg2: int = 0, arg3: int = 0
    ) -> None:
        text = await self._get(
            "/cgi-bin/ptz.cgi",
            {"action": action, "channel": 1, "code": code,
             "arg1": arg1, "arg2": arg2, "arg3": arg3},
        )
        if not cgi_accepted(text):
            raise AmcrestError(
                f"camera {self.ip}: ptz {code} rejected ({text.strip()[:120]})"
            )

    async def ptz_move(self, direction: str, speed: int = 4) -> None:
        """Start a continuous pan/tilt in `direction` at `speed` (1-8)."""
        code = self._PTZ_DIRECTION_CODE.get(direction.strip().lower())
        if code is None:
            raise ValueError(f"invalid ptz direction {direction!r}")
        await self._ptz("start", code, arg2=max(1, min(8, int(speed))))

    async def ptz_stop(self, direction: str) -> None:
        """Stop the continuous move for `direction` (same code as the start)."""
        code = self._PTZ_DIRECTION_CODE.get(direction.strip().lower())
        if code is None:
            raise ValueError(f"invalid ptz direction {direction!r}")
        await self._ptz("stop", code)

    async def ptz_step(
        self,
        direction: str,
        *,
        speed: int = STEP_SPEED,
        duration_s: float = STEP_DURATION_S,
    ) -> None:
        """One small bounded pan/tilt nudge (a single tap = "a hair").

        Runs the continuous move server-side for a tiny fixed dwell then stops
        it: ptz.cgi start(code=<dir>, speed) -> asyncio.sleep(duration_s) ->
        ptz.cgi stop(code=<dir>). This fixes the hold-to-move runaway where one
        press ran all the way — the client just taps ``action:"step"`` per nudge
        and the stop is issued here, not by the client. Supports all 8
        directions. The stop is issued even if the dwell is interrupted.
        """
        code = self._PTZ_DIRECTION_CODE.get(direction.strip().lower())
        if code is None:
            raise ValueError(f"invalid ptz direction {direction!r}")
        await self._ptz("start", code, arg2=max(1, min(8, int(speed))))
        try:
            await asyncio.sleep(max(0.0, float(duration_s)))
        finally:
            await self._ptz("stop", code)

    async def ptz_preset(self, action: str, index: int) -> None:
        """Set / goto / clear preset `index` (1-3)."""
        code = self._PTZ_PRESET_CODE.get(action.strip().lower())
        if code is None:
            raise ValueError(f"invalid ptz preset action {action!r}")
        await self._ptz("start", code, arg2=int(index))

    # ---------- reboot / snapshot ----------

    async def reboot(self) -> None:
        text = await self._get("/cgi-bin/magicBox.cgi", {"action": "reboot"})
        if not cgi_accepted(text):
            raise AmcrestError(f"camera {self.ip}: reboot rejected")

    async def snapshot(self) -> bytes:
        data = await self._get_bytes("/cgi-bin/snapshot.cgi", {"channel": 1})
        if not data or not data.startswith(b"\xff\xd8"):
            raise AmcrestError(f"camera {self.ip}: snapshot.cgi returned no JPEG")
        return data

    # ---------- time provisioning ----------
    #
    # Doorbell/camera clocks drift (one AD410 shipped stuck at year-2000 on a
    # factory Beijing timezone). We do NOT rely on NTP or the device timezone
    # index to fix this: NTP proved unreliable on these Amcrest units (a forced
    # sync did not land within 150s), and the Dahua ``NTP.TimeZone`` index ->
    # UTC-offset mapping can not be trusted (a wrong index drifts the clock
    # hours off). Instead ``provision_time`` pushes the correct local wall-clock
    # time straight to the device and DISABLES its NTP client — a reliable,
    # model-independent correction. Both CGIs are validated on the AD410:
    #   configManager.cgi?action=setConfig&NTP.Enable=false
    #   global.cgi?action=setCurrentTime&time=YYYY-MM-DD HH:MM:SS   (target-tz local)

    async def set_current_time(self, when: datetime) -> None:
        """Set the device wall-clock via global.cgi?action=setCurrentTime.

        ``when`` is a naive LOCAL datetime (the device's own zone), formatted
        ``YYYY-MM-DD HH:MM:SS`` (the Dahua format). Raises AmcrestError when
        the device rejects it.

        The query is built BY HAND rather than via a params dict, and that is
        load-bearing: httpx serializes params through urllib's ``urlencode``,
        whose default ``quote_via=quote_plus`` maps a space to ``+``. So a
        params dict puts ``time=2026-07-16+11%3A54%3A26`` on the wire.
        ``+``-as-space is an HTML *form* convention, not RFC 3986 — a
        percent-only CGI decoder reads a literal plus and the stamp is
        malformed. ``quote`` emits ``%20``, which every decoder reads as a
        space. (This docstring previously claimed httpx percent-encodes the
        space; it does not, and that claim is why the bug survived review —
        every camera silently free-ran while the sync logged success. See
        controls/time_sync smoke tests, which now assert the RAW query.)"""
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")
        text = await self._get(
            "/cgi-bin/global.cgi"
            f"?action=setCurrentTime&time={quote(stamp, safe='')}"
        )
        # Whole-word "ok" (see cgi_accepted): a plain substring test would log
        # "Broken"/"Error: Invalid token"/"Set-Cookie" as a successful clock
        # set. An EXACT match was the first fix here, but it would reject a
        # decorated success like "OK: time set" and take the whole fleet's
        # clock sync down on firmware we have never surveyed — and the
        # read-back below is what actually proves the write landed, so the body
        # check does not need to be the strict one.
        if not cgi_accepted(text):
            raise AmcrestError(
                f"camera {self.ip}: setCurrentTime rejected ({text.strip()[:80]})"
            )

    async def get_current_time(self) -> datetime:
        """Read the device wall-clock back: ``global.cgi?action=getCurrentTime``
        answers ``result=YYYY-MM-DD HH:MM:SS``. Raises AmcrestError when the
        device won't answer or the stamp is unparseable."""
        text = await self._get("/cgi-bin/global.cgi", {"action": "getCurrentTime"})
        raw = text.strip().partition("=")[2].strip()
        try:
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise AmcrestError(
                f"camera {self.ip}: getCurrentTime unparseable ({text.strip()[:80]})"
            ) from exc

    async def provision_time(
        self,
        tz_name: str,
        *,
        when: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Idempotently correct a Dahua/Amcrest camera's clock, independent of
        any (untrustworthy) on-device timezone index or NTP client.

        Pushes the current local wall-clock time for the IANA ``tz_name`` via
        ``global.cgi?action=setCurrentTime`` and DISABLES the device NTP client
        (``NTP.Enable=false``) so NTP + a wrong device timezone index can never
        drift the clock again. Re-running writes the same config, so it is safe
        to call on every reconnect AND on the periodic re-push loop.

        ``when`` overrides the computed time (tests). Disabling NTP is
        best-effort — a firmware that rejects that config key must not stop the
        clock from being set. Raises AmcrestError only when ``setCurrentTime``
        itself fails (the caller treats that as non-fatal + retryable). Returns
        a summary dict for logging/tests."""
        # Disable the on-device NTP client FIRST so a wrong device timezone
        # index can't re-drift the clock right after we set it. Best-effort: an
        # offline device / firmware that rejects the key still gets its clock
        # set below.
        try:
            await self.set_config(**{"NTP.Enable": "false"})
            ntp_disabled = True
        except AmcrestError as exc:
            log.info(
                "camera %s [%s]: NTP.Enable=false rejected (%s); setting the "
                "clock anyway",
                self.ip, self.model or "?", exc,
            )
            ntp_disabled = False

        # Compute the target time HERE — after the NTP round trip, immediately
        # before the write — NOT at the top of this function.
        #
        # That ordering is the whole fix for a residual ~12-20s lag on EVERY
        # camera. The NTP.Enable call above is a full HTTP round trip to a slow
        # embedded device, and at boot all ~12 cameras make it concurrently, so
        # it can take 20s+. A timestamp sampled before it is already that stale
        # by the time setCurrentTime lands, and the camera is dutifully set to
        # the past. The symptom was every camera logging the SAME pushed value
        # (the moment the sweep started) and reading back a consistent negative
        # offset — which looks like "the write is not landing" but is really
        # "the write landed perfectly, carrying a stale value".
        when = when or camera_local_now(tz_name)
        await self.set_current_time(when)
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")

        # READ THE CLOCK BACK. set_current_time returning cleanly proves only
        # that the device ACCEPTED the request (HTTP 200 + an "ok" body); it
        # proves nothing about the resulting clock. Without this, a camera that
        # accepts-and-ignores drifts for months behind a log line that says
        # "clock set" every 30 minutes — which is exactly what happened.
        #
        # Deliberately NON-fatal: callers piggyback further provisioning on
        # provision_time returning (see time_sync._provision, which sets the
        # G.711A audio codec AFTER this), so raising here would silently stop
        # audio provisioning fleet-wide. Report the truth; don't break a
        # working camera over a wrong clock. Firmware without getCurrentTime
        # must not fail the sync either.
        clock_delta_s: Optional[float] = None
        try:
            actual = await self.get_current_time()
            clock_delta_s = round((actual - camera_local_now(tz_name)).total_seconds(), 1)
            if abs(clock_delta_s) > CLOCK_TOLERANCE_S:
                log.warning(
                    "camera %s [%s]: setCurrentTime returned OK but the clock reads "
                    "%s — %+.0fs off. The write is NOT landing; this camera's clock "
                    "is wrong and its event timestamps are unreliable.",
                    self.ip, self.model or "?",
                    actual.strftime("%Y-%m-%d %H:%M:%S"), clock_delta_s,
                )
        except AmcrestError as exc:
            log.info(
                "camera %s [%s]: clock read-back unavailable (%s) — the set was "
                "accepted but cannot be verified",
                self.ip, self.model or "?", exc,
            )

        log.info(
            "camera %s [%s]: clock set to %s (%s); device NTP %s; read-back %s",
            self.ip, self.model or "?", stamp, tz_name,
            "disabled" if ntp_disabled else "left as-is (disable rejected)",
            "unavailable" if clock_delta_s is None else f"{clock_delta_s:+.1f}s",
        )
        return {
            "timezone": tz_name,
            "ntp_disabled": ntp_disabled,
            "current_time": stamp,
            # None = the device wouldn't tell us. A number is the MEASURED
            # offset of the device clock from the pushed time.
            "clock_delta_s": clock_delta_s,
        }

    # ---------- speaker audio: audio.cgi postAudio ----------
    # Shared by the AD410 "siren" tone and two-way talk. Format verified
    # against python-amcrest audio.py (audio_send_stream): POST
    # /cgi-bin/audio.cgi?action=postAudio&httptype=singlepart&channel=1 with
    # Content-Type "Audio/G.711A"; python-amcrest pins Content-Length to a
    # large dummy (9999999) for open-ended streams because Dahua firmwares
    # don't accept chunked transfer-encoding — httpx omits Transfer-Encoding
    # whenever Content-Length is explicit, so we do the same for streaming
    # bodies and send the exact length for complete (bytes) payloads.

    async def _post_audio(
        self,
        content: Union[bytes, bytearray, AsyncIterator[bytes]],
        *,
        timeout: httpx.Timeout,
        context: str = "audio",
    ) -> None:
        """POST G.711 A-law audio to the device speaker.

        `content` is either the complete payload (bytes — replayable if
        digest auth answers 401 first) or an async iterator of A-law chunks
        (two-way talk). Streaming bodies cannot be replayed for a digest
        re-challenge, so streaming callers must prime the challenge cache
        first (talk_stream does).

        `context` ("talk" / "siren" / "audio") only tags the logs so the
        AD410 talk investigation can tell the two callers apart.
        """
        streaming = not isinstance(content, (bytes, bytearray))
        length = 9_999_999 if streaming else len(content)
        log.info(
            "camera %s [%s]: postAudio (%s) channel=1 Content-Type=Audio/G.711A "
            "Content-Length=%s (%s)",
            self.ip, self.model or "?", context, length,
            "open-ended stream" if streaming else "fixed payload",
        )
        try:
            resp = await self._client.post(
                "/cgi-bin/audio.cgi",
                params={"action": "postAudio", "httptype": "singlepart", "channel": 1},
                content=content,
                headers={"Content-Type": "Audio/G.711A", "Content-Length": str(length)},
                timeout=timeout,
            )
        except (httpx.StreamConsumed, httpx.StreamClosed) as exc:
            raise AmcrestError(
                f"camera {self.ip}: digest re-challenge on a one-shot audio stream"
            ) from exc
        except _LOCAL_PROTOCOL_ERRORS as exc:
            if streaming:
                # The talk stream ended before the pinned dummy Content-Length
                # (h11 enforces outgoing framing; urllib3/python-amcrest just
                # close the socket the same way). Every A-law byte was already
                # written — the camera simply sees the connection end. NOTE for
                # the AD410: "written" != "played" — the doorbell can accept the
                # whole stream and still emit nothing (see module docstring).
                log.info(
                    "camera %s [%s]: postAudio (%s) stream ended; all bytes sent, "
                    "no device response read (%s)",
                    self.ip, self.model or "?", context, exc.__class__.__name__,
                )
                return
            raise AmcrestError(f"camera {self.ip}: postAudio framing error: {exc}") from exc
        except httpx.ReadTimeout as exc:
            if streaming:
                # Body fully delivered; some firmwares only answer once the
                # promised Content-Length arrives. The audio already played.
                log.info(
                    "camera %s [%s]: postAudio (%s) no response after stream end (ReadTimeout)",
                    self.ip, self.model or "?", context,
                )
                return
            raise AmcrestError(f"camera {self.ip} unreachable: ReadTimeout") from exc
        except httpx.RemoteProtocolError as exc:
            if streaming:
                # Connection dropped without a response after we finished
                # sending — outright rejections answer 400/401 instead.
                log.info(
                    "camera %s [%s]: postAudio (%s) connection closed after stream end (%s)",
                    self.ip, self.model or "?", context, exc,
                )
                return
            raise AmcrestError(f"camera {self.ip} unreachable: RemoteProtocolError") from exc
        except httpx.HTTPError as exc:
            raise AmcrestError(f"camera {self.ip} unreachable: {exc.__class__.__name__}") from exc
        log.info(
            "camera %s [%s]: postAudio (%s) device responded HTTP %d",
            self.ip, self.model or "?", context, resp.status_code,
        )
        if resp.status_code in (400, 404):
            raise AmcrestUnsupportedError(
                f"camera {self.ip}: audio.cgi postAudio not supported by this firmware"
            )
        if resp.status_code == 401:
            raise AmcrestAuthError(f"camera {self.ip}: authentication failed")
        if resp.status_code >= 400:
            raise AmcrestError(f"camera {self.ip}: postAudio failed with HTTP {resp.status_code}")

    # ---------- AD410 "siren" (alarm tone via speaker) ----------

    async def play_tone(self, duration_s: int = 10) -> None:
        """Play a generated alarm tone through the device speaker via
        audio.cgi?action=postAudio (verified CGI, python-amcrest audio.py).

        Raises AmcrestUnsupportedError when the device rejects the CGI.
        """
        payload = await asyncio.to_thread(make_alarm_tone, duration_s)
        await self._post_audio(
            payload,
            timeout=httpx.Timeout(float(min(duration_s, 30) + 15), connect=5.0),
            context="siren",
        )

    # ---------- two-way talk ----------

    def _is_ad410(self) -> bool:
        return "AD410" in self.model.upper()

    # Audio Encode formats the AD410 will actually honor for POSTed talk audio
    # (G.711 A-law / mu-law). Compared case-insensitively with spaces stripped.
    _TALK_OK_AUDIO_FORMATS = frozenset(
        {"G.711A", "G711A", "G.711MU", "G711MU", "G.711.MU", "G.711U", "G711U"}
    )

    async def talk_audio_diagnostic(self) -> Optional[dict[str, Any]]:
        """Best-effort read of the device audio-Encode config, used to explain
        AD410 two-way-talk failures without changing anything on the device.

        The AD410 only honors audio.cgi POSTed speaker audio when its audio
        Encode format is G.711A/G.711Mu at an 8 kHz sample rate; the factory
        default is 16 kHz AAC, which silently drops POSTed talk audio. Returns
        ``{audio_format, sample_rate, compatible}`` (compatible may be None when
        the format couldn't be found), or None when the config can't be read.
        Never raises — this is a diagnostic, not a control path.
        """
        try:
            cfg = await self.get_config("Encode")
        except AmcrestError:
            return None
        fmt: Optional[str] = None
        rate: Optional[str] = None
        for key, value in cfg.items():
            kl = key.lower()
            if fmt is None and kl.endswith(".audio.compression"):
                fmt = value.strip()
            if rate is None and kl.endswith(".audio.frequency"):
                rate = value.strip()
        compatible: Optional[bool] = None
        if fmt is not None:
            compatible = fmt.upper().replace(" ", "") in self._TALK_OK_AUDIO_FORMATS
            if compatible and rate is not None:
                compatible = rate.split(".")[0] == "8000"
        return {"audio_format": fmt, "sample_rate": rate, "compatible": compatible}

    async def talk_stream(self, pcm_chunks: AsyncIterator[bytes]) -> None:
        """Stream browser mic audio to the device speaker.

        `pcm_chunks` yields raw little-endian Int16 mono 8 kHz PCM (the
        browser downsamples); each chunk is transcoded to G.711 A-law and
        fed through the same audio.cgi postAudio path the siren uses.

        A streaming body cannot be replayed when digest auth answers 401,
        so prime the challenge cache with a cheap authenticated GET first —
        httpx.DigestAuth caches the challenge on the auth instance and then
        attaches Authorization preemptively to the streaming POST.

        Raises AmcrestAuthError/AmcrestError when the device is unreachable
        or rejects the stream, AmcrestUnsupportedError when the firmware
        has no postAudio.
        """
        log.info("camera %s [%s]: talk_stream priming digest challenge", self.ip, self.model or "?")
        try:
            await self.fetch_device_type()
        except AmcrestUnsupportedError:
            pass  # device answered (challenge primed); getDeviceType just missing

        # AD410 preflight: the doorbell silently drops POSTed talk audio unless
        # its audio Encode is G.711A/G.711Mu @ 8 kHz (factory default is 16 kHz
        # AAC). Read it and log an actionable warning — we do NOT rewrite the
        # device's persistent audio config behind the user's back.
        if self._is_ad410():
            diag = await self.talk_audio_diagnostic()
            if diag is None:
                log.info(
                    "camera %s [AD410]: could not read audio Encode config for talk preflight",
                    self.ip,
                )
            elif diag.get("compatible") is False:
                log.warning(
                    "camera %s [AD410]: audio Encode is %r @ %r Hz — two-way talk audio is "
                    "likely to be SILENTLY DROPPED. The AD410 only honors POSTed talk audio "
                    "when its audio format is G.711A/G.711Mu at an 8000 Hz sample rate; set the "
                    "doorbell's audio encoding accordingly. Even then, some AD410 firmwares "
                    "refuse third-party POSTed talk entirely (see client.py docstring).",
                    self.ip, diag.get("audio_format"), diag.get("sample_rate"),
                )
            else:
                log.info(
                    "camera %s [AD410]: audio Encode looks talk-compatible (%r @ %r Hz)",
                    self.ip, diag.get("audio_format"), diag.get("sample_rate"),
                )

        sent = {"chunks": 0, "bytes": 0}

        async def alaw_chunks() -> AsyncIterator[bytes]:
            async for chunk in pcm_chunks:
                encoded = pcm16le_to_alaw(chunk)
                if encoded:
                    sent["chunks"] += 1
                    sent["bytes"] += len(encoded)
                    if sent["chunks"] == 1:
                        log.info("camera %s: talk_stream first A-law chunk sent", self.ip)
                    yield encoded
            log.info(
                "camera %s: talk_stream body ended (%d chunks, %d A-law bytes)",
                self.ip, sent["chunks"], sent["bytes"],
            )

        # write= caps each chunk write, not the session — total session
        # length is bounded by the caller ending the iterator (the talk WS
        # enforces the 120 s cap).
        await self._post_audio(
            alaw_chunks(),
            timeout=httpx.Timeout(30.0, connect=5.0, read=TALK_READ_TIMEOUT_S),
            context="talk",
        )

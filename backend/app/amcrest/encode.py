"""Main-stream encode profiles: what Vigilume asks a camera's MAIN stream to be.

The main stream is what the 24/7 recorder stores, what the recorded bursts
decode for face reads, what a full-resolution snapshot is cut from,
and what fullscreen live view climbs to. Its settings decide most of the box's
decode work and disk use, and the cameras ship with the heaviest ones they
have (4K/5 MP H.265 on a 2-4 s keyframe interval). A PROFILE says what to
change and leaves the rest alone:

    {"resolution": "keep" | "720p" | "1080p" | "1440p" | "4k" | "max" | "WxH",
     "codec": "keep" | "h264" | "h265",
     "keyframe_s": None | seconds between keyframes,
     "bitrate_kbps": None | kbps}

"keep" / None never touches that setting, so an empty profile is a no-op and
nothing is ever changed on a camera nobody configured.

A resolution like "1080p" is a CEILING, not an exact size, because cameras do
not share a resolution list: the camera's largest offered resolution at or
under that height, keeping the picture's current shape (4:3 vs 16:9 — a
different shape is a different field of view) when the camera offers one of
that shape with at least MIN_SAME_SHAPE_SHARE of the pixels. "WxH" is exact,
for a single camera whose list the client has read.

Everything here is pure: it plans the setConfig keys from a camera's Encode
config + capabilities, and `client.AmcrestClient.apply_main_stream` does the
I/O (write, read back, report what did not stick).
"""
from __future__ import annotations

import re
from typing import Any, Optional

RESOLUTION_CHOICES = ("keep", "720p", "1080p", "1440p", "4k", "max")
CODEC_CHOICES = ("keep", "h264", "h265")
_HEIGHT_CAP = {"720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160}

KEYFRAME_MIN_S = 0.5
KEYFRAME_MAX_S = 10.0
BITRATE_MIN_KBPS = 256
BITRATE_MAX_KBPS = 20480

#: A same-shape resolution is preferred unless it has less than this share of
#: the pixels of the best one of any shape (a 4:3 camera asked for "1080p" gets
#: 1920x1080 rather than 1280x960).
MIN_SAME_SHAPE_SHARE = 0.6

#: Resolution names Dahua firmware uses in caps / config instead of WxH.
NAMED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "CIF": (352, 240), "VGA": (640, 480), "D1": (704, 480), "960H": (928, 480),
    "720P": (1280, 720), "1.3M": (1280, 960), "SXGA": (1280, 1024),
    "UXGA": (1600, 1200), "1080P": (1920, 1080), "3M": (2048, 1536),
    "2304X1296": (2304, 1296), "QHD": (2560, 1440), "1440P": (2560, 1440),
    "4M": (2688, 1520), "5M": (2592, 1944), "6M": (3072, 2048),
    "4K": (3840, 2160), "UHD": (3840, 2160), "2160P": (3840, 2160),
    "8M": (3840, 2160),
}
_WXH = re.compile(r"^\s*(\d{3,5})\s*[xX*]\s*(\d{3,5})\s*$")

#: Used only when the camera will not list its resolutions: standard 16:9
#: sizes, tried top-down, never above what the camera runs now. Whatever is
#: written is read back, so a size the camera refuses is reported, not assumed.
_FALLBACK_LADDER = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720)]

_MAIN_VIDEO_KEY = re.compile(r"^Encode\[0\]\.MainFormat\[(\d+)\]\.Video\.([A-Za-z]+)$")
_CAPS_KEY = re.compile(
    r"MainFormat\[0\]\.Video\.(ResolutionTypes|CompressionTypes|FPSMax|BitRateOptions)$"
)


# ---------- profile ----------

def empty_profile() -> dict[str, Any]:
    return {"resolution": "keep", "codec": "keep", "keyframe_s": None, "bitrate_kbps": None}


def normalize_profile(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """A validated copy of `raw`; raises ValueError on a bad value."""
    out = empty_profile()
    if not raw:
        return out
    res = str(raw.get("resolution") or "keep").strip().lower()
    if res not in RESOLUTION_CHOICES:
        size = parse_resolution(res)
        if size is None:
            raise ValueError(f"resolution must be one of {', '.join(RESOLUTION_CHOICES)} or WxH")
        res = f"{size[0]}x{size[1]}"
    out["resolution"] = res
    codec = str(raw.get("codec") or "keep").strip().lower().replace(".", "")
    if codec not in CODEC_CHOICES:
        raise ValueError("codec must be keep, h264 or h265")
    out["codec"] = codec
    ks = raw.get("keyframe_s")
    if ks is not None:
        ks = float(ks)
        if not KEYFRAME_MIN_S <= ks <= KEYFRAME_MAX_S:
            raise ValueError(f"keyframe_s must be {KEYFRAME_MIN_S}-{KEYFRAME_MAX_S}")
        out["keyframe_s"] = ks
    br = raw.get("bitrate_kbps")
    if br is not None:
        br = int(br)
        if not BITRATE_MIN_KBPS <= br <= BITRATE_MAX_KBPS:
            raise ValueError(f"bitrate_kbps must be {BITRATE_MIN_KBPS}-{BITRATE_MAX_KBPS}")
        out["bitrate_kbps"] = br
    return out


def is_noop(profile: dict[str, Any]) -> bool:
    return (profile.get("resolution", "keep") == "keep" and profile.get("codec", "keep") == "keep"
            and profile.get("keyframe_s") is None and profile.get("bitrate_kbps") is None)


# ---------- reading a camera ----------

def parse_resolution(label: Optional[str]) -> Optional[tuple[int, int]]:
    if not label:
        return None
    m = _WXH.match(label)
    if m:
        return int(m.group(1)), int(m.group(2))
    return NAMED_RESOLUTIONS.get(label.strip().upper())


def _codec_family(value: Optional[str]) -> Optional[str]:
    """"H.264", "H.264H", "H264B" -> "h264"; "H.265" -> "h265"."""
    v = (value or "").upper().replace(".", "").replace(" ", "")
    if v.startswith("H264"):
        return "h264"
    if v.startswith("H265") or v.startswith("HEVC"):
        return "h265"
    return v.lower() or None


def parse_caps(caps: dict[str, str]) -> dict[str, Any]:
    """encode.cgi getConfigCaps -> {resolutions: [(label, w, h)], codecs: [..],
    fps_max, bitrate_range}. Missing pieces are empty / None."""
    out: dict[str, Any] = {"resolutions": [], "codecs": [], "fps_max": None, "bitrate_range": None}
    for key, value in caps.items():
        m = _CAPS_KEY.search(key)
        if not m:
            continue
        field = m.group(1)
        items = [v.strip() for v in value.split(",") if v.strip()]
        if field == "ResolutionTypes":
            for label in items:
                size = parse_resolution(label)
                if size and all(label != r[0] for r in out["resolutions"]):
                    out["resolutions"].append((label, size[0], size[1]))
        elif field == "CompressionTypes":
            out["codecs"] = [c for c in items]
        elif field == "FPSMax":
            try:
                out["fps_max"] = int(float(value))
            except ValueError:
                pass
        elif field == "BitRateOptions" and len(items) >= 2:
            try:
                nums = [int(float(v)) for v in items]
                out["bitrate_range"] = (min(nums), max(nums))
            except ValueError:
                pass
    return out


def main_formats(cfg: dict[str, str]) -> dict[int, dict[str, str]]:
    """Encode[0].MainFormat[N].Video.* -> {N: {field: value}}, for the formats
    that actually carry video (N=0 is the regular stream; 1-2 are the motion /
    alarm variants some firmware switches the stream to during an event, which
    is why every one of them is kept in step)."""
    out: dict[int, dict[str, str]] = {}
    for key, value in cfg.items():
        m = _MAIN_VIDEO_KEY.match(key)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = value
    return {n: v for n, v in sorted(out.items()) if "Compression" in v}


def current_main(cfg: dict[str, str]) -> Optional[dict[str, Any]]:
    """The regular main stream as it is now, or None when unreadable."""
    formats = main_formats(cfg)
    if not formats:
        return None
    v = formats[min(formats)]
    size = _size_of(v)

    def num(field: str) -> Optional[int]:
        try:
            return int(float(v[field]))
        except (KeyError, ValueError):
            return None

    fps, gop = num("FPS"), num("GOP")
    return {
        "width": size[0] if size else None,
        "height": size[1] if size else None,
        "codec": _codec_family(v.get("Compression")),
        "codec_raw": v.get("Compression"),
        "fps": fps,
        "gop": gop,
        "keyframe_s": round(gop / fps, 2) if fps and gop else None,
        "bitrate_kbps": num("BitRate"),
        "bitrate_control": v.get("BitRateControl"),
        "profile": v.get("Profile"),
    }


_SUB_VIDEO_KEY = re.compile(r"^Encode\[0\]\.ExtraFormat\[0\]\.Video\.([A-Za-z]+)$")


def current_sub(cfg: dict[str, str]) -> Optional[dict[str, Any]]:
    """The SUBSTREAM (ExtraFormat[0], RTSP subtype=1) as it is now, or None.
    It is what every live tile and fullscreen view opens on, and what the
    detector ingests."""
    v = {m.group(1): val for key, val in cfg.items() if (m := _SUB_VIDEO_KEY.match(key))}
    if "Compression" not in v:
        return None
    size = _size_of(v)
    return {
        "codec": _codec_family(v.get("Compression")),
        "codec_raw": v.get("Compression"),
        "width": size[0] if size else None,
        "height": size[1] if size else None,
    }


def sub_needs_h264(cfg: dict[str, str]) -> bool:
    """Live view needs an H.264 substream: go2rtc cannot restream MJPEG at all
    (the tile shows only its poster image), and H.265 has no WebRTC decoder on
    iOS, so it falls back to slow HLS."""
    sub = current_sub(cfg)
    return sub is not None and sub["codec"] != "h264"


def _size_of(v: dict[str, str]) -> Optional[tuple[int, int]]:
    try:
        return int(v["Width"]), int(v["Height"])
    except (KeyError, ValueError):
        pass
    return parse_resolution(v.get("resolution")) or parse_resolution(v.get("CustomResolutionName"))


# ---------- choosing ----------

def choose_resolution(
    setting: str, options: list[tuple[str, int, int]], current: Optional[tuple[int, int]],
) -> tuple[Optional[tuple[str, int, int]], Optional[str]]:
    """((label, w, h) to write or None, note). None with no note = nothing to do."""
    if setting == "keep":
        return None, None
    exact = parse_resolution(setting) if setting not in RESOLUTION_CHOICES else None
    if not options:
        if current is None:
            return None, "the camera reports neither its resolution nor the ones it supports"
        if setting == "max":
            return None, "the camera does not list its resolutions, so its maximum is unknown"
        cap = exact[1] if exact else _HEIGHT_CAP[setting]
        ladder = [(w, h) for w, h in ([exact] if exact else _FALLBACK_LADDER)
                  if h <= cap and w * h <= current[0] * current[1]]
        if not ladder:
            return None, f"nothing at or under {cap}p below the current {current[0]}x{current[1]}"
        w, h = ladder[0]
        return (f"{w}x{h}", w, h), "the camera does not list its resolutions; trying a standard size"
    if exact:
        match = next((o for o in options if (o[1], o[2]) == exact), None)
        if match is None:
            return None, f"{exact[0]}x{exact[1]} is not one of this camera's resolutions"
        return match, None
    if setting == "max":
        return max(options, key=lambda o: o[1] * o[2]), None
    cap = _HEIGHT_CAP[setting]
    fitting = [o for o in options if o[2] <= cap]
    if not fitting:
        smallest = min(options, key=lambda o: o[1] * o[2])
        return None, f"the camera's smallest resolution ({smallest[1]}x{smallest[2]}) is over {cap}p"
    best = max(fitting, key=lambda o: o[1] * o[2])
    if current:
        shape = current[0] / current[1]
        same = [o for o in fitting if abs(o[1] / o[2] - shape) < 0.05]
        if same:
            best_same = max(same, key=lambda o: o[1] * o[2])
            if best_same[1] * best_same[2] >= MIN_SAME_SHAPE_SHARE * best[1] * best[2]:
                best = best_same
    return best, None


def _caps_codec_label(family: str, offered: list[str]) -> Optional[str]:
    """The camera's own spelling of a codec family, or None if not offered."""
    for label in offered:
        if _codec_family(label) == family and label.upper().replace(".", "") in ("H264", "H265"):
            return label
    for label in offered:
        if _codec_family(label) == family:
            return label
    return None


def resolution_variants(
    v: dict[str, str], target: tuple[str, int, int], extra: Optional[dict[str, str]] = None,
) -> list[dict[str, str]]:
    """Ways of asking ONE format for `target`, most likely first (field names
    without the `Encode[0].MainFormat[N].Video.` prefix).

    Amcrest firmware is not consistent about which key sets the size. Most
    report Width/Height (and a CustomResolutionName next to them); the Dahua
    API documents a `resolution` key that takes a name ("1080P") or "WxH".
    Writing all of them at once was REFUSED by real cameras, so each is tried
    on its own and the size read back after each one. `extra` (the profile's
    bitrate) rides along with every attempt: these cameras check the bitrate
    against the new size, and a 4K bitrate can block a drop to 1080p."""
    label, w, h = target
    wxh = f"{w}x{h}"
    out: list[dict[str, str]] = []

    def add(d: dict[str, str]) -> None:
        d = {**d, **(extra or {})}
        if d not in out:
            out.append(d)

    if "Width" in v and "Height" in v:
        add({"Width": str(w), "Height": str(h)})
        if "CustomResolutionName" in v:
            add({"Width": str(w), "Height": str(h), "CustomResolutionName": label})
            add({"Width": str(w), "Height": str(h), "CustomResolutionName": wxh})
    add({"resolution": label})
    add({"resolution": wxh})
    if "CustomResolutionName" in v:
        add({"CustomResolutionName": label})
        add({"CustomResolutionName": wxh})
    return out


def plan_main_stream(
    cfg: dict[str, str], caps: dict[str, Any], profile: dict[str, Any],
) -> dict[str, Any]:
    """What moving every main format to `profile` takes.

    Returns:
      primary            the regular stream's format index (the one recorded
                         and streamed; the others are the variants some
                         firmware switches to during an event)
      formats            {N: {field: value}} as read
      resolution         (label, w, h) to set, or None
      resolution_formats formats whose size differs, primary first
      settings           {N: {group: {key: value}}} for codec / keyframes /
                         bitrate (full setConfig keys)
      changes, keys      a human line per planned change and its
                         "<format>:<setting>" id, so a read-back can tell
                         which ones stuck
      notes
    Only values that differ are planned, so applying twice is a no-op."""
    changes: list[str] = []
    keys: list[str] = []
    notes: list[str] = []
    formats = main_formats(cfg)
    empty = {"primary": None, "formats": {}, "resolution": None, "resolution_formats": [],
             "settings": {}, "changes": [], "keys": [], "notes": notes}
    if not formats:
        notes.append("the camera did not report a main stream")
        return empty
    primary = min(formats)
    now = current_main(cfg) or {}
    cur_size = (now["width"], now["height"]) if now.get("width") else None

    target, note = choose_resolution(profile["resolution"], caps.get("resolutions") or [], cur_size)
    if note:
        notes.append(note)

    codec_label = None
    if profile["codec"] != "keep":
        offered = caps.get("codecs") or []
        if offered:
            codec_label = _caps_codec_label(profile["codec"], offered)
            if codec_label is None:
                notes.append(f"the camera does not offer {profile['codec'].upper()}")
        else:
            codec_label = "H.264" if profile["codec"] == "h264" else "H.265"

    bitrate = profile["bitrate_kbps"]
    if bitrate is not None and caps.get("bitrate_range"):
        lo, hi = caps["bitrate_range"]
        if not lo <= bitrate <= hi:
            notes.append(f"bitrate clamped to the camera's {lo}-{hi} kbps")
            bitrate = max(lo, min(hi, bitrate))

    resolution_formats: list[int] = []
    settings: dict[int, dict[str, dict[str, str]]] = {}
    for n, v in formats.items():
        p = f"Encode[0].MainFormat[{n}].Video."
        label = "main" if n == primary else f"main #{n}"
        size = _size_of(v)
        # A secondary format smaller than the target is left alone: "up to
        # 1080p" is a ceiling, and real cameras keep a 1280x720 format #3 that
        # refuses to be raised (it made every camera's change fail).
        smaller = (n != primary and size is not None
                   and size[0] * size[1] < target[1] * target[2]) if target else False
        if target is not None and size != (target[1], target[2]) and not smaller:
            resolution_formats.append(n)
            was = f"{size[0]}x{size[1]}" if size else "?"
            changes.append(f"{label} resolution {was} -> {target[1]}x{target[2]}")
            keys.append(f"{n}:resolution")
        mine = settings.setdefault(n, {})
        if codec_label is not None and _codec_family(v.get("Compression")) != profile["codec"]:
            mine["codec"] = {p + "Compression": codec_label}
            changes.append(f"{label} codec {v.get('Compression')} -> {codec_label}")
            keys.append(f"{n}:codec")
        if profile["keyframe_s"] is not None:
            try:
                fps = int(float(v.get("FPS", "")))
                gop = int(float(v.get("GOP", "")))
            except ValueError:
                fps = gop = 0
            if fps > 0:
                want = max(1, int(round(fps * profile["keyframe_s"])))
                if want != gop:
                    mine["keyframes"] = {p + "GOP": str(want)}
                    changes.append(f"{label} keyframe every {gop} -> {want} frames ({fps} fps)")
                    keys.append(f"{n}:keyframes")
            elif n == primary:
                notes.append("the camera does not report its frame rate; keyframes left alone")
        if bitrate is not None:
            try:
                cur_br = int(float(v.get("BitRate", "")))
            except ValueError:
                cur_br = None
            if cur_br != bitrate:
                mine["bitrate"] = {p + "BitRate": str(bitrate)}
                changes.append(f"{label} bitrate {cur_br} -> {bitrate} kbps")
                keys.append(f"{n}:bitrate")
    resolution_formats.sort(key=lambda n: (n != primary, n))
    return {"primary": primary, "formats": formats, "resolution": target,
            "resolution_formats": resolution_formats, "bitrate": bitrate,
            "settings": {n: g for n, g in settings.items() if g},
            "changes": changes, "keys": keys, "notes": notes}


#: Order settings are retried in, one at a time, when the camera rejects them
#: together. Resolution is always set first (and alone): some cameras refuse
#: H.264 at 4K but take it at 1080p.
GROUP_ORDER = ("codec", "keyframes", "bitrate")

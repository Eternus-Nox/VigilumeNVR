"""HEVC -> H.264 transcode smoke suite (browser-playback fix).

Covers native/transcode.py + its wiring into native/recorder.py (clips) and
routers/recordings.py (timeline segments). CPU-only; the subprocess is MOCKED
(this suite must pass on hosts without ffmpeg/NVENC — the real NVENC path is
confirmed by the user on their GPU box). An optional REAL section runs a genuine
libx264 transcode only when ffmpeg is on PATH (feature-detected, never installs).

Sections:

  1. codec branch — is_browser_playable / needs_transcode truth table;
     parse_probe_output for video+audio, ordering, garbage/empty tolerance.
  2. arg builders — golden argv for the ffprobe probe, the encoders listing +
     select_encoder, and build_transcode_args for {nvenc, libx264} x
     {segment mpegts, clip mp4 faststart}, incl. audio copy-vs-aac and the
     seek/duration clamp.
  3. encoder auto-select — mocked probe reports nvenc -> NVENC, else LIBX264;
     result cached; mark_nvenc_failed() downgrades to libx264 once.
  4. segment cache — probe cache (TTL), h264 -> passthrough (None), hevc ->
     transcodes + caches, cache HIT skips ffmpeg, LRU eviction by byte cap,
     concurrent same-key requests share ONE transcode (dedupe).
  5. clip plan / recorder — clip_plan chooses copy for h264 and transcode for
     hevc; extract_clip uses build_transcode_args for an HEVC source and
     build_clip_args for an H.264 source (mocked _run_ffmpeg captures argv);
     nvenc runtime failure -> libx264 retry; both fail -> stream-copy fallback.
  6. failure handling — a failing transcode (mocked rc!=0) makes
     segment_for_playback return None (serve original, no crash); disabled
     transcoder (no ffmpeg) passes everything through.
  7. REAL ffmpeg (optional) — genuine libx264 segment + clip transcodes produce
     playable H.264 output when ffmpeg is on PATH.

    python backend/tests/transcode_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-transcode-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.native.recorder import CLIP_CONCURRENCY, Recorder  # noqa: E402
from app.native.transcode import (  # noqa: E402
    LIBX264,
    NVENC,
    VAAPI,
    ClipPlan,
    ProbeResult,
    Transcoder,
    build_encoders_probe_args,
    build_probe_args,
    build_transcode_args,
    find_nvidia_device,
    find_vaapi_device,
    is_browser_playable,
    needs_transcode,
    parse_probe_output,
    select_encoder,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


DEFAULT_RECORDING = {"continuous_days": 7, "event_days": 14, "snapshot_days": 14}


class FakeSettings:

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False
    def __init__(self, recording: dict | None = None):
        self.recording = dict(recording or DEFAULT_RECORDING)


def make_config(tag: str) -> Config:
    cfg = Config()
    cfg.data_dir = TMP / tag / "data"
    cfg.media_dir = TMP / tag / "media"
    return cfg


def seg_path(cam_dir: Path, dt: datetime) -> Path:
    return cam_dir / dt.strftime("%Y-%m-%d") / dt.strftime("%H") / dt.strftime("%M.%S.ts")


def make_seg(cam_dir: Path, dt: datetime, body: bytes = b"\x47" * 188) -> Path:
    p = seg_path(cam_dir, dt)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


def make_transcoder(tag: str, **over) -> Transcoder:
    """A transcoder with injected fake binaries (so .enabled is True) and a
    dedicated cache dir. Callers monkeypatch ._run.

    Both device probes are pinned so the suite behaves identically on a box that
    has a GPU and one that does not — otherwise encoder selection would depend
    on the host running the tests. ``nvidia_present=True`` keeps the NVENC cases
    (which supply an nvenc listing) exercising the GPU branch; VAAPI cases pass
    a ``vaapi_device`` in explicitly."""
    kwargs = dict(
        cache_dir=TMP / tag / "cache", ffmpeg="/fake/ffmpeg", ffprobe="/fake/ffprobe",
        vaapi_device=None, nvidia_present=True,
    )
    kwargs.update(over)
    return Transcoder(**kwargs)


# =====================================================================
# 1. codec branch + probe parsing
# =====================================================================


def codec_branch_checks() -> None:
    print("1. codec branch — is_browser_playable / needs_transcode / parse_probe_output")
    check(is_browser_playable("h264") and is_browser_playable("H264")
          and is_browser_playable("avc1"),
          "h264/avc are browser-playable (case-insensitive)")
    check(not is_browser_playable("hevc") and not is_browser_playable("h265")
          and not is_browser_playable("mpeg4") and not is_browser_playable(None),
          "hevc/h265/mpeg4/None are NOT browser-playable")

    check(not needs_transcode("h264") and not needs_transcode("avc1"),
          "h264 -> no transcode (fast copy path)")
    check(needs_transcode("hevc") and needs_transcode("h265") and needs_transcode("mpeg4"),
          "hevc/h265/mpeg4 -> transcode")
    check(not needs_transcode(None) and not needs_transcode(""),
          "unknown/None codec -> passthrough (never transcode garbage)")

    # parse_probe_output: video before audio, audio before video, missing audio,
    # empty, and 'unknown' codec_name tolerated.
    hevc = parse_probe_output("codec_name=hevc\ncodec_type=video\ncodec_name=aac\ncodec_type=audio\n")
    check(hevc == ProbeResult("hevc", "aac"), "parse: video+audio codecs extracted")
    rev = parse_probe_output("codec_type=audio\ncodec_name=aac\ncodec_type=video\ncodec_name=hevc\n")
    check(rev == ProbeResult("hevc", "aac"), "parse: robust to type-before-name ordering")
    novid = parse_probe_output("codec_name=h264\ncodec_type=video\n")
    check(novid == ProbeResult("h264", None), "parse: missing audio -> audio None")
    check(parse_probe_output("") == ProbeResult(None, None), "parse: empty output -> (None, None)")
    check(parse_probe_output("codec_name=unknown\ncodec_type=video\n") == ProbeResult(None, None),
          "parse: 'unknown' codec_name treated as None")


# =====================================================================
# 2. golden arg builders
# =====================================================================


def arg_builder_checks() -> None:
    print("2. arg builders — probe / encoders listing / transcode argv (golden)")
    check(build_probe_args("/u/ffprobe", "/x/seg.ts") == [
        "/u/ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "default=noprint_wrappers=1",
        "/x/seg.ts",
    ], "ffprobe argv: per-stream codec_type+codec_name, no wrappers")

    check(build_encoders_probe_args("/u/ffmpeg") == ["/u/ffmpeg", "-hide_banner", "-encoders"],
          "encoders-listing argv")
    nvenc_listing = "... V..... h264_nvenc  NVIDIA NVENC H.264 ..."
    check(select_encoder(nvenc_listing, None, True) == NVENC,
          "select_encoder picks h264_nvenc when listed and an NVIDIA node exists")
    check(select_encoder("... V..... libx264  libx264 H.264 ...", None, True) == LIBX264,
          "select_encoder falls back to libx264 when nvenc absent")

    # -- VAAPI (AMD/Intel iGPU) sits between NVENC and libx264 --
    vaapi_listing = "... V..... h264_vaapi  H.264/AVC (VAAPI) ...\n V..... libx264 ..."
    both_listing = vaapi_listing + "\n V..... h264_nvenc  NVIDIA NVENC"
    check(select_encoder(vaapi_listing, "/dev/dri/renderD128") == VAAPI,
          "select_encoder picks h264_vaapi when listed AND a render node exists")
    check(select_encoder(vaapi_listing, None) == LIBX264,
          "select_encoder skips VAAPI with no render node (would fail every run)")
    check(select_encoder(vaapi_listing) == LIBX264,
          "select_encoder defaults to no devices (back-compat single-arg call)")
    check(select_encoder(both_listing, "/dev/dri/renderD128", True) == NVENC,
          "select_encoder prefers NVENC over VAAPI when a real dGPU is present")
    check(select_encoder("... V..... libx264 ...", "/dev/dri/renderD128") == LIBX264,
          "select_encoder ignores a render node when ffmpeg has no h264_vaapi")

    # THE AMD-BOX CASE. Distro ffmpeg lists h264_nvenc whether or not an NVIDIA
    # card exists, so selecting on the listing alone sends an AMD mini PC to
    # NVENC -> fail -> libx264, with the iGPU never touched. The device gate is
    # the whole reason VAAPI is reachable in practice.
    check(select_encoder(both_listing, "/dev/dri/renderD128", False) == VAAPI,
          "select_encoder: nvenc LISTED but no NVIDIA node + a render node -> VAAPI")
    check(select_encoder(both_listing, None, False) == LIBX264,
          "select_encoder: encoders listed but NO devices at all -> libx264")

    # -- exclude: a runtime failure walks to the next real candidate --
    check(select_encoder(both_listing, "/dev/dri/renderD128", True,
                         frozenset({NVENC})) == VAAPI,
          "select_encoder: excluding a failed NVENC falls through to VAAPI, not CPU")
    check(select_encoder(both_listing, "/dev/dri/renderD128", True,
                         frozenset({NVENC, VAAPI})) == LIBX264,
          "select_encoder: both hardware encoders excluded -> libx264")

    # -- NVIDIA node discovery --
    check(find_nvidia_device(exists=lambda p: p == "/dev/nvidiactl") is True,
          "find_nvidia_device: nvidiactl (injected by the nvidia runtime) counts")
    check(find_nvidia_device(exists=lambda p: False) is False,
          "find_nvidia_device: no node -> False (an AMD/Intel-only box)")

    # -- render-node discovery: env override, then the standard nodes --
    check(find_vaapi_device(env="", exists=lambda p: p == "/dev/dri/renderD128")
          == "/dev/dri/renderD128", "find_vaapi_device: probes renderD128 first")
    check(find_vaapi_device(env="", exists=lambda p: p == "/dev/dri/renderD129")
          == "/dev/dri/renderD129", "find_vaapi_device: falls through to renderD129")
    check(find_vaapi_device(env="", exists=lambda p: False) is None,
          "find_vaapi_device: no node -> None (VAAPI simply not a candidate)")
    check(find_vaapi_device(env="/dev/dri/card9", exists=lambda p: True) == "/dev/dri/card9",
          "find_vaapi_device: env override wins when it exists")
    check(find_vaapi_device(env="/dev/dri/nope",
                            exists=lambda p: p == "/dev/dri/renderD128")
          == "/dev/dri/renderD128",
          "find_vaapi_device: a missing override warns and falls back, never crashes")

    # -- segment (mpegts) --
    nv_seg = build_transcode_args(
        "ffmpeg", NVENC, container="mpegts", output="/c/out.ts",
        input_path="/r/in.ts", audio_codec="aac",
    )
    check(nv_seg == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", "/r/in.ts",
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-c:a", "copy",
        "-f", "mpegts", "/c/out.ts",
    ], "NVENC segment: full-GPU decode+encode, mpegts out, audio copy (aac source)")

    va_seg = build_transcode_args(
        "ffmpeg", VAAPI, container="mpegts", output="/c/out.ts",
        input_path="/r/in.ts", audio_codec="aac", vaapi_device="/dev/dri/renderD128",
    )
    check(va_seg == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128",
        "-hwaccel_output_format", "vaapi",
        "-i", "/r/in.ts",
        "-vf", "format=nv12|vaapi,hwupload",
        "-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", "23",
        "-c:a", "copy",
        "-f", "mpegts", "/c/out.ts",
    ], "VAAPI segment: iGPU decode+encode on the render node, mpegts out")

    va_nodev = build_transcode_args(
        "ffmpeg", VAAPI, container="mpegts", output="/c/out.ts", input_path="/r/in.ts",
    )
    check("-hwaccel_device" not in va_nodev
          and va_nodev[:7] == ["ffmpeg", "-hide_banner", "-loglevel", "warning",
                               "-nostdin", "-hwaccel", "vaapi"],
          "VAAPI without an explicit device omits -hwaccel_device (ffmpeg picks the default node)")

    x_seg = build_transcode_args(
        "ffmpeg", LIBX264, container="mpegts", output="/c/out.ts",
        input_path="/r/in.ts", audio_codec="pcm_mulaw",
    )
    check(x_seg == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-i", "/r/in.ts",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        "-f", "mpegts", "/c/out.ts",
    ], "libx264 segment: no hwaccel, mpegts out, non-aac audio -> aac")

    # -- clip (mp4 faststart, concat + seek/duration) --
    nv_clip = build_transcode_args(
        "ffmpeg", NVENC, container="mp4", output="/c/7.part.mp4",
        concat_list="/c/7.txt", seek_s=5.0, duration_s=25.0, audio_codec="aac",
    )
    check(nv_clip == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-f", "concat", "-safe", "0", "-i", "/c/7.txt",
        "-ss", "5.000", "-t", "25.000",
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-f", "mp4", "/c/7.part.mp4",
    ], "NVENC clip: concat + precise cut + faststart mp4")

    va_clip = build_transcode_args(
        "ffmpeg", VAAPI, container="mp4", output="/c/7.part.mp4",
        concat_list="/c/7.txt", seek_s=5.0, duration_s=25.0, audio_codec="aac",
        vaapi_device="/dev/dri/renderD128",
    )
    check(va_clip == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128",
        "-hwaccel_output_format", "vaapi",
        "-f", "concat", "-safe", "0", "-i", "/c/7.txt",
        "-ss", "5.000", "-t", "25.000",
        "-vf", "format=nv12|vaapi,hwupload",
        "-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-f", "mp4", "/c/7.part.mp4",
    ], "VAAPI clip: concat + precise cut + faststart mp4 on the iGPU")

    # The hwupload filter must come AFTER the cut, or ffmpeg uploads frames it
    # is about to discard (and -ss/-t would apply to the wrong filter graph).
    check(va_clip.index("-t") < va_clip.index("-vf") < va_clip.index("-c:v"),
          "VAAPI: filter sits between the output-side cut and the encoder")

    x_clip = build_transcode_args(
        "ffmpeg", LIBX264, container="mp4", output="/c/7.part.mp4",
        concat_list="/c/7.txt", seek_s=-3.0, duration_s=-1.0, audio_codec=None,
    )
    check("-hwaccel" not in x_clip and x_clip[x_clip.index("-c:v") + 1] == "libx264"
          and "+faststart" in x_clip
          and x_clip[-3:] == ["-f", "mp4", "/c/7.part.mp4"],
          "libx264 clip: no hwaccel, faststart mp4")
    check(x_clip[x_clip.index("-ss") + 1] == "0.000"
          and x_clip[x_clip.index("-t") + 1] == "0.000",
          "negative seek/duration clamp to 0.000")
    check(x_clip[x_clip.index("-c:a") + 1] == "aac",
          "no audio codec info -> re-encode audio to aac (safe default)")


# =====================================================================
# 3. encoder auto-select + runtime downgrade
# =====================================================================


async def _encoder_cases() -> None:
    # nvenc present in the (mocked) listing -> NVENC, cached (one probe).
    t = make_transcoder("enc-nvenc")
    calls = {"n": 0}

    async def run_nvenc(args, timeout=None):
        calls["n"] += 1
        return 0, b"V..... h264_nvenc  NVIDIA NVENC", b""

    t._run = run_nvenc
    check(await t.encoder() == NVENC, "encoder() -> NVENC when the listing reports it")
    check(await t.encoder() == NVENC and calls["n"] == 1,
          "encoder() caches the selection (single ffmpeg -encoders probe)")

    # nvenc absent -> libx264.
    t2 = make_transcoder("enc-x264")

    async def run_x264(args, timeout=None):
        return 0, b"V..... libx264  libx264 H.264", b""

    t2._run = run_x264
    check(await t2.encoder() == LIBX264, "encoder() -> libx264 when nvenc absent")

    # runtime downgrade: NVENC selected, then mark_nvenc_failed sticks libx264.
    t3 = make_transcoder("enc-downgrade")
    t3._run = run_nvenc
    check(await t3.encoder() == NVENC, "encoder() starts on NVENC")
    t3.mark_nvenc_failed()
    check(await t3.encoder() == LIBX264, "mark_nvenc_failed() -> libx264 thereafter")
    t3.mark_nvenc_failed()  # idempotent, no crash
    check(await t3.encoder() == LIBX264, "mark_nvenc_failed() is idempotent")

    # -- VAAPI: selected only with a render node, and downgrades the same way --
    async def run_vaapi(args, timeout=None):
        return 0, b"V..... h264_vaapi  H.264/AVC (VAAPI)\nV..... libx264 ...", b""

    t5 = make_transcoder("enc-vaapi", vaapi_device="/dev/dri/renderD128")
    t5._run = run_vaapi
    check(await t5.encoder() == VAAPI, "encoder() -> VAAPI on a box with an iGPU render node")
    check(t5.vaapi_device == "/dev/dri/renderD128",
          "vaapi_device is exposed for the recorder's own clip argv")

    t6 = make_transcoder("enc-vaapi-nodev")  # vaapi_device pinned None
    t6._run = run_vaapi
    check(await t6.encoder() == LIBX264,
          "encoder() -> libx264 when ffmpeg has h264_vaapi but no node is passed through")
    check(t6.vaapi_device is None, "no render node -> vaapi_device is None")

    t7 = make_transcoder("enc-vaapi-downgrade", vaapi_device="/dev/dri/renderD128")
    t7._run = run_vaapi
    check(await t7.encoder() == VAAPI, "encoder() starts on VAAPI")
    t7.mark_hw_failed(VAAPI)
    check(await t7.encoder() == LIBX264,
          "mark_hw_failed(VAAPI) -> libx264 thereafter (driver/permission failure)")

    # disabled transcoder (no binaries) -> libx264 default, never probes.
    # ("" forces-disabled even on a host that has ffmpeg on PATH; None means
    # auto-detect via shutil.which, which is the production behavior.)
    t4 = Transcoder(cache_dir=TMP / "enc-disabled" / "cache", ffmpeg="", ffprobe="")
    check(not t4.enabled and await t4.encoder() == LIBX264,
          "disabled transcoder defaults encoder() to libx264 without probing")


def encoder_checks() -> None:
    print("3. encoder auto-select — nvenc-when-present, cached, runtime downgrade")
    asyncio.run(_encoder_cases())


# =====================================================================
# 4. segment probe cache + transcode cache + LRU + dedupe
# =====================================================================


async def _segment_cache_cases() -> None:
    cam_dir = TMP / "segcache" / "front"
    seg = make_seg(cam_dir, datetime(2026, 7, 4, 12, 0, 0))
    ts = 1000

    # -- probe cache (TTL): one ffprobe per camera within the TTL --
    t = make_transcoder("seg-probe")
    probe_calls = {"n": 0}

    async def run_probe_hevc(args, timeout=None):
        probe_calls["n"] += 1
        return 0, b"codec_name=hevc\ncodec_type=video\ncodec_name=aac\ncodec_type=audio\n", b""

    t._run = run_probe_hevc
    p1 = await t.probe("front", seg)
    p2 = await t.probe("front", seg)
    check(p1 == ProbeResult("hevc", "aac") and probe_calls["n"] == 1,
          "probe() caches per camera (one ffprobe within TTL)")
    t.invalidate("front")
    await t.probe("front", seg)
    check(probe_calls["n"] == 2, "invalidate() forces a re-probe")

    # -- h264 source -> passthrough (None), no transcode --
    th = make_transcoder("seg-h264")

    async def run_probe_h264(args, timeout=None):
        return 0, b"codec_name=h264\ncodec_type=video\n", b""

    th._run = run_probe_h264
    check(await th.segment_for_playback("front", ts, seg) is None,
          "h264 source -> segment_for_playback returns None (serve raw)")

    # -- hevc source -> transcode + cache; second call is a cache HIT --
    tt = make_transcoder("seg-hevc")
    ops: list[str] = []

    async def run_hevc(args, timeout=None):
        if args and args[0] == "/fake/ffprobe":
            return 0, b"codec_name=hevc\ncodec_type=video\ncodec_name=aac\ncodec_type=audio\n", b""
        if args and "-encoders" in args:
            ops.append("encoders")
            return 0, b"V..... libx264  libx264", b""
        # transcode: write the output file the argv names.
        ops.append("transcode")
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"H264-TS-BYTES" * 8)
        return 0, b"", b""

    tt._run = run_hevc
    out1 = await tt.segment_for_playback("front", ts, seg)
    check(out1 is not None and out1.is_file() and out1.suffix == ".ts",
          "hevc source -> transcodes to a cached .ts")
    check(ops.count("transcode") == 1, "one transcode on the cache miss")
    out2 = await tt.segment_for_playback("front", ts, seg)
    check(out2 == out1 and ops.count("transcode") == 1,
          "second request is a cache HIT (no second transcode)")

    # -- LRU eviction by byte cap --
    tl = make_transcoder("seg-lru", cache_max_bytes=300)

    async def run_lru(args, timeout=None):
        if args and args[0] == "/fake/ffprobe":
            return 0, b"codec_name=hevc\ncodec_type=video\n", b""
        if args and "-encoders" in args:
            return 0, b"libx264", b""
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 200)  # each transcoded segment ~200 bytes
        return 0, b"", b""

    tl._run = run_lru
    first = await tl.segment_for_playback("front", 1, seg)
    await asyncio.sleep(0.02)
    second = await tl.segment_for_playback("front", 2, seg)  # total 400 > 300 cap
    remaining = list((TMP / "seg-lru" / "cache").glob("*.ts"))
    check(second is not None and second.is_file(), "newest transcoded segment kept")
    check(not first.exists() and len(remaining) == 1,
          "LRU evicts the oldest segment once the byte cap is exceeded")

    # -- dedupe: concurrent same-key requests share ONE transcode --
    td = make_transcoder("seg-dedupe")
    started = {"n": 0}
    gate = asyncio.Event()

    async def run_slow(args, timeout=None):
        if args and args[0] == "/fake/ffprobe":
            return 0, b"codec_name=hevc\ncodec_type=video\n", b""
        if args and "-encoders" in args:
            return 0, b"libx264", b""
        started["n"] += 1
        await gate.wait()  # hold the transcode open while both callers wait
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"DEDUPE")
        return 0, b"", b""

    td._run = run_slow
    # warm the probe + encoder caches so the two racing calls hit only the transcode.
    await td.probe("front", seg)
    await td.encoder()
    a = asyncio.create_task(td.segment_for_playback("front", 99, seg))
    b = asyncio.create_task(td.segment_for_playback("front", 99, seg))
    await asyncio.sleep(0.05)
    gate.set()
    ra, rb = await asyncio.gather(a, b)
    check(ra == rb and ra is not None, "concurrent same-key requests resolve to the same file")
    check(started["n"] == 1, "in-flight dedupe: exactly ONE ffmpeg for the shared segment")


def segment_cache_checks() -> None:
    print("4. segment cache — probe TTL, passthrough, transcode+cache, LRU, dedupe")
    asyncio.run(_segment_cache_cases())


# =====================================================================
# 5. clip_plan + recorder.extract_clip branch (mocked _run_ffmpeg)
# =====================================================================


async def _clip_branch_cases() -> None:
    cfg = make_config("clipbranch")
    db = Database(cfg.data_dir / "clip.db")
    await db.connect()

    t0 = datetime(2026, 7, 4, 10, 0, 0).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    for off in (0, 10, 20, 30, 40):
        make_seg(cam_dir, datetime.fromtimestamp(t0 + off))
    start_time, end_time = t0 + 20, t0 + 35

    rec = Recorder(cfg, db, FakeSettings())
    rec._ffmpeg_path = "/fake/ffmpeg"
    # Replace the recorder's transcoder with an injected (enabled) one whose
    # probe we control; the recorder RUNS the argv through its own _run_ffmpeg.
    tc = make_transcoder("clipbranch-tc")
    rec._transcode = tc

    # -- clip_plan: h264 -> copy, hevc -> transcode --
    async def probe_h264(camera, sample):
        return ProbeResult("h264", "aac")

    tc.probe = probe_h264
    plan = await tc.clip_plan("front", cam_dir)
    check(plan == ClipPlan(False, None, "h264", "aac"), "clip_plan: h264 -> stream-copy")

    async def probe_hevc(camera, sample):
        return ProbeResult("hevc", "aac")

    tc.probe = probe_hevc

    async def enc_libx264():
        return LIBX264

    tc.encoder = enc_libx264
    plan = await tc.clip_plan("front", cam_dir)
    check(plan == ClipPlan(True, LIBX264, "hevc", "aac"), "clip_plan: hevc -> transcode(libx264)")

    # -- extract_clip on an H.264 source uses build_clip_args (copy) --
    captured: dict = {}

    async def run_copy_capture(args):
        captured["args"] = list(args)
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP4" * 8)
        return 0

    tc.probe = probe_h264
    rec._run_ffmpeg = run_copy_capture
    eid = await db.insert_event("native.copy", "front", "person", 1, 0.9, start_time, end_time=end_time)
    out = await rec.extract_clip("front", "native.copy", start_time, end_time)
    check(out == rec.clip_path(eid) and "-c" in captured["args"]
          and captured["args"][captured["args"].index("-c") + 1] == "copy"
          and "h264_nvenc" not in captured["args"] and "libx264" not in captured["args"],
          "extract_clip(h264 source) uses the stream-copy argv (build_clip_args)")

    # -- extract_clip on an HEVC source uses build_transcode_args (libx264) --
    tc.probe = probe_hevc
    tc.encoder = enc_libx264
    eid2 = await db.insert_event("native.hevc", "front", "dog", 1, 0.8, start_time, end_time=end_time)
    out2 = await rec.extract_clip("front", "native.hevc", start_time, end_time)
    check(out2 == rec.clip_path(eid2)
          and captured["args"][captured["args"].index("-c:v") + 1] == "libx264"
          and "+faststart" in captured["args"]
          and captured["args"][-2] == "mp4" and captured["args"][-3] == "-f",
          "extract_clip(hevc source) uses the transcode argv (libx264, faststart mp4)")
    check((await db.get_event(eid2))["has_clip"] is True,
          "transcoded clip flips has_clip true after a non-empty file lands")

    # -- nvenc runtime failure -> libx264 retry -> success --
    async def enc_nvenc():
        return NVENC

    tc.encoder = enc_nvenc
    seen: list[str] = []

    async def run_nvenc_then_ok(args):
        vcodec = args[args.index("-c:v") + 1]
        seen.append(vcodec)
        if vcodec == NVENC:
            return 1  # simulate NVENC init failure at runtime
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP4" * 8)
        return 0

    rec._run_ffmpeg = run_nvenc_then_ok
    eid3 = await db.insert_event("native.retry", "front", "cat", 1, 0.7, start_time, end_time=end_time)
    out3 = await rec.extract_clip("front", "native.retry", start_time, end_time)
    check(out3 == rec.clip_path(eid3) and seen == [NVENC, LIBX264],
          "nvenc clip failure -> automatic libx264 retry -> clip lands")
    check(NVENC in tc._failed_encoders,
          "runtime nvenc failure is recorded (excluded for the rest of the process)")

    # -- the same retry path for VAAPI (AMD/Intel iGPU), and the render node
    #    must reach the clip argv or ffmpeg would use the wrong/no GPU --
    tcv = make_transcoder("clipbranch-vaapi", vaapi_device="/dev/dri/renderD128")
    rec._transcode = tcv
    tcv.probe = probe_hevc

    async def enc_vaapi():
        return VAAPI

    tcv.encoder = enc_vaapi
    seen_v: list[str] = []
    saw_node: list[bool] = []

    async def run_vaapi_then_ok(args):
        vcodec = args[args.index("-c:v") + 1]
        seen_v.append(vcodec)
        if vcodec == VAAPI:
            saw_node.append("/dev/dri/renderD128" in args)
            return 1  # simulate a VAAPI init failure (no driver / busy engine)
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP4" * 8)
        return 0

    rec._run_ffmpeg = run_vaapi_then_ok
    eid_v = await db.insert_event(
        "native.vaapi", "front", "cat", 1, 0.7, start_time, end_time=end_time
    )
    out_v = await rec.extract_clip("front", "native.vaapi", start_time, end_time)
    check(out_v == rec.clip_path(eid_v) and seen_v == [VAAPI, LIBX264],
          "vaapi clip failure -> automatic libx264 retry -> clip lands")
    check(saw_node == [True], "the recorder passes the render node into the clip argv")
    check(VAAPI in tcv._failed_encoders,
          "runtime vaapi failure is recorded (excluded for the rest of the process)")

    # -- both encoders fail -> stream-copy fallback so a clip still lands --
    tc2 = make_transcoder("clipbranch-fallback")
    rec._transcode = tc2
    tc2.probe = probe_hevc
    tc2.encoder = enc_libx264
    seen2: list[str] = []

    async def run_transcode_fail_copy_ok(args):
        if "-c:v" in args:  # a transcode attempt
            seen2.append(args[args.index("-c:v") + 1])
            return 1  # transcode fails
        # stream-copy fallback: build_clip_args emits "-c copy"
        seen2.append("copy")
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP4" * 8)
        return 0

    rec._run_ffmpeg = run_transcode_fail_copy_ok
    eid4 = await db.insert_event("native.fallback", "front", "car", 1, 0.6, start_time, end_time=end_time)
    out4 = await rec.extract_clip("front", "native.fallback", start_time, end_time)
    check(out4 == rec.clip_path(eid4) and seen2 == ["libx264", "copy"],
          "transcode failure -> stream-copy fallback still produces a clip (no regression)")

    await db.close()


def clip_branch_checks() -> None:
    print("5. clip_plan + recorder.extract_clip transcode/copy branch")
    asyncio.run(_clip_branch_cases())


# =====================================================================
# 5b. clip extraction concurrency cap
# =====================================================================


async def _clip_concurrency_cases() -> None:
    """schedule_clip must never run more than CLIP_CONCURRENCY extractions at
    once. Without the cap, N simultaneous events fan out to N ffmpeg re-encodes,
    which on a CPU-transcoding box starve detect ingest and push each other past
    the clip timeout — and a timed-out transcode lands an unplayable HEVC clip."""
    cfg = make_config("clipsem")
    db = Database(cfg.data_dir / "sem.db")
    await db.connect()

    rec = Recorder(cfg, db, FakeSettings())
    rec._running = True
    rec.clip_delay_s = 0.0  # the post-event wait is not what we are measuring

    live = 0
    peak = 0
    done = asyncio.Event()
    finished = 0
    total = CLIP_CONCURRENCY + 3

    async def slow_extract(camera, frigate_id, start_time, end_time):
        nonlocal live, peak, finished
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.05)  # hold the slot so overlap is observable
        live -= 1
        finished += 1
        if finished == total:
            done.set()

    rec.extract_clip = slow_extract
    for i in range(total):
        await rec.schedule_clip("front", f"native.sem{i}", 100.0, 110.0)
    await asyncio.wait_for(done.wait(), timeout=10.0)

    check(peak <= CLIP_CONCURRENCY,
          f"schedule_clip caps concurrent extractions at {CLIP_CONCURRENCY} (peak was {peak})")
    check(peak > 1, "the cap still allows real parallelism (not accidentally serialized)")
    check(finished == total, "every queued clip still runs — the cap delays, never drops")

    await db.close()


def clip_concurrency_checks() -> None:
    print("5b. clip extraction concurrency cap")
    asyncio.run(_clip_concurrency_cases())


# =====================================================================
# 6. failure handling + disabled passthrough
# =====================================================================


async def _failure_cases() -> None:
    cam_dir = TMP / "fail" / "front"
    seg = make_seg(cam_dir, datetime(2026, 7, 4, 12, 0, 0))

    # transcode rc!=0 -> segment_for_playback returns None (serve original)
    tf = make_transcoder("fail-transcode")

    async def run_fail(args, timeout=None):
        if args and args[0] == "/fake/ffprobe":
            return 0, b"codec_name=hevc\ncodec_type=video\n", b""
        if args and "-encoders" in args:
            return 0, b"libx264", b""
        return 1, b"", b"boom"  # transcode fails, no output written

    tf._run = run_fail
    check(await tf.segment_for_playback("front", 1, seg) is None,
          "transcode failure -> None (serve original, never a 500)")
    check(not list((TMP / "fail-transcode" / "cache").glob("*.ts")),
          "failed transcode leaves no cached segment behind")

    # a probe timeout (rc None, empty stdout) -> unknown codec -> passthrough
    tnull = make_transcoder("fail-probe")

    async def run_probe_timeout(args, timeout=None):
        return None, b"", b"timeout"

    tnull._run = run_probe_timeout
    check(await tnull.segment_for_playback("front", 1, seg) is None,
          "probe failure -> unknown codec -> passthrough (None)")

    # disabled transcoder (no ffmpeg/ffprobe) passes everything through
    td = Transcoder(cache_dir=TMP / "fail-disabled" / "cache", ffmpeg="", ffprobe="")
    check(not td.enabled, "no binaries -> transcoder disabled")
    check(await td.segment_for_playback("front", 1, seg) is None,
          "disabled -> segment passthrough (None)")
    check(await td.clip_plan("front", seg) == ClipPlan(False, None, None, None),
          "disabled -> clip_plan stream-copy")
    check(await td.probe("front", seg) == ProbeResult(None, None),
          "disabled -> probe returns (None, None) without spawning anything")


def failure_checks() -> None:
    print("6. failure handling — transcode/probe failure passthrough, disabled")
    asyncio.run(_failure_cases())


# =====================================================================
# 7. REAL ffmpeg (optional) — genuine libx264 transcodes
# =====================================================================


def real_ffmpeg_checks() -> None:
    print("7. REAL ffmpeg — genuine libx264 segment + clip transcode (optional)")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        print("  -- skipped (no ffmpeg/ffprobe on PATH; mocked paths cover the logic)")
        return
    asyncio.run(_real_ffmpeg_cases(ffmpeg, ffprobe))


def _probe_vcodec(ffprobe: str, path: Path) -> str:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    ).stdout
    # MPEG-TS can list the video codec per-program; take the first line.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else ""


async def _real_ffmpeg_cases(ffmpeg: str, ffprobe: str) -> None:
    root = TMP / "real"
    root.mkdir(parents=True, exist_ok=True)
    # A genuine HEVC MPEG-TS segment (the exact bug: HEVC .ts the browser can't play).
    src = root / "src.ts"
    gen = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
         "-c:v", "libx265", "-x265-params", "log-level=none", "-f", "mpegts", str(src)],
        capture_output=True, timeout=120,
    )
    if gen.returncode != 0 or not src.is_file():
        print("  -- skipped (this ffmpeg has no libx265 to synthesize an HEVC fixture)")
        return
    check(_probe_vcodec(ffprobe, src) == "hevc", "REAL: synthesized an HEVC source segment")

    # Force libx264 (mac/CI has no NVENC) and run the REAL transcode via ._run.
    t = Transcoder(cache_dir=root / "cache", ffmpeg=ffmpeg, ffprobe=ffprobe)
    t._encoder = LIBX264
    probe = await t.probe("realcam", src)
    check(probe.video_codec == "hevc", "REAL: probe() reports the HEVC source codec")

    out = await t.segment_for_playback("realcam", 12345, src)
    check(out is not None and out.is_file() and out.stat().st_size > 0,
          "REAL: HEVC segment transcoded to a non-empty .ts")
    check(_probe_vcodec(ffprobe, out) == "h264",
          "REAL: transcoded timeline segment is H.264 (browser-playable)")

    # And a REAL clip transcode via build_transcode_args (concat of one segment).
    concat = root / "concat.txt"
    concat.write_text(f"ffconcat version 1.0\nfile '{src}'\n")
    clip = root / "clip.mp4"
    argv = build_transcode_args(
        ffmpeg, LIBX264, container="mp4", output=clip,
        concat_list=concat, seek_s=0.0, duration_s=1.0, audio_codec=None,
    )
    rc, _, err = await t._run(argv, timeout=120)
    check(rc == 0 and clip.is_file() and b"ftyp" in clip.read_bytes()[:64],
          "REAL: HEVC clip transcoded to a faststart H.264 mp4 (ftyp box present)")
    check(_probe_vcodec(ffprobe, clip) == "h264", "REAL: transcoded clip is H.264")


def main() -> None:
    codec_branch_checks()
    arg_builder_checks()
    encoder_checks()
    segment_cache_checks()
    clip_branch_checks()
    clip_concurrency_checks()
    failure_checks()
    real_ffmpeg_checks()
    print(f"\nALL {PASS} CHECKS PASSED (HEVC->H.264 transcode: probe/encoder/cache/clip)")


if __name__ == "__main__":
    main()

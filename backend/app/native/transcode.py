"""Automatic HEVC → H.264 transcoding for browser playback.

The recorder stream-copies the camera MAIN restream into MPEG-TS segments and
concats those into event-clip MP4s (native/recorder.py §5). When a camera's
main stream is **H.265/HEVC** the segments and clips are HEVC too, and browsers
(Chrome/Firefox) cannot decode HEVC via HLS/MSE or ``<video>`` — the timeline
seeks and event clips fail. Live view is unaffected (go2rtc handles it).

This module makes Vigilume serve **H.264** to the browser *only when the source
is not already browser-playable*, transcoding on the GPU (``h264_nvenc``) with a
CPU (``libx264``) fallback. Recordings on disk stay HEVC — the transcode happens
at serve time (timeline segments, cached) or once at clip-extraction time.

Two integration points, both owning their own subprocess execution:

- **Timeline segments** (``routers/recordings.py`` ``seg/{ts}.ts``): the
  ``Transcoder`` probes the camera codec, and for HEVC transcodes the single
  10 s segment to an independent H.264 MPEG-TS segment, cached in a bounded
  on-disk LRU (keyed by camera+ts+codec+encoder) so re-seeks are instant.
  In-flight transcodes of the same key are de-duplicated. H.264 sources are
  served raw (unchanged fast path).
- **Event clips** (``native/recorder.py`` ``extract_clip``): the recorder asks
  the ``Transcoder`` whether to transcode and which encoder to use, then builds
  the argv here and runs it through its own ``_run_ffmpeg`` (the established,
  golden-tested subprocess path) so clip failure/size/rename semantics are
  reused unchanged.

Everything degrades safely: no ffmpeg/ffprobe → passthrough (raw HEVC, as
today); a probe/transcode failure → serve the original + a WARNING (never a
500); a runtime NVENC init failure → fall back to libx264 and log once.

The exact commands (probe + both encoders + both containers) live in the pure,
golden-tested arg builders below, so they can be verified without a real ffmpeg.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger(__name__)

# H.264 encoders, best-first. NVENC is the NVIDIA path (requires an NVIDIA GPU +
# NVIDIA_DRIVER_CAPABILITIES including "video" in the container — set in
# docker-compose.yml); VAAPI is the AMD/Intel iGPU path (requires a DRI render
# node passed into the container + the Mesa VA driver in the image); libx264 is
# the universal CPU fallback.
NVENC = "h264_nvenc"
VAAPI = "h264_vaapi"
LIBX264 = "libx264"

# Encoders that run on fixed-function video silicon. These share one property
# the fallback logic cares about: they can be present in the ffmpeg build and
# still fail at RUNTIME (no driver, no permission on the node, a busy or
# unsupported engine), so a failure re-selects (next GPU encoder, else libx264)
# instead of erroring the transcode.
HW_ENCODERS = frozenset({NVENC, VAAPI})

# Where a VAAPI-capable GPU exposes itself. renderD128 is the first render node
# on any Linux box; a second GPU lands on renderD129. Probed in order, and
# overridable for a box that enumerates differently.
VAAPI_DEVICE_ENV = "VIGILUME_VAAPI_DEVICE"
VAAPI_DEVICE_CANDIDATES = ("/dev/dri/renderD128", "/dev/dri/renderD129")

# Nodes the NVIDIA container runtime injects when it hands a GPU to a container.
# nvidiactl is the control device and exists whenever ANY card was passed in.
NVIDIA_DEVICE_CANDIDATES = ("/dev/nvidiactl", "/dev/nvidia0")

# Video codecs a browser can already play through our HLS/MP4 pipeline, so we
# stream-copy them untouched (no transcode).
_BROWSER_VIDEO_CODECS = frozenset({"h264", "avc", "avc1"})
# Audio we can stream-copy into TS/MP4 as-is; anything else is re-encoded to AAC.
_AAC_AUDIO = frozenset({"aac"})

# Encoder tuning. NVENC: constant-quality VBR (-cq drives quality, -b:v 0 lets
# it float). VAAPI: constant-QP, the only rate-control every Mesa/VA driver
# implements the same way (the quality knob is -qp, matching -cq/-crf at 23).
# libx264: -crf constant quality at a fast preset. 23 is a sane
# "visually lossless-ish" default for review footage.
_NVENC_VIDEO_OPTS = ("-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0")
_VAAPI_VIDEO_OPTS = ("-rc_mode", "CQP", "-qp", "23")
_LIBX264_VIDEO_OPTS = ("-preset", "veryfast", "-crf", "23")
_VIDEO_OPTS = {
    NVENC: _NVENC_VIDEO_OPTS,
    VAAPI: _VAAPI_VIDEO_OPTS,
    LIBX264: _LIBX264_VIDEO_OPTS,
}

# Human-readable encoder names for the one-line selection log.
_ENCODER_LABEL = {
    NVENC: "GPU NVENC",
    VAAPI: "GPU VAAPI",
    LIBX264: "CPU libx264",
}

# Bounded on-disk LRU for transcoded timeline segments (re-seek/re-buffer hits).
DEFAULT_CACHE_BYTES = 512 * 1024**2  # 512 MB
# A camera's codec effectively never changes; cache the probe per camera.
CODEC_TTL_S = 300.0
# A single 10 s segment transcodes in well under this even on CPU; clips are
# run by the recorder under its own (longer) timeout.
SEGMENT_TRANSCODE_TIMEOUT_S = 60.0
_PROBE_TIMEOUT_S = 15.0


def is_browser_playable(codec: Optional[str]) -> bool:
    """True for H.264 (the browser plays it directly); False for HEVC/H.265 and
    every other codec that needs transcoding for HLS/MSE/``<video>``."""
    return (codec or "").lower() in _BROWSER_VIDEO_CODECS


def needs_transcode(codec: Optional[str]) -> bool:
    """True only for a *known* non-browser video codec (hevc/h265/mpeg4/…).

    An unknown/``None`` codec (probe failed / not a real stream) is left alone —
    transcoding garbage would only fail, so we passthrough and serve the raw
    bytes exactly as before.
    """
    c = (codec or "").lower()
    return bool(c) and c not in _BROWSER_VIDEO_CODECS


@dataclass(frozen=True)
class ProbeResult:
    """Video + audio codec names for a media file (either may be ``None``)."""

    video_codec: Optional[str]
    audio_codec: Optional[str]


@dataclass(frozen=True)
class ClipPlan:
    """How ``extract_clip`` should produce a clip for a given source segment."""

    transcode: bool               # transcode to H.264 vs. stream-copy
    encoder: Optional[str]        # NVENC/LIBX264 when transcode, else None
    video_codec: Optional[str]    # source video codec (for logging)
    audio_codec: Optional[str]    # source audio codec (-> copy if aac else aac)


# ---------- pure argv builders (golden-tested; no ffmpeg needed) ----------


def build_probe_args(ffprobe: str, input_path: os.PathLike | str) -> list[str]:
    """ffprobe argv reporting each stream's ``codec_type``/``codec_name`` — one
    call yields both the video and audio codecs (parsed by ``parse_probe_output``)."""
    return [
        ffprobe, "-v", "error",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "default=noprint_wrappers=1",
        str(input_path),
    ]


def parse_probe_output(text: str) -> ProbeResult:
    """Parse ``build_probe_args`` output into the first video + first audio
    codec. Robust to stream ordering and to missing/garbage streams."""
    names: list[str] = []
    types: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("codec_name="):
            names.append(line.split("=", 1)[1].strip())
        elif line.startswith("codec_type="):
            types.append(line.split("=", 1)[1].strip())
    video: Optional[str] = None
    audio: Optional[str] = None
    for name, ctype in zip(names, types):
        low = ctype.lower()
        if low == "video" and video is None and name and name.lower() != "unknown":
            video = name
        elif low == "audio" and audio is None and name and name.lower() != "unknown":
            audio = name
    return ProbeResult(video, audio)


def build_encoders_probe_args(ffmpeg: str) -> list[str]:
    """ffmpeg argv listing compiled-in encoders (grepped for the hw encoders)."""
    return [ffmpeg, "-hide_banner", "-encoders"]


def find_vaapi_device(
    *,
    env: Optional[str] = None,
    exists: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """The DRI render node to hand VAAPI, or ``None`` when there isn't one.

    ``VIGILUME_VAAPI_DEVICE`` wins when set (and present); otherwise the
    standard render nodes are probed in order. ``None`` — the case on a box with
    no iGPU, or one where ``/dev/dri`` was never passed into the container — is
    normal and simply means VAAPI is not a candidate.

    ``env``/``exists`` are injectable so the selection logic stays testable
    without a real render node.
    """
    check = exists if exists is not None else os.path.exists
    override = env if env is not None else os.environ.get(VAAPI_DEVICE_ENV, "")
    override = (override or "").strip()
    if override:
        # An explicit override that is not there is an operator mistake worth
        # surfacing, not a silent downgrade to CPU.
        if check(override):
            return override
        log.warning(
            "transcode: %s=%s does not exist — ignoring and probing the "
            "standard render nodes", VAAPI_DEVICE_ENV, override,
        )
    for node in VAAPI_DEVICE_CANDIDATES:
        if check(node):
            return node
    return None


def find_nvidia_device(exists: Optional[Callable[[str], bool]] = None) -> bool:
    """Whether an NVIDIA GPU was actually passed into this container.

    Needed because ``ffmpeg -encoders`` is NOT evidence of hardware: distro
    ffmpeg builds (Debian's and Ubuntu's both) list ``h264_nvenc`` and
    ``h264_vaapi`` unconditionally, since the encoders load their drivers at
    runtime. Selecting on the listing alone therefore picks NVENC on an AMD box,
    fails at init, and — because a hardware failure downgrades to CPU — lands on
    libx264 while a perfectly good iGPU sits idle. Checking for the device node
    is what makes the choice reflect the actual box.
    """
    check = exists if exists is not None else os.path.exists
    return any(check(node) for node in NVIDIA_DEVICE_CANDIDATES)


def select_encoder(
    encoders_listing: str,
    vaapi_device: Optional[str] = None,
    nvidia_present: bool = False,
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Pick the best available H.264 encoder from an ``ffmpeg -encoders`` dump.

    Order is NVENC → VAAPI → libx264. NVENC first because a box with a discrete
    NVIDIA card is the one configuration where the dGPU beats an iGPU outright;
    VAAPI next because fixed-function AMD/Intel encoding still costs a fraction
    of libx264; libx264 last because it always works.

    Each hardware encoder needs BOTH a listing entry and its device — see
    ``find_nvidia_device`` for why the listing alone is not enough. ``exclude``
    drops encoders that already failed at runtime, so a box whose device probe
    was wrong still walks down to the next real option instead of giving up on
    hardware entirely.
    """
    if NVENC not in exclude and nvidia_present and "h264_nvenc" in encoders_listing:
        return NVENC
    if VAAPI not in exclude and vaapi_device and "h264_vaapi" in encoders_listing:
        return VAAPI
    return LIBX264


def build_transcode_args(
    ffmpeg: str,
    encoder: str,
    *,
    container: str,                                   # "mpegts" | "mp4"
    output: os.PathLike | str,                        # path or "pipe:1"
    input_path: Optional[os.PathLike | str] = None,   # single-file (segment)
    concat_list: Optional[os.PathLike | str] = None,  # concat demuxer (clip)
    seek_s: Optional[float] = None,                   # output-side -ss (clip cut)
    duration_s: Optional[float] = None,               # -t (clip cut)
    audio_codec: Optional[str] = None,                # source audio -> copy if aac else aac
    vaapi_device: Optional[str] = None,               # DRI render node (VAAPI only)
) -> list[str]:
    """ffmpeg argv to transcode a source to H.264 in ``container``.

    - ``encoder`` NVENC → full-GPU pipeline ``-hwaccel cuda
      -hwaccel_output_format cuda`` (NVDEC decodes HEVC, frames stay in GPU
      memory, ``h264_nvenc`` encodes) — the robust, fastest HEVC→H.264 path.
      VAAPI → the AMD/Intel equivalent, ``-hwaccel vaapi -hwaccel_device <node>
      -hwaccel_output_format vaapi`` (the iGPU's decode block reads HEVC, the
      surfaces never leave GPU memory, ``h264_vaapi`` encodes).
      LIBX264 → plain CPU decode + encode.
    - Segment case: ``input_path`` + ``container="mpegts"`` → an independent
      H.264 MPEG-TS segment (each source .ts is keyframe-started, so a
      per-segment transcode is a valid standalone HLS segment).
    - Clip case: ``concat_list`` + ``seek_s``/``duration_s`` +
      ``container="mp4"`` → a faststart H.264 MP4 cut, same window math as the
      stream-copy path.
    - Audio: ``-c:a copy`` when already AAC, else re-encode to ``aac``
      (TS- and MP4-legal).
    """
    args = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
    if encoder == NVENC:
        # Decode on the GPU and keep frames there for the NVENC encoder — must
        # precede the input it applies to.
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif encoder == VAAPI:
        # Same idea on AMD/Intel: the render node names WHICH GPU, and
        # -hwaccel_output_format vaapi keeps decoded frames as VA surfaces so
        # h264_vaapi encodes them in place (no GPU->CPU->GPU round trip).
        args += ["-hwaccel", "vaapi"]
        if vaapi_device:
            args += ["-hwaccel_device", str(vaapi_device)]
        args += ["-hwaccel_output_format", "vaapi"]
    if concat_list is not None:
        args += ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
    else:
        args += ["-i", str(input_path)]
    # Output-side seek/duration (after -i) cut precisely on the transcode.
    if seek_s is not None:
        args += ["-ss", f"{max(0.0, seek_s):.3f}"]
    if duration_s is not None:
        args += ["-t", f"{max(0.0, duration_s):.3f}"]
    if encoder == VAAPI:
        # Guard against a decoder that handed back SOFTWARE frames (a codec the
        # iGPU can't decode, so ffmpeg silently used the CPU decoder). h264_vaapi
        # only accepts VA surfaces, and without this it would abort. The filter
        # is a no-op when frames are already on the GPU, and uploads them when
        # they are not — so an HEVC main transcodes fully on-GPU while an exotic
        # source still encodes on the GPU instead of failing outright.
        args += ["-vf", "format=nv12|vaapi,hwupload"]
    args += ["-c:v", encoder]
    args += list(_VIDEO_OPTS.get(encoder, _LIBX264_VIDEO_OPTS))
    if (audio_codec or "").lower() in _AAC_AUDIO:
        args += ["-c:a", "copy"]
    else:
        args += ["-c:a", "aac"]
    if container == "mp4":
        args += ["-movflags", "+faststart"]
    args += ["-f", container, str(output)]
    return args


# ---------- Transcoder: probe cache + encoder select + segment LRU ----------


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


class Transcoder:
    """Codec probe (cached per camera), H.264 encoder selection (cached, with a
    one-time hardware→libx264 runtime downgrade), a bounded on-disk LRU of
    transcoded timeline segments with in-flight de-duplication, and the clip
    transcode plan the recorder consumes.

    ``ffmpeg``/``ffprobe`` default to ``shutil.which`` lookups and
    ``vaapi_device`` to a render-node probe; tests inject explicit values and
    monkeypatch ``_run`` to avoid a real ffmpeg.
    """

    def __init__(
        self,
        *,
        cache_dir: os.PathLike | str,
        ffmpeg: Optional[str] = None,
        ffprobe: Optional[str] = None,
        cache_max_bytes: int = DEFAULT_CACHE_BYTES,
        codec_ttl_s: float = CODEC_TTL_S,
        segment_timeout_s: float = SEGMENT_TRANSCODE_TIMEOUT_S,
        vaapi_device: Optional[str] = None,
        nvidia_present: Optional[bool] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._ffprobe = ffprobe if ffprobe is not None else shutil.which("ffprobe")
        self._cache_max_bytes = cache_max_bytes
        self._codec_ttl_s = codec_ttl_s
        self._segment_timeout_s = segment_timeout_s
        # Resolved ONCE at construction: a render node does not appear or vanish
        # over a container's lifetime, and probing per transcode would stat the
        # filesystem on every timeline seek.
        self._vaapi_device = vaapi_device if vaapi_device is not None else find_vaapi_device()
        self._nvidia_present = (
            nvidia_present if nvidia_present is not None else find_nvidia_device()
        )
        self._codec_cache: dict[str, tuple[ProbeResult, float]] = {}
        self._encoder: Optional[str] = None
        self._encoder_lock = asyncio.Lock()
        # The ``ffmpeg -encoders`` dump, kept so a runtime failure can re-select
        # the NEXT candidate without re-probing.
        self._encoders_listing: Optional[str] = None
        # Hardware encoders that failed at runtime; never retried this process.
        self._failed_encoders: set[str] = set()
        self._inflight: dict[str, asyncio.Future] = {}
        self._codec_inflight: dict[str, asyncio.Future] = {}
        # Completed transcodes per encoder: {encoder: [ok, failed]}. The point
        # is the gap between SELECTED and WORKING — "we picked h264_vaapi" and
        # "the iGPU actually produced frames" are different claims, and only the
        # second one answers "is my GPU doing anything". Counted here rather
        # than inferred from logs so the status endpoint can just say it.
        self._runs: dict[str, list[int]] = {}

    @property
    def enabled(self) -> bool:
        """Transcoding is possible only with both binaries present; otherwise
        every path passes through (raw HEVC served, exactly as before)."""
        return bool(self._ffmpeg and self._ffprobe)

    @property
    def ffmpeg(self) -> Optional[str]:
        return self._ffmpeg

    @property
    def vaapi_device(self) -> Optional[str]:
        """The DRI render node VAAPI transcodes run against (``None`` when
        there is no iGPU passed into the container). The recorder passes this
        into ``build_transcode_args`` for its own clip transcodes."""
        return self._vaapi_device

    def invalidate(self, camera: str) -> None:
        """Drop a cached codec probe (e.g. after a camera's encode config changed)."""
        self._codec_cache.pop(camera, None)

    # ---- subprocess (single mock point) ----

    async def _run(
        self, args: list[str], timeout: Optional[float] = None
    ) -> tuple[Optional[int], bytes, bytes]:
        """Run an ffmpeg/ffprobe child, returning (returncode, stdout, stderr).
        Never raises for a normal failure/timeout — returns rc ``None`` instead
        so callers fall back gracefully. Tests monkeypatch this."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError):
            log.warning("transcode: could not spawn %s", args[0], exc_info=True)
            return None, b"", b""
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return None, b"", b"timeout"
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
        return proc.returncode, out or b"", err or b""

    # ---- codec probe (cached per camera, TTL) ----

    async def probe(self, camera: str, sample_path: os.PathLike | str) -> ProbeResult:
        """The camera's video/audio codec, probed from one of its segment/clip
        files and cached per camera. ``ProbeResult(None, None)`` when disabled
        or the probe yields nothing (treated as passthrough by callers).
        Concurrent cache misses for the same camera share one in-flight ffprobe
        (no thundering herd when a timeline opens many segments at once)."""
        if not self.enabled:
            return ProbeResult(None, None)
        now = time.monotonic()
        hit = self._codec_cache.get(camera)
        if hit is not None and now - hit[1] < self._codec_ttl_s:
            return hit[0]
        fut = self._codec_inflight.get(camera)
        if fut is None:
            fut = asyncio.ensure_future(self._probe_uncached(camera, sample_path))
            self._codec_inflight[camera] = fut
            fut.add_done_callback(lambda _f, c=camera: self._codec_inflight.pop(c, None))
        # shield so a disconnecting client can't cancel the shared probe.
        return await asyncio.shield(fut)

    async def _probe_uncached(self, camera: str, sample_path: os.PathLike | str) -> ProbeResult:
        now = time.monotonic()
        rc, out, _ = await self._run(
            build_probe_args(self._ffprobe, sample_path), timeout=_PROBE_TIMEOUT_S
        )
        result = parse_probe_output(out.decode("utf-8", "replace")) if out else ProbeResult(None, None)
        self._codec_cache[camera] = (result, now)
        return result

    # ---- encoder selection (cached; one-time hardware->libx264 downgrade) ----

    async def encoder(self) -> str:
        """The H.264 encoder to use, detected once and cached."""
        if self._encoder is not None:
            return self._encoder
        async with self._encoder_lock:
            if self._encoder is None:
                self._encoder = await self._detect_encoder()
            return self._encoder

    async def _detect_encoder(self) -> str:
        if not self._ffmpeg:
            return LIBX264
        if self._encoders_listing is None:
            rc, out, _ = await self._run(
                build_encoders_probe_args(self._ffmpeg), timeout=_PROBE_TIMEOUT_S
            )
            self._encoders_listing = out.decode("utf-8", "replace") if out else ""
        enc = select_encoder(
            self._encoders_listing,
            self._vaapi_device,
            self._nvidia_present,
            frozenset(self._failed_encoders),
        )
        if enc == VAAPI:
            log.info(
                "transcode: selected H.264 encoder %s (%s on %s)",
                enc, _ENCODER_LABEL[enc], self._vaapi_device,
            )
        else:
            log.info(
                "transcode: selected H.264 encoder %s (%s)",
                enc, _ENCODER_LABEL.get(enc, enc),
            )
        return enc

    def mark_hw_failed(self, encoder: str = NVENC) -> str:
        """Runtime hardware-encoder init/encode failure → never use ``encoder``
        again this process. Returns the encoder to use INSTEAD, so the caller
        can retry the same job without re-deriving it.

        Permanent rather than retried: the causes (no driver in the image, no
        device passed through, a GPU that does not implement the encode profile)
        do not heal while the container runs, and retrying the hardware path on
        every clip would double the latency of each one.

        Re-selecting rather than jumping straight to libx264 matters on a box
        with two GPUs, or one where the device probe guessed wrong — the next
        hardware encoder still gets its chance before we concede to the CPU.
        """
        first_time = encoder not in self._failed_encoders
        self._failed_encoders.add(encoder)
        nxt = select_encoder(
            self._encoders_listing or "",
            self._vaapi_device,
            self._nvidia_present,
            frozenset(self._failed_encoders),
        )
        self._encoder = nxt
        if first_time:
            log.warning(
                "transcode: %s failed at runtime — using %s for all further "
                "transcodes (logged once per encoder)", encoder, nxt,
            )
        return nxt

    def mark_nvenc_failed(self) -> str:
        """Back-compat alias for ``mark_hw_failed(NVENC)``."""
        return self.mark_hw_failed(NVENC)

    # ---- timeline segment path (probe -> cache -> transcode, deduped) ----

    async def clip_plan(self, camera: str, sample_segment: os.PathLike | str) -> ClipPlan:
        """Decide how ``extract_clip`` should render a clip: stream-copy for
        H.264 (or when disabled/unknown), else transcode with the chosen
        encoder."""
        if not self.enabled:
            return ClipPlan(False, None, None, None)
        probe = await self.probe(camera, sample_segment)
        if not needs_transcode(probe.video_codec):
            return ClipPlan(False, None, probe.video_codec, probe.audio_codec)
        return ClipPlan(True, await self.encoder(), probe.video_codec, probe.audio_codec)

    async def segment_for_playback(
        self, camera: str, ts: int, source_path: os.PathLike | str
    ) -> Optional[Path]:
        """H.264 segment for the browser, or ``None`` to serve the raw source.

        Returns ``None`` (passthrough) when disabled, when the source is already
        H.264 (or unknown), or when transcoding fails. Otherwise returns a cached
        transcoded ``.ts`` path (transcoding on a cache miss, de-duplicated with
        any concurrent request for the same segment)."""
        if not self.enabled:
            return None
        probe = await self.probe(camera, source_path)
        if not needs_transcode(probe.video_codec):
            return None
        encoder = await self.encoder()
        key = self._segment_key(camera, ts, probe.video_codec, encoder)
        cached = self._cache_path(key)
        if cached.exists():
            self._touch(cached)
            return cached
        return await self._get_or_transcode_segment(
            key,
            lambda: self._transcode_segment(camera, ts, source_path, probe, encoder, cached),
        )

    async def _get_or_transcode_segment(
        self, key: str, factory: Callable[[], Awaitable[Optional[Path]]]
    ) -> Optional[Path]:
        """Share a single in-flight transcode across concurrent same-key
        requests (no N ffmpegs for one segment)."""
        fut = self._inflight.get(key)
        if fut is None:
            fut = asyncio.ensure_future(factory())
            self._inflight[key] = fut
            fut.add_done_callback(lambda _f, k=key: self._inflight.pop(k, None))
        # shield so a disconnecting client can't cancel the shared transcode.
        return await asyncio.shield(fut)

    async def _transcode_segment(
        self,
        camera: str,
        ts: int,
        source_path: os.PathLike | str,
        probe: ProbeResult,
        encoder: str,
        cached: Path,
    ) -> Optional[Path]:
        with contextlib.suppress(OSError):
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        # temp name has no .ts extension so LRU (globs *.ts) never evicts it.
        tmp = self._cache_dir / f".{cached.stem}.{uuid.uuid4().hex}.part"
        ok = await self._run_h264(
            f"segment ts={int(ts)}", camera, encoder, probe.video_codec,
            lambda enc: build_transcode_args(
                self._ffmpeg, enc, container="mpegts",
                output=tmp, input_path=source_path, audio_codec=probe.audio_codec,
                vaapi_device=self._vaapi_device,
            ),
            timeout=self._segment_timeout_s,
        )
        if not ok or not _nonempty(tmp):
            _unlink(tmp)
            log.warning(
                "transcode: segment FAILED camera=%s ts=%d — serving original "
                "(browser may not play HEVC)",
                camera, int(ts),
            )
            return None
        with contextlib.suppress(OSError):
            os.replace(tmp, cached)
        self._evict_lru()
        return cached if cached.exists() else None

    async def _run_h264(
        self,
        what: str,
        camera: str,
        encoder: str,
        source_codec: Optional[str],
        args_for: Callable[[str], list[str]],
        *,
        timeout: Optional[float],
    ) -> bool:
        """Run a transcode with the hardware→libx264 runtime fallback. Returns
        whether an ffmpeg exited 0. (Segment path only — the recorder runs clip
        transcodes through its own ``_run_ffmpeg``.)"""
        log.info(
            "transcode: %s %s camera=%s (%s->h264)",
            encoder, what, camera, source_codec or "?",
        )
        rc, _, err = await self._run(args_for(encoder), timeout=timeout)
        self._note_run(encoder, rc == 0)
        if rc == 0:
            return True
        if encoder in HW_ENCODERS:
            # Retry on whatever selection survives the failure — the next
            # hardware encoder on a two-GPU box, otherwise libx264. One retry
            # per call keeps a failing segment bounded; a second bad encoder is
            # excluded by the time the next segment is served.
            retry = self.mark_hw_failed(encoder)
            log.info(
                "transcode: %s %s camera=%s (%s retry)", retry, what, camera, encoder,
            )
            rc, _, err = await self._run(args_for(retry), timeout=timeout)
            self._note_run(retry, rc == 0)
            if rc == 0:
                return True
        if err:
            log.warning(
                "transcode: ffmpeg stderr (%s camera=%s): %s",
                what, camera, err.decode("utf-8", "replace").strip()[-300:],
            )
        return False

    def _note_run(self, encoder: str, ok: bool) -> None:
        tally = self._runs.setdefault(encoder, [0, 0])
        tally[0 if ok else 1] += 1

    async def status(self) -> dict[str, Any]:
        """What is actually encoding video, for GET /api/system/detector.

        Answers the question an operator with an iGPU actually has — "is my GPU
        doing anything?" — which log-reading answers badly and nothing else
        answered at all.

        It AWAITS the encoder selection rather than reporting "not probed yet".
        The probe is a single `ffmpeg -encoders` run, cached for the life of the
        process, and it is the same one the first timeline seek would trigger —
        so asking here costs one probe and gives a real answer instead of a
        shrug.

        `hardware` is the headline. `runs` is the evidence behind it: an encoder
        can be SELECTED and still never have produced a frame, and one that was
        selected and then failed at runtime shows up in `failed` — which is the
        case that would otherwise silently drop a box back to CPU with nobody
        the wiser.
        """
        if not self.enabled:
            return {
                "enabled": False,
                "encoder": None,
                "encoder_label": "ffmpeg not available",
                "hardware": False,
                "vaapi_device": self._vaapi_device,
                "nvidia": self._nvidia_present,
                "failed": [],
                "runs": {},
            }
        enc = await self.encoder()
        return {
            "enabled": True,
            "encoder": enc,
            "encoder_label": _ENCODER_LABEL.get(enc, enc),
            "hardware": enc in HW_ENCODERS,
            # None here means no DRI render node is visible INSIDE the
            # container — on an AMD/Intel box that is nearly always a missing
            # VAAPI_DEVICE in .env rather than a missing GPU.
            "vaapi_device": self._vaapi_device,
            "nvidia": self._nvidia_present,
            "failed": sorted(self._failed_encoders),
            "runs": {k: {"ok": v[0], "failed": v[1]} for k, v in self._runs.items()},
        }

    # ---- on-disk LRU helpers ----

    def _segment_key(self, camera: str, ts: int, vcodec: Optional[str], encoder: str) -> str:
        safe_cam = camera.replace("/", "_").replace("\\", "_")
        return f"{safe_cam}__{int(ts)}__{(vcodec or 'src')}__{encoder}"

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.ts"

    def _touch(self, path: Path) -> None:
        now = time.time()
        with contextlib.suppress(OSError):
            os.utime(path, (now, now))

    def _evict_lru(self) -> None:
        """Evict oldest transcoded segments until under the byte cap. Only
        touches finished ``*.ts`` cache files (temp parts are dot-prefixed and
        extensionless)."""
        try:
            entries: list[tuple[float, int, Path]] = []
            total = 0
            for p in self._cache_dir.glob("*.ts"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((st.st_mtime, st.st_size, p))
                total += st.st_size
            if total <= self._cache_max_bytes:
                return
            entries.sort(key=lambda e: e[0])  # oldest first
            for _mtime, size, p in entries:
                if total <= self._cache_max_bytes:
                    break
                with contextlib.suppress(OSError):
                    p.unlink()
                    total -= size
        except OSError:
            pass

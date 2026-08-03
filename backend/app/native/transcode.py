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
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)

# H.264 encoders, best-first. NVENC is the GPU path (requires an NVIDIA GPU +
# NVIDIA_DRIVER_CAPABILITIES including "video" in the container — set in
# docker-compose.yml); libx264 is the universal CPU fallback.
NVENC = "h264_nvenc"
LIBX264 = "libx264"

# Video codecs a browser can already play through our HLS/MP4 pipeline, so we
# stream-copy them untouched (no transcode).
_BROWSER_VIDEO_CODECS = frozenset({"h264", "avc", "avc1"})
# Audio we can stream-copy into TS/MP4 as-is; anything else is re-encoded to AAC.
_AAC_AUDIO = frozenset({"aac"})

# Encoder tuning. NVENC: constant-quality VBR (-cq drives quality, -b:v 0 lets
# it float). libx264: -crf constant quality at a fast preset. 23 is a sane
# "visually lossless-ish" default for review footage.
_NVENC_VIDEO_OPTS = ("-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0")
_LIBX264_VIDEO_OPTS = ("-preset", "veryfast", "-crf", "23")

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
    """ffmpeg argv listing compiled-in encoders (grepped for ``h264_nvenc``)."""
    return [ffmpeg, "-hide_banner", "-encoders"]


def select_encoder(encoders_listing: str) -> str:
    """Pick ``h264_nvenc`` when ffmpeg exposes it, else ``libx264``."""
    return NVENC if "h264_nvenc" in encoders_listing else LIBX264


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
) -> list[str]:
    """ffmpeg argv to transcode a source to H.264 in ``container``.

    - ``encoder`` NVENC → full-GPU pipeline ``-hwaccel cuda
      -hwaccel_output_format cuda`` (NVDEC decodes HEVC, frames stay in GPU
      memory, ``h264_nvenc`` encodes) — the robust, fastest HEVC→H.264 path.
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
    if concat_list is not None:
        args += ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
    else:
        args += ["-i", str(input_path)]
    # Output-side seek/duration (after -i) cut precisely on the transcode.
    if seek_s is not None:
        args += ["-ss", f"{max(0.0, seek_s):.3f}"]
    if duration_s is not None:
        args += ["-t", f"{max(0.0, duration_s):.3f}"]
    args += ["-c:v", encoder]
    args += list(_NVENC_VIDEO_OPTS if encoder == NVENC else _LIBX264_VIDEO_OPTS)
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
    one-time NVENC→libx264 runtime downgrade), a bounded on-disk LRU of
    transcoded timeline segments with in-flight de-duplication, and the clip
    transcode plan the recorder consumes.

    ``ffmpeg``/``ffprobe`` default to ``shutil.which`` lookups; tests inject
    explicit paths and monkeypatch ``_run`` to avoid a real ffmpeg.
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
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._ffprobe = ffprobe if ffprobe is not None else shutil.which("ffprobe")
        self._cache_max_bytes = cache_max_bytes
        self._codec_ttl_s = codec_ttl_s
        self._segment_timeout_s = segment_timeout_s
        self._codec_cache: dict[str, tuple[ProbeResult, float]] = {}
        self._encoder: Optional[str] = None
        self._encoder_lock = asyncio.Lock()
        self._nvenc_failed = False
        self._inflight: dict[str, asyncio.Future] = {}
        self._codec_inflight: dict[str, asyncio.Future] = {}

    @property
    def enabled(self) -> bool:
        """Transcoding is possible only with both binaries present; otherwise
        every path passes through (raw HEVC served, exactly as before)."""
        return bool(self._ffmpeg and self._ffprobe)

    @property
    def ffmpeg(self) -> Optional[str]:
        return self._ffmpeg

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

    # ---- encoder selection (cached; one-time NVENC->libx264 downgrade) ----

    async def encoder(self) -> str:
        """The H.264 encoder to use, detected once and cached. Returns libx264
        immediately once NVENC has failed at runtime."""
        if self._nvenc_failed:
            return LIBX264
        if self._encoder is not None:
            return self._encoder
        async with self._encoder_lock:
            if self._encoder is None:
                self._encoder = await self._detect_encoder()
            return self._encoder

    async def _detect_encoder(self) -> str:
        if not self._ffmpeg:
            return LIBX264
        rc, out, _ = await self._run(
            build_encoders_probe_args(self._ffmpeg), timeout=_PROBE_TIMEOUT_S
        )
        enc = select_encoder(out.decode("utf-8", "replace")) if out else LIBX264
        log.info(
            "transcode: selected H.264 encoder %s (%s)",
            enc, "GPU NVENC" if enc == NVENC else "CPU libx264",
        )
        return enc

    def mark_nvenc_failed(self) -> None:
        """Runtime NVENC init/encode failure → downgrade to libx264 (log once)."""
        if not self._nvenc_failed:
            self._nvenc_failed = True
            self._encoder = LIBX264
            log.warning(
                "transcode: h264_nvenc failed at runtime — falling back to CPU "
                "libx264 for all further transcodes (logged once)"
            )

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
        """Run a transcode with the NVENC→libx264 runtime fallback. Returns
        whether an ffmpeg exited 0. (Segment path only — the recorder runs clip
        transcodes through its own ``_run_ffmpeg``.)"""
        log.info(
            "transcode: %s %s camera=%s (%s->h264)",
            encoder, what, camera, source_codec or "?",
        )
        rc, _, err = await self._run(args_for(encoder), timeout=timeout)
        if rc == 0:
            return True
        if encoder == NVENC:
            self.mark_nvenc_failed()
            log.info("transcode: libx264 %s camera=%s (nvenc retry)", what, camera)
            rc, _, err = await self._run(args_for(LIBX264), timeout=timeout)
            if rc == 0:
                return True
        if err:
            log.warning(
                "transcode: ffmpeg stderr (%s camera=%s): %s",
                what, camera, err.decode("utf-8", "replace").strip()[-300:],
            )
        return False

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

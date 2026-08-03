"""Recordings index + HLS VOD playback API (timeline foundation).

The 24/7 recorder writes 10 s MPEG-TS segments at
``<recordings_dir>/{camera}/{YYYY-MM-DD}/{HH}/{MM.SS}.ts`` (local time). These
routes expose that tree for a scrubbable timeline:

- ``GET /api/recordings/cameras``               — which cameras have footage + bounds
- ``GET /api/recordings/{camera}/index?date=``  — one local day's segments + merged coverage ranges
- ``GET /api/recordings/{camera}/playlist.m3u8`` — HLS VOD playlist over [start,end]
- ``GET /api/recordings/{camera}/seg/{ts}.ts``   — one segment file (Range-capable)

Media-scope auth (accept ``?token=``) like the other media routes, so the
browser/OS can fetch playlist + segments without headers. Every filesystem
path is resolved strictly inside the recorder's ``recordings_dir`` — the
``{camera}`` segment must be a direct child dir (no traversal), and
``{ts}`` is an int that must map to a real segment file.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from ..auth import require_admin, require_stream_auth
from ..native.recorder import (
    SEGMENT_SECONDS,
    hour_dir,
    parse_segment_start,
    select_segments,
)

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/recordings",
    tags=["recordings"],
    dependencies=[Depends(require_stream_auth)],
)

# Cap the playlist window server-side so a bogus/huge range can't build a
# multi-thousand-line playlist (6 h @ 10 s = 2160 segments).
MAX_PLAYLIST_WINDOW_S = 6 * 3600


# ADMIN-ONLY. Placed at the router root (DELETE /api/recordings). The
# router-level require_stream_auth still runs, but require_admin is strictly
# stronger — it rejects media-scope (?token=) tokens and non-admins, so a
# push-notification image token can never trigger a purge.
@router.delete("", status_code=200, dependencies=[Depends(require_admin)])
async def delete_all_recordings(request: Request) -> dict[str, Any]:
    """ADMIN ONLY. Permanently delete ALL continuous timeline footage for every
    camera. Event clips, snapshots and the events log are NOT touched.
    Recording resumes immediately for record-enabled cameras. Irreversible."""
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder unavailable")
    try:
        result = await recorder.purge_all_recordings()
    except RuntimeError as exc:  # a purge is already running
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001 — recorders are resumed in purge's finally; never a bare 500
        log.warning("recordings purge failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Purge failed") from None
    return {"purged": True, **result}

# Cap the timeline range-export window server-side (30 min). A single export is
# an ffmpeg concat + (possibly) an HEVC→H.264 transcode, so an unbounded window
# would let one request run ffmpeg for an unbounded time / produce a huge file.
# Over-cap => 400 with a clear message.
EXPORT_MAX_SECONDS = 30 * 60

# Filesystem/header-safe filename parts (mirrors routers/events.py): collapse
# any run of non [A-Za-z0-9._-] to one underscore, then trim separators.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename_part(value: str, fallback: str) -> str:
    cleaned = _FILENAME_UNSAFE.sub("_", value).strip("._")
    return cleaned or fallback


def _export_filename(camera: str, start: float, end: float) -> str:
    """``<camera>_<start-date>_<HH-MM-SS>-<HH-MM-SS>.mp4`` (local time), each
    part sanitized to ASCII filename-safe characters."""
    cam = _sanitize_filename_part(camera, "camera")
    start_lt = time.localtime(start)
    end_lt = time.localtime(end)
    day = time.strftime("%Y-%m-%d", start_lt)
    start_hms = time.strftime("%H-%M-%S", start_lt)
    end_hms = time.strftime("%H-%M-%S", end_lt)
    return f"{cam}_{day}_{start_hms}-{end_hms}.mp4"


# ---------- path helpers (traversal-safe) ----------


def _recordings_dir(request: Request) -> Path:
    return request.app.state.config.recordings_dir


def _camera_dir(request: Request, camera: str) -> Path:
    """Resolve the camera's recording dir, refusing anything that isn't a
    direct child of recordings_dir (blocks ``..`` / absolute / nested paths)."""
    root = _recordings_dir(request).resolve()
    cam_dir = (root / camera).resolve()
    if cam_dir.parent != root:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return cam_dir


def _is_day_dir(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _scan_bounds(camera_dir: Path) -> tuple[Optional[int], Optional[int]]:
    """(earliest segment start, latest segment end), or (None, None) when there
    is no footage. ``latest`` extends the newest segment's start by one segment
    length to reflect its coverage.

    Cheap boundary scan, not a full-tree walk: day dir names (YYYY-MM-DD) and
    hour dir names (HH) sort lexicographically == chronologically, so only the
    first/last non-empty day+hour dirs are read (falling through empty or
    invalid ones)."""
    if not camera_dir.is_dir():
        return None, None
    day_dirs = sorted(
        (d for d in camera_dir.iterdir() if d.is_dir() and _is_day_dir(d.name)),
        key=lambda d: d.name,
    )
    if not day_dirs:
        return None, None

    def _bound_in_day(day_dir: Path, pick, reverse: bool) -> Optional[float]:
        hour_dirs = sorted(
            (h for h in day_dir.iterdir() if h.is_dir()),
            key=lambda h: h.name, reverse=reverse,
        )
        for hd in hour_dirs:
            times = [t for t in (parse_segment_start(s) for s in hd.glob("*.ts")) if t is not None]
            if times:
                return pick(times)
        return None

    earliest: Optional[float] = None
    for d in day_dirs:
        earliest = _bound_in_day(d, min, reverse=False)
        if earliest is not None:
            break
    latest: Optional[float] = None
    for d in reversed(day_dirs):
        latest = _bound_in_day(d, max, reverse=True)
        if latest is not None:
            break
    if earliest is None or latest is None:
        return None, None
    return int(earliest), int(latest) + SEGMENT_SECONDS


def _merge_ranges(segments: list[dict[str, int]]) -> list[dict[str, int]]:
    """Merge ascending segments into contiguous coverage ranges; a gap larger
    than one segment length (SEGMENT_SECONDS) starts a new range."""
    ranges: list[dict[str, int]] = []
    for seg in segments:
        start = seg["start"]
        end = start + seg["duration"]
        if ranges and start - ranges[-1]["end"] <= SEGMENT_SECONDS:
            ranges[-1]["end"] = max(ranges[-1]["end"], end)
        else:
            ranges.append({"start": start, "end": end})
    return ranges


# ---------- routes ----------


@router.get("/cameras")
async def recordings_cameras(request: Request) -> list[dict[str, Any]]:
    """[{camera, friendly_name, has_recordings, earliest, latest}] — one entry
    per known camera (earliest/latest are epoch seconds, null when empty)."""
    cams = await request.app.state.db.list_cameras()
    root = _recordings_dir(request)
    out: list[dict[str, Any]] = []
    for cam in cams:
        name = cam["name"]
        earliest, latest = await asyncio.to_thread(_scan_bounds, root / name)
        out.append(
            {
                "camera": name,
                "friendly_name": cam.get("friendly_name") or name,
                "has_recordings": earliest is not None,
                "earliest": earliest,
                "latest": latest,
            }
        )
    return out


@router.get("/{camera}/index")
async def recordings_index(
    camera: str,
    request: Request,
    date_str: str = Query(..., alias="date", description="local day, YYYY-MM-DD"),
) -> dict[str, Any]:
    """One local day's segments + merged coverage ranges. Tolerates a missing
    day (empty segments/ranges)."""
    cam_dir = _camera_dir(request, camera)
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    day_start = day.timestamp()          # local midnight -> epoch
    day_end = day_start + 86400.0
    tz_offset = time.localtime(day_start).tm_gmtoff or 0
    raw = await asyncio.to_thread(select_segments, cam_dir, day_start, day_end)
    segments = [
        {"start": int(start), "duration": SEGMENT_SECONDS}
        for start, _ in raw
        if day_start <= start < day_end
    ]
    return {
        "date": date_str,
        "tz_offset": tz_offset,
        "segments": segments,
        "ranges": _merge_ranges(segments),
    }


@router.get("/{camera}/playlist.m3u8")
async def recordings_playlist(
    camera: str,
    request: Request,
    start: float = Query(..., description="window start (epoch seconds)"),
    end: float = Query(..., description="window end (epoch seconds)"),
    token: Optional[str] = Query(default=None),
) -> PlainTextResponse:
    """Valid HLS VOD playlist listing the segments intersecting [start, end],
    each pointing at the seg route (relative URL, ?token= carried through).
    The window is capped server-side to bound the playlist size."""
    cam_dir = _camera_dir(request, camera)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    end = min(end, start + MAX_PLAYLIST_WINDOW_S)
    raw = await asyncio.to_thread(select_segments, cam_dir, start, end)
    suffix = f"?token={token}" if token else ""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-TARGETDURATION:{SEGMENT_SECONDS}",
    ]
    for idx, (seg_start, _) in enumerate(raw):
        # Every segment is timestamp-INDEPENDENT: the recorder muxes each 10 s
        # segment with ``-reset_timestamps`` (recorder.build_segment_args), so its
        # PTS restarts near zero, and the HEVC→H.264 transcode re-encodes each
        # segment on its own, restarting the same way. Native HLS (Safari/iOS)
        # re-anchors each segment by EXTINF, but hls.js (Chrome/Firefox — the web
        # player) does NOT: without an explicit discontinuity it overlays every
        # segment at the same ~0 s offset, so only the FIRST ~10 s ever becomes
        # seekable and the rest of the window never plays (footage "doesn't load"
        # on web / frozen tile). A DISCONTINUITY before each segment after the
        # first tells the player the timeline resets there, so the segments stitch
        # into the full window and seeking works.
        if idx > 0:
            lines.append("#EXT-X-DISCONTINUITY")
        lines.append(f"#EXTINF:{float(SEGMENT_SECONDS):.3f},")
        lines.append(f"seg/{int(seg_start)}.ts{suffix}")
    lines.append("#EXT-X-ENDLIST")
    body = "\n".join(lines) + "\n"
    return PlainTextResponse(body, media_type="application/vnd.apple.mpegurl")


@router.get("/{camera}/export.mp4")
async def recordings_export(
    camera: str,
    request: Request,
    start: float = Query(..., description="window start (epoch seconds)"),
    end: float = Query(..., description="window end (epoch seconds)"),
) -> FileResponse:
    """Export the continuous footage in ``[start, end]`` as a single
    browser-playable, downloadable H.264 faststart MP4.

    Reuses the recorder/transcoder machinery end-to-end
    (``Recorder.export_range``): the same segment selection that cuts event
    clips, concat + precise cut to the window, and stream-copy for H.264 sources
    or an NVENC→libx264 transcode for HEVC. The result is built into a bounded
    on-disk cache (identical windows share the file + one in-flight build) and
    served as ``Content-Disposition: attachment``.

    - ``end ≤ start`` ⇒ 400; a window longer than ``EXPORT_MAX_SECONDS`` ⇒ 400.
    - No footage anywhere in the window ⇒ 404.
    - ffmpeg failed / unavailable ⇒ 503 (never a bare 500; temp files cleaned up).
    Media-scope auth like the rest of the recordings routes (``?token=`` ok)."""
    _camera_dir(request, camera)  # traversal guard (404 on unknown camera)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be after start")
    if end - start > EXPORT_MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Export window too long — max {EXPORT_MAX_SECONDS // 60} minutes "
                f"({EXPORT_MAX_SECONDS}s) per export"
            ),
        )
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder unavailable")
    try:
        path = await recorder.export_range(camera, start, end)
    except Exception:  # noqa: BLE001 — an ffmpeg/build crash is a 5xx, never a bare 500
        log.warning(
            "recordings export crashed cam=%s window=%.1f..%.1f",
            camera, start, end, exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Export failed")
    if path is None:
        # export_range returns None for BOTH "no footage" and "ffmpeg failed";
        # re-check the window so an empty window is a clean 404 and a real build
        # failure is a 503 (both already logged + cleaned up by the recorder).
        has_footage = await asyncio.to_thread(
            select_segments, _camera_dir(request, camera), start, end
        )
        if not has_footage:
            raise HTTPException(
                status_code=404, detail="No footage in the requested time range"
            )
        raise HTTPException(
            status_code=503, detail="Export failed — could not assemble the recording"
        )
    filename = _export_filename(camera, start, end)
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{camera}/seg/{start_ts}.ts")
async def recordings_segment(camera: str, start_ts: int, request: Request) -> FileResponse:
    """Serve the segment whose start epoch is ``start_ts`` (FileResponse ->
    Range/seek supported). 404 when no such segment exists.

    When the camera's main stream is H.264 the raw segment is served unchanged
    (fast stream-copy path). When it is HEVC — which browsers can't decode via
    HLS/MSE — the recorder's transcoder returns a cached, independently-decodable
    H.264 MPEG-TS segment instead, so the frontend needs no change (same URL,
    same media type). The playlist is unaffected. On any transcode failure the
    raw segment is served (browser may fail, but never a 500)."""
    cam_dir = _camera_dir(request, camera)
    seg = _find_segment(cam_dir, start_ts)
    if seg is None:
        raise HTTPException(status_code=404, detail="Segment not found")
    serve_path = seg
    recorder = getattr(request.app.state, "recorder", None)
    transcoder = getattr(recorder, "transcoder", None) if recorder is not None else None
    if transcoder is not None:
        try:
            alt = await transcoder.segment_for_playback(camera, start_ts, seg)
            if alt is not None:
                serve_path = alt
        except Exception:  # noqa: BLE001 — never 500 a playback seek over a transcode
            log.warning(
                "transcode: segment serve failed camera=%s ts=%d — serving original",
                camera, start_ts, exc_info=True,
            )
    return FileResponse(serve_path, media_type="video/mp2t")


def _find_segment(camera_dir: Path, start_ts: int) -> Optional[Path]:
    """The .ts file whose parsed start epoch equals ``start_ts``, resolved
    strictly inside ``camera_dir``. None when absent."""
    hd = hour_dir(camera_dir, float(start_ts))
    if not hd.is_dir():
        return None
    root = camera_dir.resolve()
    for seg in hd.glob("*.ts"):
        if parse_segment_start(seg) == float(start_ts):
            resolved = seg.resolve()
            if root in resolved.parents:  # defense in depth
                return resolved
    return None

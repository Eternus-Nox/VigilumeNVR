"""Event routes: list/detail/annotated snapshot/clip/delete.

Media backing is native: snapshots come from /data/snapshots (annotated by
the pipeline) with a live fallback to the engine's best-frame cache; clips
are served straight from the recorder's clip files
(<media>/native/clips/{event_id}.mp4 — Starlette's FileResponse handles
Range natively, so seeking works).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from .. import annotate
from ..auth import require_admin, require_auth, require_media_auth

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

# Filesystem/header-safe filename parts: collapse every run of anything that
# isn't [A-Za-z0-9._-] to one underscore, then trim leading/trailing separators.
# Keeps download filenames ASCII-safe so the quoted Content-Disposition needs no
# RFC 5987 encoding and can't inject header/quote characters.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename_part(value: str, fallback: str) -> str:
    cleaned = _FILENAME_UNSAFE.sub("_", value).strip("._")
    return cleaned or fallback


def _download_filename(event: dict[str, Any], ext: str) -> str:
    """``<camera>_<label>_<YYYY-MM-DD_HH-MM-SS>.<ext>`` from the event row
    (local time), each part sanitized to ASCII filename-safe characters."""
    camera = _sanitize_filename_part(str(event.get("camera") or ""), "camera")
    label = _sanitize_filename_part(str(event.get("label") or ""), "event")
    ts = event.get("start_time")
    stamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S",
        time.localtime(ts if isinstance(ts, (int, float)) else time.time()),
    )
    return f"{camera}_{label}_{stamp}.{ext}"


def _attachment_headers(filename: str) -> dict[str, str]:
    """Content-Disposition forcing a download with the given (already
    sanitized) filename. Omitted entirely for the default inline responses."""
    return {"Content-Disposition": f'attachment; filename="{filename}"'}

# Backend-generated events: doorbell button presses, legacy audio rows, and
# camera_ai_only object events created straight from the camera's AI. None of
# them has an ENGINE snapshot (no tracked object, no best frame), so their image
# always comes from the saved /data/snapshots file.
_SYNTHETIC_PREFIXES = ("doorbell.", "audio.", "cameraai.")

# ...but "no engine snapshot" and "no clip" are NOT the same property, and
# doorbell events are exactly where they diverge. A press now holds its event
# open until the visitor leaves and schedules a real clip cut from the 24/7
# segments (events_pipeline._doorbell_recording), so it has a clip despite
# having no engine media. Audio and camera_ai rows still never produce one.
#
# Kept as a separate tuple rather than a special case inside _is_synthetic so
# the snapshot path below is provably unchanged for all three prefixes.
_NO_CLIP_PREFIXES = ("audio.", "cameraai.")
_DOORBELL_PREFIX = "doorbell."

# How long after an event ends we still call a missing clip "processing"
# (assembly runs ~20 s after end; a little slack covers a busy recorder).
CLIP_PROCESSING_WINDOW_S = 45.0

# clip_state -> the 404 detail the clip route returns, so the UI message
# matches exactly what the state says.
_CLIP_STATE_DETAIL = {
    "recording_disabled": "Recording is disabled for this camera",
    "processing": "Clip is still being prepared",
    "unavailable": "Clip not available",
}


async def _get_event_or_404(request: Request, event_id: int) -> dict[str, Any]:
    event = await request.app.state.db.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _is_synthetic(event: dict[str, Any]) -> bool:
    """No ENGINE media for this event — its snapshot comes from disk or not at
    all. Governs the snapshot fallback only; see _never_has_clip for clips."""
    fid = event.get("frigate_id") or ""
    return not fid or fid.startswith(_SYNTHETIC_PREFIXES)


def _never_has_clip(event: dict[str, Any]) -> bool:
    """This event can never produce a clip, so a missing file is final rather
    than "still processing". An event with no frigate_id at all was never
    associated with a recording window and qualifies."""
    fid = event.get("frigate_id") or ""
    if not fid or fid.startswith(_NO_CLIP_PREFIXES):
        return True
    if fid.startswith(_DOORBELL_PREFIX):
        # A doorbell row only gets a clip if it was HELD OPEN while the visitor
        # was there. One closed AT the press (end_time == start_time) never was
        # and never will be: every row predating the hold-open feature, a press
        # on a camera with recording off, a repeat press during a visit already
        # being recorded, and a visit abandoned because Privacy Mode engaged.
        #
        # Without this they fall through to "unavailable", which both UIs render
        # as "No recording was saved for this event" — a recorder fault that
        # never happened, retroactively across the entire doorbell history.
        # end_time <= start_time is otherwise unreachable, so it is a safe tell.
        start, end = event.get("start_time"), event.get("end_time")
        if start is not None and end is not None and end <= start:
            return True
    return False


async def _record_enabled_for(request: Request, event: dict[str, Any]) -> bool:
    """Whether the event's camera currently has 24/7 recording enabled
    (False when the camera row is gone)."""
    camera = event.get("camera")
    if not camera:
        return False
    cam = await request.app.state.db.get_camera(camera)
    return bool(cam and cam.get("record_enabled"))


def _clip_state(
    event: dict[str, Any], record_enabled: bool, file_present: bool = True
) -> str:
    """Derive the clip's lifecycle state so the UI can tell "still processing"
    apart from "never coming":

    - ``ready``               the clip file exists (has_clip)
    - ``processing``          recording on, ended recently, clip not written yet
    - ``recording_disabled``  recording off for the camera (or a synthetic
                              audio/camera-AI event that never produces a clip,
                              or a doorbell row closed at the press)
    - ``unavailable``         recording on but the clip never landed (recorder
                              was down for the window / extraction failed)

    ``file_present`` lets a caller that has already stat-ed the path say so.
    has_clip is a DB flag and the file can outlive or predecease it: the
    recorder's clip retention deletes .mp4s by mtime without clearing the flag,
    so a row can claim "ready" for a file that is gone. Both clients gate their
    player on clip_state == "ready", and would mount it against a URL that 404s.
    """
    if event.get("has_clip") and file_present:
        return "ready"
    if _never_has_clip(event) or not record_enabled:
        return "recording_disabled"
    # end_time is NULL while an event is still open — a doorbell visit in
    # progress, or a detection the engine has not ended yet. That is the
    # earliest possible moment, so it reads as "processing", which is exactly
    # what it is.
    end_time = event.get("end_time")
    ended_ago = (time.time() - end_time) if end_time is not None else 0.0
    return "processing" if ended_ago < CLIP_PROCESSING_WINDOW_S else "unavailable"


@router.get("", dependencies=[Depends(require_auth)])
async def list_events(
    request: Request,
    camera: Optional[str] = None,
    label: Optional[str] = None,
    after: Optional[float] = None,
    before: Optional[float] = None,
    # Up to 1000 so the timeline can pull a full day of event markers in one
    # request (it asks for 500); the default stays 50 for the events list.
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    events, total = await request.app.state.db.list_events(
        camera=camera, label=label, after=after, before=before, limit=limit, offset=offset
    )
    return {"events": events, "total": total}


@router.get("/{event_id}", dependencies=[Depends(require_auth)])
async def get_event(event_id: int, request: Request) -> dict[str, Any]:
    event = await _get_event_or_404(request, event_id)
    record_enabled = await _record_enabled_for(request, event)
    # Additive fields (existing consumers keep working): record_enabled and a
    # derived clip_state let the UI show an accurate clip status instead of a
    # dead "recording unavailable" whenever a clip is late or never coming.
    return {
        **event,
        "clip_url": f"/api/events/{event_id}/clip.mp4",
        "snapshot_url": f"/api/events/{event_id}/snapshot.jpg",
        "record_enabled": record_enabled,
        "clip_state": _clip_state(event, record_enabled),
    }


@router.get("/{event_id}/snapshot.jpg", dependencies=[Depends(require_media_auth)])
async def event_snapshot(
    event_id: int,
    request: Request,
    download: bool = Query(default=False, description="attach as a download when true"),
):
    event = await _get_event_or_404(request, event_id)
    # Default is inline (no Content-Disposition — the browser shows it); with
    # ?download=1 the same bytes are served as an attachment with a friendly,
    # sanitized filename.
    headers = _attachment_headers(_download_filename(event, "jpg")) if download else None
    path = request.app.state.config.snapshots_dir / f"{event_id}.jpg"
    if path.is_file():
        return FileResponse(path, media_type="image/jpeg", headers=headers)
    # Annotated copy not saved (yet) — serve the engine's clean best frame.
    if not _is_synthetic(event):
        jpeg = await request.app.state.media.event_snapshot(event["frigate_id"], retries=1)
        if jpeg:
            return Response(content=jpeg, media_type="image/jpeg", headers=headers)
    raise HTTPException(status_code=404, detail="Snapshot not available")


@router.get("/{event_id}/clip.mp4", dependencies=[Depends(require_media_auth)])
async def event_clip(
    event_id: int,
    request: Request,
    download: bool = Query(default=False, description="attach as a download when true"),
):
    event = await _get_event_or_404(request, event_id)
    if _never_has_clip(event):
        raise HTTPException(status_code=404, detail="This event has no clip")
    path = request.app.state.media.clip_path(event["id"])
    if not path.is_file():
        # Clip assembly runs ~20 s after event end; before that (or if the
        # recorder was down for the window) there is nothing to serve. The
        # 404 detail mirrors clip_state so the UI message is accurate
        # (processing vs. recording disabled vs. never coming).
        record_enabled = await _record_enabled_for(request, event)
        # file_present=False: we are here BECAUSE the file is missing, so a
        # stale has_clip=1 must not be allowed to answer "ready" — that maps to
        # no detail string at all and 404s with a bare fallback message.
        state = _clip_state(event, record_enabled, file_present=False)
        detail = _CLIP_STATE_DETAIL.get(state, "Clip not available")
        raise HTTPException(status_code=404, detail=detail)
    # Inline by default (Range/seek works either way — Starlette's FileResponse
    # honours Range regardless of Content-Disposition); ?download=1 adds the
    # attachment header + sanitized filename so the browser saves the file.
    headers = _attachment_headers(_download_filename(event, "mp4")) if download else None
    return FileResponse(path, media_type="video/mp4", headers=headers)


def _purge_event_media(request: Request, event: dict[str, Any]) -> None:
    """Remove an event's on-disk snapshot + clip (best-effort)."""
    event_id = int(event["id"])
    path = request.app.state.config.snapshots_dir / f"{event_id}.jpg"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove snapshot file %s", path)
    try:
        request.app.state.media.clip_path(event_id).unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove clip file for event %d", event_id)


def _purge_all_event_media(request: Request) -> int:
    """Remove EVERY event snapshot + clip file (best-effort). Returns the count
    removed. Scoped strictly to the snapshots + event-clips dirs — continuous
    recordings live under a different tree and are never touched here."""
    removed = 0
    config = request.app.state.config
    for directory, pattern in ((config.snapshots_dir, "*.jpg"), (config.clips_dir, "*.mp4")):
        try:
            entries = list(directory.glob(pattern))
        except OSError:
            log.warning("could not list event media dir %s", directory)
            continue
        for f in entries:
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except OSError:
                log.warning("could not remove event media file %s", f)
    return removed


@router.delete("/{event_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_event(event_id: int, request: Request) -> Response:
    event = await _get_event_or_404(request, event_id)
    await request.app.state.db.delete_event(event_id)
    _purge_event_media(request, event)
    return Response(status_code=204)


def _normalized_foot(
    box: Any, detect_dims: Optional[tuple[int, int]]
) -> Optional[tuple[float, float]]:
    """Normalized (0..1) bottom-center of a detect-pixel [x1,y1,x2,y2] box, or
    None when the box is malformed or detect dims are missing/zero."""
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not detect_dims:
        return None
    dw, dh = detect_dims
    if not dw or not dh:
        return None
    try:
        x1, _y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    fx = min(max(((x1 + x2) / 2.0) / dw, 0.0), 1.0)
    fy = min(max(y2 / dh, 0.0), 1.0)
    return (fx, fy)


@router.post("/{event_id}/reject", status_code=201, dependencies=[Depends(require_admin)])
async def reject_event(event_id: int, request: Request) -> dict[str, Any]:
    """ADMIN: mark an event as a false detection ("not a real <object>"). Learns
    a per-camera/label suppression at the detection's foot-point — future
    matching detections near that spot are dropped before they open an event —
    saves a cropped thumbnail, and deletes this event. Irreversible."""
    event = await _get_event_or_404(request, event_id)
    box = event.get("box") or []
    camera = str(event.get("camera") or "")
    label = str(event.get("label") or "")
    media = request.app.state.media
    detect_dims = await media.detect_dims(camera)
    foot = _normalized_foot(box, detect_dims)
    if not camera or not label or foot is None:
        raise HTTPException(status_code=400, detail="Event has no detection box to reject")
    foot_x, foot_y = foot

    # Crop a thumbnail from the saved (or engine-cached) snapshot — best-effort.
    config = request.app.state.config
    jpeg: Optional[bytes] = None
    snap_path = config.snapshots_dir / f"{event_id}.jpg"
    try:
        if snap_path.is_file():
            jpeg = await asyncio.to_thread(snap_path.read_bytes)
    except OSError:
        jpeg = None
    if jpeg is None and not _is_synthetic(event):
        jpeg = await media.event_snapshot(event["frigate_id"], retries=1)
    thumb: Optional[bytes] = None
    if jpeg is not None:
        try:
            thumb = await asyncio.to_thread(annotate.crop_thumbnail, jpeg, list(box), detect_dims)
        except Exception:  # noqa: BLE001 — the thumbnail is optional; never fail the reject over it
            log.warning("suppression thumbnail crop failed for event %d", event_id, exc_info=True)
            thumb = None

    db = request.app.state.db
    sid = await db.insert_suppression(camera, label, foot_x, foot_y, has_thumb=bool(thumb))
    if thumb is not None:
        thumbs_dir = config.suppression_thumbs_dir
        try:
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread((thumbs_dir / f"{sid}.jpg").write_bytes, thumb)
        except OSError:
            log.warning("could not write suppression thumbnail %d", sid)

    # Remove the false event + its media, then reload the engine to arm the
    # sample so future matching detections are dropped.
    await db.delete_event(event_id)
    _purge_event_media(request, event)
    try:
        await request.app.state.engine.reload()
    except Exception:  # noqa: BLE001 — the suppression is stored; a reload retry self-heals
        log.warning("engine reload after reject failed", exc_info=True)
    return {
        "id": sid,
        "camera": camera,
        "label": label,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "has_thumb": bool(thumb),
        "created_at": time.time(),
    }


@router.delete("", status_code=200, dependencies=[Depends(require_admin)])
async def delete_all_events(request: Request) -> dict[str, int]:
    """ADMIN ONLY. Permanently delete EVERY event plus all event snapshots and
    clips. Continuous timeline recordings are NOT touched. Irreversible."""
    deleted = await request.app.state.db.delete_all_events()
    files_removed = await asyncio.to_thread(_purge_all_event_media, request)
    ws = getattr(request.app.state, "ws", None)
    if ws is not None:
        try:
            await ws.broadcast({"type": "events_cleared"})
        except Exception:  # noqa: BLE001 — live-refresh is best-effort
            log.warning("events_cleared broadcast failed", exc_info=True)
    log.warning("ADMIN: deleted all events (%d rows, %d media files)", deleted, files_removed)
    return {"deleted": deleted, "files_removed": files_removed}



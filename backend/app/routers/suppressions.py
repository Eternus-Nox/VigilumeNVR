"""Reject-to-suppress management API: list / remove learned detection
suppressions and serve their thumbnails.

The reject itself is ``POST /api/events/{id}/reject`` (events router); these
routes manage the resulting samples. List + delete are admin-only (Bearer); the
thumbnail is media-scope (``?token=``) so a plain ``<img>`` / ``AsyncImage`` can
load it, mirroring the event-snapshot route. Removing a suppression reloads the
engine so the sample stops dropping detections immediately.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from ..auth import require_admin, require_media_admin

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detection/suppressions", tags=["suppressions"])


def _with_thumb_url(s: dict[str, Any]) -> dict[str, Any]:
    """Attach a thumbnail URL when the suppression has a saved crop."""
    if s.get("has_thumb"):
        return {**s, "thumbnail_url": f"/api/detection/suppressions/{s['id']}/thumb.jpg"}
    return s


@router.get("", dependencies=[Depends(require_admin)])
async def list_suppressions(
    request: Request, camera: Optional[str] = Query(default=None)
) -> list[dict[str, Any]]:
    """ADMIN: all learned suppressions (newest first), optionally one camera."""
    items = await request.app.state.db.list_suppressions(camera)
    return [_with_thumb_url(s) for s in items]


@router.delete("/{suppression_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_suppression(suppression_id: int, request: Request) -> Response:
    """ADMIN: remove one suppression (re-enables those detections) and reload
    the engine so it stops matching immediately."""
    ok = await request.app.state.db.delete_suppression(suppression_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Suppression not found")
    try:
        (request.app.state.config.suppression_thumbs_dir / f"{suppression_id}.jpg").unlink(
            missing_ok=True
        )
    except OSError:
        log.warning("could not remove suppression thumbnail %d", suppression_id)
    try:
        await request.app.state.engine.reload()
    except Exception:  # noqa: BLE001 — the row is gone; a later reload self-heals
        log.warning("engine reload after suppression delete failed", exc_info=True)
    return Response(status_code=204)


@router.get("/{suppression_id}/thumb.jpg", dependencies=[Depends(require_media_admin)])
async def suppression_thumb(suppression_id: int, request: Request) -> FileResponse:
    """ADMIN: cropped thumbnail of the rejected detection.

    Media-style auth (?token= works, so <img>/AsyncImage can load it without
    headers) but admin-gated: suppressions are an admin-only feature and the
    ids are sequential, so require_media_auth alone would let any authenticated
    viewer walk them."""
    path = request.app.state.config.suppression_thumbs_dir / f"{suppression_id}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No thumbnail")
    return FileResponse(path, media_type="image/jpeg")

"""Software Privacy Mode API — per-camera / per-group capture kill switch.

ADMIN-ONLY, both verbs. Privacy Mode is an operator control: who is being
watched and who is not is an administrator's decision, so a viewer may neither
change it (POST) nor enumerate the configuration (GET).

The viewer UI does NOT need this endpoint. A viewer still gets the one bit it
needs — the per-camera `private` flag on `GET /api/cameras` — which is what the
dashboard renders the "Privacy Mode" overlay from. That flag is the RESOLVED
effect, not the configuration: it says "this camera is not being captured"
without revealing the camera/group selection behind it. An earlier revision made
GET any-authenticated for the overlay; the overlay never used it.

Deliberately its OWN router + its own persisted key (see app/privacy.py) — kept
entirely off the `/api/settings` surface so a full-replace PUT can never wipe it
and silently resume capture.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import privacy as privacy_state
from ..auth import require_admin

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


class PrivacyUpdate(BaseModel):
    """Partial update: send only the field(s) you're changing. `cameras` are
    camera names put directly into Privacy Mode; `groups` are camera-group ids
    whose members go private. Omitted field = left unchanged."""

    cameras: Optional[list[str]] = Field(default=None, max_length=512)
    groups: Optional[list[int]] = Field(default=None, max_length=512)


def _payload(raw: dict[str, Any], resolved: frozenset[str]) -> dict[str, Any]:
    return {
        "cameras": sorted(raw["cameras"]),
        "groups": sorted(raw["groups"]),
        # The effective set the gates enforce (direct ∪ group members, existing
        # cameras only). Admin-facing: the viewer overlay reads the per-camera
        # `private` flag on /api/cameras instead.
        "private_cameras": sorted(resolved),
        "enabled": bool(resolved),
    }


@router.get("", dependencies=[Depends(require_admin)])
async def get_privacy(request: Request) -> dict[str, Any]:
    db = request.app.state.db
    raw = await privacy_state.load_raw(db)
    return _payload(raw, await privacy_state.resolve(db, raw))


@router.post("", dependencies=[Depends(require_admin)])
async def set_privacy(request: Request, body: PrivacyUpdate) -> dict[str, Any]:
    state = request.app.state
    raw = await privacy_state.load_raw(state.db)
    if body.cameras is not None:
        raw["cameras"] = sorted({c.strip() for c in body.cameras if c.strip()})
    if body.groups is not None:
        raw["groups"] = sorted({int(g) for g in body.groups})

    # Persist FIRST, and treat a persistence failure as a failed toggle: if we
    # acked success but the write was lost, a restart would revert to the old
    # persisted state and resume capture the admin believes is stopped.
    try:
        await privacy_state.save_raw(state.db, raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"failed to persist privacy state: {exc}") from exc

    # Recompute the resolved set, THEN drive the live reconcile. refresh() before
    # apply() so every subsystem's gate sees the new set. apply() awaits all stop
    # hooks before we return — stop-then-ack, never footage-flowing-then-ack.
    resolved = await privacy_state.refresh(state)
    await privacy_state.apply(state)
    return _payload(raw, resolved)

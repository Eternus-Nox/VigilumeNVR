"""Camera groups CRUD (docs/CONTRACTS.md — Groups & camera order addendum).

Groups are named, ordered subsets of cameras for the dashboard selector and
TV mode. The stored `cameras` JSON array is the display order; unknown or
deleted camera names are tolerated in storage and filtered out of every API
response (a camera recreated with the same name re-appears in its groups).
"""
from __future__ import annotations

from typing import Any, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from ..auth import require_admin, require_auth, role_from_claims
from .. import privacy

router = APIRouter(prefix="/api/groups", tags=["groups"], dependencies=[Depends(require_auth)])


async def _guard_privacy_group(request: Request, group_id: int) -> None:
    """A NON-ADMIN must not mutate a group that Software Privacy Mode selects.

    Group membership feeds the resolved private set (privacy.resolve: direct
    cameras UNION the members of every selected group), and both mutating routes
    below call ``privacy.on_change`` — so without this guard a viewer could edit
    the capture kill switch that only an admin is allowed to set:

      * remove cameras from a private group (or DELETE the group outright) and
        those cameras leave Privacy Mode — recording, detection and live view
        RESUME on cameras an admin deliberately switched off;
      * add cameras to a private group and they go dark — a denial of
        surveillance on the rest of the fleet.

    Deliberately scoped to privacy-selected groups only: ordinary group CRUD
    stays any-authenticated (the existing product decision), so a viewer keeps
    full control of every group Privacy Mode does not depend on.
    """
    claims = await require_auth(request)
    if role_from_claims(claims) == "admin":
        return
    raw = await privacy.load_raw(request.app.state.db)
    if group_id in set(raw.get("groups") or ()):
        raise HTTPException(
            status_code=403,
            detail="This group is used by Privacy Mode; only an admin can change it",
        )


def _clean_cameras(v: Optional[list[str]]) -> Optional[list[str]]:
    """Strip/skip blanks and duplicates, keep order. Unknown names are NOT an
    error (filtered at read time per contract)."""
    if v is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in v:
        item = item.strip()
        if item and len(item) <= 64 and item not in seen:
            cleaned.append(item)
            seen.add(item)
    return cleaned


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    cameras: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("group name must not be blank")
        return v

    @field_validator("cameras")
    @classmethod
    def _cameras_clean(cls, v: list[str]) -> list[str]:
        return _clean_cameras(v) or []


class GroupUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    cameras: Optional[list[str]] = None
    position: Optional[int] = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def _name_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("group name must not be blank")
        return v

    @field_validator("cameras")
    @classmethod
    def _cameras_clean(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return _clean_cameras(v)


async def _filtered(request: Request, group: dict[str, Any]) -> dict[str, Any]:
    """Group response with unknown/deleted camera names filtered out."""
    known = {cam["name"] for cam in await request.app.state.db.list_cameras()}
    return {**group, "cameras": [n for n in group["cameras"] if n in known]}


# READING is any-auth — a viewer needs the group tabs to navigate the cameras
# they are allowed to watch. WRITING is admin: which cameras are grouped, and
# under what name, is shared configuration, and a viewer is view-only.
@router.get("")
async def list_groups(request: Request) -> list[dict[str, Any]]:
    db = request.app.state.db
    known = {cam["name"] for cam in await db.list_cameras()}
    return [
        {**group, "cameras": [n for n in group["cameras"] if n in known]}
        for group in await db.list_groups()
    ]


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
async def create_group(body: GroupCreate, request: Request) -> dict[str, Any]:
    group = await request.app.state.db.create_group(body.name, body.cameras)
    if group is None:
        raise HTTPException(status_code=409, detail=f"Group '{body.name}' already exists")
    return await _filtered(request, group)


@router.put("/{group_id}", dependencies=[Depends(require_admin)])
async def update_group(group_id: int, body: GroupUpdate, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    if await db.get_group(group_id) is None:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    await _guard_privacy_group(request, group_id)
    try:
        group = await db.update_group(
            group_id, name=body.name, cameras=body.cameras, position=body.position
        )
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Group '{body.name}' already exists")
    if group is None:  # deleted between the check and the update
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    # A membership change to a group that's in Privacy Mode changes who is
    # effectively private — propagate to capture live (no-op if unaffected).
    await privacy.on_change(request.app.state)
    return await _filtered(request, group)


@router.delete(
    "/{group_id}", status_code=204, dependencies=[Depends(require_admin)]
)
async def delete_group(group_id: int, request: Request) -> Response:
    await _guard_privacy_group(request, group_id)
    if not await request.app.state.db.delete_group(group_id):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    await privacy.on_change(request.app.state)
    return Response(status_code=204)

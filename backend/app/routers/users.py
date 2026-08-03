"""User-management API (admin only) — docs/CONTRACTS.md RBAC addendum.

The whole router is admin-gated (require_admin -> 403 for viewers). The
built-in admin ("admin") is env-controlled and is NEVER a DB row, so it can't
be created, targeted, demoted, or deleted here.

- GET    /api/users            -> [{id, username, role, created_at}]  (no hashes)
- POST   /api/users            -> 201 {id, username, role, created_at}
- PUT    /api/users/{id}       -> {id, username, role, created_at}    (reset pw / role)
- DELETE /api/users/{id}       -> 204
- POST   /api/users/me/password (any authenticated DB user) -> 204   (self password)
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..auth import (
    ADMIN_USERNAME,
    hash_password,
    require_admin,
    require_auth,
    verify_password_hash,
)

# Self-service password change is any-authenticated (a DB user changes their
# own password); everything else is admin-only.
router = APIRouter(prefix="/api/users", tags=["users"])

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
_PASSWORD_MIN = 8
_PASSWORD_MAX = 256


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    role: Literal["admin", "viewer"] = "viewer"


class UserUpdate(BaseModel):
    password: Optional[str] = Field(default=None, min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    role: Optional[Literal["admin", "viewer"]] = None


class SelfPasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)


def _validate_username(username: str) -> str:
    username = username.strip().lower()
    if username == ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail="'admin' is a reserved username")
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="username must be 3-32 chars: lowercase letters, digits, '_', '.', '-'",
        )
    return username


# ---------- self-service (any authenticated DB user) ----------


@router.post("/me/password", status_code=204)
async def change_own_password(
    body: SelfPasswordChange, request: Request, claims: dict = Depends(require_auth)
) -> Response:
    """A DB user changes their own password. The built-in admin's password is
    env-controlled and cannot be changed here (400)."""
    username = claims.get("sub") or ADMIN_USERNAME
    if username == ADMIN_USERNAME:
        raise HTTPException(
            status_code=400,
            detail="The built-in admin password is set via ADMIN_PASSWORD (env), not the API",
        )
    user = await request.app.state.db.get_user_by_username(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password_hash(body.current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    await request.app.state.db.update_user(user["id"], password_hash=hash_password(body.new_password))
    return Response(status_code=204)


# ---------- admin-only CRUD ----------


@router.get("", dependencies=[Depends(require_admin)])
async def list_users(request: Request) -> list[dict]:
    return await request.app.state.db.list_users()


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
async def create_user(body: UserCreate, request: Request) -> dict:
    username = _validate_username(body.username)
    created = await request.app.state.db.create_user(
        username, hash_password(body.password), body.role
    )
    if created is None:
        raise HTTPException(status_code=409, detail=f"User '{username}' already exists")
    return created


@router.put("/{user_id}", dependencies=[Depends(require_admin)])
async def update_user(user_id: int, body: UserUpdate, request: Request) -> dict:
    db = request.app.state.db
    existing = await db.get_user(user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Guard: never leave the DB without an admin. Demoting the last DB admin is
    # rejected (the built-in env admin is separate and always present, but the
    # last *user* admin is still protected per the RBAC contract).
    if (
        body.role is not None
        and body.role != "admin"
        and existing["role"] == "admin"
        and await db.count_admin_users() <= 1
    ):
        raise HTTPException(status_code=400, detail="Cannot demote the last admin")
    password_hash = hash_password(body.password) if body.password is not None else None
    updated = await db.update_user(user_id, password_hash=password_hash, role=body.role)
    if updated is None:  # deleted between the check and the update
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@router.delete("/{user_id}", status_code=204, dependencies=[Depends(require_admin)])
async def delete_user(user_id: int, request: Request) -> Response:
    db = request.app.state.db
    if await db.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete_user(user_id)
    return Response(status_code=204)

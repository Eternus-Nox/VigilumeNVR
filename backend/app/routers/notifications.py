"""Notification routes: VAPID key, push subscribe/unsubscribe, test send,
APNs (iOS) device register/unregister (docs/push-architecture.md)."""
from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..auth import require_admin, require_auth

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# PushKit VoIP registration for the CallKit doorbell ring lives under /api/push
# (the cross-role FEATURE CONTRACT pins POST /api/push/voip). Included by main.py
# alongside `router`.
push_router = APIRouter(prefix="/api/push", tags=["push"])

# APNs device tokens are opaque hex; currently 64 hex chars but documented as
# variable-length — same generous range as the relay (relay/main.py).
_APNS_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64,160}$")
_APNS_ENVIRONMENTS = ("sandbox", "production")
_APNS_KEY_BYTES = 32
_APNS_NAME_MAX = 64


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscription(BaseModel):
    endpoint: str = Field(min_length=8)
    keys: SubscriptionKeys
    expirationTime: Optional[float] = None


class UnsubscribeRequest(BaseModel):
    endpoint: str


@router.get("/vapid-public-key")
async def vapid_public_key(request: Request) -> dict:
    # No auth: fetched pre-subscribe by the service worker context.
    return {"key": request.app.state.push.public_key}


@router.post("/subscribe", status_code=204, dependencies=[Depends(require_auth)])
async def subscribe(body: PushSubscription, request: Request) -> Response:
    subscription = body.model_dump(exclude_none=True)
    await request.app.state.db.upsert_subscription(body.endpoint, subscription)
    log.info("push subscription registered (%s...)", body.endpoint[:60])
    return Response(status_code=204)


@router.post("/unsubscribe", status_code=204, dependencies=[Depends(require_auth)])
async def unsubscribe(body: UnsubscribeRequest, request: Request) -> Response:
    await request.app.state.db.delete_subscription(body.endpoint)
    return Response(status_code=204)


class ApnsRegisterRequest(BaseModel):
    device_token: str
    device_name: Optional[str] = None
    key_b64: str
    environment: str = "production"


class ApnsUnregisterRequest(BaseModel):
    device_token: str


@router.post("/apns/register", status_code=204, dependencies=[Depends(require_auth)])
async def apns_register(body: ApnsRegisterRequest, request: Request) -> Response:
    """Register (upsert) an iOS device for APNs push. Any authenticated role —
    a viewer's phone gets pushes too. Contract validation (400, not 422):
    hex token 64-160 chars stored LOWERCASED; key_b64 decoding to exactly 32
    bytes; device_name capped at 64 chars; environment sandbox|production."""
    if not _APNS_TOKEN_RE.fullmatch(body.device_token):
        raise HTTPException(status_code=400, detail="device_token must be 64-160 hex characters")
    try:
        key = base64.b64decode(body.key_b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="key_b64 is not valid base64")
    if len(key) != _APNS_KEY_BYTES:
        raise HTTPException(status_code=400, detail="key_b64 must decode to exactly 32 bytes")
    environment = (body.environment or "").strip().lower()
    if environment not in _APNS_ENVIRONMENTS:
        raise HTTPException(status_code=400, detail="environment must be 'sandbox' or 'production'")
    token = body.device_token.lower()
    device_name = (body.device_name or "").strip()[:_APNS_NAME_MAX]
    await request.app.state.db.upsert_apns_device(token, device_name, body.key_b64, environment)
    log.info("apns device registered token=%s… env=%s", token[:8], environment)
    return Response(status_code=204)


@router.delete("/apns/register", status_code=204, dependencies=[Depends(require_auth)])
async def apns_unregister(body: ApnsUnregisterRequest, request: Request) -> Response:
    """Unregister an APNs device. Idempotent: 204 whether or not it existed."""
    await request.app.state.db.delete_apns_device(body.device_token.strip().lower())
    return Response(status_code=204)


@router.get("/apns/devices", dependencies=[Depends(require_auth)])
async def apns_devices(request: Request) -> list[dict[str, Any]]:
    """Registered APNs devices — 8-char token prefixes only (enough for the
    UI to disambiguate without exposing the full capability)."""
    rows = await request.app.state.db.list_apns_devices()
    return [
        {
            "device_token_prefix": row["device_token"][:8],
            "device_name": row["device_name"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


class VoipRegisterRequest(BaseModel):
    # `token` is the contract field name (POST /api/push/voip {token, ...});
    # `device_token` is accepted too for parity with the APNs route.
    token: Optional[str] = None
    device_token: Optional[str] = None
    device_name: Optional[str] = None
    environment: str = "production"


class VoipUnregisterRequest(BaseModel):
    token: Optional[str] = None
    device_token: Optional[str] = None


def _voip_token(body: "VoipRegisterRequest | VoipUnregisterRequest") -> str:
    return (body.token or body.device_token or "").strip()


@push_router.post("/voip", status_code=204, dependencies=[Depends(require_auth)])
async def voip_register(body: VoipRegisterRequest, request: Request) -> Response:
    """Register (upsert) a PushKit VoIP token for the doorbell CallKit ring. Any
    authenticated role (a viewer's phone rings too). Same token validation as
    APNs (hex 64-160, stored lowercased); no encryption key (the VoIP payload is
    minimal + not E2E-encrypted)."""
    raw = _voip_token(body)
    if not _APNS_TOKEN_RE.fullmatch(raw):
        raise HTTPException(status_code=400, detail="token must be 64-160 hex characters")
    environment = (body.environment or "").strip().lower()
    if environment not in _APNS_ENVIRONMENTS:
        raise HTTPException(status_code=400, detail="environment must be 'sandbox' or 'production'")
    token = raw.lower()
    device_name = (body.device_name or "").strip()[:_APNS_NAME_MAX]
    await request.app.state.db.upsert_voip_device(token, device_name, environment)
    log.info("voip device registered token=%s… env=%s", token[:8], environment)
    return Response(status_code=204)


@push_router.delete("/voip", status_code=204, dependencies=[Depends(require_auth)])
async def voip_unregister(body: VoipUnregisterRequest, request: Request) -> Response:
    """Unregister a VoIP token. Idempotent: 204 whether or not it existed."""
    await request.app.state.db.delete_voip_device(_voip_token(body).lower())
    return Response(status_code=204)


@push_router.get("/voip/devices", dependencies=[Depends(require_auth)])
async def voip_devices(request: Request) -> list[dict[str, Any]]:
    """Registered VoIP devices — 8-char token prefixes only."""
    rows = await request.app.state.db.list_voip_devices()
    return [
        {
            "device_token_prefix": row["device_token"][:8],
            "device_name": row["device_name"],
            "environment": row["environment"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.post("/test", dependencies=[Depends(require_admin)])
async def test_notification(request: Request) -> dict[str, Any]:
    """Send a test Web Push to every registered subscription.

    - 400: no push subscriptions registered.
    - 502: subscriptions exist but every send failed (detail names the
      first error).
    - 200: at least one delivery succeeded -> {push_sent}.
    """
    state = request.app.state
    public_url = state.settings.public_url
    payload = {
        "title": "Vigilume NVR test",
        "body": "Notifications are working.",
        "tag": "sentinel-test",
        "data": {"url": f"{public_url}/" if public_url else "/"},
    }
    push_result = await state.push.send_to_all(payload)

    if push_result.attempted == 0:
        raise HTTPException(
            status_code=400,
            detail="No push subscriptions registered — enable notifications on a device first",
        )
    if push_result.sent == 0:
        reason = push_result.errors[0] if push_result.errors else "unknown push failure"
        raise HTTPException(status_code=502, detail=f"All push sends failed: {reason}")
    return {"push_sent": push_result.sent}

"""APNs push (iOS) — E2E-encrypted, via the owner-hosted relay.

docs/push-architecture.md is the PINNED CONTRACT for every shape here.

Two modes (settings.notifications.apns.mode):

- ``relay``: POST the E2E ciphertext to a relay run by the app owner, who
  holds the Apple ``.p8`` (see ``relay/main.py``). The self-hoster needs no
  Apple developer account at all. This backend signs NOTHING and never sees an
  Apple credential; it speaks plain HTTPS to ``{relay_url}/api/push``.
- ``off`` (default): no-op.

The third mode, ``direct`` — this server holding its own ``.p8`` and talking
straight to Apple over HTTP/2 with an ES256 provider JWT — is RETIRED. A stored
``mode="direct"`` is migrated to ``off`` in ``settings_store._strip_legacy``
(and the ``direct`` block popped); it must never reach the pydantic Literal,
which would 422 every settings save and lock the admin out of the settings page.

**notify/ntfy.py** also needs no Apple account, and it stays — but it is a
different product, not a substitute: an ntfy alert lands in the *ntfy app*, so
there is no CallKit ring and no native UI. That is exactly why the relay exists.

Encryption (contract §2): AES-256-GCM with the per-registration 32-byte key,
fresh random 12-byte nonce, no AAD; wire format
``base64(nonce || ciphertext || 16-byte tag)`` — CryptoKit's
``AES.GCM.SealedBox.combined`` layout, so the iOS extension opens it directly.
The relay is blind to all of it.

Registration hygiene: the relay's **410** (``unregistered``) or **400**
``bad_device_token`` deletes the registration row. NOTE the vocabulary: every
reason string on the wire is the RELAY's snake_case, NOT Apple's CamelCase —
the relay collapses Apple's reasons into a closed set (``relay/main.py``
``_err``). Comparing against ``"BadDeviceToken"`` here compiles fine and simply
never prunes.

This service NEVER raises into the events pipeline; per-send errors are
logged with an 8-char token prefix only (never full tokens, never payloads).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..db import Database

log = logging.getLogger(__name__)

# Contract §2 size budget: plaintext JSON <= 2500 bytes (truncate body first).
PLAINTEXT_MAX_BYTES = 2500

# Transient-failure retry policy (contract §3: 502 -> retry with backoff, max ~3).
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 0.5  # doubled per retry; module-level so tests can shrink it

# Outcomes of a single-device send (internal).
_OK, _PRUNED, _ERROR = "ok", "pruned", "error"


def build_plaintext(
    title: str,
    body: str,
    event_id: str,
    snapshot_url: Optional[str],
    camera: Optional[str] = None,
    camera_label: Optional[str] = None,
) -> bytes:
    """The UTF-8 JSON plaintext of contract §2:
    ``{"title", "body", "event_id", "snapshot_url"}`` plus the OPTIONAL
    ``camera`` (slug, used by the iOS extension as the notification
    ``threadIdentifier`` so same-camera events stack) and ``camera_label``
    (friendly name for the collapsed-group summary). Both are omitted entirely
    when absent so the wire stays back-compatible (the extension tolerates
    their absence). Kept under PLAINTEXT_MAX_BYTES by truncating ``body`` first
    (the camera slug/label are tiny and never truncated)."""
    payload: dict[str, Any] = {
        "title": title,
        "body": body,
        "event_id": event_id,
        "snapshot_url": snapshot_url,
    }
    if camera:
        payload["camera"] = camera
    if camera_label:
        payload["camera_label"] = camera_label
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    while len(raw) > PLAINTEXT_MAX_BYTES and payload["body"]:
        excess = len(raw) - PLAINTEXT_MAX_BYTES
        payload["body"] = payload["body"][: -max(1, excess)]
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return raw


def encrypt_payload(key: bytes, plaintext: bytes) -> str:
    """AES-256-GCM, fresh 12-byte nonce, no AAD ->
    ``base64(nonce || ciphertext || tag)`` (AESGCM appends the 16-byte tag to
    the ciphertext, so ``nonce + ct`` IS the combined layout)."""
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ct).decode("ascii")


@dataclass
class ApnsSendResult:
    """Outcome of a send_to_all fan-out (PushSendResult twin)."""

    attempted: int = 0  # registered devices targeted
    sent: int = 0  # successful deliveries
    pruned: int = 0  # registrations deleted (410 / bad_device_token)
    errors: list[str] = field(default_factory=list)  # one message per failed send


class ApnsService:
    """Fan-out APNs sender. `transport` is injectable for tests (MockTransport)."""

    def __init__(self, db: Database, settings: Any, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._db = db
        self._settings = settings  # SettingsStore (needs .notifications)
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        # httpx logs request URLs at INFO. Ours is {relay_url}/api/push — the
        # device token rides in the BODY, which httpx never logs, so this is no
        # longer the token-leak muzzle it was under direct mode
        # (/3/device/<full token>). Kept because relay_url may embed a hostname
        # the operator treats as private, and it costs nothing. ntfy.py muzzles
        # the same loggers for a sharper reason (its URL IS the secret).
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---------- config / client ----------

    def _cfg(self) -> dict[str, Any]:
        apns = self._settings.notifications.get("apns")
        return apns if isinstance(apns, dict) else {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No base_url: the relay URL is a live setting, built per send.
            # No http2 either — APNs is HTTP/2-only but the RELAY is not, and
            # h2 left this backend with the .p8-era `direct` transport.
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---------- send ----------

    async def send_to_all(
        self,
        *,
        title: str,
        body: str,
        event_id: str,
        snapshot_url: Optional[str],
        camera: Optional[str] = None,
        camera_label: Optional[str] = None,
        priority: str = "high",
        collapse_id: Optional[str] = None,
    ) -> ApnsSendResult:
        """Encrypt + send to every registered APNs device. Never raises.

        ``camera``/``camera_label`` ride INSIDE the encrypted plaintext (the
        relay stays blind to them); the iOS extension uses ``camera`` as the
        notification thread id so same-camera events group on the lock screen.
        """
        result = ApnsSendResult()
        try:
            cfg = self._cfg()
            if not cfg:  # legacy settings blob without an apns block -> no-op
                return result
            # "relay" is the only transport: `direct` (this server holding its
            # own Apple .p8) is retired, and a stored mode="direct" is migrated
            # to "off" by settings_store._strip_legacy.
            # "off"/missing/garbage -> no-op.
            if cfg.get("mode") != "relay":
                return result
            devices = await self._db.list_apns_devices()
            result.attempted = len(devices)
            if not devices:
                return result
            plaintext = build_plaintext(
                title, body, event_id, snapshot_url, camera, camera_label
            )
            outcomes = await asyncio.gather(
                *(
                    self._send_one(cfg, device, plaintext, priority, collapse_id)
                    for device in devices
                ),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    log.warning("apns send raised unexpectedly: %r", outcome)
                    result.errors.append(str(outcome))
                elif outcome[0] == _OK:
                    result.sent += 1
                elif outcome[0] == _PRUNED:
                    result.pruned += 1
                    result.errors.append(outcome[1])
                else:
                    result.errors.append(outcome[1])
        except Exception:  # noqa: BLE001 — never raise into the pipeline
            log.exception("apns send_to_all failed")
        return result

    # ---------- VoIP (PushKit) — CallKit doorbell ring ----------

    async def send_voip_to_all(
        self, *, camera: str, event_id: str
    ) -> ApnsSendResult:
        """Send a VoIP APNs push to every registered PushKit token so the phone
        rings like a real call (iOS reports it via CallKit on receipt).

        The payload is MINIMAL and NOT E2E-encrypted (the app must read it
        immediately to report the incoming call): ``{type:"doorbell", camera,
        event_id}``. ``camera`` is the friendly name used as the CallKit handle.
        NB the relay CAN read this one — contract §3a is explicit that §1's
        "ciphertext only" does not cover the VoIP route.

        Same relay, different endpoint: ``/api/push/voip``. The topic
        (``<bundle>.voip``), push-type (``voip``) and expiration are the
        RELAY's business now, not ours. Never raises."""
        result = ApnsSendResult()
        try:
            cfg = self._cfg()
            if not cfg:  # legacy settings blob without an apns block -> no-op
                return result
            # "relay" is the only transport: `direct` (this server holding its
            # own Apple .p8) is retired, and a stored mode="direct" is migrated
            # to "off" by settings_store._strip_legacy.
            # "off"/missing/garbage -> no-op.
            if cfg.get("mode") != "relay":
                return result
            devices = await self._db.list_voip_devices()
            result.attempted = len(devices)
            if not devices:
                return result
            payload = {"type": "doorbell", "camera": camera, "event_id": event_id}
            outcomes = await asyncio.gather(
                *(self._send_voip_one(cfg, device, payload) for device in devices),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    log.warning("voip send raised unexpectedly: %r", outcome)
                    result.errors.append(str(outcome))
                elif outcome[0] == _OK:
                    result.sent += 1
                elif outcome[0] == _PRUNED:
                    result.pruned += 1
                    result.errors.append(outcome[1])
                else:
                    result.errors.append(outcome[1])
        except Exception:  # noqa: BLE001 — never raise into the pipeline
            log.exception("apns voip send_to_all failed")
        return result

    async def _prune_voip(self, token: str, reason: str) -> tuple[str, str]:
        await self._db.delete_voip_device(token)
        log.info("pruned voip registration token=%s… (%s)", token[:8], reason)
        return (_PRUNED, f"voip registration {token[:8]}… pruned ({reason})")

    async def _send_voip_one(
        self,
        cfg: dict[str, Any],
        device: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        token = device["device_token"]
        tok8 = token[:8]
        try:
            return await self._send_voip_relay(cfg, device, payload)
        except Exception as exc:  # noqa: BLE001 — network layer; never crash callers
            log.warning("voip send failed token=%s…: %s", tok8, type(exc).__name__)
            return (_ERROR, f"voip send failed ({tok8}…): {type(exc).__name__}")

    async def _send_one(
        self,
        cfg: dict[str, Any],
        device: dict[str, Any],
        plaintext: bytes,
        priority: str,
        collapse_id: Optional[str],
    ) -> tuple[str, str]:
        token = device["device_token"]
        tok8 = token[:8]
        try:
            key = base64.b64decode(device.get("key_b64") or "")
            if len(key) != 32:
                log.warning("apns token=%s…: stored key is not 32 bytes — skipping", tok8)
                return (_ERROR, f"apns {tok8}…: stored key is not 32 bytes")
            payload_b64 = encrypt_payload(key, plaintext)
            return await self._send_relay(cfg, device, payload_b64, priority, collapse_id)
        except Exception as exc:  # noqa: BLE001 — network layer; never crash callers
            log.warning("apns send failed token=%s…: %s", tok8, type(exc).__name__)
            return (_ERROR, f"apns send failed ({tok8}…): {type(exc).__name__}")

    async def _prune(self, token: str, reason: str) -> tuple[str, str]:
        await self._db.delete_apns_device(token)
        log.info("pruned apns registration token=%s… (%s)", token[:8], reason)
        return (_PRUNED, f"apns registration {token[:8]}… pruned ({reason})")

    async def _post_with_retry(
        self, url: str, json_body: dict[str, Any], headers: Optional[dict[str, str]] = None
    ) -> Optional[httpx.Response]:
        """POST with backoff on transient failures (network error / 5xx),
        max _RETRY_ATTEMPTS. Returns None when every attempt failed to connect.

        This is contract §3's table: 502 -> retried (max ~3, 0.5s doubling),
        429 -> returned immediately (never retry-storm the relay's limiter),
        410/400 -> immediate so the prune runs exactly once."""
        client = self._get_client()
        resp: Optional[httpx.Response] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await client.post(url, json=json_body, headers=headers)
            except httpx.HTTPError:
                resp = None
            if resp is not None and resp.status_code < 500:
                return resp
            if attempt < _RETRY_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_BACKOFF_S * (2**attempt))
        return resp

    @staticmethod
    def _reason_of(resp: httpx.Response) -> str:
        """The relay's closed-vocabulary reason (snake_case — `relay/main.py`
        `_err`), NOT one of Apple's CamelCase strings.

        The non-JSON tolerance is load-bearing: a Cloudflare tunnel whose origin
        is down answers with an HTML error page under 502/530. 530 is >= 500 so
        _post_with_retry retries it, and this returns "" -> the generic error
        branch. That is correct; do not "fix" it into an exception."""
        try:
            data = resp.json()
            return str(data.get("reason", "")) if isinstance(data, dict) else ""
        except (json.JSONDecodeError, ValueError):
            return ""

    # ---------- relay mode (contract §3) ----------

    def _relay_url(self, cfg: dict[str, Any]) -> str:
        return str(cfg.get("relay_url") or "").rstrip("/")

    async def _send_relay(
        self,
        cfg: dict[str, Any],
        device: dict[str, Any],
        payload_b64: str,
        priority: str,
        collapse_id: Optional[str],
    ) -> tuple[str, str]:
        token = device["device_token"]
        tok8 = token[:8]
        base = self._relay_url(cfg)
        if not base:
            return (_ERROR, "apns relay mode is not configured (relay_url)")
        body: dict[str, Any] = {
            "device_token": token,
            "payload_b64": payload_b64,
            "priority": priority if priority in ("high", "normal") else "high",
        }
        if collapse_id:
            body["collapse_id"] = collapse_id[:64]
        # THE FIX. The relay pins no host at startup; it routes per request.
        # Sent only when the row HAS one, so a legacy NULL row falls back to the
        # relay's APNS_ENV rather than being forced to a wrong host from here.
        env = device.get("environment")
        if env:
            body["environment"] = env
        resp = await self._post_with_retry(f"{base}/api/push", body)
        if resp is None:
            # The RELAY is unreachable (tunnel down / cloudflared not running) —
            # NOT Apple. A healthy relay that cannot reach Apple answers
            # 502 apns_unreachable below. Different pages of the runbook.
            log.warning("relay unreachable token=%s…", tok8)
            return (_ERROR, f"relay unreachable ({tok8}…)")
        reason = self._reason_of(resp)  # snake_case now, NOT Apple's CamelCase
        if resp.status_code == 200:
            return (_OK, "")
        if resp.status_code == 410:  # prune on the STATUS, never the reason
            return await self._prune(token, reason or "unregistered")
        if resp.status_code == 400 and reason == "bad_device_token":
            return await self._prune(token, "bad_device_token")
        if resp.status_code == 502 and reason == "apns_auth":
            log.error("relay reports an APNs auth failure — check the relay's "
                      "APNS_KEY_P8/APNS_KEY_ID/APNS_TEAM_ID. NOT transient: this "
                      "burns %d attempts per device per event.", _RETRY_ATTEMPTS)
        log.warning("apns relay send failed token=%s… status=%s reason=%s",
                    tok8, resp.status_code, reason)
        return (_ERROR, f"apns relay send failed ({tok8}…): HTTP {resp.status_code} {reason}")

    async def _send_voip_relay(
        self, cfg: dict[str, Any], device: dict[str, Any], payload: dict[str, Any]
    ) -> tuple[str, str]:
        token = device["device_token"]
        tok8 = token[:8]
        base = self._relay_url(cfg)
        if not base:
            return (_ERROR, "apns relay mode is not configured (relay_url)")
        # `payload` is a PLAINTEXT dict, not payload_b64. That asymmetry is
        # FORCED by CallKit (the PushKit delegate must report the call
        # immediately; there is no extension window in which to decrypt, and
        # failing to report gets the app terminated). Do not "unify" the paths.
        body: dict[str, Any] = {"device_token": token, "payload": payload}
        env = device.get("environment")
        if env:
            body["environment"] = env
        resp = await self._post_with_retry(f"{base}/api/push/voip", body)
        if resp is None:
            log.warning("voip relay unreachable token=%s…", tok8)
            return (_ERROR, f"voip relay unreachable ({tok8}…)")
        reason = self._reason_of(resp)
        if resp.status_code == 200:
            return (_OK, "")
        if resp.status_code == 410:
            return await self._prune_voip(token, reason or "unregistered")
        if resp.status_code == 400 and reason == "bad_device_token":
            return await self._prune_voip(token, "bad_device_token")
        log.warning("voip relay send failed token=%s… status=%s reason=%s",
                    tok8, resp.status_code, reason)
        return (_ERROR, f"voip relay send failed ({tok8}…): HTTP {resp.status_code} {reason}")

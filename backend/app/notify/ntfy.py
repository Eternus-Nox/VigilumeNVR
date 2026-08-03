"""ntfy (ntfy.sh or self-hosted) push — the no-Apple-account channel.

Peer of notify/push.py (web push) and notify/apns.py: same never-raise policy,
same injectable transport, same Result shape, config read live per send.

WHY THIS EXISTS. On iOS every push ultimately traverses APNs, which needs an
Apple developer account and a signing key. That left a self-hoster two bad
options: buy an Apple account, or trust a relay run by someone else holding a
.p8 on their behalf. ntfy removes both — ntfy.sh already runs that plumbing, or
you run your own ntfy.

IT IS NOT A RELAY REPLACEMENT. It was treated as one for part of a day, and the
relay was deleted on that basis; the relay is back and ntfy stays beside it.
ntfy's notifications arrive in the *ntfy app* — no CallKit doorbell ring, no
native Vigilume UI, no inline snapshot. Different question, different answer:
ntfy = push at all without Apple; relay = the real app experience.

NO REGISTRATION TABLE. Unlike push.py/apns.py there is no `db` and no fan-out:
the TOPIC is the subscriber list, and it lives in settings. One destination, one
request, no prune.

SECURITY — THE TOPIC IS A PASSWORD. ntfy's own docs say it outright: on a
default-allow server (including ntfy.sh) anyone who knows a topic receives every
message published to it. The publish URL is `{server}/{topic}`, so the URL *is*
the secret. Two consequences enforced here:
  1. Nothing logs a full topic, a full URL, or a raw exception (an httpx error
     string embeds the request URL — i.e. the secret). Same discipline as
     apns.py, which muzzles httpx/httpcore for the same reason.
  2. The snapshot is LINKED via the `Attach` header, never uploaded, so the
     image itself never touches the ntfy server — the phone fetches it straight
     from this NVR. NB that link carries a media token; see attach_snapshot in
     the settings model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

# apns.py parity: a slow ntfy must never become this pipeline's problem.
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass
class NtfySendResult:
    """Outcome of one publish. Twin of PushSendResult / ApnsSendResult.

    `attempted` is 0 when the channel is off or unconfigured (so a caller can
    tell "not set up" from "tried and failed") and 1 otherwise.
    """

    attempted: int = 0
    sent: int = 0
    errors: list[str] = field(default_factory=list)


# ntfy renders a `Tags` value that matches an emoji SHORTCODE as an emoji in
# front of the title — that is the only per-notification "icon" that works on
# BOTH iOS and Android (ntfy's `Icon` header, which takes an image URL, is
# Android-only and the iOS app ignores it). Anything that is NOT a shortcode is
# rendered as a literal #hashtag, which is why the internal dedup tag must never
# be sent here.
_ICONS: dict[str, str] = {
    "person": "walking",
    "dog": "dog",
    "cat": "cat",
    "car": "blue_car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bike",
    "package": "package",
}
NTFY_ICON_DOORBELL = "bell"
NTFY_ICON_DEFAULT = "video_camera"


def ntfy_icon(labels: list[str]) -> str:
    """Emoji shortcode for an event's classes. First recognised label wins;
    anything unmapped falls back to a generic camera glyph."""
    for label in labels:
        icon = _ICONS.get((label or "").strip().lower())
        if icon:
            return icon
    return NTFY_ICON_DEFAULT



class NtfyService:
    """Publishes one message per event. `transport` is injectable for tests."""

    def __init__(
        self,
        settings: Any,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        # httpx logs every request URL at INFO. Here that URL is
        # {server}/{topic} — the shared secret — on every single event. Cap the
        # client libraries at WARNING; this module's own lines only ever carry a
        # short topic prefix. (apns.py does the same for device tokens.)
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def _cfg(self) -> dict[str, Any]:
        ntfy = self._settings.notifications.get("ntfy")
        return ntfy if isinstance(ntfy, dict) else {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Deliberately no base_url: baking the server in would make a
            # settings change require a restart. The URL is built per send.
            self._client = httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(
        self,
        *,
        title: str,
        body: str,
        click_url: Optional[str] = None,
        attach_url: Optional[str] = None,
        tag: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> NtfySendResult:
        """Publish one notification. NEVER raises — the pipeline calls this."""
        result = NtfySendResult()
        try:
            cfg = self._cfg()
            if not cfg.get("enabled"):
                return result
            server = str(cfg.get("server") or "").rstrip("/")
            topic = str(cfg.get("topic") or "").strip().strip("/")
            if not server or not topic:
                # Enabled but half-configured. attempted stays 0 so this reads
                # as "not set up" rather than a failure the operator must chase.
                return result
            result.attempted = 1

            headers: dict[str, str] = {
                "Title": title,
                # Per-message override beats the configured default. Used to
                # escalate a doorbell press to max when there is no CallKit ring
                # to carry it (see events_pipeline._send_notification).
                "Priority": str(priority or cfg.get("priority") or 4),
            }
            if tag:
                headers["Tags"] = tag
            if click_url:
                headers["Click"] = click_url
            # Only attach when the operator opted in: the URL carries a media
            # token, and on a default-allow topic every subscriber can read it.
            if attach_url and cfg.get("attach_snapshot", True):
                headers["Attach"] = attach_url
            token = str(cfg.get("auth_token") or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"

            resp = await self._get_client().post(
                f"{server}/{topic}", content=body.encode("utf-8"), headers=headers
            )
            if resp.status_code < 300:
                result.sent = 1
            else:
                # Status only. An ntfy error body can echo the request.
                result.errors.append(f"ntfy publish failed: HTTP {resp.status_code}")
                log.warning(
                    "ntfy publish -> HTTP %d (topic=%s…) — check the server URL, "
                    "topic and auth token",
                    resp.status_code, topic[:4],
                )
        except Exception as exc:  # noqa: BLE001 — never raise into the pipeline
            # type(exc).__name__ only: an httpx exception's str() embeds the
            # request URL, which is {server}/{topic} — the secret.
            result.errors.append(f"ntfy publish failed: {type(exc).__name__}")
            log.warning("ntfy publish failed: %s", type(exc).__name__)
        return result

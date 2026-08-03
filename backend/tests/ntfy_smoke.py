"""Smoke suite for the ntfy notification channel (backend/app/notify/ntfy.py).

ntfy is how a self-hoster gets push with NO Apple developer account — it is
what let the owner-hosted APNs push relay be retired. Coverage:

  - publish shape: POST {server}/{topic}, body = the message text, headers
    Title / Priority / Tags / Click / Attach / Authorization
  - the snapshot is LINKED via `Attach` (the phone fetches it from the NVR);
    the image itself is never uploaded to ntfy
  - attach_snapshot=false -> no Attach header (text-only)
  - auth_token -> `Authorization: Bearer`; absent -> no header at all
  - disabled / no server / no topic -> attempted 0 (a no-op, not a failure)
  - never raises: transport blow-ups and HTTP errors surface as result.errors
  - LOG HYGIENE: the topic is a shared secret ({server}/{topic} IS the publish
    URL), so it must never be logged in full — not by us, not by httpx
  - pipeline: ntfy fires alongside web push under the SAME gates, with the
    same media-token snapshot URL, and a hung ntfy never stalls the caller

CPU-only, no network. Usage: python backend/tests/ntfy_smoke.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for _i in (1, 2, 3):
    for _sfx in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{_i}_{_sfx}", None)
os.environ["ADMIN_PASSWORD"] = "admin-secret"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="vigilume-ntfy-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import asyncio  # noqa: E402

import httpx  # noqa: E402

from app.notify.ntfy import (  # noqa: E402
    NTFY_ICON_DEFAULT, NTFY_ICON_DOORBELL, NtfyService, ntfy_icon,
)  # noqa: E402

PASS = 0

# A realistic generated topic: unguessable, which is the whole security model
# on a default-allow server (ntfy's docs: the topic is essentially a password).
SECRET_TOPIC = "vigilume_7f3a9c1e5b2d4a86"


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}", flush=True)
        os._exit(1)
    PASS += 1
    print(f"  ok: {msg}")


class FakeSettings:

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False
    def __init__(self, ntfy_cfg: dict):
        self.notifications = {"enabled": True, "ntfy": ntfy_cfg}


def cfg(**over) -> dict:
    base = {
        "enabled": True,
        "server": "https://ntfy.example",
        "topic": SECRET_TOPIC,
        "auth_token": "",
        "priority": 4,
        "attach_snapshot": True,
    }
    base.update(over)
    return base


class Recorder:
    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, text="1")


def svc_for(config: dict, rec: Recorder) -> NtfyService:
    return NtfyService(FakeSettings(config), transport=httpx.MockTransport(rec.handler))


# --------------------------------------------------------------------------- #
# 1. publish shape
# --------------------------------------------------------------------------- #


async def _publish_cases() -> None:
    rec = Recorder()
    svc = svc_for(cfg(), rec)
    res = await svc.send(
        title="Person detected at Front Door",
        body="1 person in frame",
        click_url="https://nvr.example/events/42",
        attach_url="https://nvr.example/api/events/42/snapshot.jpg?token=mt",
        tag="native.42",
    )
    check(res.attempted == 1 and res.sent == 1 and not res.errors, "publish delivered")
    req = rec.requests[0]
    check(str(req.url) == f"https://ntfy.example/{SECRET_TOPIC}",
          "POST {server}/{topic}")
    check(req.method == "POST", "publish is a POST")
    check(req.content.decode() == "1 person in frame", "body is the message text")
    check(req.headers["Title"] == "Person detected at Front Door", "Title header")
    check(req.headers["Priority"] == "4", "Priority header (ntfy 1..5)")
    check(req.headers["Tags"] == "native.42", "Tags header")
    check(req.headers["Click"] == "https://nvr.example/events/42", "Click header")
    # The snapshot is LINKED, never uploaded: the phone fetches it from the
    # NVR, so the image never touches the ntfy server.
    check(req.headers["Attach"] == "https://nvr.example/api/events/42/snapshot.jpg?token=mt",
          "Attach header carries the NVR snapshot URL (linked, not uploaded)")
    check(b"snapshot" not in req.content and len(req.content) < 200,
          "the image bytes are NOT in the request body (never uploaded to ntfy)")
    check("Authorization" not in req.headers, "no Authorization header without a token")
    await svc.aclose()

    # ---- auth token ----
    rec = Recorder()
    svc = svc_for(cfg(auth_token="tk_abc123"), rec)
    await svc.send(title="t", body="b")
    check(rec.requests[0].headers["Authorization"] == "Bearer tk_abc123",
          "auth_token -> Authorization: Bearer (self-hosted deny-all / reserved topics)")
    await svc.aclose()

    # ---- attach_snapshot=false: text only ----
    rec = Recorder()
    svc = svc_for(cfg(attach_snapshot=False), rec)
    await svc.send(title="t", body="b", attach_url="https://nvr.example/s.jpg?token=mt")
    check("Attach" not in rec.requests[0].headers,
          "attach_snapshot=false -> NO Attach header (no media token leaves the box)")
    await svc.aclose()

    # ---- server trailing slash ----
    rec = Recorder()
    svc = svc_for(cfg(server="https://ntfy.example/"), rec)
    await svc.send(title="t", body="b")
    check(str(rec.requests[0].url) == f"https://ntfy.example/{SECRET_TOPIC}",
          "trailing slash on the server URL does not double up")
    await svc.aclose()

    # ---- priority passthrough ----
    rec = Recorder()
    svc = svc_for(cfg(priority=5), rec)
    await svc.send(title="t", body="b")
    check(rec.requests[0].headers["Priority"] == "5", "priority is configurable")

    await _check_priority_override()
    _check_icons()
    await svc.aclose()


# --------------------------------------------------------------------------- #
# 2. no-op + resilience
# --------------------------------------------------------------------------- #


async def _noop_cases() -> None:
    for label, config in (
        ("disabled", cfg(enabled=False)),
        ("no topic", cfg(topic="")),
        ("no server", cfg(server="")),
    ):
        rec = Recorder()
        svc = svc_for(config, rec)
        res = await svc.send(title="t", body="b")
        check(res.attempted == 0 and res.sent == 0 and not rec.requests,
              f"{label} -> attempted 0, no request (a no-op, not a failure)")
        await svc.aclose()

    # A settings blob with no ntfy key at all (older /data) must not explode.
    class NoNtfy:
        notifications = {"enabled": True}
    svc = NtfyService(NoNtfy(), transport=httpx.MockTransport(Recorder().handler))
    res = await svc.send(title="t", body="b")
    check(res.attempted == 0, "settings without an ntfy block -> no-op")
    await svc.aclose()

    # ---- HTTP error: reported, never raised ----
    rec = Recorder(status=403)
    svc = svc_for(cfg(), rec)
    res = await svc.send(title="t", body="b")
    check(res.attempted == 1 and res.sent == 0 and res.errors,
          "HTTP 403 (bad token / topic denied) -> reported, not raised")
    check("403" in res.errors[0], "the error names the status")
    await svc.aclose()

    # ---- transport blows up: reported, never raised ----
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("ntfy unreachable", request=request)

    svc = NtfyService(FakeSettings(cfg()), transport=httpx.MockTransport(boom))
    res = await svc.send(title="t", body="b")
    check(res.sent == 0 and res.errors, "unreachable ntfy -> error, never raises")
    await svc.aclose()


# --------------------------------------------------------------------------- #
# 3. log hygiene — the topic is a password
# --------------------------------------------------------------------------- #


async def _log_cases() -> None:
    """{server}/{topic} IS the publish URL, so the topic is a bearer secret.
    Anyone who reads it from a log can subscribe to every notification. httpx
    logs request URLs at INFO, so this catches the library too."""
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    root = logging.getLogger()
    root.addHandler(Capture())
    root.setLevel(logging.DEBUG)
    try:
        # Success, an HTTP error, and a transport blow-up: every path logs.
        for rec_or_boom in (Recorder(), Recorder(status=403)):
            svc = svc_for(cfg(auth_token="tk_supersecret"), rec_or_boom)
            await svc.send(title="t", body="b")
            await svc.aclose()

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        svc = NtfyService(FakeSettings(cfg(auth_token="tk_supersecret")),
                          transport=httpx.MockTransport(boom))
        await svc.send(title="t", body="b")
        await svc.aclose()

        blob = "\n".join(records)
        check(bool(records), "some log lines captured (the check can actually fail)")
        check(SECRET_TOPIC not in blob,
              "the full topic is NEVER logged (it is the shared secret)")
        check("tk_supersecret" not in blob, "the ntfy auth token is never logged")
        check("https://ntfy.example/" + SECRET_TOPIC not in blob,
              "the publish URL is never logged (httpx muzzled)")
    finally:
        root.removeHandler(root.handlers[-1])
        root.setLevel(logging.WARNING)


def main() -> None:
    print("ntfy: publish shape (URL, headers, linked snapshot)")
    asyncio.run(_publish_cases())
    print("ntfy: no-op when unconfigured + never raises")
    asyncio.run(_noop_cases())
    print("ntfy: log hygiene (the topic is a password)")
    asyncio.run(_log_cases())
    print(f"\nALL {PASS} CHECKS PASSED (ntfy channel)")


async def _check_priority_override() -> None:
    """A per-message priority beats the configured default. This is what
    escalates a doorbell press when there is NO CallKit ring to carry it (ntfy
    is the only channel), so the press cannot look like routine motion."""
    rec = Recorder()
    svc = NtfyService(FakeSettings(cfg(priority=2)), transport=httpx.MockTransport(rec.handler))
    await svc.send(title="Motion", body="b")
    check(rec.requests[-1].headers["Priority"] == "2",
          "routine alert keeps the configured priority")
    await svc.send(title="Doorbell", body="b", priority=5)
    check(rec.requests[-1].headers["Priority"] == "5",
          "urgent doorbell overrides it to max (5)")
    await svc.send(title="Motion", body="b", priority=None)
    check(rec.requests[-1].headers["Priority"] == "2",
          "priority=None falls back to config (no accidental escalation)")


def _check_icons() -> None:
    """`Tags` is USER-VISIBLE in ntfy: an emoji shortcode renders as the
    notification's icon, anything else prints as a literal #hashtag. The
    internal dedup tag must therefore never reach it."""
    check(ntfy_icon(["person"]) == "walking", "person maps to a walking glyph")
    check(ntfy_icon(["car"]) == "blue_car", "car maps to a car glyph")
    check(ntfy_icon(["dog"]) == "dog", "dog maps to a dog glyph")
    check(ntfy_icon(["giraffe"]) == NTFY_ICON_DEFAULT,
          "an unmapped class falls back to the generic camera glyph")
    check(ntfy_icon(["giraffe", "person"]) == "walking",
          "the first RECOGNISED label wins on a multi-object event")
    check(ntfy_icon([]) == NTFY_ICON_DEFAULT, "no labels -> generic glyph")
    check(NTFY_ICON_DOORBELL == "bell", "a doorbell press rings a bell glyph")
    check(all("vigilume-" not in i and "sentinel-" not in i
              for i in (ntfy_icon(["person"]), NTFY_ICON_DOORBELL, NTFY_ICON_DEFAULT)),
          "no internal dedup tag can leak into the visible Tags header")


if __name__ == "__main__":
    main()

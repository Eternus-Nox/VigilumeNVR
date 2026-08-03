"""Smoke suite for APNs (iOS) push + the hide-boxes setting.

Pinned contract: docs/push-architecture.md. Coverage:

  - db migration v7: apns_devices table, upsert (latest key wins), idempotent
    delete, list order/shape
  - POST /api/notifications/apns/register: contract validation (400s: bad hex
    token, bad base64, wrong key length, bad environment), lowercased storage,
    device_name cap, ANY authenticated role (viewer registers too), 401 unauth
  - DELETE .../apns/register idempotent 204; GET .../apns/devices returns
    8-char prefixes only (never the full token)
  - settings.notifications.apns pydantic block (mode/relay_url) + draw_boxes
    default true; the retired "direct" mode is rejected and migrated away
  - encryption round-trip: payload_b64 = base64(nonce||ct||tag) decrypts with
    a python AESGCM using the registered key; plaintext JSON shape; 2500-byte
    body truncation
  - relay mode (httpx MockTransport): POST {relay_url}/api/push request shape,
    priority/collapse_id, `environment` sent ONLY when the row has one (a NULL
    row must fall back to the relay's APNS_ENV, not be forced to a host from
    here), 410 prunes the row, 400 bad_device_token prunes, other 400/429/
    relay-down do NOT prune, never raises
  - THE REASON VOCABULARY: every reason on the wire is the RELAY's snake_case
    (relay/main.py `_err`), never Apple's CamelCase. A prune keyed on
    "BadDeviceToken" compiles fine and silently never fires — so the prune
    cases below assert the snake_case strings deliberately.
  - pipeline: APNs fires alongside web push under the SAME cooldown gate,
    same media-token snapshot URL as the web-push image; a slow/hung relay
    never stalls the notify caller (APNs isolated on its own pipeline task)
  - draw_boxes=false: snapshot keeps the count banner but has NO box pixels
    at the detection region (annotate + full pipeline); missing key = true

CPU-only, no network. Usage: python backend/tests/apns_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Clean env before app config is instantiated (mirrors rbac_smoke.py).
for _i in (1, 2, 3):
    for _sfx in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{_i}_{_sfx}", None)
os.environ["ADMIN_PASSWORD"] = "admin-secret"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-apns-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from app import annotate  # noqa: E402
from app.auth import AuthService  # noqa: E402
from app.config import DEFAULT_SETTINGS  # noqa: E402
from app.db import Database  # noqa: E402
from app.events_pipeline import EventsPipeline  # noqa: E402
from app.notify import apns as apns_mod  # noqa: E402
from app.notify.apns import (  # noqa: E402
    ApnsService,
    build_plaintext,
    encrypt_payload,
)
from app.notify.push import PushSendResult  # noqa: E402
from app.settings_store import SettingsStore  # noqa: E402
from app.ws import WSManager  # noqa: E402

PASS = 0
BG = (114, 114, 114)  # uniform detect-frame background

# Shrink the transient-retry backoff so retry paths don't slow the suite.
apns_mod._RETRY_BACKOFF_S = 0.01

TOKEN_A = "a1" * 32  # 64 hex chars
TOKEN_B = "b2" * 32
KEY_A = bytes(range(32))
KEY_B = bytes(range(32, 64))
KEY_A_B64 = base64.b64encode(KEY_A).decode()
KEY_B_B64 = base64.b64encode(KEY_B).decode()


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}", flush=True)
        # Hard exit: a sys.exit inside asyncio.run leaves aiosqlite worker
        # threads alive and hangs interpreter shutdown.
        os._exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class FakeSettings:
    """Minimal SettingsStore stand-in for the ApnsService unit tests."""

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False

    def __init__(self, apns_cfg: dict, ntfy_cfg: dict | None = None):
        self.notifications = {"enabled": True, "apns": apns_cfg}
        if ntfy_cfg is not None:
            self.notifications["ntfy"] = ntfy_cfg


RELAY_URL = "https://relay.test"


def relay_cfg(**over) -> dict:
    """A valid apns "relay" config — the only APNs transport. There is no p8
    and no keypair anywhere in this suite by design: the backend signs nothing,
    the .p8 lives only in relay/."""
    cfg = {"mode": "relay", "relay_url": RELAY_URL}
    cfg.update(over)
    return cfg


class ApnsRecorder:
    """Captures the relay requests the service makes, and replays canned
    responses. `responses` is a list of (status, json); the last one repeats."""

    def __init__(self) -> None:
        self.requests: list[tuple[httpx.Request, dict]] = []
        self.responses: list[tuple[int, dict]] = [(200, {"ok": True})]

    def handler(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        self.requests.append((request, body))
        status, payload = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        return httpx.Response(status, json=payload)


def decrypt_combined(key: bytes, payload_b64: str) -> dict:
    """The python twin of CryptoKit's AES.GCM.open(SealedBox(combined:))."""
    blob = base64.b64decode(payload_b64)
    nonce, rest = blob[:12], blob[12:]
    plaintext = AESGCM(key).decrypt(nonce, rest, None)
    return json.loads(plaintext.decode("utf-8"))


async def new_db(name: str) -> Database:
    db = Database(TMP / name / "nvr.db")
    await db.connect()
    return db


def _uniform_jpeg(w: int, h: int) -> bytes:
    frame = np.full((h, w, 3), BG, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(buf)


def _changed_mask(region: np.ndarray) -> np.ndarray:
    diff = np.abs(region.astype(np.int32) - np.array(BG, dtype=np.int32))
    return np.any(diff > 30, axis=2)


def _frame_region(result: np.ndarray, h: int) -> np.ndarray:
    return result[result.shape[0] - h :, :, :]


# --------------------------------------------------------------------------- #
# 1. DB: migration + upsert/delete/list
# --------------------------------------------------------------------------- #


async def _db_cases() -> None:
    db = await new_db("dbunit")
    cur = await db.conn.execute("PRAGMA user_version")
    check((await cur.fetchone())[0] >= 7, "schema at v7+ (apns_devices migration)")

    await db.upsert_apns_device(TOKEN_A, "Adam's iPhone", KEY_A_B64, "production")
    rows = await db.list_apns_devices()
    check(len(rows) == 1 and rows[0]["device_token"] == TOKEN_A, "device row inserted")
    check(rows[0]["key_b64"] == KEY_A_B64 and rows[0]["environment"] == "production",
          "row stores key_b64 + environment")

    # Re-register: latest key/name/env win, still one row.
    await db.upsert_apns_device(TOKEN_A, "Renamed", KEY_B_B64, "sandbox")
    rows = await db.list_apns_devices()
    check(len(rows) == 1 and rows[0]["key_b64"] == KEY_B_B64
          and rows[0]["device_name"] == "Renamed" and rows[0]["environment"] == "sandbox",
          "re-register upserts (latest key/name/environment win)")

    check(await db.delete_apns_device(TOKEN_A) is True, "delete removes the row")
    check(await db.delete_apns_device(TOKEN_A) is False, "second delete is a no-op (idempotent)")
    check(await db.list_apns_devices() == [], "list empty after delete")
    await db.close()


def db_checks() -> None:
    print("db: apns_devices migration + upsert/delete/list")
    asyncio.run(_db_cases())


# --------------------------------------------------------------------------- #
# 2. settings models
# --------------------------------------------------------------------------- #


def settings_model_checks() -> None:
    print("settings: apns block + draw_boxes (pydantic round-trip)")
    from app.routers.settings import (
        ApnsSettings, AppSettings, NotificationSettings, NtfySettings,
    )

    n = NotificationSettings()
    check(n.draw_boxes is True, "draw_boxes defaults true")
    # Default "off", NOT "relay": mode="relay" with an empty relay_url would
    # error on every event, and a baked-in owner URL would make a fresh install
    # push through someone else's Apple credentials the moment a phone
    # registers. The admin opts in.
    check(n.apns.mode == "off", "apns defaults to mode 'off' (admin opts in)")
    check(n.apns.relay_url == "", "apns.relay_url defaults empty (no baked-in owner URL)")
    check(not hasattr(n.apns, "direct"),
          "ApnsSettings no longer has a `direct` block (this server signs nothing)")

    check(DEFAULT_SETTINGS["notifications"]["draw_boxes"] is True
          and DEFAULT_SETTINGS["notifications"]["apns"]["mode"] == "off"
          and DEFAULT_SETTINGS["notifications"]["apns"]["relay_url"] == "",
          "DEFAULT_SETTINGS carries the same defaults (store merge source)")
    # update() deep-merges over DEFAULT_SETTINGS, so a `direct` key here would
    # stamp a dead block into every saved document.
    check("direct" not in DEFAULT_SETTINGS["notifications"]["apns"],
          "DEFAULT_SETTINGS has NO `direct` key (it would be merged into every save)")

    bad = False
    try:
        ApnsSettings(mode="carrier-pigeon")
    except Exception:
        bad = True
    check(bad, "unknown apns mode rejected (422-shaped)")

    bad = False
    try:
        ApnsSettings(mode="direct")
    except Exception:
        bad = True
    check(bad, "the retired 'direct' mode is rejected by the model")
    check(ApnsSettings(mode="relay", relay_url=RELAY_URL).mode == "relay",
          "'relay' is accepted (the only APNs transport)")

    # ---- relay_url validator ----
    bad = False
    try:
        ApnsSettings(mode="relay", relay_url="ftp://nope")
    except Exception:
        bad = True
    check(bad, "non-http(s) relay_url rejected")
    check(ApnsSettings(relay_url="https://r.example/").relay_url == "https://r.example",
          "relay_url trailing slash stripped")
    # THE one config that actually works for the owner: the compose service name
    # on the shared default network. A validator demanding https or a public TLD
    # would lock him out of it.
    check(ApnsSettings(mode="relay", relay_url="http://push-relay:8090").relay_url
          == "http://push-relay:8090",
          "http://push-relay:8090 accepted (the compose service name, plain http)")

    # ---- the direct migration (settings_store._strip_legacy) ----
    # Because `mode` is a Literal, a stored "direct" MUST be migrated before it
    # can reach the validator above — otherwise every PUT/PATCH /api/settings
    # 422s and the admin is locked out of the settings page entirely,
    # including out of changing the mode.
    from app.settings_store import _strip_legacy  # noqa: PLC0415

    migrated = _strip_legacy({"notifications": {"apns": {
        "mode": "direct", "relay_url": "",
        "direct": {"key_id": "K", "team_id": "T", "bundle_id": "b", "p8": "PEM"},
    }}})["notifications"]["apns"]
    check(migrated["mode"] == "off",
          "stored mode='direct' -> 'off' (never silently -> 'relay')")
    check("direct" not in migrated,
          "the dead `direct` block (and the p8 it held) is dropped")

    # A stored relay config is a LIVE value, not legacy — it must survive intact.
    kept_relay = _strip_legacy({"notifications": {"apns": {
        "mode": "relay", "relay_url": "http://push-relay:8090",
    }}})["notifications"]["apns"]
    check(kept_relay == {"mode": "relay", "relay_url": "http://push-relay:8090"},
          "a stored 'relay' config is untouched by the migration")

    # The migrated value must actually satisfy the model it's protecting.
    check(ApnsSettings(**migrated).mode == "off",
          "a migrated blob validates (no 422 settings lockout)")

    # ntfy must NOT be stripped: it was popped as legacy while ntfy support
    # was removed, which would silently delete the channel's config on save.
    kept = _strip_legacy({"notifications": {"ntfy": {"enabled": True, "topic": "abc"}}})
    check(kept["notifications"]["ntfy"]["topic"] == "abc",
          "_strip_legacy PRESERVES the ntfy block (no longer popped as legacy)")

    # ---- ntfy: push with no Apple account (what replaced the relay) ----
    check(n.ntfy.enabled is False, "ntfy defaults disabled")
    check(n.ntfy.server == "https://ntfy.sh", "ntfy defaults to the public server")
    # SECURITY: the topic is a bearer secret — on a default-allow server
    # (ntfy.sh included) anyone who knows it reads every message. So there is
    # NO default topic; the UI must generate an unguessable one.
    check(n.ntfy.topic == "", "ntfy has NO default topic (it is a password, not a name)")
    check(n.ntfy.attach_snapshot is True, "ntfy attaches the snapshot by default")

    bad = False
    try:
        NtfySettings(server="ftp://nope")
    except Exception:
        bad = True
    check(bad, "non-http(s) ntfy server rejected")
    check(NtfySettings(server="https://n.example/").server == "https://n.example",
          "ntfy server trailing slash stripped")

    # The topic goes straight into the publish URL, so it must stay ONE path
    # segment — no slashes, no query, no traversal.
    for evil in ("a/b", "../admin", "t?x=1", "t#frag", "a b", "t;drop"):
        bad = False
        try:
            NtfySettings(topic=evil)
        except Exception:
            bad = True
        check(bad, f"ntfy topic {evil!r} rejected (cannot smuggle a path/query into the URL)")
    check(NtfySettings(topic="vigilume_a9f3-XY").topic == "vigilume_a9f3-XY",
          "ntfy topic allows A-Z a-z 0-9 _ -")

    for bad_prio in (0, 6):
        bad = False
        try:
            NtfySettings(priority=bad_prio)
        except Exception:
            bad = True
        check(bad, f"ntfy priority {bad_prio} rejected (ntfy scale is 1..5)")

    dumped = AppSettings().model_dump()
    check(dumped["notifications"]["apns"] == {"mode": "off", "relay_url": ""}
          and "draw_boxes" in dumped["notifications"],
          "AppSettings round-trips apns + draw_boxes through PUT /api/settings")

    full = AppSettings(**{"notifications": {"apns": {
        "mode": "relay", "relay_url": "http://push-relay:8090/",
    }, "draw_boxes": False}}).model_dump()
    apns_out = full["notifications"]["apns"]
    check(apns_out == {"mode": "relay", "relay_url": "http://push-relay:8090"},
          "apns relay config round-trips (slash stripped, no stray direct block)")
    check(full["notifications"]["draw_boxes"] is False, "draw_boxes=false round-trips")


# --------------------------------------------------------------------------- #
# 3. encryption
# --------------------------------------------------------------------------- #


def encryption_checks() -> None:
    print("crypto: AES-256-GCM combined layout + plaintext shape + size budget")
    plaintext = build_plaintext("Person detected", "1 person in frame", "42",
                                "https://nvr.example/api/events/42/snapshot.jpg?token=x")
    payload_b64 = encrypt_payload(KEY_A, plaintext)
    blob = base64.b64decode(payload_b64)
    check(len(blob) == 12 + len(plaintext) + 16,
          "wire blob is exactly nonce(12) + ct(len) + tag(16)")
    decrypted = decrypt_combined(KEY_A, payload_b64)
    check(decrypted == {"title": "Person detected", "body": "1 person in frame",
                        "event_id": "42",
                        "snapshot_url": "https://nvr.example/api/events/42/snapshot.jpg?token=x"},
          "round-trip decrypt yields the exact contract JSON")
    check(list(decrypted.keys()) == ["title", "body", "event_id", "snapshot_url"],
          "plaintext carries exactly title/body/event_id/snapshot_url when camera omitted")

    # ----- per-camera grouping fields survive build + encrypt + AES-GCM round-trip -----
    cam_plain = build_plaintext("Person detected at Backyard", "1 person in frame", "99",
                                None, "backyard", "Backyard")
    cam_dec = decrypt_combined(KEY_A, encrypt_payload(KEY_A, cam_plain))
    check(cam_dec["camera"] == "backyard" and cam_dec["camera_label"] == "Backyard",
          "camera + camera_label present in built+encrypted plaintext (survive AES-GCM round-trip)")
    check(cam_dec["title"] == "Person detected at Backyard" and cam_dec["event_id"] == "99"
          and cam_dec["snapshot_url"] is None,
          "grouping fields ride alongside the existing title/event_id/snapshot_url")
    # back-compat: absent camera -> keys omitted entirely (older-extension safe)
    check("camera" not in json.loads(build_plaintext("t", "b", "1", None))
          and "camera_label" not in json.loads(build_plaintext("t", "b", "1", None)),
          "camera/camera_label omitted when not supplied (back-compat wire shape)")
    # only one of the two supplied -> only that key appears
    only_slug = json.loads(build_plaintext("t", "b", "1", None, camera="frontdoor"))
    check(only_slug.get("camera") == "frontdoor" and "camera_label" not in only_slug,
          "camera can appear without camera_label")

    # fresh nonce per message -> distinct ciphertexts for the same plaintext
    check(encrypt_payload(KEY_A, plaintext) != payload_b64, "fresh random nonce per message")

    # wrong key must fail (proves it is really AES-GCM with the device key)
    failed = False
    try:
        decrypt_combined(KEY_B, payload_b64)
    except Exception:
        failed = True
    check(failed, "wrong key fails authentication (GCM tag verified)")

    # snapshot_url null when there is no image
    no_img = json.loads(build_plaintext("t", "b", "7", None))
    check(no_img["snapshot_url"] is None, "snapshot_url is null when no image")

    long = build_plaintext("t", "x" * 10_000, "7", None)
    check(len(long) <= 2500, "oversized body truncated to keep plaintext <= 2500 bytes")
    parsed = json.loads(long)
    check(parsed["title"] == "t" and parsed["event_id"] == "7" and parsed["body"].startswith("xxx"),
          "truncation trims body only (title/event_id intact)")


# --------------------------------------------------------------------------- #
# 4. relay mode (MockTransport) — the only APNs transport
# --------------------------------------------------------------------------- #


async def _relay_cases() -> None:
    db = await new_db("relay")
    await db.upsert_apns_device(TOKEN_A, "prod phone", KEY_A_B64, "production")
    await db.upsert_apns_device(TOKEN_B, "dev phone", KEY_B_B64, "sandbox")

    cfg = relay_cfg()
    rec = ApnsRecorder()
    svc = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(rec.handler))

    res = await svc.send_to_all(title="Dog detected at Yard", body="1 dog in frame",
                                event_id="77", snapshot_url=None, priority="high",
                                collapse_id="77")
    check(res.attempted == 2 and res.sent == 2, "relay mode sent to both devices")
    check(all(str(r.url) == "https://relay.test/api/push" for r, _ in rec.requests),
          "POST {relay_url}/api/push — ONE URL for every device (the relay routes the host)")
    by_token = {b["device_token"]: b for _, b in rec.requests}
    check(set(by_token) == {TOKEN_A, TOKEN_B}, "one request per registered device token")

    prod = by_token[TOKEN_A]
    check(prod["priority"] == "high" and prod["collapse_id"] == "77",
          "body carries priority + collapse_id (contract §3 fields)")
    check(prod["environment"] == "production" and by_token[TOKEN_B]["environment"] == "sandbox",
          "each device's environment rides in the body (the relay routes the host per request)")
    check("enc" not in prod and "aps" not in prod,
          "no `aps`/`enc` wrapping here — building the APNs payload is the RELAY's job")
    check(decrypt_combined(KEY_A, prod["payload_b64"])["title"] == "Dog detected at Yard",
          "payload_b64 decrypts with the production device's key (E2E round-trip)")
    check(decrypt_combined(KEY_B, by_token[TOKEN_B]["payload_b64"])["title"] == "Dog detected at Yard",
          "each device encrypted with ITS OWN registered key")

    # ----- no Apple credential ever leaves this process -----
    check(all("authorization" not in r.headers for r, _ in rec.requests),
          "no authorization header — the backend signs NOTHING (the .p8 lives in the relay)")

    # ----- priority: normal passes through; garbage is coerced to high -----
    rec.requests.clear()
    await svc.send_to_all(title="t", body="b", event_id="78", snapshot_url=None,
                          priority="normal")
    check(all(b["priority"] == "normal" for _, b in rec.requests), "priority='normal' forwarded")
    rec.requests.clear()
    await svc.send_to_all(title="t", body="b", event_id="78", snapshot_url=None,
                          priority="URGENT!!")
    check(all(b["priority"] == "high" for _, b in rec.requests),
          "an out-of-vocabulary priority is coerced to 'high' (relay 400s bad_priority)")
    # collapse_id omitted entirely when absent (the relay treats absent != empty)
    rec.requests.clear()
    await svc.send_to_all(title="t", body="b", event_id="79", snapshot_url=None)
    check(all("collapse_id" not in b for _, b in rec.requests),
          "collapse_id omitted from the body when not supplied")

    # ----- `environment` is sent ONLY when the row has one -----
    # THE FIX: the old relay pinned one APNs host at startup and silently sent a
    # sandbox dev-build token to production; Apple said BadDeviceToken and every
    # push vanished. A row with no environment must fall back to the relay's
    # APNS_ENV, NOT be forced to a host from here.
    # NB the apns_devices schema is `environment TEXT NOT NULL DEFAULT
    # 'production' CHECK(...)` at v7 AND today, so a NULL row cannot be created
    # through the DB — this guard is exercised at the _send_relay seam directly.
    rec.requests.clear()
    out = await svc._send_relay(cfg, {"device_token": TOKEN_A}, "cGF5", "high", None)
    check(out[0] == "ok" and "environment" not in rec.requests[0][1],
          "a device row with NO environment -> the key is OMITTED (relay falls back to APNS_ENV)")
    rec.requests.clear()
    await svc._send_relay(cfg, {"device_token": TOKEN_A, "environment": None}, "cGF5", "high", None)
    check("environment" not in rec.requests[0][1],
          "environment=None is omitted too (never sent as a null the relay would reject)")

    # ----- 410 unregistered prunes (on the STATUS, not the reason) -----
    # The relay's own 410 branch: `return _err(410, "unregistered")`.
    def status_handler(status: int, payload: dict):
        def h(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}")
            # only the sandbox device gets the failure
            if body.get("device_token") == TOKEN_B:
                return httpx.Response(status, json=payload)
            return httpx.Response(200, json={"ok": True})
        return h

    svc_410 = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        status_handler(410, {"ok": False, "reason": "unregistered"})))
    res = await svc_410.send_to_all(title="t", body="b", event_id="80", snapshot_url=None)
    tokens = {d["device_token"] for d in await db.list_apns_devices()}
    check(res.pruned == 1 and tokens == {TOKEN_A}, "relay 410 pruned the sandbox device only")

    # ----- 400 bad_device_token prunes — THE REASON-VOCABULARY TRAP -----
    # relay/main.py: `return _err(400, "bad_device_token")`. snake_case, NOT
    # Apple's "BadDeviceToken" — the relay collapses Apple's CamelCase into its
    # own closed vocabulary before the backend ever sees it.
    svc_bad = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"ok": False, "reason": "bad_device_token"})))
    res = await svc_bad.send_to_all(title="t", body="b", event_id="81", snapshot_url=None)
    check(res.pruned == 1 and await db.list_apns_devices() == [],
          "relay 400 'bad_device_token' (snake_case) pruned the registration")

    # The inverse, pinning the vocabulary: Apple's CamelCase is NOT the relay's
    # and must NOT prune. If this ever passes-by-pruning, someone compared
    # against Apple's string and every real dead token now leaks instead.
    await db.upsert_apns_device(TOKEN_A, "x", KEY_A_B64, "production")
    svc_camel = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"ok": False, "reason": "BadDeviceToken"})))
    res = await svc_camel.send_to_all(title="t", body="b", event_id="82", snapshot_url=None)
    check(res.pruned == 0 and len(await db.list_apns_devices()) == 1,
          "Apple's CamelCase 'BadDeviceToken' is NOT the relay's vocabulary -> no prune")

    # ----- other 400s do NOT prune -----
    svc_400 = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"ok": False, "reason": "bad_payload"})))
    res = await svc_400.send_to_all(title="t", body="b", event_id="83", snapshot_url=None)
    check(res.pruned == 0 and res.errors and len(await db.list_apns_devices()) == 1,
          "400 bad_payload errors but does NOT prune (row kept)")

    # ----- 429 is returned immediately: no retry-storm into the relay's limiter -----
    n429 = {"n": 0}

    def h429(request: httpx.Request) -> httpx.Response:
        n429["n"] += 1
        return httpx.Response(429, json={"ok": False, "reason": "rate_limited"})

    svc_429 = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(h429))
    res = await svc_429.send_to_all(title="t", body="b", event_id="84", snapshot_url=None)
    check(n429["n"] == 1, "429 is NOT retried (one request, no retry-storm — contract §3)")
    check(res.pruned == 0 and res.errors and len(await db.list_apns_devices()) == 1,
          "429 errors without pruning")

    # ----- 502 apns_auth: a healthy relay whose KEY is wrong. Not transient,
    # but it is a 5xx, so it burns the full retry budget per device per event. -----
    nauth = {"n": 0}

    def hauth(request: httpx.Request) -> httpx.Response:
        nauth["n"] += 1
        return httpx.Response(502, json={"ok": False, "reason": "apns_auth"})

    svc_auth = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(hauth))
    res = await svc_auth.send_to_all(title="t", body="b", event_id="85", snapshot_url=None)
    check(nauth["n"] == apns_mod._RETRY_ATTEMPTS,
          f"502 apns_auth burns all {apns_mod._RETRY_ATTEMPTS} attempts (5xx; logged distinctly)")
    check(res.pruned == 0 and any("apns_auth" in e for e in res.errors),
          "502 apns_auth surfaces the reason and does NOT prune")

    # ----- a Cloudflare tunnel with a DEAD ORIGIN answers HTML, not JSON -----
    # 530 is >= 500 so it retries, and _reason_of returns "" -> generic error.
    # This path is correct as-is; it must not raise.
    nhtml = {"n": 0}

    def hhtml(request: httpx.Request) -> httpx.Response:
        nhtml["n"] += 1
        return httpx.Response(530, html="<html><body>error 1033</body></html>")

    svc_html = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(hhtml))
    res = await svc_html.send_to_all(title="t", body="b", event_id="86", snapshot_url=None)
    check(nhtml["n"] == apns_mod._RETRY_ATTEMPTS, "530 HTML retried like any 5xx")
    check(res.sent == 0 and res.pruned == 0 and res.errors
          and len(await db.list_apns_devices()) == 1,
          "an HTML 530 (dead tunnel origin) -> generic error, never raises, never prunes")

    # ----- the RELAY itself unreachable (tunnel down) -> resp is None -----
    ndown = {"n": 0}

    def hdown(request: httpx.Request) -> httpx.Response:
        ndown["n"] += 1
        raise httpx.ConnectError("relay down")

    svc_down = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(hdown))
    res = await svc_down.send_to_all(title="t", body="b", event_id="87", snapshot_url=None)
    check(ndown["n"] == apns_mod._RETRY_ATTEMPTS, "an unreachable relay is retried")
    check(res.sent == 0 and res.pruned == 0
          and any("relay unreachable" in e for e in res.errors)
          and len(await db.list_apns_devices()) == 1,
          "relay unreachable (NOT 'apns unreachable' — different runbook page); no prune")

    # ----- mode=relay with no relay_url -> clean error, no crash, no prune -----
    svc_nourl = ApnsService(db, FakeSettings({"mode": "relay", "relay_url": ""}),
                            transport=httpx.MockTransport(rec.handler))
    res = await svc_nourl.send_to_all(title="t", body="b", event_id="88", snapshot_url=None)
    check(res.sent == 0 and any("relay_url" in e for e in res.errors)
          and len(await db.list_apns_devices()) == 1,
          "unconfigured relay mode errors cleanly (row kept)")

    # ----- mode off / retired direct -> cheap no-op, nothing sent -----
    rec.requests.clear()
    svc_off = ApnsService(db, FakeSettings(relay_cfg(mode="off")),
                          transport=httpx.MockTransport(rec.handler))
    res = await svc_off.send_to_all(title="t", body="b", event_id="89", snapshot_url=None)
    check(res.attempted == 0 and not rec.requests, "apns mode off -> no-op, no request")
    svc_dead = ApnsService(db, FakeSettings(relay_cfg(mode="direct")),
                           transport=httpx.MockTransport(rec.handler))
    res = await svc_dead.send_to_all(title="t", body="b", event_id="90", snapshot_url=None)
    check(res.attempted == 0 and not rec.requests,
          "a stale mode='direct' that dodged the migration -> no-op (never a relay send)")

    for s in (svc, svc_410, svc_bad, svc_camel, svc_400, svc_429, svc_auth,
              svc_html, svc_down, svc_nourl, svc_off, svc_dead):
        await s.aclose()
    await db.close()


def relay_checks() -> None:
    print("relay mode: /api/push shape, environment, reason vocabulary, prune, retries")
    asyncio.run(_relay_cases())


# --------------------------------------------------------------------------- #
# 5b. VoIP (PushKit) — CallKit doorbell ring
# --------------------------------------------------------------------------- #


async def _voip_cases() -> None:
    db = await new_db("voip")

    # ----- DB: upsert/list/delete (no key column) -----
    await db.upsert_voip_device(TOKEN_A, "Adam's iPhone", "production")
    rows = await db.list_voip_devices()
    check(len(rows) == 1 and rows[0]["device_token"] == TOKEN_A
          and rows[0]["environment"] == "production" and "key_b64" not in rows[0],
          "voip device row stored (no encryption key)")
    await db.upsert_voip_device(TOKEN_A, "Renamed", "sandbox")
    rows = await db.list_voip_devices()
    check(len(rows) == 1 and rows[0]["device_name"] == "Renamed"
          and rows[0]["environment"] == "sandbox", "voip re-register upserts (latest name/env win)")

    # ----- relay mode: POST {relay_url}/api/push/voip -----
    await db.upsert_voip_device(TOKEN_A, "iPhone", "production")
    await db.upsert_voip_device(TOKEN_B, "dev", "sandbox")
    cfg = relay_cfg()
    rec = ApnsRecorder()

    svc_r = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(rec.handler))
    res = await svc_r.send_voip_to_all(camera="Front Door", event_id="77")
    check(res.attempted == 2 and res.sent == 2, "voip relay sent to both devices")
    check(all(str(r.url) == "https://relay.test/api/push/voip" for r, _ in rec.requests),
          "voip goes to the SEPARATE {relay_url}/api/push/voip endpoint")
    by_token = {b["device_token"]: b for _, b in rec.requests}
    prod = by_token[TOKEN_A]
    # The topic (<bundle>.voip), push-type and expiration are the RELAY's job —
    # this backend knows no bundle id at all any more.
    check(prod["payload"] == {"type": "doorbell", "camera": "Front Door", "event_id": "77"},
          "voip body carries the minimal doorbell payload")
    check("payload_b64" not in prod and "enc" not in prod,
          "voip payload is PLAINTEXT, not encrypted (CallKit must report immediately)")
    check(prod["environment"] == "production" and by_token[TOKEN_B]["environment"] == "sandbox",
          "voip environment rides per request too (relay routes the host)")

    # environment omitted when the row has none (same fallback fix as /api/push)
    rec.requests.clear()
    await svc_r._send_voip_relay(cfg, {"device_token": TOKEN_A}, {"type": "doorbell"})
    check("environment" not in rec.requests[0][1],
          "voip: no environment on the row -> key omitted (relay falls back to APNS_ENV)")

    # ----- prune keys on the RELAY's vocabulary here too -----
    svc_v410 = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(410, json={"ok": False, "reason": "unregistered"})
        if json.loads(r.content)["device_token"] == TOKEN_B
        else httpx.Response(200, json={"ok": True})))
    res = await svc_v410.send_voip_to_all(camera="Front Door", event_id="78")
    toks = {d["device_token"] for d in await db.list_voip_devices()}
    check(res.pruned == 1 and toks == {TOKEN_A}, "voip 410 pruned the sandbox token only")

    svc_vbad = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"ok": False, "reason": "bad_device_token"})))
    res = await svc_vbad.send_voip_to_all(camera="Front Door", event_id="79")
    check(res.pruned == 1 and await db.list_voip_devices() == [],
          "voip 400 'bad_device_token' (snake_case) pruned the registration")

    await db.upsert_voip_device(TOKEN_A, "iPhone", "production")
    svc_vcamel = ApnsService(db, FakeSettings(cfg), transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"ok": False, "reason": "BadDeviceToken"})))
    res = await svc_vcamel.send_voip_to_all(camera="Front Door", event_id="80")
    check(res.pruned == 0 and len(await db.list_voip_devices()) == 1,
          "voip: Apple's CamelCase is NOT the relay's vocabulary -> no prune")

    # ----- no devices / mode off -> no-op -----
    await db.delete_voip_device(TOKEN_A)
    res = await svc_r.send_voip_to_all(camera="Front Door", event_id="81")
    check(res.attempted == 0 and res.sent == 0, "no voip tokens -> no-op")
    await db.upsert_voip_device(TOKEN_A, "iPhone", "production")
    svc_off = ApnsService(db, FakeSettings(relay_cfg(mode="off")),
                          transport=httpx.MockTransport(rec.handler))
    res = await svc_off.send_voip_to_all(camera="Front Door", event_id="82")
    check(res.attempted == 0, "apns mode off -> voip is a cheap no-op")

    for s in (svc_r, svc_v410, svc_vbad, svc_vcamel, svc_off):
        await s.aclose()
    await db.close()


def voip_checks() -> None:
    print("voip: PushKit CallKit ring — relay request shape, prune, no-op")
    asyncio.run(_voip_cases())


# --------------------------------------------------------------------------- #
# 6. pipeline: shared cooldown + media-token snapshot URL + draw_boxes
# --------------------------------------------------------------------------- #


class FakePush:
    public_key = "fake"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def send_to_all(self, payload: dict) -> PushSendResult:
        self.payloads.append(payload)
        return PushSendResult(attempted=1, sent=1)


class FakeMedia:
    def __init__(self, jpeg: bytes) -> None:
        self._jpeg = jpeg

    async def event_snapshot(self, fid: str, retries: int = 1):
        return self._jpeg

    async def detect_dims(self, camera: str):
        return None


def native_payload(fid: str, camera: str = "cam", label: str = "person",
                   box=None) -> dict:
    now = time.time()
    box = box or [50, 50, 150, 300]
    return {
        "type": "new",
        "after": {
            "id": fid, "camera": camera, "label": label, "top_score": 0.95,
            "start_time": now, "has_snapshot": True,
            "snapshot": {"box": box, "frame_time": now, "score": 0.95},
        },
    }


async def _wait(pred, timeout=8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


async def _pipeline_cases() -> None:
    root = TMP / "pipeline"
    db = Database(root / "nvr.db")
    await db.connect()
    settings = SettingsStore(db)
    await settings.load()
    await settings.update({
        "system": {"public_url": "https://nvr.example"},
        "notifications": {"apns": relay_cfg()},
    })
    await db.upsert_apns_device(TOKEN_A, "iPhone", KEY_A_B64, "production")

    rec = ApnsRecorder()
    rec.responses = [(200, {"ok": True})]
    apns = ApnsService(db, settings, transport=httpx.MockTransport(rec.handler))
    push = FakePush()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    media = FakeMedia(_uniform_jpeg(640, 480))
    pipeline = EventsPipeline(db, media, WSManager(), push, settings, auth,  # type: ignore[arg-type]
                              root / "snapshots", apns=apns)

    await pipeline.handle_event(native_payload("native.1"))
    ok = await _wait(lambda: len(push.payloads) == 1 and len(rec.requests) == 1)
    check(ok, "event -> ONE web push AND ONE apns send")

    web = push.payloads[0]
    check(str(rec.requests[0][0].url) == "https://relay.test/api/push",
          "the pipeline's apns send goes to {relay_url}/api/push")
    # The relay is BLIND to all of this: it only ever sees payload_b64.
    sent = decrypt_combined(KEY_A, rec.requests[0][1]["payload_b64"])
    check(sent["snapshot_url"] == web.get("image"),
          "apns snapshot_url == web-push image URL (same media token)")
    check(sent["snapshot_url"].startswith("https://nvr.example/api/events/")
          and "?token=" in sent["snapshot_url"],
          "snapshot_url is the tokened public media URL")
    check(sent["title"] == web["title"] and sent["body"] == web["body"],
          "apns title/body match the web push")
    # camera slug rides inside the ciphertext for per-camera grouping; no
    # camera row inserted here, so the friendly label falls back to the slug.
    check(sent["camera"] == "cam" and sent["camera_label"] == "cam",
          "pipeline puts the event's camera + friendly label inside the encrypted payload")
    row = await db.get_event_by_frigate_id("native.1")
    # A BODY field, not a header: the relay maps it onto apns-collapse-id.
    check(rec.requests[0][1]["collapse_id"] == str(row["id"]),
          "relay body collapse_id = event id")

    # ----- SAME cooldown gates both transports -----
    await pipeline.handle_event(native_payload("native.2"))
    await asyncio.sleep(0.4)
    check(len(push.payloads) == 1 and len(rec.requests) == 1,
          "second event within cooldown -> NO web push and NO apns (shared gate)")

    # different label -> its own cooldown key, both transports fire again
    await pipeline.handle_event(native_payload("native.3", label="dog"))
    ok = await _wait(lambda: len(push.payloads) == 2 and len(rec.requests) == 2)
    check(ok, "different label passes the shared gate on BOTH transports")

    # label not in the enabled list -> neither transport fires
    await pipeline.handle_event(native_payload("native.4", label="giraffe"))
    await asyncio.sleep(0.4)
    check(len(push.payloads) == 2 and len(rec.requests) == 2,
          "label filter gates apns exactly like web push")

    # ----- draw_boxes default: box pixels ARE drawn -----
    row1 = await db.get_event_by_frigate_id("native.1")
    snap1 = root / "snapshots" / f"{row1['id']}.jpg"
    check(snap1.is_file(), "annotated snapshot written")
    img = cv2.imread(str(snap1))
    mask = _changed_mask(_frame_region(img, 480))
    check(bool(mask[50:301, 50:151].any()), "draw_boxes default true -> box pixels present")

    # ----- draw_boxes=false: banner stays, box region clean -----
    # Re-send the apns block: SettingsStore.update merges over DEFAULT_SETTINGS,
    # NOT over the current settings, so anything omitted here reverts to its
    # default — and the apns default is now mode="off", which would silently
    # disable every APNs assertion below this point.
    await settings.update({
        "system": {"public_url": "https://nvr.example"},
        "notifications": {"draw_boxes": False, "apns": relay_cfg()},
    })
    check(settings.notifications["draw_boxes"] is False, "draw_boxes=false persisted")
    pipeline._cooldowns.clear()
    await pipeline.handle_event(native_payload("native.5", camera="cam2"))
    row5_holder: dict = {}

    async def _snap5_ready() -> bool:
        row5 = await db.get_event_by_frigate_id("native.5")
        if row5 and row5["has_snapshot"]:
            row5_holder["row"] = row5
            return True
        return False

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and not await _snap5_ready():
        await asyncio.sleep(0.05)
    check("row" in row5_holder, "draw_boxes=false event still produced a snapshot")
    img5 = cv2.imread(str(root / "snapshots" / f"{row5_holder['row']['id']}.jpg"))
    check(img5.shape[0] > 480, "banner strip still present with draw_boxes=false")
    mask5 = _changed_mask(_frame_region(img5, 480))
    check(not bool(mask5[50:301, 50:151].any()),
          "NO box pixels at the detection region when draw_boxes=false (pipeline)")

    # ----- task isolation: a slow/hung relay never stalls the notify caller
    # (the doorbell watcher awaits _send_notification inline; the enrich task
    # holds the per-event `enriching` flag while inside it) -----
    slow_rec = ApnsRecorder()
    slow_rec.responses = [(200, {"ok": True})]

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.8)  # simulated relay hang, >> the 0.5s budget below
        return slow_rec.handler(request)

    apns_slow = ApnsService(db, settings, transport=httpx.MockTransport(slow_handler))
    pipeline_slow = EventsPipeline(db, FakeMedia(_uniform_jpeg(64, 48)), WSManager(),
                                   push, settings, auth,  # type: ignore[arg-type]
                                   root / "snapshots", apns=apns_slow)
    t0 = time.monotonic()
    await pipeline_slow._send_notification(title="t", body="b", event_id=1,
                                           tag="x", with_image=False)
    elapsed = time.monotonic() - t0
    check(elapsed < 0.5,
          f"hung relay does NOT stall the notification caller ({elapsed:.2f}s < 0.5s)")
    ok = await _wait(lambda: len(slow_rec.requests) == 1)
    check(ok, "isolated apns send still completes in the background")
    await pipeline_slow.shutdown()
    await apns_slow.aclose()

    await pipeline.shutdown()
    await apns.aclose()
    await db.close()


def pipeline_checks() -> None:
    print("pipeline: apns alongside web push, shared cooldown, draw_boxes wiring")
    asyncio.run(_pipeline_cases())


# --------------------------------------------------------------------------- #
# 7. annotate: draw_boxes unit behavior
# --------------------------------------------------------------------------- #


def annotate_checks() -> None:
    print("annotate: draw_boxes=false keeps the banner, skips boxes")
    w, h = 640, 480
    jpeg = _uniform_jpeg(w, h)
    box = [50, 50, 150, 300]
    scene = [{"box": box, "label": "person", "score": 0.9},
             {"box": [250, 60, 360, 320], "label": "dog", "score": 0.8}]

    on = annotate.annotate_event_snapshot(jpeg, None, "person", 0.9, 2, None, scene, True)
    img_on = cv2.imdecode(np.frombuffer(on, np.uint8), cv2.IMREAD_COLOR)
    mask_on = _changed_mask(_frame_region(img_on, h))
    check(bool(mask_on[50:301, 50:151].any()), "draw_boxes=true draws the box (control)")

    off = annotate.annotate_event_snapshot(jpeg, None, "person", 0.9, 2, None, scene, False)
    check(off is not None, "draw_boxes=false still returns a JPEG")
    img_off = cv2.imdecode(np.frombuffer(off, np.uint8), cv2.IMREAD_COLOR)
    check(img_off.shape[0] > h and img_off.shape[1] == w,
          "draw_boxes=false keeps the banner strip (height grew)")
    mask_off = _changed_mask(_frame_region(img_off, h))
    check(not bool(mask_off.any()), "draw_boxes=false leaves the WHOLE frame body untouched")

    # default parameter (legacy callers) draws boxes
    default = annotate.annotate_event_snapshot(jpeg, box, "person", 0.9, 1, None, None)
    img_d = cv2.imdecode(np.frombuffer(default, np.uint8), cv2.IMREAD_COLOR)
    check(bool(_changed_mask(_frame_region(img_d, h))[50:301, 50:151].any()),
          "omitted draw_boxes arg defaults to drawing (legacy-safe)")

    # single-box legacy path also respects draw_boxes=False
    off1 = annotate.annotate_event_snapshot(jpeg, box, "person", 0.9, 1, None, None, False)
    m1 = _changed_mask(_frame_region(cv2.imdecode(np.frombuffer(off1, np.uint8), cv2.IMREAD_COLOR), h))
    check(not bool(m1.any()), "single-box path also skips drawing when disabled")


# --------------------------------------------------------------------------- #
# 8. HTTP API (register / unregister / devices) via the real app
# --------------------------------------------------------------------------- #


def api_checks() -> None:
    print("api: apns register validation, roles, unregister, devices")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.post("/api/auth/login", json={"password": "admin-secret"})
        assert r.status_code == 200, r.text
        admin = {"Authorization": f"Bearer {r.json()['token']}"}

        # a viewer account — register must be open to ANY authenticated role
        r = client.post("/api/users", headers=admin,
                        json={"username": "phoneuser", "password": "viewerpass1", "role": "viewer"})
        assert r.status_code == 201, r.text
        r = client.post("/api/auth/login", json={"username": "phoneuser", "password": "viewerpass1"})
        viewer = {"Authorization": f"Bearer {r.json()['token']}"}

        reg = {"device_token": TOKEN_A.upper(), "device_name": "Adam's iPhone",
               "key_b64": KEY_A_B64, "environment": "production"}

        check(client.post("/api/notifications/apns/register", json=reg).status_code == 401,
              "register without auth -> 401")

        r = client.post("/api/notifications/apns/register", headers=viewer, json=reg)
        check(r.status_code == 204, "VIEWER can register a device (204)")

        r = client.get("/api/notifications/apns/devices", headers=admin)
        devices = r.json()
        check(r.status_code == 200 and len(devices) == 1, "devices list has the row")
        check(devices[0]["device_token_prefix"] == TOKEN_A[:8],
              "uppercase token stored LOWERCASED (prefix matches lowercase)")
        check("device_token" not in devices[0] and len(devices[0]["device_token_prefix"]) == 8,
              "devices list exposes the 8-char prefix only, never the full token")
        check(devices[0]["device_name"] == "Adam's iPhone", "device_name round-trips")

        # upsert: same token re-registered with a new key -> still one row
        r = client.post("/api/notifications/apns/register", headers=admin,
                        json={**reg, "key_b64": KEY_B_B64, "device_name": "n" * 100})
        check(r.status_code == 204, "re-register (rotated key) -> 204 upsert")
        devices = client.get("/api/notifications/apns/devices", headers=admin).json()
        check(len(devices) == 1 and devices[0]["device_name"] == "n" * 64,
              "upsert kept one row; device_name capped at 64 chars")

        # ----- validation: 400s -----
        bad = {**reg, "device_token": "zz" * 32}
        check(client.post("/api/notifications/apns/register", headers=admin, json=bad).status_code == 400,
              "non-hex device_token -> 400")
        bad = {**reg, "device_token": "abc123"}
        check(client.post("/api/notifications/apns/register", headers=admin, json=bad).status_code == 400,
              "too-short device_token -> 400")
        bad = {**reg, "key_b64": base64.b64encode(b"short").decode()}
        check(client.post("/api/notifications/apns/register", headers=admin, json=bad).status_code == 400,
              "key_b64 of the wrong length (not 32 bytes) -> 400")
        bad = {**reg, "key_b64": "!!!not base64!!!"}
        check(client.post("/api/notifications/apns/register", headers=admin, json=bad).status_code == 400,
              "key_b64 that is not base64 -> 400")
        bad = {**reg, "environment": "staging"}
        check(client.post("/api/notifications/apns/register", headers=admin, json=bad).status_code == 400,
              "unknown environment -> 400")
        check(client.post("/api/notifications/apns/register", headers=admin,
                          json={"device_token": TOKEN_B}).status_code == 422,
              "missing key_b64 -> 422 (shape error)")

        # environment defaults to production when omitted
        r = client.post("/api/notifications/apns/register", headers=viewer,
                        json={"device_token": TOKEN_B, "key_b64": KEY_B_B64})
        check(r.status_code == 204, "environment omitted -> defaults to production (204)")

        # ----- unregister: idempotent 204 -----
        r = client.request("DELETE", "/api/notifications/apns/register", headers=viewer,
                           json={"device_token": TOKEN_B.upper()})
        check(r.status_code == 204, "DELETE unregister -> 204 (uppercase input matches)")
        r = client.request("DELETE", "/api/notifications/apns/register", headers=viewer,
                           json={"device_token": TOKEN_B})
        check(r.status_code == 204, "second DELETE still 204 (idempotent)")
        check(client.request("DELETE", "/api/notifications/apns/register",
                             json={"device_token": TOKEN_B}).status_code == 401,
              "unregister without auth -> 401")

        # ----- settings surface: GET returns the apns block (admin) -----
        s = client.get("/api/settings", headers=admin).json()
        check(s["notifications"]["apns"] == {"mode": "off", "relay_url": ""},
              "GET /api/settings exposes the apns defaults (mode off, empty relay_url)")
        check("direct" not in s["notifications"]["apns"],
              "GET /api/settings exposes NO `direct` block (this server holds no .p8)")
        check(s["notifications"]["draw_boxes"] is True, "GET /api/settings exposes draw_boxes")
        # ntfy is a first-class block again (it was stripped as legacy while
        # ntfy support was removed) — the UI needs it to render the channel.
        check("ntfy" in s["notifications"], "GET /api/settings exposes the ntfy block")
        check(s["notifications"]["ntfy"]["topic"] == ""
              and s["notifications"]["ntfy"]["enabled"] is False,
              "ntfy ships disabled with NO default topic (the topic is a secret)")
        s["notifications"]["apns"] = {"mode": "relay", "relay_url": "http://push-relay:8090"}
        s["notifications"]["draw_boxes"] = False
        r = client.put("/api/settings", headers=admin, json=s)
        check(r.status_code == 200
              and r.json()["notifications"]["apns"] == {
                  "mode": "relay", "relay_url": "http://push-relay:8090"},
              "PUT round-trips the apns relay config (incl. the compose-service URL)")
        check(r.json()["notifications"]["draw_boxes"] is False,
              "PUT round-trips draw_boxes=false")

        # A stored "direct" must be MIGRATED on the way in, not 422 the save —
        # otherwise an old /data volume locks the admin out of this page.
        r = client.put("/api/settings", headers=admin, json={**s, "notifications": {
            **s["notifications"], "apns": {"mode": "relay", "relay_url": "ftp://nope"}}})
        check(r.status_code == 422, "PUT a non-http(s) relay_url -> 422 (not a 500)")

        # ----- PATCH: the ntfy block round-trips, and a bad one 422s -----
        r = client.patch("/api/settings", headers=admin, json={"notifications": {"ntfy": {
            "enabled": True, "server": "https://ntfy.example/", "topic": "vigilume_9f3a",
            "auth_token": "tk_x", "priority": 5, "attach_snapshot": False,
        }}})
        check(r.status_code == 200, "PATCH an ntfy block -> 200")
        n = r.json()["notifications"]["ntfy"]
        check(n["topic"] == "vigilume_9f3a" and n["priority"] == 5
              and n["attach_snapshot"] is False and n["server"] == "https://ntfy.example",
              "PATCH round-trips ntfy (server trailing slash stripped)")
        check(r.json()["notifications"]["apns"] == {
            "mode": "relay", "relay_url": "http://push-relay:8090"},
              "the ntfy PATCH did not touch the APNs relay config (deep-merge, not replace)")

        # A validator that raises ValueError must give a 422 — NOT a 500.
        # pydantic v2 puts the raw ValueError OBJECT in each error's ctx, so
        # handing exc.errors() straight to HTTPException makes FastAPI fail to
        # encode the body and return 500. Every custom validator here (ntfy
        # topic/server, public_url, mqtt topic, timezone) hits that path.
        for bad_topic in ("../admin", "a/b", "t?x=1"):
            r = client.patch("/api/settings", headers=admin, json={
                "notifications": {"ntfy": {"topic": bad_topic}}})
            check(r.status_code == 422,
                  f"PATCH ntfy topic {bad_topic!r} -> 422 (not a 500)")
            check("topic" in str(r.json().get("detail")),
                  f"  the 422 detail names the field for {bad_topic!r}")
        r = client.patch("/api/settings", headers=admin,
                         json={"system": {"public_url": "ftp://nope"}})
        check(r.status_code == 422, "PATCH a bad public_url -> 422 (not a 500)")

        # ----- VoIP (PushKit) registration route: POST /api/push/voip -----
        vtok = "cd" * 32
        check(client.post("/api/push/voip", json={"token": vtok}).status_code == 401,
              "voip register without auth -> 401")
        r = client.post("/api/push/voip", headers=viewer,
                        json={"token": vtok.upper(), "device_name": "Adam's iPhone"})
        check(r.status_code == 204, "VIEWER can register a VoIP token (204)")
        devs = client.get("/api/push/voip/devices", headers=admin).json()
        check(len(devs) == 1 and devs[0]["device_token_prefix"] == vtok[:8]
              and "device_token" not in devs[0],
              "voip devices list exposes the 8-char prefix only (token stored lowercased)")
        # device_token alias also accepted; bad token -> 400
        check(client.post("/api/push/voip", headers=viewer,
                          json={"device_token": "zz" * 32}).status_code == 400,
              "non-hex voip token -> 400")
        check(client.post("/api/push/voip", headers=viewer,
                          json={"token": vtok, "environment": "staging"}).status_code == 400,
              "bad voip environment -> 400")
        r = client.request("DELETE", "/api/push/voip", headers=viewer, json={"token": vtok})
        check(r.status_code == 204, "voip unregister -> 204")
        check(client.request("DELETE", "/api/push/voip", headers=viewer,
                             json={"token": vtok}).status_code == 204,
              "second voip unregister still 204 (idempotent)")


def main() -> None:
    db_checks()
    settings_model_checks()
    encryption_checks()
    relay_checks()
    voip_checks()
    pipeline_checks()
    annotate_checks()
    api_checks()
    print(f"\nALL {PASS} CHECKS PASSED (apns push + hide-boxes)")


if __name__ == "__main__":
    main()

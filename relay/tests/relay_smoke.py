"""Smoke suite for the Vigilume APNs relay (relay/main.py).

Pinned contract: docs/push-architecture.md §1 trust model, §3 the API, §3a VoIP.
This suite drives the REAL app through httpx.ASGITransport and fakes Apple with
an httpx.MockTransport injected via create_app(transport=...). No network, no
pytest, no .p8 from disk beyond a throwaway P-256 key we mint here.

Coverage:

  - /healthz: shape, env echo, Cache-Control: no-store
  - /api/push request shape: URL/host, method, apns-topic (from CONFIG, never
    the caller), push-type, priority, expiration, collapse-id, apns-id UUID
  - the APNs body: the `aps` LITERAL, `enc` == the submitted ciphertext, and
    that NO caller key can reach `aps` (the §1 invariant that makes a public
    relay safe)
  - the provider JWT: ES256/kid/iss/iat, CACHED across pushes, re-signed past
    JWT_MAX_AGE_S, and HOST-INDEPENDENT (one token serves both APNs hosts)
  - the body cap: Content-Length fast-reject, the chunked stream-and-abort, and
    a Content-Length that LIES LOW (10 declared, 1 MB streamed)
  - input validation, each asserting ZERO APNs calls (nothing signs or reaches
    Apple on bad input): tokens, payload_b64, priority, collapse_id BYTES
  - *** environment ***: the regression guard. See _env_checks.
  - APNs response -> relay response mapping, including 410 keyed on STATUS
  - the per-token limiter, the GLOBAL limiter, the limiter's LRU bound, the
    BadDeviceToken circuit breaker, and the concurrency gate
  - the constructed-payload size assert (APNS_ALERT_MAX), and that the bytes
    SENT are the bytes MEASURED
  - /api/push/voip: the whitelist, the caps, expiration 0, topic <bundle>.voip
  - LOG HYGIENE: no ciphertext, no plaintext, no full token, no .p8, no JWT,
    no CF-Connecting-IP in any log line at DEBUG — plus the httpx/httpcore
    muzzle, and the one thing that MUST be logged (the host on BadDeviceToken)
  - load_config: path vs inline PEM, and every way it must fail at BOOT

Offline, CPU-only. Usage: python relay/tests/relay_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

RELAY = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, RELAY)

import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402

import httpx  # noqa: E402
import jwt as pyjwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="vigilume-relay-smoke-"))

# One throwaway P-256 key for the whole suite: ES256 keygen is slow, and every
# test that needs "a valid .p8" is indifferent to which key it is.
_KEY = ec.generate_private_key(ec.SECP256R1())
P8_PEM = _KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
P8_PUB = _KEY.public_key()

# Written to a FILE so the APNS_KEY_P8-as-a-PATH branch — the one production is
# supposed to use (an inline PEM leaks the signing key into `docker inspect`) —
# is the branch the whole suite exercises.
P8_PATH = TMP / "AuthKey_SMOKE1234.p8"
P8_PATH.write_text(P8_PEM)

KEY_ID = "SMOKE12345"
TEAM_ID = "TEAM123456"
BUNDLE_ID = "com.vigilume.app"

# EVERY env var is set here, at import, before any load_config() can run —
# create_app() reads the environment on each call, so a stray leftover from the
# ambient shell would silently reconfigure an app under test.
BASE_ENV = {
    "APNS_KEY_P8": str(P8_PATH),
    "APNS_KEY_ID": KEY_ID,
    "APNS_TEAM_ID": TEAM_ID,
    "APNS_BUNDLE_ID": BUNDLE_ID,
    "APNS_ENV": "production",
    "RELAY_MAX_BODY": "4096",
    "RELAY_RATE_LIMIT": "30",
    "RELAY_RATE_WINDOW": "60",
    "RELAY_MAX_CONCURRENCY": "32",
    "RELAY_APNS_TTL": "1800",
    "RELAY_GLOBAL_RATE_LIMIT": "600",
    "RELAY_MAX_LIMITER_KEYS": "10000",
    "RELAY_ACQUIRE_TIMEOUT_MS": "2000",
    "RELAY_BREAKER_THRESHOLD": "50",
    "RELAY_BREAKER_COOLDOWN": "60",
}
os.environ.update(BASE_ENV)

import main  # noqa: E402

PASS = 0

TOKEN_A = "a1" * 32          # 64 hex chars — the low end of TOKEN_RE
TOKEN_B = "b2" * 32
TOKEN_LONG = "c3" * 80       # 160 hex chars — the high end
PROD_HOST = "api.push.apple.com"
SANDBOX_HOST = "api.sandbox.push.apple.com"

# The relay never sees this — it is sealed inside the ciphertext on the device's
# key. It exists so log hygiene can assert on a string that MUST NOT appear.
PLAINTEXT = "PLAINTEXT-Person-at-Front-Door-SECRET"
PAYLOAD_B64 = base64.b64encode(PLAINTEXT.encode()).decode()

# Behind the Cloudflare Tunnel every request carries this. It is client PII and
# must never reach a log line, so it rides on EVERY request this suite makes.
CF_IP = "203.0.113.77"
DEFAULT_HEADERS = {"CF-Connecting-IP": CF_IP}


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}", flush=True)
        # Hard exit: a sys.exit inside asyncio.run raises SystemExit into the
        # loop, which leaves worker threads alive and hangs interpreter
        # shutdown. backend/tests/apns_smoke.py documents the same trap.
        os._exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class Recorder:
    """Fake APNs. Records every request; replays canned responses.

    `status_by_token` lets one app produce different Apple answers per device
    token, so a section can walk the whole response-mapping table without
    rebuilding the app (and resetting its breaker/limiters) each time.
    """

    def __init__(self, status: int = 200, body: dict | None = None,
                 raise_exc: Exception | None = None, delay: float = 0.0) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = {} if body is None else body
        self.raise_exc = raise_exc
        self.delay = delay
        self.status_by_token: dict[str, tuple[int, dict]] = {}

    @property
    def bodies(self) -> list:
        return [json.loads(r.content.decode()) for r in self.requests]

    @property
    def calls(self) -> int:
        return len(self.requests)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_exc is not None:
            raise self.raise_exc
        token = request.url.path.rsplit("/", 1)[-1]
        if token in self.status_by_token:
            status, body = self.status_by_token[token]
            return httpx.Response(status, json=body)
        return httpx.Response(self.status, json=self.body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def make_app(rec: Recorder, **over) -> object:
    """Build a real app with `rec` standing in for Apple.

    Env overrides are applied around create_app() only — load_config() runs
    inside it, so this is the only window in which they matter, and leaving them
    set would leak into the next section.

    Defaults: the rate limiters are effectively OFF unless a test is explicitly
    about them. They are keyed per-app, but a section that fires 30+ pushes at
    one token would otherwise start 429ing for reasons it never asked about.
    """
    over.setdefault("RELAY_RATE_LIMIT", "1000")
    over.setdefault("RELAY_GLOBAL_RATE_LIMIT", "0")
    saved = {k: os.environ.get(k) for k in over}
    try:
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return main.create_app(transport=rec.transport())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://relay.test",
        headers=DEFAULT_HEADERS,
    )


def push_body(**over) -> dict:
    body = {"device_token": TOKEN_A, "payload_b64": PAYLOAD_B64}
    body.update(over)
    return {k: v for k, v in body.items() if v is not _OMIT}


class _Omit:
    def __repr__(self) -> str:
        return "<omitted>"


_OMIT = _Omit()


def closure_of(app, fn_name: str, var: str):
    """Reach the limiter/breaker the app ACTUALLY uses.

    create_app() keeps them as closure locals (correctly — they are per-app
    state, not module globals), so there is no public handle. Asserting on a
    freshly built lookalike would prove nothing about the app; this proves the
    real one is bounded.
    """
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or getattr(fn, "__name__", None) != fn_name:
            continue
        free = fn.__code__.co_freevars
        if var in free:
            return fn.__closure__[free.index(var)].cell_contents
    raise AssertionError(f"no closure var {var!r} on {fn_name!r} — test needs updating")


class shifted_clock:
    """Jump main.time.<attr> forward by `secs` for the duration of the block.

    main.time IS the stdlib time module, so this is a process-wide patch — the
    shift is applied as an OFFSET on the real clock rather than a frozen
    constant, so asyncio's loop.time() (which is time.monotonic()) stays
    monotonic and its timers keep the same relative deadlines.
    """

    def __init__(self, attr: str, secs: float) -> None:
        self.attr = attr
        self.secs = secs

    def __enter__(self) -> "shifted_clock":
        self.orig = getattr(main.time, self.attr)
        orig, secs = self.orig, self.secs
        setattr(main.time, self.attr, lambda: orig() + secs)
        return self

    def __exit__(self, *exc) -> None:
        setattr(main.time, self.attr, self.orig)


def jwt_of(request: httpx.Request) -> str:
    return request.headers["authorization"][len("bearer "):]


# --------------------------------------------------------------------------- #
# 1. /healthz
# --------------------------------------------------------------------------- #


async def _health_cases() -> None:
    rec = Recorder()
    app = make_app(rec, APNS_ENV="sandbox")
    async with client_for(app) as c:
        r = await c.get("/healthz")
        check(r.status_code == 200, "GET /healthz -> 200")
        check(r.json()["ok"] is True, "/healthz ok is True")
        check(r.json()["env"] == "sandbox", "/healthz env echoes APNS_ENV")
        # A health check that a "Cache Everything" rule froze at {"ok":true}
        # forever is worse than no health check at all.
        check(r.headers.get("cache-control") == "no-store",
              "/healthz sends Cache-Control: no-store")


def health_checks() -> None:
    print("health: shape, env echo, no-store")
    asyncio.run(_health_cases())


# --------------------------------------------------------------------------- #
# 2. /api/push request shape + APNs body
# --------------------------------------------------------------------------- #


async def _shape_cases() -> None:
    rec = Recorder()
    app = make_app(rec)  # APNS_ENV=production
    async with client_for(app) as c:
        before = main.time.time()
        r = await c.post("/api/push", json=push_body(collapse_id="event-42"))
        after = main.time.time()

        check(r.status_code == 200 and r.json() == {"ok": True},
              "push -> 200 {\"ok\":true}")
        check(rec.calls == 1, "exactly ONE APNs request per push")
        req = rec.requests[0]
        check(str(req.url) == f"https://{PROD_HOST}/3/device/{TOKEN_A}",
              "URL is https://api.push.apple.com/3/device/<token> for production")
        check(req.method == "POST", "APNs request is a POST")
        check(req.headers["apns-topic"] == BUNDLE_ID,
              "apns-topic comes from CONFIG (a caller who could set it would aim "
              "the owner's .p8 at another app's tokens)")
        check(req.headers["apns-push-type"] == "alert", "apns-push-type: alert")
        check(req.headers["apns-priority"] == "10", "priority 'high' -> apns-priority 10")
        exp = int(req.headers["apns-expiration"])
        check(int(before) + 1800 <= exp <= int(after) + 1800,
              "apns-expiration is now + RELAY_APNS_TTL (1800)")
        check(req.headers["apns-collapse-id"] == "event-42", "collapse_id forwarded as a header")
        apns_id = req.headers["apns-id"]
        check(str(uuid.UUID(apns_id)) == apns_id, "apns-id present and a valid UUID")

        r = await c.post("/api/push", json=push_body(priority="normal"))
        check(r.status_code == 200 and rec.requests[-1].headers["apns-priority"] == "5",
              "priority 'normal' -> apns-priority 5")
        r = await c.post("/api/push", json=push_body())
        check(r.status_code == 200 and rec.requests[-1].headers["apns-priority"] == "10",
              "priority omitted -> defaults to high (apns-priority 10)")
        check("apns-collapse-id" not in rec.requests[-1].headers,
              "no collapse_id -> no apns-collapse-id header")

        # ----- the APNs body -----
        body = rec.bodies[0]
        check(body["aps"]["mutable-content"] == 1,
              "aps mutable-content=1 (what triggers the iOS service extension)")
        check(body["aps"]["alert"] == {"title": "Vigilume", "body": "Encrypted notification"},
              "aps alert is the generic decrypt-failed fallback")
        check(body["enc"] == PAYLOAD_B64, "enc is the exact payload_b64 submitted")
        check(set(body.keys()) == {"aps", "enc"},
              "the sent body's top-level keys are exactly {aps, enc}")

        # *** §1's security invariant. *** `aps` is a server-built literal;
        # caller bytes only ever land in "enc". A caller who could merge into
        # `aps` could set sound/content-available/badge — or apns-topic — on a
        # push signed with the owner's .p8.
        r = await c.post("/api/push", json=push_body(**{
            "aps": {"alert": "PWNED", "badge": 99},
            "sound": "hacked.caf",
            "content-available": 1,
            "apns-topic": "com.attacker.app",
            "enc": "attacker-enc",
        }))
        check(r.status_code == 200, "a body stuffed with aps/sound/apns-topic still 200s")
        sent = rec.bodies[-1]
        check(sent["aps"] == {"mutable-content": 1,
                              "alert": {"title": "Vigilume", "body": "Encrypted notification"}},
              "NO caller key enters aps — it is byte-identical to the literal")
        check(sent["enc"] == PAYLOAD_B64 and set(sent.keys()) == {"aps", "enc"},
              "caller's own 'enc'/extra keys dropped; body still exactly {aps, enc}")
        check(rec.requests[-1].headers["apns-topic"] == BUNDLE_ID,
              "a caller-supplied apns-topic in the JSON body never reaches the header")


def shape_checks() -> None:
    print("push: request shape + the aps literal (no caller key may enter it)")
    asyncio.run(_shape_cases())


# --------------------------------------------------------------------------- #
# 3. the provider JWT
# --------------------------------------------------------------------------- #


async def _jwt_cases() -> None:
    rec = Recorder()
    app = make_app(rec)
    async with client_for(app) as c:
        await c.post("/api/push", json=push_body())
        req = rec.requests[0]
        auth = req.headers["authorization"]
        check(auth.startswith("bearer "), "authorization is 'bearer <jwt>'")
        tok = jwt_of(req)
        header = pyjwt.get_unverified_header(tok)
        check(header["alg"] == "ES256", "JWT header alg is ES256")
        check(header["kid"] == KEY_ID, "JWT header kid is APNS_KEY_ID")
        claims = pyjwt.decode(tok, P8_PUB, algorithms=["ES256"])
        check(claims["iss"] == TEAM_ID, "JWT claim iss is APNS_TEAM_ID")
        check(abs(claims["iat"] - main.time.time()) < 5, "JWT claim iat is ~now")

        # Signing is off the request path: one ES256 signature per 45 min
        # regardless of volume. If this ever regresses to per-request, Apple's
        # per-connection TooManyProviderTokenUpdates becomes reachable.
        await c.post("/api/push", json=push_body())
        check(jwt_of(rec.requests[1]) == tok, "the JWT is CACHED (two pushes reuse it)")

        # Apple: refresh no more than once per 20 min, no less than once per 60.
        with shifted_clock("time", main.JWT_MAX_AGE_S + 60):
            await c.post("/api/push", json=push_body())
        tok2 = jwt_of(rec.requests[2])
        check(tok2 != tok, "past JWT_MAX_AGE_S the token is RE-SIGNED (differs)")
        # verify_iat off: the re-signed token was minted on the SHIFTED clock,
        # so its iat is 45 min in this process's future. That is the test's
        # doing, not the relay's — the signature is what matters here.
        claims2 = pyjwt.decode(tok2, P8_PUB, algorithms=["ES256"],
                               options={"verify_iat": False})
        check(claims2["iss"] == TEAM_ID,
              "the re-signed token is still a valid ES256 provider token")
        check(claims2["iat"] > claims["iat"],
              "the re-signed token carries a FRESH iat (Apple rejects a stale one)")


async def _jwt_host_independence() -> None:
    """One token serves BOTH hosts, so the cache is keyed on NOTHING.

    That is an INVARIANT, not an accident: if a future version keys this cache
    on the caller's `environment`, an attacker alternating environments thrashes
    it into signing on every request.
    """
    rec = Recorder()
    app = make_app(rec)
    async with client_for(app) as c:
        await c.post("/api/push", json=push_body(environment="sandbox"))
        await c.post("/api/push", json=push_body(environment="production"))
    hosts = [r.url.host for r in rec.requests]
    check(hosts == [SANDBOX_HOST, PROD_HOST], "one app served sandbox then production")
    a, b = jwt_of(rec.requests[0]), jwt_of(rec.requests[1])
    check(a == b, "the SAME JWT string served both APNs hosts (cache keyed on nothing)")
    check(pyjwt.decode(b, P8_PUB, algorithms=["ES256"])["iss"] == TEAM_ID,
          "and it is still a validly signed token on both")


def jwt_checks() -> None:
    print("jwt: ES256 provider token, cached, re-signed, host-independent")
    asyncio.run(_jwt_cases())
    asyncio.run(_jwt_host_independence())


# --------------------------------------------------------------------------- #
# 4. body cap
# --------------------------------------------------------------------------- #


def raw_push(payload_b64: str, token: str = TOKEN_A) -> bytes:
    """The relay body as EXACT bytes, so a cap test can hit the cap exactly.

    json.dumps would be at the mercy of key order and separator defaults.
    """
    return (b'{"device_token":"' + token.encode()
            + b'","payload_b64":"' + payload_b64.encode() + b'"}')


async def _chunks(total: int, chunk: int = 65536):
    sent = 0
    while sent < total:
        n = min(chunk, total - sent)
        yield b"x" * n
        sent += n


async def _cap_cases() -> None:
    cap = 4096
    rec = Recorder()
    app = make_app(rec, RELAY_MAX_BODY=cap)
    async with client_for(app) as c:
        # ---- Content-Length fast reject ----
        # Fires when a hoster sets cloudflared's disableChunkedEncoding=true and
        # cloudflared buffers + declares a length.
        r = await c.post("/api/push", content=b"x" * (cap + 1),
                         headers={"content-type": "application/json"})
        check(r.status_code == 400 and r.json()["reason"] == "too_large",
              "Content-Length over the cap -> 400 too_large")
        check(rec.calls == 0, "  and ZERO APNs calls (nothing signs on an oversized body)")

        # ---- chunked, no Content-Length ----
        # The default path behind a Cloudflare Tunnel (disableChunkedEncoding
        # defaults to false, so chunked passes straight through).
        r = await c.post("/api/push", content=_chunks(cap * 4),
                         headers={"content-type": "application/json"})
        check(r.status_code == 400 and r.json()["reason"] == "too_large",
              "chunked body over the cap (no Content-Length) -> 400 too_large")
        check(rec.calls == 0, "  and ZERO APNs calls")

        # ---- the Content-Length that LIES LOW ----
        # THE reason Content-Length is only ever a fast-reject optimisation.
        # Declare 10, stream 1 MB: if the code trusted the header and called
        # request.body(), this buffers a megabyte.
        r = await c.post("/api/push", content=_chunks(1_000_000),
                         headers={"content-type": "application/json",
                                  "content-length": "10"})
        check(r.status_code == 400 and r.json()["reason"] == "too_large",
              "Content-Length: 10 + a 1 MB stream -> 400 too_large (the stream is the bound)")
        check(rec.calls == 0, "  and ZERO APNs calls (the declared length was a lie)")

        # ---- exactly at the cap ----
        # raw_push is 100 bytes of scaffolding + the payload.
        n = cap - 100
        assert n % 4 == 0, "payload_b64 must stay base64-shaped"
        at_cap = raw_push("A" * n)
        check(len(at_cap) == cap, "test fixture really is exactly RELAY_MAX_BODY bytes")
        r = await c.post("/api/push", content=at_cap,
                         headers={"content-type": "application/json"})
        check(r.status_code == 200 and rec.calls == 1,
              "a body of EXACTLY the cap passes (the cap is >, not >=)")

        r = await c.post("/api/push", content=at_cap + b" ",
                         headers={"content-type": "application/json"})
        check(r.status_code == 400 and r.json()["reason"] == "too_large" and rec.calls == 1,
              "cap+1 rejected")


def cap_checks() -> None:
    print("cap: Content-Length fast path, the chunked stream-and-abort, a lying length")
    asyncio.run(_cap_cases())


# --------------------------------------------------------------------------- #
# 5. input validation — every case asserts ZERO APNs calls
# --------------------------------------------------------------------------- #


async def _bad_input_cases() -> None:
    rec = Recorder()
    app = make_app(rec)
    async with client_for(app) as c:

        async def reject(body, status, reason, msg, raw=False):
            n = rec.calls
            if raw:
                r = await c.post("/api/push", content=body,
                                 headers={"content-type": "application/json"})
            else:
                r = await c.post("/api/push", json=body)
            ok = (r.status_code == status and r.json()["reason"] == reason
                  and rec.calls == n)
            check(ok, f"{msg} -> {status} {reason}, zero APNs calls")

        await reject(push_body(device_token="z" * 64), 400, "bad_device_token",
                     "non-hex token")
        await reject(push_body(device_token="a" * 63), 400, "bad_device_token",
                     "63-char token (below TOKEN_RE's floor)")
        await reject(push_body(device_token="a" * 161), 400, "bad_device_token",
                     "161-char token (above TOKEN_RE's ceiling)")
        # THE fullmatch guard. Python's `$` also matches BEFORE a trailing
        # newline, so `^...$` + .match() would accept this and the newline would
        # land in the /3/device/<token> path.
        await reject(push_body(device_token="a" * 64 + "\n"), 400, "bad_device_token",
                     "token with a TRAILING NEWLINE (the fullmatch guard)")
        await reject(push_body(device_token=_OMIT), 400, "bad_device_token",
                     "missing device_token")
        await reject(push_body(device_token=12345), 400, "bad_device_token",
                     "non-string device_token")

        await reject(b"not json at all", 400, "bad_payload", "a non-JSON body", raw=True)
        await reject([1, 2, 3], 400, "bad_payload", "a JSON array (not a dict)")
        await reject(push_body(payload_b64=_OMIT), 400, "bad_payload", "missing payload_b64")
        await reject(push_body(payload_b64=""), 400, "bad_payload", "empty payload_b64")
        # validate=True: base64 that would otherwise decode by silently
        # discarding the junk. The device's AES-GCM open would fail on it.
        await reject(push_body(payload_b64="not!valid!base64!"), 400, "bad_payload",
                     "payload_b64 that is not valid base64 (validate=True)")

        await reject(push_body(priority="urgent"), 400, "bad_priority",
                     "priority 'urgent' (closed vocabulary)")
        await reject(push_body(collapse_id="c" * 65), 400, "bad_collapse_id",
                     "collapse_id of 65 bytes")
        await reject(push_body(collapse_id=42), 400, "bad_collapse_id",
                     "non-string collapse_id")

        check(rec.calls == 0, "the ENTIRE bad-input section made zero APNs calls")

        # ----- collapse_id: Apple's limit is 64 BYTES, not characters -----
        r = await c.post("/api/push", json=push_body(collapse_id="c" * 64))
        check(r.status_code == 200 and rec.requests[-1].headers["apns-collapse-id"] == "c" * 64,
              "64-char ASCII collapse_id passes (exactly at Apple's 64-byte limit)")
        # 23 chars — a naive len() check calls this fine — but 65 UTF-8 bytes,
        # which is what APNs measures and 400 BadCollapseIds on.
        wide = "あ" * 21 + "ab"
        assert len(wide) == 23 and len(wide.encode()) == 65
        n = rec.calls
        r = await c.post("/api/push", json=push_body(collapse_id=wide))
        check(r.status_code == 400 and r.json()["reason"] == "bad_collapse_id" and rec.calls == n,
              "23 chars / 65 BYTES of UTF-8 collapse_id -> 400 bad_collapse_id (bytes, not chars)")

        # ----- lowercasing -----
        # The backend stores tokens lowercased; if the relay forwarded the
        # caller's casing, a 410 would come back with a token the backend's
        # prune lookup cannot match, and the dead row would live forever.
        await c.post("/api/push", json=push_body(device_token=TOKEN_A.upper()))
        check(rec.requests[-1].url.path == f"/3/device/{TOKEN_A}",
              "an UPPERCASE token reaches APNs LOWERCASED (the 410-prune lookup matches)")
        # The 160-char end of the band must actually work.
        await c.post("/api/push", json=push_body(device_token=TOKEN_LONG))
        check(rec.requests[-1].url.path == f"/3/device/{TOKEN_LONG}",
              "a 160-char token (TOKEN_RE's ceiling) is accepted")


def bad_input_checks() -> None:
    print("input: tokens, payload_b64, priority, collapse_id bytes — all zero-call")
    asyncio.run(_bad_input_cases())


# --------------------------------------------------------------------------- #
# 6. *** environment *** — the regression guard
# --------------------------------------------------------------------------- #


async def _env_cases() -> None:
    """THE bug this rebuild exists to not repeat.

    The previous relay pinned ONE APNs host at startup (AsyncClient(base_url=))
    and never read `environment`. A sandbox (Xcode dev build) token was posted
    to the production host, Apple answered BadDeviceToken, and every push
    silently vanished. So: the host is resolved PER REQUEST, an unknown value is
    a LOUD 400 rather than a silent fallback, and the value is never
    interpolated into a URL — only looked up in the fixed APNS_HOSTS map.
    """
    rec = Recorder()
    app = make_app(rec, APNS_ENV="production")
    async with client_for(app) as c:
        await c.post("/api/push", json=push_body(environment="sandbox"))
        check(rec.requests[-1].url.host == SANDBOX_HOST,
              "environment=sandbox -> the sandbox host (even though APNS_ENV=production)")
        await c.post("/api/push", json=push_body(environment="production"))
        check(rec.requests[-1].url.host == PROD_HOST,
              "environment=production -> the production host")
        await c.post("/api/push", json=push_body())
        check(rec.requests[-1].url.host == PROD_HOST,
              "environment ABSENT -> falls back to the APNS_ENV host")
        await c.post("/api/push", json=push_body(environment="  SANDBOX  "))
        check(rec.requests[-1].url.host == SANDBOX_HOST,
              "environment 'SANDBOX' (case/whitespace) -> the sandbox host")

    # A sandbox-default relay must fall back to SANDBOX, not to production —
    # the fallback is the config's, not a hardcoded default.
    rec2 = Recorder()
    app2 = make_app(rec2, APNS_ENV="sandbox")
    async with client_for(app2) as c:
        await c.post("/api/push", json=push_body())
        check(rec2.requests[-1].url.host == SANDBOX_HOST,
              "APNS_ENV=sandbox + environment absent -> the sandbox host")

    rec3 = Recorder()
    app3 = make_app(rec3)
    async with client_for(app3) as c:
        # NOT a silent fallback to production. A silent fallback reproduces the
        # exact debugging experience of the original bug: push accepted, Apple
        # says BadDeviceToken, notification vanishes.
        r = await c.post("/api/push", json=push_body(environment="staging"))
        check(r.status_code == 400 and r.json()["reason"] == "bad_environment",
              "environment='staging' -> 400 bad_environment (NOT a silent fallback)")
        check(rec3.calls == 0, "  and ZERO APNs calls — the .p8 never signed anything")

        r = await c.post("/api/push", json=push_body(environment=123))
        check(r.status_code == 400 and r.json()["reason"] == "bad_environment" and rec3.calls == 0,
              "environment=123 (non-string) -> 400 bad_environment, zero calls")

        r = await c.post("/api/push", json=push_body(environment=""))
        check(r.status_code == 400 and r.json()["reason"] == "bad_environment" and rec3.calls == 0,
              "environment='' -> 400 bad_environment (empty is PRESENT, not absent)")

        # *** The fixed-map guard. *** APNS_HOSTS is a dict LOOKUP, never an
        # f-string / urljoin / base_url. An unknown value cannot BECOME a URL,
        # it can only fail to match — so a caller cannot steer .p8-signed
        # requests at a host of their choosing.
        r = await c.post("/api/push", json=push_body(environment="https://evil.example"))
        check(r.status_code == 400 and r.json()["reason"] == "bad_environment",
              "environment='https://evil.example' -> 400 bad_environment")
        check(rec3.calls == 0
              and not any("evil" in str(q.url) for q in rec.requests + rec2.requests + rec3.requests),
              "  'evil.example' NEVER appears in any APNs URL (fixed-map lookup, not interpolation)")


def env_checks() -> None:
    print("environment: per-request host routing — the regression guard")
    asyncio.run(_env_cases())


# --------------------------------------------------------------------------- #
# 7. APNs response -> relay response mapping
# --------------------------------------------------------------------------- #


async def _mapping_cases() -> None:
    rec = Recorder()
    # Breaker off: this section deliberately fires BadDeviceTokens and must not
    # trip a breaker into masking the later assertions.
    app = make_app(rec, RELAY_BREAKER_THRESHOLD="0")
    cases = [
        ((200, {}), 200, None, "APNs 200 -> 200 {\"ok\":true}"),
        ((410, {"reason": "Unregistered"}), 410, "unregistered",
         "APNs 410 Unregistered -> 410 unregistered (the backend prunes the row)"),
        # 410 carries BOTH Unregistered AND ExpiredToken. Key on the STATUS,
        # never the reason string, or dead tokens leak and live forever.
        ((410, {"reason": "ExpiredToken"}), 410, "unregistered",
         "APNs 410 ExpiredToken -> 410 unregistered (keyed on STATUS, not reason)"),
        ((400, {"reason": "BadDeviceToken"}), 400, "bad_device_token",
         "APNs 400 BadDeviceToken -> 400 bad_device_token"),
        ((400, {"reason": "DeviceTokenNotForTopic"}), 400, "bad_device_token",
         "APNs 400 DeviceTokenNotForTopic -> 400 bad_device_token"),
        ((400, {"reason": "BadTopic"}), 400, "bad_payload",
         "APNs 400 BadTopic -> 400 bad_payload (a config error, not a dead token)"),
        ((413, {}), 400, "too_large", "APNs 413 -> 400 too_large"),
        ((400, {"reason": "PayloadTooLarge"}), 400, "too_large",
         "APNs 400 PayloadTooLarge -> 400 too_large (reason as well as status)"),
        # The reason is COLLAPSED on the way out: it denies a caller an oracle
        # for how this .p8 is scoped. The raw string goes to the log instead.
        ((403, {"reason": "InvalidProviderToken"}), 502, "apns_auth",
         "APNs 403 InvalidProviderToken -> 502 apns_auth"),
        ((403, {"reason": "BadEnvironmentKeyIdInToken"}), 502, "apns_auth",
         "APNs 403 BadEnvironmentKeyIdInToken -> 502 apns_auth (env-scoped key, 2025-02-17)"),
        ((429, {"reason": "TooManyRequests"}), 429, "apns_rate_limited",
         "APNs 429 TooManyRequests -> 429 apns_rate_limited"),
        ((500, {"reason": "InternalServerError"}), 502, "apns_error",
         "APNs 500 -> 502 apns_error"),
        ((503, {}), 502, "apns_error", "APNs 503 -> 502 apns_error"),
    ]
    async with client_for(app) as c:
        for i, ((status, body), want_status, want_reason, msg) in enumerate(cases):
            rec.status = status
            rec.body = body
            n = rec.calls
            r = await c.post("/api/push", json=push_body())
            ok = r.status_code == want_status and rec.calls == n + 1
            if want_reason is None:
                ok = ok and r.json() == {"ok": True}
            else:
                ok = ok and r.json() == {"ok": False, "reason": want_reason}
            check(ok, msg)

    # A transport-level failure (Apple unreachable, DNS, TLS) is NOT an APNs
    # answer — it must not be confused with one, and must never raise out of the
    # handler into a 500.
    rec2 = Recorder(raise_exc=httpx.ConnectError("nope"))
    app2 = make_app(rec2)
    async with client_for(app2) as c:
        r = await c.post("/api/push", json=push_body())
        check(r.status_code == 502 and r.json()["reason"] == "apns_unreachable",
              "transport raises httpx.ConnectError -> 502 apns_unreachable (never a 500)")

    # A 200 with a body that is not JSON at all must still be a success — Apple
    # sends an empty body on success, and the reason-parsing must not become a
    # failure mode of its own.
    rec3 = Recorder()
    rec3.handler = lambda request: httpx.Response(200, content=b"")  # type: ignore[assignment]
    app3 = make_app(rec3)
    async with client_for(app3) as c:
        r = await c.post("/api/push", json=push_body())
        check(r.status_code == 200 and r.json() == {"ok": True},
              "APNs 200 with an empty (non-JSON) body still -> 200 ok")


def mapping_checks() -> None:
    print("mapping: every APNs status -> the contract's error envelope")
    asyncio.run(_mapping_cases())


# --------------------------------------------------------------------------- #
# 8. rate limiting + the limiter's LRU bound
# --------------------------------------------------------------------------- #


async def _rate_cases() -> None:
    rec = Recorder()
    app = make_app(rec, RELAY_RATE_LIMIT="3", RELAY_RATE_WINDOW="60",
                   RELAY_GLOBAL_RATE_LIMIT="0")
    async with client_for(app) as c:
        for i in range(3):
            r = await c.post("/api/push", json=push_body())
            check(r.status_code == 200, f"per-token push {i + 1}/3 within the limit -> 200")
        n = rec.calls
        r = await c.post("/api/push", json=push_body())
        check(r.status_code == 429 and r.json()["reason"] == "rate_limited",
              "the 4th push at RELAY_RATE_LIMIT=3 -> 429 rate_limited")
        check(rec.calls == n, "  and ZERO APNs calls (rate limiting is before the .p8)")

        # Keyed per token: one noisy device must not silence another.
        r = await c.post("/api/push", json=push_body(device_token=TOKEN_B))
        check(r.status_code == 200 and rec.calls == n + 1,
              "a DIFFERENT token is unaffected (the limiter is per-token)")

        # monotonic(), not time(): a clock STEP must not open the window. The
        # shift here is the window sliding, not the wall clock moving.
        with shifted_clock("monotonic", 61):
            r = await c.post("/api/push", json=push_body())
        check(r.status_code == 200 and rec.calls == n + 2,
              "past RELAY_RATE_WINDOW the window slides and the token recovers")

    # ----- the GLOBAL limiter -----
    # The per-token limiter is keyed on caller-controlled input, so it offers NO
    # global protection: 10k distinct hex strings each get a fresh window.
    rec2 = Recorder()
    app2 = make_app(rec2, RELAY_RATE_LIMIT="1000", RELAY_GLOBAL_RATE_LIMIT="2")
    async with client_for(app2) as c:
        r1 = await c.post("/api/push", json=push_body(device_token="1" * 64))
        r2 = await c.post("/api/push", json=push_body(device_token="2" * 64))
        r3 = await c.post("/api/push", json=push_body(device_token="3" * 64))
        check(r1.status_code == 200 and r2.status_code == 200,
              "two DISTINCT tokens inside the global limit -> 200")
        check(r3.status_code == 429 and r3.json()["reason"] == "rate_limited" and rec2.calls == 2,
              "a THIRD distinct token trips the GLOBAL limiter (per-token cannot see this)")


async def _limiter_bound_cases() -> None:
    """Keys are device tokens — caller-controlled input in an all-in-memory
    process. Unbounded, a caller sending distinct valid-looking tokens grows the
    dict until the container OOMs.
    """
    max_keys = 4
    rec = Recorder()
    app = make_app(rec, RELAY_RATE_LIMIT="2", RELAY_MAX_LIMITER_KEYS=str(max_keys),
                   RELAY_GLOBAL_RATE_LIMIT="0")
    limiter = closure_of(app, "push", "per_token")
    async with client_for(app) as c:
        tokens = [f"{i:02d}" * 32 for i in range(max_keys + 6)]
        for t in tokens:
            r = await c.post("/api/push", json=push_body(device_token=t))
            check_silent = r.status_code == 200
            assert check_silent, r.text
        check(len(limiter._hits) <= max_keys,
              f"after {len(tokens)} distinct tokens the limiter dict is <= "
              f"max_limiter_keys ({len(limiter._hits)} <= {max_keys})")

        # LRU evicts the OLDEST, never the key being used — otherwise the bound
        # would be a rate-limit bypass (sweep N junk tokens, evict the hot one,
        # repeat). The most recent token must still be tracked.
        hot = tokens[-1]  # already has 1 hit
        r = await c.post("/api/push", json=push_body(device_token=hot))
        check(r.status_code == 200, "the most-recent token's 2nd push (at the limit) -> 200")
        r = await c.post("/api/push", json=push_body(device_token=hot))
        check(r.status_code == 429 and r.json()["reason"] == "rate_limited",
              "the most-recent token STILL rate-limits after eviction (LRU never drops the hot key)")


def rate_checks() -> None:
    print("limits: per-token, global, sliding window, and the LRU bound")
    asyncio.run(_rate_cases())
    asyncio.run(_limiter_bound_cases())


# --------------------------------------------------------------------------- #
# 9. circuit breaker
# --------------------------------------------------------------------------- #


async def _breaker_cases() -> None:
    """Protects the TEAM's APNs reputation — shared-fate, and nothing else
    guards it. Every Vigilume install shares ONE bundle id and ONE .p8, and
    Apple disconnects sooner on BadDeviceToken than on other errors, so an
    anonymous sweep of junk tokens degrades delivery for every hoster at once.
    """
    rec = Recorder(status=400, body={"reason": "BadDeviceToken"})
    app = make_app(rec, RELAY_BREAKER_THRESHOLD="3", RELAY_BREAKER_COOLDOWN="30",
                   RELAY_RATE_LIMIT="1000", RELAY_GLOBAL_RATE_LIMIT="0")
    async with client_for(app) as c:
        for i in range(3):
            r = await c.post("/api/push", json=push_body(device_token=f"{i}" * 64))
            check(r.status_code == 400 and r.json()["reason"] == "bad_device_token",
                  f"BadDeviceToken {i + 1}/3 forwarded normally (breaker still closed)")
        check(rec.calls == 3, "  all 3 reached Apple")

        r = await c.post("/api/push", json=push_body(device_token="9" * 64))
        check(r.status_code == 429 and r.json()["reason"] == "rate_limited",
              "the threshold'th BadDeviceToken OPENS the breaker -> next push 429 rate_limited")
        check(rec.calls == 3, "  and ZERO further APNs calls while open")

        # A self-DoS lever if it tripped tight, hence the high default and the
        # cooldown: it must close again on its own.
        rec.status, rec.body = 200, {}
        with shifted_clock("monotonic", 31):
            r = await c.post("/api/push", json=push_body())
        check(r.status_code == 200 and rec.calls == 4,
              "after RELAY_BREAKER_COOLDOWN the breaker closes and forwards again")


def breaker_checks() -> None:
    print("breaker: BadDeviceToken storm opens it, cooldown closes it")
    asyncio.run(_breaker_cases())


# --------------------------------------------------------------------------- #
# 10. concurrency gate
# --------------------------------------------------------------------------- #


async def _concurrency_cases() -> None:
    """A gate with an infinite waiting room is a memory amplifier wearing a
    safety vest: without the acquire timeout, excess requests each park holding
    a task, a socket and a buffered body for the full httpx timeout, and
    Cloudflare 524s the caller at 120s anyway. Shed at the door instead.
    """
    rec = Recorder(delay=1.0)  # a hanging Apple
    app = make_app(rec, RELAY_MAX_CONCURRENCY="1", RELAY_ACQUIRE_TIMEOUT_MS="200",
                   RELAY_RATE_LIMIT="1000", RELAY_GLOBAL_RATE_LIMIT="0")
    async with client_for(app) as c:
        first = asyncio.create_task(c.post("/api/push", json=push_body()))
        await asyncio.sleep(0.05)  # let `first` take the only semaphore slot
        t0 = main.time.monotonic()
        second = await c.post("/api/push", json=push_body(device_token=TOKEN_B))
        waited = main.time.monotonic() - t0
        check(second.status_code == 429 and second.json()["reason"] == "rate_limited",
              "with the gate full, a concurrent push is SHED with 429 rate_limited")
        check(waited < 0.6,
              f"  it shed at ~the acquire timeout rather than queueing ({waited:.2f}s < 0.6s)")
        r = await first
        check(r.status_code == 200 and rec.calls == 1,
              "the in-flight request still completes normally (the shed one never called Apple)")


def concurrency_checks() -> None:
    print("concurrency: the gate sheds at the door instead of queueing")
    asyncio.run(_concurrency_cases())


# --------------------------------------------------------------------------- #
# 11. the constructed-payload size assert
# --------------------------------------------------------------------------- #


async def _payload_size_cases() -> None:
    """RELAY_MAX_BODY does not imply the APNs ceiling.

    A 4096-byte relay body minus the JSON scaffolding leaves ~3996 chars of
    payload_b64; the aps envelope puts that back at ~4095 against Apple's 4096.
    It fits — with ~1 byte of margin, and nothing else enforces it. So the
    CONSTRUCTED body is what gets asserted. (Raise RELAY_MAX_BODY, as a hoster
    might, and the margin is gone entirely — which is what this tests.)
    """
    rec = Recorder()
    app = make_app(rec, RELAY_MAX_BODY="8192")
    async with client_for(app) as c:
        # 4000 chars of payload_b64: relay body 4100 (fits 8192), aps body 4099
        # (over APNS_ALERT_MAX). PayloadTooLarge is permanent and non-retryable,
        # so this must be caught here, not by Apple.
        pb = "A" * 4000
        r = await c.post("/api/push", content=raw_push(pb),
                         headers={"content-type": "application/json"})
        check(r.status_code == 400 and r.json()["reason"] == "too_large",
              "a payload that fits RELAY_MAX_BODY but blows the 4096 aps ceiling -> 400 too_large")
        check(rec.calls == 0, "  and ZERO APNs calls")

        # ...and the largest one that DOES fit is sent, byte-for-byte as
        # measured. `raw` is both what len() checked and what client.post sends
        # — if those two ever diverge, the assert protects nothing.
        pb = "A" * 3996
        expected = (b'{"aps":{"mutable-content":1,"alert":{"title":"Vigilume",'
                    b'"body":"Encrypted notification"}},"enc":"' + pb.encode() + b'"}')
        check(len(expected) == 4095, "fixture: the largest fitting aps body is 4095 bytes")
        r = await c.post("/api/push", content=raw_push(pb),
                         headers={"content-type": "application/json"})
        check(r.status_code == 200 and rec.calls == 1, "4095-byte aps body is sent")
        check(rec.requests[0].content == expected,
              "the body SENT is byte-identical to the body that was MEASURED")


def payload_size_checks() -> None:
    print("payload: the aps body is asserted against APNS_ALERT_MAX, and sent as measured")
    asyncio.run(_payload_size_cases())


# --------------------------------------------------------------------------- #
# 12. /api/push/voip
# --------------------------------------------------------------------------- #


async def _voip_cases() -> None:
    """The VoIP payload is the ONLY caller-controlled JSON structure that
    reaches Apple — it CANNOT be ciphertext (PushKit must report to CallKit
    immediately; there is no service-extension window to decrypt in, and failing
    to report gets the app terminated and then cut off from VoIP pushes
    entirely). So the mitigation is constraint: whitelist + caps.
    """
    rec = Recorder()
    app = make_app(rec, APNS_ENV="production")
    async with client_for(app) as c:
        payload = {"type": "doorbell", "camera": "Front Door", "event_id": "77"}
        r = await c.post("/api/push/voip",
                         json={"device_token": TOKEN_A, "payload": payload})
        check(r.status_code == 200 and rec.calls == 1, "voip push -> 200, one APNs request")
        req = rec.requests[0]
        check(req.url.path == f"/3/device/{TOKEN_A}", "voip URL is /3/device/<token>")
        check(req.headers["apns-topic"] == f"{BUNDLE_ID}.voip",
              "voip apns-topic is <bundle>.voip (Apple's fixed suffix)")
        check(req.headers["apns-push-type"] == "voip", "voip apns-push-type: voip")
        check(req.headers["apns-priority"] == "10", "voip apns-priority is always 10")
        # *** 0, NOT now+ttl. *** A 30-minute-old doorbell ring firing as a live
        # CallKit call is a real defect and an attack amplifier (queue rings at
        # an offline phone, they all land at once when it comes back).
        check(req.headers["apns-expiration"] == "0",
              "voip apns-expiration is 0 — never deliver a stale ring later")
        check(rec.bodies[0] == payload,
              "voip body is the payload VERBATIM (no aps wrapper, no enc)")

        # The whitelist. Without it a caller POSTing {"aps": {...}} controls the
        # aps dict on the .voip topic — the exact injection the alert path is
        # immune to.
        r = await c.post("/api/push/voip", json={"device_token": TOKEN_A, "payload": {
            "type": "doorbell", "aps": {"alert": "PWNED", "badge": 9}}})
        check(r.status_code == 200 and rec.bodies[-1] == {"type": "doorbell"},
              "an 'aps' key in the voip payload is DROPPED, not forwarded (the whitelist)")
        r = await c.post("/api/push/voip", json={"device_token": TOKEN_A, "payload": {
            "type": "doorbell", "evil": "x", "content-available": 1}})
        check(rec.bodies[-1] == {"type": "doorbell"},
              "unknown voip keys are dropped")

        # `camera` renders as the CallKit handle on a lock screen. Cap it.
        r = await c.post("/api/push/voip", json={"device_token": TOKEN_A, "payload": {
            "type": "doorbell", "camera": "C" * 500}})
        check(rec.bodies[-1]["camera"] == "C" * 128,
              "a 500-char voip camera is capped at 128 (it is the CallKit handle)")

        async def voip_reject(payload, msg, status=400, reason="bad_payload"):
            n = rec.calls
            r = await c.post("/api/push/voip",
                             json={"device_token": TOKEN_A, "payload": payload})
            check(r.status_code == status and r.json()["reason"] == reason and rec.calls == n,
                  f"{msg} -> {status} {reason}, zero APNs calls")

        await voip_reject({"type": "doorbell", "camera": {"$ne": 1}},
                          "non-string voip camera")
        await voip_reject("doorbell", "voip payload that is not a dict")
        await voip_reject({}, "empty voip payload")
        # Everything whitelisted away is the same as empty.
        await voip_reject({"evil": "x"}, "a voip payload of only unknown keys")
        await voip_reject(None, "missing voip payload")

        n = rec.calls
        r = await c.post("/api/push/voip",
                         json={"device_token": "z" * 64, "payload": payload})
        check(r.status_code == 400 and r.json()["reason"] == "bad_device_token"
              and rec.calls == n, "voip non-hex token -> 400 bad_device_token, zero calls")
        r = await c.post("/api/push/voip", json={"device_token": TOKEN_A, "payload": payload,
                                                 "environment": "staging"})
        check(r.status_code == 400 and r.json()["reason"] == "bad_environment"
              and rec.calls == n, "voip environment='staging' -> 400 bad_environment, zero calls")

        # Same per-request routing as the alert path — a dev-build doorbell must
        # ring on the sandbox host.
        await c.post("/api/push/voip", json={"device_token": TOKEN_A, "payload": payload,
                                             "environment": "sandbox"})
        check(rec.requests[-1].url.host == SANDBOX_HOST,
              "voip environment routes the host per request too")

    rec2 = Recorder(status=410, body={"reason": "Unregistered"})
    app2 = make_app(rec2)
    async with client_for(app2) as c:
        r = await c.post("/api/push/voip", json={"device_token": TOKEN_A,
                                                 "payload": {"type": "doorbell"}})
        check(r.status_code == 410 and r.json()["reason"] == "unregistered",
              "voip 410 -> 410 unregistered (the backend prunes the VoIP row)")


def voip_checks() -> None:
    print("voip: whitelist, caps, expiration 0, <bundle>.voip topic")
    asyncio.run(_voip_cases())


# --------------------------------------------------------------------------- #
# 13. LOG HYGIENE
# --------------------------------------------------------------------------- #


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # a formatting bug must not hide a leak
            self.records.append(f"{record.name} <unformattable> {record.msg!r} {record.args!r}")

    @property
    def text(self) -> str:
        return "\n".join(self.records)


async def _log_traffic(rec: Recorder, app) -> None:
    async with client_for(app) as c:
        rec.status, rec.body = 200, {}
        await c.post("/api/push", json=push_body(collapse_id="42"))
        await c.post("/api/push/voip", json={"device_token": TOKEN_A,
                                             "payload": {"type": "doorbell",
                                                         "camera": "Front Door"}})
        await c.post("/api/push", json=push_body(device_token="z" * 64))  # client-side 400
        rec.status, rec.body = 400, {"reason": "BadDeviceToken"}
        await c.post("/api/push", json=push_body())                        # APNs 400
        rec.status, rec.body = 410, {"reason": "Unregistered"}
        await c.post("/api/push", json=push_body())                        # 410
        rec.status, rec.body = 500, {"reason": "InternalServerError"}
        await c.post("/api/push", json=push_body())                        # 502
        rec.status, rec.body = 403, {"reason": "InvalidProviderToken"}
        await c.post("/api/push", json=push_body())                        # 502 apns_auth


def log_checks() -> None:
    print("logs: no ciphertext, no plaintext, no full token, no .p8, no JWT, no client IP")
    root = logging.getLogger()
    cap = Capture()
    old_level = root.level
    # Deliberately un-muzzle the client libs FIRST, so the assertion below
    # proves create_app() re-muzzles them rather than inheriting a lucky state.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.DEBUG)
    root.setLevel(logging.DEBUG)
    root.addHandler(cap)
    try:
        rec = Recorder()
        app = make_app(rec, RELAY_BREAKER_THRESHOLD="0")
        check(logging.getLogger("httpx").level == logging.WARNING,
              "create_app muzzles the httpx logger to WARNING (it logs the URL — "
              "and our URL is /3/device/<FULL TOKEN> — at INFO)")
        check(logging.getLogger("httpcore").level == logging.WARNING,
              "create_app muzzles the httpcore logger to WARNING")
        asyncio.run(_log_traffic(rec, app))
        text = cap.text
    finally:
        root.removeHandler(cap)
        root.setLevel(old_level)

    check(len(cap.records) > 0, "the capture actually saw log lines (a vacuous pass would not count)")

    check(PAYLOAD_B64 not in text, "no payload_b64 CIPHERTEXT in any log line")
    check(PLAINTEXT not in text, "no notification plaintext in any log line")
    # A device token is an unguessable capability — logging it whole hands
    # anyone with log access the ability to push to that device via this relay.
    check(TOKEN_A not in text and TOKEN_A.upper() not in text,
          "no FULL device token in any log line")
    check(TOKEN_A[:8] in text,
          "the 8-char token PREFIX is present (32 bits of a >=256-bit capability — "
          "traceable, not a reconstruction path)")
    for line in [ln for ln in P8_PEM.splitlines() if ln.strip()]:
        assert line not in text, f"the .p8 leaked into a log line: {line[:20]}"
    check(True, "no line of the .p8 PEM in any log line")
    leaked_jwt = [j for j in (jwt_of(r) for r in rec.requests) if j and j in text]
    check(not leaked_jwt, "no authorization/JWT value in any log line")
    # Behind the tunnel every request carries CF-Connecting-IP. Logging the
    # HEADER DICT — the easy debugging reflex — logs client PII.
    check(CF_IP not in text, "no CF-Connecting-IP value in any log line (never log the header dict)")

    # The one thing that MUST be there. An env mismatch looks EXACTLY like a
    # BadDeviceToken, and "which host did we use" is the first question anyone
    # debugging it asks — this assertion is the whole point of the env fix.
    bad_lines = [ln for ln in cap.records if "bad_device_token" in ln]
    check(bool(bad_lines), "a BadDeviceToken from Apple produces a log line")
    check(any(PROD_HOST in ln for ln in bad_lines),
          "  and that line NAMES THE HOST it used (the env-mismatch breadcrumb)")
    check(any("env=production" in ln for ln in bad_lines),
          "  and the resolved environment")
    # The 403 collapse is a security feature on the WIRE, not in the log: the
    # operator still needs Apple's raw reason to tell BadTopic from
    # BadEnvironmentKeyIdInToken.
    check("InvalidProviderToken" in text,
          "a 403's RAW Apple reason is logged (collapsed only in the response)")


# --------------------------------------------------------------------------- #
# 14. load_config
# --------------------------------------------------------------------------- #


def _load_with(**over):
    saved = {k: os.environ.get(k) for k in over}
    try:
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return main.load_config()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _raises(**over) -> bool:
    try:
        _load_with(**over)
    except RuntimeError:
        return True
    return False


def config_checks() -> None:
    print("config: env only — path vs inline PEM, and every boot-time failure")
    cfg = _load_with(APNS_KEY_P8=str(P8_PATH))
    check(cfg.key_pem == P8_PEM, "APNS_KEY_P8 as a PATH loads the key from disk")
    check(cfg.key_id == KEY_ID and cfg.team_id == TEAM_ID and cfg.bundle_id == BUNDLE_ID
          and cfg.env == "production", "config carries key_id/team_id/bundle_id/env")

    # Supported, but a path is preferred: an inline PEM puts the signing key in
    # `docker inspect` output and in the box .env.
    # .strip()ed, because a PEM pasted into a .env almost always arrives with
    # trailing whitespace — and cryptography accepts it without the final
    # newline, so this must still be a signing key, not just a string.
    cfg = _load_with(APNS_KEY_P8=P8_PEM)
    check(cfg.key_pem == P8_PEM.strip(),
          "APNS_KEY_P8 as an inline PEM loads (sniffed on -----BEGIN)")
    signed = pyjwt.encode({"iss": TEAM_ID}, cfg.key_pem, algorithm="ES256")
    check(pyjwt.decode(signed, P8_PUB, algorithms=["ES256"])["iss"] == TEAM_ID,
          "the inline-PEM key really signs ES256 (not merely a string that loaded)")

    # The key is read at STARTUP so a chmod/mount problem fails loudly at boot
    # instead of at the first doorbell press.
    check(_raises(APNS_KEY_P8=str(TMP / "does-not-exist.p8")),
          "a missing/unreadable .p8 path raises at load_config (at BOOT, not at first push)")
    junk = TMP / "not-a-key.txt"
    junk.write_text("this is not a key\n")
    check(_raises(APNS_KEY_P8=str(junk)),
          "a readable file that is not a PEM raises at load_config")
    check(_raises(APNS_KEY_P8=""), "APNS_KEY_P8 empty raises")
    check(_raises(APNS_KEY_P8=None), "APNS_KEY_P8 unset raises")

    # An unknown APNS_ENV must not become a URL, and must not silently mean
    # production — same fixed-map discipline as the per-request value.
    check(_raises(APNS_ENV="staging"), "APNS_ENV='staging' raises at load_config")
    check(_load_with(APNS_ENV=None).env == "production", "APNS_ENV unset defaults to production")
    check(_load_with(APNS_ENV="SANDBOX").env == "sandbox", "APNS_ENV is case-insensitive")

    check(_raises(APNS_KEY_ID=None), "missing APNS_KEY_ID raises")
    check(_raises(APNS_TEAM_ID=None), "missing APNS_TEAM_ID raises")
    check(_raises(APNS_BUNDLE_ID=None), "missing APNS_BUNDLE_ID raises")
    check(_raises(RELAY_MAX_BODY="lots"), "a non-integer RELAY_MAX_BODY raises (named, at boot)")

    check(main.APNS_HOSTS == {"production": "https://api.push.apple.com",
                              "sandbox": "https://api.sandbox.push.apple.com"},
          "APNS_HOSTS is exactly the two Apple hosts — the only hosts this relay may reach")


def main_() -> None:
    health_checks()
    shape_checks()
    jwt_checks()
    cap_checks()
    bad_input_checks()
    env_checks()
    mapping_checks()
    rate_checks()
    breaker_checks()
    concurrency_checks()
    payload_size_checks()
    voip_checks()
    log_checks()
    config_checks()
    print(f"\nALL {PASS} CHECKS PASSED (relay)")


if __name__ == "__main__":
    main_()

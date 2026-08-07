"""Vigilume APNs push relay — E2E ciphertext in, APNs out.

docs/push-architecture.md is the PINNED CONTRACT: §1 trust model, §2 the E2E
scheme, §3 this API, §3a the VoIP/CallKit endpoint. If a shape here disagrees
with the doc, the doc wins — and if a change is needed, update the doc first.

WHAT THIS PROCESS HOLDS: the Apple .p8 signing key, and nothing else. It never
sees notification plaintext (that is AES-256-GCM ciphertext in `payload_b64`,
opened only on the device), never sees a camera name on the alert path, never
sees a snapshot, and stores nothing on disk. Restarts are free. Never mount a
writable volume.

THE BUG THIS REBUILD EXISTS TO NOT REPEAT: the previous relay pinned ONE APNs
host at startup (`AsyncClient(base_url=...)`) and never read `environment` —
so a sandbox (Xcode dev build) token was posted to the production host, Apple
answered BadDeviceToken, and the push silently vanished. Now: `environment` is
accepted per request, validated against the fixed APNS_HOSTS map (NEVER
interpolated into a URL — a caller must not be able to steer .p8-signed
requests at a host of their choosing), falls back to APNS_ENV only when the
field is ABSENT, and the host is chosen per request. BadDeviceToken names the
host it used, because that error is exactly what an env mismatch looks like.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("relay")

# ── Fixed facts ──────────────────────────────────────────────────────────────

# The ONLY hosts this relay will ever talk to. A dict LOOKUP (never an f-string,
# never urljoin, never base_url) is what makes a caller-supplied `environment`
# safe: an unknown value cannot become a URL, it can only fail to match.
APNS_HOSTS = {
    "production": "https://api.push.apple.com",
    "sandbox": "https://api.sandbox.push.apple.com",
}

# OUR constraint, not Apple's. Apple explicitly says do not assume a device
# token size — but a relay must reject garbage BEFORE signing anything with the
# .p8, so we band it. If Apple ever widens tokens this is a one-line fix rather
# than a mystery 400.
# fullmatch (not match): Python's `$` also matches BEFORE a trailing newline, so
# `^...$` + .match() would accept "abcd…\n" and that newline would land in the
# /3/device/<token> path. The backend's register route uses fullmatch too.
TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64,160}$")

# Apple: refresh no more than once per 20 min, no less than once per 60.
# 45 min sits centred in that window.
JWT_MAX_AGE_S = 45 * 60

# Apple's payload ceilings. We assert the CONSTRUCTED body against these rather
# than inferring them from RELAY_MAX_BODY: a 4096-byte relay body minus the JSON
# scaffolding leaves ~3850-3940 chars of payload_b64, and the `aps` envelope
# lands at ~3950-4045 against 4096. It fits, but the margin is ~50-150 bytes and
# nothing else enforces it. PayloadTooLarge (413) is permanent and non-retryable.
APNS_ALERT_MAX = 4096
APNS_VOIP_MAX = 5120

# §3a: the VoIP payload is the ONLY caller-controlled JSON structure that
# reaches Apple. Whitelist it. Without this, a caller POSTing {"aps": {...}}
# controls the aps dict on the .voip topic — the exact injection the alert path
# is immune to (there, `aps` is a server-built literal and caller bytes only
# ever land in "enc"). `camera` is rendered as the CallKit handle on a lock
# screen: cap it, and treat it as untrusted display text on the iOS side too.
_VOIP_FIELDS = {"type": 32, "camera": 128, "event_id": 128}


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    key_pem: str
    key_id: str
    team_id: str
    bundle_id: str
    env: str
    max_body: int
    rate_limit: int
    rate_window: int
    max_concurrency: int
    apns_ttl: int
    global_rate_limit: int
    max_limiter_keys: int
    acquire_timeout: float
    breaker_threshold: int
    breaker_cooldown: int


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer (got {raw!r})") from exc


def load_config() -> Config:
    """Env ONLY — this process has no config file and no DB.

    APNS_KEY_P8 is a PATH **or** an inline PEM (sniffed for the PEM header; a
    path never contains one). Prefer the path: an inline PEM puts the signing
    key in `docker inspect` output and in the box .env.

    The key is read HERE, at startup, so a chmod/mount problem fails loudly at
    boot instead of at the first doorbell press.
    """
    raw = (os.environ.get("APNS_KEY_P8") or "").strip()
    if not raw:
        raise RuntimeError("APNS_KEY_P8 is required (path to the .p8, or inline PEM)")
    if "-----BEGIN" in raw:
        key_pem = raw
    else:
        try:
            with open(raw, "r", encoding="utf-8") as fh:
                key_pem = fh.read()
        except OSError as exc:
            # The overwhelmingly likely cause: the relay runs USER nobody and
            # the key is not world-readable, or ./secrets was bind-mounted as a
            # FILE (which fails on Unraid's shfs) instead of a DIRECTORY.
            raise RuntimeError(
                f"APNS_KEY_P8 path {raw!r} is not readable ({exc}) — the relay "
                "runs as USER nobody; the key must be world-readable and its "
                "DIRECTORY mounted (./secrets:/keys:ro)"
            ) from exc
    if "-----BEGIN" not in key_pem:
        raise RuntimeError("APNS_KEY_P8 does not look like a PEM private key")

    key_id = (os.environ.get("APNS_KEY_ID") or "").strip()
    team_id = (os.environ.get("APNS_TEAM_ID") or "").strip()
    bundle_id = (os.environ.get("APNS_BUNDLE_ID") or "").strip()
    for name, val in (("APNS_KEY_ID", key_id), ("APNS_TEAM_ID", team_id),
                      ("APNS_BUNDLE_ID", bundle_id)):
        if not val:
            raise RuntimeError(f"{name} is required")

    env = (os.environ.get("APNS_ENV") or "production").strip().lower()
    if env not in APNS_HOSTS:
        raise RuntimeError(f"APNS_ENV must be one of {sorted(APNS_HOSTS)} (got {env!r})")

    return Config(
        key_pem=key_pem,
        key_id=key_id,
        team_id=team_id,
        bundle_id=bundle_id,
        env=env,
        max_body=_int_env("RELAY_MAX_BODY", 4096),
        rate_limit=_int_env("RELAY_RATE_LIMIT", 30),
        rate_window=_int_env("RELAY_RATE_WINDOW", 60),
        max_concurrency=_int_env("RELAY_MAX_CONCURRENCY", 32),
        apns_ttl=_int_env("RELAY_APNS_TTL", 1800),
        # A GLOBAL bound. The per-token limiter is keyed on caller-controlled
        # input, so it provides NO global protection: 10k distinct hex strings
        # each get a fresh 30/min window. 0 disables.
        global_rate_limit=_int_env("RELAY_GLOBAL_RATE_LIMIT", 600),
        max_limiter_keys=_int_env("RELAY_MAX_LIMITER_KEYS", 10000),
        acquire_timeout=float(_int_env("RELAY_ACQUIRE_TIMEOUT_MS", 2000)) / 1000.0,
        breaker_threshold=_int_env("RELAY_BREAKER_THRESHOLD", 50),
        breaker_cooldown=_int_env("RELAY_BREAKER_COOLDOWN", 60),
    )


# ── JWT ──────────────────────────────────────────────────────────────────────

class JWTProvider:
    """ES256 provider token, cached; re-signed past JWT_MAX_AGE_S.

    HOST-INDEPENDENT — one token serves BOTH APNs hosts, so this cache is keyed
    on NOTHING. That was universally true until 2025-02-17, when Apple shipped
    environment-scoped keys. A LEGACY key (created before then, or any existing
    key) still works for both environments; a key created today can be pinned to
    Sandbox or Production. So host-independence is now a property of the
    OPERATOR'S KEY, not of APNs. A hoster who mints a Sandbox-scoped key gets
    403 BadEnvironmentKeyIdInToken the moment we route to production — which is
    why that reason is logged loudly and distinctly below. The clean eventual
    answer is per-environment keys (APNS_KEY_P8_SANDBOX/_PRODUCTION); v2.

    *** INVARIANT: THE CACHE KEY MUST NEVER TAKE A PER-REQUEST INPUT. ***
    There is nothing to key on today, which is exactly why this needs saying: if
    v2 keys this cache on the caller's `environment`, then (a) an attacker
    alternating environments thrashes it into signing on EVERY request, and (b)
    an unvalidated value grows the dict without bound. Validate against
    APNS_HOSTS first, key on the RESULT, never on the input.

    Signing is off the request path entirely: one ES256 signature per 45 min
    regardless of volume. Apple's TooManyProviderTokenUpdates (which is 429, not
    403, and is scoped per CONNECTION) is unreachable from a single cached JWT.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._token: Optional[str] = None
        self._issued_at = 0.0

    def token(self) -> str:
        now = time.time()
        if self._token is None or now - self._issued_at > JWT_MAX_AGE_S:
            self._token = jwt.encode(
                {"iss": self._cfg.team_id, "iat": int(now)},
                self._cfg.key_pem,
                algorithm="ES256",
                headers={"kid": self._cfg.key_id},
            )
            self._issued_at = now
        return self._token


# ── Rate limiting ────────────────────────────────────────────────────────────

class SlidingWindowLimiter:
    """Per-key sliding window, BOUNDED.

    The bound is not hygiene, it is a fix. Keys are device tokens — i.e.
    caller-controlled input — in a process that deliberately holds all state in
    memory. Unbounded, a caller sending distinct valid-looking tokens grows this
    dict until the container OOMs. LRU eviction is safe here precisely because
    evicting only ever LOOSENS a limit, and the global limiter is the real
    backstop.

    monotonic(), not time(): a clock step must not open the window.
    """

    def __init__(self, limit: int, window: int, max_keys: int = 10000) -> None:
        self._limit = limit
        self._window = window
        self._max_keys = max_keys
        self._hits: "OrderedDict[str, deque[float]]" = OrderedDict()

    def allow(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        now = time.monotonic()
        dq = self._hits.get(key)
        if dq is None:
            dq = deque()
            self._hits[key] = dq
        self._hits.move_to_end(key)
        cutoff = now - self._window
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= self._limit:
            return False
        dq.append(now)
        while len(self._hits) > self._max_keys:
            self._hits.popitem(last=False)  # LRU
        return True


class BadTokenBreaker:
    """Protects the TEAM's APNs reputation — shared-fate, and nothing else guards it.

    Every Vigilume install on earth shares ONE bundle id and ONE .p8. Apple
    throttles throughput on too many 4XX and DISCONNECTS on excess errors, with
    BadDeviceToken disconnecting SOONER than others. So an anonymous shell loop
    of junk tokens degrades APNs delivery for every hoster at once. The
    per-token limiter cannot see this (the attacker picks the key).

    Deliberately trips HIGH and logs loudly rather than tripping tight: this is
    also a self-DoS lever (trip it and the owner's real pushes stop). At the
    default of 50 BadDeviceTokens in 60s, a one-device owner will never see it;
    a 10k-token sweep will. RELAY_BREAKER_THRESHOLD=0 disables.
    """

    def __init__(self, threshold: int, window: int, cooldown: int) -> None:
        self._threshold = threshold
        self._window = window
        self._cooldown = cooldown
        self._events: "deque[float]" = deque()
        self._open_until = 0.0

    def is_open(self) -> bool:
        return self._threshold > 0 and time.monotonic() < self._open_until

    def record_bad_token(self) -> None:
        if self._threshold <= 0:
            return
        now = time.monotonic()
        self._events.append(now)
        cutoff = now - self._window
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()
        if len(self._events) >= self._threshold and not self.is_open():
            self._open_until = now + self._cooldown
            self._events.clear()
            log.error(
                "CIRCUIT BREAKER OPEN: %d BadDeviceToken in %ds — refusing pushes "
                "for %ds to protect this team's APNs reputation. Either a caller "
                "is sweeping junk tokens, or an APNS_ENV/environment mismatch is "
                "making every real token look dead.",
                self._threshold, self._window, self._cooldown,
            )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _err(status: int, reason: str) -> JSONResponse:
    """The contract's uniform error envelope (§3). `reason` is a CLOSED
    vocabulary and never echoes caller input — a reflected device_token would
    land in the caller's logs and log aggregator."""
    return JSONResponse({"ok": False, "reason": reason}, status_code=status)


async def _read_capped(request: Request, cap: int) -> Optional[bytes]:
    """Body cap, enforced BEFORE anything else parses or allocates.

    Content-Length is a FAST-REJECT OPTIMISATION, NEVER THE BOUND — it is
    attacker-controlled and can lie LOW ("Content-Length: 100", then stream
    100 MB). The stream-and-abort loop is the sole authority and always runs.
    Never `await request.body()` on the strength of a small declared length.

    Both paths are live in production, not theoretical: behind a Cloudflare
    Tunnel, cloudflared's `disableChunkedEncoding` defaults to false (chunked
    passes through -> the stream path fires), but a hoster who flips it to true
    makes cloudflared buffer and present a Content-Length (-> the fast path
    fires). Which branch runs depends on someone else's config, so both are
    smoke-tested.

    Honest scope: once traffic is inside Cloudflare the abort no longer saves
    upstream bandwidth — it only protects THIS process's memory. That is still
    worth having; don't claim more for it.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > cap:
                return None
        except ValueError:
            return None
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > cap:
            return None  # abort the instant the cap is crossed
    return bytes(buf)


def _clean_voip_payload(payload: Any) -> Optional[dict[str, str]]:
    """Whitelist + cap. Unknown keys are DROPPED, not forwarded — see _VOIP_FIELDS."""
    if not isinstance(payload, dict):
        return None
    out: dict[str, str] = {}
    for key, cap in _VOIP_FIELDS.items():
        val = payload.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            return None
        out[key] = val[:cap]
    if not out:
        return None
    return out


# ── App factory ──────────────────────────────────────────────────────────────

def create_app(transport: Optional[httpx.AsyncBaseTransport] = None) -> FastAPI:
    """`transport` is injectable so the smoke suite can drive APNs with an
    httpx.MockTransport and stay entirely offline."""
    cfg = load_config()
    app = FastAPI(title="Vigilume APNs relay", docs_url=None, redoc_url=None)

    # *** LOG HYGIENE ***
    # httpx logs the request URL at INFO, and our URL is /3/device/<FULL TOKEN>
    # — a device token is an unguessable capability. Muzzle the client libs.
    # This is a mitigation of a LIBRARY DEFAULT: any new HTTP client added here
    # needs the same treatment.
    # Our own lines carry an 8-char token prefix (32 bits of a >=256-bit
    # capability — not a reconstruction path) plus the apns-id UUID.
    # NEVER log: payloads, ciphertext, full tokens, the .p8, or THE HEADER DICT
    # — behind the tunnel every request carries CF-Connecting-IP, which is
    # client PII. The key_id IS safe to log: it rides in the JWT `kid` header in
    # cleartext on every APNs request anyway.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    jwt_provider = JWTProvider(cfg)
    per_token = SlidingWindowLimiter(cfg.rate_limit, cfg.rate_window, cfg.max_limiter_keys)
    global_limiter = SlidingWindowLimiter(cfg.global_rate_limit, cfg.rate_window, max_keys=1)
    breaker = BadTokenBreaker(cfg.breaker_threshold, cfg.rate_window, cfg.breaker_cooldown)
    sem = asyncio.Semaphore(cfg.max_concurrency)

    # NO base_url. The host is chosen PER REQUEST — pinning one here at startup
    # is precisely the bug that ate the sandbox push.
    # http2=(transport is None): APNs is HTTP/2-ONLY, so h2 is mandatory in
    # production (httpx[http2]); tests inject a transport and need no h2.
    # An explicit timeout is not optional — the semaphore bounds concurrency but
    # a hung request would hold its slot forever.
    client = httpx.AsyncClient(
        http2=(transport is None),
        transport=transport,
        timeout=httpx.Timeout(10.0, connect=5.0),
    )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    def _resolve_host(value: Any) -> tuple[Optional[str], Optional[str]]:
        """Fixed-map lookup. Returns (host, env) or (None, None) when the caller
        sent an unrecognised value.

        ABSENT -> fall back to APNS_ENV. PRESENT BUT UNKNOWN -> 400
        bad_environment, NOT a silent fallback to production: a silent fallback
        reproduces the exact debugging experience of the bug this fix exists to
        kill (push accepted, Apple says BadDeviceToken, notification vanishes).
        """
        if value is None:
            return APNS_HOSTS[cfg.env], cfg.env
        if not isinstance(value, str):
            return None, None
        env = value.strip().lower()
        host = APNS_HOSTS.get(env)
        if host is None:
            return None, None
        return host, env

    async def _forward(
        *, host: str, env: str, token: str, apns_body: dict[str, Any],
        headers: dict[str, str], max_payload: int, kind: str,
    ) -> JSONResponse:
        tok8 = token[:8]
        apns_id = headers["apns-id"]

        # Assert the CONSTRUCTED payload, and send exactly what we measured.
        raw = json.dumps(apns_body, separators=(",", ":")).encode("utf-8")
        if len(raw) > max_payload:
            log.warning("%s too_large token=%s… bytes=%d", kind, tok8, len(raw))
            return _err(400, "too_large")

        if breaker.is_open():
            return _err(429, "rate_limited")

        # Shed load at the door. A gate with an infinite waiting room is a
        # memory amplifier wearing a safety vest: without this, 33+ concurrent
        # requests each park holding a task, a socket and a buffered body for
        # the full httpx timeout, and Cloudflare 524s the caller at 120s anyway.
        try:
            await asyncio.wait_for(sem.acquire(), timeout=cfg.acquire_timeout)
        except asyncio.TimeoutError:
            return _err(429, "rate_limited")
        try:
            try:
                resp = await client.post(
                    f"{host}/3/device/{token}", content=raw, headers=headers,
                )
            except httpx.HTTPError as exc:
                log.warning("%s apns unreachable token=%s… id=%s: %s",
                            kind, tok8, apns_id, type(exc).__name__)
                return _err(502, "apns_unreachable")
        finally:
            sem.release()

        try:
            data = resp.json()
            reason = str(data.get("reason", "")) if isinstance(data, dict) else ""
        except (json.JSONDecodeError, ValueError):
            reason = ""

        if resp.status_code == 200:
            return JSONResponse({"ok": True})

        # 410 carries BOTH Unregistered AND ExpiredToken. Key on the STATUS,
        # never the reason string, or dead tokens leak. Apple explicitly does
        # not count 410 as an error condition.
        if resp.status_code == 410:
            log.info("%s unregistered token=%s… id=%s (%s)", kind, tok8, apns_id, reason)
            return _err(410, "unregistered")

        if resp.status_code == 413 or reason == "PayloadTooLarge":
            return _err(400, "too_large")

        if resp.status_code == 400:
            if reason in ("BadDeviceToken", "DeviceTokenNotForTopic"):
                # Name the HOST. This error is exactly what an environment
                # mismatch looks like, and "which host did we use" is the first
                # question anyone debugging it will ask.
                breaker.record_bad_token()
                log.warning(
                    "%s bad_device_token token=%s… id=%s host=%s env=%s reason=%s "
                    "(a sandbox token sent to production, or vice versa, looks "
                    "EXACTLY like this)",
                    kind, tok8, apns_id, host, env, reason,
                )
                return _err(400, "bad_device_token")
            log.warning("%s apns 400 token=%s… id=%s reason=%s", kind, tok8, apns_id, reason)
            return _err(400, "bad_payload")

        if resp.status_code == 403:
            # Log Apple's RAW reason, return the collapsed one. The collapse is
            # a security feature, not tidiness: it denies a caller an oracle for
            # how this .p8 is scoped (see JWTProvider). But BadTopic,
            # InvalidPushType and BadEnvironmentKeyIdInToken are all
            # misconfigurations that look identical under a collapsed reason, so
            # the operator needs the raw string in the log.
            if reason in ("BadEnvironmentKeyIdInToken", "UnrelatedKeyIdInToken"):
                log.error(
                    "APNS KEY SCOPE ERROR (%s) host=%s env=%s — your APNs key is "
                    "environment-scoped (Apple, 2025-02-17) and does not cover "
                    "this environment. Use a legacy/dual-environment key, or run "
                    "one relay per environment with its own key.",
                    reason, host, env,
                )
            else:
                log.error("apns auth failure (%s) — check APNS_KEY_P8/APNS_KEY_ID/"
                          "APNS_TEAM_ID", reason)
            return _err(502, "apns_auth")

        if resp.status_code == 429:
            log.warning("%s apns rate limited token=%s… id=%s", kind, tok8, apns_id)
            return _err(429, "apns_rate_limited")

        log.warning("%s apns error token=%s… id=%s status=%s reason=%s",
                    kind, tok8, apns_id, resp.status_code, reason)
        return _err(502, "apns_error")

    # ── /healthz ─────────────────────────────────────────────────────────────

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        # no-store is cheap insurance, not a fix for a known bug. Cloudflare
        # does not cache JSON by default and this is extensionless — but that is
        # a DEFAULT, and a hoster with a "Cache Everything" rule would serve a
        # stale {"ok":true} forever. A health check that lies is worse than none.
        return JSONResponse({"ok": True, "env": cfg.env},
                            headers={"Cache-Control": "no-store"})

    # ── /api/push ────────────────────────────────────────────────────────────

    @app.post("/api/push")
    async def push(request: Request) -> JSONResponse:
        # Validation order is deliberate: cap -> json -> dict -> token ->
        # payload -> priority -> collapse_id -> environment -> rate limit ->
        # send. Nothing signs, allocates or reaches Apple until the input is
        # known-good.
        raw = await _read_capped(request, cfg.max_body)
        if raw is None:
            return _err(400, "too_large")
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _err(400, "bad_payload")
        if not isinstance(body, dict):
            return _err(400, "bad_payload")

        token = body.get("device_token")
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
            # LOG IT. `bad_device_token` is returned from two places that mean
            # opposite things: HERE (the string never passed the format gate —
            # Apple was never contacted) and in _forward() (Apple itself
            # rejected a well-formed token: an environment/topic mismatch).
            # Silent here, they are indistinguishable to the operator, who then
            # goes hunting an APNs environment bug that does not exist. That
            # cost a real debugging session. Never log the token itself — it is
            # a capability and it may be someone's real one; length + shape is
            # what identifies the mistake (a truncated copy-paste is the usual
            # culprit) without putting it in the log.
            log.warning(
                "alert bad_device_token REJECTED ON FORMAT (Apple NOT contacted): "
                "type=%s len=%s — expected 64-160 hex chars, ^[0-9a-fA-F]{64,160}$",
                type(token).__name__, len(token) if isinstance(token, str) else "n/a",
            )
            return _err(400, "bad_device_token")
        # Lowercase, so the backend's 410-prune lookups match what it stored
        # (contract §2: the backend stores lowercased "the relay lowercases too").
        token = token.lower()

        payload_b64 = body.get("payload_b64")
        if not isinstance(payload_b64, str) or not payload_b64:
            return _err(400, "bad_payload")
        try:
            base64.b64decode(payload_b64, validate=True)
        except (binascii.Error, ValueError):
            return _err(400, "bad_payload")

        priority = body.get("priority", "high")
        if priority not in ("high", "normal"):
            return _err(400, "bad_priority")

        collapse_id = body.get("collapse_id")
        if collapse_id is not None:
            if not isinstance(collapse_id, str):
                return _err(400, "bad_collapse_id")
            # Apple's limit is 64 BYTES, not characters — a 64-char string with
            # any non-ASCII exceeds it and APNs answers 400 BadCollapseId.
            if len(collapse_id.encode("utf-8")) > 64:
                return _err(400, "bad_collapse_id")

        host, env = _resolve_host(body.get("environment"))
        if host is None:
            return _err(400, "bad_environment")

        if not global_limiter.allow("*") or not per_token.allow(token):
            return _err(429, "rate_limited")

        apns_id = str(uuid.uuid4())
        headers = {
            "authorization": f"bearer {jwt_provider.token()}",
            # PINNED SERVER-SIDE, never caller-supplied — §1 names this as the
            # security invariant that makes a public relay safe: APNs only
            # honours a token for this bundle id. A caller who could set
            # apns-topic could aim the owner's .p8 at another app's tokens.
            "apns-topic": cfg.bundle_id,
            "apns-push-type": "alert",
            "apns-priority": "10" if priority == "high" else "5",
            "apns-expiration": str(int(time.time()) + cfg.apns_ttl),
            "apns-id": apns_id,  # Apple-recommended; makes logs traceable, leaks nothing
            "content-type": "application/json",
        }
        if collapse_id:
            headers["apns-collapse-id"] = collapse_id

        # A LITERAL. Never merge caller JSON into `aps`. mutable-content:1 is
        # what triggers the iOS notification service extension; the alert is the
        # fallback shown when decryption fails — which is also what an attacker
        # with a stolen token gets, since their bytes fail GCM authentication.
        # Their `enc` is opaque to us and unopenable by them: spam, not spoofing.
        apns_body = {
            "aps": {
                "mutable-content": 1,
                "alert": {"title": "Vigilume", "body": "Encrypted notification"},
            },
            "enc": payload_b64,
        }
        return await _forward(host=host, env=env, token=token, apns_body=apns_body,
                              headers=headers, max_payload=APNS_ALERT_MAX, kind="push")

    # ── /api/push/voip ───────────────────────────────────────────────────────

    @app.post("/api/push/voip")
    async def push_voip(request: Request) -> JSONResponse:
        """PushKit/CallKit doorbell ring (contract §3a).

        *** WHY `payload` IS A PLAINTEXT DICT AND `payload_b64` IS NOT. ***
        This asymmetry is FORCED, not an oversight. Do not "unify" the two
        paths. A VoIP push MUST be reported to CallKit via
        CXProvider.reportNewIncomingCall essentially immediately: the PushKit
        delegate has no notification-service-extension window in which to
        decrypt. Fail to report and iOS TERMINATES the app; keep failing and iOS
        STOPS DELIVERING VoIP PUSHES ENTIRELY. So the payload cannot be
        ciphertext, and the mitigation is constraint (whitelist + caps above),
        not encryption.

        Consequence to be honest about: §1's "the content is ciphertext the
        relay cannot read" does NOT cover this endpoint. Minimise what goes in
        the payload, and see the README on putting Access in front of this route
        specifically.
        """
        raw = await _read_capped(request, cfg.max_body)
        if raw is None:
            return _err(400, "too_large")
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _err(400, "bad_payload")
        if not isinstance(body, dict):
            return _err(400, "bad_payload")

        token = body.get("device_token")
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
            # See the identical guard on the alert path: silent, this is
            # indistinguishable from Apple rejecting a well-formed token.
            log.warning(
                "voip bad_device_token REJECTED ON FORMAT (Apple NOT contacted): "
                "type=%s len=%s — expected 64-160 hex chars, ^[0-9a-fA-F]{64,160}$",
                type(token).__name__, len(token) if isinstance(token, str) else "n/a",
            )
            return _err(400, "bad_device_token")
        token = token.lower()

        payload = _clean_voip_payload(body.get("payload"))
        if payload is None:
            return _err(400, "bad_payload")

        host, env = _resolve_host(body.get("environment"))
        if host is None:
            return _err(400, "bad_environment")

        if not global_limiter.allow("*") or not per_token.allow(token):
            return _err(429, "rate_limited")

        apns_id = str(uuid.uuid4())
        headers = {
            "authorization": f"bearer {jwt_provider.token()}",
            "apns-topic": f"{cfg.bundle_id}.voip",  # Apple's fixed suffix
            "apns-push-type": "voip",
            "apns-priority": "10",
            # *** 0, NOT now+apns_ttl. *** Apple explicitly directs VoIP requests
            # to expire immediately or within a few seconds, so the system does
            # not deliver a stale call later. The old contract applied
            # now+1800 uniformly: a 30-minute-old doorbell ring firing as a live
            # CallKit call is a real defect, and an attack amplifier (queue rings
            # at an offline phone, they all land when it comes back).
            "apns-expiration": "0",
            "apns-id": apns_id,
            "content-type": "application/json",
        }
        return await _forward(host=host, env=env, token=token, apns_body=payload,
                              headers=headers, max_payload=APNS_VOIP_MAX, kind="voip")

    return app

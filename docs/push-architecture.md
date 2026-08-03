# Vigilume Push Architecture (iOS / APNs)

**Status: PINNED CONTRACT — backend implemented.** The hoster-backend side of
§2 and §4 is implemented (`backend/app/notify/apns.py`,
`routers/notifications.py` APNs routes, `apns_devices` table at schema v7,
`settings.notifications.apns`, and the events-pipeline hook — smoke-tested by
`backend/tests/apns_smoke.py`). If a change is needed, update this doc first.

> **THE RELAY IS BACK; `mode: "direct"` IS RETIRED (2026-07-16).** This reverses
> a retirement notice that stood in this file for part of one day — read the
> history before re-litigating it:
>
> 1. The relay was retired in favour of **ntfy**, on the reasoning that nobody
>    should have to host a `.p8` on strangers' behalf.
> 2. That shipped, and ntfy turned out **not to be good enough**: on iOS its
>    alerts land in the *ntfy app*. No CallKit doorbell ring, no native UI. The
>    doorbell ring is the product.
> 3. So `relay/`, the `push-relay` compose service and `apns.mode = "relay"` are
>    **restored**, and `mode: "direct"` — this server holding its own Apple
>    key — is **deleted** instead. Direct was the fallback for a problem the
>    relay solves; keeping two APNs transports meant two send paths to test.
>
> **ntfy is not going anywhere** (§7). It is the no-Apple-account channel and it
> stays beside the relay. Modes are now `relay` | `off`.
>
> A stored `mode: "direct"` is migrated to **`off`** in
> `settings_store._strip_legacy` (never to `relay` — relay with an empty
> `relay_url` errors on every event, and we must never silently start pushing
> through a relay the admin never configured). That migration is load-bearing:
> `ApnsSettings.mode` is a pydantic `Literal`, so an unmigrated blob would 422
> **every** settings save and lock the admin out of the settings page —
> including out of changing the mode. Migrating on load *and* on save means the
> dead value can never reach the validator. The visible cost is real: APNs push
> stops until the admin sets `mode=relay` + a `relay_url` (ntfy and web push are
> unaffected). A stop you can see beats one you can't.

## 1. Flow and trust model

Vigilume's iOS app belongs to one developer (bundle id, Apple team, and the
APNs `.p8` auth key are his), but **any** self-hosted Vigilume server must be
able to push to it — with a real native notification and a CallKit doorbell
ring. This mirrors Home Assistant's mobile push: a public relay run by the app
owner holds the Apple credentials and hosters talk to it over plain HTTPS.

```
┌──────────────────┐   HTTPS POST /api/push    ┌──────────────────┐  HTTP/2 + ES256 JWT  ┌──────┐
│  hoster server    │ ────────────────────────> │  push relay      │ ───────────────────> │ APNs │
│ (any Vigilume     │  {device_token,           │ (app owner runs  │  POST /3/device/<t>  └──┬───┘
│  install)         │   payload_b64=ciphertext} │  it publicly;    │                         │
│                   │                           │  holds the .p8)  │                         ▼
│  holds: per-reg   │                           │  sees: token +   │                    ┌─────────┐
│  E2E key, events, │ <─── ext fetches snapshot │  ciphertext only │                    │ device  │
│  snapshots        │      directly (tokened    └──────────────────┘                    │ decrypts│
└──────────────────┘       URL, not relayed)                                            └─────────┘
```

Who holds what:

| Secret | Hoster server | Relay | iOS app |
| --- | --- | --- | --- |
| APNs `.p8` key / Key ID / Team ID | never | **yes** | never |
| Per-registration E2E key (32 bytes) | **yes** (DB) | never | **yes** (Keychain) |
| Notification plaintext (title/body/event/snapshot URL) | yes | never (ciphertext only) | yes (after decrypt) |
| APNs device token | yes (DB) | in transit only | yes |

Why the relay is safe to expose publicly: device tokens are unguessable
random capabilities that APNs only honors for this app's bundle id
(`apns-topic` is pinned server-side in the relay); the content is ciphertext
the relay cannot read; the relay stores nothing and enforces rate limits.
Snapshot images never transit the relay.

## 2. E2E encryption scheme (exact)

**Registration.** The app generates a random 32-byte key
(`SymmetricKey(size: .bits256)`), stores it in the Keychain, and registers
with the **hoster** server:

```
POST /api/notifications/apns/register
{
  "device_token": "<hex APNs token>",
  "device_name": "Adam's iPhone",
  "key_b64": "<base64 of the 32-byte key>",
  "environment": "production"        // or "sandbox" (Xcode dev builds); optional, default "production"
}
```

Backend validation on register: `device_token` must match
`^[0-9a-fA-F]{64,160}$` and is stored **lowercased** (the relay lowercases
too, so 410-prune lookups match); `key_b64` must base64-decode to exactly
32 bytes; `device_name` is a display string, cap at 64 chars; `environment`
must be `sandbox` or `production`. Reject violations with 400.

`environment` is stored **per device** and forwarded on every push (§3): the
origin server is the only party that knows how each device registered, so one
relay instance serves both environments and a sandbox phone and a TestFlight
phone can share it. (Earlier drafts ran one relay per environment. They don't
have to.)

The hoster server stores `(device_token, device_name, key_b64, environment,
created_at)` in its DB (`apns_devices`, schema v7). Re-registering the same
`device_token` upserts (token may rotate; key may rotate — latest wins).
`DELETE .../register` is idempotent: 204 whether or not the token existed.

**Encryption (hoster server, per notification).**

- Plaintext: UTF-8 JSON
  `{"title": str, "body": str, "event_id": str, "snapshot_url": str|null,
  "camera": str?, "camera_label": str?}`
  — `snapshot_url` is an absolute URL pointing at the hoster server, carrying
  its own auth token, valid long enough for the extension to fetch (~5 min).
  `camera` (the camera slug) and `camera_label` (its friendly name) are
  **optional** and, when present, let the iOS extension group same-camera
  notifications: `camera` becomes the notification `threadIdentifier` (iOS
  stacks a thread and lets the user tap through each event) and `camera_label`
  becomes the collapsed-group `summaryArgument` (e.g. "3 more from Backyard").
  They travel INSIDE the ciphertext — the relay never sees the camera (privacy
  invariant). Back-compat: older servers omit both keys and the extension
  falls back to a constant thread id; the generic undecryptable-fallback
  notification is never grouped. Both keys are tiny; the 2500-byte plaintext
  budget still truncates `body` first.
- Cipher: **AES-256-GCM**, key = the registered 32-byte key, fresh random
  **12-byte nonce** per message, no AAD.
- Wire format: `ciphertext_b64 = base64(nonce || ciphertext || 16-byte tag)`
  (i.e. nonce first, GCM tag appended — this is CryptoKit's
  `AES.GCM.SealedBox.combined` layout, so the extension can do
  `AES.GCM.open(try SealedBox(combined: data), using: key)` directly).
- That `ciphertext_b64` is exactly the `payload_b64` sent to the relay.
- **Size budget:** the relay caps the whole request at 4096 bytes and APNs
  caps the wrapped alert payload at 4 KB. base64 of `12 + len(plaintext) +
  16` bytes plus the JSON envelope means: keep the plaintext JSON ≤ **2500
  bytes** (truncate `body` first if needed). Typical payloads are ~200 bytes.

**Decryption (iOS notification service extension).** Read `enc` from the
push userinfo, base64-decode, open the sealed box with the Keychain key,
parse the JSON, replace the notification title/body, fetch `snapshot_url`
from the hoster server (direct, never via relay) and attach it. On any
failure, deliver the pushed fallback alert unchanged
(`Vigilume / Encrypted notification`).

## 3. Relay API (implemented in `relay/`)

`POST {relay_url}/api/push` — request body (JSON, ≤ `RELAY_MAX_BODY`,
default 4096 bytes; this keeps the wrapped APNs payload under Apple's 4 KB
alert limit):

```json
{
  "device_token": "hex, 64-160 chars",
  "payload_b64": "<ciphertext_b64>",
  "priority": "high" | "normal",   // optional, default "high" (APNs 10 vs 5)
  "collapse_id": "string <= 64",   // optional, maps to apns-collapse-id
  "environment": "sandbox" | "production"  // optional; falls back to APNS_ENV
}
```

**Always send `environment`.** Only the origin server knows how each device
registered (§2 stores it per device). Omitting it falls back to the relay's
`APNS_ENV`, and a wrong environment is the worst failure mode in this whole
document: Apple rejects a sandbox token at the production host with
`BadDeviceToken` and **the push simply vanishes** — no error the user ever
sees. An unknown value is `400 bad_environment`; the host map is a fixed
two-entry constant, never interpolated from the request.

The relay forwards to APNs (`api.push.apple.com` or sandbox) as an alert
push with headers `apns-topic=<bundle id>`, `apns-push-type=alert`,
`apns-priority=10|5`, `apns-expiration=now+1800`, ES256 JWT auth refreshed
every 45 min, and body:

```json
{
  "aps": {
    "mutable-content": 1,
    "alert": {"title": "Vigilume", "body": "Encrypted notification"}
  },
  "enc": "<payload_b64>"
}
```

Responses the hoster backend must handle:

| Status | Body | Backend action |
| --- | --- | --- |
| 200 | `{"ok": true}` | delivered |
| 400 | `{"ok": false, "reason": "bad_device_token" \| "bad_payload" \| "too_large" \| ...}` | log; if `bad_device_token`, prune the registration |
| 410 | `{"ok": false, "reason": "unregistered"}` | **delete the registration** (APNs says the token is dead) |
| 429 | `{"ok": false, "reason": "rate_limited" \| "apns_rate_limited"}` | drop or back off; do not retry-storm |
| 502 | `{"ok": false, "reason": "apns_error" \| "apns_auth" \| "apns_unreachable"}` | transient; retry with backoff (max ~3) |

`GET {relay_url}/healthz` → `{"ok": true, "env": "sandbox"|"production"}`,
`Cache-Control: no-store`. `env` is the **fallback** default, not a constraint.

## 3a. VoIP API — the doorbell ring (implemented in `relay/`)

This is the reason the relay exists and ntfy cannot replace it. A VoIP push
wakes the app via PushKit even when it is not running, and the app reports it to
CallKit — the doorbell rings like a phone call, full-screen, on a locked phone.
`POST {relay_url}/api/push/voip`:

```json
{
  "device_token": "hex, 64-160 chars",     // the PushKit token, NOT the APNs alert token
  "payload": {"type": "doorbell", "camera": "front_door", "event_id": "..."},
  "environment": "sandbox" | "production"  // optional; same rules as §3
}
```

**The VoIP payload is plaintext, by force.** PushKit hands the dictionary to the
app, and iOS requires the CallKit report to happen in that same wake — there is
no notification-service-extension hop to decrypt in, as there is on the alert
path (§2). So this path, and only this path, shows the relay a camera slug.
That is a real and deliberate hole in §1's privacy invariant: **the relay
operator can see which camera rang and when.** Nothing else — no snapshot, no
title, no body. Callers who won't accept that should send `type` only.

**`payload` is whitelist-rebuilt, never forwarded.** `_VOIP_FIELDS` in
`relay/main.py` is the whole contract:

| Key | Cap | Meaning |
| --- | --- | --- |
| `type` | 32 chars | `"doorbell"` — the only value the backend sends. The relay caps the length but does NOT validate the value, and `CallManager.swift` never reads it; it is a discriminator for future push kinds, not a check. |
| `camera` | 128 chars | camera slug, for the CallKit caller name |
| `event_id` | 128 chars | correlates the ring with the event |

Unknown keys are **dropped**. Non-string values are `400`. An empty result is
`400`. The relay constructs the outgoing dict from scratch, because unlike the
alert path — where the caller's bytes land in an opaque `enc` string — here the
caller's JSON *is* the `aps`-adjacent dict. Merging it would hand any origin
server control of the payload iOS acts on.

Headers the relay pins (never the caller): `apns-topic = <bundle_id>.voip`
(Apple's fixed suffix — a VoIP push to the plain bundle id is rejected),
`apns-push-type = voip`, `apns-priority = 10`, and **`apns-expiration = 0`**.
Zero is deliberate: a doorbell ring that arrives after the visitor left is worse
than no ring, and iOS terminates an app that takes a VoIP push without reporting
a call. Never let it queue. Size cap is `APNS_VOIP_MAX` (5120), Apple's VoIP
limit — larger than the 4 KB alert limit.

Responses match §3's table.

## 4. Backend deltas (implemented — see docs/CONTRACTS.md for the API surface)

Endpoints (hoster server, authenticated like the rest of the API):

```
POST   /api/notifications/apns/register    {device_token, device_name, key_b64}
DELETE /api/notifications/apns/register    {device_token}          # unregister
GET    /api/notifications/apns/devices     -> [{device_token_prefix, device_name, created_at}]
```

`device_token_prefix` = first 8 hex chars of the (lowercased) token — enough
for the UI to disambiguate devices without exposing the full capability.

Settings (`settings.notifications.apns`):

```jsonc
{
  "mode": "relay" | "off",   // default "off"
  "relay_url": ""            // required when mode == "relay"; max 256 chars
}
```

`mode` defaults to **`off`**, not `relay`: there is no default public relay URL
to point at, and a mode that errors on every event out of the box is worse than
one that is plainly disabled. `direct` is gone (see the notice at the top).

**If you run the relay yourself, `relay_url` is `http://push-relay:8090`** — the
Docker-internal name, not your own public hostname. Your push then never depends
on your tunnel, your DNS, or your internet being up.

Send path:

- Hooks the **same** notification pipeline as web push (same event filters,
  per-camera toggles, and cooldown/dedup windows) — one decision, two
  transports.
- For each registered APNs device: encrypt per §2, POST to
  `{relay_url}/api/push` with `collapse_id = event_id`, the device's stored
  `environment`, and `priority` `"high"` for detections, `"normal"` for
  housekeeping.
- On HTTP **410** (or 400 `bad_device_token`): delete that registration row.

## 5. iOS deltas (implemented)

Shipped: `ios/Shared/PushCrypto.swift` (the §2 crypto),
`ios/NotificationService/` (the decrypt-and-attach extension), and
`ios/Sentinel/Sources/Core/CallManager.swift` (PushKit → CallKit, §3a). The
spec below is the contract they implement.

- On enabling notifications: request authorization, get the APNs token,
  generate the 32-byte key (`SymmetricKey(size: .bits256)`), persist in
  Keychain, `POST /api/notifications/apns/register` to the connected server.
  Hex-encode the token as
  `deviceToken.map { String(format: "%02x", $0) }.joined()` (lowercase).
  Re-register on token rotation (`didRegisterForRemoteNotifications…`) and
  on server reconnect.
- Notification Service Extension (`mutable-content: 1` triggers it):
  1. `guard let enc = userInfo["enc"] as? String`
  2. `AES.GCM.open(SealedBox(combined: Data(base64Encoded: enc)!), using: key)`
     (CryptoKit; combined layout matches §2's `nonce||ct||tag`).
  3. Parse JSON → set `title`/`body`; fetch `snapshot_url` (direct to the
     hoster server) with a short timeout (~4 s) and attach as
     `UNNotificationAttachment`.
  4. Any failure → call the content handler with the original request's
     content (generic "Vigilume / Encrypted notification" shows).
- Multiple servers: one key per server registration (key is per
  registration, not per device).

## 6. Relay deployment (ops)

Implemented in `docker-compose.yml` as `push-relay`, profile-gated (`relay`) —
only the app owner runs it, not regular hosters. `relay/README.md` is the
deploy reference (env table, Apple portal setup, publishing); the invariants:

- **No persistence.** In-memory rate-limit counters only; restarts are free.
  `read_only: true`, no writable volume, ever.
- **The key is a `:ro` DIRECTORY mount** (`./secrets:/keys:ro`), not
  `./secrets/apns.p8:/keys/apns.p8:ro`. On Unraid `/mnt/user` is shfs (FUSE)
  and a single-FILE bind mount fails in runc ("not a directory"). `APNS_KEY_P8`
  is a **path**, not inline PEM — inline puts the signing key in `docker
  inspect` and in the box `.env`. `deploy.sh` excludes `secrets/`, so the key
  exists only on the box; Apple lets you download a `.p8` exactly once.
- **The relay runs `USER nobody`**, so `chown nobody:users secrets/apns.p8 &&
  chmod 400`. A 0600 key owned by root is the failure you will actually hit:
  the mount succeeds and the relay dies on startup unable to read it.
- **Port 8090 is published on all interfaces**, not bound to loopback, so a
  tunnel or reverse proxy in a *different* Docker project (which has its own
  network and cannot resolve `push-relay`) can reach `http://<box-lan-ip>:8090`.
  This is why an earlier `relay_url=http://192.168.1.253:8090` could never work
  against the old `127.0.0.1:8090:8090` binding. See `relay/README.md` for why
  exposure is acceptable by design.
- **One instance serves both environments** — `environment` rides on each
  request (§2, §3). `APNS_ENV` is only the fallback for a request that omits it.
- **Rate limits.** 30 pushes/token/min sliding window, a global cap, 32
  concurrent APNs requests, 4 KB request cap, and a `BadDeviceToken` circuit
  breaker so one broken origin server can't burn the team's APNs reputation.
- **Logging:** status + a short token prefix only; never payloads, never full
  tokens. `httpx`/`httpcore` are muzzled to WARNING — httpx would otherwise log
  the full `/3/device/<token>` URL at INFO under a verbose config.
- **Verified by `relay/tests/relay_smoke.py`** (no network; Apple faked via
  `httpx.MockTransport`, the app driven via `ASGITransport`).

## 7. ntfy — push without Apple credentials (beside the relay)

`backend/app/notify/ntfy.py`, `settings.notifications.ntfy`, smoke-tested by
`backend/tests/ntfy_smoke.py`. A third channel beside web push and APNs, fired
from the same `events_pipeline._send_notification` fan-out, under the same
enabled/label/min_score/cooldown gates, and **spawned** (never awaited) so a
slow ntfy can't stall the doorbell watcher.

**What it is for.** On iOS every push traverses APNs, which needs an Apple
developer account and a signing key. ntfy.sh already runs that plumbing, so a
self-hoster needs no Apple account, no `.p8`, and nobody has to host a signing
key for strangers. Self-hosted ntfy wakes the phone via ntfy.sh but the app
fetches the message from *your* server, so content stays private.

**Why it is not a relay replacement.** It briefly was, for part of one day (see
the notice at the top). ntfy's notifications arrive in the **ntfy app**: no
CallKit doorbell ring, no native Vigilume UI, no snapshot in-line. That is the
whole trade. The two channels answer different questions — ntfy asks "how do I
get pushed at all without Apple?", the relay asks "how do I get the real app
experience from someone else's server?" — and both fire from the same
`_send_notification` fan-out under the same gates.

**Publish.** `POST {server}/{topic}`, body = the message text, headers:

| Header | Value |
| --- | --- |
| `Title` | notification title |
| `Priority` | `1`..`5` (ntfy scale; default 4) |
| `Tags` | the event tag |
| `Click` | tap target — the event's click URL |
| `Attach` | the event snapshot URL (see below) |
| `Authorization` | `Bearer tk_...` when `auth_token` is set |

**Snapshots are LINKED, never uploaded.** `Attach` carries the same
media-token snapshot URL web push and APNs already use; the **phone** fetches
it straight from the NVR, so the image never touches the ntfy server. This
needs `system.public_url` reachable from the phone. `attach_snapshot: false`
sends text only.

**Security — the topic is a password.** ntfy's own docs say so: on a
default-allow server (ntfy.sh included) **anyone who knows the topic receives
every message on it**, and the `Attach` URL carries a media token. So:

- `topic` **defaults to empty** and the UI must generate an unguessable one —
  never a memorable name like `vigilume`.
- Self-host with `auth-default-access: deny-all` and set `auth_token` for a
  real permission model.
- The publish URL *is* the secret, so nothing logs a full topic, a full URL, or
  a raw exception (an httpx error string embeds the request URL). `httpx` and
  `httpcore` are muzzled to WARNING for the same reason.
- Note the notification **text alone** leaks — camera names plus timing map
  when a house is empty — so a public topic is a poor idea even with
  `attach_snapshot: false`.

**Settings** (`notifications.ntfy`): `enabled` (default false), `server`
(default `https://ntfy.sh`), `topic` (`^[A-Za-z0-9_-]{1,64}$` — one path
segment, so it can't smuggle a path or query into the publish URL),
`auth_token`, `priority` (1..5), `attach_snapshot` (default true).

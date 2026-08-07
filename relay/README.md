# Vigilume APNs push relay

Forwards E2E-encrypted push to Apple on behalf of any Vigilume server. Pinned
contract: [`docs/push-architecture.md`](../docs/push-architecture.md) §1-§3a.

**What it holds:** the Apple `.p8` signing key. Nothing else. It never sees
notification plaintext, camera names (alert path), or snapshots, and writes
nothing to disk. **Never mount a writable volume.**

## Why this exists

Vigilume's iOS app belongs to one Apple team. Any self-hosted Vigilume server
must be able to push to it — with a real native notification and a CallKit
doorbell ring. ntfy is a fine channel but its notifications land in the *ntfy
app*: no CallKit ring, no native UI. The relay is the only way another
self-hoster gets the real app experience. ntfy stays, beside it.

## Configuration (env only)

| Var | Default | Notes |
|---|---|---|
| `APNS_KEY_P8` | *required* | **A path** (`/keys/apns.p8`) or an inline PEM. Prefer the path — inline puts the key in `docker inspect`. |
| `APNS_KEY_ID` | *required* | Apple Key ID. Safe to log (it rides in the JWT `kid` in cleartext). |
| `APNS_TEAM_ID` | *required* | |
| `APNS_BUNDLE_ID` | *required* | `apns-topic`; VoIP uses `<bundle>.voip`. |
| `APNS_ENV` | `production` | `sandbox` \| `production`. **Fallback only** — used when a request omits `environment`. |
| `RELAY_MAX_BODY` | `4096` | Request cap, bytes. |
| `RELAY_RATE_LIMIT` | `30` | Per device token, per window. |
| `RELAY_RATE_WINDOW` | `60` | Seconds. |
| `RELAY_MAX_CONCURRENCY` | `32` | Concurrent APNs requests. |
| `RELAY_APNS_TTL` | `1800` | `apns-expiration` offset (alert path only; VoIP is always `0`). |
| `RELAY_GLOBAL_RATE_LIMIT` | `600` | Across **all** tokens, per window. `0` disables. |
| `RELAY_BREAKER_THRESHOLD` | `50` | BadDeviceTokens in a window before the breaker opens. `0` disables. |
| `RELAY_BREAKER_COOLDOWN` | `60` | Seconds the breaker stays open. |

`APNS_ENV=sandbox` is what an Xcode dev build needs. TestFlight and App Store
builds are production. **A dev build's token sent to the production host returns
`BadDeviceToken` and the push vanishes** — the single most common failure here.
Check `GET /healthz` to see which environment this instance defaults to.

## API

See contract §3 / §3a. Summary:

- `POST /api/push` — `{device_token, payload_b64, priority?, collapse_id?, environment?}`
- `POST /api/push/voip` — `{device_token, payload, environment?}`
- `GET /healthz` → `{"ok": true, "env": "..."}`, `Cache-Control: no-store`

`environment` is optional on both and validated against a fixed host map;
absent falls back to `APNS_ENV`, unknown is `400 bad_environment`.

## Apple portal setup

1. Keys → **+** → enable **Apple Push Notification service (APNs)**.
2. Download the `.p8`. **Apple lets you download it exactly once, ever.** Keep
   Apple's filename, `AuthKey_<KEYID>.p8` — it is the only record of the key ID.
3. Note the **Key ID** and your **Team ID**.
4. If you create the key today you may be offered environment scoping. A
   **dual-environment (unscoped) key** is what this relay's single-JWT design
   expects. A scoped key returns `403 BadEnvironmentKeyIdInToken` for the other
   environment; the relay logs that distinctly.

## Deploy

Profile-gated — only the app owner runs it:

```sh
# .env
COMPOSE_PROFILES=relay
APNS_KEY_ID=ABCDE12345
APNS_TEAM_ID=FGHIJ67890
APNS_BUNDLE_ID=com.vigilume.app
APNS_ENV=sandbox          # Xcode build. TestFlight/App Store -> production.

# the key. deploy.sh excludes secrets/, so this lives only on the box.
cp AuthKey_ABCDE12345.p8 secrets/apns.p8
chown nobody:users secrets/apns.p8 && chmod 400 secrets/apns.p8   # runs USER nobody

docker compose up -d --build --remove-orphans
```

`chmod 400` **owned by root** is the failure you will actually hit: the mount
succeeds and the relay dies on startup unable to read the key. `chown` first.

### Publishing it

Other people's servers need a public HTTPS address for the relay. It is built to
face the internet (below), so put it behind whatever you already run — a
Cloudflare Tunnel, an nginx/Traefik reverse proxy, anything:

```
public hostname  ->  http://<box-lan-ip>:8090
```

Compose publishes 8090 on **all** interfaces for exactly this. A tunnel or proxy
in a *different* Docker project has its own network and cannot resolve
`push-relay`, so it needs the box's LAN IP. (Loopback binding is why an earlier
`relay_url=http://192.168.1.253:8090` could never work: nothing answered on the
LAN IP, and a containerised tunnel can't reach the host's loopback either.)

**Your own backend must not use the public URL.** Set Settings → Notifications →
APNs → Relay URL to `http://push-relay:8090` — the Docker-internal name. Your
push then never depends on the tunnel, on DNS, or on your internet being up.

### Why this is safe to expose

- **Bodies are ciphertext.** The alert path takes `payload_b64` the relay cannot
  read. It never sees camera names or snapshots.
- **Device tokens are unguessable** (64 hex, `fullmatch`-validated) and are the
  only addressing. There is no account, no login, and nothing to enumerate.
- **It stores nothing.** `read_only: true`, no volumes but the key (`:ro`), no
  database. Rate-limit counters are in memory; restarts are free.
- **It is rate-limited** per token and globally, with a `BadDeviceToken` breaker
  so a broken or hostile origin server cannot burn the team's APNs reputation.

What exposure does cost you: anyone who learns a device token can send that
device a push the app will show. That is inherent to the design — it is what lets
someone else's server reach the app at all.


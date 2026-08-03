# Vigilume iOS — Native App Design (research findings)

Status: **verified 2026-07-09** against the pinned go2rtc v1.9.14 source, the live
repo (`frontend/nginx.conf.template`, `backend/app/native/streams.py`,
`frontend/src/lib/api.ts`, `frontend/src/lib/talk.ts`, `docs/CONTRACTS.md`) and
current Apple documentation. This document gives the build agents exact
endpoints, payloads and project-generation shapes — no guessing.

App identity: name **Vigilume**, deployment target **iOS 17**, bundle id
placeholder **`com.sentinelnvr.app`** (user sets his own team/bundle in Xcode;
automatic signing, `DEVELOPMENT_TEAM` left unset). Project lives in
`/Users/adamrpowell94/Desktop/Security/ios/`.

---

## 1. Server reachability & auth model

The app talks to ONE user-configured absolute base URL (the nginx `web`
container), e.g. `http://192.168.1.50:8080` or `https://nvr.example.com:8443`
(Caddy TLS profile). Everything is under that origin:

| Path prefix | Backs onto | Auth |
|---|---|---|
| `/api/*` | FastAPI backend | `Authorization: Bearer <JWT>`; media routes + WS also accept `?token=` |
| `/go2rtc/*` | go2rtc 1.9.14 API (port 1984, internal) | **NONE — verified** (see below) |

**Login:** `POST {base}/api/auth/login` body `{"username": "...", "password": "..."}`
→ `{token, role, username}`. Roles: `admin` / `viewer`. Store the token in the
**Keychain** (not UserDefaults). Any 401 ⇒ token invalid ⇒ return to login.
`GET /api/auth/me` → `{username, role}` validates a stored session at launch.

**`/go2rtc/` proxy has no auth — verified fact, not an assumption.**
`frontend/nginx.conf.template` lines 83–97: the `location /go2rtc/` block does
`rewrite ^/go2rtc/(.*)$ /$1 break; proxy_pass $go2rtc_upstream;` with WebSocket
upgrade headers and **no** token/header check of any kind, and the backend is
not in that path. The web PWA already relies on this (its live player opens
`/go2rtc/api/ws?src=<camera>` with no token). The iOS app therefore reaches
go2rtc endpoints with **plain unauthenticated GETs** through the same proxy.
Security posture is unchanged from the existing web client: anyone who can
reach port 8080 can pull live streams. Document-honestly note: if that ever
tightens server-side, the app's stream URLs gain a `?token=` the same way the
web client's would.

---

## 2. Live video (v1 = HLS via AVPlayer)

### 2.1 The exact endpoint

go2rtc v1.9.14 registers (verified in `internal/hls/hls.go` at the v1.9.14
tag): `api/stream.m3u8` (master playlist), `api/hls/playlist.m3u8` (media
playlist), `api/hls/segment.ts`, `api/hls/init.mp4`, `api/hls/segment.m4s`.

Two flavors of `api/stream.m3u8`:

| URL | Segments | Codecs |
|---|---|---|
| `api/stream.m3u8?src=NAME` | MPEG-TS | H.264 only |
| `api/stream.m3u8?src=NAME&mp4` | **fMP4** | H.264, **H.265/HEVC**, AAC |

**Use the `&mp4` (fMP4) flavor.** Reasons, all from go2rtc's own docs/source:

- fMP4 HLS has **lower latency than HLS/TS** (go2rtc README states this
  explicitly).
- The Amcrest mains may be HEVC; Apple's HLS stack does **not** play HEVC in
  MPEG-TS segments, but AVPlayer plays HEVC in fMP4 natively (iOS 11+). fMP4
  works for both H.264 and H.265 cameras with zero config.
- Audio: the go2rtc `{name}` main stream carries an **AAC** track (the backend
  generates `streams: {name}: [<main_url>, "ffmpeg:{name}#audio=aac"]` —
  `backend/app/native/streams.py::build_config`), so plain `&mp4` gets audio.
  `&mp4=flac` (PCMA/PCMU passthrough) is **not needed** for main streams.

Full app-side URLs (absolute base + proxy prefix):

```
Camera screen / full-screen (full-res + AAC audio):
  {base}/go2rtc/api/stream.m3u8?src={camera}&mp4

Grid tiles (low-res, muted) — ALSO the automatic SD fallback:
  {base}/go2rtc/api/stream.m3u8?src={camera}_sub&mp4
```

### 2.1.1 HEVC over HLS/fMP4 — codec-filter verification (2026-07-09) + fallback

Re-verified against the pinned v1.9.14 source after a real-device report of
a black main-stream player (Amcrest mains are H.265):

- `internal/hls/hls.go handlerStream`: `medias := mp4.ParseQuery(query)` —
  any present `mp4` param (including the bare `&mp4`) selects the fMP4
  consumer; without it you get MPEG-TS (H.264 only).
- `pkg/mp4/helpers.go ParseQuery`: the media filter for **every** `mp4`
  value — `""` (bare, "legacy"), `flac`, or anything else — includes video
  codecs **H264 AND H265**; the value only widens the *audio* list
  (`&mp4=flac` adds PCMA/PCMU/PCM/PCML, other values add Opus/MP3). **There
  is no video-codec query knob and none is needed: the pinned URL
  `stream.m3u8?src={camera}&mp4` already requests HEVC-capable fMP4.** Do not
  add `&video=...` — a `video`/`audio` query without `mp4` also selects the
  fMP4 consumer (`core.ParseQuery`), but buys nothing over `&mp4`.
- `pkg/mp4/mime.go`: HEVC is advertised as `hvc1.1.6.L153.B0` — the
  `hvc1` tag Safari/AVPlayer accepts (`hev1` is the one Apple rejects), so
  the master playlist's `CODECS=` attribute is iOS-compatible too.

So go2rtc CAN serve HEVC HLS and the URL form is already right. The observed
device failure is therefore environmental, not a codec-filter bug. Known
sharp edge in the same source: `internal/hls/session.go Session.Init()`
waits only **~3 s** (60×50 ms) for the init segment + first fragment; the
main stream's on-demand `ffmpeg:{name}#audio=aac` producer can take longer
than that to spin up cold, yielding an empty `init.mp4` → AVPlayer fails →
black box. Level/profile quirks of a given camera's HEVC encode are also
possible.

**App-side answer (implemented in `LivePlayerModel`): HD-first with
automatic SD fallback.** The big player attaches the main stream; if the
item fails or no frame renders within **~5 s**, it switches to the
`{camera}_sub` H.264 substream, shows an "SD (compat)" badge with an "HD"
retry button, and keeps the usual watchdog/backoff self-heal on whichever
source is active. If the substream *also* fails repeatedly, the UI shows
the real AVFoundation error text (never a silent black box). A stall on a
source that already rendered retries in place instead of downgrading.

`{camera}_sub` publishes only the raw substream (no `#audio=aac` source), so
its G.711 audio is filtered out by the fMP4 codec filter → **video-only** —
exactly right for muted grid tiles, and it rides the single shared RTSP
session Vigilume already opens per substream.

### 2.2 Proxy-path correctness (relative URIs — verified)

The master playlist references the media playlist and segments with
**relative** URIs plus an `id=` session query (`hls/playlist.m3u8?id=...`,
`init.mp4?id=...`, `segment.m4s?id=...`). AVPlayer resolves them against the
playlist URL, so fetched as `{base}/go2rtc/api/stream.m3u8?...` they resolve to
`{base}/go2rtc/api/hls/...`, and nginx's `rewrite ^/go2rtc/(.*)$ /$1` (query
string preserved) maps them straight onto go2rtc's `api/hls/...`. **No URL
rewriting needed in the app.**

go2rtc HLS sessions expire after **5 s without a segment request**; AVPlayer's
continuous polling keeps the session alive. A paused/backgrounded player will
lose the session — treat resume as "create a fresh `AVPlayerItem` from the
stream URL", never try to reuse a stale one.

### 2.3 AVPlayer setup + latency expectations (honest numbers)

```swift
let item = AVPlayerItem(url: hlsURL)
item.preferredForwardBufferDuration = 1        // keep the buffer small
let player = AVPlayer(playerItem: item)
player.automaticallyWaitsToMinimizeStalling = false
player.play()
// Optional catch-up when drift accumulates: seek to livePosition, or
// player.rate = 1.05 briefly. Mute grid tiles: player.isMuted = true.
```

- **HLS/fMP4 via AVPlayer: expect ~3–6 s glass-to-glass**, occasionally more.
  go2rtc cuts short segments, but AVPlayer holds a multi-segment buffer and
  Apple's own docs call regular HLS a seconds-class protocol. Do **not**
  promise sub-second live view in v1; label the UI "LIVE" without a latency
  claim.
- go2rtc's README ranks latency: **WebRTC (best, sub-second) > MSE/MP4
  (~1 s) > HLS (worst)**. MSE is a browser API — not applicable natively.

**Future upgrade — WHEP/WebRTC (documented, not built in v1):** go2rtc 1.9.x
exposes a WHEP-compatible SDP exchange at `POST {base}/go2rtc/api/webrtc?src={camera}`
(also the WebSocket signaling at `/go2rtc/api/ws?src=`). A native client needs
a WebRTC stack (e.g. `stasel/WebRTC` SPM binary of libwebrtc, or LiveKit's
`webrtc-xcframework`) plus TURN-less ICE against the operator-configured
`ip:8555` candidates (`settings.system.webrtc_candidates`). That drops live
latency to <1 s. v1 ships HLS only; the player view should be built behind a
small `LivePlayerBackend` protocol so WHEP can slot in later.

### 2.4 Recorded timeline + event clips (AVPlayer, direct)

- Timeline: `GET {base}/api/recordings/{camera}/playlist.m3u8?start={epoch}&end={epoch}&token={jwt}`
  is a fully valid **HLS VOD** playlist (`#EXT-X-PLAYLIST-TYPE:VOD`, relative
  `seg/{ts}.ts` URIs that inherit `?token=` server-side). AVPlayer plays it
  directly — segments are MPEG-TS **H.264** (the backend transcodes HEVC
  sources for exactly this reason), which Apple HLS accepts in TS. Window is
  capped server-side at 6 h; request ~1 h around the playhead like the web
  client does.
- Segment index for the scrubber: `GET /api/recordings/{camera}/index?date=YYYY-MM-DD`
  (Bearer) → `{segments, ranges}`; availability: `GET /api/recordings/cameras`.
- Event clips: `GET /api/events/{id}/clip.mp4?token=` — H.264 faststart MP4
  with Range support; AVPlayer-direct. Snapshot:
  `GET /api/events/{id}/snapshot.jpg?token=`. Respect `clip_state` from
  `GET /api/events/{id}` (`ready | processing | recording_disabled | unavailable`)
  for the "clip still being prepared" UX.
- Because AVPlayer/URLSession subsystems fetch these without app-controlled
  headers in some paths, **always use the `?token=` form for media URLs**
  (mirrors `frontend/src/lib/api.ts::mediaUrl`).

---

## 3. APNs (token-based p8 / ES256)

### 3.1 Server-side contract (for the future backend APNs sender)

The backend today only does Web Push; APNs delivery is a backend work item.
This is the contract both sides build to (current Apple spec, verified):

- **Connection:** HTTP/2 (ALPN `h2`) to `https://api.push.apple.com:443`
  (sandbox: `https://api.sandbox.push.apple.com:443`), request
  `POST /3/device/{device-token-hex}`.
- **Auth:** JWT signed **ES256** with the `.p8` AuthKey. Header
  `{"alg":"ES256","kid":"<10-char Key ID>"}`, claims
  `{"iss":"<10-char Team ID>","iat":<unix-seconds>}`. APNs rejects tokens
  older than **1 hour** (`403 ExpiredProviderToken`) and throttles re-issuing
  more than every ~20 min — **cache the JWT and refresh every 30–50 min**, do
  not sign per-push. One key works for all the team's apps; no cert renewal
  treadmill.
- **Headers per push:**
  - `authorization: bearer <jwt>`
  - `apns-topic: com.sentinelnvr.app` (the APP bundle id — never the
    extension's)
  - `apns-push-type: alert` (required)
  - `apns-priority: 10` (immediate; use `5` only for silent/content-available)
  - `apns-expiration: <unix>` — set `now + 3600`; a stale motion alert is noise
  - optional `apns-collapse-id: <event-id>` to coalesce update/end re-sends
- **Payload** (≤ 4 KB):

```json
{
  "aps": {
    "alert": { "title": "Person detected at Front Yard", "body": "2 in frame" },
    "sound": "default",
    "thread-id": "front_yard",
    "mutable-content": 1
  },
  "event_id": "123",
  "snapshot_url": "https://nvr.example.com/api/events/123/snapshot.jpg?token=<media-scope-jwt>",
  "deep_link": "sentinel://events/123"
}
```

  `mutable-content: 1` is what routes the push through the Notification
  Service Extension. `snapshot_url` must be an **absolute** URL (built from
  `settings.system.public_url`) carrying a **media-scope token** (the media
  routes accept `?token=`; media-scope tokens are rejected on non-media routes
  — safe to embed). `event_id`/`deep_link` drive tap → Event Detail.
- **Device registration:** the app needs a backend route to store its APNs
  token — proposed shape, mirroring the Web Push pair:
  `POST /api/notifications/apns/register {device_token, environment: "prod"|"sandbox"}`
  and `POST /api/notifications/apns/unregister {device_token}` (Bearer, any
  authenticated role — viewers may toggle push on their own device per RBAC).
  Until that route exists the app builds and gates the UI on a 404 ("server
  does not support native push yet").

### 3.2 iOS-side pieces

App target (SwiftUI lifecycle → needs an `UIApplicationDelegateAdaptor`):

```swift
// 1. Ask permission (Settings screen or first-run):
let ok = try await UNUserNotificationCenter.current()
    .requestAuthorization(options: [.alert, .sound, .badge])
// 2. Then, on the main actor:
UIApplication.shared.registerForRemoteNotifications()
// 3. AppDelegate:
func application(_ app: UIApplication,
                 didRegisterForRemoteNotificationsWithDeviceToken token: Data) {
    let hex = token.map { String(format: "%02x", $0) }.joined()
    // POST hex to /api/notifications/apns/register
}
// Also implement didFailToRegisterForRemoteNotificationsWithError (log + surface).
```

- Re-run `registerForRemoteNotifications()` on every launch — the token can
  rotate; re-POST when it changes.
- `UNUserNotificationCenterDelegate.didReceive response:` reads
  `userInfo["deep_link"] / ["event_id"]` and routes to Event Detail.
  `willPresent` returns `[.banner, .sound]` so foreground pushes still show.
- Entitlement: `aps-environment` = `development` in the entitlements file —
  Xcode automatic signing flips it to `production` for distribution builds.
  The user's paid developer account has full push entitlement.
- `UIBackgroundModes: [remote-notification]` in Info.plist (enables
  `content-available` wake-ups later; harmless for alert pushes).

### 3.3 Notification Service Extension (snapshot attachment)

Second target `NotificationService` (type `app-extension`,
`NSExtensionPointIdentifier: com.apple.usernotifications.service`). It runs on
`mutable-content: 1` pushes, has ~30 s, and attaches the snapshot:

```swift
final class NotificationService: UNNotificationServiceExtension {
    private var handler: ((UNNotificationContent) -> Void)?
    private var content: UNMutableNotificationContent?

    override func didReceive(_ request: UNNotificationRequest,
                             withContentHandler h: @escaping (UNNotificationContent) -> Void) {
        handler = h
        content = (request.content.mutableCopy() as? UNMutableNotificationContent)
        guard let content,
              let urlString = request.content.userInfo["snapshot_url"] as? String,
              let url = URL(string: urlString) else { h(request.content); return }
        URLSession.shared.downloadTask(with: url) { tmp, _, _ in
            defer { h(content) }
            guard let tmp else { return }
            let dst = tmp.deletingLastPathComponent()
                .appendingPathComponent(UUID().uuidString + ".jpg") // extension required
            try? FileManager.default.moveItem(at: tmp, to: dst)
            if let a = try? UNNotificationAttachment(identifier: "snapshot", url: dst) {
                content.attachments = [a]
            }
        }.resume()
    }

    override func serviceExtensionTimeWillExpire() {
        if let content { handler?(content) } // deliver un-attached rather than default
    }
}
```

**Target requirements (confirmed):**
- Its own bundle id **prefixed by the app's**: `com.sentinelnvr.app.NotificationService`.
- Same automatic signing; **no entitlements needed** on the extension
  (`aps-environment` belongs to the app target only).
- **App Groups are NOT needed** — the attachment is handed to the system from
  the extension's own container; nothing is shared with the app. (A shared
  Keychain group is also unnecessary because the snapshot URL carries its own
  `?token=`.)
- Extension deployment target ≤ app's (use iOS 17 for both).
- ATS applies inside the extension too — the ATS keys in §5 go in **both**
  Info.plists so an http-LAN snapshot URL still downloads.

---

## 4. Two-way talk (push-to-talk uplink)

Contract (CONTRACTS.md + `frontend/src/lib/talk.ts`, the reference
implementation): `WS {base}/api/cameras/{name}/talk?token=<session-jwt>`
(media-scope tokens rejected; capability-gated on `capabilities.speaker` —
only the AD410). Client sends **binary frames of raw little-endian Int16 PCM,
mono, 8 kHz**. Text frames ignored except `{"type":"stop"}`. Server closes:
`4003` no speaker, `4009` busy (someone else talking), `4502` camera rejected
audio/unreachable, `1000` clean stop or the **120 s** session cap.

### Native pipeline (feasible, all first-party APIs)

```
AVAudioSession (.playAndRecord, .voiceChat)          — routes + echo profile
  → AVAudioEngine.inputNode (setVoiceProcessingEnabled(true))  — AEC/NS like the web's echoCancellation:true
  → installTap(onBus: 0, bufferSize: 4096, format: nil)        — hardware rate (typ. 48 kHz) Float32
  → AVAudioConverter(from: hwFormat,
        to: AVAudioFormat(commonFormat: .pcmFormatInt16,
                          sampleRate: 8000, channels: 1, interleaved: true))
  → accumulate into 320-sample frames (40 ms = 640 bytes)
  → URLSessionWebSocketTask.send(.data(frame))
```

Notes for the implementer:
- The web client uses **40 ms frames (320 samples / 640 bytes)**; match it
  (spec tolerance 20–60 ms). Keep a small carry buffer so converter output
  chunks (which won't align to 320) roll into the next frame; flush the
  partial frame then send `{"type":"stop"}` on release — exactly what
  `talk.ts` does.
- Tap `bufferSize: 4096` at 48 kHz ≈ 85 ms per callback → ~2 WS frames per
  tap; total uplink ~128 kbit/s raw — trivial. End-to-end added buffering
  ≈ 40–125 ms; fine for PTT.
- `AVAudioConverter` handles the 48 000→8 000 non-integer ratio with proper
  filtering (better than the web client's averaging decimator).
- Mic permission: `NSMicrophoneUsageDescription` (Info.plist, §5) + first-use
  prompt. Request lazily on first talk press, mirroring the web UX.
- One `TalkSession` per press; `stop()` idempotent: flush partial frame → send
  stop → `cancel(with: .normalClosure)` → remove tap → deactivate the audio
  session. Surface 4003/4009/4502 with the same fault copy the web uses.
- `setVoiceProcessingEnabled(true)` must be called before the engine starts
  and changes the input format — always read `inputNode.inputFormat(forBus: 0)`
  after enabling it, never hardcode 48 kHz.

---

## 5. Project generation (xcodegen)

**xcodegen 2.45.4 is installed** at `/opt/homebrew/bin/xcodegen` (verified);
Xcode 26.6 / iOS 26.5 SDK present. No brew install needed. Layout:

```
ios/
  project.yml
  Vigilume/                  # app sources (SwiftUI)
    Vigilume.entitlements
  NotificationService/       # extension sources
    NotificationService.swift
```

`ios/project.yml` — the verified shape:

```yaml
name: SentinelNVR
options:
  bundleIdPrefix: com.sentinelnvr
  deploymentTarget:
    iOS: "17.0"
  createIntermediateGroups: true
settings:
  base:
    SWIFT_VERSION: "5.9"
    CODE_SIGN_STYLE: Automatic
    # DEVELOPMENT_TEAM deliberately unset — the user selects his team in Xcode.
targets:
  SentinelNVR:
    type: application
    platform: iOS
    sources: [Vigilume]
    dependencies:
      - target: NotificationService     # app-extension deps embed into PlugIns automatically
    entitlements:
      path: Vigilume/Vigilume.entitlements
      properties:
        aps-environment: development     # auto-signing flips to production on distribution
    info:
      path: Vigilume/Info.plist
      properties:
        CFBundleDisplayName: Vigilume
        UILaunchScreen: {}
        UIBackgroundModes: [remote-notification]
        NSMicrophoneUsageDescription: >-
          Vigilume uses the microphone for two-way talk through your
          doorbell/camera speaker.
        NSAppTransportSecurity:
          NSAllowsArbitraryLoads: true      # see ATS decision below
          NSAllowsLocalNetworking: true
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.sentinelnvr.app
        MARKETING_VERSION: "1.0"
        CURRENT_PROJECT_VERSION: "1"
        TARGETED_DEVICE_FAMILY: "1,2"
  NotificationService:
    type: app-extension
    platform: iOS
    sources: [NotificationService]
    info:
      path: NotificationService/Info.plist
      properties:
        CFBundleDisplayName: SentinelNotificationService
        NSExtension:
          NSExtensionPointIdentifier: com.apple.usernotifications.service
          NSExtensionPrincipalClass: $(PRODUCT_MODULE_NAME).NotificationService
        NSAppTransportSecurity:
          NSAllowsArbitraryLoads: true
          NSAllowsLocalNetworking: true
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.sentinelnvr.app.NotificationService
        MARKETING_VERSION: "1.0"
        CURRENT_PROJECT_VERSION: "1"
schemes:
  SentinelNVR:
    build:
      targets: { SentinelNVR: all }
    run:
      config: Debug
```

Notes:
- xcodegen auto-generates both Info.plists from the `info:` blocks — do not
  hand-maintain plist files.
- The scheme name must stay `SentinelNVR` (the BUILD GATE references it).
- No app groups, no push SPM packages, no third-party deps in v1.

**BUILD GATE (every phase):**

```bash
cd /Users/adamrpowell94/Desktop/Security/ios \
  && xcodegen generate \
  && xcodebuild -project SentinelNVR.xcodeproj -scheme SentinelNVR \
       -sdk iphonesimulator -destination "generic/platform=iOS Simulator" \
       build CODE_SIGNING_ALLOWED=NO
```

### The ATS decision (honest)

Facts (Apple ATS behavior, verified against Apple docs/forums):
- ATS is enforced only for **public qualified hostnames**. Connections to
  **raw IP addresses** (`http://192.168.1.50:8080`), unqualified names and
  `.local` are **exempt from ATS by default** — most LAN self-hosters need no
  exception at all.
- `NSAllowsLocalNetworking: true` formalizes the local-hosts/.local carve-out
  without disabling ATS globally.
- The gap: self-hosters who reach the NVR through a plain-http **DNS
  hostname** — `http://nvr.lan:8080`... is unqualified (fine), but
  `http://myhost.tailnet-name.ts.net:8080` or an internal FQDN is a qualified
  hostname and **ATS blocks it** unless arbitrary loads are allowed (per-domain
  exceptions are impossible: the domain is user-entered at runtime).

**Decision: ship with `NSAllowsArbitraryLoads: true` + `NSAllowsLocalNetworking: true`,**
and have the app UI nudge toward https (show a "not encrypted" badge for http
base URLs). Rationale: the server URL is user-configured and frequently plain
http on trusted LANs/VPNs; this is the same trust model as every self-hosted
NVR client (Home Assistant's app does the same). Trade-off stated plainly:
ATS protections are off for ALL connections, so a user who enters an http URL
over hostile Wi-Fi sends his JWT in cleartext — exactly as the PWA over http
does today. APNs itself is unaffected (Apple's push transport, not ours). App
Review: sideload/TestFlight-personal distribution doesn't gate on it; if the
app ever goes to the App Store, the justification is "user-configured
connection to a private self-hosted server" (accepted category).

---

## 6. Everything else the build agents need (quick contract recap)

- **WS live updates:** `wss?://{base}/api/ws?token=` — JSON frames
  `event_new | event_update | event_end | doorbell` (`{type, event:{...}}`),
  `camera_status`, `model_status`. Reconnect with backoff; refetch lists on
  reconnect.
- **Cameras:** `GET /api/cameras` → capabilities gate UI (talk button on
  `speaker`, siren on `siren`, spotlight on `white_light`). Admin-only
  controls: settings/light/siren/reboot/probe/talk (talk is admin-only per the
  matrix), plus all `/api/settings`, `/api/users`, `/api/detection/*` —
  viewers get 403; hide those affordances by `role`.
- **Events list:** `GET /api/events?camera=&label=&after=&before=&limit=&offset=` →
  `{events, total}`; snapshots/clips via `?token=` URLs (§2.4).
- **Groups:** full CRUD allowed for both roles (shared across users).
- **Snapshot polling for tiles (fallback/low-power):**
  `GET /api/cameras/{name}/snapshot.jpg` (Bearer or `?token=`).
- Reference client for shapes and edge-cases: `frontend/src/lib/api.ts`
  (read-only; mirrors CONTRACTS.md exactly).

## 7. Open items handed to later phases

1. Backend APNs sender + `POST /api/notifications/apns/register` route
   (§3.1) — backend is owned by other workflows right now; the app gates its
   push UI on the route's existence.
2. WHEP/WebRTC live backend behind `LivePlayerBackend` (§2.3) — needs a
   WebRTC lib; drops latency to <1 s.
3. If `/go2rtc/` proxy auth is ever added server-side, thread `?token=` into
   the stream URLs (§1).

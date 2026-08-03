# Vigilume iOS app — build, install & self-hosting guide

The native SwiftUI companion app lives in `ios/` (`SentinelNVR.xcodeproj`,
targets `SentinelNVR` app + `NotificationService` extension). Design decisions
are recorded in [ios-design.md](ios-design.md); the backend API it speaks is
[CONTRACTS.md](CONTRACTS.md).

## Requirements

- macOS with **Xcode 26** (iOS 17+ SDK). Deployment target is **iOS 17.0**.
- An **Apple Developer account**. A paid account is what this project assumes
  (full push entitlement; no 7-day reinstall limit — that restriction only
  applies to free "personal team" signing).
- Your Vigilume server reachable from the phone (LAN, VPN/Tailscale, or a
  public HTTPS URL via the `tls` profile / your own reverse proxy).

## Build & install on your iPhone

1. `open ios/SentinelNVR.xcodeproj` in Xcode.
2. Select the **SentinelNVR** target → *Signing & Capabilities*:
   - **Automatically manage signing** is already on.
   - Pick your **Team**. Repeat for the **NotificationService** target
     (same team).
   - The bundle id ships as the placeholder `com.sentinelnvr.app`
     (extension: `com.sentinelnvr.app.NotificationService`). With your own
     team you will normally change these — see **Self-hosters** below.
3. Plug in your iPhone (or pick it under *Devices*), select it as the run
   destination, and **Run** (⌘R). Xcode installs the app; the first launch on
   a new device asks you to trust the developer certificate under
   *Settings → General → VPN & Device Management* (paid-account builds signed
   with a development profile).
4. Because this is a paid developer account, the installed build does **not**
   expire after 7 days (that limit is only for free accounts). Development
   builds still expire with the provisioning profile (~1 year) — just Run
   again from Xcode, or distribute to yourself via **TestFlight** for
   long-lived installs.

Sanity build from the command line (no signing, simulator):

```sh
cd ios
xcodegen generate     # if you edited project.yml (brew install xcodegen)
xcodebuild -project SentinelNVR.xcodeproj -scheme SentinelNVR \
  -sdk iphonesimulator -destination "generic/platform=iOS Simulator" \
  build CODE_SIGNING_ALLOWED=NO
```

## App UI at a glance

- **Cameras** — a single-column list of full-width 16:9 live tiles (name +
  online status overlaid), with group filter chips. **Tap a tile to open the
  camera screen**: a big live player on top (full-res main stream, tap to
  unmute, expand-to-fullscreen button) with **every control visible below
  it** — IR mode, spotlight (with brightness), hold-to-confirm siren,
  hold-to-talk, reboot, the recent-events strip, and (admin) status &
  credentials. Nothing hides behind a long-press; the tile's long-press menu
  is only an optional shortcut (e.g. jump straight to full-screen live). If
  the HD main stream won't start within ~5 s the player automatically drops
  to the H.264 substream with an "SD (compat)" badge and an "HD" retry
  button; if both fail you get the real error text, never a black box. Only
  on-screen tiles hold a stream.
- **Events** — two dropdown filters (Camera, Object) above the paged list.
  Opening an event **autoplays** the clip. If the clip is still being cut,
  the event screen shows that camera's **live view** with a "Clip
  processing…" badge and polls; when the clip lands, a "Clip ready — tap to
  watch" button swaps to the clip player. A dedicated **Save** card (also
  mirrored in the share toolbar menu) offers per-media actions: **Save to
  Photos** downloads the clip/snapshot with a progress bar and adds it to
  your photo library (iOS asks once for "Add Photos Only" access), and
  **Save to Files / Share** hands the server's friendly-filename download
  URL to the share sheet for Files, AirDrop, Messages, etc.
- **Timeline** — one scrub bar per day with green event markers (tap a marker
  to seek). When the playhead sits inside an event, the detection is named
  under the bar next to the time readout (e.g. "Person · Backyard"). The
  compact **Range** button (top row, next to Day/1h) toggles drag-to-select
  export of up to 30 minutes.
- The home-screen **app icon** (`ios/Vigilume/Assets.xcassets/AppIcon.appiconset`)
  is generated from the same shield/lens geometry as the web PWA icons
  (`frontend/scripts/gen-icons.mjs`).

## First sign-in

Enter the server URL exactly as you'd open the web UI, e.g.
`http://192.168.1.50:8080` or `https://nvr.example.com:8443`, plus your
username/password (admins get device controls & talk; viewers get
live/events/timeline and per-device push, per the RBAC contract). You can save
multiple servers under *Settings → Servers* and switch between them; each
server keeps its own login token (in the iOS Keychain) and push preference.

## Plain-HTTP LAN servers (ATS note)

iOS blocks cleartext HTTP by default (App Transport Security). The app ships
with `NSAllowsArbitraryLoads` + `NSAllowsLocalNetworking` enabled so that a
plain-`http://` LAN or Tailscale server URL works out of the box — the UI
shows an "unencrypted" badge for such servers, and your credentials/JWT travel
in cleartext on that network. If you only ever use HTTPS, you can tighten this
by removing the `NSAppTransportSecurity` override in `ios/project.yml` (both
targets) and regenerating the project. If you ever submit a build to App
Review, expect to justify the arbitrary-loads exception (self-hosted LAN NVR
is a standard justification).

## Push notifications (APNs)

Setup is one toggle: *Settings → Push notifications* → on → allow the
permission prompt. Status shows **"Active on this device"** when registered.
Until your server has the APNs feature
(`POST /api/notifications/apns/register`), the toggle reports **"Server not
updated yet"** and re-registers automatically once the route appears.
Web-push (PWA) notifications are unaffected.

Under the hood, native push is **end-to-end encrypted**
([push-architecture.md](push-architecture.md)) — nothing user-visible
changes, but it's worth knowing what travels where:

- On registration the app generates a random **32-byte key per server**,
  stores it in the iOS Keychain (in a keychain group shared with the
  notification extension), and sends it to *your* server:
  `{device_token, device_name, key_b64, environment}`. The key never leaves
  the app + your server.
- Your server encrypts each notification
  (`{title, body, event_id, snapshot_url}`) with **AES-256-GCM** under that
  key and hands only the ciphertext to the **push relay** (run by the app
  owner, which holds the Apple APNs credentials). The relay — and Apple —
  see ciphertext, never your camera events.
- On the phone, the Notification Service extension decrypts the alert,
  fills in the real title/body, and fetches the snapshot image **directly
  from your server** (tokened URL, never via the relay) to attach it. If
  decryption isn't possible (e.g. key rotated), the generic
  "Vigilume / Encrypted notification" alert is shown instead.

Debug (Xcode) installs register against the APNs **sandbox** gateway;
TestFlight/App Store builds use **production** — the app sends the matching
`environment` automatically, and automatic signing flips the
`aps-environment` entitlement for distribution builds.

Tapping an event push deep-links straight to that event's detail screen.
With multiple saved servers, the app knows which server's key decrypted the
push and switches to that server before opening the event.

## Self-hosters

- **Set your own identity**: change `PRODUCT_BUNDLE_IDENTIFIER` for both
  targets (in Xcode's Signing editor, or in `ios/project.yml` +
  `xcodegen generate`) to something under a domain you control, e.g.
  `com.yourname.sentinel` / `com.yourname.sentinel.NotificationService`, and
  select your own Team. `DEVELOPMENT_TEAM` is deliberately unset in the
  project.
- **Push with your own bundle id needs your own APNs setup**: a relay only
  pushes to the bundle id it was configured with (`apns-topic` is pinned
  relay-side and must equal *your* app's). If you rebuild under your own
  identifier, create an APNs `.p8` in your Apple developer account and run
  your own relay (`relay/`, see [push-architecture.md](push-architecture.md)
  §6) with `APNS_BUNDLE_ID` set to it, then point
  `settings.notifications.apns.relay_url` at it. (`mode: "direct"` — the
  server holding its own `.p8` — is retired; the relay is the only APNs
  transport. `notifications.ntfy` needs no Apple account at all, at the cost
  of the doorbell ring.) The E2E encryption layer is unchanged either way.
- The app has no hard-coded server: any Vigilume backend that implements
  [CONTRACTS.md](CONTRACTS.md) works, and multiple servers can be saved.

## Troubleshooting

- **"Could not attach to the server" / spinner on tiles** — confirm the same
  URL opens the web UI in Safari on the phone; live view uses
  `{base}/go2rtc/api/stream.m3u8`, which must be proxied by the same nginx.
- **Live tiles reconnect forever** — go2rtc down or the camera substream is
  offline; check *Settings → About → go2rtc* and the server logs.
- **Push toggle says "Server not updated yet"** — expected until the backend
  APNs route ships; the device re-registers automatically afterwards.
- **No event clip yet** — clips are cut ~20 s after an event ends; while
  *processing*, the event screen shows the camera's live view, polls, and
  offers "Clip ready — tap to watch" when done. *Recording disabled /
  unavailable* states are explained in place.

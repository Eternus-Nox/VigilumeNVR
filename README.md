# Vigilume NVR

A lightweight, **fully standalone**, self-hosted security camera NVR for Amcrest
cameras: its own GPU object detection ([D-FINE](https://github.com/Peterande/D-FINE)
ONNX on an NVIDIA card), its own 24/7 recording, its own low-latency live view
(WebRTC/MSE via go2rtc), smart push notifications with annotated snapshots, and a
mobile-installable PWA. No cloud, no subscriptions, no port forwarding — and no
external NVR software: the whole stack is four small containers you run yourself.

The binding architecture spec lives in [docs/CONTRACTS.md](docs/CONTRACTS.md); the
engine design record (model pins, decode math, ffmpeg patterns) in
[docs/native-mode-design.md](docs/native-mode-design.md).

## Features

- **24/7 recording** — ffmpeg stream-copy (no transcode) into 10-second segments, with
  separate retention for continuous footage, event clips, and snapshots; every detected
  event auto-cuts a playable clip ([recordings & clips](docs/recordings.md))
- **Timeline** — scrub each camera's continuous footage, jump to any moment, with your
  detected events as markers; streamed as HLS VOD on demand
  ([timeline](docs/recordings.md#timeline))
- **GPU object detection** (person, dog, cat, car by default) — D-FINE ONNX on the
  NVIDIA GPU via onnxruntime, running against low-resolution camera substreams; model
  downloaded and SHA-256-verified automatically on first boot
- **Smart notifications** — Web Push to your phone with a
  [Supervision](https://github.com/roboflow/supervision)-annotated snapshot (bounding
  boxes, labels, and an in-frame count like "2 people"); tap opens the event clip
- **Doorbell events** — AD410 button presses trigger an instant push with snapshot
- **Live view** — WebRTC (sub-second) with automatic MSE fallback, via a
  Vigilume-managed go2rtc restream (one camera connection shared by detection,
  recording, and live view). WebRTC candidates are derived automatically so LAN
  live view is fast with no hand-configuration ([docs/live-latency.md](docs/live-latency.md))
- **Camera control** — IR/night-vision mode, spotlight, siren (AD410), two-way talk
  (AD410), reboot, OSD, and more over the Amcrest HTTP API, capability-gated per model
- **PWA** — installable on iOS (16.4+) and Android, dark theme, works at phone widths
- **Zero config files** — cameras and settings are managed entirely in the web UI; the
  backend generates the go2rtc config and runs the detection/recording engines itself

## Architecture

```
   IP5M-T1277EW-AI ──┐
   IP8M-2779EW-AI ───┤  RTSP (main + sub streams, one session per stream)
   AD410 doorbell ───┤
                     │        ┌────── doorbell button (HTTP event stream) ──────┐
                     ▼        │                                                 │
        ┌──── go2rtc (restream fan-out) ────┐                                   │
        │   {cam} main + {cam}_sub streams  │                                   │
        └──┬──────────────┬─────────────────┘                                   │
           │ live view    │ RTSP restream                                       │
           │ (WebRTC/MSE) ▼                                                     ▼
           │  ┌────── backend (FastAPI) ──────────────────────────────────────────┐
           │  │  detector (D-FINE ONNX, NVIDIA GPU)   recorder (ffmpeg copy,      │
           │  │    └─► ByteTrack ─► events ─┐           10 s TS segments + clips) │
           │  │                             ▼                                     │
           │  │  event DB • Supervision-annotated snapshots • Web Push            │
           │  │  Amcrest device control • go2rtc config generator • auth          │
           │  └───────────────┬───────────────────────────────────────────────────┘
           │                  │ /api, /api/ws
           ▼                  ▼
   ┌──────────────── web (nginx, port 8080) ────────────────┐
   │  React PWA  ◄──  /  +  /api  +  /go2rtc (one origin)   │
   └───────────────────────────┬─────────────────────────────┘
                               │  optional: caddy, port 8443 (HTTPS —
                               │  required for PWA install + push on phones)
                               ▼
                      Phone (installed PWA)
```

Everything is served from a single origin (port 8080, or 8443 with TLS), so one proxied
port is all you need for live view, API, and push.

## Hardware requirements

| Component | Requirement |
|-----------|-------------|
| Host      | Linux box running Docker + Docker Compose v2 |
| GPU       | NVIDIA GPU (GTX 16-series / RTX or newer recommended), NVIDIA driver + [NVIDIA Container Toolkit](docs/setup-nvidia.md). The default D-FINE-S model needs only a small slice of a modern card |
| RAM       | 4 GB+ (backend + model + go2rtc are modest) |
| Storage   | Sized for retention — roughly 130 GB/day for the default 3-camera setup, ~1 TB per week ([storage math](docs/faq.md#how-much-storage-do-i-need)) |
| Network   | PoE switch/injectors for the two turret cameras; solid 2.4/5 GHz Wi-Fi coverage at the front door for the AD410 |
| Phone     | iOS 16.4+ or Android with Chrome, for the PWA |

The macOS checkout is for development only; the stack deploys on the Linux host.
(A CPU-only backend build target exists for dev/tests — see `backend/Dockerfile` — but
is not a supported production deployment.)

## Quick start

Prerequisites on the NVR host: NVIDIA driver + NVIDIA Container Toolkit installed and
verified — [docs/setup-nvidia.md](docs/setup-nvidia.md) steps 1–2.

```bash
# 1. Get the code onto the NVR host
git clone <your-repo-url> vigilume-nvr && cd vigilume-nvr

# 2. Configure
cp .env.example .env
#    Edit .env: set ADMIN_PASSWORD (required), TZ, MEDIA_PATH (big disk).
#    Optionally fill the CAM1..3_* blocks to pre-seed your cameras —
#    or skip them and add cameras in the UI afterwards.

# 3. Build and start
docker compose up -d --build

# 4. Open the UI and add cameras
#    http://<nvr-host>:8080  — log in with ADMIN_PASSWORD, then
#    Settings → Cameras → Add camera (name, model, IP, credentials).
#    Camera prep (static IPs, RTSP users): docs/cameras-amcrest.md
```

First boot: the backend downloads the detection model (~40 MB, SHA-256 verified), runs
the GPU self-test, writes the go2rtc config, and starts recording as soon as cameras
exist. Verify with the first-boot checklist in
[docs/setup-nvidia.md](docs/setup-nvidia.md#4-first-boot-checklist) — in short:
**Settings → System** shows the detector ready on `cuda`, live view plays, and a
walk-test in front of a camera produces an event with an annotated snapshot and, ~30
seconds after it ends, a [playable clip](docs/recordings.md#clip-lifecycle--why-a-fresh-event-briefly-shows-processing).

### Then: install the app on your phone

Push notifications and PWA install require HTTPS with a certificate your phone trusts.
Set that up first — [docs/remote-access.md](docs/remote-access.md) (Tailscale
recommended, LAN-only Caddy alternative) — then follow
[docs/mobile-pwa.md](docs/mobile-pwa.md) to install and enable notifications. Set
`PUBLIC_URL` (or **Settings → System → Public URL**) to the HTTPS URL you chose so
notification taps open the right address.

## Ports

| Port | Service | Notes |
|------|---------|-------|
| 8080 | web (nginx) | The app. PWA + `/api` + `/go2rtc`, single origin |
| 8443 | caddy | Optional HTTPS (set `COMPOSE_PROFILES=tls` in `.env`) |
| 8554 | go2rtc | RTSP restream for VLC etc. on the LAN |
| 8555 tcp+udp | go2rtc | WebRTC media — add this host's LAN/Tailscale IPs (`ip:8555`) under **Settings → System → WebRTC addresses** for sub-second live view |

**Never port-forward any of these to the internet.** See
[docs/remote-access.md](docs/remote-access.md).

## Self-hosting

Vigilume is built to be run by anyone, anywhere — there are no cloud dependencies.

**Server** — clone this repo, `cp .env.example .env`, set `ADMIN_PASSWORD` and
`MEDIA_PATH`, then `docker compose up -d --build` on a Linux host with an NVIDIA GPU
(driver + container toolkit: [docs/setup-nvidia.md](docs/setup-nvidia.md); Unraid
walkthrough: [docs/deploy-unraid.md](docs/deploy-unraid.md)). Detection models download
automatically (checksum-pinned); web-push keys self-generate; all data stays local.
Optional: Home Assistant MQTT ([docs/home-assistant.md](docs/home-assistant.md)).

**iOS app** — the native app ([docs/ios-app.md](docs/ios-app.md)) has no hardcoded
server: add any Vigilume URL and log in.

### Push notifications: pick one

Apple requires every push to be signed by a key belonging to the Apple team that
owns the app's bundle ID. That single fact decides your options — it is a
cryptographic constraint, not a networking one, so pushes for someone else's App
Store build can never be signed by your server, no matter what address it has.

| Option | Apple developer account | Event alerts | Doorbell rings like a call | Runs entirely on your box |
|---|---|---|---|---|
| **ntfy** | Not needed | Yes | No | Yes |
| **Your own build + your own relay** | Required (paid) | Yes | Yes | Yes |
| **Someone else's App Store build** | Not needed | Yes | Yes | No — needs that owner's relay |

- **ntfy** — install the ntfy app, generate a topic in Settings → Notifications, done.
  No Apple account, no relay, nothing exposed. Alerts carry a linked snapshot your
  phone fetches straight from your server. It cannot ring the phone like a call,
  because a CallKit ring needs PushKit VoIP, which is APNs.
- **Your own build + your own relay** — build the app in Xcode with your own bundle
  ID and Apple team, create your own APNs `.p8`, and run the bundled relay beside
  the NVR (`COMPOSE_PROFILES=relay`, `relay_url` = `http://push-relay:8090`). Full
  feature set including the CallKit doorbell ring, and no dependency on anyone.
- **Someone else's App Store build** — pushes from *your* server travel through that
  owner's relay **end-to-end encrypted**: the relay sees only ciphertext and a device
  token, and snapshots are fetched directly from your server, never through it
  ([docs/push-architecture.md](docs/push-architecture.md)). You need a relay URL from
  whoever publishes that build; this project does not operate a shared public relay.

**Push relay** — the tiny stateless container in [`relay/`](relay/) that holds the
APNs signing key. Everything it needs is in `relay/README.md`. The `.p8` is the only
secret; the Key ID, Team ID and Bundle ID are public identifiers (the Key ID rides in
the JWT header in cleartext) and are safe to commit.

**Updating a deployment** — edit/pull the source, then `./deploy.sh` syncs it to the
run location without touching `.env` or runtime data, and
`docker compose up -d --build` applies it.

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Architecture contract: services, API shapes, native engine, settings schema, capability map |
| [docs/native-mode-design.md](docs/native-mode-design.md) | Engine design record: model pins + hashes, decode math, tracker settings, ffmpeg/go2rtc patterns |
| [docs/recordings.md](docs/recordings.md) | Timeline scrubbing, event clips, the `processing` clip state, recorder logs, file layout + retention |
| [docs/cameras-amcrest.md](docs/cameras-amcrest.md) | Per-model camera onboarding: IPs, RTSP users, stream settings, AD410 specifics |
| [docs/mobile-pwa.md](docs/mobile-pwa.md) | Installing the PWA on iOS/Android, enabling notifications |
| [docs/remote-access.md](docs/remote-access.md) | HTTPS + remote access via Tailscale, LAN-only TLS alternative, WebRTC over the tailnet |
| [docs/live-latency.md](docs/live-latency.md) | Why live view is fast or slow: the WebRTC/MSE/HLS protocol map, zero-config WebRTC candidates + the `VIGILUME_WEBRTC_HOST` override, and the camera I-frame-interval fix |
| [docs/setup-nvidia.md](docs/setup-nvidia.md) | NVIDIA driver + container toolkit setup, GPU first-boot checklist |
| [docs/faq.md](docs/faq.md) | GPU checks, storage sizing, model choice, tuning, backup/restore, upgrades |

## Roadmap (documented, not built)

- Audio event detection (bark / scream / yell) — the old implementation rode on
  Frigate's audio classifier and was removed with it; a native classifier over the RTSP
  audio track is the planned replacement
- Zones (polygon-gated alerting)
- Native app-store wrappers (the PWA covers phones today)

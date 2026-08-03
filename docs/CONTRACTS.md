# Vigilume NVR — Architecture Contract

This document is the **source of truth** for service topology, API shapes, ports, env
vars, the native detection/recording engine, and per-camera capabilities. All components
MUST conform to it. Engine internals (model research, decode math, tracker settings,
ffmpeg patterns) are recorded in [native-mode-design.md](native-mode-design.md).

## What this system is

A lightweight, **fully standalone**, self-hosted security NVR built around three Amcrest
cameras (IP5M-T1277EW-AI, IP8M-2779EW-AI, AD410 doorbell). The backend owns everything:
object detection (D-FINE ONNX on an NVIDIA GPU), 24/7 recording (ffmpeg stream-copy),
live view (a Vigilume-managed go2rtc), smart event notifications with annotated
snapshots and in-frame counts, and a mobile-installable PWA. There is no Frigate and no
deployment-mode switch. The only MQTT is an **optional, outbound** Home Assistant
auto-discovery *publisher* (off by default; see **MQTT / Home Assistant** below) — it is
not a dependency, and it is unrelated to the removed inbound Frigate→MQTT feed.

## Service topology (docker-compose — ONE compose file, no override files)

| Service   | Image / build | Role | Ports (host) |
|-----------|---------------|------|--------------|
| `backend` | `./backend`, build target **`gpu`** (default): minimal `nvidia/cuda:12.9.1-base-ubuntu24.04` base + `onnxruntime-gpu==1.27.0` (CUDA-12 build) + pinned pip CUDA 12 / cuDNN 9 `nvidia-*-cu12` runtime wheels (on `LD_LIBRARY_PATH`) + apt `ffmpeg`; ~5 GB image | Detection, 24/7 recording + clips, events DB, notifications, Amcrest control, settings, auth, go2rtc config generator | — (proxied by `web`) |
| `go2rtc`  | `alexxit/go2rtc:1.9.14` (pinned) | RTSP restream fan-out (one camera session shared by live view + detect + record), MSE + WebRTC live view | 8554 (RTSP), 8555/tcp+udp (WebRTC). API port 1984 is **internal only** — nginx proxies it |
| `web`     | `./frontend` (nginx) | Serves PWA, reverse-proxies `/api` + `/go2rtc` | 8080 |
| `caddy`   | `caddy:2` (profile `tls`) | Optional LAN HTTPS via a self-signed internal CA — a secure context for the mic/PWA/push, fully offline | 8443 |

Internal DNS names: `backend`, `go2rtc`, `web`. The backend reaches the go2rtc API at
`http://go2rtc:1984` and its RTSP restream at `rtsp://go2rtc:8554`.

**Startup order:** `backend` starts first, writes `./go2rtc/config/go2rtc.yaml` from its
camera DB during boot, and `go2rtc` has `depends_on: backend: condition:
service_healthy` — so go2rtc always starts with a current config. The backend
healthcheck (`GET /api/system/health`) goes green as soon as the app itself is up;
detector/recorder trouble never fails the healthcheck (it shows up as
`detector.ready: false` instead).

**GPU:** the compose file gives `backend` an nvidia device reservation
(`deploy.resources.reservations.devices`) and sets `VIGILUME_REQUIRE_GPU=1` by default.
Host prerequisites (driver + NVIDIA Container Toolkit): [setup-nvidia.md](setup-nvidia.md).

**CPU build target:** `backend` also has a **`cpu`** Dockerfile target
(`python:3.12-slim` + CPU `onnxruntime`) for **dev/tests only** — build it explicitly
with `docker build --target cpu backend/`. Never deploy it for production detection.

**Profiles:** `relay` (the standalone APNs push relay) and `tls` (the LAN HTTPS
terminator — a self-signed Caddy on 8443 for a secure context on the local network).
Enable either via `COMPOSE_PROFILES=relay,tls` in `.env`.

## Native engine

### Detection model (D-FINE, pinned by revision and SHA-256)

The detector is **D-FINE** (Apache-2.0, DETR-family, NMS-free), run through ONNX
`settings.detection.backend` picks the SILICON: `"gpu"` (default — D-FINE ONNX on
CUDA, highest accuracy) or `"coral"` (SSDLite MobileDet on an Edge TPU, ~2 W but
COCO mAP ~33 vs ~54). Unlike `model`/`confidence`, which reconfigure the live
detector, `backend` is read once at boot by `build_detector` — **a change needs a
backend restart**. An explicitly-set `VIGILUME_DETECTOR` env var OVERRIDES the
stored value, so an operator can always force a backend without touching the DB.
Coral requires the hardware fitted; with no Edge TPU the detector reports
`ready: false` and detection does not run.

Runtime. Six models are supported via `settings.detection.model`; artifacts are the
`onnx-community` conversions of the official `ustc-community/dfine-*` checkpoints,
pinned by immutable revision URL + SHA-256 (pin table lives in
`backend/app/native/detector.py`). The COCO tiers share the COCO-80 output space; the
`_obj365` model uses the Objects365 vocabulary (see **Class vocabulary** below). All
share the SAME graph I/O and preprocessing, so the decode is identical — only the
`logits` class-count (80 vs 366) and the labelmap differ:

| key | vocabulary | mAP (val) | bytes | SHA-256 |
|---|---|---|---|---|
| `dfine_n` | COCO (80) | 42.8 (COCO) | 15,258,358 | `0f684f409618ee8a822410e754a29caa817d1aa16283ce89cad936d0a48e2f35` |
| `dfine_s` (**default**) | COCO (80) | 48.5 (COCO) | 41,535,197 | `cd8a49a945feda6d28c6304ae8ae85c2759ba1d78a5a83a22c5ce8db82ef7238` |
| `dfine_m` | COCO (80) | 52.3 (COCO) | 78,624,257 | `70aaa837978a06ba44ad17398c7079ae5a1a7b1a9032b5d7053981e1ada02d6b` |
| `dfine_l` | COCO (80) | 54.0 (COCO) | 125,348,332 | `d678f3baebfb909d3a20f21d1d807544d0172ed47fa1ab88e7fcdec7e365b236` |
| `dfine_x` | COCO (80) | 55.8 (COCO) | 251,138,448 | `644fb5124c9c035a6082f23419da693c19ac857bd984ab7af5e353779368a03b` |
| `dfine_l_obj365` | Objects365 (365) | 44.7 (Objects365) | 125,936,360 | `cd0dfa92a2e0e2ab3d4a7c2e6252ebc094aa77b67f2f9dc1a010dd350a9a2f3e` |

**Model bootstrap:** at first boot the backend downloads the selected model into
`/data/models/{key}.onnx` (`.part` file → SHA-256 verify → atomic rename → `{key}.json`
sidecar recording url/sha256/downloaded_at). Download or verification failure ⇒ loud
log, background retry with backoff, `detector.ready: false` — **the app keeps serving
the UI, it never crashes**. Offline installs: place the file at that path yourself and
check its hash against the table above.

**GPU requirement (`VIGILUME_REQUIRE_GPU`):** ONNX Runtime silently falls back to CPU
when the CUDA execution provider can't load; Vigilume refuses to let that pass
unnoticed.

- `VIGILUME_REQUIRE_GPU=1` (compose default): CUDA EP unavailable ⇒ **hard detector
  failure** — `detector.ready: false`, `device: null`, ERROR log
  `GPU UNAVAILABLE — set VIGILUME_REQUIRE_GPU=0 to accept CPU`. The UI, recording and
  live view keep working; only detection is down.
- `VIGILUME_REQUIRE_GPU=0` (dev): CPU inference is accepted — `device: "cpu"`,
  `ready: true`, WARNING log with the measured per-frame cost.
- CUDA active ⇒ `device: "cuda"` and an INFO log
  `GPU OK — provider=CUDAExecutionProvider model=<key> warmup_ms=<x> infer_ms=<y>`.

Inference contract: 640×640 input (plain resize, no letterbox/normalize), sigmoid decode
of 300 queries (per-query argmax over the model's class-count — 80 or 366), threshold at
`settings.detection.confidence`, boxes emitted directly in detect-stream pixels. No NMS
anywhere.

**Class vocabulary.** The decode returns contiguous `class_id`s; the string label is
resolved from the ACTIVE model's labelmap (`backend/app/native/coco_labels.py`
`LABELMAPS`: `coco` = COCO-80, `obj365` = Objects365 366-entry, id 0 a background
`"none"` placeholder). `coco_labels.ID_TO_LABEL` is a single live view the engine imports
once and resolves `class_id → label` through; the detector re-points it (in place) to the
loaded model's labelmap on every model swap, so switching to `dfine_l_obj365` maps ids to
Objects365 labels with no change to the tracking/annotate path (labels are just strings;
per-camera `detect_objects` filtering compares against the active labelmap). The active
model's ordered vocabulary is served by `GET /api/detection/labels` (below) so the
per-camera object picker lists exactly the classes the running model can detect.

### Detector backend selection (`VIGILUME_DETECTOR`)

The inference backend is chosen by `VIGILUME_DETECTOR` (parsed in `config.py`; a factory
`native/detector.build_detector` constructs it). Both backends implement the SAME
interface the ingest worker + self-heal supervisor drive (`ready`/`device`/`kind`/
`model_key`/`status()`/`detect()→sv.Detections`/`start`/`stop`/`reconfigure` +
`note_detect_ok`/`note_detect_failure`/`needs_reinit`/`reinit`), so everything downstream
(tracking/events/annotate/recording/notify) is unchanged.

| `VIGILUME_DETECTOR` | Backend | `kind` | `device` | `VIGILUME_REQUIRE_GPU` |
|---|---|---|---|---|
| `onnx` (**default**) | D-FINE ONNX (above) | `onnx` | `cuda`/`cpu`/`null` | honored |
| `onnx_cpu` | D-FINE ONNX forced to CPU | `onnx` | `cpu`/`null` | overridden (off) |

An unknown value falls back to `onnx` with a WARNING. Precedence: `VIGILUME_REQUIRE_GPU`
is meaningful ONLY for the `onnx` kind — `onnx_cpu` forces CPU regardless.

### Tracking → events

Pinned stack: `supervision>=0.29.1,<0.30` + `trackers==2.4.0` (`ByteTrackTracker`, one
instance **per camera**). Do NOT use `sv.ByteTrack` (removed in supervision 0.30) and do
NOT bump `trackers` to 2.5.0 (broken/empty wheel on PyPI).

Event state machine (per camera, per label — `backend/app/native/engine.py`):

- A track **confirms** after 3 frames carrying its tracker_id; the first confirmed track
  of a label opens an event (`type:"new"`).
- One open event per `(camera, label)`; more simultaneous objects of that label raise
  `count`, not extra events. `update` emits on best-score +0.02, active-count change, or
  a 10 s heartbeat.
- `end` after 5 s without the label; `end_time` = last time it was seen. The engine then
  asks the recorder to cut a clip.
- The engine synthesizes **Frigate-shaped** `{type: new|update|end, after: {...}}`
  payloads and calls the existing `EventsPipeline.handle_event()` **in-process**, and
  feeds the live in-frame count cache via `pipeline.update_count()` — so counts are
  exact and the events/notify/WS/UI surface is unchanged.
- Snapshots: the engine keeps the raw detect-res frame with the best track confidence
  per open event; the pipeline annotates it with Supervision (boxes, labels, count
  banner) and stores `/data/snapshots/{id}.jpg`.

### Event id conventions (`frigate_id` column — name kept for schema stability)

| Prefix | Meaning | Media |
|---|---|---|
| `native.` | Engine-detected object event: `native.{epoch_ms}-{6 hex}` | snapshot + clip |
| `cameraai.` | Camera-AI-only object event (`camera_ai_only` mode; no server inference ran): `cameraai.{epoch_ms}-{label}` | snapshot only (no clip) |
| `doorbell.` | Synthetic AD410 button-press event: `doorbell.{epoch_ms}` | snapshot only (no clip) |
| `audio.` | Legacy audio-event rows from pre-standalone installs — **no longer produced**, tolerated in the DB | none |

The `source` column on cameras (`'manual'`) is likewise kept harmlessly; nothing sets
`'frigate'` anymore.

### Frame ingest

One ffmpeg child per **detect-enabled** camera decoding the substream via Vigilume's own
restream (`rtsp://go2rtc:8554/{name}_sub` — one RTSP session per camera stream total,
respecting Amcrest concurrent-session limits), `-vf fps={detect_fps}`, rawvideo bgr24 to
stdout. Latest-frame **drop** (never queue), 15 s staleness watchdog, kill + respawn
with 2 s → 60 s backoff. A single inference worker services all cameras.

**Self-healing (the detection path can never silently wedge).** The path is purely
frame-driven, so four safety nets keep it live when frames stop, the detector goes
un-ready, or a `detect()` call hangs:

- **Worker heartbeat** — the single worker blocks on its wake Event with a
  `HEARTBEAT_TICK_S` (1 s) timeout, never `wait()` forever. Each tick, a source with a
  **fresh** frame is processed as before; a source that has gone stale (no fresh frame
  for ≥ 1 s) is ticked with **empty observations** (`frame_bgr=None`) so the engine still
  ends its open events by absence during an ingest stall. Stale ticks re-run **no**
  inference (no wasted/duplicate `detect`, no busy-spin — the 1 s timeout bounds CPU).
- **Per-inference timeout** — each `detect()` runs under `asyncio.wait_for` with an
  `DETECT_TIMEOUT_S` (8 s) bound. On timeout or exception the frame yields empty
  observations and the detector's consecutive-failure counter ticks; the worker is never
  blocked past the timeout (a timed-out inference thread may linger — bounded, logged).
- **Detector auto-reinit** — `DETECT_FAILURE_THRESHOLD` (3) consecutive failures flag the
  detector for reinit. A supervisor coroutine (in the ingest manager) rebuilds the ORT
  session in the background (`OnnxDetector.reinit`) — cooldown-guarded (`REINIT_COOLDOWN_S`
  ≥ 20 s) so it never thrashes. On success `ready` flips true again (`detector recovered`
  INFO); on failure it stays `ready:false` and retries after the cooldown. This recovers a
  CUDA/GPU dropout without a container restart.
- **Frame-starvation visibility** — a source that produces **no** frame for > 90 s across
  respawns logs `ingest {cam}: no frames for {n}s despite {k} respawns — check the
  camera/go2rtc sub-stream`, so the real cause is visible (and surfaced in status, below).

**go2rtc RTSP transport (researched, not changed):** go2rtc's native RTSP client already
uses **TCP** (interleaved RTP over the control connection — verified in `pkg/rtsp/conn.go`;
there is no UDP client and no `#transport=tcp` keyword, `transport` selects WebSocket) and
already sends RTSP `OPTIONS` **keepalive** (~25 s) with a read deadline (~30 s default). The
detect ffmpeg pipe also passes `-rtsp_transport tcp`. The only safe, source-verified lever
is the `#timeout=N` (seconds) fragment on an rtsp source; because TCP + keepalive are
already the default, the generated URLs are left **as-is** (per "don't guess-break the
working config") and recovery is handled by the self-heal nets above.

### Camera-AI-gated detection (offload the GPU to the cameras' own AI)

The Amcrest/Dahua models (`IP8M-2779EW-AI`, `IP5M-T1277EW-AI`, `AD410` — the
`ai_on_camera` capability) run **on-device AI**: SMD human/vehicle + IVS tripwire/intrusion.
Vigilume can use that native event stream to **gate** server-side GPU inference so the
detector only works when a camera actually flags something — the load win when many cameras
share one GPU.

**Per-camera `detect_mode`** (schema v11; `cameras.detect_mode`, nullable) + the fallback
**`settings.detection.default_mode`** (default `always`) for cameras whose stored mode is
NULL/unset. `effective_detect_mode(stored, default)` resolves the two; an unknown value can
never disable detection — it degrades to `always`.

- **`always`** — continuous server inference (historical behavior; unchanged).
- **`camera_ai`** — the camera's ffmpeg ingest still runs (live view / frame cache work), but
  the per-frame `detector.detect()` call is **skipped unless the camera's on-board AI is
  active**. "Active" = from an AI **Start** until its **Stop** plus an `AI_ACTIVE_COOLDOWN_S`
  (8 s) tail (a momentary IVS **Pulse** stays active for that tail; a missed Stop is bounded
  by `AI_ACTIVE_MAX_S` = 300 s). The gate is a cheap mode-check + one state lookup **per
  frame**, not per detection; a gated-off frame is still fed to the engine with **empty
  observations** (frame cache + absence-based event ending keep working).
- **`camera_ai_only`** — **no server inference at all** (the camera spawns no ingest ffmpeg
  source). Events are created **directly** from the camera's ONVIF AI notifications: the topic
  (+ any `ObjectType` SimpleItem) maps to a label (Human/Person→person, Vehicle/Car→car, else
  a generic `motion`), a live snapshot is grabbed for the event image, and the event respects
  the camera's
  `detect_objects` filter + a per-`(camera,label)` de-dupe cooldown; notifications ride the
  same enabled/labels/min_score/cooldown gates as engine object events. These rows use the
  `cameraai.` id prefix (snapshot-only, no clip).

**AI event listener** (`backend/app/amcrest/ai_events.py`): camera-AI activity is sourced
over **ONVIF PullPoint**, not the Dahua CGI event stream. The CGI attach
(`eventManager.cgi?action=attach`) pushes **nothing** on the Amcrest AI turrets (firmware
limitation, validated live against a 1277EW-AI), but ONVIF PullPoint works. Per
`ai_on_camera` camera in a camera-AI mode, the listener (via `onvif-zeep`, camera HTTP
port 80, stored admin creds) builds an ONVIF events service + PullPoint subscription and
loops `PullMessages({"Timeout": 30 s, "MessageLimit": 100})` — every blocking ONVIF call
runs in a **thread** (`asyncio.to_thread`), never on the event loop (zeep is synchronous).
Each `NotificationMessage` is parsed for its **Topic** + **SimpleItem**s: a fire topic
(motion / tamper / line-cross / intrusion / object-/human-/vehicle-detect …) with
`IsMotion=true` or MotionAlarm `State=true` is a **Start** (motion active), `false` a
**Stop**; a fire topic with no boolean item is a momentary **Pulse**; any other topic is
logged `[unmapped]` and ignored. Labels: a Human/Person topic or `ObjectType` item →
`person`, Vehicle/Car → `car`, else generic `motion` (the server D-FINE refines it for
`camera_ai`). The subscription auto-reconnects with 5 s→120 s backoff and is best-effort
renewed; a camera that is offline / not ONVIF-capable / auth-fails logs + backs off and
never errors the app. The listener exposes the live "AI active" state to the ingest gate
(`camera_ai`) + status API and creates events for `camera_ai_only`. Every parsed
notification is logged one line prefixed `ai_event` — surface real events with
`docker compose logs backend | grep ai_event`. Dependency: `onvif-zeep` (bundles the ONVIF
WSDL tree as package data; imported lazily so the app boots without a camera reachable).

### Recording + clips

- One ffmpeg child per **record-enabled** camera, stream-copying the go2rtc **main**
  restream (audio already AAC via the `#audio=aac` source) into **10 s MPEG-TS
  segments**: `/media/native/recordings/{camera}/{YYYY-MM-DD}/{HH}/{MM.SS}.ts`.
  Watchdog: process exit or no new segment for 30 s ⇒ respawn with backoff.
- **Event clips:** ~20 s after `end` (guarantees the covering segment closed), the
  recorder concat+stream-copies segments spanning `[start−5 s, end+5 s]` into
  `/media/native/clips/{event_id}.mp4` (`-movflags +faststart`; `event_id` is the DB row
  id), then sets `has_clip`. Missing segments ⇒ log once, `has_clip` stays false, no
  retry loop.
- **Retention** (driven by `settings.recording`): recording hour-dirs older than
  `continuous_days`; clip files older than `event_days`; event rows + snapshots pruned
  at max(`event_days`, `snapshot_days`). Low-disk guard: <5 GB free on the media
  filesystem ⇒ prune oldest recording hours regardless of retention, loudly.

### go2rtc config (generated by the backend)

`backend/app/native/streams.py` renders `./go2rtc/config/go2rtc.yaml` from the camera DB
at boot, after camera CRUD, and after `system.webrtc_candidates` changes:

```yaml
api:    { listen: ":1984" }
rtsp:   { listen: ":8554" }
webrtc:
  listen: ":8555"
  candidates: [<settings.system.webrtc_candidates...>, "stun:8555"]
streams:
  {name}:      [<main_url>, "ffmpeg:{name}#audio=aac"]   # record + live
  {name}_sub:  [<sub_url>]                               # detect ingest
```

`main_url`/`sub_url` are the per-camera override columns; blank means "derive the
Amcrest default from ip + credentials" (subtype=0/1, credentials **percent-encoded**).
The `#audio=aac` source transcodes G.711 → AAC so recording (MPEG-TS), MSE and WebRTC
all get legal audio. Writes are idempotent (skip when content unchanged). Stream-only
changes (routine camera CRUD) are applied **incrementally** via `PUT/DELETE /api/streams`
(PUT carries the stream's full source list) so live streams for other cameras keep
running; listener/candidate changes and the first sync after backend boot use
`POST /api/restart`, falling back to per-stream PUTs. go2rtc being down never breaks
camera CRUD.

**WebRTC candidates:** inside Docker go2rtc can't guess host IPs — operators add this
server's LAN and/or Tailscale IPs as `ip:8555` entries under **Settings → System →
WebRTC addresses**. If no candidate is reachable the frontend's player falls back to MSE
automatically ([remote-access.md](remote-access.md)).

## Nginx routing (inside `web`)

- `/` → SPA static files (PWA: manifest, service worker at root scope)
- `/api/` → `http://backend:8000/api/` (incl. WebSocket upgrade on `/api/ws`)
- `/go2rtc/` → the container's `GO2RTC_UPSTREAM` env (default `http://go2rtc:1984/`,
  host-side override `VIGILUME_GO2RTC_UPSTREAM` in `.env`; WebSocket upgrade; live
  view signaling, e.g. `/go2rtc/api/ws?src=<camera>` — WebRTC first, MSE fallback)

Single origin = no CORS, and the service worker can intercept everything.

## Backend REST API (all under `/api`, JSON, Bearer auth except where noted)

Auth & roles (RBAC — two roles, `admin` and `viewer`):
- `POST /api/auth/login` `{username?, password}` → `{token, role, username}` (JWT HS256;
  secret auto-generated, persisted in `/data`). `username` defaults to `"admin"`, so
  legacy single-password clients that POST only `{password}` keep logging in as the
  built-in admin. Verification: `username=="admin"` → compare `ADMIN_PASSWORD` env (always
  role `admin`); any other username → look up the `users` row + verify its pbkdf2 hash
  (role from the row). Existing login rate-limit is unchanged. Wrong credentials → 401.
- The session JWT carries `sub=<username>` and `role=<role>`; media tokens carry `role`
  too. **Backward compat:** a legacy token with `sub:"admin"` and NO `role` claim is
  treated as role `admin`, so pre-RBAC sessions keep working.
- `GET /api/auth/me` (any authenticated) → `{username, role}`.
- All other routes require `Authorization: Bearer <token>`; WS accepts `?token=`.
- `require_auth` = any authenticated; `require_admin` = authenticated AND role `admin`
  (else **403** `"Admin access required"`).
- **Roles:** `admin` = full access. `viewer` may VIEW live/events/recordings/timeline/grid,
  use GROUPING (create/edit/delete/reorder camera groups — groups are SHARED across users),
  and enable/disable PUSH on their own device. A viewer may NOT reach camera
  settings/device controls, global settings, detection-model management, camera CRUD, or
  user management (all → 403).
- Media routes (event snapshot/clip, camera snapshot: `GET /api/events/{id}/snapshot.jpg`,
  `GET /api/events/{id}/clip.mp4`, `GET /api/cameras/{name}/snapshot.jpg`; and all recordings
  media — `playlist.m3u8`, `seg/{ts}.ts`, `export.mp4`) additionally accept
  `?token=` carrying a session **or media-scope** JWT — push-notification images are fetched by
  the browser/OS without headers. Media-scope tokens are rejected on all non-media routes.

Users (DB-backed accounts; **admin only** except self password change; `routers/users.py`):
- Table `users(id, username UNIQUE, password_hash, role CHECK(admin|viewer), created_at)`,
  migrated via `SCHEMA_VERSION` (v5). Passwords are stored as
  `pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>` (stdlib `hashlib.pbkdf2_hmac`, ≥200 000
  iters, per-user `secrets.token_bytes` salt; verified with `hmac.compare_digest`). The
  built-in admin (`username "admin"`) is env-controlled and is **never** a row here —
  `"admin"` is a RESERVED username.
- `GET /api/users` → `[{id, username, role, created_at}]` (**never** returns hashes).
- `POST /api/users` `{username, password, role?}` → **201** `{id, username, role, created_at}`
  (role default `viewer`; **409** duplicate; **400** reserved name `"admin"`; username charset
  `[a-z0-9][a-z0-9_.-]{2,31}`; password min length 8 → **422**).
- `PUT /api/users/{id}` `{password?, role?}` → reset password / change role. **404** unknown
  id; **400** demoting the last DB admin. (The built-in admin has no id, so it can never be
  targeted.)
- `DELETE /api/users/{id}` → **204** (**404** unknown id).
- `POST /api/users/me/password` (any authenticated **DB** user) `{current_password, new_password}`
  → **204**; **400** for the built-in admin (env-controlled), **401** wrong current password.

Authorization matrix (every route classified):
- **Any authenticated (admin OR viewer):** `GET /api/cameras`; `GET /api/cameras/{name}/snapshot.jpg`;
  all `GET /api/events*` + snapshot + clip + `DELETE /api/events/{id}`; all `GET /api/recordings/*`;
  `/api/groups` (GET/POST/PUT/DELETE) **except** that `PUT`/`DELETE` on a group SELECTED BY
  Privacy Mode are admin-only — see the Privacy Mode note below;
  `GET /api/notifications/vapid-public-key` (no-auth),
  `POST /api/notifications/subscribe`, `POST /api/notifications/unsubscribe`,
  `POST`/`DELETE /api/notifications/apns/register`, `GET /api/notifications/apns/devices`
  (prefixes only); `WS /api/ws`;
  `POST /api/cameras/{name}/ptz` and `WS /api/cameras/{name}/talk` — **viewer-accessible
  by product decision**: aiming a camera and talking through it are live-interaction
  actions (they store nothing and are bounded by the camera's own limits), the same class
  as watching the stream. Both remain capability-gated (`ptz` / `speaker`). The talk WS
  still refuses **media-scope** tokens (1008) — those are the long-lived, widely shared
  image tokens from notifications/MQTT and must never open a live mic;
  `GET /api/auth/me`; `GET /api/system/health` (no-auth).
- **Admin only (`require_admin` → 403 for viewers):** `POST`/`PUT`/`DELETE /api/cameras` +
  `PUT /api/cameras/order`; `GET`/`PUT /api/cameras/{name}/settings` + `POST .../light` +
  `POST .../siren` + `POST .../reboot` + `POST .../probe`;
  `GET`/`PUT`/`PATCH /api/settings`; ALL `/api/detection/models/*`; `GET /api/system/detector`;
  `POST /api/integrations/mqtt/test`;
  `POST /api/notifications/test`; ALL `/api/users/*` (except `POST /api/users/me/password`).

System:
- `GET /api/system/health` (**no auth**, used by the compose healthcheck) →

```json
{
  "status": "ok",
  "version": "2.0.0",
  "detector": {"kind": "onnx", "ready": true, "device": "cuda", "model": "dfine_s"},
  "go2rtc": true,
  "cameras_online": 3
}
```

  `detector.kind` is `"onnx"`; `detector.device` is
  `"cuda" | "cpu" | null`. Returns 200 as soon as the app is
  up — subsystem trouble is reported in the fields, never as a failing healthcheck.

- **`GET /api/system/detector`** (auth) — full detector self-test:

```json
{
  "kind": "onnx",
  "ready": true,
  "device": "cuda",
  "model": "dfine_s",
  "model_sha_ok": true,
  "last_inference_ms": 9.4,
  "consecutive_failures": 0,
  "needs_reinit": false,
  "last_reinit_age_s": null,
  "model_state": "ready",
  "model_progress_pct": 100,
  "per_camera": [
    {"name": "front_yard", "ingest_ok": true, "fps": 5.0, "last_frame_age_s": 0.2,
     "stalled": false, "respawns": 0, "ai_active": false,
     "ai_events": {"mode": "camera_ai", "connected": true, "ai_active": false,
                   "ai_labels": [], "fire_count": 3, "last_fire_age_s": 12.5}}
  ]
}
```

  `per_camera` covers detect-enabled cameras (`ingest_ok` = a frame arrived <15 s ago;
  `stalled` = no frame for ≥ the 15 s watchdog window / never; `respawns` = ffmpeg
  respawn count for that source). `ai_active` is the live camera-AI gate state; `ai_events`
  is the per-camera AI-listener detail (`mode`, stream `connected`, `ai_labels`,
  `fire_count`, `last_fire_age_s`) or `null` when no AI watcher runs for that camera (mode
  `always` / no on-board AI). The top-level `consecutive_failures` / `needs_reinit` /
  `last_reinit_age_s` (seconds since the last reinit attempt, or `null`) surface the
  detector's self-heal state, so the user/UI can see WHICH camera is starved or that the
  detector's auto-reinit fired (see **Frame ingest → Self-healing**).
  `model_state`/`model_progress_pct` mirror the ModelStore state of the **active**
  model (`"absent" | "downloading" | "verifying" | "ready" | "error"`, 0–100) so the
  System detector card can show download/load progress. See **Detection models** below.

Cameras:
- `GET /api/cameras` → `[{name, friendly_name, model, ip, online, source, needs_credentials, capabilities: {ir, white_light, siren, mic, speaker, doorbell, ai_on_camera, backchannel, ptz, night_vision}, detect_objects: string[], exempt_zones: [{name, points: [[x,y],…]}], detect: {enabled}, record: {enabled}, detect_fps, detect_mode, detect_mode_stored, ai_active, private, main_url, sub_url}]`
  - `private` (bool) is Software Privacy Mode's RESOLVED effect for this camera: the
    backend is recording/detecting/streaming NOTHING for it. **Any-authenticated on
    purpose** — it is the only privacy signal a viewer gets (`/api/privacy` itself is
    admin-only), and it reveals the effect without revealing the camera/group selection
    behind it. Every client renders its "Privacy Mode" panel from THIS field, and that
    panel must win over `online`: the camera's go2rtc streams have been deleted, so
    showing "offline" would misreport a deliberate choice as a fault. Surfaces: web
    Dashboard tile, TV mode, camera detail; iOS dashboard tile, camera detail,
    fullscreen.
  - `detect_mode` is the camera's **effective** server-detection mode — one of `always` |
    `camera_ai` | `camera_ai_only` — resolving `detect_mode_stored` (the raw stored value,
    which may be **null** = "inherit") against `settings.detection.default_mode`. See
    **Camera-AI-gated detection** below. `ai_active` is a live boolean: `true` while the
    camera's own on-board AI is currently firing (only meaningful in `camera_ai`/
    `camera_ai_only`; always present, `false` otherwise).
  - `detect_objects` is the camera's **stored** tracked-object list, returned verbatim.
    An **empty list `[]` means record-only**: the camera records but detects nothing (no
    events, and no ingest ffmpeg/inference session at all). A non-empty list is the exact
    set of labels tracked. The API never coerces `[]` to the defaults, so the object picker
    round-trips exactly what is stored.
  - `exempt_zones` are per-camera **exempt (privacy / ignore) detection zones**: a list of
    polygons, each `{name: string, points: [[x,y],…]}` where `points` are **normalized
    0..1** coords (origin top-left, resolution-independent). Any detected object whose box
    **foot-center** — the bottom-edge midpoint `((x1+x2)/2, y2)`, i.e. where a person/vehicle
    meets the ground — falls inside **any** polygon is suppressed: no event, no live count,
    no notification, no annotation (masking happens before track confirmation). The engine
    converts each polygon to detect-stream pixels (`x*detect_width`, `y*detect_height`) once
    per camera-row change and runs a ray-cast point-in-polygon test; polygons with fewer than
    3 points are ignored. An **empty list `[]` = no masking** (unchanged behavior). Returned
    verbatim, so the zone editor round-trips exactly what is stored.
  - `detect.enabled` gates the native ingest+detection loop for the camera;
    `record.enabled` gates its 24/7 recorder process.
  - `detect_fps` (1–10, default 5): rate the engine samples the substream.
  - `detect_mode` (`always` | `camera_ai` | `camera_ai_only`, or **blank/null** to inherit
    `settings.detection.default_mode`): gates whether/when the GPU runs on this camera — see
    **Camera-AI-gated detection** below.
  - `main_url`/`sub_url`: optional RTSP overrides (non-Amcrest cameras, odd ports);
    empty string = Amcrest default derived from ip + credentials.
  - `needs_credentials` = stored username **or** password is empty.
  - Rows are ordered by the stored `position`, then `name` (the "All cameras" dashboard
    order); new cameras append at the end.
- `PUT /api/cameras/order` `{names: string[]}` — assigns display positions in the given
  order; names not listed keep their relative order after the listed ones (unknown names
  ignored). Returns the reordered camera list (same shape as `GET /api/cameras`).
- `POST /api/cameras` — add camera `{name, friendly_name, model, ip, username, password, detect_objects?, exempt_zones?, detect_enabled?, record_enabled?, detect_fps?, detect_mode?, main_url?, sub_url?}`
  (toggles default `true`; `main_url`/`sub_url` are 0–512 chars and must start with
  `rtsp://` when non-empty; `name` is a lowercase slug `[a-z][a-z0-9_]{0,31}` — it
  becomes the stream name). `detect_objects` **omitted** stores the defaults
  `["person","dog","cat","car"]`; an explicit **`[]`** stores a record-only camera
  (detects nothing). The add-camera form prefills the defaults, so new cameras detect
  unless the operator clears the picker. `exempt_zones` **omitted** stores `[]` (no
  masking); each submitted zone is `{name?, points: [[x,y],…]}` with coords **clamped to
  [0,1]**, and zones with fewer than 3 points are dropped. `detect_mode` **omitted** (or
  blank `""`) stores **null** (inherit `settings.detection.default_mode`); a valid mode
  (`always`/`camera_ai`/`camera_ai_only`) is stored verbatim, and an unknown value is
  **rejected** (422) so a typo can never silently disable/gate detection.
- **Migration (schema v6):** the empty-list semantics changed — `[]` used to mean "the
  defaults" and now means "record-only". A one-time boot migration backfills every existing
  camera whose stored `detect_objects` was NULL / `''` / `'[]'` with the defaults, so
  deployed cameras keep detecting `person/dog/cat/car`. After the migration an empty list
  only ever results from an explicit user action (clearing the object picker / sending `[]`).
- **Migration (schema v8):** adds `cameras.exempt_zones TEXT NOT NULL DEFAULT '[]'`. Existing
  rows default to `[]` (no masking, unchanged behavior); no data is rewritten.
- **Migration (schema v9):** adds `events.box TEXT NOT NULL DEFAULT '[]'` (the best
  detection box in detect-stream px; used to annotate event snapshots) and the
  `detection_suppressions` table (`id, camera, label, foot_x, foot_y, created_at`,
  indexed by `camera`). The v8→v9 migration preserves existing rows and rewrites no
  data. **The reject-to-suppress feature that once used this table was removed** —
  the table + migration remain in place (**dormant**; no code reads or writes it).
  Use **exempt zones** to ignore detections instead. `events.box` is retained (it
  backs snapshot annotation).
- **Migration (schema v11):** camera-AI-gated detection. Adds nullable `cameras.detect_mode
  TEXT` (no SQL default). Existing rows get **NULL** = "unset" → inherit
  `settings.detection.default_mode` (itself defaulting to `always`), so deployed cameras keep
  the historical continuous-inference behavior untouched. The v10→v11 migration preserves
  existing rows and rewrites no data.
- `PUT /api/cameras/{name}` — edit (same shape; omitted optional fields keep stored
  values, blank username/password keep stored credentials). Sending `detect_objects: []`
  empties the list (record-only) and, via the CRUD resync below, stops that camera's
  detection ingest without a restart; re-adding an object restarts it. Omitting
  `detect_objects` keeps the stored list. **`exempt_zones`** follows the same rule: omitting
  it keeps the stored zones, an explicit `[]` clears them, and any other list replaces them
  wholesale (the engine reload picks up the new masking without a restart). When the stored
  model is `"unknown"`/empty and the update supplies working credentials, the backend
  re-probes `getDeviceType` and adopts the matched model (best-effort). **`detect_mode`**:
  omitting it keeps the stored value; a blank `""` clears it back to inherit; a valid mode is
  stored verbatim. A mode change re-gates ingest and starts/stops the camera-AI watcher live.
- `DELETE /api/cameras/{name}` → 204.
- **CRUD side effects:** every add/edit/delete regenerates + syncs the go2rtc config,
  reloads the detection engine and recorder, and **resyncs the camera-AI event watchers**
  (start/stop/reconnect per the new modes/credentials) so running ffmpeg children and AI
  streams match the camera set. No restarts of anything else.
- `GET /api/cameras/{name}/settings` → Amcrest device state `{ir_mode: "auto"|"on"|"off", day_night?: "color"|"black_white"|"brightness", night_vision_mode?: "auto"|"color"|"bw", white_light?: {mode: "off"|"on"|"auto", brightness: 0-100}, flip, osd_name, motion_detect, volume?: {mic, speaker}}` (fields present only if capability exists; `night_vision_mode` only when `capabilities.night_vision`)
- `PUT /api/cameras/{name}/settings` — patch of the same shape, applied to the device over the Amcrest HTTP API (a `white_light` patch may carry `mode` and/or `brightness`). `night_vision_mode` (`"auto"|"color"|"bw"`, capability-gated on `night_vision` — the IP4M-1056E) writes the Dahua `VideoInDayNight[0][N].Mode` (`Auto|Color|BlackWhite`) to **every** exposed day/night profile; a non-`night_vision` camera → **400**. Distinct from `ir_mode` (IR illuminator) which is unchanged.
- `POST /api/cameras/{name}/light` `{mode: "off"|"on"|"auto", brightness?: 0-100}` (white light / spotlight, capability-gated: the two EW turrets. Driven via the Dahua `Lighting_V2` white-LED slot — config names verified against the rroller/dahua HA integration, see `backend/app/amcrest/client.py`. Returns **501** if the firmware rejects the CGI)
- `POST /api/cameras/{name}/siren` `{duration_s?: int=10}` (capability-gated; AD410 = **generated alarm tone played via `audio.cgi` `postAudio`** — there is no public siren CGI. Returns **501** if the firmware rejects the tone playback)
- `POST /api/cameras/{name}/probe` → `{ok: bool, model: string|null, capabilities: {...}, detail: string|null}` — probes the device with the stored credentials right now (`getDeviceType` + capability probe, **8 s total cap**). On success adopts a matched model when the stored one is `"unknown"`/empty and refreshes cached capabilities; on failure `ok: false` with `detail` `"authentication failed"` or `"camera unreachable"`.
- `WS /api/cameras/{name}/talk?token=` — push-to-talk uplink (capability-gated on `speaker`; **session** JWT only — media-scope tokens rejected). The client sends **binary** frames of raw little-endian Int16 PCM, **mono, 8 kHz** (the browser downsamples); the backend transcodes to G.711 A-law and streams to the camera over the same `audio.cgi` `postAudio` mechanism as the siren. One active talker per camera. Text frames are ignored except `{"type":"stop"}`. Close codes: `4003` no speaker, `4009` busy (second talker), `4502` camera rejected audio/unreachable, `1000` clean stop (client stop/close or the **120 s** session cap).
- `POST /api/cameras/{name}/ptz` `{action, direction?, speed?, index?}` — pan/tilt + preset control, capability-gated on `ptz` (the IP3M-941B dome; non-PTZ camera → **400**). Drives the camera's Dahua `ptz.cgi`:
  - `action: "move"` — continuous move; `direction ∈ {up,down,left,right,upleft,upright,downleft,downright}` (required), `speed: 1-8` (default 4) → `ptz.cgi?action=start&channel=1&code=<Up|Down|Left|Right|LeftUp|RightUp|LeftDown|RightDown>&arg1=0&arg2=<speed>&arg3=0`
  - `action: "stop"` — `direction` required → `ptz.cgi?action=stop&…` with the same `code`
  - `action: "preset_set"|"preset_goto"|"preset_clear"` — `index: 1-3` (required) → `ptz.cgi?action=start&channel=1&code=<SetPreset|GotoPreset|ClearPreset>&arg1=0&arg2=<index>&arg3=0`

  Returns **204** on success; missing `direction`/`index` for the action → **422**; firmware without `ptz.cgi` → **501**; transient device rejection → **502**. (Talk on the IP3M-941B auto-routes to the RTSP backchannel via `capabilities.backchannel`, same as the AD410 — no separate control.)
- `POST /api/cameras/{name}/reboot`
- `GET /api/cameras/{name}/snapshot.jpg` — live snapshot from the engine's latest
  decoded frame; when ingest is down (or detection disabled) the backend falls back to
  the camera's own CGI snapshot.

Camera groups (named, ordered camera subsets for the dashboard selector / TV mode):
- DB table `camera_groups` (`id` INTEGER PK AUTOINCREMENT, `name` TEXT UNIQUE NOT NULL,
  `cameras` TEXT NOT NULL default `'[]'` — JSON array of camera names **in display
  order**, `position` INTEGER NOT NULL), migrated via the `SCHEMA_VERSION` pattern.
  Unknown/deleted camera names stored in a group are **tolerated**: filtered out of API
  responses at read time, never an error (a camera recreated with the same name
  re-appears in its groups).
- `GET /api/groups` → `[{id, name, cameras: string[], position}]` ordered by `position`.
- `POST /api/groups` `{name, cameras?: string[]}` → **201** created row
  (`position` = max+1). **409** on duplicate name.
- `PUT /api/groups/{id}` `{name?, cameras?, position?}` — partial update; `cameras`
  REPLACES the full ordered list (reorder = PUT with the new order). **409** on rename
  to an existing name, **404** unknown id.
- `DELETE /api/groups/{id}` → 204 (**404** unknown id).
- **Privacy Mode interlock (403):** `PUT`/`DELETE` on a group whose id appears in the
  Privacy Mode `groups` selection require **admin**; a viewer gets **403** with detail
  `"This group is used by Privacy Mode; only an admin can change it"`. Group membership
  feeds the resolved private set, so without this a viewer could empty or delete a
  private group and RESUME capture on cameras an admin switched off (or add cameras and
  blind them). Groups Privacy Mode does not reference stay fully viewer-writable.

Events (backend's own SQLite DB, produced by the native engine):
- `GET /api/events?camera=&label=&after=&before=&limit=50&offset=0` → `{events: [{id, frigate_id, camera, label, count, score, start_time, end_time, has_clip, has_snapshot, zones: [], box: [x1,y1,x2,y2]|[]}], total}`
  - `box` is the best detection box in **detect-stream px** (`[]` for
    doorbell/audio/legacy rows). It backs snapshot annotation; the UI does not
    need to render it.
  - `has_clip` reflects **reality**: it is `false` until the recorder has
    written a non-empty clip file for the event. The engine NEVER sets it
    optimistically at event end — the recorder flips it to `true` only after
    `extract_clip` produces a non-empty `<clips>/{id}.mp4` (extraction failure
    ⇒ stays `false`, WARNING logged with the reason).
- `GET /api/events/{id}` → same row + `{clip_url, snapshot_url, record_enabled, clip_state}`
  - `record_enabled` (bool): whether the event's camera currently records 24/7.
  - `clip_state` ∈ `ready | processing | recording_disabled | unavailable`,
    derived so the UI can tell "the clip is still coming" from "it is never
    coming":
    - `ready` — `has_clip` (the clip file exists);
    - `processing` — recording on, event ended < 45 s ago, clip not written yet;
    - `recording_disabled` — the camera isn't recording (also synthetic
      `doorbell.`/`audio.` events, which never produce a clip);
    - `unavailable` — recording on but the clip never landed (recorder was down
      for the window / extraction failed).
  - The `GET /api/events/{id}/clip.mp4` 404 `detail` mirrors `clip_state`
    (`"Clip is still being prepared"` / `"Recording is disabled for this
    camera"` / `"Clip not available"`) so the UI message is accurate.
- `GET /api/events/{id}/snapshot.jpg` — **Supervision-annotated** snapshot (boxes,
  labels, count banner) from `/data/snapshots/`; before the annotated copy lands it
  serves the engine's clean best frame.
- `GET /api/events/{id}/clip.mp4` — the recorder's clip file
  (`/media/native/clips/{id}.mp4`), served with Range support (seeking works). 404 until
  clip assembly finishes (~20 s after event end) or when the recorder missed the window.
  Clips are already H.264 (see *Recording + clips*), so they play in the browser.
- **Download vs. inline** — both media routes accept **`?download=1`** (default
  inline). With it set, the response carries
  `Content-Disposition: attachment; filename="<camera>_<label>_<YYYY-MM-DD_HH-MM-SS>.<ext>"`
  (`.mp4` for the clip, `.jpg` for the snapshot; the timestamp is the event's
  local start time). The filename parts are sanitized to `[A-Za-z0-9._-]`
  (unsafe runs collapse to `_`), so the header is injection-safe. Media-scope
  auth (`?token=`) and Range are unchanged either way.
- `DELETE /api/events/{id}` — also unlinks the snapshot + clip files. **admin only**.
  (The old `POST /api/events/{id}/reject` reject-to-suppress endpoint was **removed**
  — use **exempt zones** to ignore unwanted detections instead. See *Exempt zones*
  under the camera routes.)

Recordings (24/7 continuous footage — the timeline/scrubber source; the recorder
writes 10 s MPEG-TS segments at `<recordings_dir>/{camera}/{YYYY-MM-DD}/{HH}/{MM.SS}.ts`
in **local** time). All routes are **media-scope** like the other media routes
(accept `?token=` so the browser/OS can fetch playlist + segments without
headers); `{camera}` is resolved strictly to a direct child of the recordings
dir (no path traversal), `{start_ts}` must map to a real segment file:
- `GET /api/recordings/cameras` → `[{camera, friendly_name, has_recordings: bool, earliest: epoch|null, latest: epoch|null}]`
  — one entry per known camera; `earliest`/`latest` are epoch seconds
  (`earliest` = first segment start, `latest` = last segment start + segment
  length), `null` when the camera has no footage.
- `GET /api/recordings/{camera}/index?date=YYYY-MM-DD` (local day) →
  `{date, tz_offset, segments: [{start: epoch, duration: sec}], ranges: [{start: epoch, end: epoch}]}`
  — `segments` are that local day's segments (sorted, `duration` = 10 s
  nominal); `ranges` = merged contiguous coverage (a gap larger than one
  segment length starts a new range). A missing day returns empty
  `segments`/`ranges`; a malformed `date` ⇒ 400.
- `GET /api/recordings/{camera}/playlist.m3u8?start=<epoch>&end=<epoch>` → a valid
  **HLS VOD** playlist (`#EXTM3U`, `#EXT-X-VERSION:3`, `#EXT-X-PLAYLIST-TYPE:VOD`,
  `#EXT-X-TARGETDURATION`, `#EXTINF` + `seg/{start_ts}.ts` per segment,
  `#EXT-X-ENDLIST`) listing the segments intersecting `[start, end]`. Each URI
  is **relative** to the playlist and carries `?token=` through when the
  playlist was fetched with one. The window is **capped server-side to 6 h** to
  bound playlist size; `end ≤ start` ⇒ 400. `Content-Type:
  application/vnd.apple.mpegurl`.
- `GET /api/recordings/{camera}/seg/{start_ts}.ts` → the segment whose start
  epoch is `start_ts`, `Content-Type: video/mp2t`, Range/seek supported
  (FileResponse). 404 when no such segment exists.
- `GET /api/recordings/{camera}/export.mp4?start=<epoch>&end=<epoch>` → the
  continuous footage in `[start, end]` as **one downloadable H.264 faststart
  MP4**. Reuses the recorder/transcoder machinery: the same segment selection
  that cuts event clips, concat + precise cut to the window, and **stream-copy
  for H.264 sources or an NVENC→libx264 transcode for HEVC** (so the export is
  always browser-playable). Served as
  `Content-Disposition: attachment; filename="<camera>_<start-date>_<HH-MM-SS>-<HH-MM-SS>.mp4"`
  (local time, sanitized), `Content-Type: video/mp4`. The window is **capped
  server-side to `EXPORT_MAX_SECONDS` (1800 s / 30 min)** — a longer window ⇒
  **400** with a clear detail; `end ≤ start` ⇒ **400**; **no footage** anywhere
  in the window ⇒ **404**; an ffmpeg build failure ⇒ **503** (never a bare 500;
  temp files are cleaned up). Builds land in a bounded on-disk cache (`<MEDIA>/native/tmp/export-cache`,
  LRU) with in-flight de-duplication, so identical windows share one build and
  the cached file.

Notifications:
- `GET /api/notifications/vapid-public-key` → `{key}` (**no auth**)
- `POST /api/notifications/subscribe` — Web Push subscription JSON; upsert by endpoint
- `POST /api/notifications/unsubscribe` `{endpoint}`
- `POST /api/notifications/test` — sends a test Web Push to every registered subscription.
  Responses: **200** `{push_sent: int}` when at least one delivery succeeded;
  **400** `"No push subscriptions registered — enable notifications on a device first"`
  when there are no subscriptions;
  **502** when subscriptions exist but every send failed (detail names the first push error)

APNs (iOS) push — **pinned contract: [push-architecture.md](push-architecture.md)**
(E2E-encrypted; sent alongside Web Push under the SAME cooldown/label/min_score
gates; DB table `apns_devices`, schema v7). All three routes are **any
authenticated role** — a viewer's phone gets pushes too:
- `POST /api/notifications/apns/register`
  `{device_token, device_name?, key_b64, environment?}` → **204** upsert by
  token (re-register rotates key/name/env; latest wins). Validation (**400**):
  `device_token` `^[0-9a-fA-F]{64,160}$`, stored **lowercased**; `key_b64` must
  base64-decode to exactly **32 bytes**; `environment` ∈ `sandbox | production`
  (default `production`); `device_name` capped at 64 chars.
- `DELETE /api/notifications/apns/register` `{device_token}` → **204**,
  idempotent (204 whether or not the token existed).
- `GET /api/notifications/apns/devices` →
  `[{device_token_prefix, device_name, created_at}]` — first 8 hex chars only,
  the full token capability is never returned.
- Send path: per device, the notification JSON
  `{title, body, event_id, snapshot_url}` (snapshot_url = the same tokened
  media URL as the web-push image) is AES-256-GCM-encrypted with the
  registration key (`base64(nonce||ct||tag)`) and delivered per
  `settings.notifications.apns.mode` — `relay` (POST to `{relay_url}/api/push`,
  forwarding the device's stored `environment` so the relay picks the sandbox vs
  production APNs host) or `off` (default). `direct` is RETIRED; ntfy (below) is
  how a hoster pushes without any Apple credentials, though only the relay can
  ring the doorbell. A **410** (`unregistered`) or **400** `bad_device_token`
  response **deletes the registration row**. Sends never raise into the
  pipeline; logs carry 8-char token prefixes only.
- ntfy send path: the SAME gates and the same tokened media URL, published as
  `POST {server}/{topic}` with the snapshot LINKED via the `Attach` header (the
  phone fetches it from this NVR — the image never reaches ntfy). Spawned, so a
  slow ntfy never stalls the caller. See push-architecture.md §7.

Settings:
- `GET /api/settings` / `PUT /api/settings` / `PATCH /api/settings` →
```json
{
  "notifications": {
    "enabled": true,
    "labels": ["person", "dog", "cat", "car"],
    "cooldown_seconds": 60,
    "min_score": 0.7,
    "draw_boxes": true,
    "apns": {"mode": "off", "relay_url": ""},
    "ntfy": {
      "enabled": false, "server": "https://ntfy.sh", "topic": "",
      "auth_token": "", "priority": 4, "attach_snapshot": true
    }
  },
  "recording": {"continuous_days": 7, "event_days": 14, "snapshot_days": 14},
  "detection": {"model": "dfine_s", "confidence": 0.5, "default_mode": "always",
                "backend": "gpu"},
  "system": {"public_url": "", "webrtc_candidates": []},
  "mqtt": {
    "enabled": false, "host": "", "port": 1883,
    "username": "", "password": "",
    "discovery_prefix": "homeassistant", "base_topic": "sentinel"
  }
}
```
  - **PUT is a FULL REPLACE; PATCH deep-merges.** Every field of the PUT model
    has a default, so a PUT that omits a block **silently resets that block to
    its default** — it does not leave it alone. This has destroyed a live secret:
    an iOS client PUT-ing one unrelated toggle wiped the stored APNs key, because
    it never sent one. **Any partial update MUST use `PATCH`** (iOS and the web
    settings pages do). PUT is for a client that GET-ed the whole document,
    edited it, and is sending all of it back. Both validate identically and both
    run the legacy migrations below.
  - `recording` drives Vigilume's own recorder retention + pruning (see Native engine).
  - `detection.model` ∈ `dfine_n | dfine_s | dfine_m | dfine_l | dfine_x | dfine_l_obj365`;
    a change triggers model download (if absent) + engine reload. `detection.confidence`
    ∈ 0.2–0.9 (decode threshold). `dfine_n…dfine_x` are COCO-80; `dfine_l_obj365` swaps
    the output vocabulary to Objects365 (365 categories) — see **Class vocabulary** below.
  - `detection.default_mode` ∈ `always | camera_ai | camera_ai_only`: the **effective
    server-detection mode for cameras whose per-camera `detect_mode` is unset/NULL** (see
    **Camera-AI-gated detection**). A change re-gates ingest and starts/stops camera-AI
    watchers live (no restart). Cameras with an explicit `detect_mode` are unaffected.
  - `system.webrtc_candidates`: up to 16 `ip:8555` strings (≤64 chars each) appended to
    go2rtc's ICE candidates; a change regenerates + syncs the go2rtc config.
  - `public_url` is the externally reachable base URL used in push payload click-links.
  - `notifications.draw_boxes` (default `true`): draw detection boxes + labels on
    event snapshots. `false` ⇒ `annotate_event_snapshot` skips box/label drawing but
    keeps the count banner. **Legacy-safe:** a stored settings blob without the key
    behaves as `true`.
  - `notifications.apns` configures APNs (iOS) push per the pinned
    [push-architecture.md](push-architecture.md): `mode` ∈ `relay | off`
    (default `off`), `relay_url` (≤256 chars) — required when `mode == "relay"`.
    The relay is how a native notification + a **CallKit doorbell ring** reach
    the iOS app from any self-hosted server; `notifications.ntfy` is the
    no-Apple-account channel beside it, not a replacement (its alerts land in
    the ntfy app: no ring, no native UI). If you run the bundled `push-relay`
    yourself, `relay_url` is `http://push-relay:8090` — the Docker-internal
    name, so push never depends on your tunnel, DNS, or internet.
    **`mode: "direct"` (this server holding its own Apple `.p8`) is RETIRED**,
    and with it the `direct.{key_id, team_id, bundle_id, p8}` block. A stored
    `mode: "direct"` migrates to **`off`** — never to `relay`, which with an
    empty `relay_url` would error on every event — on load and on save, and
    `direct` is dropped. That migration is required, not cosmetic: `mode` is a
    pydantic `Literal`, so an unmigrated blob would 422 every settings write and
    lock the admin out of the settings page, including out of fixing the mode.
  - `notifications.ntfy` configures the **ntfy** channel — push with no Apple
    developer account, for self-hosters (see push-architecture.md §7):
    `enabled` (default `false`), `server` (http(s), default `https://ntfy.sh`),
    `topic` (`^[A-Za-z0-9_-]{1,64}$`, **default empty**), `auth_token`
    (`Authorization: Bearer`), `priority` (1..5, default 4), `attach_snapshot`
    (default `true` — links the event snapshot via ntfy's `Attach` header so the
    PHONE fetches it from this NVR; the image never touches the ntfy server).
    **The topic is a shared secret**: on a default-allow server (ntfy.sh
    included) anyone who knows it receives every message, so it defaults to
    empty and the UI generates an unguessable one. Never logged in full.
  - `mqtt` configures the **outbound MQTT + Home Assistant auto-discovery publisher**
    (see **MQTT / Home Assistant** below). Admin-only (whole router is `require_admin`);
    changing any `mqtt` field **restarts the publisher live** (reconnect, no app
    restart). `discovery_prefix`/`base_topic` are validated as single topic segments
    (no `+`/`#`/spaces, slashes stripped). The stored password is returned by
    `GET /api/settings` (admin-only).
  - Legacy blocks persisted by older versions — `detection.audio_events`/
    `audio_labels` (the removed Frigate audio classifier), `notifications.apns.direct`
    (the retired own-Apple-key mode), and `time_sync.auto_ntp`/`ntp_server` (the
    camera clocks are now pushed directly, with the camera NTP client disabled)
    — are **silently dropped or migrated** on load and on write; an old `/data`
    volume must never 500. (`notifications.apns.relay_url` and
    `notifications.ntfy` were each dropped this way during the day both features
    were briefly retired. Both are supported and round-trip again — do not
    re-add a `pop()` for either without also deleting its settings model.)

Detection models (in-app tier download + activate — the ModelStore is the single
downloader, `backend/app/native/model_store.py`, built on the pin table above):

Tiers map to the model keys — a static metadata table in `model_store.py`
(`MODEL_TIERS`) is the single source of truth. Each tier advertises its class
`vocabulary` (`"COCO (80)"` vs `"Objects365 (365)"`) and `labels_count` so the
picker can show what a model detects. `approx_map` is **COCO** AP(val) for the
COCO tiers and **Objects365** AP(val) for the obj365 tier (`map_dataset` names
which — the two benchmarks are not comparable):

| tier | key | one-line tradeoff | vocabulary | approx mAP |
|---|---|---|---|---|
| Lightweight | `dfine_n` | Fastest, lowest latency — quick load, weak GPUs / CPU fallback | COCO (80) | 42.8 (COCO) |
| Balanced (**default**) | `dfine_s` | Best speed/accuracy tradeoff for most setups | COCO (80) | 48.5 (COCO) |
| Heavy | `dfine_m` | Higher accuracy, higher latency — GPUs with headroom | COCO (80) | 52.3 (COCO) |
| Accurate | `dfine_l` | High accuracy — a solid GPU with headroom | COCO (80) | 54.0 (COCO) |
| Maximum | `dfine_x` | Most accurate COCO tier, heaviest — GPU only | COCO (80) | 55.8 (COCO) |
| Big Vocabulary | `dfine_l_obj365` | Detects 365 object types (far beyond COCO's 80); ~`dfine_l` speed | Objects365 (365) | 44.7 (Objects365) |

Per-key download **state machine** (owned by the ModelStore, one source of truth for
both the detector and the API): `"absent" → "downloading"` (with `progress_pct` 0–100)
`→ "verifying" → "ready"`, or `→ "error"` (with a short `detail`; a SHA-256 mismatch
deletes the bad file). Downloads are **non-blocking** (background task), **idempotent**
(a second download while one is in-flight is a no-op), and a failed/partial file is
cleaned and retryable. The detector's boot + reconfigure go through the store (ONE
downloader); a model swap adopts the new model in the background once its download +
session build complete, so activate/PUT return immediately and boot never blocks.

- `GET /api/detection/models` → `{active, device, models: [...]}`:

```json
{
  "active": "dfine_s",
  "device": "cuda",
  "models": [
    {
      "key": "dfine_n", "tier": "lightweight", "label": "Lightweight",
      "blurb": "Fastest, lowest latency — quick to load, good for weak GPUs.",
      "size_bytes": 15258358, "input_size": 640,
      "approx_map": 42.8, "map_dataset": "COCO",
      "recommended_for": "Weak GPUs, CPU fallback, fastest startup",
      "vocabulary": "coco", "num_classes": 80,
      "state": "ready", "progress_pct": 100,
      "active": false, "loaded": false, "sha_ok": true, "detail": null
    }
  ]
}
```

  `models` is ordered fastest-COCO → heaviest-COCO → big-vocabulary (`dfine_n`, `dfine_s`,
  `dfine_m`, `dfine_l`, `dfine_x`, `dfine_l_obj365`). Each entry carries `vocabulary`
  (short machine name `"coco"`/`"objects365"`), `num_classes` (80/365 selectable classes),
  and `map_dataset` (which benchmark `approx_map` is on). `active` = `settings.detection.model`;
  `loaded` = the model currently loaded in the detector **and** ready (lets the UI
  distinguish the active setting from what's actually running). `device` is the
  detector's `"cuda" | "cpu" | null`.
- `GET /api/detection/labels[?model={key}]` → `{model, vocabulary, count, labels: [...]}`
  — the ordered, user-selectable class vocabulary for a model, defaulting to the **active**
  model. Lets the per-camera object picker list exactly what the running model can detect
  (COCO-80, or the 365 real Objects365 classes — the model's id-0 `"none"` background
  placeholder is dropped from the list). `vocabulary` is the short machine name
  (`"coco"`/`"objects365"`); `count == labels.length`. Admin-gated (all `/api/detection/*`
  routes are). **404** unknown key.
- `POST /api/detection/models/{key}/download` → **202** `{key, state, progress_pct}` —
  starts (or no-ops) a background download. **404** unknown key.
- `POST /api/detection/models/{key}/activate` → **202** `{key, state, active: true,
  loaded}` — persists `settings.detection.model = {key}` and reconfigures the detector
  (starting the download in the background if the model is absent; the detector adopts it
  when ready). This is the SAME activate path `PUT /api/settings` model changes route
  through — both stay consistent. **404** unknown key.
- `DELETE /api/detection/models/{key}` → `{key, state: "absent"}` — deletes the
  downloaded file + sidecar, freeing disk. **409** if `{key}` is the active model;
  **404** unknown key.
- Progress delivery: a `{type:"model_status", key, tier, state, progress_pct, active,
  loaded}` frame is broadcast on the WS (below) on every state/progress change (progress
  throttled to ~1/sec). GET stays authoritative for a polling fallback.

Learned detection suppressions (reject-to-suppress): **removed.** The
`GET /api/detection/suppressions`, `DELETE /api/detection/suppressions/{id}`, and
`GET /api/detection/suppressions/{id}/thumb.jpg` routes no longer exist, and the
engine no longer drops detections by learned samples. The `detection_suppressions`
table stays in the schema but **dormant** (no code reads or writes it). To ignore
unwanted detections, draw **exempt zones** on the camera snapshot (see the
`exempt_zones` camera field) — a foot-center in a drawn polygon is masked before
any event/count/notification.

WebSocket:
- `WS /api/ws?token=` — server pushes `{type:"event_new"|"event_update"|"event_end"|"doorbell", event:{...}}`, `{type:"camera_status", ...}`, and `{type:"model_status", key, tier, state, progress_pct, active, loaded}` for live UI updates.

## Event & notification pipeline (backend, in-process)

1. The native engine calls `EventsPipeline.handle_event()` with Frigate-shaped payloads
   (see Native engine). On `type:"new"` (and meaningful `update`s): store row, take the
   engine's best detect-res frame, read the in-frame count from the live count cache
   (fed exactly by the engine's `update_count()` calls — counts are always current),
   annotate with **Supervision** (`sv.Detections` from the event's box +
   `sv.BoxAnnotator` + `sv.LabelAnnotator`, plus a count banner like "2 people"), save
   to `/data/snapshots/{id}.jpg`.
2. Notify (respecting per-(camera,label) cooldown, min_score, enabled labels — ONE
   decision, two transports):
   - **Web Push** (pywebpush, VAPID): title `"Person detected at {friendly_name}"`, body
     `"{count} in frame"` (pluralized label), `image` = annotated snapshot URL,
     `data.url` = `{public_url}/events/{id}`.
   - **APNs (iOS)** (`notify/apns.py`, [push-architecture.md](push-architecture.md)):
     the same title/body/event, E2E-encrypted per registered device with its 32-byte
     key, `snapshot_url` = the SAME tokened media URL as the web-push `image`,
     `collapse_id` = event id, priority `high`. Relayed or direct per
     `settings.notifications.apns.mode`; never raises into the pipeline, and the
     send runs on its own pipeline task so a hung/down relay (per-request
     timeouts + bounded retries, ~30 s worst case) never delays web push, the
     doorbell watcher, or event enrichment.
3. On `type:"end"`: update row (end_time, has_clip); the recorder cuts the clip ~20 s
   later.
4. AD410 doorbell button: backend maintains a long-poll attach to
   `http://<ad410>/cgi-bin/eventManager.cgi?action=attach&codes=[All]` (digest auth).
   Button-press event codes (`Invite` / `CallNoAnswered` / `_DoTalkAction_`) → push
   `"Doorbell pressed at {friendly_name}"` with snapshot, bypassing label filters.

## MQTT / Home Assistant (outbound publisher — optional, off by default)

`backend/app/integrations/mqtt_ha.py` publishes to the operator's MQTT broker with
Home Assistant **auto-discovery** so cameras/detections appear as HA entities with no
manual YAML, plus optional two-way control. Publish-only; **not** a dependency. Enabled
via `settings.mqtt` (above). Full operator guide: [home-assistant.md](home-assistant.md).

- **Resilience:** `aiomqtt` is imported **lazily** (like onnxruntime) — importing
  `app.main` needs neither aiomqtt nor a broker. A single background task owns one
  connection with auto-reconnect + capped backoff; a down/unreachable broker is logged
  and retried and **never** crashes the app. Wired into the lifespan like the other
  background tasks; `PUT /api/settings` restarts it on any `mqtt` change.
- **Availability:** Last-Will on `<base_topic>/status` = `"offline"` (retained); a
  retained `"online"` birth is published on connect. All entity **state** topics are
  retained.
- **Discovery:** retained configs under
  `<discovery_prefix>/<component>/<base_topic>_<camera>_<slug>/config` with **stable
  `unique_id`s**. **One HA device per camera**, all grouped under a "Vigilume NVR"
  bridge device via `via_device`. Per camera:
  - a **`binary_sensor` per tracked label** (from `detect_objects`), `device_class`
    `occupancy` (person/cat/dog) or `motion` (else), ON while in-frame (engine
    count / event new‥end) → OFF on clear. State: `<base>/<camera>/<label>/state`.
  - a **`binary_sensor` "Connectivity"** (`device_class: connectivity`) driven by the
    `CameraProber` online/offline broadcast. State: `<base>/<camera>/connectivity/state`.
  - a **`sensor` "Last event"** — state = last label; JSON attributes
    `{count, score, started, ended, snapshot_url}` (`snapshot_url` = `<public_url>/api/
    events/{id}/snapshot.jpg?token=<media>`). Attributes topic
    `<base>/<camera>/last_event/attributes`.
  - an **`image` "Last snapshot"** — only when `public_url` is set; fed the annotated
    snapshot **URL** via `url_topic` (`<base>/<camera>/image/url`), not raw bytes
    (guards broker size).
- **Two-way control (capability-gated via `amcrest/features`):** for cameras whose caps
  include them — a `switch` for **IR**, a `switch` for **spotlight** (`white_light`), a
  `button` for **siren**. Command topics `<base>/<camera>/{ir|spotlight|siren}/set` are
  subscribed; a command calls the **same** control paths `routers/cameras` uses
  (`set_ir_mode` / `set_white_light` / `play_tone`) and echoes switch state to
  `<base>/<camera>/{ir|spotlight}/state`. Best-effort: a device error is logged and
  never drops the MQTT link.
- **Test route:** `POST /api/integrations/mqtt/test` (admin-only) — attempts a
  connect+publish with the CURRENT saved settings, or with `{mqtt: {...}}` in the body
  (same validation as PUT) to test before saving. Returns `{ok, detail}` (connected /
  authentication failed / unreachable). Never mutates state; the live publisher is
  untouched.

## Amcrest RTSP URLs

- Main: `rtsp://{user}:{pass}@{ip}:554/cam/realmonitor?channel=1&subtype=0`
- Sub:  `rtsp://{user}:{pass}@{ip}:554/cam/realmonitor?channel=1&subtype=1`

(All three models, including AD410, support this. AD410 must be on current firmware;
RTSP creds = device creds set in the Amcrest Smart Home app. Credentials are
percent-encoded into generated URLs. Per-camera `main_url`/`sub_url` overrides replace
these wholesale when set.)

## Per-model capability map (backend `amcrest/features.py`)

| Capability        | IP5M-T1277EW-AI | IP8M-2779EW-AI | AD410 | IP3M-941B | IP4M-1041B | IP4M-1056E |
|-------------------|-----------------|----------------|-------|-----------|------------|------------|
| IR night vision   | ✅              | ✅             | ✅    | ✅        | ✅         | ✅         |
| White light LED   | ✅ (spotlight via `Lighting_V2`) | ✅ (spotlight via `Lighting_V2`) | ❌ (status LED only) | ❌ | ❌ | ❌ (auto white LED via `night_vision` day/night, not an on-demand spotlight) |
| Siren             | ❌              | ❌             | ✅ (generated alarm tone via `audio.cgi` `postAudio`; 501 if firmware rejects) | ❌ | ❌ | ❌ |
| Microphone (audio in RTSP) | ✅     | ✅             | ✅    | ✅        | ✅         | ❌         |
| Speaker (two-way) | ❌              | ❌             | ✅    | ✅ (via `backchannel`) | ✅ (via `backchannel`) | ❌ |
| Doorbell button   | ❌              | ❌             | ✅    | ❌        | ❌         | ❌         |
| On-camera AI      | ✅ (unused — Vigilume detects) | ✅ (unused) | ✅ (unused) | ❌ | ❌ | ❌ |
| `backchannel` (RTSP two-way talk) | ❌ | ❌ | ✅ | ✅ (codec pinned G.711A) | ✅ (codec pinned G.711A) | ❌ |
| `ptz` (pan/tilt + 3 presets) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `night_vision` (full-colour) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

(`ptz` gates `POST /api/cameras/{name}/ptz`; `night_vision` gates the `night_vision_mode` settings field. `backchannel`-capable cameras route talk to the RTSP audio backchannel rather than the `audio.cgi` `postAudio` CGI.)

Backend should ALSO probe the device (`magicBox.cgi?action=getDeviceType`, Lighting config presence)
at registration and merge probed capabilities over this static map, so future models work.

Amcrest control = Dahua-compatible CGI over HTTP digest auth (use `httpx` with `DigestAuth`; the
`python-amcrest` library is sync — prefer direct async CGI calls):
- IR mode: `configManager.cgi?action=setConfig&Lighting[0][0].Mode={Auto|Manual|Off}`
- White light / spotlight (EW turrets): `configManager.cgi?action=setConfig&Lighting_V2[0][0][1].Mode={Off|Manual|Auto}` and `…&Lighting_V2[0][0][1].MiddleLight[0].Light={0-100}`; read via `getConfig&name=Lighting_V2`. Slot `[0]` is the IR illuminator, slot `[1]` the white LED on dual-illuminator hardware (verified against the rroller/dahua HA integration's flood-light/doorbell functions).
- Speaker audio (siren tone + two-way talk): `audio.cgi?action=postAudio&httptype=singlepart&channel=1` with `Content-Type: Audio/G.711A` and a pinned large Content-Length for open-ended streams (verified against python-amcrest `audio.py`)
- Night-vision mode (IP4M-1056E full-colour, `night_vision` cap): `configManager.cgi?action=setConfig&VideoInDayNight[0][N].Mode={Auto|Color|BlackWhite}` written to EVERY exposed profile index `N` (the camera obeys whichever day/night profile is active, so writing only `[0][0]` is silently ignored — same reason `set_ir` writes all `Lighting` profiles); read via `getConfig&name=VideoInDayNight`
- PTZ (IP3M-941B dome, `ptz` cap): `ptz.cgi?action={start|stop}&channel=1&code=<Up|Down|Left|Right|LeftUp|RightUp|LeftDown|RightDown|SetPreset|GotoPreset|ClearPreset>&arg1=0&arg2=<speed 1-8 | preset index 1-3>&arg3=0`
- Flip/mirror, OSD channel title, motion detect enable, snap: standard Dahua CGI endpoints
- The camera's own lens mask (`LeLensMask`) is NOT exposed as a control. It was replaced by
  software Privacy Mode (`/api/privacy`, **admin-only on both verbs**), which gates
  Vigilume's capture instead of
  reconfiguring the device — a device-side mask survives Vigilume and can only be undone from
  the camera's own web UI. The backend clears `LeLensMask` the first time each camera
  is reachable (`amcrest/lens_mask.py`, on the same prober on-connect hook as time-sync)
  so no camera is stranded blind by the removal.
- Reboot: `magicBox.cgi?action=reboot`
- AD410 events: `eventManager.cgi?action=attach&codes=[All]` (multipart stream)

## Frontend (React + TS + Vite PWA)

Pages:
- **Dashboard** — camera grid, live players (go2rtc via `/go2rtc/api/ws?src=<name>` using vendored `video-rtc.js` from go2rtc; WebRTC first, MSE fallback), per-tile status + last event
- **Camera detail** — large live view + capability-gated controls (IR mode, spotlight, siren, talk, reboot, settings), recent events for that camera
- **Events** — filterable timeline (camera/label/date), annotated snapshot thumbnails, click → clip playback (`/api/events/{id}/clip.mp4`)
- **Event detail** (`/events/:id`) — clip player + metadata; this is the push-notification click target
- **Settings** — tabs: Cameras (add/edit/delete incl. detect_fps + RTSP overrides, Amcrest device settings), Groups (dashboard camera groups), Notifications (enable push on this device, labels, cooldown), Recording (retention + disk-usage math, detection model + confidence), System (public URL, WebRTC addresses, health + detector status)
- **Login**

PWA: `vite-plugin-pwa`, installable manifest, service worker handles `push` (showNotification with image) + `notificationclick` (focus/open `data.url`). Dark theme by default, clean and minimal; must look good at 380px width.

## Env vars (.env)

See `.env.example`. Backend seeds its camera DB on first boot from `CAM{1..3}_*` vars if
the DB is empty.

| Var | Default | Meaning |
|-----|---------|---------|
| `ADMIN_PASSWORD` | — (required) | Web UI / API password |
| `TZ` | `America/New_York` | Container timezone (recording paths use local dates) |
| `PUBLIC_URL` | empty | Externally reachable base URL for push click-links (also settable in the UI) |
| `VIGILUME_REQUIRE_GPU` | `1` | `1`: missing CUDA EP ⇒ detector hard-fails (ready:false). `0`: allow CPU inference (dev). Only meaningful for `VIGILUME_DETECTOR=onnx` |
| `VIGILUME_DETECTOR` | `onnx` | Inference backend: `onnx` \| `onnx_cpu` |
| `MEDIA_PATH` | `./media` | Host path for 24/7 recordings + clips (point at the big disk) |
| `COMPOSE_PROFILES` | empty | `tls` to enable the Caddy HTTPS terminator |
| `VIGILUME_GO2RTC_UPSTREAM` | `http://go2rtc:1984` | nginx `/go2rtc/` proxy target (becomes the web container's `GO2RTC_UPSTREAM` env). Deliberately NOT named `GO2RTC_UPSTREAM` at the `.env` level: pre-standalone installs set that name to the external Frigate host, and a stale leftover line must never hijack the live-view proxy |
| `CAM{1..3}_*` | — | First-boot camera seeds (name/friendly/model/ip/user/pass) |

> **Legacy env names:** the `VIGILUME_*` tunables above were named `SENTINEL_*`
> before the rename. Both docker-compose and the backend still read the old
> `SENTINEL_*` name as a fallback (`VIGILUME_X` wins when both are set), so an
> existing `.env` keeps working untouched.

Backend-internal overrides (tests/dev): `DATA_DIR`, `MEDIA_DIR`, `GO2RTC_CONFIG_DIR`,
`GO2RTC_URL`, `GO2RTC_RTSP_URL`.

## Data layout

- `./data` → backend volume `/data`: `nvr.db` (SQLite), `snapshots/`, `models/`
  (downloaded ONNX models + sidecars), `secrets.json` (JWT secret, VAPID keys)
- `./media` (or `MEDIA_PATH`) → backend volume `/media`:
  `native/recordings/{camera}/{date}/{hour}/*.ts` (24/7), `native/clips/{event_id}.mp4`
- `./go2rtc/config` → backend `/go2rtc-config` (writes `go2rtc.yaml`), go2rtc `/config`
  (reads it)

## Non-goals / roadmap (document, don't build)

- **Audio events** (bark / scream / yell): the previous audio classifier was a Frigate
  feature and was removed with it. Legacy `audio.` event rows remain viewable. Roadmap:
  a native audio classifier over the RTSP audio track.
- **Zones** (polygon-gated alerting): native engine v1 tracks whole-frame; `zones` stays
  `[]` in the API.
- **Per-track events**: overlapping arrivals/departures of the same label merge into one
  event with `count` — same UX as before; splitting per track is a future option.
- **TensorRT execution provider / cross-camera batching**: CUDA EP with a single
  inference worker is deliberate at this scale (see design doc §2.3–2.4).
- Native app-store apps (PWA now; Expo wrapper later if wanted).

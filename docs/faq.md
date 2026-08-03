# FAQ & operations

## How do I check that the GPU is actually being used?

Three ways, most specific first:

1. **Settings → System** in the app: the detector card shows
   `ready` / `device: cuda` / the model and the rolling inference time.
   (The same data is at `GET /api/system/detector`.)
2. Backend logs — first boot prints exactly one of:

   ```
   GPU OK — provider=CUDAExecutionProvider model=dfine_s warmup_ms=... infer_ms=...
   GPU UNAVAILABLE — set VIGILUME_REQUIRE_GPU=0 to accept CPU
   ```

   ```bash
   docker logs vigilume-backend | grep -i -e "GPU OK" -e "GPU UNAVAILABLE"
   ```

3. On the NVR host:

   ```bash
   nvidia-smi                            # the backend python process holds GPU memory
   docker exec vigilume-backend nvidia-smi   # GPU visible inside the container
   ```

   Utilization spikes with motion; an idle scene sits near 0% with brief
   bursts, which is normal — detection runs at ~5 fps per camera.

If the detector reports `ready: false` with `device: null`, the CUDA provider
didn't load: re-check [setup-nvidia.md](setup-nvidia.md) (driver + NVIDIA
Container Toolkit + the compose GPU reservation). By default
(`VIGILUME_REQUIRE_GPU=1`) Vigilume refuses to silently fall back to CPU
detection — everything else (UI, live view, recording) keeps working while
you fix it.

## How much storage do I need?

24/7 recording is stream-copy (no transcode), so disk usage is exactly the
**main-stream bitrates** you set on the cameras
([cameras-amcrest.md](cameras-amcrest.md#3-stream-settings-both-turrets)). The math:

```
GB/day per camera ≈ bitrate in Mbps × 10.8
```

With the bitrates recommended in the camera guide:

| Camera | Bitrate | Per day | 7 days |
|---|---|---|---|
| IP8M-2779EW-AI (4K) | 6 Mbps | ~65 GB | ~455 GB |
| IP5M-T1277EW-AI (5MP) | 4 Mbps | ~43 GB | ~300 GB |
| AD410 (2K) | 2 Mbps | ~22 GB | ~150 GB |
| **Total** | | **~130 GB/day** | **~0.9 TB** |

Add a little headroom for event clips and snapshots kept past the continuous
window (defaults: continuous 7 days, events 14, snapshots 14 —
**Settings → Recording**). A 2 TB disk comfortably holds the default
retention; 4 TB gives you roughly two weeks of continuous footage or room for
a fourth camera.

Practical notes:

- Point `MEDIA_PATH` in `.env` at the big disk **before** first start.
- Prefer a disk rated for continuous writes (NAS/surveillance class) — this
  workload writes ~1.5 MB/s per camera around the clock.
- H.265 on the main streams roughly halves these numbers, at the cost of
  clip-playback compatibility
  ([trade-off](cameras-amcrest.md#3-stream-settings-both-turrets)).
- Retention changes apply live — the recorder's hourly retention pass and the
  event pruner read the current settings; nothing restarts.
- Low-disk guard: if the media filesystem drops under 5 GB free, Vigilume
  prunes the oldest recording hours regardless of retention and logs loudly.

## Which detection model should I pick?

**Settings → Recording → Object detection** (`dfine_s` by default). All three
are D-FINE COCO models, downloaded and hash-verified automatically on change:

| Model | Accuracy (COCO mAP) | Size | When |
|---|---|---|---|
| `dfine_n` | 42.8 | 15 MB | CPU-only operation (`VIGILUME_REQUIRE_GPU=0`) or a very weak GPU |
| `dfine_s` (default) | 48.5 | 41 MB | The sweet spot for person/dog/cat/car on any dGPU — leave it here |
| `dfine_m` | 52.3 | 78 MB | GPU headroom to spare and you want the extra accuracy |

The confidence threshold (default 0.5) is next to it: it gates which raw
detections exist *at all*. For notification noise, tune
`notifications.min_score` first (below) — it filters without losing events.

## How do I add a fourth camera?

**Settings → Cameras → Add camera**: name (lowercase/underscores), friendly
name, model, IP, username, password. The backend regenerates the go2rtc
streams and starts detection + recording for it immediately — the new camera
appears on the dashboard in seconds. The `CAM1..3_*` values in `.env` only
seed the database on first boot — after that the UI is the source of truth,
and you are not limited to three cameras.

Prepare the camera first (fixed IP, dedicated user, stream settings):
[cameras-amcrest.md](cameras-amcrest.md). Amcrest/Dahua-family cameras get
full device control; for known models capabilities come from the
[capability map](CONTRACTS.md#per-model-capability-map-backend-amcrestfeaturespy),
and the backend probes the device at registration so other Dahua-compatible
models work with whatever features they report. Non-Amcrest RTSP cameras
work too: set the **Main/Sub stream URL** overrides in the add form
(detection + recording + live view; device controls stay off).

Budget for it: each camera adds a little CPU (substream decode), a slice of
GPU (the detector is shared across cameras) and disk (see storage math
above).

## I'm getting notifications for things that aren't there

Tune in this order (**Settings → Notifications**):

1. **`min_score`** — raise from the default `0.7` to `0.8`–`0.85`. This
   filters which events *notify*; Vigilume still records and stores every
   event, so you lose no footage, just noise.
2. **Labels** — drop labels you don't care about. The notification label list
   is global; to stop tracking an object on one camera entirely (e.g. `car`
   on a camera facing a busy street), edit that camera in
   **Settings → Cameras** and change its detect-objects list — the engine
   then never tracks it there.
3. **Cooldown** — `cooldown_seconds` (default 60) rate-limits repeat pushes
   per camera+label pair.
4. As a last resort, raise the **confidence threshold**
   (**Settings → Recording → Object detection**, default 50%) — this discards
   low-confidence detections before tracking, for all cameras.

**About zones** (only alert when the object is inside a drawn polygon — the
strongest fix for a road at the edge of frame): not built yet; the native
engine v1 tracks the whole frame. It's on the roadmap.

Also worth checking: camera stream quality at night — IR glare, spider webs,
and focus issues are classic false-positive factories. The event snapshots
show exactly what the detector saw, box and score included.

## What happened to Frigate (and MQTT)?

Earlier versions ran on top of [Frigate](https://github.com/blakeblackshear/frigate)
with an MQTT broker in between. Vigilume is now **fully standalone**: its own
D-FINE detector on the GPU, its own ffmpeg recorder, its own go2rtc for live
view, and in-process events (no broker). Fewer containers, no config-file
generation for a third-party NVR, and the same UI, API and notifications.

If you're upgrading from a Frigate-based install:

- **Clean up `.env`:** delete the lines the old modes used — `SENTINEL_MODE`,
  `FRIGATE_URL`, `MQTT_HOST`/`MQTT_PORT`/`MQTT_USER`/`MQTT_PASS`, and
  `GO2RTC_UPSTREAM`. Nothing reads any of them anymore (the live-view proxy
  override is now `VIGILUME_GO2RTC_UPSTREAM` — the legacy `SENTINEL_GO2RTC_UPSTREAM`
  name is still honored — precisely so an old
  `GO2RTC_UPSTREAM=<frigate-host>` line can't point live view at the retired
  Frigate box), so they're harmless — but they're also misleading cruft.
- Event history and snapshots carry over. **Old events' clips were served by
  Frigate** and will 404 once it's gone; new events get native clips.
- 24/7 recording history starts fresh under `media/native/`.
- Cameras whose credentials were never stored (Frigate redacted them on
  import) must be given their RTSP username/password in
  **Settings → Cameras** — the native engine connects to cameras directly.

## What happened to audio events (bark / scream / yell)?

Removed — audio classification was a Frigate feature. Old `audio.` events
remain viewable in history, and doorbell-press notifications are unaffected
(they never depended on it). A native audio classifier over the cameras' RTSP
audio track is on the roadmap; camera audio is still recorded and audible in
clips and live view either way.

## What does "lightweight" actually mean here?

- **One inference engine, on the GPU.** A single D-FINE model shared by all
  cameras, fed by one inference worker. The cameras' on-board AI is unused;
  nothing else runs CV.
- **Detection on substreams.** Object detection samples the low-res sub
  stream (~D1/720p) at 5 fps; the full-res main stream is never decoded.
- **Recording is stream-copy.** The 24/7 recorder remuxes the camera's own
  H.264/H.265 into 10 s segments — no transcoding, ~2 % overhead. Event clips
  are concat + copy of those segments.
- **No duplicate inference for pretty snapshots.** The annotated notification
  images (boxes, labels, count banner) are drawn by Supervision *from the
  detector's own output* — annotation is a drawing operation, not a second
  model pass.
- **One connection per camera stream.** go2rtc restreams each camera once;
  detection, recording, and live view all share it (Amcrest firmware limits
  concurrent RTSP clients, so this matters).
- **No live-view transcoding.** The browser plays the camera's own H.264 via
  WebRTC or MSE (only the G.711 audio is transcoded to AAC).
- **Small footprint elsewhere:** SQLite, one FastAPI process, static PWA from
  nginx, in-process event bus.

## How do I back up and restore?

Everything that matters lives next to `docker-compose.yml`:

| Path | Contents | Size |
|---|---|---|
| `./data` | SQLite DB (cameras, events, settings, push subscriptions), annotated snapshots, downloaded models, `secrets.json` (JWT secret + VAPID keys) | small (models add ~15–80 MB) |
| `./go2rtc/config` | generated go2rtc config — **regenerated at every boot**, no need to back up | tiny |
| `./media` (or `MEDIA_PATH`) | 24/7 recordings + event clips | huge |

**Backup** (config + history, excluding video):

```bash
docker compose stop backend        # quiesce the SQLite DB
tar czf vigilume-backup-$(date +%F).tgz .env data
docker compose start backend
```

Back up `./media` too if you want the footage itself — rsync to a NAS is the
usual move; it's append-mostly so incremental syncs are cheap.

**Restore:** on a fresh host, restore the tarball into the project directory,
put media back at `MEDIA_PATH` (or accept losing old footage), then
`docker compose up -d --build`. Models re-download automatically if you
excluded `data/models/` from the backup.

Do not lose `data/secrets.json`: the VAPID keys live there, and new keys
invalidate every phone's push subscription (each device would need to
re-enable notifications). The JWT secret is there too — losing it just forces
everyone to log in again.

## How do I upgrade?

1. Back up first (above — the small tarball is enough).
2. Pull/update the code, then:

```bash
docker compose pull            # go2rtc image
docker compose build --pull    # rebuild backend + web images
docker compose up -d
```

3. Sanity-check: dashboard shows all cameras live, **Settings → System**
   health is green (detector ready on `cuda`), and a test notification
   arrives.

The moving parts are pinned (go2rtc image, onnxruntime, model artifacts by
SHA-256), so upgrades are deliberate code updates, not surprise image drift.
If an upgrade goes sideways: `docker compose down`, restore the backup, and
`docker compose up -d --build`.

## Can I talk through the AD410 from the app?

Yes — press-and-hold the **Hold to talk** button on the doorbell's camera
page. It needs HTTPS (browsers only allow microphone access in a secure
context — see [remote-access.md](remote-access.md)) and has
walkie-talkie-style latency (~0.5–2 s), so hold to speak, release to listen.
The Amcrest Smart Home app still does full phone-style calls and coexists
fine with Vigilume.
Details: [cameras-amcrest.md](cameras-amcrest.md#ad410-2k-video-doorbell).

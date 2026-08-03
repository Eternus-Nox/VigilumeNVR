# Recordings, clips & the timeline

Vigilume is a full NVR: its own recorder keeps **24/7 continuous footage** per
camera, cuts a **clip** for every detected event, and exposes a per-camera
**timeline** you scrub through to review any moment. This page covers what those
are, where the files live, how to read the recorder's logs, and how to tell
"the clip is still coming" from "the clip is never coming".

The binding shapes (recorder paths, recordings API, `clip_state`) are in
[CONTRACTS.md](CONTRACTS.md) — see *Native engine → Recording + clips* and the
*Recordings* / *Events* API sections. Storage sizing is in
[faq.md](faq.md#how-much-storage-do-i-need).

## Timeline

The timeline lets you **scrub each camera's continuous 24/7 footage** and jump
straight to any point in time.

- **Per camera.** The recorder keeps a separate continuous archive per camera,
  so the timeline is built one camera at a time — pick a camera and a local day.
- **Coverage & gaps.** The day index reports which stretches of the day actually
  have footage (contiguous segments merged into coverage *ranges*), so the
  scrubber can draw filled vs. empty spans. A camera that was offline, or whose
  **Record** toggle was off, leaves a visible gap.
- **Event markers.** Your detected events (person / dog / car / …) mark points on
  the timeline; jump to a marker to review that moment or open its
  [clip](#recordings--clips).
- **HLS under the hood.** Playback streams as **HLS VOD**: for whatever window
  you're viewing, the backend builds a standard `.m3u8` playlist of the underlying
  10-second segments and serves each segment with HTTP Range. The player pulls
  only the segments it needs and seeks land on the nearest one, so scrubbing stays
  responsive over long windows (the playlist is capped at 6 h per request to keep
  it small).

### The recordings API (timeline / scrubber source)

The timeline is served by the recordings API. These routes are **media-scope**
(they accept `?token=` so the browser/OS can fetch the playlist and segments
without auth headers); the camera path segment is resolved strictly inside the
recordings dir (no traversal). Full request/response shapes are in the
*Recordings* section of [CONTRACTS.md](CONTRACTS.md).

| Route | Purpose |
|---|---|
| `GET /api/recordings/cameras` | Which cameras have footage, and each one's `earliest`/`latest` epoch bounds |
| `GET /api/recordings/{camera}/index?date=YYYY-MM-DD` | One local day's `segments` + merged coverage `ranges` (drives the scrubber's filled/empty spans) |
| `GET /api/recordings/{camera}/playlist.m3u8?start=<epoch>&end=<epoch>` | HLS VOD playlist over the window (capped server-side to 6 h) |
| `GET /api/recordings/{camera}/seg/{start_ts}.ts` | One 10 s segment, Range/seek-capable |
| `GET /api/recordings/{camera}/export.mp4?start=<epoch>&end=<epoch>` | Download the footage in the window as one H.264 MP4 (capped to 30 min) |

Event markers come from the events list —
`GET /api/events?camera=&after=&before=` — overlaid on the same time axis.

### Exporting a slice of the timeline

Pick a **start/end range** on a camera's timeline and download it as a single
file: `GET /api/recordings/{camera}/export.mp4?start=<epoch>&end=<epoch>`
returns that span as one **browser-playable H.264 faststart MP4**, served as a
download (`Content-Disposition: attachment` with a
`<camera>_<start-date>_<HH-MM-SS>-<HH-MM-SS>.mp4` filename).

It reuses the exact same machinery as an event clip — there is no second video
pipeline:

- **Same segment selection + cut.** The recorder picks the 10 s segments
  intersecting the window (the same `select_segments` that cuts event clips),
  concatenates them and cuts precisely to `[start, end]`.
- **Same H.264 guarantee (GPU transcode for HEVC).** H.264 cameras stream-copy
  (fast, no re-encode). HEVC cameras are transcoded to H.264 using the **same
  NVIDIA GPU path** as timeline segments and clips (`h264_nvenc`, falling back to
  CPU `libx264`), so the export always plays in a browser — see
  [Browser playback](#browser-playback-automatic-h264-transcoding-for-hevc-cameras).
- **30-minute cap.** The window is capped server-side at **`EXPORT_MAX_SECONDS`
  = 1800 s (30 min)** so one request can't run ffmpeg (and a transcode) for an
  unbounded time or produce a huge file. A longer window ⇒ `400`; an inverted
  window (`end ≤ start`) ⇒ `400`; **no footage** in the window ⇒ `404`; an
  ffmpeg failure ⇒ `5xx` with a message (never a hung request or a bare 500).
- **Cached + de-duplicated.** Finished exports live in a bounded on-disk LRU
  (`<MEDIA_PATH>/native/tmp/export-cache`); requesting the same window twice
  serves the cached file, and two identical in-flight exports share one build.

Because it's a normal media route it accepts `?token=` (so the browser can start
the download without auth headers), same as the playlist/segment routes.

## Recordings & clips

### Recording requires the camera's Record toggle

24/7 recording runs **only for cameras with Record enabled**
(**Settings → Cameras →** the camera's *Record* toggle; `record.enabled` in the
API). Turn it off and that camera produces **no segments and no event clips** —
its timeline shows a gap, and its events report `clip_state: recording_disabled`.
Detection and live view are independent toggles, so a camera can detect without
recording (you'll get notifications and snapshots but no clips).

### Where footage lives

One ffmpeg child per record-enabled camera stream-copies (no transcode) the
camera's **main** stream into **10-second MPEG-TS segments**, and event clips are
concatenated out of those same segments:

```
<MEDIA_PATH>/native/recordings/{camera}/{YYYY-MM-DD}/{HH}/{MM.SS}.ts   # 24/7 segments (local time)
<MEDIA_PATH>/native/clips/{event_id}.mp4                               # one clip per event
```

`MEDIA_PATH` is the media disk you set in `.env`; inside the backend container
these live under `/media/native/…`.

### Browser playback: automatic H.264 transcoding for HEVC cameras

Browsers (Chrome/Firefox) **cannot decode H.265/HEVC** via HLS/MSE or `<video>`.
If a camera's main stream is HEVC, the raw segments and clips are HEVC too, so
the timeline would fail every seek ("playback error, try another time") and event
clips wouldn't play. (Live view is unaffected — go2rtc handles it separately.)

Vigilume fixes this transparently: **when — and only when — the source is HEVC
it serves H.264 to the browser**, transcoding on the fly.

- **Recordings stay HEVC on disk.** Transcoding happens at *serve* time, so the
  archive keeps HEVC's ~50 % storage saving. Nothing on disk is rewritten.
- **Timeline segments** are transcoded per 10 s segment on request and cached to
  a bounded on-disk LRU (under `<MEDIA_PATH>/native/tmp/transcode-cache`), so
  re-seeks and re-buffering are instant and playback stays smooth. Concurrent
  requests for the same segment share one transcode. H.264 cameras skip all of
  this and serve the raw segment (fast stream-copy path, unchanged).
- **Event clips** are transcoded **once**, at extraction time, so
  `GET /api/events/{id}/clip.mp4` serves a browser-playable faststart H.264 MP4.
  H.264 cameras keep the fast concat + stream-copy path.
- **GPU-accelerated with a CPU fallback.** Transcoding uses the NVIDIA GPU
  (`h264_nvenc`, the same GPU that runs detection) when available, falling back
  to CPU (`libx264`) otherwise. The container already exposes NVENC via
  `NVIDIA_DRIVER_CAPABILITIES=…,video` (set in `docker-compose.yml`).
- **Never fails hard.** If a transcode fails, Vigilume serves the original
  segment/clip and logs a `WARNING` (no 500). If NVENC init fails at runtime it
  falls back to libx264 and logs once.

You don't have to configure anything — H.264 and H.265 main streams both work
(see [cameras-amcrest.md](cameras-amcrest.md#3-stream-settings-both-turrets)). To
confirm on a GPU box which encoder is in use, watch the recorder logs:

```bash
docker compose logs -f backend | grep 'transcode:'
```

- `transcode: selected H.264 encoder h264_nvenc (GPU NVENC)` — the GPU path is active
- `transcode: h264_nvenc segment camera=… (hevc->h264)` / `transcode: h264_nvenc clip event=… (hevc->h264)` — a transcode ran on the GPU
- `transcode: selected H.264 encoder libx264 (CPU libx264)` or a
  `h264_nvenc failed at runtime — falling back to CPU libx264` line — CPU fallback

To double-check NVENC is really doing the work, run `nvidia-smi dmon` (or
`nvidia-smi -q -d UTILIZATION`) while scrubbing an HEVC camera's timeline — the
**enc** column should tick up.

### Clip lifecycle — why a fresh event briefly shows "processing"

Clips are **cut ~20 seconds after an event ends** — the recorder waits for the
MPEG-TS segment covering the end of the event to close, then concat +
stream-copies the segments spanning `[start − 5 s, end + 5 s]` into the clip file
and only then marks the event `has_clip`.

So a **just-ended event has no clip yet**. During that gap its `clip_state` is
`processing` (the clip endpoint returns `404 "Clip is still being prepared"`) and
the event view shows the annotated snapshot; ~20–30 s later the clip lands and
becomes playable. `has_clip` reflects **reality** — it is `false` until a
non-empty clip file exists and is never set optimistically, so "still coming" and
"never coming" stay distinguishable:

| `clip_state` | Meaning | What you see |
|---|---|---|
| `ready` | The clip file exists | Playable clip |
| `processing` | Recording on, event ended < 45 s ago, clip not written yet | Annotated snapshot, clip arriving shortly |
| `recording_disabled` | The camera isn't recording (also doorbell / audio events, which never get a clip) | Snapshot only, no clip is coming |
| `unavailable` | Recording was on but the clip never landed (recorder down for the window / extraction failed) | Player shows *"Clip unavailable"* |

`clip_state` is on `GET /api/events/{id}`, and the clip route's 404 `detail`
mirrors it so the message is always accurate.

**Downloading a clip or snapshot.** Both event media routes take `?download=1`
(default is inline): `GET /api/events/{id}/clip.mp4?download=1` and
`.../snapshot.jpg?download=1` add a `Content-Disposition: attachment` header
with a friendly `<camera>_<label>_<timestamp>` filename so the browser saves the
file instead of playing/showing it. Clips are already H.264, so the downloaded
file plays anywhere. To grab an arbitrary span rather than one event's clip, use
[timeline export](#exporting-a-slice-of-the-timeline).

### Reading the recorder logs (prefix `recorder:`)

Every recorder action logs with a **`recorder:`** prefix, so one grep tells you
whether segments are being written and whether a given event's clip was cut:

```bash
docker compose logs -f backend | grep recorder:
```

Healthy 24/7 recording:

- `recorder: recording {camera} from rtsp://…` — the recorder started for that camera
- `recorder: first segment written for {camera}` — footage is flowing

Trouble producing segments (the main stream stalled or won't open):

- `recorder: {camera} no new segment for 30 s — killing ffmpeg (respawn)`
- `recorder: {camera} produced NO segment within … s of start — …`

Clip extraction:

- `recorder: clip extract start event=… window=… candidates=N` — assembling a clip
- `recorder: clip ready event=… -> …/clips/{id}.mp4 (bytes=…, N segments)` — clip landed (`has_clip` set)
- `recorder: clip FAILED event=… — no segments in window […]` — no footage for that time (record off, or the stream was down during the event) → the event ends up `unavailable`
- `recorder: clip FAILED event=… — ffmpeg exited …` / `… empty output …` / `… ffmpeg unavailable` — extraction error

Rule of thumb: steady segment lines **plus** `clip ready` = healthy. A
`clip FAILED … no segments in window` on an event that shows *"recording
unavailable"* points straight at the recorder not having footage for that window.

### Retention & storage

Retention is set in **Settings → Recording** and applied live by an hourly pass
(nothing restarts):

- `continuous_days` — how long 24/7 segments are kept (default **7**)
- `event_days` — how long event clips are kept (default **14**)
- `snapshot_days` — how long snapshots + event rows are kept (default **14**)

**Low-disk guard:** if the media filesystem drops under **5 GB free**, Vigilume
prunes the oldest recording hours regardless of retention and logs it loudly.

Point `MEDIA_PATH` at the big disk **before** first start. Storage math (≈130 GB/day
for the default three cameras, ≈0.9 TB/week) is in
[faq.md](faq.md#how-much-storage-do-i-need). H.265 halves those numbers with a
clip-playback trade-off — see
[cameras-amcrest.md](cameras-amcrest.md#3-stream-settings-both-turrets).

### When an event says "recording unavailable"

Walk through the recorder-log / stream / record-toggle / disk checklist in the
Unraid guide: [deploy-unraid.md → Troubleshooting](deploy-unraid.md#troubleshooting).

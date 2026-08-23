# Deploying Vigilume NVR on Unraid

Vigilume runs on Unraid as a small Docker Compose stack. This guide is written
around the NVIDIA path (the fastest for detection), but the stack also runs on
an **AMD/Intel iGPU or CPU-only box** — the NVIDIA reservation ships commented
out, so nothing here is mandatory. If that is your setup, read
[6c. Switching between CPU, NVIDIA and AMD](#6c-switching-between-cpu-nvidia-and-amd)
first and treat the NVIDIA steps as optional. The big picture is in the
[README](../README.md); GPU internals are in [setup-nvidia.md](setup-nvidia.md).

> **Why the Mac build failed:** the GPU image is built around NVIDIA CUDA and
> only runs on a real NVIDIA card with the Linux NVIDIA driver — it cannot run
> under Docker Desktop/Colima on a Mac (no GPU passthrough) and the CUDA base
> image is x86-only. Build and run it **on the Unraid box**, not the Mac.

## 1. Prerequisites on Unraid

1. **Community Applications** plugin installed (Apps tab).
2. **Nvidia Driver** plugin (by *ich777*) — **only if you have an NVIDIA card.**
   Skip this entirely on an AMD/Intel or CPU-only box; see
   [6c. Switching between CPU, NVIDIA and AMD](#6c-switching-between-cpu-nvidia-and-amd)
   for what those boxes do instead. Apps → search "Nvidia Driver" → install →
   **reboot**. This installs the NVIDIA kernel driver and the container runtime
   that lets Docker see the GPU.
   - Verify in the Unraid terminal:
     ```bash
     nvidia-smi          # should list your card
     nvidia-smi -L       # copy the "GPU-xxxxxxxx-..." UUID
     ```
   - Put that UUID in `.env` as `NVIDIA_VISIBLE_DEVICES=GPU-...`.
   - Then **uncomment the `deploy:` block** in the backend service — it ships
     commented out so the stack deploys on GPU-less boxes.
3. **Docker Compose** — either the *Docker Compose Manager* plugin (by
   *dcflachs*), or just run `docker compose` from the Unraid terminal
   (available on current Unraid).

### Give Docker enough image storage

The slim GPU image (minimal `nvidia/cuda:*-base` + pip CUDA 12 / cuDNN 9 libs +
onnxruntime + ffmpeg) is **~5 GB** — down from the ~9–10 GB it was on the
old full `-cudnn-runtime` base. Unraid's default Docker image (`docker.img`) is
only **20 GB** and is often already partly full, so a too-small store can still
fail the build with *"no space left on device"*.

**Build peak ≠ final size.** The finished image is ~5 GB, but *building* it
briefly needs **~10–15 GB free** — pip holds the downloaded CUDA wheels
(~1.8 GB) and their unpacked form (~2.9 GB) and the onnxruntime layer all at
once. Stacked on your other containers, the default 20 GB `docker.img` fills up
mid-unpack and fails with `No space left on device` during
`pip install` — which is exactly the error to expect here.

Fix it in **Settings → Docker** (stop the Docker service first to edit):
- **Best:** switch to **directory mode** pointing at your **cache/NVMe pool**
  (e.g. `/mnt/cache/docker`) — no fixed cap, uses the pool's free space, and
  fast. This is the robust fix; you won't hit the wall again, **or**
- Increase the **vDisk size** to **≥ 64 GB** (not just 40 — leave headroom for
  the build peak + your existing containers).

Check free space first with `df -h /var/lib/docker` in the Unraid terminal.

**Critical for speed:** the Docker image store must live on a **cache/NVMe
pool**, NOT the array. If `docker.img` (or the directory) is on the array
(`/mnt/user/...` via Unraid's FUSE layer, or a spinning `/mnt/diskN`),
unpacking the CUDA image crawls and looks *stuck on extracting* for many
minutes. On a cache SSD/NVMe the same extract is quick. The Unraid default
(`docker.img` on the cache pool) is correct — only a manual move to the array
causes this. Re-enable Docker afterward.

## 2. Put the project on the array

Copy the whole `Security` folder to a share, e.g. `/mnt/user/appdata/vigilume/`
(SMB from the Mac, or `git clone` in the terminal). The `./data` and
`./go2rtc/config` bind mounts live here — small (DB, snapshots, the ~100 MB
model, go2rtc config), fine on appdata/cache.

## 3. Create the recordings share (array, not cache)

Recordings are large (~135 GB/day for three cameras) and must land on the
**array**, never a cache SSD. Create a share (e.g. **`vigilume`**) and make sure
`.env`'s `MEDIA_PATH` matches its `/mnt/user` path:

```dotenv
MEDIA_PATH=/mnt/user/vigilume/media
```

Plan retention against your array's free space in Settings → Recording after
first boot. 135 GB/day × 7 days ≈ 1 TB for a week of continuous footage.

## 4. Fill in `.env`

Already mostly done. Confirm:
- `ADMIN_PASSWORD` — your web/app login (set).
- `NVIDIA_VISIBLE_DEVICES` — the GPU UUID from step 1 (uncomment + paste).
- `MEDIA_PATH` — the array share from step 3.
- `PUBLIC_URL` — your HTTPS URL (`https://…`) if you reverse-proxy it; see §7.

## 5. Build & start

From the project folder in the Unraid terminal:

```bash
docker compose up -d --build
```

First run pulls the CUDA base image and builds — **several minutes**, and the
very first backend start also downloads the detection model (~100 MB) and
compiles GPU kernels, so give it a minute before the detector reports ready.

## 6. Verify the GPU is actually being used

```bash
docker compose logs -f backend        # watch for the detector self-test line
```

You want the detector self-test line
`GPU OK — provider=CUDAExecutionProvider model=dfine_s warmup_ms=<x> infer_ms=<y>`
(the on-box acceptance gate — `detector.device` becomes `"cuda"`). If instead
it says `GPU UNAVAILABLE` with `VIGILUME_REQUIRE_GPU=1`, the container can't see
the card — see troubleshooting below. (A missing CUDA/cuDNN library can't be the
cause: the image build runs a sanity check that fails if any required `.so` is
absent.) Then:

```bash
# From a browser or curl on the LAN (port 8080 is the web UI):
curl -s http://<unraid-ip>:8080/api/system/health
# -> {"status":"ok","detector":{"ready":true,"device":"cuda","model":"dfine_s"},...}

nvidia-smi   # should show the vigilume-backend python process on the GPU
```

Open `http://<unraid-ip>:8080`, log in, and add your three cameras in
Settings → Cameras (model dropdown + the camera's own IP/username/password).
Detection, recording, live view, and notifications light up from there.

## 6b. HEVC cameras: iGPU transcoding on an AMD/Intel box

Skip this if your cameras are all H.264, or if you have an NVIDIA card (NVENC
is picked automatically).

H.265/HEVC cameras need an HEVC→H.264 transcode for browser playback, and on a
box with no usable GPU that runs on the CPU (`libx264`). Most Unraid boxes built
on a Ryzen APU or an Intel CPU with Quick Sync have an iGPU that does this in
fixed-function silicon instead — it just has to be passed into the container.
Background: [recordings.md → Transcoding hardware](recordings.md#transcoding-hardware-nvenc-vaapi-libx264).

**1. Check the render node exists.** In the Unraid terminal:

```bash
ls -l /dev/dri
# want: renderD128 (and card0). renderD129 = a second GPU.
```

Unraid's stock kernel carries both `amdgpu` and `i915`, so on most boards the
node is simply there. If `/dev/dri` is missing or empty, the iGPU is usually
disabled in the BIOS (look for *IGD / iGPU Multi-Monitor / Primary Display* and
make sure the integrated GPU stays enabled even with a discrete card fitted).
ich777's **Radeon TOP** (AMD) / **Intel GPU TOP** (Intel) plugins from Community
Applications are the usual way to confirm and monitor it.

**2. Point Vigilume at it.** Add the render node to `.env` next to the project:

```bash
# .env
VAAPI_DEVICE=/dev/dri/renderD128
```

Then `docker compose up -d backend`. Leave it unset and nothing changes (the
mapping is inert).

Unlike the Jellyfin/Plex guides you may have followed on Unraid, **no
`group_add: video` is needed** — the backend container runs as root, so it can
open the render node whatever group owns it on the host.

**3. Confirm it took.**

```bash
docker logs vigilume-backend 2>&1 | grep 'transcode:'
# want: transcode: selected H.264 encoder h264_vaapi (GPU VAAPI on /dev/dri/renderD128)
```

`selected H.264 encoder libx264 (CPU libx264)` means the node never arrived —
recheck step 1 and that `VAAPI_DEVICE` is in the `.env` compose actually reads
(the one beside `docker-compose.yml`). A `h264_vaapi failed at runtime — using
libx264` line means the opposite: the node arrived, but the image has no VA
driver — see step 4.

**4. The image must include the Mesa VA driver.** VAAPI needs
`mesa-va-drivers` inside the backend image. This stack **builds** the backend
from `backend/Dockerfile`, which installs it, so a rebuild is all it takes:

```bash
docker compose up -d --build backend
```

An image built before VAAPI support was added logs the runtime-failure line
above no matter how the node is passed through. If you ever switch to a
prebuilt GHCR image instead of building, the same applies — pull one new
enough to contain the driver.

## 6c. Switching between CPU, NVIDIA and AMD

Two independent jobs use hardware, and they need not run on the same chip — a
Coral can do detection while an AMD iGPU does transcoding:

- **Detection** — NVIDIA CUDA, a Coral Edge TPU, or the CPU.
- **Transcoding** — HEVC→H.264 for browser playback, and *only* for H.265
  cameras. NVIDIA NVENC, an AMD/Intel iGPU via VAAPI, or the CPU (`libx264`).

Everything is switched with **stack environment variables**, with exactly one
exception: the NVIDIA reservation, which is a `deploy:` block and cannot be
made conditional with a variable. It therefore ships **commented out** in both
compose files, so the stack deploys unchanged on AMD, Intel and CPU-only boxes.

| Your box | `deploy:` block | Variables to set |
|---|---|---|
| **NVIDIA GPU** | **uncomment** | *(none — defaults are right)*; optionally `NVIDIA_VISIBLE_DEVICES=GPU-…` |
| **AMD / Intel iGPU + Coral** | leave commented | `VAAPI_DEVICE=/dev/dri/renderD128`, `CORAL_DEVICE=/dev/apex_0`, and pick *Coral* in Settings → Recording → Detection hardware |
| **AMD / Intel iGPU, no Coral** | leave commented | `VAAPI_DEVICE=/dev/dri/renderD128`, `VIGILUME_DETECTOR=onnx_cpu`, `VIGILUME_REQUIRE_GPU=0` |
| **CPU only** | leave commented | `VIGILUME_DETECTOR=onnx_cpu`, `VIGILUME_REQUIRE_GPU=0` |

The block to uncomment (backend service, both compose files):

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Uncommenting it **without** the Nvidia Driver plugin installed fails the whole
stack with *could not select device driver "nvidia" with capabilities:
[[gpu]]* — `docker compose up` aborts before any container starts, so nothing
is left half-running. Installing the plugin but forgetting to uncomment is the quieter
failure: the stack comes up and detection logs `GPU UNAVAILABLE` with
`ready:false` (because `VIGILUME_REQUIRE_GPU` defaults to `1`), which is the
intended loud-rather-than-slow behavior.

Two things that trip people up when switching:

- **`VIGILUME_DETECTOR` is an override, not the setting.** Leave it unset and
  the stored *Settings → Recording → Detection hardware* choice wins. Set it to
  a non-empty value and that choice can never take effect again — a box with
  `VIGILUME_DETECTOR=onnx` in its environment cannot be switched to the Coral
  from the UI, no matter what you click. Only set it to force CPU (`onnx_cpu`).
- **VAAPI is independent of all of the above.** It uses the render node, never
  CUDA, so it works the same whether detection runs on a Coral, the CPU, or an
  NVIDIA card you also happen to have.

Confirm what actually got picked:

```bash
docker logs vigilume-backend 2>&1 | grep -E 'transcode:|GPU OK|GPU UNAVAILABLE'
```

## 7. Remote access / HTTPS (`nvr.example.com`)

Phones require HTTPS for PWA install + push, and browsers require it for the
mic (push-to-talk). Two common Unraid routes:
- **Your existing reverse proxy** (SWAG / Nginx Proxy Manager / Traefik):
  proxy `nvr.example.com` → `http://<unraid-ip>:8080`, and forward
  WebSockets (they do by default). Set `PUBLIC_URL=https://nvr.example.com`.
  Leave `COMPOSE_PROFILES` empty (the proxy terminates TLS, not Caddy).
- **Tailscale** (Unraid has a Tailscale plugin): simplest for private access —
  see [remote-access.md](remote-access.md).

**WebRTC live view:** after it's up, add this Unraid host's LAN IP (and
Tailscale IP if used) with port 8555 in **Settings → System → WebRTC
addresses**, so browsers can negotiate the low-latency stream. Without it,
live view still works over the MSE fallback.

## Troubleshooting

- **`no space left on device` during build** → Docker image store too small;
  see §1 "Give Docker enough image storage".
- **Detector logs `GPU UNAVAILABLE` / `device: null`** → the container isn't
  getting the GPU. Check `nvidia-smi` works on the host (driver plugin +
  reboot done), the UUID in `.env` is correct, then try adding `runtime: nvidia`
  to the `backend:` service in `docker-compose.yml` (some Unraid setups need the
  explicit runtime name in addition to the `deploy.devices` block). `docker
  compose up -d` to apply.
- **Cameras show offline / no events** → the camera IP/password in Settings is
  the camera's own RTSP credential (for the AD410, the password set in the
  Amcrest Smart Home app). Test it with the "Save & Test" button on the camera
  page.
- **Event says "recording unavailable"** (its clip won't play) → the recorder
  had no footage to cut from for that event's window. Work through:
  1. **Read the recorder logs** — segment production *and* clip extraction share
     the `recorder:` prefix:
     ```bash
     docker compose logs backend | grep recorder:
     ```
     Healthy looks like steady `recording {camera} …` / `first segment written`
     lines plus a `clip ready event=… -> …/clips/{id}.mp4` for the event. A
     `clip FAILED … no segments in window […]` or repeated `no new segment for
     30 s — killing ffmpeg` means the recorder wasn't producing segments.
  2. **Confirm Record is enabled** for that camera (Settings → Cameras). With it
     off there are no segments and no clips (the event shows
     `clip_state: recording_disabled`).
  3. **Verify the camera's main stream works** — ffprobe the go2rtc main
     restream the recorder pulls from (the backend image ships ffmpeg):
     ```bash
     docker compose exec backend ffprobe -v error -rtsp_transport tcp \
       rtsp://go2rtc:8554/<camera>
     ```
     No streams / a hang means the main stream is down or unplayable, which shows
     up as `no new segment` respawns above.
  4. **Check free disk** on `MEDIA_PATH` (`df -h /mnt/user/vigilume/media`) —
     under 5 GB free the low-disk guard prunes the oldest footage regardless of
     retention, which can erase an old event's segments.

  Full clip lifecycle, log reference, and file layout:
  [recordings.md](recordings.md).
- **To try without a GPU** (e.g. on a GPU-less box) → build the CPU target:
  `docker compose build --build-arg nothing backend` won't switch targets;
  instead set the backend `build.target: cpu` in a `docker-compose.override.yml`
  and `VIGILUME_REQUIRE_GPU=0`. Detection runs on CPU (slow). Not for production.

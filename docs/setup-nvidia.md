# NVIDIA host setup (Linux)

Vigilume's backend container runs object detection with **ONNX Runtime's CUDA
execution provider** (D-FINE model, `onnxruntime-gpu==1.27.0`). The GPU image
is **slim** (~5 GB): it uses a minimal `nvidia/cuda:12.9.1-base` base and
ships the CUDA 12 runtime + cuDNN 9 that the CUDA EP needs as pinned pip
`nvidia-*-cu12` wheels, rather than the multi-GB `-cudnn-runtime` base. The
NVIDIA **driver** is still provided by the host via the container toolkit — we
bundle only the userspace CUDA/cuDNN libraries, not a driver. The Linux host
needs two things before `docker compose up`:

1. an NVIDIA driver,
2. the NVIDIA Container Toolkit wired into Docker.

There is no model export step — the backend downloads its pinned,
SHA-256-verified model on first boot. Everything below assumes Ubuntu
22.04/24.04 or Debian 12; adjust package commands for other distros.

## 1. NVIDIA driver

Vigilume pins the **CUDA 12.9** userspace libraries (cuDNN 9), which run on the
**R550+** driver series via CUDA's minor-version forward-compatibility (a
driver shipped for an earlier 12.x still runs 12.9 runtime code; the
`nvidia/cuda:*-base` image also carries the CUDA forward-compat layer). R575+
gives native 12.9 support with no reliance on forward-compat. Just install the
current stable branch from your distro:

```bash
# Ubuntu — pick the recommended driver automatically:
sudo ubuntu-drivers install
# or explicitly, e.g.:
sudo apt install nvidia-driver-550
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

You should see the GPU, driver version, and CUDA version. If `nvidia-smi`
fails, stop here — nothing downstream will work.

> Headless/server tip: the `-server` driver variants (e.g.
> `nvidia-driver-550-server`) are fine; Vigilume needs CUDA compute, which
> they include.

> Looking ahead: ONNX Runtime 1.27 is the last comfortable pin on the CUDA 12
> line — its **default** PyPI wheel already switched to CUDA 13, so Vigilume
> installs the CUDA-12 build of 1.27.0 from Microsoft's official
> `onnxruntime-cuda-12` package feed (see `backend/Dockerfile`). When Vigilume
> moves to a CUDA 13 build, an R580+ driver becomes the requirement and the
> pinned `nvidia-*-cu12` wheels become `-cu13` — release notes will call it out.

## 2. NVIDIA Container Toolkit

Installs the runtime hooks that let Docker containers see the GPU:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Wire it into Docker and restart the daemon:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify with a CUDA container

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

If this prints the same table as the host `nvidia-smi`, Docker↔GPU wiring is
done. `docker-compose.yml` requests the GPU declaratively for the backend, so
no `--gpus` flag is needed for the stack itself:

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 3. `VIGILUME_REQUIRE_GPU` — what it does

ONNX Runtime **silently falls back to CPU** when the CUDA provider can't load
(missing toolkit wiring, driver mismatch, no device reservation). Vigilume
refuses to let that pass unnoticed:

- `VIGILUME_REQUIRE_GPU=1` (**the shipped default**): if CUDA is unavailable,
  the detector hard-fails — `GET /api/system/health` reports
  `detector: {ready: false, device: null, ...}` and the log shows
  `GPU UNAVAILABLE — set VIGILUME_REQUIRE_GPU=0 to accept CPU`. The web UI,
  live view and 24/7 recording keep working; only detection is down until you
  fix the GPU wiring.
- `VIGILUME_REQUIRE_GPU=0`: CPU inference is consciously allowed (dev
  machines / temporary operation). The detector comes up with
  `device: "cpu"` and a WARNING log stating the measured per-frame cost.

## 4. First-boot checklist

Run through this after the first `docker compose up -d --build`:

1. **GPU visible inside the backend container:**

   ```bash
   docker exec vigilume-backend nvidia-smi
   ```

2. **Model download + GPU self-test in the logs:**

   ```bash
   docker logs -f vigilume-backend
   ```

   You should see the model land in `/data/models/` (~40 MB for the default
   `dfine_s`, SHA-256 verified) followed by exactly one of:

   - `GPU OK — provider=CUDAExecutionProvider model=dfine_s warmup_ms=<x> infer_ms=<y>`
     — you're done; expect single-digit-millisecond inference on any recent
     card.
   - `GPU UNAVAILABLE — set VIGILUME_REQUIRE_GPU=0 to accept CPU` — the CUDA
     provider didn't load; see Troubleshooting below. (The image is built with
     a sanity check that fails if a required CUDA/cuDNN `.so` is absent, so this
     at runtime points at the host GPU wiring — driver/toolkit/device
     reservation — not a missing library.)

3. **Detector self-test endpoint** — log in to the UI, or call it with a
   token:

   ```
   GET /api/system/detector
   → {ready: true, device: "cuda", model: "dfine_s", model_sha_ok: true,
      last_inference_ms: ..., per_camera: [{name, ingest_ok, fps, last_frame_age_s}]}
   ```

   Every detect-enabled camera should show `ingest_ok: true` and `fps` near
   its configured detect rate (default 5). **Settings → System** shows the
   same data.

4. **Live view** plays on the dashboard (WebRTC needs your host IPs under
   **Settings → System → WebRTC addresses** — until then it falls back to
   MSE automatically).

5. **Walk test:** walk in front of a camera → an event appears with an
   annotated snapshot and a push notification (if set up); ~30 s after you
   leave the frame the event's clip becomes playable.

6. **Recovery test (optional but recommended):** unplug a camera — logs show
   ingest/recorder backoff, the dashboard tile goes offline, and everything
   resumes on reconnect. Reboot the host — the stack comes back by itself and
   segments recorded before the cut still play.

## Troubleshooting quick hits

- `nvidia-smi` works on the host but the CUDA container fails → rerun
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- Health shows `detector.ready: false` with the `GPU UNAVAILABLE` log →
  confirm the compose GPU reservation is present and
  `docker exec vigilume-backend nvidia-smi` works; after driver updates,
  containers can lose GPU access until Docker restarts
  (`sudo systemctl restart docker`).
- Health shows `detector.ready: false` with a model download error → the host
  can't reach huggingface.co, or the hash check failed. Retries are
  automatic; for offline installs drop the model file at
  `./data/models/<key>.onnx` manually and verify its SHA-256 against the pin
  table in [CONTRACTS.md](CONTRACTS.md#detection-model-d-fine-pinned-by-revision-and-sha-256).
- Detection much slower than expected while `device` is `"cuda"` → another
  process is hogging the GPU (`nvidia-smi` shows who), or the card is in a
  low-power state.
- Cameras online but `per_camera[].ingest_ok: false` → the go2rtc restream
  isn't up (`docker logs vigilume-go2rtc`) or the camera's RTSP credentials
  are wrong (test with `ffprobe`/VLC — see
  [cameras-amcrest.md](cameras-amcrest.md#3-stream-settings-both-turrets)).

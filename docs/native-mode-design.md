# Native Mode Design — `SENTINEL_MODE=native`

> **NOTE (2026-07, final direction):** deployment modes were DROPPED in favor
> of full replacement. There is no `SENTINEL_MODE`; bundled/external (Frigate
> + MQTT) were deleted outright and the "native" engine described here is the
> ONLY architecture. This document stays as the **design record** for the
> engine internals (detector pins/hashes, decode math, tracker settings,
> ingest/recorder/go2rtc patterns, test plan). Where it talks about mode
> switches, `docker-compose.native.yml`, `is_native` branches, MQTT, or
> Frigate compatibility shims, the standalone contract addendum overrides it:
> one compose file, `app.state.media` (no `frigate` alias), health shape
> `{status, version, detector:{ready,device,model}, go2rtc, cameras_online}`,
> native event ids prefixed `native.`, and no audio-event settings.

**Status:** design record (see note above); originally: ready to implement.
**Date:** 2026-07-06. All third-party facts below were verified against live sources on this date; items that could not be verified on the dev Mac are called out in §8.

Native mode makes Vigilume fully standalone: **its own object detection
(onnxruntime-gpu + D-FINE), its own 24/7 recording (ffmpeg stream-copy), and
its own live view (Vigilume-owned go2rtc, WebRTC-first)**. Frigate and MQTT
are not involved at all. The public API surface (`/api/events`,
`/api/cameras`, media routes, WS messages, notifications) is unchanged — the
frontend works as-is except for small settings-tab differences noted in §7.

Design north star: the existing `EventsPipeline` (`backend/app/events_pipeline.py`)
already consumes Frigate-*shaped* event payloads defensively. Native mode
**synthesizes those same payloads in-process** and swaps the media backend
(`FrigateClient`) for a native implementation with the same call surface.
Events, enrichment, annotation, notifications, WS broadcast, and the UI work
unchanged.

---

## 0. Verified version pins (mid-2026)

| Thing | Pin | Verified how |
|---|---|---|
| Detector model | D-FINE-S COCO, ONNX fp32 from `onnx-community` (HF) | Downloaded, ran inference on CPU EP, decoded a real image correctly (see §1.4) |
| supervision | `supervision>=0.29.1,<0.30` | 0.29.1 released 2026-06-23 (PyPI). `sv.ByteTrack` is **deprecated since 0.28.0 and removed in 0.30.0** (deprecation warning reproduced locally) — do not use it |
| tracker | `trackers==2.4.0` (`ByteTrackTracker`) | **`trackers` 2.5.0 wheel on PyPI is broken/empty (9.7 kB, contains no package code — verified by installing it; import fails).** 2.4.0 (126 kB wheel) imports and tracks correctly. Pin 2.4.0 exactly; re-check 2.5.x at build time |
| onnxruntime-gpu | `onnxruntime-gpu==1.27.0`, **CUDA-12 build** (CUDA 12.x + cuDNN 9.x) | **CORRECTION (2026-07, verified against the 1.27.0 wheel metadata):** the *default* PyPI `onnxruntime-gpu==1.27.0` wheel is **CUDA 13** — its `[cuda,cudnn]` extras pull `-cu13` wheels — because ORT 1.27 deprecated CUDA 12. To keep CUDA 12.9 + cuDNN 9 we install the **CUDA-12 build** of 1.27.0 from Microsoft's official `onnxruntime-cuda-12` feed and supply the cu12 libs as pip wheels. See §2.1. (Supersedes the earlier "default PyPI still targets CUDA 12" claim.) |
| CPU fallback | `onnxruntime==1.27.0` | Installed in the scratchpad venv (py3.14 wheel exists); ran D-FINE-S at ~92 ms/frame on an M-series Mac |
| CUDA base image | `nvidia/cuda:12.9.1-base-ubuntu24.04` (minimal; CUDA/cuDNN libs come from pip — see §2.1) | Tag verified via Docker Hub API 2026-07 (also verified: `13.0.1-base-…` for the CUDA-13 upgrade path). The former `-cudnn-runtime` base bundled multi-GB CUDA/cuDNN, most unused → replaced to shrink the image ~9–10 GB → ~5 GB |
| go2rtc | `alexxit/go2rtc:1.9.14` | Latest release v1.9.14 (2026-01-19), config keys verified against README (`streams:`, `webrtc: listen/candidates`, `api: listen`) |

---

## 1. Detection model

### 1.1 Decision: D-FINE-S (default), D-FINE-N / D-FINE-M as options

[D-FINE](https://github.com/Peterande/D-FINE) (ICLR 2025 spotlight), **Apache-2.0**
(license verified on the repo). DETR-family → **NMS-free**: output is a fixed
300 queries with per-class scores; decoding is a sigmoid + threshold, no NMS,
no anchor/grid decode. COCO 80-class. Accuracy/speed (repo table, COCO val,
T4 TensorRT):

| Model | mAP | T4 latency | Params | GFLOPs | ONNX size |
|---|---|---|---|---|---|
| D-FINE-N | 42.8 | 2.12 ms | 4 M | 7 | 15.3 MB |
| **D-FINE-S** | **48.5** | **3.49 ms** | 10 M | 25 | 41.5 MB |
| D-FINE-M | 52.3 | 5.62 ms | 19 M | 57 | 78.6 MB |

For comparison, Frigate-bundled-mode's YOLOv9-t (320) and the strongest
permissive classic (YOLOX-S, 40.5 mAP) are both well below D-FINE-S. For
person/dog/cat/car at 5 fps on a dGPU, D-FINE-S at 640 is the sweet spot;
D-FINE-N exists for CPU-fallback operation and D-FINE-M for operators with
GPU headroom.

**Candidates evaluated and rejected:**

- **YOLOX** (Apache-2.0): the only candidate with truly *first-party* hosted
  ONNX (GitHub release assets, verified live:
  `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx`,
  35.9 MB). Rejected as primary: 2021-era accuracy (S = 40.5 mAP, 8 points
  below D-FINE-S at similar cost), raw-grid output needing stride decode +
  NMS, letterbox preprocessing. **Documented fallback** if HF hosting is ever
  unacceptable — keep the loader model-agnostic enough that adding a YOLOX
  decode path later is contained to `native/detector.py`.
- **RT-DETRv2/v3** (Apache-2.0): good, but no official hosted ONNX (.pth only,
  export needs torch); D-FINE supersedes it on the same codebase lineage.
- **RF-DETR** (Roboflow, core models Apache-2.0, ICLR 2026): strong accuracy,
  but ONNX export runs through the `rfdetr` pip package **with torch at
  runtime**; no stable hosted ONNX artifact. Revisit if Roboflow starts
  hosting ONNX.
- **YOLO-NAS**: architecture Apache-2.0 but the *weights* are under Deci's
  restrictive license — fails the license criterion.
- **DAMO-YOLO** (Apache-2.0): stale since 2023, ONNX on ModelScope only.
- **Ultralytics YOLOv8/26, YOLOv9, YOLOv10**: AGPL-3.0 / GPL-3.0 — excluded
  by contract.

### 1.2 Pinned artifacts

The official D-FINE repo ships `.pth` only (verified). The pinned ONNX
artifacts are the **`onnx-community` conversions of the official
`ustc-community/dfine-*-coco` checkpoints** (the HF-maintained transformers.js
org — stable, revision-addressable URLs). Pin by **revision + SHA-256**;
`native/model_store.py` verifies the hash after download (see §7.5):

| key | labelmap | URL (immutable, revision-pinned) | bytes | SHA-256 |
|---|---|---|---|---|
| `dfine_n` | coco | `https://huggingface.co/onnx-community/dfine_n_coco-ONNX/resolve/380d2839c327efaf65dd0fe0c2c10ab7fadd5473/onnx/model.onnx` | 15,258,358 | `0f684f409618ee8a822410e754a29caa817d1aa16283ce89cad936d0a48e2f35` |
| `dfine_s` (default) | coco | `https://huggingface.co/onnx-community/dfine_s_coco-ONNX/resolve/a3cf03147a9b86c78475139115c8ac142577352d/onnx/model.onnx` | 41,535,197 | `cd8a49a945feda6d28c6304ae8ae85c2759ba1d78a5a83a22c5ce8db82ef7238` |
| `dfine_m` | coco | `https://huggingface.co/onnx-community/dfine_m_coco-ONNX/resolve/489756db2825cb068a588fe930af239a656d1fe1/onnx/model.onnx` | 78,624,257 | `70aaa837978a06ba44ad17398c7079ae5a1a7b1a9032b5d7053981e1ada02d6b` |
| `dfine_l` | coco | `https://huggingface.co/onnx-community/dfine_l_coco-ONNX/resolve/e09b218185400510b68d067bbd0c9379d905ffcd/onnx/model.onnx` | 125,348,332 | `d678f3baebfb909d3a20f21d1d807544d0172ed47fa1ab88e7fcdec7e365b236` |
| `dfine_x` | coco | `https://huggingface.co/onnx-community/dfine_x_coco-ONNX/resolve/b67b16d98f1f95e2af0b6d0b8f74ad956557fe8c/onnx/model.onnx` | 251,138,448 | `644fb5124c9c035a6082f23419da693c19ac857bd984ab7af5e353779368a03b` |
| `dfine_l_obj365` | obj365 | `https://huggingface.co/onnx-community/dfine_l_obj365-ONNX/resolve/32323e9a0c74b22803a2ad5b315a8fed61801e6a/onnx/model.onnx` | 125,936,360 | `cd0dfa92a2e0e2ab3d4a7c2e6252ebc094aa77b67f2f9dc1a010dd350a9a2f3e` |

(N and M hashes from HF `x-linked-etag` LFS headers; S, L, X and the obj365
hash additionally verified by downloading each pinned-revision file and running
`shasum -a 256` locally.) `labelmap` names the output vocabulary
(`coco_labels.LABELMAPS`): the COCO tiers emit `logits[1,300,80]`; `dfine_l_obj365`
emits `logits[1,300,366]` (Objects365, id 0 a background `none` placeholder). The
graph I/O and preprocessing (§1.3) are identical across every model, so the decode
is shared — only the class-count and labelmap differ. The detector re-points the
engine's live `ID_TO_LABEL` view to the loaded model's labelmap on each swap, so
tracking/annotate need no change (labels are just strings). Licensing: D-FINE code
+ COCO weights are Apache-2.0; the obj365 weights are trained on the Objects365
dataset (its own non-commercial-research terms) — fine for a self-hosted NVR, noted
here for downstream redistribution awareness.

Storage: `/data/models/{key}.onnx` (+ `{key}.json` sidecar recording url,
sha256, downloaded_at). `/data` is already a persistent volume.

### 1.3 Inference contract (verified by running the actual model)

Graph I/O (printed from the downloaded `dfine_s` ONNX via onnxruntime):

```
inputs : pixel_values  float32 [batch, 3, height, width]   (dynamic; use 1×3×640×640)
outputs: logits        float32 [batch, 300, 80]
         pred_boxes    float32 [batch, 300, 4]   # (cx, cy, w, h), normalized 0..1
```

**Preprocessing** (from the repo's `preprocessor_config.json` —
`RTDetrImageProcessor`, verified: `do_resize` 640×640 bilinear, `do_rescale`
1/255, `do_normalize: false`, `do_pad: false`):

1. BGR frame (from ffmpeg pipe) → RGB.
2. **Plain resize to 640×640** (`cv2.resize`, `INTER_LINEAR`). **No letterbox,
   no padding, no mean/std normalization.** Aspect distortion is part of the
   model's training regime.
3. `astype(float32) / 255.0`, HWC→CHW, add batch dim.

**Decode (NMS-free)** — validated end-to-end on a real image (D-FINE-S on CPU
EP detected `dog 0.955` with a correct box, single detection, no duplicates):

```python
scores = 1.0 / (1.0 + np.exp(-logits[0]))        # [300, 80] sigmoid
conf   = scores.max(axis=1)                       # best class score per query
cls    = scores.argmax(axis=1)
keep   = conf >= confidence_threshold             # settings.detection.confidence, default 0.5
cx, cy, w, h = pred_boxes[0][keep].T              # normalized
# scale into DETECT-STREAM pixel space (camera detect_width/height, e.g. 704x480)
xyxy = np.stack([(cx - w/2) * dw, (cy - h/2) * dh,
                 (cx + w/2) * dw, (cy + h/2) * dh], axis=1).astype(np.float32)
```

Boxes are produced directly in detect-stream pixels so the **existing
annotator maps them 1:1** (`annotate.py` docstring assumption holds; the
native snapshot is the raw detect-res frame).

Class mapping: contiguous COCO-80 (verified from the repo `config.json`
`id2label`): `0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck,
15=cat, 16=dog, …`. Ship the full 80-entry list as `native/coco_labels.py`;
filter to the camera's `detect_objects` (default
`["person","dog","cat","car"]`) *after* decode, *before* tracking.

### 1.4 supervision + tracking (exact, pinned APIs)

```python
import supervision as sv                     # ==0.29.1 (pin <0.30)
from trackers import ByteTrackTracker       # trackers==2.4.0  (2.5.0 wheel is broken on PyPI)

detections = sv.Detections(
    xyxy=xyxy,                               # float32 [N,4], detect-stream px
    confidence=conf[keep].astype(np.float32),
    class_id=cls[keep].astype(int),
)

tracker = ByteTrackTracker(                  # one instance PER CAMERA
    lost_track_buffer=25,                    # frames to keep lost tracks ≈ 5 s @ 5 fps
    frame_rate=float(detect_fps),            # camera's detect_fps (default 5)
    track_activation_threshold=0.5,
    minimum_consecutive_frames=2,
    minimum_iou_threshold=0.1,
    high_conf_det_threshold=0.6,
)
tracked = tracker.update(detections)         # -> sv.Detections with .tracker_id
```

Verified signatures (introspected from the installed 2.4.0 package):
`ByteTrackTracker(lost_track_buffer=30, frame_rate=30.0,
track_activation_threshold=0.7, minimum_consecutive_frames=2,
minimum_iou_threshold=0.1, high_conf_det_threshold=0.6, state_estimator_class=…)`;
`update(detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections`.
`tracker_id` values start at **0** (verified) — never treat 0 as falsy.
`trackers` is Apache-2.0 and "speaks `supervision.Detections` natively".

`annotate.py` needs **zero changes** — it already builds `sv.Detections` from
a single box for the snapshot annotation.

---

## 2. Runtime (GPU + CPU)

### 2.1 Container

- Base image (GPU build target): **`nvidia/cuda:12.9.1-base-ubuntu24.04`**
  (tag verified on Docker Hub 2026-07), a **minimal** base (~200 MB) — NOT the
  multi-GB `-cudnn-runtime` base. It carries only the container-toolkit driver
  injection env (`NVIDIA_VISIBLE_DEVICES`,
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so the host still mounts
  `libcuda.so.1`) plus the CUDA forward-compat layer. `python:3.12-slim` was
  rejected because it does not set `NVIDIA_DRIVER_CAPABILITIES`, so the toolkit
  would not inject the driver. Ubuntu 24.04 → glibc 2.39, satisfies the
  manylinux_2_27/2_28 wheels; ORT 1.27 ships cp312.

- **Slim-image approach (2026-07 revision — cuts ~9–10 GB → ~5 GB):**
  instead of bundling the whole CUDA runtime + cuDNN via the base image (most of
  which D-FINE inference never touches), supply ONLY the libraries the CUDA
  Execution Provider dlopens, as pinned pip `nvidia-*-cu12` wheels, and point
  `LD_LIBRARY_PATH` at their `site-packages/nvidia/<comp>/lib` dirs. This is the
  exact set ORT declares via its `onnxruntime-gpu[cuda,cudnn]` extra (the extra
  is named `-cu13` on 1.27; we pin the CUDA-12 equivalents). Complete set
  (requirements-gpu.txt, versions verified on PyPI 2026-07):

  | pip wheel | pin | provides (SONAME) |
  |---|---|---|
  | `nvidia-cuda-runtime-cu12` | 12.9.79 | `libcudart.so.12` |
  | `nvidia-cuda-nvrtc-cu12` | 12.9.86 | `libnvrtc.so.12` (+ builtins) |
  | `nvidia-cublas-cu12` | 12.9.2.10 | `libcublas.so.12`, `libcublasLt.so.12` |
  | `nvidia-cudnn-cu12` | 9.24.0.43 | `libcudnn.so.9` + graph/ops/cnn/adv/engines-precompiled/engines-runtime-compiled/heuristic |
  | `nvidia-cufft-cu12` | 11.4.1.4 | `libcufft.so.11` |
  | `nvidia-curand-cu12` | 10.3.10.19 | `libcurand.so.10` |
  | `nvidia-nvjitlink-cu12` | 12.9.86 | `libnvJitLink.so.12` |

  cuDNN (~800 MB) + cuBLAS (~580 MB) dominate; the cu12 libs total ~2.9 GB installed (~1.8 GB download; .so files unpack larger).
  `cufft`/`curand` (~270 MB combined) are in ORT's declared CUDA extra but are
  only used by FFT/random ops D-FINE does not exercise — they can be dropped for
  a further ~270 MB if a future pass wants it; kept for now to match ORT's
  extra exactly and avoid any missing-lib surprise. The Dockerfile runs a
  build-time sanity check (ctypes/`os.path.exists` over the SONAMEs above +
  `CUDAExecutionProvider in get_available_providers()`) that fails the build if
  a lib is absent — cheap insurance since the GPU image can't be run on the
  Mac dev host.

- **ORT 1.27.0 is CUDA 13 by default — we pin its CUDA-12 build.** As of 1.27
  the DEFAULT PyPI `onnxruntime-gpu` wheel switched to CUDA 13 (CUDA 12
  deprecated; the `-cu13` component wheels are not yet cleanly on public PyPI).
  To keep the CUDA 12.9 / cuDNN 9 pairing and the documented R550+ host driver,
  the Dockerfile installs the CUDA-12 build of the same `1.27.0` release from
  Microsoft's official `onnxruntime-cuda-12` package feed
  (`--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`,
  installed `--no-deps` so its deps come from PyPI and the CUDA-13 PyPI wheel is
  never pulled). This supersedes the earlier note below that assumed the default
  PyPI wheel was still CUDA 12.
- Python: Ubuntu 24.04's `python3.12` (+ `python3.12-venv`). ORT 1.27 has
  cp312 wheels. Also install `ffmpeg` from apt (used by ingest §3 and
  recording §5).
- The existing slim image stays the default build (`bundled`/`external`
  modes). Implement as a **multi-stage Dockerfile with two final targets**:
  `runtime` (current python:3.11-slim base, unchanged) and `runtime-cuda`
  (the CUDA base above). `requirements.txt` stays shared;
  `requirements-native.txt` adds `onnxruntime-gpu==1.27.0` (cuda target) or
  `onnxruntime==1.27.0` (CPU/dev), plus `trackers==2.4.0`.
- CUDA-12 deprecation note (ORT 1.27 release notes: *"CUDA 12 packages are
  deprecated, please move to CUDA 13 ASAP"*): when a future pass moves to
  ORT ≥1.28, switch the base image to `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04`
  (tag verified) and install from ORT's CUDA-13 package index. CUDA 13
  requires an R580+ host driver — flag that in setup-nvidia.md when it
  happens. Today's pin works with the R550+ driver already documented in
  `docs/setup-nvidia.md`.

### 2.2 Session setup + CPU EP fallback

```python
providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
active = sess.get_providers()[0]   # what ORT actually picked
```

ORT silently falls back to CPU when CUDA/cuDNN/driver is missing — **the
engine must detect and surface this** (§8.2): log
`detector: provider=CUDAExecutionProvider` or a loud
`detector: GPU NOT AVAILABLE — running on CPU (…)` line. Env
`VIGILUME_REQUIRE_GPU=1` (native compose default) turns silent fallback into
a hard startup failure with a clear message; set it to `0` to consciously run
CPU-only. When on CPU, the engine auto-caps aggregate inference at
`~1000 / cpu_ms_per_frame` fps (measured at startup, §8.2) and drops frames
beyond it (log once per minute), so a mis-deployed stack degrades instead of
melting.

### 2.3 TensorRT EP: not worth it at this scale — CUDA EP only (v1)

- Load: 3–6 cameras × 5 fps = **15–30 inferences/s**. D-FINE-S is 3.49 ms on
  T4 **TensorRT**; CUDA EP is typically 2–3× slower → est. **7–12 ms/frame**
  on a T4-class dGPU → 30 fps aggregate ≈ 20–35 % of the GPU, leaving NVDEC
  and headroom free. No latency SLA exists (events, not interactive).
- TRT EP costs: minutes-long engine build on first boot per (model, GPU,
  TRT version), engine cache invalidation on every driver/ORT upgrade, a much
  fatter image, and a second native-lib compat matrix. Frigate's own
  0.17-on-ONNX warm-up pain (documented in CONTRACTS) is exactly this.
- Decision: **CUDA EP only** in v1. Leave a `SENTINEL_ORT_TRT=1` escape hatch
  that prepends `TensorrtExecutionProvider` (with
  `trt_engine_cache_enable=true, trt_engine_cache_path=/data/models/trt`) for
  operators who want it; document as unsupported/experimental.

### 2.4 Per-frame budget (estimates; GPU numbers unverifiable on the dev Mac)

| Stage | Budget (per frame) | Basis |
|---|---|---|
| ffmpeg decode (D1 substream, CPU) | ~2–5 % of one core per cam | small substreams, §3 |
| BGR→RGB + resize + /255 (CPU) | 1–2 ms | 704×480 → 640×640, cv2 |
| D-FINE-S, CUDA EP | est. 7–12 ms | 2–3× the 3.49 ms T4-TRT figure |
| D-FINE-S, CPU EP | **92 ms measured** (M-series Mac) | scratchpad venv, §0 |
| decode + ByteTrack + state machine | <1 ms | numpy on 300 queries |

A single inference worker (thread) servicing a frame queue is sufficient
through 6 cams × 5 fps even at the pessimistic 12 ms (=36 % duty at 30 fps).
The `pixel_values` input has a dynamic batch dim — cross-camera batching is a
documented future optimization, not v1.

---

## 3. Frame ingest

### 3.1 Decision: ffmpeg subprocess → rawvideo pipe (Frigate's proven pattern)

Per enabled camera, one ffmpeg child process decoding the **substream** at
`detect_fps`, writing raw BGR frames to stdout; the engine reads exact-size
chunks (`width*height*3` bytes):

```
ffmpeg -hide_banner -loglevel warning -nostdin
  -rtsp_transport tcp
  -timeout 5000000                       # µs, socket I/O timeout (input option)
  -fflags nobuffer -flags low_delay
  -i rtsp://go2rtc:8554/{name}_sub
  -vf fps={detect_fps}
  -f rawvideo -pix_fmt bgr24 pipe:1
```

Reader: blocking `readexactly(frame_bytes)` on the process stdout (via
`asyncio.subprocess` + a size-1 latest-frame slot per camera — **drop, never
queue**: if the inference worker is busy, overwrite the slot so detection
always sees the freshest frame).

Why not the alternatives:

- **OpenCV `VideoCapture(CAP_FFMPEG)`**: internal buffering you can't fully
  disable (`CAP_PROP_BUFFERSIZE` is a hint, backend-dependent), no read
  timeout → a flaky camera blocks the thread indefinitely, reconnect state is
  opaque, and error reporting is a boolean. Known-bad fit for 24/7 unattended
  ingest.
- **PyAV**: good control (timeouts, error codes) but adds a large binary
  wheel, its own ffmpeg build alongside the apt ffmpeg we need anyway for
  recording, and you still hand-write the reconnect loop. More code, more
  compat surface, no capability we lack.
- **ffmpeg subprocess** (chosen): process death IS the error signal; restart
  is `kill + respawn` with backoff (2 s → 60 s cap, mirroring
  `MqttListener.run`). A **staleness watchdog** (no frame for 15 s → kill →
  respawn) catches silent stalls. ffmpeg's RTSP-over-TCP handles the Amcrest
  quirks; this is exactly Frigate's architecture, proven on these cameras.

Bandwidth sanity: 704×480×3 B × 5 fps × 6 cams ≈ 30 MB/s over a local pipe —
trivial.

### 3.2 Source: via Vigilume's go2rtc restream (not direct)

Ingest consumes `rtsp://go2rtc:8554/{name}_sub` (go2rtc restream, §6) rather
than the camera directly. One RTSP session per camera stream total (Amcrest
firmware limits concurrent RTSP clients; live view + detect + record would
otherwise open three). Same pattern as bundled Frigate
(`preset-rtsp-restream`). If go2rtc is down, ingest reconnects with backoff —
same behavior as camera-down. (`ffmpeg -i <camera sub_url>` direct is a
one-line fallback documented in faq if an operator wants to bypass go2rtc.)

### 3.3 hwaccel: not needed for detect

Substreams are D1/VGA H.264 at 5 fps post-filter. Software decode of six such
streams costs well under one core total. NVDEC would add the CUDA-context
dance to ffmpeg children for no measurable win. Recording (§5) is stream-copy
(no decode at all). Decision: **CPU decode, no hwaccel**, revisit only if an
operator feeds 4K substreams.

---

## 4. Event lifecycle → existing pipeline

### 4.1 Integration principle

`EventsPipeline.handle_event()` accepts `{type: new|update|end, after: {...}}`
dicts and already tolerates field variance. The native engine **calls it
directly (in-process, no MQTT)** with payloads shaped like Frigate's, and
feeds the live-count cache via the existing `pipeline.update_count()`. What
the pipeline consumes (verified against `events_pipeline.py`):

| Field | Native source |
|---|---|
| `after.id` | `f"ntv-{int(start_ts*1000)}-{secrets.token_hex(3)}"` — must NOT start with `doorbell.`/`audio.` (those are the synthetic-no-media prefixes in `routers/events.py`) |
| `after.camera`, `after.label` | camera name; COCO label string |
| `after.top_score` | max confidence seen for the track so far |
| `after.start_time`, `after.end_time` | epoch floats |
| `after.snapshot` = `{frame_time, score, box}` | best-frame metadata (§4.3); `box` = `[x1,y1,x2,y2]` in detect px |
| `after.box` | same box (belt & suspenders — `_box_of` checks both) |
| `after.has_clip` | `True` (recording is always-on in native; `False` if the camera's `record_enabled` is off) |
| `after.has_snapshot` | `True` once a best frame exists |
| `entered_zones`/`current_zones` | `[]` (zones are out of scope for native v1) |

`pipeline.current_count()` reads the `(camera,label)` cache; the engine calls
`pipeline.update_count(camera, label, n_active_tracks)` **whenever the number
of live confirmed tracks for that label changes** — so counts are exact in
native mode (no cold-cache fallback ever needed, but the fallback stays
harmless).

### 4.2 State machine (per camera, in `native/tracking.py`)

Per frame (after decode → label filter → `tracker.update`):

1. Group tracked detections by `tracker_id`. A track becomes **confirmed**
   after `MIN_HITS = 3` frames carrying a tracker_id (ByteTrack's own
   `minimum_consecutive_frames=2` gates id assignment; the extra hit
   suppresses one-frame flickers). Confirmation of the **first** track of a
   label with no open event → emit `type:"new"`.
2. While an event for `(camera,label)` is open: on each frame where either
   (a) the best score improved by ≥0.02, (b) the active-track count changed,
   or (c) 10 s elapsed since last emit → emit `type:"update"` (matches the
   pipeline's "meaningful updates" contract; it max-pools `count` itself via
   `state["max_count"]`).
3. **Event end**: when zero active (non-lost) tracks of that label remain for
   `ABSENCE_TIMEOUT_S = 5` (i.e. `detect_fps * 5` consecutive frames — same
   order as Frigate's default) → emit `type:"end"` with `end_time` = last
   time the label was seen. `lost_track_buffer=25` means brief occlusions
   don't end events (ByteTrack revives the id).
4. One event per `(camera,label)` at a time (mirrors what the pipeline's
   count-based model expects); multiple simultaneous objects of a label raise
   `count`, not extra events. Different labels → independent events.

On engine shutdown/restart, open events are ended with `end_time=now`
(pipeline `_on_end` handles rows it doesn't have in `_active` already).

### 4.3 Best-snapshot selection

The engine keeps, per open event, the **raw detect-res BGR frame with the
highest best-track confidence so far** (copied out of the ring slot, JPEG-
encoded lazily on first request, held in memory; one frame per open event +
last-frame-per-camera ≈ a few MB). Every time the best frame is replaced, the
next `update` payload carries a new `snapshot.frame_time` — which is exactly
the trigger `_on_update` uses to re-enrich (`needs_snapshot = has_snapshot
and snap_time != state["snap_time"]`). Enrichment then calls
`media.event_snapshot(fid)` (§4.4) and annotates with the box from the same
payload. Because the frame is detect-res and the box is in detect px,
`detect_dims` equals the image dims and the annotator's rescale is a no-op.

### 4.4 `NativeMediaProvider` — the FrigateClient stand-in

`main.py` currently injects `FrigateClient` into the pipeline and routers as
`app.state.frigate`. Native mode injects `NativeMediaProvider` implementing
the **same methods actually consumed** (verified by grepping call sites):

| Method (same signature) | Native implementation |
|---|---|
| `event_snapshot(fid, retries)` | JPEG of the event's best frame from the engine store (open events) or `/data/snapshots/{id}.jpg` fallback; `None` if unknown |
| `latest_jpg(camera, height=None)` | last decoded frame for the camera, JPEG-encoded (downscale if `height`); `None` if ingest is down → `routers/cameras.py` already falls back to the Amcrest CGI snapshot |
| `detect_dims(camera)` | `(detect_width, detect_height)` from the camera row |
| `is_healthy()` | engine running flag (used by health route) |
| `get_config() / config_cameras()` | `(True, None)` / `(True, [])` — only used by external-mode import & `in_frigate`, both disabled in native |
| `build_clip_request / send_stream` | **not implemented** — `routers/events.py` gets a small branch instead (§5.4); raising `NotImplementedError` guards regressions |
| `restart()` | no-op returning True |
| `aclose()` | no-op |

Type it as a `MediaProvider` Protocol in `native/media.py`; do NOT touch
`frigate.py` (bundled/external must keep working byte-for-byte).

---

## 5. Recording

### 5.1 24/7 segments: ffmpeg stream-copy → MPEG-TS, 10 s segments

One ffmpeg child per `record_enabled` camera, consuming the go2rtc **main**
restream (which transcodes G.711 → AAC per §6.2, so audio is TS-legal):

```
ffmpeg -hide_banner -loglevel warning -nostdin
  -rtsp_transport tcp -timeout 5000000
  -i rtsp://go2rtc:8554/{name}
  -c copy -map 0
  -f segment -segment_time 10 -segment_atclocktime 1 -reset_timestamps 1
  -strftime 1
  /media/native/recordings/{name}/%Y-%m-%d/%H/%M.%S.ts
```

The recorder pre-creates the day/hour dirs on rollover (a small clock task —
ffmpeg's strftime does not mkdir) and applies the same watchdog/backoff as
ingest (§3.1): process exit or no new segment file for 30 s → respawn.

**Container choice — TS over fragmented MP4:** a power cut corrupts at most
the in-flight 10 s `.ts` segment, and even that usually remains decodable up
to the cut (TS is a self-syncing stream format with no finalization step).
fMP4 achieves similar resilience only with careful `-movflags
+frag_keyframe+empty_moov` tuning and still risks an unreadable tail moof;
plain MP4 segments lose the whole in-flight segment (moov written at close).
TS also concat-copies trivially (§5.3) and carries H.264 and H.265 alike.
Cost: ~2 % mux overhead and one remux at clip-extraction time — accepted.

10 s segments (Frigate-equivalent) keep event-clip latency low: the tail
segment needed for a clip closes ≤10 s after the event window ends. Volume:
8,640 files/day/cam — fine on ext4/xfs with the day/hour directory fan-out.

### 5.2 Storage layout + retention + disk math

```
/media/native/
  recordings/{camera}/{YYYY-MM-DD}/{HH}/{MM.SS}.ts   # 24/7, continuous_days
  clips/{event_id}.mp4                                # event clips, event_days
```

(`./media` host dir is already the media volume in compose; native mounts it
at `/media` in the backend container.)

Retention task (extend the existing `_prune_loop` cadence with an hourly
native pass): delete recording **hour-directories** whose newest mtime <
`now - continuous_days*86400` (then empty day dirs); delete
`clips/{id}.mp4` older than `event_days` (the existing event-row pruner
already removes rows + snapshots; add the clip file to its unlink list).
Plus a **low-disk guard**: if the media filesystem has <5 GB free, prune the
oldest recording hours regardless of retention and log loudly.

Disk math (bitrate is per-camera main-stream config; Amcrest defaults):

| Camera | Main stream | Typical bitrate |
|---|---|---|
| IP5M-T1277EW-AI | 2688×1520 H.265 | ~4 Mbps |
| IP8M-2779EW-AI | 3840×2160 H.265 | ~6–8 Mbps |
| AD410 | 1920×1080 H.264 | ~2 Mbps |

Total ≈ 12–14 Mbps ≈ **1.6 MB/s ≈ 135 GB/day ≈ 0.95 TB/week**. Formula for
docs: `GB/day = Σ bitrate_Mbps × 10.8`. Default `continuous_days: 7` needs
~1 TB; surface this in the Recording tab copy.

### 5.3 Event clip extraction

Scheduled by the recorder **20 s after `type:"end"`** (guarantees the segment
covering `end+5 s` has closed; 10 s segment + margin). Window:
`[start_time - 5, end_time + 5]`.

1. Select segments: parse timestamps from filenames within
   `[window_start - 10, window_end]` for that camera (a segment starting up
   to 10 s before the window can still contain its head), sorted.
2. Write a concat list file and cut with stream copy:

```
# /tmp/clip-{id}.txt:   file '/media/native/recordings/cam/2026-07-06/14/03.20.ts' ...
ffmpeg -hide_banner -loglevel warning -nostdin
  -f concat -safe 0 -i /tmp/clip-{id}.txt
  -ss {window_start - first_segment_start} -t {window_end - window_start}
  -c copy -movflags +faststart
  /media/native/clips/{event_id}.mp4
```

`-ss` after concat input with `-c copy` cuts at the previous keyframe —
sub-second slop, same tradeoff Frigate makes. On success: DB
`update_event(id, has_clip=True)` (idempotent) if it wasn't already set.
H.265 in MP4 plays in Safari/Chrome ≥ 2025 releases; document that H.265
cameras + old browsers may need the camera set to H.264 (existing
cameras-amcrest.md caveat applies).

Failure (missing segments — recorder was down): log, leave `has_clip` false →
UI shows no clip, `/clip.mp4` 404s. Never retry-loop.

### 5.4 API integration (same paths, native branch)

`routers/events.py` `event_clip()` gets a mode branch at the top:

```python
if request.app.state.config.is_native:
    path = MEDIA_DIR / "clips" / f"{event_id}.mp4"
    if not path.is_file():
        raise HTTPException(404, "Clip not available")
    return FileResponse(path, media_type="video/mp4")   # Starlette ≥0.38 serves Range natively
```

(Starlette's `FileResponse` has handled `Range` since 0.38 — current FastAPI
pins well above; verify `starlette.__version__` in the smoke test.)
Snapshot route is untouched (files under `/data/snapshots/` as today).
`routers/cameras.py` `camera_snapshot` is untouched — `latest_jpg` comes from
the provider, CGI fallback already exists.

---

## 6. Live view — Vigilume-owned go2rtc

### 6.1 Service

`alexxit/go2rtc:1.9.14` (pinned) as compose service `go2rtc` under the
`native` deployment (§7.4). Ports: `8554` (RTSP restream, host-exposed for
VLC parity with bundled mode), `8555/tcp + 8555/udp` (WebRTC), `1984`
(HTTP API, **internal only** — nginx proxies it). Volume:
`./go2rtc/config:/config` (backend mounts the same dir at `/go2rtc-config`).

`web` container env: `GO2RTC_UPSTREAM=http://go2rtc:1984` — the nginx
`/go2rtc/` proxy is already env-driven (CONTRACTS), **zero nginx changes**.
The frontend's `video-rtc.js` negotiates WebRTC over the same
`/go2rtc/api/ws?src=<name>` proxy (mode order already
`webrtc,mse,hls,mjpeg` per the contract addendum) and falls back to MSE when
ICE fails.

### 6.2 Generated config (`native/go2rtc_config.py` → `/go2rtc-config/go2rtc.yaml`)

Rendered from the camera DB (same trigger points as `frigate_config.regenerate`
today: boot + camera CRUD + relevant settings changes):

```yaml
api:
  listen: ":1984"
rtsp:
  listen: ":8554"
webrtc:
  listen: ":8555"
  candidates:
    - {lan_ip}:8555            # settings.system.webrtc_candidates, see below
    - {tailscale_ip}:8555      # optional entry
    - stun:8555                # resolve public IP via STUN (for completeness)
streams:
  {name}:                      # main
    - {main_url}               # rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0
    - "ffmpeg:{name}#audio=aac"   # only when the camera speaks G.711 (all three Amcrest models)
  {name}_sub:
    - {sub_url}                # …subtype=1
```

`main_url`/`sub_url` come from the new per-camera override columns, defaulting
to the Amcrest URLs built from `ip`+creds (§7.2). Credentials are
percent-encoded into the URL. The `#audio=aac` ffmpeg source gives every
consumer (record §5.1, MSE, WebRTC) AAC audio — same trick as the bundled
Frigate template.

Change application: write the YAML, then `POST /go2rtc api /api/restart`
falling back to per-stream `PUT /api/streams?name={n}&src={url}` (both from
go2rtc's documented API; **verify exact endpoints against the 1.9.14 README
at build time** — if the restart endpoint differs, a `docker restart` note in
docs is the fallback). Config writes are idempotent (skip when content
unchanged, like `frigate_config.py`).

### 6.3 WebRTC that actually works (LAN + Tailscale)

go2rtc's WebRTC needs **reachable ICE candidates**; inside Docker it can't
guess host IPs, so candidates are explicit:

- New setting `system.webrtc_candidates: string[]` (default `[]`), UI in the
  System tab: "WebRTC addresses — add this server's LAN IP and (if used)
  Tailscale IP, port 8555" with an autodetect hint listing the host's IPs
  (backend best-effort via the default-route socket trick). Entries are
  `ip:8555`; the generator passes them through verbatim and always appends
  `stun:8555`.
- LAN: `192.168.x.y:8555/tcp+udp` published by compose → sub-second latency.
- **Tailscale**: add the tailnet IP (`100.a.b.c:8555`). WebRTC media then
  flows over the tailnet (UDP). This is the documented go2rtc pattern for
  VPN/overlay networks — candidates are offered to the browser and ICE picks
  the one that connects. No TURN server needed inside a tailnet.
- If no candidate is reachable (e.g. phone on LTE hitting a Caddy-exposed
  HTTPS origin without the tailnet), ICE fails and `video-rtc.js` falls back
  to MSE over the existing WebSocket proxy automatically — live view never
  breaks, it just isn't sub-second. Document in remote-access.md.

---

## 7. Config surface, compose, migration

### 7.1 DB migration (SCHEMA_VERSION — coordinate the slot)

The groups/ordering pass (contract addendum) claims the next version(s) for
`camera_groups` + `position`. Native mode takes **the next free version after
whatever is merged first** (implementation detail, pattern per `db.py`):

```sql
ALTER TABLE cameras ADD COLUMN main_url TEXT NOT NULL DEFAULT '';
ALTER TABLE cameras ADD COLUMN sub_url  TEXT NOT NULL DEFAULT '';
```

Empty string = "derive Amcrest default from ip+creds" (`subtype=0`/`1`, the
CONTRACTS URLs). `camera_row_to_dict` gains both keys; `_CAMERA_INSERT_SQL`
and the upsert add both columns.

### 7.2 Per-camera fields (native additions, reuse everywhere else)

| Field | Status |
|---|---|
| `detect_fps` | already exists (default 5) — becomes user-visible in native camera modal (1–10) |
| `detect_enabled` | already exists — gates ingest+detection per camera |
| `record_enabled` | already exists — gates the recorder process |
| `detect_width/height` | already exist — detect-space dims for box scaling (Amcrest sub defaults per model) |
| `main_url`, `sub_url` | **new** — optional RTSP overrides (non-Amcrest cameras, odd ports). Blank = Amcrest default |

`GET /api/cameras` additionally returns `main_url`/`sub_url` (empty strings
when default) in native mode; POST/PUT accept them (optional, 0–512 chars,
must start with `rtsp://` when non-empty).

### 7.3 Settings additions

```json
"detection": {
  "audio_events": true,          // existing; HIDDEN in native UI (no audio classifier in v1)
  "audio_labels": [...],         // existing; inert in native
  "model": "dfine_s",            // native: dfine_n | dfine_s | dfine_m
  "confidence": 0.5              // native: decode threshold, 0.2–0.9
},
"system": {
  "public_url": "",
  "webrtc_candidates": []        // native: ["192.168.1.10:8555", "100.64.0.7:8555"]
}
```

`SettingsStore` merges over defaults, so old `/data` loads cleanly (same
silent-tolerance rule as the ntfy removal). In native mode `PUT
/api/settings` **accepts** `recording`/`detection` again (the external-mode
400 stays external-only); a `detection.model` change triggers model download
(if absent) + engine reload; `recording` changes only affect the pruner;
`webrtc_candidates` changes regenerate go2rtc config.

### 7.4 Compose (native deployment)

`SENTINEL_MODES` becomes `("bundled", "external", "native")` (config.py; add
`is_native`). MQTT is not started at all in native (`main.py` skips
`MqttListener`; the event bus is in-process). No frigate, no mosquitto.

Compose shape — an **override file**, because the base `backend` service
can't conditionally gain a GPU reservation via profiles:

```yaml
# docker-compose.native.yml  (docker compose -f docker-compose.yml -f docker-compose.native.yml up -d)
services:
  backend:
    build: { context: ./backend, target: runtime-cuda }
    environment:
      - VIGILUME_REQUIRE_GPU=${VIGILUME_REQUIRE_GPU:-1}
    volumes:
      - ./data:/data
      - ${MEDIA_PATH:-./media}:/media
      - ./go2rtc/config:/go2rtc-config
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
  go2rtc:
    image: alexxit/go2rtc:1.9.14
    container_name: vigilume-go2rtc
    restart: unless-stopped
    volumes: [ ./go2rtc/config:/config ]
    ports: [ "8554:8554", "8555:8555/tcp", "8555:8555/udp" ]
    depends_on: { backend: { condition: service_healthy } }
  web:
    environment: [ GO2RTC_UPSTREAM=http://go2rtc:1984 ]
```

`.env`: `SENTINEL_MODE=native`, `COMPOSE_PROFILES=` (empty; `tls` still
composable). `FRIGATE_URL`/`MQTT_*` are ignored in native.

### 7.5 Model bootstrap

At native startup, `model_store.ensure(model_key)`:
1. `/data/models/{key}.onnx` exists and sidecar sha matches pin → done.
2. Else download from the pinned revision URL (httpx, streamed, 120 s
   timeout, `.part` file), verify SHA-256, atomic rename, write sidecar.
3. Failure → retry with backoff; the app stays healthy (health reports
   `detector: "downloading"`/`"error"`), detection starts when the model
   lands. Offline installs: drop the file in place manually (docs note +
   the sha to check).

### 7.6 Migration path for THE user (external @ 192.168.1.253 → native)

Cameras are already imported and credentialed (credentials card), so the DB
is ready — the flip is config-only:

1. `docker compose down` (Vigilume only; the old Frigate keeps running until
   you're satisfied, they don't conflict — Vigilume's go2rtc binds 8554/8555
   on the *Vigilume* host only. If Vigilume runs on the same host as the old
   Frigate, stop Frigate first or the ports collide).
2. Edit `.env`: `SENTINEL_MODE=native` (keep `COMPOSE_PROFILES=` empty;
   `FRIGATE_URL`, `GO2RTC_UPSTREAM` overrides can be deleted).
3. `docker compose -f docker-compose.yml -f docker-compose.native.yml build backend`
   then `… up -d`.
4. First boot: model downloads (~40 MB), GPU self-test logs (§8.2), go2rtc
   config generated from the stored cameras, recording starts immediately.
5. Verify: `GET /api/system/health` → `mode:"native"`; `/api/system/detector`
   → `provider:"CUDAExecutionProvider"`; walk in front of a camera → event
   with snapshot + push; wait ~30 s after event end → clip plays.
6. Caveats to document: event history carries over, but **old events' clips
   were Frigate's** — they 404 once Frigate is off (snapshots, stored in
   `/data/snapshots`, all survive). Recording history starts empty.
   `needs_credentials` cameras must be fixed *before* the flip (native needs
   RTSP creds for go2rtc; the import stored empty creds when Frigate
   redacted them).

Rollback = revert `.env`, `docker compose up -d` with the old files.

---

## 8. Risks & GPU-first-boot verification

### 8.1 What CANNOT be verified on the dev Mac (no NVIDIA GPU)

| Unverifiable locally | Mitigation |
|---|---|
| CUDA EP actually loads (wheel ↔ cuDNN 9 ↔ driver triplet) | §8.2 self-test + `VIGILUME_REQUIRE_GPU=1`; base image and wheel pins are the ORT-documented pairing |
| Real GPU per-frame latency | budget has 3× headroom over the worst estimate; detector endpoint reports measured `avg_infer_ms` |
| NVDEC/hwaccel | intentionally unused (§3.3) |
| go2rtc WebRTC through 8555 on the LAN/tailnet | MSE fallback is automatic; candidates are config, not code |
| Long-run recorder behavior on real disks | TS format bounds corruption; watchdogs bound outages; smoke test covers command construction, not weeks of runtime |

What IS verified on the Mac already (this doc's research): model artifact
URLs + hashes, ONNX graph I/O, preprocessing constants, decode correctness on
a real image, CPU EP latency, supervision/trackers exact APIs (including the
broken `trackers` 2.5.0 wheel and the `sv.ByteTrack` 0.30 removal), CUDA
image tags, go2rtc release/config keys.

### 8.2 First-boot-on-GPU verification structure

**Startup GPU check (blocking, before camera loops start):**
1. Create the ORT session; read `sess.get_providers()[0]`.
2. Run 3 warm-up inferences on a zero tensor; time the 3rd.
3. Log exactly one of:
   - `INFO  native.detector: GPU OK — provider=CUDAExecutionProvider model=dfine_s warmup_ms=<x> infer_ms=<y>`
   - `ERROR native.detector: GPU UNAVAILABLE — falling back to CPU (infer_ms=<y>); set VIGILUME_REQUIRE_GPU=0 to accept, see docs/native-mode.md#gpu` → and if `VIGILUME_REQUIRE_GPU=1` (native default) **exit non-zero** so compose surfaces the failure instead of silently cooking a CPU.

**Self-test endpoint** `GET /api/system/detector` (auth):

```json
{
  "provider": "CUDAExecutionProvider",         // or CPUExecutionProvider
  "model": "dfine_s", "model_sha256": "cd8a…", "model_state": "ready",  // downloading|ready|error
  "warmup_ms": 812, "avg_infer_ms": 9.4,       // rolling 60 s
  "frames": {"front_door": {"fps": 5.0, "last_frame_age_s": 0.2}, ...},
  "recorder": {"front_door": {"recording": true, "last_segment_age_s": 4}, ...},
  "queue_depth": 0, "dropped_frames_60s": 0
}
```

`GET /api/system/health` in native: `mode:"native"`, `frigate: true` ⇔ engine
task running (key kept for frontend compat), `mqtt: true` constant (no broker
exists; document). SystemTab shows the detector block in native mode.

**Frame-ingest self-heal (implemented).** The detection path is purely frame-driven with
a single inference worker, so it self-heals rather than silently wedging (see
[CONTRACTS.md → Frame ingest → Self-healing](CONTRACTS.md#frame-ingest) for the contract):

- Worker **heartbeat** — the worker's wake wait has a 1 s timeout (`HEARTBEAT_TICK_S`);
  a source with no fresh frame is ticked with **empty** observations so open events end by
  absence during an ingest stall, without re-running inference (no busy-spin).
- Per-inference **timeout** — `detect()` runs under `asyncio.wait_for` (8 s,
  `DETECT_TIMEOUT_S`); a timeout/exception yields empty observations and increments the
  detector's failure counter without blocking the worker.
- Detector **auto-reinit** — 3 consecutive failures (`DETECT_FAILURE_THRESHOLD`) flag the
  detector; the ingest supervisor calls `OnnxDetector.reinit()`, which rebuilds the ORT
  session in the background, cooldown-guarded (`REINIT_COOLDOWN_S` ≥ 20 s), recovering a
  CUDA dropout (`ready:false`) with no container restart.
- **Starvation** WARNING — a source with no frame for > 90 s across respawns logs the
  camera/go2rtc sub-stream as the culprit; `/api/system/detector` exposes per-camera
  `stalled`/`respawns` + detector `consecutive_failures`/`last_reinit_age_s`.

**go2rtc RTSP transport** was researched and **left unchanged**: go2rtc's native RTSP
client is already TCP (interleaved) with built-in `OPTIONS` keepalive (verified in
`pkg/rtsp/conn.go`); there is no `#transport=tcp` forcing keyword (`transport` selects
WebSocket), and forcing it risks breaking the working config for zero benefit. The only
safe lever is the `#timeout=N` fragment, deferred in favour of the self-heal nets above.

### 8.3 Other risks

- **`trackers` 2.5.0 PyPI wheel is empty** — the pin MUST be `==2.4.0`; add
  an import-smoke assertion (`ByteTrackTracker` importable).
- **supervision 0.30** will delete `sv.ByteTrack`; we never use it, and the
  `<0.30` cap protects `annotate.py` from unrelated churn.
- **onnx-community hosting**: not the model authors (it's the HF/transformers.js
  org). Hash-pinned downloads make tampering detectable; the sidecar makes
  mirroring trivial (docs: how to self-host the file). YOLOX documented as
  the first-party-hosted fallback detector (§1.1).
- **ORT CUDA-12 line is deprecated** (1.27 is the last comfortable pin);
  CUDA-13 migration (base image `13.0.1-cudnn-runtime-ubuntu24.04` + ORT
  CUDA-13 index + R580 driver) is a known, contained follow-up.
- **Amcrest concurrent-RTSP limits**: mitigated by restream fan-out (§3.2);
  if go2rtc flaps, everything reconnects (watchdogs).
- **One event per (camera,label)**: two people arriving/leaving in an
  overlapping window merge into one event with count=2 — same UX as today's
  count model; per-track events are a future option.
- **No zones / audio classification in v1**: fields stay `[]`/inert; the UI
  already hides audio config outside bundled mode.

---

## 9. File-by-file implementation plan

New package `backend/app/native/` (nothing under it imports in
bundled/external mode except types):

| File | Contents | ~LOC |
|---|---|---|
| `native/coco_labels.py` | 80-label tuple, `LABEL_TO_ID`, `ID_TO_LABEL` | 40 |
| `native/model_store.py` | `MODELS` pin table (url/sha256/size per §1.2), `async ensure(key) -> Path`, hash verify, atomic download | 120 |
| `native/detector.py` | `OnnxDetector`: session init w/ provider selection + `VIGILUME_REQUIRE_GPU`, warm-up timing, `detect(frame_bgr, dw, dh) -> sv.Detections` (preprocess §1.3 + decode; runs in the inference worker thread) | 150 |
| `native/ingest.py` | `FrameSource`: ffmpeg child (§3.1 args), `readexactly` loop, latest-frame slot, staleness watchdog, backoff respawn | 160 |
| `native/tracking.py` | Per-camera `ByteTrackTracker` + `EventStateMachine` (§4.2): confirm/update/end, best-frame keeper (§4.3), payload synthesis (§4.1) | 220 |
| `native/engine.py` | `DetectionEngine`: owns sources + single inference worker + state machines; calls `pipeline.handle_event` / `update_count`; `reload(cameras, settings)`; exposes `latest_frame(camera)`, `event_best_jpeg(fid)`, stats for the detector endpoint | 220 |
| `native/media.py` | `MediaProvider` Protocol + `NativeMediaProvider` (§4.4) | 100 |
| `native/recorder.py` | `Recorder`: per-camera segment ffmpeg (§5.1), dir rollover, watchdog, retention + low-disk prune (§5.2), `extract_clip(event)` (§5.3) scheduled off pipeline `event_end` WS-adjacent hook (engine calls it directly after emitting `end`) | 260 |
| `native/go2rtc_config.py` | YAML generation (§6.2), idempotent write, restart/PUT sync | 120 |

Touched existing files (surgical):

| File | Change |
|---|---|
| `config.py` | `SENTINEL_MODES += ("native",)`, `is_native`, `MEDIA_DIR`/`GO2RTC_CONFIG_DIR` envs, `VIGILUME_REQUIRE_GPU`, `DEFAULT_SETTINGS` additions (§7.3) |
| `db.py` | schema bump: `main_url`/`sub_url` (§7.1), row dict + insert/upsert columns |
| `main.py` | native branch in lifespan: skip MQTT + Frigate config regen; `model_store.ensure` → `OnnxDetector` → `DetectionEngine` → `Recorder` → go2rtc config write; inject `NativeMediaProvider` as `app.state.frigate`-replacement (`app.state.media`, with `app.state.frigate = media` alias so routers/pipeline are untouched); shutdown ordering: engine → recorder → provider |
| `events_pipeline.py` | **no changes** (constructor already takes the client; type it as `MediaProvider`) |
| `routers/events.py` | native branch in `event_clip` (§5.4) |
| `routers/cameras.py` | accept/return `main_url`/`sub_url`; on CRUD in native: regenerate go2rtc config + `engine.reload()` + `recorder.reload()` instead of Frigate regen; `POST /api/cameras/import` → 400 `"Import is only available in external mode"` (existing bundled message generalizes) |
| `routers/settings.py` | native: allow recording/detection; model/confidence/webrtc_candidates side effects (§7.3) |
| `routers/system.py` | health native fields; new `GET /api/system/detector` |
| `backend/Dockerfile` | multi-stage `runtime` / `runtime-cuda` targets; apt `ffmpeg` in cuda target; `requirements-native.txt` |
| `docker-compose.native.yml` | new (§7.4) |
| frontend `api.ts` | types: `mode: 'native'`, camera `main_url/sub_url`, settings additions, detector-status type |
| frontend `SystemTab.tsx` | detector status card (native) + webrtc_candidates editor |
| frontend `CamerasTab.tsx` | native: show detect_fps + URL-override fields in the modal (advanced collapsible) |
| frontend `RecordingTab.tsx` | native: re-enabled (it's hidden in external); disk-usage note w/ §5.2 math |
| docs | new `docs/native-mode.md` (operator guide distilled from this design), CONTRACTS addendum for the API deltas, setup-nvidia.md native section, faq/remote-access notes (§6.3) |

**API contract deltas (summary for CONTRACTS.md):**
- `SENTINEL_MODE` gains `native`; health `mode:"native"`, `frigate` = engine-up, `mqtt: true` const.
- NEW `GET /api/system/detector` (shape §8.2).
- `GET/POST/PUT /api/cameras`: `main_url`, `sub_url` (native-meaningful, stored in all modes).
- `PUT /api/settings`: recording/detection editable in native; `detection.model|confidence`, `system.webrtc_candidates` added.
- `POST /api/cameras/import` → 400 in native.
- Media routes: identical paths/semantics, native backing (`clip.mp4` served from `/media/native/clips`).
- WS messages: unchanged.

## 10. Test plan

**Durable backend tests** (extend `backend/tests/`, same style as
`import_smoke.py` — must run on the Mac venv, CPU-only, no network, no GPU):

1. `native_smoke.py` (new, target ≥60 checks):
   - decode math: golden `logits`/`pred_boxes` fixture (small `.npz` generated
     once from the real model, committed) → expected labels/boxes/confidences;
     threshold edge cases; empty-frame → empty `sv.Detections`.
   - preprocessing: known 704×480 input → shape/dtype/range assertions, BGR→RGB
     order (checkerboard fixture).
   - state machine: scripted detection sequences → assert exact
     `handle_event` payload stream (new after 3 hits; update on count/score
     change; end after 25 absent frames; occlusion gap < buffer keeps the
     event open); `update_count` call sequence.
   - payload compatibility: every synthesized payload round-trips through the
     real `EventsPipeline` handlers against a temp DB (pipeline is already
     testable with a fake media provider).
   - recorder: segment-filename parser, clip window → concat-list + ffmpeg
     argv construction (golden argv, no ffmpeg exec), retention selection on
     a synthetic dir tree, low-disk pruning order.
   - go2rtc config: camera rows → golden YAML (creds encoding, audio=aac
     inclusion, overrides win, candidates injection); idempotent-write check.
   - model_store: sha mismatch → reject + retry path (local file fixtures).
   - media provider: `event_snapshot`/`latest_jpg`/`detect_dims` against a
     stub engine; `build_clip_request` raises.
   - imports: `ByteTrackTracker` importable, `supervision.__version__` <0.30,
     `starlette` FileResponse Range support present.
2. `import_smoke.py` / `controls_smoke.py`: stay green (native code must not
   import GPU-only deps at module import time — keep `onnxruntime` import
   inside `detector.py` functions/class init).
3. Optional local live test (manual, Mac): `SENTINEL_MODE=native` uvicorn run
   with `dfine_n`, one RTSP file-loop source via local ffmpeg — validates
   ingest→detect→event→snapshot end-to-end on CPU.

**On-GPU first-boot checklist** (docs/native-mode.md, executed on the NVR):
`nvidia-smi` in backend container → detector log line `GPU OK` →
`/api/system/detector` provider/latency → live WebRTC (LAN + tailnet) →
walk-test event: push arrives with annotated snapshot, count correct →
clip playable ~30 s after end → pull power on a camera: ingest/record logs
show backoff + recovery → reboot host: everything resumes, segments before
the cut play.

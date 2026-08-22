# Vigilume NVR backend — standalone NVR brain (detection + recording + API).
#
# Two final targets:
#   gpu (DEFAULT, used by docker-compose.yml): MINIMAL nvidia/cuda:*-base image
#       + onnxruntime-gpu==1.27.0 (CUDA-12 build) with the CUDA 12 / cuDNN 9
#       runtime supplied as pinned pip nvidia-*-cu12 wheels on LD_LIBRARY_PATH,
#       apt ffmpeg for ingest/recording (~5 GB, vs ~9–10 GB on the old
#       -cudnn-runtime base). Requires the NVIDIA container toolkit on the host.
#   cpu (dev/tests ONLY): python slim + CPU onnxruntime. Never deploy this
#       for production detection — build it explicitly with
#       `docker build --target cpu` when you need a GPU-less stack.
#
# NOTE on OpenCV: supervision/trackers depend on the GUI OpenCV build, so
# both targets install libgl1 + libglib2.0-0 (tiny next to ffmpeg/CUDA).

# ---------------------------------------------------------------- cpu (dev)
FROM python:3.12-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# mesa-va-drivers + libva-drm2 supply the VAAPI backend for AMD/Intel iGPU
# transcoding; see the gpu target below for the full rationale. Present in both
# targets so a Coral box (which has no reason to run the CUDA image) still gets
# hardware HEVC->H.264 instead of falling back to libx264.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg \
        mesa-va-drivers libva-drm2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-cpu.txt ./
RUN pip install -r requirements.txt -r requirements-cpu.txt

# ---- OPTIONAL Coral Edge TPU runtime (settings.detection.backend="coral") ----
#
# TWO pieces are needed and BOTH must be in the image:
#   1. ai-edge-litert  — the Python runtime.
#   2. libedgetpu.so.1 — the .so that load_delegate() opens.
# Only the DEVICE NODE (/dev/apex_0) comes from the host, via the optional
# CORAL_DEVICE passthrough in docker-compose.yml. A shared library does NOT
# cross the container boundary.
#
# In its OWN layer, last, ON PURPOSE: ai-edge-litert is deliberately NOT in
# requirements.txt, because touching that file invalidates the COPY above it and
# with it the ~5 GB CUDA layer, forcing a full re-download on every rebuild
# (this is what filled docker.img once already). Here it costs ~17 MB + ~1 MB.
#
# Installed UNCONDITIONALLY so switching backends is a settings change plus a
# restart, not a rebuild. Nothing is imported at module load; app/native/coral.py
# only touches litert inside _make_interpreter.
#
# This target is python:3.12-slim => Debian BOOKWORM.
#
# dpkg-deb -x, NOT apt install: the .deb declares "Depends: libgcc1", a package
# that no longer exists (it is libgcc-s1), so apt refuses. Extraction bypasses
# dpkg's dep check; the .so's real ELF NEEDED list (libusb-1.0, libstdc++, libm,
# libgcc_s, libc) is satisfied by the apt line below.
#
# -std, not -max: -max ships a preinst raising a critical debconf EULA prompt,
# which aborts a non-interactive build.
#
# The trailing `ldconfig -p | grep` FAILS THE BUILD if the library did not land,
# rather than shipping an image where choosing Coral dies at runtime.
#
# Inert with no Coral fitted: the library loads, finds no device, and
# CoralDetector logs "CORAL UNAVAILABLE" instead of starting.
RUN pip install "ai-edge-litert>=2.0.2"
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates libusb-1.0-0 \
 && curl -fL -o /tmp/libedgetpu.deb \
      https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.bookworm_amd64.deb \
 && dpkg-deb -x /tmp/libedgetpu.deb / \
 && ldconfig \
 && rm -f /tmp/libedgetpu.deb \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig -p | grep -q libedgetpu.so.1

COPY app ./app

EXPOSE 8000
# --proxy-headers: requests only ever arrive via the nginx "web" container
# (port 8000 is not published to the host), so trust its X-Forwarded-For —
# otherwise the login rate limiter keys every client to the proxy's IP.
# --forwarded-allow-ips: the Docker bridge ONLY, never "*". With "*" uvicorn's
# _TrustedHosts enters always_trust mode and returns the LEFTMOST X-Forwarded-For
# entry — which nginx's $proxy_add_x_forwarded_for prepends straight from the
# client. So request.client.host was whatever the caller typed, and the login
# rate limiter (keyed on it) could be reset every request by rotating a header.
# Narrowed, uvicorn walks the XFF chain right-to-left and stops at the first host
# OUTSIDE the trusted set — discarding the client-supplied prefix.
#
# /12 not the box's exact /16 (172.18.0.0/16 as of 2026-07-20): compose recreating
# the network can renumber it, and 172.16.0.0/12 covers Docker's whole default
# pool. Everything in it is on this host either way.
#
# --timeout-graceful-shutdown bounds uvicorn's CONNECTION-DRAIN phase, which runs
# BEFORE the app's lifespan shutdown and is otherwise unbounded. Without it a
# client slowly pulling a 30-minute export holds the whole process in shutdown
# indefinitely. The lifespan teardown that follows has its own separate deadline
# (routers/system.py _TEARDOWN_DEADLINE_S).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "172.16.0.0/12", "--timeout-graceful-shutdown", "10"]

# ------------------------------------------------------------ gpu (default)
# SLIM CUDA image (~5 GB, down from ~9–10 GB): a MINIMAL nvidia/cuda
# *base* image instead of the multi-GB `-cudnn-runtime` base. The `-base`
# tag is ~200 MB and carries only the driver-injection env
# (NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES=compute,utility, so the
# host container-toolkit still mounts libcuda.so.1) plus the CUDA
# forward-compat layer — NOT the CUDA runtime or cuDNN. The exact CUDA 12
# runtime libs + cuDNN 9 that onnxruntime-gpu's CUDA Execution Provider
# dlopens are supplied as pinned pip `nvidia-*-cu12` wheels
# (requirements-gpu.txt) and exposed via LD_LIBRARY_PATH below. python:slim
# was rejected: it does not set NVIDIA_DRIVER_CAPABILITIES, so the toolkit
# would not inject the driver.
#
# CUDA 12 vs 13: onnxruntime-gpu 1.27.0's DEFAULT PyPI wheel is now CUDA 13
# (CUDA 12 was deprecated in ORT 1.27; the cu13 component wheels are not yet
# cleanly on public PyPI). We deliberately keep CUDA 12.9 + cuDNN 9 — matching
# the R550+ host driver this stack documents — by installing the CUDA-12 build
# of the SAME 1.27.0 release from Microsoft's official onnxruntime-cuda-12
# package feed (the --no-deps step below). Ubuntu 24.04 glibc 2.39 satisfies
# the manylinux_2_27/2_28 wheels; ORT 1.27 ships cp312 wheels.
FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# mesa-va-drivers + libva-drm2: the VAAPI userspace for an AMD (VCN) or Intel
# (Quick Sync) iGPU. Ubuntu's ffmpeg is already built with VAAPI, but without
# the Mesa driver it finds no VA backend and h264_vaapi cannot initialize, so
# HEVC->H.264 falls back to CPU libx264. Only useful when a render node is also
# passed through (VAAPI_DEVICE in docker-compose.yml); inert otherwise, and it
# costs ~30 MB against a ~5 GB CUDA base. NVIDIA boxes are unaffected — NVENC
# is selected ahead of VAAPI (app/native/transcode.py select_encoder).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        ffmpeg libglib2.0-0 libgl1 \
        mesa-va-drivers libva-drm2 \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ORT's CUDA EP resolves the pip-provided CUDA 12 / cuDNN 9 runtime libs here.
# One dir per pinned nvidia-*-cu12 wheel (site-packages/nvidia/<comp>/lib).
# Keep in sync with requirements-gpu.txt; the sanity check below fails the
# build if any listed .so is missing.
# The trailing ${LD_LIBRARY_PATH} PRESERVES the base image's driver-injection
# dirs (/usr/local/nvidia/lib[64]) — the NVIDIA Container Toolkit mounts the
# host driver (libcuda.so.1) there. On Unraid's classic (non-CDI) nvidia
# runtime that dir is how the driver is found; overwriting it would break
# GPU detection at run time.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

WORKDIR /srv
COPY requirements.txt requirements-gpu.txt ./

# Step 1 — the CUDA-12 build of onnxruntime-gpu 1.27.0 from Microsoft's
# official onnxruntime-cuda-12 feed (plain PyPI 1.27.0 is the CUDA-13 wheel).
# --no-deps: its deps (numpy + the pinned nvidia-*-cu12 wheels) install from
# PyPI in step 2. Installing ORT first means step 2 sees onnxruntime-gpu==1.27.0
# already satisfied and never pulls the CUDA-13 PyPI wheel.
RUN pip install --no-deps \
      --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      onnxruntime-gpu==1.27.0

# Step 2 — everything else from PyPI (shared deps + pinned CUDA 12 / cuDNN 9
# runtime wheels + detection stack).
RUN pip install -r requirements.txt -r requirements-gpu.txt

# Build-time insurance — we CANNOT run the GPU image on the Mac dev host, so
# fail the build now if any CUDA/cuDNN .so the CUDA EP loads is missing from
# LD_LIBRARY_PATH, or if this ORT build does not expose the CUDA provider.
# (libcuda.so.1 is host-injected at run time and is NOT needed to map these.)
RUN python - <<'PY'
import ctypes, os, sys
required = [
    "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9",
    "libcufft.so.11", "libcurand.so.10", "libnvrtc.so.12", "libnvJitLink.so.12",
    # cuDNN 9 is split: libcudnn.so.9 is a thin loader that dlopens these at
    # inference time — a missing sub-lib passes the stub check but fails on the
    # first detection, so assert them at build.
    "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9", "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9", "libcudnn_heuristic.so.9",
]
dirs = os.environ["LD_LIBRARY_PATH"].split(":")
missing = [so for so in required
           if not any(os.path.exists(os.path.join(d, so)) for d in dirs)]
if missing:
    sys.exit("FATAL: CUDA/cuDNN libs missing from LD_LIBRARY_PATH: %s" % missing)
# The three ORT hard-requires should also dlopen. Existence is already gated
# above, so FATAL only when the error names the target lib itself (a corrupt/
# ABI-broken wheel); a missing DEPENDENCY such as the host-injected
# libcuda.so.1 (absent at build time) names that dep, not the target, and is
# tolerated as a note.
for so in ("libcudart.so.12", "libcublas.so.12", "libcudnn.so.9"):
    try:
        ctypes.CDLL(so)
    except OSError as e:
        if so in str(e):
            sys.exit("FATAL: %s present on disk but not loadable: %s" % (so, e))
        print("note: %s load deferred (dependency absent at build, e.g. host libcuda.so.1): %s" % (so, e))
import onnxruntime as ort
if ort.__version__ != "1.27.0":
    sys.exit("FATAL: expected onnxruntime 1.27.0 (CUDA-12 feed build), got %s" % ort.__version__)
provs = ort.get_available_providers()
if "CUDAExecutionProvider" not in provs:
    sys.exit("FATAL: onnxruntime build lacks CUDAExecutionProvider: %s" % provs)
print("sanity OK — ort %s, CUDA/cuDNN libs present, CUDAExecutionProvider available: %s" % (ort.__version__, provs))
PY

# ---- OPTIONAL Coral Edge TPU runtime (settings.detection.backend="coral") ----
#
# TWO pieces are needed and BOTH must be in the image:
#   1. ai-edge-litert  — the Python runtime.
#   2. libedgetpu.so.1 — the .so that load_delegate() opens.
# Only the DEVICE NODE (/dev/apex_0) comes from the host, via the optional
# CORAL_DEVICE passthrough in docker-compose.yml. A shared library does NOT
# cross the container boundary.
#
# In its OWN layer, last, ON PURPOSE: ai-edge-litert is deliberately NOT in
# requirements.txt, because touching that file invalidates the COPY above it and
# with it the ~5 GB CUDA layer, forcing a full re-download on every rebuild
# (this is what filled docker.img once already). Here it costs ~17 MB + ~1 MB.
#
# Installed UNCONDITIONALLY so switching backends is a settings change plus a
# restart, not a rebuild. Nothing is imported at module load; app/native/coral.py
# only touches litert inside _make_interpreter.
#
# This target is nvidia/cuda:...-ubuntu24.04 => UBUNTU NOBLE.
#
# dpkg-deb -x, NOT apt install: the .deb declares "Depends: libgcc1", a package
# that no longer exists (it is libgcc-s1), so apt refuses. Extraction bypasses
# dpkg's dep check; the .so's real ELF NEEDED list (libusb-1.0, libstdc++, libm,
# libgcc_s, libc) is satisfied by the apt line below.
#
# -std, not -max: -max ships a preinst raising a critical debconf EULA prompt,
# which aborts a non-interactive build.
#
# The trailing `ldconfig -p | grep` FAILS THE BUILD if the library did not land,
# rather than shipping an image where choosing Coral dies at runtime.
#
# Inert with no Coral fitted: the library loads, finds no device, and
# CoralDetector logs "CORAL UNAVAILABLE" instead of starting.
RUN pip install "ai-edge-litert>=2.0.2"
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates libusb-1.0-0 \
 && curl -fL -o /tmp/libedgetpu.deb \
      https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.ubuntu24.04_amd64.deb \
 && dpkg-deb -x /tmp/libedgetpu.deb / \
 && ldconfig \
 && rm -f /tmp/libedgetpu.deb \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig -p | grep -q libedgetpu.so.1

COPY app ./app

EXPOSE 8000
# --forwarded-allow-ips: the Docker bridge ONLY, never "*". With "*" uvicorn's
# _TrustedHosts enters always_trust mode and returns the LEFTMOST X-Forwarded-For
# entry — which nginx's $proxy_add_x_forwarded_for prepends straight from the
# client. So request.client.host was whatever the caller typed, and the login
# rate limiter (keyed on it) could be reset every request by rotating a header.
# Narrowed, uvicorn walks the XFF chain right-to-left and stops at the first host
# OUTSIDE the trusted set — discarding the client-supplied prefix.
#
# /12 not the box's exact /16 (172.18.0.0/16 as of 2026-07-20): compose recreating
# the network can renumber it, and 172.16.0.0/12 covers Docker's whole default
# pool. Everything in it is on this host either way.
#
# --timeout-graceful-shutdown bounds uvicorn's CONNECTION-DRAIN phase, which runs
# BEFORE the app's lifespan shutdown and is otherwise unbounded. Without it a
# client slowly pulling a 30-minute export holds the whole process in shutdown
# indefinitely. The lifespan teardown that follows has its own separate deadline
# (routers/system.py _TEARDOWN_DEADLINE_S).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "172.16.0.0/12", "--timeout-graceful-shutdown", "10"]

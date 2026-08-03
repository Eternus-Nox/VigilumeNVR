"""OnnxDetector — D-FINE ONNX object detection (docs/native-mode-design.md §1–§2, §8.2).

Model artifacts are pinned in ``MODELS`` (revision-pinned onnx-community
URLs + SHA-256). ``start()`` ensures ``models_dir/{key}.onnx`` exists and
hashes to the pin (download to a ``.part`` file, verify, atomic rename,
``{key}.json`` sidecar with url/sha256/downloaded_at), then builds the ORT
session and runs a 3-inference warm-up. Failures log loudly, leave ``ready``
False and retry with backoff — the app never crashes over the detector.

GPU semantics (VIGILUME_REQUIRE_GPU):
- ``require_gpu`` True (compose default) and the CUDA EP is unavailable or
  ORT silently picked the CPU EP => ``ready`` stays False, ``device`` None,
  ERROR log ``GPU UNAVAILABLE — set VIGILUME_REQUIRE_GPU=0 to accept CPU``.
  The availability pre-check runs BEFORE any model download so a
  mis-deployed CPU host fails fast and cheap.
- ``require_gpu`` False: CPU inference is accepted (``device: "cpu"``,
  WARNING log with the measured per-frame cost).
- CUDA active: ``device: "cuda"``, INFO ``GPU OK — provider=... model=...``.

``import onnxruntime`` happens ONLY inside methods — CPU test hosts and the
web-only failure mode must import this module fine.

Threading model: ``detect()`` is a blocking call invoked from exactly one
place (the engine's single inference worker, via ``asyncio.to_thread``).
``start()``/``stop()``/``reconfigure()`` run on the event loop and serialize
session (re)builds behind an ``asyncio.Lock``; the session swap itself is a
single attribute assignment, and an in-flight ``detect()`` keeps the old
session object alive via its local reference — no further locking needed.
Stats properties are plain attribute reads (cheap, thread-safe).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import cv2
import numpy as np
import supervision as sv

from ..config import env_dual
from .coco_labels import set_active_labelmap

if TYPE_CHECKING:  # pragma: no cover
    import httpx

    from .model_store import ModelStore

# progress(downloaded_bytes, total_bytes) — called while streaming a download.
ProgressFn = Callable[[int, int], None]

log = logging.getLogger(__name__)

# Input side of the pinned graphs (dynamic dims, but 640x640 is the trained
# resolution — see preprocessor_config.json notes in the design doc).
INPUT_SIZE = 640

_DOWNLOAD_TIMEOUT_S = 120.0
_RETRY_INITIAL_S = 2.0
_RETRY_CAP_S = 300.0
_WARMUP_RUNS = 3

# Self-heal: consecutive detect() timeouts/exceptions before the detector is
# flagged for a background session rebuild, and the minimum spacing between
# reinit attempts so a persistently-broken detector never thrashes the host.
DETECT_FAILURE_THRESHOLD = 3
REINIT_COOLDOWN_S = 20.0

# Revision-pinned, SHA-256-verified artifacts (design doc §1.2). Every SHA-256
# below was verified by a local download + shasum of the pinned-revision file.
# ``labelmap`` names an entry in coco_labels.LABELMAPS (the model's output
# vocabulary); all onnx-community D-FINE exports share the SAME graph I/O
# (``pixel_values`` -> ``logits``/``pred_boxes``) and preprocessing (640x640
# plain resize, /255, no normalize), so the NMS-free decode is identical across
# every tier — only the class-count of ``logits`` (80 vs 366) and the labelmap
# differ.
MODELS: dict[str, dict[str, Any]] = {
    "dfine_n": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_n_coco-ONNX/resolve/"
            "380d2839c327efaf65dd0fe0c2c10ab7fadd5473/onnx/model.onnx"
        ),
        "bytes": 15_258_358,
        "sha256": "0f684f409618ee8a822410e754a29caa817d1aa16283ce89cad936d0a48e2f35",
        "labelmap": "coco",
    },
    # Face recognition models (OpenCV Zoo, permissive licences). NOT detectors —
    # deliberately absent from TIER_ORDER so the model picker never lists them.
    "dfine_s": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_s_coco-ONNX/resolve/"
            "a3cf03147a9b86c78475139115c8ac142577352d/onnx/model.onnx"
        ),
        "bytes": 41_535_197,
        "sha256": "cd8a49a945feda6d28c6304ae8ae85c2759ba1d78a5a83a22c5ce8db82ef7238",
        "labelmap": "coco",
    },
    "dfine_m": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_m_coco-ONNX/resolve/"
            "489756db2825cb068a588fe930af239a656d1fe1/onnx/model.onnx"
        ),
        "bytes": 78_624_257,
        "sha256": "70aaa837978a06ba44ad17398c7079ae5a1a7b1a9032b5d7053981e1ada02d6b",
        "labelmap": "coco",
    },
    "dfine_l": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_l_coco-ONNX/resolve/"
            "e09b218185400510b68d067bbd0c9379d905ffcd/onnx/model.onnx"
        ),
        "bytes": 125_348_332,
        "sha256": "d678f3baebfb909d3a20f21d1d807544d0172ed47fa1ab88e7fcdec7e365b236",
        "labelmap": "coco",
    },
    "dfine_x": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_x_coco-ONNX/resolve/"
            "b67b16d98f1f95e2af0b6d0b8f74ad956557fe8c/onnx/model.onnx"
        ),
        "bytes": 251_138_448,
        "sha256": "644fb5124c9c035a6082f23419da693c19ac857bd984ab7af5e353779368a03b",
        "labelmap": "coco",
    },
    # Objects365 vocabulary (365 categories) — same D-FINE graph/decode, but
    # ``logits`` is [1, 300, 366] and the labelmap is obj365 (id 0 background).
    "dfine_l_obj365": {
        "url": (
            "https://huggingface.co/onnx-community/dfine_l_obj365-ONNX/resolve/"
            "32323e9a0c74b22803a2ad5b315a8fed61801e6a/onnx/model.onnx"
        ),
        "bytes": 125_936_360,
        "sha256": "cd0dfa92a2e0e2ab3d4a7c2e6252ebc094aa77b67f2f9dc1a010dd350a9a2f3e",
        "labelmap": "obj365",
    },
}

DEFAULT_MODEL = "dfine_s"


def model_labelmap(key: str) -> str:
    """Labelmap name for a model key (``"coco"`` for legacy pins missing the
    field, e.g. tests that patch MODELS with a bare fake pin)."""
    return MODELS[key].get("labelmap", "coco")


class ModelVerifyError(RuntimeError):
    """Downloaded/on-disk model artifact failed SHA-256 verification."""


# ---------- model store (download + SHA-256 pinning) ----------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(models_dir: Path, key: str) -> Path:
    return models_dir / f"{key}.onnx"


async def ensure_model(
    models_dir: Path,
    key: str,
    client: Optional["httpx.AsyncClient"] = None,
    progress: Optional["ProgressFn"] = None,
) -> Path:
    """Ensure the pinned artifact for ``key`` is on disk and hash-verified.

    Single attempt (callers own retry policy). Raises ``ModelVerifyError``
    on a hash mismatch (both for a corrupt existing file that then fails to
    re-download correctly, and for a corrupt download) and propagates
    network errors. On success the ``.onnx`` file is in place and a
    ``{key}.json`` sidecar records url/sha256/downloaded_at.

    ``progress`` (optional) is called as ``progress(downloaded_bytes,
    total_bytes)`` while streaming a fresh download — the ModelStore uses it
    to drive its per-key progress + state machine. It is NOT called for an
    already-present, still-valid file.
    """
    pin = MODELS[key]
    path = model_path(models_dir, key)
    if path.is_file():
        digest = await asyncio.to_thread(sha256_file, path)
        if digest == pin["sha256"]:
            return path
        log.warning(
            "model %s on disk hashes to %s… (pin %s…) — re-downloading",
            key, digest[:12], pin["sha256"][:12],
        )
        path.unlink(missing_ok=True)

    import httpx  # local import keeps module import light

    models_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".onnx.part")
    log.info("downloading detector model %s (%d bytes) from %s", key, pin["bytes"], pin["url"])
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT_S), follow_redirects=True)
    try:
        digest = hashlib.sha256()
        downloaded = 0
        total = int(pin["bytes"])
        with part.open("wb") as fh:
            async with client.stream("GET", pin["url"]) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(1024 * 256):
                    fh.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
        if digest.hexdigest() != pin["sha256"]:
            part.unlink(missing_ok=True)
            raise ModelVerifyError(
                f"model {key} download hashed to {digest.hexdigest()} — pin is {pin['sha256']}"
            )
        part.replace(path)
        sidecar = {
            "url": pin["url"],
            "sha256": pin["sha256"],
            "bytes": pin["bytes"],
            "downloaded_at": time.time(),
        }
        (models_dir / f"{key}.json").write_text(json.dumps(sidecar, indent=2))
        log.info("model %s downloaded and SHA-256 verified", key)
        return path
    finally:
        part.unlink(missing_ok=True)
        if own_client:
            await client.aclose()


# ---------- pure inference math (unit-tested in native_smoke) ----------


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> float32 [1, 3, 640, 640] tensor.

    Verified against the repo preprocessor_config.json: BGR->RGB, PLAIN
    resize to 640x640 (bilinear — no letterbox, no padding), /255.0, no
    mean/std normalization, HWC->CHW, batch dim.

    ONE FUSED OpenCV CALL, deliberately. The obvious spelling of this is four
    chained numpy/cv2 steps (cvtColor -> resize -> astype/255 -> transpose +
    ascontiguousarray), which allocates ~12 MB and makes four full passes over
    the image PER FRAME — including a strided, cache-hostile gather for the
    HWC->CHW transpose. At 12 cameras x 5 fps that is ~60 of those per second,
    all on the GIL-holding side of the to_thread hop, competing with the
    detection the box actually exists to do.

    blobFromImage implements exactly the contract above in a single C++ pass
    with one output allocation: swapRB=True is the BGR->RGB, size= is the plain
    bilinear resize, scalefactor is the /255, crop=False means no letterbox, and
    the NCHW layout is what it returns natively.

    Measured on this tree: 0.970 -> 0.403 ms/frame (2.4x) and ~12 MB/frame of
    allocation removed. The output is NOT bit-identical to the chained form —
    it differs by up to 5.96e-08 (one float32 ULP near 1.0) because the scale
    is folded into the resize rather than applied after it. That is ~1e-5 of a
    single 8-bit quantisation step, far below anything the detector resolves;
    native_smoke pins the tolerance so a real divergence would still fail.
    """
    return cv2.dnn.blobFromImage(
        frame_bgr,
        scalefactor=1.0 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False,
    )


def decode(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    confidence: float,
    detect_width: int,
    detect_height: int,
) -> sv.Detections:
    """NMS-free D-FINE decode (design doc §1.3), validated on a real image.

    logits [1, 300, C], pred_boxes [1, 300, 4] (cx, cy, w, h normalized)
    -> sv.Detections with float32 xyxy in detect-stream pixels, confidence,
    and contiguous class_id, thresholded at ``confidence`` (>=). ``C`` is the
    active model's class count (80 for COCO, 366 for Objects365); the math is
    class-count agnostic (per-query sigmoid + argmax), so no per-model branch
    is needed. class_id -> label happens downstream via the active labelmap.
    """
    scores = 1.0 / (1.0 + np.exp(-logits[0]))       # [300, 80] sigmoid
    conf = scores.max(axis=1)                        # best class score per query
    cls = scores.argmax(axis=1)
    keep = conf >= confidence
    if not np.any(keep):
        return sv.Detections.empty()
    cx, cy, w, h = pred_boxes[0][keep].T             # normalized cx,cy,w,h
    dw, dh = float(detect_width), float(detect_height)
    xyxy = np.stack(
        [(cx - w / 2) * dw, (cy - h / 2) * dh, (cx + w / 2) * dw, (cy + h / 2) * dh],
        axis=1,
    ).astype(np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=conf[keep].astype(np.float32),
        class_id=cls[keep].astype(int),
    )


class OnnxDetector:
    """D-FINE ONNX detector (see module docstring for the full contract)."""

    def __init__(
        self,
        models_dir: Path,
        model_key: str = DEFAULT_MODEL,
        confidence: float = 0.5,
        require_gpu: bool = True,
        force_cpu: bool = False,
        store: Optional["ModelStore"] = None,
    ):
        if model_key not in MODELS:
            log.warning("unknown detector model %r — falling back to %s", model_key, DEFAULT_MODEL)
            model_key = DEFAULT_MODEL
        self._models_dir = models_dir
        self._model_key = model_key
        self._confidence = confidence
        self._require_gpu = require_gpu
        # DISTINCT from ``require_gpu`` and the reason VIGILUME_DETECTOR=onnx_cpu
        # used to be a no-op on a CUDA box:
        #   require_gpu=False -> "CPU is ACCEPTABLE if CUDA is missing"
        #   force_cpu=True    -> "do not offer CUDA to ORT at all"
        # Only the second actually keeps inference off the GPU. Without it,
        # onnx_cpu still handed ORT a CUDA provider and ORT took it, so the
        # documented GPU-dropout fallback silently ran on the GPU it was meant
        # to stand in for.
        self._force_cpu = force_cpu
        # The ModelStore is the ONE downloader when present; without it (unit
        # tests constructing a bare detector) we fall back to ensure_model.
        self._store = store
        self._ready = False
        self._device: Optional[str] = None  # "cuda" | "cpu" | None
        # Tri-state: None = unknown / (re)loading (SHA not yet confirmed for the
        # active model), True = verified + loaded, False = a DEFINITIVE checksum
        # failure. A model swap sets None (not False) so the UI shows "loading",
        # not a false "model broken" banner, while the new model downloads.
        self._model_sha_ok: Optional[bool] = None
        self._last_inference_ms: Optional[float] = None
        self._session: Optional[Any] = None  # ort.InferenceSession
        self._input_name = "pixel_values"
        self._boot_lock = asyncio.Lock()
        self._started = False
        self._stopped = False
        # Self-heal state: the worker reports detect() outcomes here; once the
        # failure run crosses DETECT_FAILURE_THRESHOLD the detector is flagged
        # for a cooldown-guarded background reinit (rebuild ORT session).
        self._consecutive_failures = 0
        self._needs_reinit = False
        self._last_reinit_monotonic: Optional[float] = None
        # A model-swap reload runs as a background task so activate/PUT return
        # immediately even when the new model still has to download.
        self._reload_task: Optional[asyncio.Task] = None

    # ---------- lifecycle (event loop) ----------

    async def start(self) -> None:
        """Ensure the model artifact, build the session, warm up, publish
        ready/device. Never raises (CancelledError excepted); retries model
        download/verification with backoff. A GPU-required host without a
        usable CUDA EP is a terminal not-ready state (no pointless retry —
        the provider set can't change without a restart)."""
        self._started = True
        backoff = _RETRY_INITIAL_S
        while not self._stopped:
            try:
                async with self._boot_lock:
                    if await self._bootstrap_once():
                        return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — detector trouble never crashes the app
                log.exception("detector bootstrap failed — retrying in %.0f s", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_CAP_S)

    async def _bootstrap_once(self) -> bool:
        """One bootstrap attempt. Returns True when a terminal state was
        reached (ready, or ready=False with no point retrying). Raises on
        retryable failures (download/verify/session errors)."""
        import onnxruntime as ort  # deferred: GPU wheel absent on test hosts

        if self._require_gpu and "CUDAExecutionProvider" not in ort.get_available_providers():
            log.error(
                "GPU UNAVAILABLE — CUDAExecutionProvider not in this onnxruntime build "
                "(%s); set VIGILUME_REQUIRE_GPU=0 to accept CPU",
                ort.get_available_providers(),
            )
            self._ready = False
            self._device = None
            return True  # terminal: no download, no retry

        # Model artifact (download + SHA-256 verify) — raises to retry loop.
        # The ModelStore is the single downloader when wired; it surfaces
        # progress on the WS while this awaits. A bare detector (unit tests)
        # falls back to the ensure_model primitive directly.
        key = self._model_key
        try:
            if self._store is not None:
                path = await self._store.ensure_ready(key)
            else:
                path = await ensure_model(self._models_dir, key)
        except ModelVerifyError:
            # A genuine checksum failure (NOT a transient network/download error)
            # — flag the active model definitively broken for the UI, then
            # re-raise so start()/reload keep retrying. Transient errors leave
            # model_sha_ok as None ("verifying") so the UI shows loading, not a
            # false "model broken" banner.
            self._model_sha_ok = False
            raise
        self._model_sha_ok = True

        session, provider, warmup_ms, infer_ms = await asyncio.to_thread(
            self._build_session_blocking, ort, path
        )

        if provider != "CUDAExecutionProvider" and self._require_gpu:
            log.error(
                "GPU UNAVAILABLE — ORT picked %s (infer_ms=%.1f); "
                "set VIGILUME_REQUIRE_GPU=0 to accept CPU",
                provider, infer_ms,
            )
            self._ready = False
            self._device = None
            return True  # terminal until restart/reconfigure

        self._session = session
        # Point the engine's live class_id->label view at this model's output
        # vocabulary. Done together with the session swap so labels track the
        # actually-loaded model (COCO-80 or Objects365) with no engine change.
        set_active_labelmap(model_labelmap(key))
        self._last_inference_ms = round(infer_ms, 2)
        if provider == "CUDAExecutionProvider":
            self._device = "cuda"
            log.info(
                "GPU OK — provider=CUDAExecutionProvider model=%s warmup_ms=%.0f infer_ms=%.1f",
                key, warmup_ms, infer_ms,
            )
        else:
            self._device = "cpu"
            log.warning(
                "detector running on CPU (provider=%s model=%s infer_ms=%.1f) — "
                "acceptable for dev only",
                provider, key, infer_ms,
            )
        self._ready = True
        # A clean session build clears any pending self-heal state.
        self._consecutive_failures = 0
        self._needs_reinit = False
        # The model is now loaded in the detector — nudge the store so the WS
        # model_status flips loaded:true (state was already ready).
        if self._store is not None:
            self._store.notify(key)
        return True

    def _build_session_blocking(self, ort: Any, path: Path) -> tuple[Any, str, float, float]:
        """Create the ORT session + run the warm-up (blocking; to_thread).

        Returns (session, active_provider, total_warmup_ms, third_infer_ms).
        """
        providers: list[Any] = ["CPUExecutionProvider"]
        cuda_available = (
            not self._force_cpu
            and "CUDAExecutionProvider" in ort.get_available_providers()
        )
        if cuda_available:
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": 0}))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # Don't let ORT's thread pools busy-wait between inferences — this box
        # also runs the recorder/transcoder and the idle spin burns whole cores.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.inter_op_num_threads = 1
        if not cuda_available:
            # CPU inference: cap the intra-op pool so a heavy D-FINE run does
            # not oversubscribe the shared host. Tunable; default leaves cores
            # free for ffmpeg while keeping per-frame latency reasonable.
            so.intra_op_num_threads = int(env_dual("ORT_INTRA_THREADS", "4"))
        session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        active = session.get_providers()[0]
        self._input_name = session.get_inputs()[0].name

        zero = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        t_start = time.perf_counter()
        infer_ms = 0.0
        for _ in range(_WARMUP_RUNS):
            t0 = time.perf_counter()
            session.run(["logits", "pred_boxes"], {self._input_name: zero})
            infer_ms = (time.perf_counter() - t0) * 1000.0
        warmup_ms = (time.perf_counter() - t_start) * 1000.0
        return session, active, warmup_ms, infer_ms

    async def stop(self) -> None:
        """Release the session; subsequent detect() calls fail cleanly."""
        self._stopped = True
        self._ready = False
        self._device = None
        self._session = None
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()

    async def reconfigure(self, model_key: str, confidence: float) -> None:
        """Apply settings.detection changes. Confidence is picked up by the
        next detect() call; a model swap re-runs the bootstrap (download if
        absent + session rebuild) in the BACKGROUND so activate/PUT return
        immediately — the detector adopts the new model once its download and
        session build complete. Never raises."""
        self._confidence = float(confidence)
        if model_key not in MODELS:
            log.warning("unknown detector model %r ignored (keeping %s)", model_key, self._model_key)
            return
        if model_key == self._model_key:
            return
        old_key = self._model_key
        self._model_key = model_key
        # None ("verifying"), NOT False — the new model just isn't confirmed yet.
        # Reporting False here is what made the UI flash "model broken" on every
        # switch. It only becomes True (verified) or False (real checksum fail).
        self._model_sha_ok = None
        if not self._started or self._stopped:
            return  # start() hasn't run yet — it will pick up the new key
        log.info("detector model change %s -> %s — reloading in the background", old_key, model_key)
        self._ready = False
        self._start_reload_task()

    def _start_reload_task(self) -> None:
        """Launch (or replace) the background reload loop. Tolerates a model
        that is not yet downloaded: the store's ensure_ready waits on the
        in-flight download and the bootstrap adopts the model when it lands."""
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()
        self._reload_task = asyncio.create_task(self._reload_loop(), name="detector-reload")

    async def _reload_loop(self) -> None:
        backoff = _RETRY_INITIAL_S
        while not self._stopped:
            try:
                async with self._boot_lock:
                    if await self._bootstrap_once():
                        return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — detector trouble never crashes the app
                log.exception("detector reload failed — retrying in %.0f s", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_CAP_S)

    # ---------- self-heal (worker reports detect outcomes; supervisor reinits) ----------

    def note_detect_ok(self) -> None:
        """A detect() call returned normally — clear the failure run. Called on
        the event loop from the ingest worker after each successful inference."""
        self._consecutive_failures = 0

    def note_detect_failure(self) -> None:
        """A detect() call timed out or raised. Once the consecutive-failure run
        crosses DETECT_FAILURE_THRESHOLD, flag the detector for a background
        reinit (the ingest supervisor rebuilds the session)."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= DETECT_FAILURE_THRESHOLD:
            if not self._needs_reinit:
                log.warning(
                    "detector: %d consecutive detect failures — flagging session reinit",
                    self._consecutive_failures,
                )
            self._needs_reinit = True

    @property
    def needs_reinit(self) -> bool:
        return self._needs_reinit

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def reinit(self, *, force: bool = False) -> bool:
        """Rebuild the ORT session in the background to recover from a detect
        timeout/failure storm or a CUDA/GPU dropout (ready:false) WITHOUT a
        container restart. Cooldown-guarded (>= REINIT_COOLDOWN_S between
        attempts) so it never thrashes. On success ``ready`` flips true again
        and an INFO ``detector recovered`` is logged; on failure ``ready`` stays
        false, a WARNING is logged and the needs-reinit flag is left set so the
        supervisor retries after the cooldown. Never raises."""
        if self._stopped or not self._started:
            return self._ready
        now = time.monotonic()
        if (
            not force
            and self._last_reinit_monotonic is not None
            and now - self._last_reinit_monotonic < REINIT_COOLDOWN_S
        ):
            return self._ready  # within cooldown — skip (anti-thrash)
        self._last_reinit_monotonic = now
        # Clear the flag up front; a failed attempt re-sets it below so the
        # supervisor picks it up again on its next pass.
        self._needs_reinit = False
        log.info("detector reinit: rebuilding ORT session (model=%s)", self._model_key)
        try:
            async with self._boot_lock:
                self._ready = False
                await self._bootstrap_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — reinit trouble never crashes the app
            log.exception("detector reinit failed — will retry after cooldown")
            self._needs_reinit = True
            return False
        if self._ready:
            log.info("detector recovered — reinit rebuilt the session (device=%s)", self._device)
            return True
        log.warning(
            "detector reinit did not restore readiness (device unavailable) — "
            "retrying after cooldown"
        )
        self._needs_reinit = True
        return False

    def last_reinit_age_s(self) -> Optional[float]:
        """Seconds since the last reinit attempt (None = never reinited)."""
        if self._last_reinit_monotonic is None:
            return None
        return round(time.monotonic() - self._last_reinit_monotonic, 2)

    # ---------- inference (engine's worker thread via asyncio.to_thread) ----------

    def detect(
        self, frame_bgr: np.ndarray, detect_width: int, detect_height: int
    ) -> sv.Detections:
        """BLOCKING single-frame inference.

        Returns ``sv.Detections`` with float32 ``xyxy`` in detect-stream
        pixels (0..detect_width/height), ``confidence`` and COCO-80
        ``class_id``, already thresholded at ``self.confidence``. Label
        filtering to the camera's detect_objects happens downstream (engine).
        Raises RuntimeError while not ready.
        """
        session = self._session
        if not self._ready or session is None:
            raise RuntimeError("detector not ready")
        tensor = preprocess(frame_bgr)
        t0 = time.perf_counter()
        logits, pred_boxes = session.run(
            ["logits", "pred_boxes"], {self._input_name: tensor}
        )
        self._last_inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return decode(logits, pred_boxes, self._confidence, detect_width, detect_height)

    # ---------- stats (cheap, thread-safe, sync) ----------

    @property
    def kind(self) -> str:
        return "onnx"

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def model_sha_ok(self) -> Optional[bool]:
        return self._model_sha_ok

    @property
    def last_inference_ms(self) -> Optional[float]:
        return self._last_inference_ms

    def status(self) -> dict[str, Any]:
        """Detector block for /api/system/detector."""
        return {
            "kind": "onnx",
            "ready": self._ready,
            "device": self._device,
            "model": self._model_key,
            "model_sha_ok": self._model_sha_ok,
            "last_inference_ms": self._last_inference_ms,
            # Self-heal visibility (so the user/UI can see reinit fired).
            "consecutive_failures": self._consecutive_failures,
            "needs_reinit": self._needs_reinit,
            "last_reinit_age_s": self.last_reinit_age_s(),
        }


# ---------- detector factory (VIGILUME_DETECTOR selection) ----------


class AutoDetector:
    """Edge TPU when the box has one, GPU otherwise — decided at start().

    THE POINT: a Coral should be plug-and-play. Fit one, restart, and detection
    uses it; pull it out and detection keeps working on the GPU. Nobody should
    have to know a setting exists.

    HOW IT DECIDES. `start()` boots the Coral first. Its bootstrap never raises
    — it either binds its devices and reports ready, or logs CORAL UNAVAILABLE
    and reports not-ready — so "did a TPU answer?" is just `coral.ready`. On
    not-ready we fall through to ONNX. Probing separately would be worse than
    useless: libedgetpu hands a delegate its device EXCLUSIVELY, so a probe that
    succeeded would be holding the very device the real detector then wants.

    ONE-WAY, and deliberately so. Once ONNX is chosen this object IS the ONNX
    detector for the process lifetime; a Coral that appears later is picked up
    on the next restart. Hot-swapping mid-run would mean tearing down a live
    inference path on a security system to chase a device that was not there at
    boot — a bad trade for a case a restart already covers.

    Everything after `start()` is a straight delegation to whichever detector
    won, so the ingest worker and self-heal supervisor keep seeing exactly one
    detector object with the same duck-typed surface. That is the rule this
    whole backend is built on.
    """

    def __init__(self, *, coral: Any, make_onnx: Callable[[], Any]) -> None:
        self._coral = coral
        # ONNX is built EAGERLY and is the active detector until Coral proves
        # itself. Constructing it is cheap — no download, no session, that all
        # happens in start() — and it means every status read BEFORE the
        # decision describes the backend we will most likely end up on, rather
        # than describing a Coral that may not exist. (Defaulting to the Coral
        # instance made `model_key` report an Edge TPU model on a GPU-only box,
        # which broke the detector self-test's model-state reporting.)
        self._onnx = make_onnx()
        self._active: Any = self._onnx
        self._decided = False

    async def start(self) -> None:
        if self._decided:
            await self._active.start()
            return
        await self._coral.start()
        if self._coral.ready:
            self._active = self._coral
            self._decided = True
            log.info(
                "detector AUTO: Edge TPU detected (%d device(s)) — using Coral",
                self._coral.status().get("device_count", 1),
            )
            return
        # No TPU (or it failed to bind): release anything it claimed, then GPU.
        await self._coral.stop()
        self._active = self._onnx
        self._decided = True
        log.info("detector AUTO: no Edge TPU available — using the GPU/ONNX backend")
        await self._onnx.start()

    # ---- everything below is pure delegation to the chosen detector ----

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes this class does NOT define, which is every
        # member of the detector contract except start(). Keeping it dynamic
        # means a new member on either backend needs no change here.
        return getattr(self._active, name)



def build_detector(
    *,
    config: Any,
    models_dir: Path,
    model_key: str,
    confidence: float,
    store: Optional["ModelStore"] = None,
    backend: Optional[str] = None,
    coral_model_key: Optional[str] = None,
) -> Any:
    """Construct the detector chosen by ``config.detector`` (VIGILUME_DETECTOR).

    Returns an ``OnnxDetector`` implementing the detector interface the ingest
    worker + self-heal supervisor call (``ready``/``device``/``kind``/
    ``model_key``/``status()``/``detect()``/``start()``/``stop()``/
    ``reconfigure()`` + the self-heal hooks).

    Selection (see config.Config.detector / docs):
    - ``onnx``     (default) -> ``OnnxDetector``; ``VIGILUME_REQUIRE_GPU`` gates
      CUDA-vs-CPU as before.
    - ``onnx_cpu``           -> ``OnnxDetector`` with ``require_gpu`` forced off
      (conscious CPU inference; ``VIGILUME_REQUIRE_GPU`` is overridden).
    """
    # PRECEDENCE, and it is deliberate: an explicitly-set VIGILUME_DETECTOR wins
    # (the low-level/CI override), otherwise the stored settings.detection.backend
    # decides. That way the UI switch is the normal path but an operator can
    # always force a backend from the environment without touching the DB — which
    # matters when the stored choice is the thing that broke detection.
    kind = str(getattr(config, "detector", "onnx") or "onnx")
    if backend and not os.environ.get("VIGILUME_DETECTOR"):
        kind = backend
    if kind == "auto":
        # AUTO: use Edge TPUs when the box has them, else the GPU. The probe IS
        # the claim (libedgetpu hands a delegate its device exclusively), so
        # rather than detect-then-construct — which would race itself for the
        # device — this constructs the Coral detector and lets its own bootstrap
        # decide. CoralDetector.start() never raises: it either binds and
        # reports ready, or logs CORAL UNAVAILABLE and reports not-ready.
        #
        # The FALLBACK is what makes this safe to default to. `AutoDetector`
        # below wraps both and swaps to ONNX if Coral does not come up, so a box
        # with no TPU (or a TPU that stops answering) still detects. Detection
        # never silently stops — that is the rule this whole file serves.
        from .coral import CoralDetector  # noqa: PLC0415 — optional backend

        log.info("detector selection: AUTO — trying Edge TPU, falling back to GPU")
        return AutoDetector(
            coral=CoralDetector(
                models_dir=models_dir, confidence=confidence,
                model_key=coral_model_key or "",
            ),
            make_onnx=lambda: OnnxDetector(
                models_dir=models_dir,
                model_key=model_key,
                confidence=confidence,
                require_gpu=bool(getattr(config, "require_gpu", True)),
                store=store,
            ),
        )

    if kind == "coral":
        # Edge TPU. Deliberately constructed and returned HERE and nowhere else:
        # nothing outside this branch knows Coral exists, so removing the
        # backend again is deleting these lines plus its VALID_DETECTORS entry.
        from .coral import CoralDetector  # noqa: PLC0415 — optional backend

        log.info("detector selection: Coral Edge TPU (VIGILUME_DETECTOR=coral)")
        # model_key is the CORAL registry key (settings.detection.coral_model),
        # NOT a D-FINE tier — the two model lists are disjoint. An unknown key
        # falls back to the Coral default inside CoralDetector.
        return CoralDetector(
            models_dir=models_dir, confidence=confidence, model_key=coral_model_key or ""
        )

    force_cpu = kind == "onnx_cpu"
    require_gpu = bool(getattr(config, "require_gpu", True)) and not force_cpu
    if force_cpu:
        log.info(
            "detector selection: ONNX on CPU (VIGILUME_DETECTOR=onnx_cpu; "
            "CUDA is not offered to onnxruntime, require_gpu overridden)"
        )
    return OnnxDetector(
        models_dir=models_dir,
        model_key=model_key,
        confidence=confidence,
        require_gpu=require_gpu,
        force_cpu=force_cpu,
        store=store,
    )

"""Native engine smoke suite (docs/native-mode-design.md §10 test plan).

Sections are owned by the implementation passes that build the code under
test.

DETECTOR / INGEST / PIPELINE sections (run first):

  1. pins & import hygiene — ByteTrackTracker importable, supervision
     >=0.29.1 <0.30, MODELS revision+SHA pin table, importing the app never
     imports onnxruntime at module scope
  2. preprocess — shape/dtype/range + BGR->RGB order on a two-tone frame
  3. decode — hand-built logits/pred_boxes vectors -> exact detect-pixel
     boxes, sigmoid math, >= threshold edge, empty frames, tracker on empty
  4. model store — fake-pin download via httpx.MockTransport: sidecar,
     idempotent re-ensure, tampered-file re-download, corrupt download =>
     ModelVerifyError with nothing left behind
  5. SENTINEL_REQUIRE_GPU — CUDA EP absent (this host) => terminal
     ready=False/device=None BEFORE any download; detect() raises
  6. REAL dfine_n — download (persistent cache) + SHA verify + CPU
     bootstrap + inference on synthetic frames; reconfigure/stop
  7. REAL dfine_s — dog-fixture detection quality, moving dog frames ->
     ByteTrack continuity -> DetectionEngine -> real EventsPipeline on a
     temp DB -> native.-prefixed event row + exact live counts + annotated
     snapshot + web-push payload (fake push) + absence end + clip request
  8. ingest — golden ffmpeg argv, latest-frame-drop slot, staleness
     watchdog respawn, EOF backoff, IngestManager reload reconciliation
     (detect_enabled / fps change / removal / ffmpeg-less host) + worker
  9. app boot — /api/system/health + /api/system/detector shapes with
     SENTINEL_REQUIRE_GPU=1 on a GPU-less host (last section)

Real-model downloads (~15+41 MB on first run) land in a persistent cache so
reruns are offline: $SENTINEL_TEST_MODEL_CACHE or
~/.cache/sentinel-tests/models.

RECORDER sections:

  A. ffmpeg argv builders — golden-exact segment + clip commands, concat
     list rendering/escaping, path layout
  B. segment index — filename-timestamp parsing and clip window selection
     including every boundary case (±10 s head rule, window_end exclusive,
     hour/day rollover, garbage tolerance)
  C. retention — hour-dir pruning on fake trees, empty-day-dir cleanup,
     clip pruning, low-disk guard (fake disk sizes) incl. active-dir skip
  D. clip extraction — full extract path against a real SQLite event row
     with a mocked ffmpeg runner (argv + concat capture, has_clip update,
     failure/no-segments/unknown-id paths, schedule_clip fire-and-forget);
     runs the REAL ffmpeg end-to-end when one is on PATH
  E. lifecycle — start/stop/reload idempotence, ffmpeg-absent degradation,
     supervised child bookkeeping via a fake process
  F. media route — GET /api/events/{id}/clip.mp4 serves recorder clip files
     through the real app (Range support, synthetic-event 404s, delete
     cleanup)

STREAMS / MEDIA-PROVIDER sections are added by their own passes — keep the
check() style and section layout.

CPU-only, no network, no GPU; ffmpeg is feature-detected (this suite must
pass on hosts without it). Usage:

    python backend/tests/native_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Environment must be clean before app config is instantiated (lifespan).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
# REQUIRE_GPU=1 on this GPU-less test host makes every app boot's detector
# hard-fail fast and cheap (no 41 MB model download in the background during
# route tests); detector component tests construct OnnxDetector with an
# explicit require_gpu=False instead.
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
# Unroutable local ports -> instant refusal: go2rtc syncs never block and
# any real ffmpeg child a recorder might spawn dies immediately.
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-native-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import contextlib  # noqa: E402
import logging  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main  # noqa: E402,F401 — import hygiene: must not pull onnxruntime
from app.auth import AuthService  # noqa: E402
from app.config import APP_VERSION, Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.events_pipeline import EventsPipeline  # noqa: E402
from app.native import detector as detector_module  # noqa: E402
from app.native import ingest as ingest_module  # noqa: E402
from app.native import recorder as recorder_module  # noqa: E402
from app.native.coco_labels import LABEL_TO_ID  # noqa: E402
from app.native.detector import (  # noqa: E402
    DEFAULT_MODEL,
    MODELS,
    ModelVerifyError,
    OnnxDetector,
    decode,
    ensure_model,
    model_path,
    preprocess,
    sha256_file,
)
from app.native.engine import DetectionEngine, observations_from_supervision  # noqa: E402
from app.native.ingest import FrameSource, IngestManager, build_ingest_args  # noqa: E402
from app.native.media import NativeMediaProvider  # noqa: E402
from app.native import streams as streams_module  # noqa: E402
from app.native.streams import (  # noqa: E402
    build_config,
    default_stream_url,
    is_doorbell,
    stream_sources,
    sub_stream_name,
    webrtc_status,
)
from app.notify.push import PushSendResult  # noqa: E402
from app.settings_store import SettingsStore  # noqa: E402
from app.ws import WSManager  # noqa: E402
from app.native.recorder import (  # noqa: E402
    CLIP_PAD_S,
    LOW_DISK_BYTES,
    Recorder,
    SEGMENT_SECONDS,
    build_clip_args,
    build_concat_list,
    build_segment_args,
    hour_dir,
    iter_hour_dirs,
    parse_segment_start,
    prune_clips,
    prune_recordings,
    select_segments,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- fixtures / helpers ----------------

DEFAULT_RECORDING = {"continuous_days": 7, "event_days": 14, "snapshot_days": 14}


class FakeSettings:
    """Duck-typed SettingsStore: the recorder only reads .recording."""

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False

    def __init__(self, recording: dict | None = None):
        self.recording = dict(recording or DEFAULT_RECORDING)


def make_config(tag: str) -> Config:
    cfg = Config()
    cfg.data_dir = TMP / tag / "data"
    cfg.media_dir = TMP / tag / "media"
    return cfg


def seg_path(cam_dir: Path, dt: datetime) -> Path:
    return cam_dir / dt.strftime("%Y-%m-%d") / dt.strftime("%H") / dt.strftime("%M.%S.ts")


def make_seg(cam_dir: Path, dt: datetime, mtime: float | None = None) -> Path:
    p = seg_path(cam_dir, dt)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x47" * 188)  # TS sync bytes; content is irrelevant
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def camera_row(name: str, record_enabled: bool = True) -> dict:
    return {
        "name": name,
        "friendly_name": name.title(),
        "model": "AD410",
        "ip": "192.0.2.50",
        "username": "admin",
        "password": "pw",
        "detect_objects": [],
        "detect_width": 640,
        "detect_height": 480,
        "detect_fps": 5,
        "detect_enabled": True,
        "record_enabled": record_enabled,
        "capabilities": {},
    }


# =====================================================================
# Detector / ingest / pipeline sections (see docstring, sections 1-9)
# =====================================================================

MODEL_CACHE = Path(
    os.environ.get("SENTINEL_TEST_MODEL_CACHE")
    or Path.home() / ".cache" / "sentinel-tests" / "models"
)
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOG_JPG = FIXTURES / "dog_704x480.jpg"


def detector_pin_checks() -> None:
    print("detector 1: pins & import hygiene")
    check("onnxruntime" not in sys.modules,
          "importing the app tree does not import onnxruntime")

    sv_version = tuple(int(p) for p in sv.__version__.split(".")[:2])
    check((0, 29) <= sv_version < (0, 30), f"supervision {sv.__version__} is >=0.29,<0.30")

    from trackers import ByteTrackTracker  # noqa: PLC0415

    check(callable(ByteTrackTracker), "trackers ByteTrackTracker importable (2.4.0 pin)")

    check(set(MODELS) == {"dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x",
                          "dfine_l_obj365"},
          "MODELS pins the COCO tiers (n/s/m/l/x) + the Objects365 model")
    check(DEFAULT_MODEL == "dfine_s", "default model is dfine_s")
    for key, pin in MODELS.items():
        ok_pin = (
            pin["url"].startswith("https://huggingface.co/onnx-community/")
            and "/resolve/" in pin["url"]
            and len(pin["sha256"]) == 64
            and int(pin["bytes"]) > 1_000_000
            and pin["labelmap"] in ("coco", "obj365")
        )
        check(ok_pin, f"{key}: revision-pinned URL + SHA-256 + size + labelmap")
    check(MODELS["dfine_s"]["sha256"].startswith("cd8a49a945feda"),
          "dfine_s pin matches CONTRACTS.md")
    check(all(MODELS[k]["labelmap"] == "coco" for k in
              ("dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x"))
          and MODELS["dfine_l_obj365"]["labelmap"] == "obj365",
          "COCO tiers map to coco; the obj365 model maps to obj365")


def make_tracker():
    from trackers import ByteTrackTracker  # noqa: PLC0415

    return ByteTrackTracker(
        lost_track_buffer=25, frame_rate=5.0, track_activation_threshold=0.5,
        minimum_consecutive_frames=2, minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )


def preprocess_checks() -> None:
    print("detector 2: preprocessing")
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    frame[:, :352] = (255, 0, 0)   # left half pure BLUE in BGR
    frame[:, 352:] = (0, 0, 255)   # right half pure RED in BGR
    tensor = preprocess(frame)
    check(tensor.shape == (1, 3, 640, 640), "output shape [1,3,640,640]")
    check(tensor.dtype == np.float32, "output dtype float32")
    check(float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0, "range [0,1] (/255, no normalize)")
    # BGR->RGB: a blue pixel must land in the B plane (idx 2), not R (idx 0).
    check(tensor[0, 2, 320, 100] > 0.99 and tensor[0, 0, 320, 100] < 0.01,
          "BGR->RGB: left (blue) half lands in the B plane")
    check(tensor[0, 0, 320, 500] > 0.99 and tensor[0, 2, 320, 500] < 0.01,
          "BGR->RGB: right (red) half lands in the R plane")


def _make_outputs(entries):
    """entries: [(query_idx, class_id, logit, (cx, cy, w, h))]"""
    logits = np.full((1, 300, 80), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    for qi, cid, logit, box in entries:
        logits[0, qi, cid] = logit
        boxes[0, qi] = box
    return logits, boxes


def decode_checks() -> None:
    print("detector 3: NMS-free decode unit vectors")
    logits, boxes = _make_outputs([(0, 16, 3.0, (0.5, 0.5, 0.25, 0.5))])
    dets = decode(logits, boxes, 0.5, 704, 480)
    check(len(dets) == 1, "single query above threshold -> one detection")
    check(int(dets.class_id[0]) == 16, "class_id 16 (dog) preserved")
    expected_conf = 1.0 / (1.0 + np.exp(-3.0))
    check(abs(float(dets.confidence[0]) - expected_conf) < 1e-6, "confidence == sigmoid(logit)")
    check(np.allclose(dets.xyxy[0], [264.0, 120.0, 440.0, 360.0], atol=1e-3),
          "cxcywh -> xyxy scaled into 704x480 detect pixels")
    check(dets.xyxy.dtype == np.float32, "xyxy dtype float32")

    dets2 = decode(logits, boxes, 0.5, 640, 360)
    check(np.allclose(dets2.xyxy[0], [240.0, 90.0, 400.0, 270.0], atol=1e-3),
          "box scaling follows detect_width/height")

    logits, boxes = _make_outputs(
        [(0, 0, 0.0, (0.5, 0.5, 0.2, 0.2)), (1, 2, -0.1, (0.3, 0.3, 0.1, 0.1))]
    )
    dets = decode(logits, boxes, 0.5, 704, 480)
    check(len(dets) == 1 and int(dets.class_id[0]) == 0,
          "sigmoid(0)==0.5 kept at threshold 0.5 (>=), just-below dropped")

    logits, boxes = _make_outputs(
        [(5, 0, 4.0, (0.2, 0.2, 0.1, 0.1)), (9, 2, 2.0, (0.7, 0.7, 0.2, 0.2))]
    )
    dets = decode(logits, boxes, 0.5, 704, 480)
    check(len(dets) == 2 and set(dets.class_id.tolist()) == {0, 2},
          "two queries decode independently (no NMS)")

    logits, boxes = _make_outputs([])
    empty = decode(logits, boxes, 0.5, 704, 480)
    check(len(empty) == 0, "all-background frame -> empty sv.Detections")
    check(len(make_tracker().update(empty)) == 0, "ByteTrackTracker accepts an empty frame")

    # trackers 2.4.0 marks not-yet-activated detections with tracker_id -1
    # (ids start at 0) — those must never reach the engine as observations.
    unactivated = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([16]),
    )
    unactivated.tracker_id = np.array([-1])
    check(observations_from_supervision(unactivated) == [],
          "tracker_id -1 (unactivated track) is dropped from observations")


async def _model_store_cases() -> None:
    payload = b"fake-onnx-bytes-" * 1024
    good_sha = hashlib.sha256(payload).hexdigest()
    MODELS["fake_test"] = {
        "url": "https://models.example/fake.onnx", "bytes": len(payload), "sha256": good_sha,
    }
    served = {"body": payload}
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, content=served["body"])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    models_dir = TMP / "modelstore"
    try:
        path = await ensure_model(models_dir, "fake_test", client=client)
        check(path == model_path(models_dir, "fake_test") and path.is_file(),
              "download lands at models_dir/{key}.onnx")
        check(path.read_bytes() == payload, "downloaded bytes intact")
        check(hits["n"] == 1, "exactly one HTTP fetch")
        sidecar = json.loads((models_dir / "fake_test.json").read_text())
        check(sidecar["sha256"] == good_sha
              and sidecar["url"] == MODELS["fake_test"]["url"]
              and sidecar["downloaded_at"] > 0,
              "sidecar records url/sha256/downloaded_at")
        check(not list(models_dir.glob("*.part")), "no .part file left after success")

        await ensure_model(models_dir, "fake_test", client=client)
        check(hits["n"] == 1, "re-ensure with a verified file skips the download")

        path.write_bytes(b"tampered!")
        await ensure_model(models_dir, "fake_test", client=client)
        check(path.read_bytes() == payload and hits["n"] == 2,
              "tampered on-disk file is re-downloaded + re-verified")

        served["body"] = b"corrupted-download"
        path.unlink()
        raised = False
        try:
            await ensure_model(models_dir, "fake_test", client=client)
        except ModelVerifyError:
            raised = True
        check(raised, "corrupt download raises ModelVerifyError")
        check(not path.exists() and not list(models_dir.glob("*.part")),
              "failed verify leaves no artifact behind")

        # download failure (HTTP 404): clean refusal, nothing left behind
        MODELS["fake_404"] = {
            "url": "https://models.example/gone.onnx", "bytes": 1, "sha256": "0" * 64,
        }
        client404 = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(404, content=b"gone"))
        )
        try:
            raised = False
            try:
                await ensure_model(models_dir, "fake_404", client=client404)
            except httpx.HTTPStatusError:
                raised = True
            check(raised, "HTTP 404 download raises cleanly (HTTPStatusError)")
            check(not model_path(models_dir, "fake_404").exists()
                  and not (models_dir / "fake_404.json").exists()
                  and not list(models_dir.glob("*.part")),
                  "404 failure leaves no model, sidecar or .part behind")

            # detector bootstrap with a failing download: start() keeps
            # retrying in the background, ready stays False, nothing raises
            # out (the app keeps serving while the detector is down).
            real_ensure = detector_module.ensure_model

            async def _failing_ensure(*args, **kwargs):
                raise RuntimeError("simulated model download failure")

            detector_module.ensure_model = _failing_ensure
            try:
                det = OnnxDetector(models_dir=TMP / "dl-fail", model_key="dfine_n",
                                   confidence=0.5, require_gpu=False)
                task = asyncio.create_task(det.start())
                await asyncio.sleep(0.3)
                check(det.ready is False and det.model_sha_ok is None,
                      "download failure -> not-ready, sha unknown/None (retrying, not 'broken')")
                check(not task.done(), "start() keeps retrying with backoff after the failure")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            finally:
                detector_module.ensure_model = real_ensure
        finally:
            del MODELS["fake_404"]
            await client404.aclose()
    finally:
        del MODELS["fake_test"]
        await client.aclose()


def model_store_checks() -> None:
    print("detector 4: model store (fake pin, MockTransport)")
    asyncio.run(_model_store_cases())


async def _require_gpu_cases() -> None:
    import onnxruntime as ort  # noqa: PLC0415

    # This section assumes a host without the CUDA EP (dev Mac / CI).
    assert "CUDAExecutionProvider" not in ort.get_available_providers()

    models_dir = TMP / "gpu-required-models"
    det = OnnxDetector(models_dir=models_dir, model_key="dfine_n",
                       confidence=0.5, require_gpu=True)

    class _LogCapture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    capture = _LogCapture()
    det_logger = logging.getLogger("app.native.detector")
    det_logger.addHandler(capture)
    try:
        await asyncio.wait_for(det.start(), timeout=10.0)  # terminal, no retry loop
    finally:
        det_logger.removeHandler(capture)
    check(any(r.levelno == logging.ERROR and "GPU UNAVAILABLE" in r.getMessage()
              for r in capture.records),
          "hard GPU failure logs a loud ERROR ('GPU UNAVAILABLE ... SENTINEL_REQUIRE_GPU=0')")
    check(det.ready is False, "require_gpu + no CUDA EP -> ready stays False")
    check(det.device is None, "require_gpu + no CUDA EP -> device is None")
    check(not models_dir.exists() or not any(models_dir.iterdir()),
          "GPU pre-check fails BEFORE any model download")
    raised = False
    try:
        det.detect(np.zeros((480, 704, 3), np.uint8), 704, 480)
    except RuntimeError:
        raised = True
    check(raised, "detect() raises RuntimeError while not ready")
    check(det.status() == {"kind": "onnx", "ready": False, "device": None,
                           "model": "dfine_n",
                           "model_sha_ok": None, "last_inference_ms": None,
                           "consecutive_failures": 0, "needs_reinit": False,
                           "last_reinit_age_s": None},
          "status() reports the hard-failed detector (sha None — GPU gate, model never verified)")


def require_gpu_checks() -> None:
    print("detector 5: SENTINEL_REQUIRE_GPU on a GPU-less host")
    asyncio.run(_require_gpu_cases())


async def _dfine_n_cases() -> None:
    path = await ensure_model(MODEL_CACHE, "dfine_n")
    check(path.stat().st_size == MODELS["dfine_n"]["bytes"], "dfine_n artifact size matches pin")
    check(sha256_file(path) == MODELS["dfine_n"]["sha256"], "dfine_n SHA-256 matches pin")

    det = OnnxDetector(models_dir=MODEL_CACHE, model_key="dfine_n",
                       confidence=0.5, require_gpu=False)
    await asyncio.wait_for(det.start(), timeout=120.0)
    check(det.ready is True, "require_gpu=0 -> CPU bootstrap reaches ready")
    check(det.device == "cpu", "device reported as 'cpu'")
    check(det.model_sha_ok is True, "model_sha_ok True after verified load")
    check(det.last_inference_ms is not None and det.last_inference_ms > 0,
          "warm-up published last_inference_ms")

    gray = np.full((480, 704, 3), 114, dtype=np.uint8)
    dets = det.detect(gray, 704, 480)
    check(isinstance(dets, sv.Detections) and len(dets) == 0,
          "uniform synthetic frame -> zero detections at 0.5")
    noise = np.random.default_rng(7).integers(0, 255, (480, 704, 3), dtype=np.uint8)
    check(len(det.detect(noise, 704, 480)) <= 3, "noise frame decodes without junk floods")

    session_before = det._session
    await det.reconfigure("dfine_n", 0.35)
    check(det._session is session_before and det.confidence == 0.35,
          "confidence-only reconfigure keeps the session")
    await det.reconfigure("not_a_model", 0.5)
    check(det.model_key == "dfine_n", "unknown model key ignored")

    st = det.status()
    check(st["ready"] is True and st["device"] == "cpu" and st["model"] == "dfine_n"
          and st["model_sha_ok"] is True and st["last_inference_ms"] > 0,
          "status() self-test block is complete")

    await det.stop()
    check(det.ready is False, "stop() clears ready")
    raised = False
    try:
        det.detect(gray, 704, 480)
    except RuntimeError:
        raised = True
    check(raised, "detect() after stop raises RuntimeError")


def dfine_n_checks() -> None:
    print("detector 6: REAL dfine_n download + SHA verify + CPU inference")
    asyncio.run(_dfine_n_cases())


class FakePush:
    public_key = "fake"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def send_to_all(self, payload: dict) -> PushSendResult:
        self.payloads.append(payload)
        return PushSendResult(attempted=1, sent=1)


class ClipSpyRecorder:
    """Mocked recorder: records the clip request and immediately 'finishes'
    the clip (writes the file at the event row's clip path) like the real
    Recorder does ~20 s after event end."""

    def __init__(self, clips_dir: Path, db=None) -> None:
        self._clips_dir = clips_dir
        self._db = db
        self.clip_requests: list[tuple] = []

    def clip_path(self, event_id: int) -> Path:
        return self._clips_dir / f"{event_id}.mp4"

    async def schedule_clip(self, camera: str, frigate_id: str, start: float, end: float) -> None:
        self.clip_requests.append((camera, frigate_id, start, end))
        if self._db is not None:
            row = await self._db.get_event_by_frigate_id(frigate_id)
            if row is not None:
                self._clips_dir.mkdir(parents=True, exist_ok=True)
                self.clip_path(int(row["id"])).write_bytes(b"\x00\x00\x00\x18ftypisom-stub")
                # Mirror the real recorder: has_clip flips to true only AFTER a
                # non-empty clip file exists — never optimistically at event end.
                await self._db.update_event(int(row["id"]), has_clip=True)


class FakeWSClient:
    """Captures WSManager broadcasts like a connected /api/ws client."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def accept(self, subprotocol=None) -> None:
        # Mirrors starlette's signature. WSManager.connect passes the echoed
        # subprotocol (the bearer-token handshake), and a fake that refuses the
        # kwarg raises INSIDE the endpoint — which then hangs the suite on
        # aiosqlite's non-daemon worker at exit rather than failing loudly.
        self.subprotocol = subprotocol
        return None

    async def send_text(self, text: str) -> None:
        self.messages.append(json.loads(text))

    def types(self) -> list[str]:
        return [m.get("type") for m in self.messages]


def _dog_frame(x: int, patch: np.ndarray) -> np.ndarray:
    canvas = np.full((480, 704, 3), 114, dtype=np.uint8)
    canvas[140:340, x:x + 260] = patch
    return canvas


async def _pipeline_cases() -> None:
    check(DOG_JPG.is_file(), "dog fixture image present")
    dog = cv2.imread(str(DOG_JPG))
    check(dog is not None and dog.shape == (480, 704, 3), "dog fixture decodes at 704x480")
    patch = cv2.resize(dog, (260, 200), interpolation=cv2.INTER_AREA)

    det = OnnxDetector(models_dir=MODEL_CACHE, model_key="dfine_s",
                       confidence=0.5, require_gpu=False)
    await asyncio.wait_for(det.start(), timeout=300.0)
    check(det.ready and det.device == "cpu" and det.model_sha_ok, "dfine_s bootstrapped on CPU")

    # -- straight detection quality on the fixture --
    dets = det.detect(dog, 704, 480)
    dog_hits = [i for i, cid in enumerate(dets.class_id) if int(cid) == LABEL_TO_ID["dog"]]
    check(len(dog_hits) == 1, "dfine_s finds exactly one dog on the fixture")
    best = dog_hits[0]
    check(float(dets.confidence[best]) > 0.8, "dog confidence > 0.8")
    x1, y1, x2, y2 = (float(v) for v in dets.xyxy[best])
    check(-5 <= x1 < x2 <= 709 and -5 <= y1 < y2 <= 485
          and (x2 - x1) * (y2 - y1) > 0.3 * 704 * 480,
          "dog box inside the frame and plausibly large")

    # -- full stack: temp DB + real settings/pipeline, fake push --
    config = Config()
    (TMP / "pipeline").mkdir(parents=True, exist_ok=True)
    db = Database(TMP / "pipeline" / "nvr.db")
    await db.connect()
    await db.upsert_camera({
        "name": "yard", "friendly_name": "Back Yard", "model": "IP5M-T1277EW-AI",
        "ip": "127.0.0.1", "username": "u", "password": "p",
        "detect_objects": ["dog"], "detect_width": 704, "detect_height": 480,
        "detect_fps": 5, "detect_enabled": True, "record_enabled": True,
        "capabilities": {}, "created_at": time.time(),
    })
    settings = SettingsStore(db)
    await settings.load()
    ws = WSManager()
    ws_client = FakeWSClient()
    await ws.connect(ws_client)  # capture broadcasts like a live /api/ws client
    push = FakePush()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    recorder = ClipSpyRecorder(TMP / "pipeline" / "clips", db=db)
    snapshots_dir = TMP / "pipeline" / "snapshots"

    engine = DetectionEngine(db, det, recorder, settings, config)
    media = NativeMediaProvider(db, engine, recorder)
    pipeline = EventsPipeline(db, media, ws, push, settings, auth, snapshots_dir)
    engine.set_pipeline(pipeline)
    await engine.reload()  # camera rows only; no ingest/housekeeping tasks here

    tracker = make_tracker()
    t0 = time.time()
    tracker_ids: set[int] = set()
    per_frame_obs: list[int] = []
    for i in range(8):
        frame = _dog_frame(60 + i * 30, patch)
        tracked = tracker.update(det.detect(frame, 704, 480))
        obs = observations_from_supervision(tracked)
        per_frame_obs.append(len(obs))
        tracker_ids.update(o.tracker_id for o in obs if o.label == "dog")
        await engine.process("yard", t0 + i * 0.2, obs, frame_bgr=frame)

    check(sum(per_frame_obs[-4:]) >= 4, "moving dog keeps producing tracked observations")
    check(len(tracker_ids) == 1, "ByteTrack keeps ONE track id across the moving frames")

    key = ("yard", "dog")
    st = engine._events.get(key)
    check(st is not None and st.fid.startswith("native."), "open event uses the native. id prefix")
    fid = st.fid
    check(st.best_score > 0.8 and st.best_frame is not None, "engine retained a best frame + score")
    check(pipeline.counts.get(key) == 1, "engine feeds update_count -> live count is exactly 1")

    row = await db.get_event_by_frigate_id(fid)
    check(row is not None and row["camera"] == "yard" and row["label"] == "dog",
          "event row stored through the real EventsPipeline")
    event_id = int(row["id"])
    check(not bool(row["has_clip"]),
          "has_clip is FALSE at event start — the recorder flips it only after "
          "the clip file lands (no optimistic true)")

    new_msgs = [m for m in ws_client.messages if m.get("type") == "event_new"]
    check(len(new_msgs) == 1 and new_msgs[0]["event"]["id"] == event_id
          and new_msgs[0]["event"]["camera"] == "yard"
          and new_msgs[0]["event"]["label"] == "dog",
          "WS broadcast: exactly one event_new carrying the DB row")

    jpeg = engine.event_best_jpeg(fid)
    check(jpeg is not None and jpeg[:2] == b"\xff\xd8", "event_best_jpeg serves the retained frame as JPEG")
    live = await media.latest_jpg("yard", height=240)
    check(live is not None and live[:2] == b"\xff\xd8", "media provider latest_jpg serves the frame cache")
    check(await media.detect_dims("yard") == (704, 480), "detect_dims from the camera row")

    # enrichment + notification run as background tasks — wait for them
    deadline = time.monotonic() + 15.0
    snap_path = snapshots_dir / f"{event_id}.jpg"
    row = None
    while time.monotonic() < deadline:
        row = await db.get_event(event_id)
        if row and row["has_snapshot"] and snap_path.is_file() and push.payloads:
            break
        await asyncio.sleep(0.05)
    check(snap_path.is_file() and snap_path.stat().st_size > 1000,
          "annotated snapshot written to snapshots/{row_id}.jpg")
    annotated = cv2.imread(str(snap_path))
    check(annotated is not None and annotated.shape[1] == 704 and annotated.shape[0] >= 480,
          "annotated snapshot keeps detect-res width (+ count banner strip)")
    check(row is not None and bool(row["has_snapshot"]), "row has_snapshot set after enrichment")
    check(len(push.payloads) == 1, "exactly one push notification (cooldown + notified-once)")
    payload = push.payloads[0]
    check(payload["title"] == "Dog detected at Back Yard", "push title uses label + friendly name")
    check(payload["body"] == "1 dog in frame", "push body carries the in-frame count")
    check(f"/api/events/{event_id}/snapshot.jpg?token=" in payload.get("image", ""),
          "push image URL is the media-token snapshot route")
    check(payload["data"]["url"].endswith(f"/events/{event_id}"),
          "push click-through targets the event page")

    # absence -> end (one empty frame past ABSENCE_TIMEOUT_S)
    gray = np.full((480, 704, 3), 114, dtype=np.uint8)
    last_seen = st.last_seen
    await engine.process("yard", t0 + 8 * 0.2 + 5.5, [], frame_bgr=gray)
    check(key not in engine._events, "label absence past 5 s ends the event")
    row = await db.get_event(event_id)
    check(row is not None and abs(row["end_time"] - last_seen) < 1e-3,
          "end_time == last time the label was seen")
    check(pipeline.counts.get(key) == 0, "live count zeroed on event end")
    check(recorder.clip_requests == [("yard", fid, st.start_time, last_seen)],
          "engine asked the recorder for the event clip")
    check(recorder.clip_path(event_id).is_file(),
          "mocked recorder finished -> clip file exists at the event's clip path")
    check(bool((await db.get_event(event_id))["has_clip"]),
          "has_clip flips true ONLY after the recorder wrote the clip file")
    check(engine.event_best_jpeg(fid) is not None,
          "best frame survives event end for late enrichment")

    types = ws_client.types()
    check("event_update" in types, "WS broadcast: event_update sent during the event")
    end_msgs = [m for m in ws_client.messages if m.get("type") == "event_end"]
    check(len(end_msgs) == 1 and end_msgs[0]["event"]["id"] == event_id
          and end_msgs[0]["event"]["end_time"] is not None
          and not bool(end_msgs[0]["event"]["has_clip"]),
          "WS broadcast: event_end carries end_time; has_clip is still false "
          "(the clip is assembled afterward)")
    check(types.index("event_new") < types.index("event_end"),
          "WS broadcast ordering: event_new precedes event_end")

    await pipeline.shutdown()
    await det.stop()
    await db.close()


def pipeline_checks() -> None:
    print("detector 7: REAL dfine_s -> tracker -> engine -> EventsPipeline")
    asyncio.run(_pipeline_cases())


class _NullMedia:
    """Media stub: no snapshot (enrichment stops early). The `labels`
    accumulation happens in _on_new/_on_update BEFORE enrichment, so a null
    media provider is enough to exercise it."""

    async def event_snapshot(self, fid: str, retries: int = 1):
        return None

    async def detect_dims(self, camera: str):
        return None


def _scene_payload(etype: str, fid: str, camera: str, primary: str,
                   scene_labels: list[str]) -> dict:
    """A native-engine-shaped payload whose scene carries multiple classes."""
    now = time.time()
    scene = [{"box": [10 + 40 * i, 10, 60 + 40 * i, 120], "label": lbl, "score": 0.9}
             for i, lbl in enumerate(scene_labels)]
    return {
        "type": etype,
        "after": {
            "id": fid, "camera": camera, "label": primary, "top_score": 0.9,
            "start_time": now, "has_snapshot": False,
            "snapshot": {"box": scene[0]["box"], "frame_time": now, "score": 0.9},
            "scene": scene,
        },
    }


async def _labels_cases() -> None:
    root = TMP / "labels"
    db = Database(root / "nvr.db")
    await db.connect()
    settings = SettingsStore(db)
    await settings.load()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    pipeline = EventsPipeline(db, _NullMedia(), WSManager(), FakePush(), settings, auth,
                              root / "snapshots")

    # cur cache: the pipeline reads current_count(camera, label); prime it.
    pipeline.update_count("yard", "person", 1)

    # -- new event: person + car in the opening scene --
    await pipeline.handle_event(
        _scene_payload("new", "native.multi", "yard", "person", ["person", "car"]))
    row = await db.get_event_by_frigate_id("native.multi")
    check(row is not None, "multi-object event row created")
    check(row["label"] == "person", "primary label preserved (back-compat)")
    check(row["labels"] == ["person", "car"],
          "labels lists BOTH classes present at event open")

    # -- update: a dog appears later -> labels grows, order preserved --
    await pipeline.handle_event(
        _scene_payload("update", "native.multi", "yard", "person", ["person", "car", "dog"]))
    row = await db.get_event_by_frigate_id("native.multi")
    check(row["labels"] == ["person", "car", "dog"],
          "labels accumulates a newly-appearing class across the event")
    check(row["label"] == "person", "primary label unchanged when the set grows")

    # -- an update with no NEW class does not duplicate labels --
    await pipeline.handle_event(
        _scene_payload("update", "native.multi", "yard", "person", ["car", "person"]))
    row = await db.get_event_by_frigate_id("native.multi")
    check(row["labels"] == ["person", "car", "dog"], "no duplicate labels on re-observation")

    # -- single-object event: labels is just [label] (back-compat) --
    pipeline.update_count("yard", "cat", 1)
    await pipeline.handle_event(
        _scene_payload("new", "native.solo", "yard", "cat", ["cat"]))
    row = await db.get_event_by_frigate_id("native.solo")
    check(row["labels"] == ["cat"], "single-object event -> labels == [label]")

    # -- doorbell/synthetic row (no scene, insert_event defaults) -> [label] --
    now = time.time()
    eid = await db.insert_event(frigate_id=f"doorbell.{int(now*1000)}", camera="yard",
                                label="doorbell", count=1, score=1.0, start_time=now,
                                end_time=now, has_clip=False, has_snapshot=False)
    drow = await db.get_event(eid)
    check(drow["labels"] == ["doorbell"],
          "doorbell insert (no labels arg) falls back to [label]")

    # -- legacy row (labels column literally '[]') falls back in the serializer --
    await db.conn.execute("UPDATE events SET labels='[]' WHERE id=?", (eid,))
    await db.conn.commit()
    drow = await db.get_event(eid)
    check(drow["labels"] == ["doorbell"],
          "empty labels column serializes back to [primary label] (legacy-safe)")

    await pipeline.shutdown()
    await db.close()


def labels_checks() -> None:
    print("multi-object: events accumulate a `labels` set (primary label kept)")
    asyncio.run(_labels_cases())


# ---- ingest fakes ----


class PipeProc:
    """Stands in for the ffmpeg ingest child: StreamReader stdout + kill/wait."""

    def __init__(self, chunks: list[bytes], eof: bool = False):
        self.stdout = asyncio.StreamReader()
        for chunk in chunks:
            self.stdout.feed_data(chunk)
        if eof:
            self.stdout.feed_eof()
        self.killed = False
        self.returncode: int | None = None
        self.pid = 4242
        self._done = asyncio.Event()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode or 0


class ScriptedSource(FrameSource):
    """FrameSource whose _spawn pops from a scripted process list."""

    def __init__(self, procs: list[PipeProc], **kwargs):
        super().__init__(**kwargs)
        self._procs = list(procs)

    async def _spawn(self):
        if not self._procs:
            await asyncio.sleep(3600)
        return self._procs.pop(0)


INGEST_W, INGEST_H = 4, 3
INGEST_FRAME_BYTES = INGEST_W * INGEST_H * 3


def _raw_frame(fill: int) -> bytes:
    return bytes([fill]) * INGEST_FRAME_BYTES


async def _frame_source_cases() -> None:
    # latest-frame DROP: three frames queued, slot keeps only the newest;
    # then silence past the (shrunk) stall timeout => watchdog respawn.
    proc1 = PipeProc([_raw_frame(1), _raw_frame(2), _raw_frame(3)])
    proc2 = PipeProc([])  # respawn target that never produces
    src = ScriptedSource(
        [proc1, proc2], name="cam", url="rtsp://x/cam_sub",
        width=INGEST_W, height=INGEST_H, detect_fps=5, on_frame=lambda: None,
        stall_timeout_s=0.25, backoff_initial_s=0.02, backoff_cap_s=0.1,
    )
    task = asyncio.create_task(src.run())
    await asyncio.sleep(0.1)
    item = src.take_latest()
    check(item is not None, "frames flow from the pipe into the slot")
    frame, ts = item
    check(frame.shape == (INGEST_H, INGEST_W, 3) and int(frame[0, 0, 0]) == 3,
          "slot holds only the NEWEST frame (drop, never queue)")
    check(abs(ts - time.time()) < 5.0, "frame carries an epoch timestamp")
    check(src.take_latest() is None, "take_latest clears the slot")

    await asyncio.sleep(0.45)
    check(proc1.killed, "staleness watchdog killed the stalled ffmpeg child")
    check(src.spawn_count >= 2, "watchdog respawned after the stall")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # EOF (child death) => backoff respawns until frames flow again
    procs = [PipeProc([], eof=True), PipeProc([], eof=True), PipeProc([_raw_frame(9)])]
    src2 = ScriptedSource(
        procs, name="cam2", url="rtsp://x/cam2_sub",
        width=INGEST_W, height=INGEST_H, detect_fps=5, on_frame=lambda: None,
        stall_timeout_s=5.0, backoff_initial_s=0.02, backoff_cap_s=0.1,
    )
    task2 = asyncio.create_task(src2.run())
    deadline = time.monotonic() + 3.0
    while src2.spawn_count < 3 and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    check(src2.spawn_count == 3, "ffmpeg exit triggers backoff respawns")
    await asyncio.sleep(0.05)
    item = src2.take_latest()
    check(item is not None and int(item[0][0, 0, 0]) == 9, "recovered source delivers frames again")
    task2.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task2
    check(procs[2].killed, "cancel kills the live ffmpeg child")


class DummySource:
    """Patched-in FrameSource replacement for IngestManager tests."""

    def __init__(self, name, url, width, height, detect_fps, on_frame, ffmpeg, **kwargs):
        self.name, self.url = name, url
        self.width, self.height, self.detect_fps = width, height, detect_fps
        self.ffmpeg = ffmpeg
        self._latest = None
        self.cancelled = False

    def take_latest(self):
        item = self._latest
        self._latest = None
        return item

    async def run(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def process(self, camera, frame_time, observations, frame_bgr=None):
        self.calls.append((camera, frame_time, list(observations), frame_bgr))


class StubDetector:
    def __init__(self) -> None:
        self.ready = False
        self.result = sv.Detections.empty()
        self.needs_reinit = False
        self.ok_calls = 0
        self.fail_calls = 0

    def detect(self, frame, dw, dh):
        return self.result

    # widened detector interface consumed by the self-heal ingest worker
    def note_detect_ok(self) -> None:
        self.ok_calls += 1

    def note_detect_failure(self) -> None:
        self.fail_calls += 1

    async def reinit(self, *, force: bool = False) -> bool:
        self.needs_reinit = False
        return True


def _ingest_cam(name: str, **over) -> dict:
    row = {
        "name": name, "detect_enabled": True, "detect_fps": 5,
        "detect_width": 704, "detect_height": 480,
        # Non-empty by default so the camera is a normal detecting camera; an
        # empty detect_objects now means record-only (no ingest source).
        "detect_objects": ["person"],
        "ip": "10.0.0.9", "username": "u", "password": "p",
        "main_url": "", "sub_url": "",
    }
    row.update(over)
    return row


async def _ingest_manager_cases() -> None:
    config = Config()
    stub_engine = StubEngine()
    stub_detector = StubDetector()

    real_source = ingest_module.FrameSource
    ingest_module.FrameSource = DummySource  # type: ignore[misc]
    try:
        mgr = IngestManager(stub_engine, stub_detector, config, ffmpeg_path="/fake/ffmpeg")
        await mgr.start()
        await mgr.reload([_ingest_cam("front"), _ingest_cam("side", detect_enabled=False)])
        await asyncio.sleep(0)  # let the source run() task actually start
        check(set(mgr._sources) == {"front"},
              "reload spawns sources only for detect-enabled cameras")
        check(mgr._sources["front"].url == f"{config.go2rtc_rtsp_url}/front_sub",
              "ingest consumes the go2rtc {name}_sub restream")
        check("front" in mgr._trackers, "one ByteTrackTracker per camera")

        # record-only: detect-enabled BUT empty detect_objects -> NO source
        # (the camera just records; no ingest ffmpeg / inference / GPU).
        await mgr.reload([
            _ingest_cam("front"),
            _ingest_cam("reconly", detect_objects=[]),
        ])
        await asyncio.sleep(0)
        check(set(mgr._sources) == {"front"},
              "empty detect_objects (record-only) spawns no ingest source")

        first = mgr._sources["front"]
        await mgr.reload([_ingest_cam("front"), _ingest_cam("side", detect_enabled=False)])
        check(mgr._sources["front"] is first, "unchanged camera keeps its running source")

        await mgr.reload([_ingest_cam("front", detect_fps=8)])
        check(mgr._sources["front"] is not first and first.cancelled,
              "detect_fps change restarts the camera's ffmpeg child")
        check(mgr._sources["front"].detect_fps == 8, "new source picked up the new fps")

        # worker path: detector NOT ready -> engine still fed (frame cache)
        frame = np.full((480, 704, 3), 50, dtype=np.uint8)
        stub_engine.calls.clear()  # ignore any prior empty heartbeat ticks
        mgr._sources["front"]._latest = (frame, 1234.5)
        # Nudge the single inference worker's wake Event.
        mgr._wake.set()
        await asyncio.sleep(0.1)
        fresh = [c for c in stub_engine.calls if c[3] is frame]
        check(len(fresh) == 1 and fresh[0][0] == "front" and fresh[0][2] == [],
              "worker feeds the engine even while the detector is down")

        # worker path: detector ready -> ByteTrack -> observations
        stub_detector.ready = True
        stub_detector.result = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 220.0, 220.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([16]),
        )
        for i in range(4):
            mgr._sources["front"]._latest = (frame, 1235.0 + i)
            mgr._wake.set()
            await asyncio.sleep(0.1)
        confirmed = [obs for (_, _, obs, _) in stub_engine.calls if obs]
        check(bool(confirmed) and confirmed[-1][0].label == "dog"
              and confirmed[-1][0].tracker_id >= 0,
              "worker: detect -> ByteTrack -> dog observations reach the engine")

        await mgr.reload([])
        check(not mgr._sources, "camera removal stops its source")

        mgr2 = IngestManager(stub_engine, stub_detector, config, ffmpeg_path=None)
        await mgr2.reload([_ingest_cam("front")])
        check(not mgr2._sources, "ffmpeg missing -> ingest disabled gracefully")
        await mgr2.stop()
        await mgr.stop()
    finally:
        ingest_module.FrameSource = real_source  # type: ignore[misc]


def ingest_checks() -> None:
    print("detector 8: ingest (argv, drop slot, watchdog, manager reload, worker)")
    check(
        build_ingest_args("rtsp://go2rtc:8554/yard_sub", 5, 704, 480)
        == [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", "tcp", "-timeout", "5000000",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", "rtsp://go2rtc:8554/yard_sub",
            "-vf", "fps=5,scale=704:480",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        "golden ffmpeg ingest argv (design doc §3.1 + byte-exact scale)",
    )
    asyncio.run(_frame_source_cases())
    asyncio.run(_ingest_manager_cases())


def boot_endpoint_checks() -> None:
    print("detector 9: app boot — health/detector endpoints (REQUIRE_GPU=1)")
    from app.main import app  # noqa: PLC0415

    with TestClient(app) as client:
        time.sleep(1.0)  # detector-start task reaches its terminal GPU check

        resp = client.get("/api/system/health")
        check(resp.status_code == 200, "GET /api/system/health is 200 without auth")
        health = resp.json()
        check(health["status"] == "ok" and health["version"] == APP_VERSION,
              "health status/version")
        check(health["detector"] == {"kind": "onnx", "ready": False, "device": None,
                                      "model": "dfine_s"},
              "health.detector: REQUIRE_GPU=1, GPU-less host -> ready:false, device:null")
        check(isinstance(health["go2rtc"], bool), "health.go2rtc is a boolean")
        check(isinstance(health["cameras_online"], int), "health.cameras_online is an int")
        check(set(health) == {"status", "version", "detector", "go2rtc", "cameras_online"},
              "health carries exactly the contract keys (no legacy fields)")

        models_dir = Path(os.environ["DATA_DIR"]) / "models"
        check(not list(models_dir.glob("*.onnx*")) if models_dir.exists() else True,
              "REQUIRE_GPU hard failure downloads nothing")

        resp = client.get("/api/system/detector")
        check(resp.status_code in (401, 403), "detector self-test requires auth")

        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        resp = client.get("/api/system/detector", headers={"Authorization": f"Bearer {token}"})
        check(resp.status_code == 200, "GET /api/system/detector with auth is 200")
        state = resp.json()
        check(state["ready"] is False and state["device"] is None
              and state["model"] == "dfine_s" and state["model_sha_ok"] is None
              and state["last_inference_ms"] is None,
              "detector self-test block matches the hard-failed state")
        check(state["per_camera"] == [], "per_camera empty with no cameras")


# ---------------- A. argv builders ----------------


def argv_builder_checks() -> None:
    print("recorder A: ffmpeg argv builders (golden)")
    args = build_segment_args(
        "rtsp://go2rtc:8554/front_door",
        "/media/native/recordings/front_door/%Y-%m-%d/%H/%M.%S.ts",
    )
    check(args == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",
        "-i", "rtsp://go2rtc:8554/front_door",
        # Video stream-copy, audio RE-ENCODED to AAC. NOT a blanket "-c copy":
        # the cameras' native G.711A (pcm_alaw) has no MPEG-TS mapping, so
        # copying it silently produced audio-less recordings — and therefore
        # audio-less event clips — on every camera. See build_segment_args.
        "-map", "0", "-c:v", "copy", "-c:a", "aac",
        "-f", "segment",
        "-segment_time", "10",
        "-segment_atclocktime", "1",
        "-reset_timestamps", "1",
        "-strftime", "1",
        "/media/native/recordings/front_door/%Y-%m-%d/%H/%M.%S.ts",
    ], "segment argv: video copy + AAC audio (pcm_alaw is not TS-muxable)")

    clip = build_clip_args(Path("/tmp/c.txt"), 5.0, 25.0, Path("/media/native/clips/7.mp4"))
    check(clip == [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", "/tmp/c.txt",
        "-ss", "5.000",
        "-t", "25.000",
        "-c", "copy", "-movflags", "+faststart",
        "/media/native/clips/7.mp4",
    ], "clip argv matches the design-doc command exactly (faststart, stream copy)")

    neg = build_clip_args(Path("/tmp/c.txt"), -3.2, -1.0, Path("/o.mp4"))
    check(neg[neg.index("-ss") + 1] == "0.000", "negative seek clamps to 0.000")
    check(neg[neg.index("-t") + 1] == "0.000", "negative duration clamps to 0.000")

    text = build_concat_list([Path("/a/b.ts"), Path("/a/c.ts")])
    check(
        text == "ffconcat version 1.0\nfile '/a/b.ts'\nfile '/a/c.ts'\n",
        "concat list: header + ordered file lines + trailing newline",
    )
    quoted = build_concat_list([Path("/a/it's.ts")])
    check("file '/a/it'\\''s.ts'" in quoted, "concat list escapes single quotes ffmpeg-style")

    cfg = make_config("paths")
    rec = Recorder(cfg, None, FakeSettings())
    check(
        rec.segment_pattern("cam1")
        == str(cfg.media_dir / "native" / "recordings" / "cam1" / "%Y-%m-%d" / "%H" / "%M.%S.ts"),
        "segment pattern lives under <media>/native/recordings/<cam>/",
    )
    check(
        rec.clip_path(42) == cfg.media_dir / "native" / "clips" / "42.mp4",
        "clip path is <media>/native/clips/<event_row_id>.mp4",
    )
    check(
        rec.segment_input_url("cam1") == "rtsp://127.0.0.1:1/cam1",
        "segment input consumes the go2rtc MAIN restream for the camera",
    )


# ---------------- B. segment parsing + window selection ----------------


def segment_index_checks() -> None:
    print("recorder B: filename timestamps + clip window selection")
    base = datetime(2026, 7, 1, 12, 0, 0)
    cam_dir = TMP / "segidx" / "front"

    p = seg_path(cam_dir, base)
    check(parse_segment_start(p) == base.timestamp(), "segment path round-trips to its epoch")
    check(parse_segment_start(Path("/x/2026-07-01/12/garbage.ts")) is None,
          "unparseable segment filename -> None")
    check(parse_segment_start(Path("/x/notadate/12/00.00.ts")) is None,
          "unparseable day directory -> None")
    check(parse_segment_start(Path("/x/2026-07-01/99/00.00.ts")) is None,
          "impossible hour (99) -> None")

    window_start = base.timestamp()                 # 12:00:00
    window_end = window_start + 30.0                # 12:00:30
    offsets = {
        -20: False,  # 11:59:40 — more than SEGMENT_SECONDS before the window
        -10: True,   # 11:59:50 — exactly window_start - 10: can hold the head
        0: True,
        10: True,
        20: True,
        29: True,    # just inside the window end
        30: False,   # exactly window_end: cannot intersect [start, end)
    }
    for off in offsets:
        make_seg(cam_dir, datetime.fromtimestamp(window_start + off))
    # garbage that must be tolerated, never selected
    (cam_dir / "2026-07-01" / "12" / "junk.ts").write_bytes(b"x")
    (cam_dir / "stray.txt").write_text("not a day dir")

    got = select_segments(cam_dir, window_start, window_end)
    got_offsets = [round(ts - window_start) for ts, _ in got]
    check(got_offsets == [-10, 0, 10, 20, 29],
          "selection = segments starting in [start-10, end), sorted ascending")
    check(all(path.is_file() for _, path in got), "selected entries are real files")
    check(-20 not in got_offsets, "segment older than the 10 s head rule excluded")
    check(30 not in got_offsets, "segment starting exactly at window_end excluded")

    # hour/day rollover: window spanning midnight picks from both day dirs
    cam2 = TMP / "segidx" / "gate"
    midnight = datetime(2026, 7, 2, 0, 0, 0)
    before = datetime(2026, 7, 1, 23, 59, 50)
    after = datetime(2026, 7, 2, 0, 0, 10)
    for dt in (before, midnight, after):
        make_seg(cam2, dt)
    got2 = select_segments(cam2, midnight.timestamp() - 2, midnight.timestamp() + 15)
    check([p.parent.parent.name for _, p in got2] == ["2026-07-01", "2026-07-02", "2026-07-02"],
          "window spanning midnight selects across day directories in order")

    check(select_segments(TMP / "segidx" / "missing", 0.0, 100.0) == [],
          "missing camera dir -> empty selection")
    check(select_segments(cam_dir, window_end, window_start) == [],
          "inverted window -> empty selection")


# ---------------- C. retention + low-disk guard ----------------


def retention_checks() -> None:
    print("recorder C: retention pruning + low-disk guard (fake trees)")
    now = time.time()
    root = TMP / "retention" / "recordings"

    old_a1 = hour_tree_at(root, "camA", "2026-06-20", "10", now - 8 * 86400)
    old_a2 = hour_tree_at(root, "camA", "2026-06-20", "11", now - 8 * 86400 + 10)
    new_a = hour_tree_at(root, "camA", "2026-07-05", "10", now - 100)
    old_b = hour_tree_at(root, "camB", "2026-06-19", "05", now - 9 * 86400)
    empty_old = root / "camA" / "2026-06-21" / "23"
    empty_old.mkdir(parents=True)
    os.utime(empty_old, (now - 8 * 86400, now - 8 * 86400))

    hours = iter_hour_dirs(root)
    check([h for _, h in hours][0] == old_b, "iter_hour_dirs sorts oldest first across cameras")
    check(len(hours) == 5, "iter_hour_dirs finds every hour dir incl. empty ones")

    removed = prune_recordings(root, now - 7 * 86400)
    check(set(removed) == {old_a1, old_a2, old_b, empty_old},
          "hour dirs older than continuous_days removed (incl. empty hour dir)")
    check(new_a.is_dir() and (new_a / "00.00.ts").is_file(), "fresh hour dir untouched")
    check(not (root / "camA" / "2026-06-20").exists()
          and not (root / "camB" / "2026-06-19").exists(),
          "emptied day dirs removed after pruning")
    check((root / "camA").is_dir(), "camera dir itself is kept")

    clips = TMP / "retention" / "clips"
    clips.mkdir(parents=True)
    old_clip = clips / "1.mp4"
    new_clip = clips / "2.mp4"
    stale_part = clips / ".9.part.mp4"
    note = clips / "notes.txt"
    for f, age in ((old_clip, 15 * 86400), (new_clip, 100), (stale_part, 15 * 86400), (note, 15 * 86400)):
        f.write_bytes(b"data")
        os.utime(f, (now - age, now - age))
    removed_clips = prune_clips(clips, now - 14 * 86400)
    check(set(removed_clips) == {old_clip, stale_part},
          "clips older than event_days removed (stale .part leftovers swept too)")
    check(new_clip.is_file() and note.is_file(), "fresh clip + non-mp4 files untouched")

    # low-disk guard: free space derived from how many hour dirs remain
    cfg = make_config("lowdisk")
    rec = Recorder(cfg, None, FakeSettings())
    lroot = cfg.recordings_dir
    h1 = hour_tree_at(lroot, "cam", "2026-07-01", "01", now - 5 * 86400)
    h2 = hour_tree_at(lroot, "cam", "2026-07-01", "02", now - 4 * 86400)
    h3 = hour_tree_at(lroot, "cam", "2026-07-01", "03", now - 3 * 86400)
    h4 = hour_tree_at(lroot, "cam", "2026-07-02", "04", now)  # actively written

    def fake_free() -> int:
        remaining = len(iter_hour_dirs(lroot))
        return (6 - remaining) * 2 * 1024**3  # 4 dirs -> 4 GB, 3 -> 6 GB

    rec._disk_free = fake_free
    check(fake_free() < LOW_DISK_BYTES, "fixture starts below the 5 GB floor")
    forced = rec._low_disk_prune(now)
    check(forced == [h1], "low-disk guard deletes exactly the oldest hour dir, then stops")
    check(h2.is_dir() and h3.is_dir() and h4.is_dir(), "newer + active hour dirs survive")

    # all dirs actively written -> nothing deletable, loud log, no crash
    cfg2 = make_config("lowdisk2")
    rec2 = Recorder(cfg2, None, FakeSettings())
    active = hour_tree_at(cfg2.recordings_dir, "cam", "2026-07-02", "05", now)
    rec2._disk_free = lambda: 1 * 1024**3
    check(rec2._low_disk_prune(now) == [] and active.is_dir(),
          "low-disk guard never touches actively-written hour dirs")

    rec2._disk_free = lambda: 50 * 1024**3
    check(rec2._low_disk_prune(now) == [], "low-disk guard is a no-op with plenty free")

    # retention_pass end-to-end honors the settings values
    cfg3 = make_config("retpass")
    rec3 = Recorder(cfg3, None, FakeSettings({"continuous_days": 1, "event_days": 2, "snapshot_days": 2}))
    keep_hour = hour_tree_at(cfg3.recordings_dir, "cam", "2026-07-05", "10", now - 3600)
    drop_hour = hour_tree_at(cfg3.recordings_dir, "cam", "2026-07-03", "10", now - 2 * 86400)
    cfg3.clips_dir.mkdir(parents=True, exist_ok=True)
    drop_clip = cfg3.clips_dir / "3.mp4"
    keep_clip = cfg3.clips_dir / "4.mp4"
    for f, age in ((drop_clip, 3 * 86400), (keep_clip, 3600)):
        f.write_bytes(b"x")
        os.utime(f, (now - age, now - age))
    rec3._disk_free = lambda: 50 * 1024**3
    result = rec3.retention_pass(now)
    check(result["hours"] == [drop_hour] and keep_hour.is_dir(),
          "retention_pass prunes recordings by settings.recording.continuous_days")
    check(result["clips"] == [drop_clip] and keep_clip.is_file(),
          "retention_pass prunes clips by settings.recording.event_days")
    check(result["low_disk"] == [], "retention_pass reports no forced deletions when disk is fine")


def hour_tree_at(root: Path, cam: str, day: str, hour: str, mtime: float) -> Path:
    hd = root / cam / day / hour
    hd.mkdir(parents=True, exist_ok=True)
    f = hd / "00.00.ts"
    f.write_bytes(b"\x47" * 188)
    os.utime(f, (mtime, mtime))
    return hd


# ---------------- D. clip extraction ----------------


def clip_extraction_checks() -> None:
    print("recorder D: event clip extraction (mocked ffmpeg subprocess)")
    asyncio.run(_clip_extraction_cases())
    _real_ffmpeg_case()


async def _clip_extraction_cases() -> None:
    cfg = make_config("clips")
    db = Database(cfg.data_dir / "clip.db")
    await db.connect()
    await db.upsert_camera(camera_row("front"))

    # Segments every 10 s around the event; the event runs 10:00:20-10:00:35
    # => window [10:00:15, 10:00:40], head rule pulls in the 10:00:10 segment.
    t0 = datetime(2026, 7, 2, 10, 0, 0).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    segs = {off: make_seg(cam_dir, datetime.fromtimestamp(t0 + off)) for off in (0, 10, 20, 30, 40)}
    start_time, end_time = t0 + 20, t0 + 35
    event_id = await db.insert_event(
        "native.1751-aaa", "front", "person", 1, 0.91, start_time, end_time=end_time
    )

    rec = Recorder(cfg, db, FakeSettings())
    rec._ffmpeg_path = "/fake/ffmpeg"  # force-enable without a real binary
    captured: dict = {}

    async def fake_run_ok(args: list[str]) -> int:
        captured["args"] = list(args)
        concat = Path(args[args.index("-i") + 1])
        captured["concat"] = concat.read_text()
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE-MP4")
        return 0

    rec._run_ffmpeg = fake_run_ok
    out_path = await rec.extract_clip("front", "native.1751-aaa", start_time, end_time)

    check(out_path == rec.clip_path(event_id), "extract_clip returns clip_path(event row id)")
    check(out_path.is_file() and out_path.read_bytes() == b"FAKE-MP4",
          "clip file lands at <clips>/<event_id>.mp4 (atomic rename from .part)")
    expected_concat = cfg.clips_dir / f".{event_id}.concat.txt"
    expected_part = cfg.clips_dir / f".{event_id}.part.mp4"
    check(captured["args"] == build_clip_args(
        expected_concat,
        (start_time - CLIP_PAD_S) - (t0 + 10),   # window_start - first_segment_start
        (end_time + CLIP_PAD_S) - (start_time - CLIP_PAD_S),
        expected_part,
    ), "clip argv: golden build_clip_args with computed seek into the first segment")
    check(captured["args"][captured["args"].index("-ss") + 1] == "5.000"
          and captured["args"][captured["args"].index("-t") + 1] == "25.000",
          "seek 5.000 s / duration 25.000 s for a 15 s event with 5 s pads")
    check(captured["concat"] == build_concat_list([segs[10], segs[20], segs[30]]),
          "concat list = exactly the covering segments (head rule in, end boundary out)")
    check(not expected_concat.exists() and not expected_part.exists(),
          "concat list + .part temp files cleaned up after success")
    row = await db.get_event(event_id)
    check(row["has_clip"] is True, "event row flipped to has_clip=True")

    # ffmpeg failure: no clip file, temps cleaned, has_clip untouched
    event2 = await db.insert_event(
        "native.1751-bbb", "front", "dog", 1, 0.8, start_time, end_time=end_time
    )

    async def fake_run_fail(args: list[str]) -> int:
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"HALF")
        return 1

    rec._run_ffmpeg = fake_run_fail
    out2 = await rec.extract_clip("front", "native.1751-bbb", start_time, end_time)
    check(out2 is None and not rec.clip_path(event2).exists(),
          "ffmpeg rc!=0 -> no clip file served")
    check(not (cfg.clips_dir / f".{event2}.part.mp4").exists()
          and not (cfg.clips_dir / f".{event2}.concat.txt").exists(),
          "failed extraction leaves no temp files behind")
    check((await db.get_event(event2))["has_clip"] is False,
          "failed extraction never sets has_clip")

    # no segments in the window (recorder was down) -> logged skip, no retry
    event3 = await db.insert_event(
        "native.1751-ccc", "front", "car", 1, 0.7, t0 + 5000, end_time=t0 + 5010
    )
    rec._run_ffmpeg = fake_run_ok
    out3 = await rec.extract_clip("front", "native.1751-ccc", t0 + 5000, t0 + 5010)
    check(out3 is None and (await db.get_event(event3))["has_clip"] is False,
          "no covering segments -> None, has_clip stays false")

    check(await rec.extract_clip("front", "native.does-not-exist", start_time, end_time) is None,
          "unknown frigate_id -> None, no crash")

    rec_no_ffmpeg = Recorder(cfg, db, FakeSettings())
    rec_no_ffmpeg._ffmpeg_path = None
    check(await rec_no_ffmpeg.extract_clip("front", "native.1751-aaa", start_time, end_time) is None,
          "ffmpeg absent -> extract_clip degrades to None")

    # schedule_clip: returns immediately, extraction happens after the delay
    event4 = await db.insert_event(
        "native.1751-ddd", "front", "cat", 1, 0.9, start_time, end_time=end_time
    )
    rec._running = True
    rec.clip_delay_s = 0.2
    await rec.schedule_clip("front", "native.1751-ddd", start_time, end_time)
    check(not rec.clip_path(event4).exists(),
          "schedule_clip returns before extraction runs (fire-and-forget)")
    deadline = time.monotonic() + 5.0
    while not rec.clip_path(event4).exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    check(rec.clip_path(event4).is_file() and (await db.get_event(event4))["has_clip"] is True,
          "scheduled clip lands after the post-end delay and flips has_clip")
    await rec.stop()
    await db.close()


def _real_ffmpeg_case() -> None:
    """End-to-end concat with the real ffmpeg — only when one is on PATH
    (per project rules: feature-detect, never install)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("  -- real-ffmpeg clip concat skipped (no ffmpeg on this host; mocked path covered)")
        return
    asyncio.run(_real_ffmpeg_async(ffmpeg))


async def _real_ffmpeg_async(ffmpeg: str) -> None:
    cfg = make_config("realclip")
    db = Database(cfg.data_dir / "real.db")
    await db.connect()
    t0 = datetime(2026, 7, 2, 9, 0, 10).timestamp()
    cam_dir = cfg.recordings_dir / "front"
    gen_fail = False
    for off in (0, 10):
        dest = seg_path(cam_dir, datetime.fromtimestamp(t0 + off))
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
             "-c:v", "mpeg4", "-f", "mpegts", str(dest)],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not dest.is_file():
            gen_fail = True
            break
    if gen_fail:
        print("  -- real-ffmpeg clip concat skipped (test segment generation failed)")
        await db.close()
        return
    # Window covers both segments with a 0.5 s seek into the first.
    start_time = t0 + 5.5
    end_time = start_time
    event_id = await db.insert_event(
        "native.real-1", "front", "person", 1, 0.9, start_time, end_time=end_time
    )
    rec = Recorder(cfg, db, FakeSettings())
    rec._ffmpeg_path = ffmpeg
    out = await rec.extract_clip("front", "native.real-1", start_time, end_time)
    check(out is not None and out.is_file() and out.stat().st_size > 0,
          "REAL ffmpeg: concat + stream-copy cut produced a clip file")
    head = out.read_bytes()[:64]
    check(b"ftyp" in head, "REAL ffmpeg: output is an mp4 (ftyp box near the start)")
    check((await db.get_event(event_id))["has_clip"] is True,
          "REAL ffmpeg: has_clip flipped on the event row")
    await db.close()


# ---------------- E. lifecycle / supervision ----------------


class FakeProc:
    """Stand-in for asyncio.subprocess.Process (never exits on its own)."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stderr = None
        self.terminated = False
        self._done = asyncio.Event()

    async def wait(self) -> int | None:
        await self._done.wait()
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._done.set()

    def kill(self) -> None:
        self.returncode = -9
        self._done.set()


def lifecycle_checks() -> None:
    print("recorder E: lifecycle, ffmpeg-absent degradation, supervision bookkeeping")
    asyncio.run(_lifecycle_cases())


async def _lifecycle_cases() -> None:
    orig_detect = recorder_module._ffmpeg_available
    try:
        # -- ffmpeg absent: one warning, zero cameras managed, still answers
        cfg = make_config("lifecycle-noff")
        db = Database(cfg.data_dir / "life.db")
        await db.connect()
        await db.upsert_camera(camera_row("livecam"))
        rec = Recorder(cfg, db, FakeSettings())
        await rec.reload()  # before start(): must be a harmless no-op
        check(rec.status() == {}, "reload() before start() is a no-op")
        recorder_module._ffmpeg_available = lambda: None
        await rec.start()
        check(rec._ffmpeg_path is None and rec.status() == {},
              "no ffmpeg -> acts as if zero cameras are record-enabled; status() answers")
        check(cfg.recordings_dir.is_dir() and cfg.clips_dir.is_dir(),
              "start() creates the media directory tree")
        await rec.reload()
        check(rec.status() == {}, "reload() without ffmpeg stays empty")
        await rec.stop()
        await rec.stop()
        check(True, "stop() is idempotent (double stop)")

        # -- fake child process: spawn args, status, reload-driven teardown
        cfg2 = make_config("lifecycle-fake")
        db2 = Database(cfg2.data_dir / "life2.db")
        await db2.connect()
        await db2.upsert_camera(camera_row("livecam"))
        rec2 = Recorder(cfg2, db2, FakeSettings())
        recorder_module._ffmpeg_available = lambda: "/fake/ffmpeg"
        spawned: list[list[str]] = []
        procs: list[FakeProc] = []

        async def fake_spawn(args: list[str]) -> FakeProc:
            spawned.append(list(args))
            proc = FakeProc()
            procs.append(proc)
            return proc

        rec2._spawn = fake_spawn
        await rec2.start()
        await asyncio.sleep(0.1)  # let the camera task reach its watch loop
        check(spawned and spawned[0] == build_segment_args(
            "rtsp://127.0.0.1:1/livecam", rec2.segment_pattern("livecam")
        ), "per-camera child spawned with the golden segment argv")
        status = rec2.status()
        check(status.get("livecam", {}).get("recording") is True,
              "status() reports recording=true while the child lives")
        check(status["livecam"]["last_segment_age_s"] is None,
              "no segments yet -> last_segment_age_s null")
        now = time.time()
        check(hour_dir(rec2.camera_dir("livecam"), now).is_dir()
              and hour_dir(rec2.camera_dir("livecam"), now + 3600).is_dir(),
              "current + next hour dirs pre-created (ffmpeg strftime does not mkdir)")

        await db2.upsert_camera(camera_row("livecam", record_enabled=False))
        await rec2.reload()
        check(rec2.status() == {}, "reload() stops the recorder when record_enabled flips off")
        check(procs[0].terminated, "the ffmpeg child is terminated on teardown")

        await db2.upsert_camera(camera_row("livecam", record_enabled=True))
        await rec2.reload()
        await asyncio.sleep(0.1)
        check(rec2.status().get("livecam", {}).get("recording") is True,
              "reload() restarts the recorder when record_enabled flips back on")
        await rec2.stop()
        check(rec2.status() == {} and procs[-1].returncode is not None,
              "stop() tears down every camera child")
        await db.close()
        await db2.close()
    finally:
        recorder_module._ffmpeg_available = orig_detect


# ---------------- F. media route: clip serving through the app ----------------


def media_route_checks() -> None:
    print("recorder F: GET /api/events/{id}/clip.mp4 through the real app")
    from app.main import app  # noqa: PLC0415 — after env setup at module top

    with TestClient(app) as client:
        resp = client.post("/api/auth/login", json={"password": "test-password"})
        assert resp.status_code == 200, resp.text
        token = resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Rows inserted out-of-band (sqlite3 side connection; WAL allows it) —
        # the route only needs the row + the recorder's clip file on disk.
        db_file = Path(os.environ["DATA_DIR"]) / "nvr.db"
        conn = sqlite3.connect(db_file)
        now = time.time()
        cur = conn.execute(
            "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
            " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("native.9000-fff", "front", "person", 2, 0.88, now - 60, now - 30, 1, 0, "[]"),
        )
        native_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
            " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("doorbell.front.123", "front", "doorbell", 1, 1.0, now - 50, now - 50, 0, 0, "[]"),
        )
        doorbell_id = cur.lastrowid
        conn.commit()
        conn.close()

        resp = client.get(f"/api/events/{native_id}/clip.mp4", headers=headers)
        check(resp.status_code == 404, "clip 404s before the recorder has written the file")
        # The row carries has_clip=1 while the file is still absent. That used to
        # make _clip_state answer "ready" — a state with no entry in
        # _CLIP_STATE_DETAIL — so the 404 fell back to a generic "Clip not
        # available". Harmless-looking here, but the same "ready" answer is what
        # both clients gate their player on, so they would mount a video element
        # against a URL that 404s. The route now tells _clip_state the file is
        # missing, so the answer is a real state with a real message.
        #
        # Asserted as "one of the real states" rather than a specific string:
        # which one depends on whether the camera row exists in the fixture,
        # and that is not what this check is about.
        from app.routers.events import _CLIP_STATE_DETAIL
        check(resp.json()["detail"] in set(_CLIP_STATE_DETAIL.values()),
              "...and the 404 detail names a REAL clip state, not the generic fallback")

        clips_dir = Path(os.environ["MEDIA_DIR"]) / "native" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        data = bytes(range(48)) * 4  # 192 recognizable bytes
        clip_file = clips_dir / f"{native_id}.mp4"
        clip_file.write_bytes(data)

        resp = client.get(f"/api/events/{native_id}/clip.mp4", headers=headers)
        check(resp.status_code == 200 and resp.content == data,
              "clip served from <media>/native/clips/<event_id>.mp4")
        check(resp.headers["content-type"] == "video/mp4", "clip content-type video/mp4")

        resp = client.get(f"/api/events/{native_id}/clip.mp4",
                          headers={**headers, "Range": "bytes=4-9"})
        check(resp.status_code == 206 and resp.content == data[4:10],
              "Range requests served (206 partial content)")
        check(resp.headers.get("content-range", "").startswith("bytes 4-9/"),
              "Content-Range header present for partial responses")

        resp = client.get(f"/api/events/{native_id}/clip.mp4?token={token}")
        check(resp.status_code == 200, "media route also accepts ?token= (push-image path)")

        # A doorbell press now holds its event open while the visitor is there
        # and schedules a real clip, so `doorbell.` is no longer a blanket
        # never-has-a-clip prefix. What decides it is the row's SHAPE:
        #
        #   end_time == start_time  -> closed AT the press; never eligible.
        #                              (every row predating this feature, a press
        #                              with recording off, a repeat press during
        #                              a visit, a privacy-aborted visit)
        #   end_time  > start_time  -> genuinely held open; clip-eligible.
        #
        # This fixture row is the FIRST shape (both stamps are now-50), so it
        # must still be refused outright — that is what keeps the operator's
        # doorbell history from retroactively reading as a recorder failure.
        resp = client.get(f"/api/events/{doorbell_id}/clip.mp4", headers=headers)
        check(resp.status_code == 404 and resp.json()["detail"] == "This event has no clip",
              "a doorbell row closed AT the press is still refused outright (marker shape)")

        # A HELD-OPEN doorbell row with an assembled clip must actually serve —
        # the whole point of the feature.
        conn = sqlite3.connect(db_file)
        cur = conn.execute(
            "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
            " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("doorbell.front.456", "front", "doorbell", 1, 1.0, now - 90, now - 45, 1, 0, "[]"),
        )
        held_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO events (frigate_id, camera, label, count, score, start_time,"
            " end_time, has_clip, has_snapshot, zones) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("cameraai.front.7", "front", "person", 1, 1.0, now - 40, now - 20, 0, 0, "[]"),
        )
        cameraai_id = cur.lastrowid
        conn.commit()
        conn.close()
        (clips_dir / f"{held_id}.mp4").write_bytes(data)
        resp = client.get(f"/api/events/{held_id}/clip.mp4", headers=headers)
        check(resp.status_code == 200 and resp.content == data,
              "a HELD-OPEN doorbell event with an assembled clip IS served")

        # ...while the other synthetic kinds stay refused regardless of shape —
        # note this cameraai row has end_time > start_time, so it would slip
        # through if the split were keyed on shape alone rather than prefix.
        resp = client.get(f"/api/events/{cameraai_id}/clip.mp4", headers=headers)
        check(resp.status_code == 404 and resp.json()["detail"] == "This event has no clip",
              "cameraai.* is STILL refused even with a held-open shape (split did not widen)")

        detail = client.get(f"/api/events/{native_id}", headers=headers).json()
        check(detail["clip_url"] == f"/api/events/{native_id}/clip.mp4",
              "event detail advertises the same clip URL the recorder backs")

        resp = client.delete(f"/api/events/{native_id}", headers=headers)
        check(resp.status_code == 204 and not clip_file.exists(),
              "deleting the event removes the recorder's clip file")


def streams_config_checks() -> None:
    print("streams: doorbell _sub uses the main source (AD410 has no substream)")

    # Non-doorbell camera: `_sub` keeps the derived subtype=1 substream so
    # multi-camera walls stay on the low-bitrate feed.
    turret = camera_row("turret")
    turret["model"] = "IP5M-T1277EW-AI"
    turret["ip"] = "192.0.2.80"
    check(not is_doorbell(turret), "turret is not a doorbell")
    t_sources = stream_sources(turret)
    t_main = t_sources["turret"][0]
    t_sub = t_sources[sub_stream_name("turret")][0]
    check(t_main == default_stream_url(turret, 0), "turret main source is subtype=0")
    check(t_sub == default_stream_url(turret, 1),
          "non-doorbell _sub still uses subtype=1 (bandwidth)")
    check(t_sub != t_main, "non-doorbell _sub differs from main")

    # Doorbell (AD410): `_sub` must equal the MAIN source (subtype=0), because
    # the AD410 serves no usable subtype=1 substream.
    door = camera_row("doorbell")  # camera_row defaults model=AD410
    check(is_doorbell(door), "AD410 detected as doorbell by model")
    d_sources = stream_sources(door)
    d_main = d_sources["doorbell"][0]
    d_sub = d_sources[sub_stream_name("doorbell")][0]
    check(d_main == default_stream_url(door, 0), "doorbell main source is subtype=0")
    check(d_sub == d_main,
          "doorbell _sub source equals the main source (subtype=0)")
    check(d_sub == default_stream_url(door, 0),
          "doorbell _sub is the subtype=0 URL, not subtype=1")

    # Detection is robust to model-string casing/whitespace and to the probed
    # capability flag standing in for the model string.
    lower = camera_row("db_lower")
    lower["model"] = "  ad410 "
    check(is_doorbell(lower), "doorbell detected case-insensitively / trimmed")
    by_cap = camera_row("db_cap")
    by_cap["model"] = "UNKNOWN-MODEL"
    by_cap["capabilities"] = {"doorbell": True}
    check(is_doorbell(by_cap), "doorbell detected via capabilities flag")
    cap_sources = stream_sources(by_cap)
    check(cap_sources[sub_stream_name("db_cap")][0] == cap_sources["db_cap"][0],
          "capability-flagged doorbell also gets main-sourced _sub")

    # A doorbell with a main_url override: `_sub` follows the override.
    over = camera_row("db_over")
    over["main_url"] = "rtsp://192.0.2.99:7447/main"
    o_sources = stream_sources(over)
    check(o_sources[sub_stream_name("db_over")][0] == "rtsp://192.0.2.99:7447/main",
          "doorbell _sub honors the main_url override")


def webrtc_candidate_checks() -> None:
    print("streams: WebRTC auto host-candidate derivation + readiness surfacing")
    stun = f"stun:{streams_module.WEBRTC_PORT}"
    env_key = streams_module.WEBRTC_HOST_ENV
    saved_env = os.environ.get(env_key)
    saved_auto = streams_module._auto_lan_ipv4

    def base_settings(candidates=None, public_url=""):
        return {"system": {"webrtc_candidates": list(candidates or []),
                           "public_url": public_url}}

    try:
        # --- pure helpers ---
        check(streams_module._is_docker_bridge_ipv4("172.18.0.2"),
              "172.18.0.2 flagged as a docker-bridge address (skipped)")
        check(streams_module._is_docker_bridge_ipv4("172.17.0.9"),
              "172.17.0.9 (docker0) flagged as a docker-bridge address")
        check(not streams_module._is_docker_bridge_ipv4("192.168.1.5"),
              "192.168.1.5 is a real LAN IP (not docker-bridge)")
        check(not streams_module._is_docker_bridge_ipv4("10.0.0.4"),
              "10.0.0.4 is a real LAN IP (not docker-bridge)")
        check(streams_module._public_url_ipv4("http://192.168.4.4:8443") == "192.168.4.4",
              "PUBLIC_URL IP literal host extracted")
        check(streams_module._public_url_ipv4("https://nvr.tailnet.ts.net") is None,
              "PUBLIC_URL hostname yields no IP candidate")
        check(streams_module._host_to_candidate("192.168.1.10") == "192.168.1.10:8555",
              "bare host gets the :8555 port")
        check(streams_module._host_to_candidate("192.168.1.10:9000") == "192.168.1.10:9000",
              "explicit host:port kept verbatim")

        # Force the auto-derivation off so the derived host comes only from the
        # source under test (this GPU-less/CI host would otherwise return its
        # real LAN IP and make the assertions host-dependent).
        streams_module._auto_lan_ipv4 = lambda: None
        os.environ.pop(env_key, None)

        # --- (d) none available: stun-only, ready:false, still valid ---
        st = webrtc_status(base_settings())
        check(st["candidates"] == [stun], "no host anywhere -> stun-only candidate list")
        check(st["ready"] is False and st["detected_ip"] is None and st["source"] is None,
              "no host -> ready:false, no detected_ip/source")
        cfg = build_config([], base_settings())
        check(cfg["webrtc"]["candidates"] == [stun],
              "build_config stays valid (stun-only) when no host is detected")

        # --- manual candidates present: ready even without a derived host ---
        st = webrtc_status(base_settings(["10.0.0.5:8555"]))
        check(st["candidates"] == ["10.0.0.5:8555", stun], "manual candidate + stun, order-stable")
        check(st["ready"] is True, "a manual candidate makes WebRTC ready")
        check(st["detected_ip"] is None and st["source"] is None,
              "manual-only readiness reports no auto-detected host")

        # --- (c) PUBLIC_URL IP literal derives a host candidate ---
        st = webrtc_status(base_settings(public_url="http://192.168.4.4:8443"))
        check(st["candidates"] == ["192.168.4.4:8555", stun],
              "PUBLIC_URL IP literal -> derived host candidate + stun")
        check(st["ready"] is True and st["detected_ip"] == "192.168.4.4"
              and st["source"] == "public_url",
              "PUBLIC_URL-derived host reported as ready/source=public_url")

        # --- (a) env override wins over PUBLIC_URL, accepts a hostname ---
        os.environ[env_key] = "nvr.local"
        st = webrtc_status(base_settings(public_url="http://192.168.4.4:8443"))
        check(st["candidates"] == ["nvr.local:8555", stun],
              "SENTINEL_WEBRTC_HOST hostname wins over PUBLIC_URL and gets :8555")
        check(st["detected_ip"] == "nvr.local" and st["source"] == "env",
              "env host reported as detected_ip with source=env")

        # env host already present in the manual list -> no duplicate
        os.environ[env_key] = "192.168.1.10"
        st = webrtc_status(base_settings(["192.168.1.10:8555"]))
        check(st["candidates"] == ["192.168.1.10:8555", stun],
              "derived host equal to a manual entry is de-duplicated")

        # --- (b) auto-derived LAN IPv4 used when no env/public url ---
        os.environ.pop(env_key, None)
        streams_module._auto_lan_ipv4 = lambda: "192.168.7.7"
        st = webrtc_status(base_settings())
        check(st["candidates"] == ["192.168.7.7:8555", stun]
              and st["ready"] is True and st["source"] == "auto",
              "auto-derived LAN IPv4 appended as a host candidate (source=auto)")

        # a docker-bridge auto result is already filtered inside _auto_lan_ipv4;
        # simulate that filter returning None -> falls through to stun-only.
        streams_module._auto_lan_ipv4 = lambda: None
        st = webrtc_status(base_settings())
        check(st["ready"] is False, "auto-derivation returning None -> not ready")
    finally:
        streams_module._auto_lan_ipv4 = saved_auto
        if saved_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = saved_env


def main() -> None:
    # detector / ingest / pipeline sections (pins first: import hygiene)
    detector_pin_checks()
    preprocess_checks()
    decode_checks()
    model_store_checks()
    require_gpu_checks()
    dfine_n_checks()
    pipeline_checks()
    labels_checks()
    ingest_checks()
    streams_config_checks()
    webrtc_candidate_checks()
    # recorder sections
    argv_builder_checks()
    segment_index_checks()
    retention_checks()
    clip_extraction_checks()
    lifecycle_checks()
    media_route_checks()
    # app-boot endpoint shapes last (boots the real app twice)
    boot_endpoint_checks()
    print(f"\nALL {PASS} CHECKS PASSED (detector/ingest/pipeline + recorder sections)")


if __name__ == "__main__":
    main()

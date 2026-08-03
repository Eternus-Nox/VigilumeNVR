"""Multi-box annotation smoke suite.

Goal under test: "All items that are detected and counted should be boxed in
the screenshot." The annotated event snapshot must draw EVERY detected/counted
object (each with its own label + score), rescale every box the same way, and
summarize all counts in the banner — while the legacy single-box path stays
intact for doorbell/audio/legacy rows.

Covers all three edited files:

  1. annotate.annotate_event_snapshot — multi-box scene drawing, uniform
     rescale of every box, per-label banner, single-box backward compat,
     empty-scene fallback, box clamping.
  2. native.engine.DetectionEngine — synthesized payload carries
     ``after["scene"]`` with every counted object (all labels), refreshed with
     the best frame.
  3. events_pipeline.EventsPipeline — a real enrich over a native scene payload
     produces a saved snapshot with multiple boxes drawn.

CPU-only, no network, no GPU, no real model. Usage:

    python backend/tests/annotate_multibox_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Clean env before app config is instantiated (mirrors native_smoke.py).
for _i in (1, 2, 3):
    for _sfx in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{_i}_{_sfx}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-multibox-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import asyncio  # noqa: E402
import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.annotate import (  # noqa: E402
    _scene_banner,
    annotate_event_snapshot,
    plural_label,
)
from app.auth import AuthService  # noqa: E402
from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402
from app.events_pipeline import EventsPipeline, _scene_of  # noqa: E402
from app.native.engine import (  # noqa: E402
    DetectionEngine,
    Observation,
    _CameraState,
)
from app.native.media import NativeMediaProvider  # noqa: E402
from app.notify.push import PushSendResult  # noqa: E402
from app.settings_store import SettingsStore  # noqa: E402
from app.ws import WSManager  # noqa: E402

PASS = 0
BG = (114, 114, 114)  # uniform detect-frame background


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- pixel helpers ----------------


def _uniform_jpeg(w: int, h: int) -> bytes:
    frame = np.full((h, w, 3), BG, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(buf)


def _frame_region(result: np.ndarray, h: int) -> np.ndarray:
    """Strip the banner (vstacked ON TOP) and return the h-row image body."""
    strip = result.shape[0] - h
    return result[strip:, :, :]


def _changed_mask(region: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that differ from the flat background by more than
    JPEG noise (drawn box borders / label chips are strongly saturated)."""
    diff = np.abs(region.astype(np.int32) - np.array(BG, dtype=np.int32))
    return np.any(diff > 30, axis=2)


def _box_drawn(mask: np.ndarray, box) -> bool:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return bool(mask[y1 : y2 + 1, x1 : x2 + 1].any())


def _region_empty(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    return not bool(mask[y1:y2, x1:x2].any())


# ---------------- 1. annotate: multi-box scene ----------------


def annotate_scene_checks() -> None:
    print("annotate 1: multi-box scene draws every counted object")
    w, h = 640, 480
    jpeg = _uniform_jpeg(w, h)
    a = [50, 50, 150, 300]    # person
    b = [250, 60, 360, 320]   # person
    d = [450, 200, 600, 400]  # dog
    scene = [
        {"box": a, "label": "person", "score": 0.87},
        {"box": b, "label": "person", "score": 0.80},
        {"box": d, "label": "dog", "score": 0.95},
    ]
    # box=None on purpose: with a scene present the single-box arg is ignored.
    out = annotate_event_snapshot(jpeg, None, "person", 0.87, 2, None, scene)
    check(out is not None and out[:2] == b"\xff\xd8", "annotated JPEG decodes")
    result = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    check(result is not None, "result image decodes")
    check(result.shape[1] == w and result.shape[0] > h,
          "width preserved; height grew by the banner strip")
    mask = _changed_mask(_frame_region(result, h))
    check(_box_drawn(mask, a), "box drawn at person A region")
    check(_box_drawn(mask, b), "box drawn at person B region")
    check(_box_drawn(mask, d), "box drawn at dog region")
    check(_region_empty(mask, w - 35, 0, w, 45),
          "empty corner (no object) has no drawn content")


def annotate_rescale_checks() -> None:
    print("annotate 2: every scene box rescaled by the SAME detect_dims factor")
    w, h = 640, 480
    jpeg = _uniform_jpeg(w, h)
    # Scene given in 320x240 detect pixels; image is 640x480 -> x2 rescale.
    scene = [
        {"box": [150, 100, 200, 180], "label": "person", "score": 0.9},  # -> 300,200,400,360
        {"box": [20, 20, 60, 90], "label": "dog", "score": 0.8},          # -> 40,40,120,180
    ]
    out = annotate_event_snapshot(jpeg, None, "person", 0.9, 2, (320, 240), scene)
    check(out is not None, "rescaled annotation decodes")
    result = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    mask = _changed_mask(_frame_region(result, h))
    check(_box_drawn(mask, [300, 200, 400, 360]), "first box landed at 2x-scaled coords")
    check(_box_drawn(mask, [40, 40, 120, 180]), "second box landed at 2x-scaled coords")
    check(_region_empty(mask, 150, 110, 190, 170),
          "nothing drawn at the un-rescaled coords (proves the rescale ran)")


def annotate_clamp_checks() -> None:
    print("annotate 3: boxes are clamped to the image (no crash, no OOB)")
    w, h = 640, 480
    jpeg = _uniform_jpeg(w, h)
    scene = [
        {"box": [600, 400, 900, 700], "label": "person", "score": 0.9},  # spills off-frame
        {"box": [-50, -50, 100, 120], "label": "dog", "score": 0.8},     # negative origin
    ]
    out = annotate_event_snapshot(jpeg, None, "person", 0.9, 2, None, scene)
    check(out is not None, "out-of-bounds boxes annotate without error")
    result = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    check(result is not None and result.shape[1] == w, "clamped result still decodes at width")
    mask = _changed_mask(_frame_region(result, h))
    check(_box_drawn(mask, [600, 400, 639, 479]), "spill box drawn clamped inside bottom-right")
    check(_box_drawn(mask, [0, 0, 100, 120]), "negative box drawn clamped to origin")


def annotate_banner_checks() -> None:
    print("annotate 4: banner summarizes every counted label")
    scene = [
        {"box": [0, 0, 1, 1], "label": "person", "score": 0.9},
        {"box": [0, 0, 1, 1], "label": "person", "score": 0.8},
        {"box": [0, 0, 1, 1], "label": "dog", "score": 0.7},
    ]
    check(_scene_banner(scene) == "2 people, 1 dog",
          "scene banner: '2 people, 1 dog' (count desc, pluralized)")
    check(_scene_banner([{"box": [0, 0, 1, 1], "label": "car", "score": 0.5}]) == "1 car",
          "single-object scene banner")
    check(_scene_banner([]) == "", "empty scene -> empty banner (single-label fallback)")
    check(plural_label("person", 2) == "2 people", "plural_label irregular plural intact")


def annotate_single_box_checks() -> None:
    print("annotate 5: single-box backward-compat path (scene absent/empty)")
    w, h = 640, 480
    jpeg = _uniform_jpeg(w, h)
    a = [50, 50, 150, 300]
    # scene=None -> legacy single box
    out = annotate_event_snapshot(jpeg, a, "person", 0.9, 1, None, None)
    check(out is not None, "legacy single-box annotation decodes")
    result = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    mask = _changed_mask(_frame_region(result, h))
    check(_box_drawn(mask, a), "the single box is drawn")
    check(_region_empty(mask, 400, 50, 600, 300),
          "no phantom second box drawn on the single-box path")

    # scene=[] -> falls back cleanly to the single box + single-label banner
    out2 = annotate_event_snapshot(jpeg, a, "person", 0.9, 3, None, [])
    check(out2 is not None, "empty scene falls back without error")
    result2 = cv2.imdecode(np.frombuffer(out2, np.uint8), cv2.IMREAD_COLOR)
    mask2 = _changed_mask(_frame_region(result2, h))
    check(_box_drawn(mask2, a), "empty-scene fallback still draws the single box")

    # No box at all + empty scene: banner-only, still decodes.
    out3 = annotate_event_snapshot(jpeg, None, "person", 0.0, 1, None, None)
    check(out3 is not None and cv2.imdecode(np.frombuffer(out3, np.uint8),
          cv2.IMREAD_COLOR).shape[0] > h,
          "no-box annotation still returns a banner-topped frame")


# ---------------- 2. engine: scene in the synthesized payload ----------------


class CapturePipeline:
    """Stand-in EventsPipeline: captures the engine's synthesized payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_event(self, payload: dict) -> None:
        self.payloads.append(payload)

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = count


async def _engine_scene_cases() -> None:
    engine = DetectionEngine(None, None, None, None, Config())  # type: ignore[arg-type]
    cap = CapturePipeline()
    engine.set_pipeline(cap)  # type: ignore[arg-type]
    engine._cameras["cam"] = _CameraState(
        row={"name": "cam", "detect_objects": ["person", "dog"], "record_enabled": True}
    )

    frame = np.full((480, 640, 3), BG, dtype=np.uint8)
    obs = [
        Observation("person", 0, 0.90, (50.0, 50.0, 150.0, 300.0)),
        Observation("person", 1, 0.80, (250.0, 60.0, 360.0, 320.0)),
        Observation("dog", 2, 0.95, (450.0, 200.0, 600.0, 400.0)),
    ]
    t0 = time.time()
    for i in range(4):  # >= MIN_HITS frames so all three tracks confirm
        await engine.process("cam", t0 + i * 0.2, obs, frame_bgr=frame)

    new_person = [
        p for p in cap.payloads if p["type"] == "new" and p["after"]["label"] == "person"
    ]
    check(len(new_person) == 1, "engine emitted exactly one person 'new' payload")
    after = new_person[0]["after"]
    check("scene" in after and isinstance(after["scene"], list),
          "payload carries after['scene'] as a list")
    scene = after["scene"]
    check(len(scene) == 3, "scene holds ALL counted objects (2 person + 1 dog), all labels")
    labels: dict[str, int] = {}
    for obj in scene:
        check(isinstance(obj["box"], list) and len(obj["box"]) == 4,
              "each scene object has a 4-value box")
        check(isinstance(obj["score"], float), "each scene object has a float score")
        labels[obj["label"]] = labels.get(obj["label"], 0) + 1
    check(labels == {"person": 2, "dog": 1}, "scene label multiset is 2 person + 1 dog")
    boxes = {tuple(o["box"]) for o in scene}
    check((450.0, 200.0, 600.0, 400.0) in boxes, "the dog box (other label) rides along in the scene")

    # Backward-compat single-object fields still present alongside the scene.
    check(after["snapshot"]["box"] == list(after["box"]) and len(after["box"]) == 4,
          "legacy snapshot.box / box fields retained for backward compatibility")

    # The dog event's scene is the same full scene (every counted object).
    new_dog = [p for p in cap.payloads if p["type"] == "new" and p["after"]["label"] == "dog"]
    check(len(new_dog) == 1 and len(new_dog[0]["after"]["scene"]) == 3,
          "the dog event's snapshot scene also boxes every counted object")

    # Scene tracks the retained best frame.
    st = engine._events[("cam", "person")]
    check(len(st.best_scene) == 3 and st.best_frame is not None,
          "EventState.best_scene matches the retained best frame")


def engine_scene_checks() -> None:
    print("engine 6: synthesized payload carries the full scene")
    asyncio.run(_engine_scene_cases())


# ---------------- 3. pipeline: real enrich -> multi-box snapshot ----------------


class FakePush:
    public_key = "fake"

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def send_to_all(self, payload: dict) -> PushSendResult:
        self.payloads.append(payload)
        return PushSendResult(attempted=1, sent=1)


class FakeWSClient:
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


class StubRecorder:
    def __init__(self, clips_dir: Path) -> None:
        self._clips_dir = clips_dir

    def clip_path(self, event_id: int) -> Path:
        return self._clips_dir / f"{event_id}.mp4"

    async def schedule_clip(self, *a, **k) -> None:  # not exercised here
        return None


async def _pipeline_multibox_cases() -> None:
    root = TMP / "pipeline"
    root.mkdir(parents=True, exist_ok=True)
    db = Database(root / "nvr.db")
    await db.connect()
    await db.upsert_camera({
        "name": "cam", "friendly_name": "Cam", "model": "IP5M-T1277EW-AI",
        "ip": "127.0.0.1", "username": "u", "password": "p",
        "detect_objects": ["person", "dog"], "detect_width": 640, "detect_height": 480,
        "detect_fps": 5, "detect_enabled": True, "record_enabled": True,
        "capabilities": {}, "created_at": time.time(),
    })
    settings = SettingsStore(db)
    await settings.load()
    ws = WSManager()
    await ws.connect(FakeWSClient())
    push = FakePush()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    recorder = StubRecorder(root / "clips")
    snapshots_dir = root / "snapshots"
    config = Config()

    # Real engine (retains the frame + scene) feeding a real EventsPipeline.
    engine = DetectionEngine(db, None, recorder, settings, config)  # type: ignore[arg-type]
    media = NativeMediaProvider(db, engine, recorder)  # type: ignore[arg-type]
    pipeline = EventsPipeline(db, media, ws, push, settings, auth, snapshots_dir)
    engine.set_pipeline(pipeline)
    engine._cameras["cam"] = _CameraState(
        row={"name": "cam", "detect_objects": ["person", "dog"],
             "record_enabled": True, "detect_enabled": True}
    )

    frame = np.full((480, 640, 3), BG, dtype=np.uint8)
    obs = [
        Observation("person", 0, 0.90, (50.0, 50.0, 150.0, 300.0)),
        Observation("person", 1, 0.80, (250.0, 60.0, 360.0, 320.0)),
        Observation("dog", 2, 0.95, (450.0, 200.0, 600.0, 400.0)),
    ]
    t0 = time.time()
    for i in range(4):
        await engine.process("cam", t0 + i * 0.2, obs, frame_bgr=frame)

    st = engine._events[("cam", "person")]
    row = await db.get_event_by_frigate_id(st.fid)
    check(row is not None and row["label"] == "person", "person event row stored via the pipeline")
    event_id = int(row["id"])

    # Enrichment runs as a spawned task — wait for the annotated snapshot.
    snap_path = snapshots_dir / f"{event_id}.jpg"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if snap_path.is_file() and (await db.get_event(event_id))["has_snapshot"]:
            break
        await asyncio.sleep(0.05)
    check(snap_path.is_file() and snap_path.stat().st_size > 1000,
          "real enrich wrote the annotated snapshot")

    result = cv2.imread(str(snap_path))
    check(result is not None and result.shape[1] == 640 and result.shape[0] > 480,
          "snapshot keeps detect width + banner strip")
    mask = _changed_mask(_frame_region(result, 480))
    check(_box_drawn(mask, [50, 50, 150, 300]), "enriched snapshot boxes person A")
    check(_box_drawn(mask, [250, 60, 360, 320]), "enriched snapshot boxes person B")
    check(_box_drawn(mask, [450, 200, 600, 400]), "enriched snapshot boxes the dog too")
    check(_region_empty(mask, 605, 0, 640, 45),
          "enriched snapshot: empty corner stays clean")

    await pipeline.shutdown()
    await db.close()


def pipeline_multibox_checks() -> None:
    print("pipeline 7: real EventsPipeline enrich -> multi-box snapshot")
    asyncio.run(_pipeline_multibox_cases())


# ---------------- _scene_of guard (pipeline helper) ----------------


def scene_of_checks() -> None:
    print("pipeline 8: _scene_of parses native scene / rejects legacy rows")
    good = {"scene": [{"box": [1, 2, 3, 4], "label": "person", "score": 0.9}]}
    parsed = _scene_of(good)
    check(parsed is not None and parsed[0]["box"] == [1.0, 2.0, 3.0, 4.0]
          and parsed[0]["label"] == "person" and isinstance(parsed[0]["score"], float),
          "well-formed scene parses to float boxes/scores")
    check(_scene_of({}) is None, "missing scene -> None (doorbell/audio/legacy fallback)")
    check(_scene_of({"scene": []}) is None, "empty scene -> None (single-box fallback)")
    check(_scene_of({"scene": [{"label": "x"}]}) is None,
          "scene entries without a valid box are dropped -> None")


def main() -> None:
    annotate_scene_checks()
    annotate_rescale_checks()
    annotate_clamp_checks()
    annotate_banner_checks()
    annotate_single_box_checks()
    engine_scene_checks()
    pipeline_multibox_checks()
    scene_of_checks()
    print(f"\nALL {PASS} CHECKS PASSED (multi-box annotation: annotate + engine + pipeline)")


if __name__ == "__main__":
    main()

"""Coral Edge TPU backend smoke — runs with NO Coral hardware and no litert.

Everything risky about this backend is in the pure functions: the SSD box order,
the normalised->pixel scaling, and the sparse COCO-90 -> COCO-80 remap. All three
fail SILENTLY in production (the engine resolves labels through a .get() that
never raises), so they are pinned here rather than discovered on a live camera.

The interpreter itself is faked, so this suite also proves the class satisfies
the detector contract without an Edge TPU present.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.native.coral import (  # noqa: E402
    CORAL_MODELS,
    CoralDetector,
    decode,
    preprocess,
)

MODEL_INPUT = CORAL_MODELS["ssdlite_mobiledet"]["input"]
from app.native.coco_labels import COCO_LABELS, coco90_to_label  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"  FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


def label_remap() -> None:
    print("\nCOCO-90 -> COCO-80 remap")
    # Anchors from google-coral/test_data/coco_labels.txt (90 lines, 0-indexed,
    # "n/a" at the ten gap rows). An off-by-one in the gap set still yields 80
    # entries, so these pin ALIGNMENT, not just count.
    for sparse, want in [
        (0, "person"), (2, "car"), (3, "motorcycle"), (5, "bus"), (7, "truck"),
        (9, "traffic_light"), (12, "stop_sign"), (15, "bird"), (16, "cat"),
        (17, "dog"), (18, "horse"),
    ]:
        check(coco90_to_label(sparse) == want, f"COCO-90 {sparse} -> {want}")
    for gap in (11, 25, 28, 29, 44, 65, 67, 68, 70, 82):
        check(coco90_to_label(gap) is None, f"COCO-90 gap {gap} -> None (dropped, never guessed)")
    check(coco90_to_label(90) is None, "out-of-range id -> None")
    check(coco90_to_label(-1) is None, "negative id -> None")


def preprocessing() -> None:
    print("\npreprocess")
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure BLUE in BGR
    t = preprocess(frame, MODEL_INPUT)
    check(t.shape == (1, MODEL_INPUT, MODEL_INPUT, 3), f"shape is 1x{MODEL_INPUT}x{MODEL_INPUT}x3")
    check(t.dtype == np.uint8, "dtype is uint8 (quantised model input, NOT float)")
    # BGR->RGB must actually happen: blue in BGR is channel 0; after conversion
    # it must land in channel 2. Getting this backwards silently degrades every
    # detection rather than erroring.
    check(int(t[0, 0, 0, 2]) == 255 and int(t[0, 0, 0, 0]) == 0,
          "BGR->RGB conversion applied (blue moved to channel 2)")


def decoding() -> None:
    print("\ndecode: box order, scaling, thresholding")
    W, H = 704, 480
    # SSD emits [ymin, xmin, ymax, xmax] NORMALISED. This box is the left half,
    # top quarter -> x 0..352, y 0..120 once scaled.
    boxes = np.array([[[0.0, 0.0, 0.25, 0.5]]], dtype=np.float32)
    ids = np.array([[0.0]], dtype=np.float32)          # COCO-90 0 = person
    scores = np.array([[0.9]], dtype=np.float32)
    det = decode(boxes, ids, scores, 1, 0.5, W, H)
    check(len(det) == 1, "one detection above threshold survives")
    x1, y1, x2, y2 = det.xyxy[0]
    check(abs(x1 - 0.0) < 0.01 and abs(y1 - 0.0) < 0.01, "xyxy origin correct")
    check(abs(x2 - 352.0) < 0.01, f"xmax scaled by WIDTH not height (got {x2}, want 352)")
    check(abs(y2 - 120.0) < 0.01, f"ymax scaled by HEIGHT not width (got {y2}, want 120)")
    check(COCO_LABELS[int(det.class_id[0])] == "person", "class id remapped to person")

    # Below-threshold detections must be dropped by the detector, not downstream.
    quiet = decode(boxes, ids, np.array([[0.2]], dtype=np.float32), 1, 0.5, W, H)
    check(len(quiet) == 0, "below-confidence detection is dropped here")

    # An unassigned COCO-90 id must vanish, NOT become class 0 (= person).
    gap = decode(boxes, np.array([[11.0]], dtype=np.float32),
                 np.array([[0.99]], dtype=np.float32), 1, 0.5, W, H)
    check(len(gap) == 0, "a gap class id is DROPPED, never coerced to person")

    # count governs, not the array length — SSD pads its output tensors.
    padded_boxes = np.array([[[0.0, 0.0, 0.25, 0.5], [0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    padded_ids = np.array([[0.0, 0.0]], dtype=np.float32)
    padded_scores = np.array([[0.9, 0.9]], dtype=np.float32)
    one = decode(padded_boxes, padded_ids, padded_scores, 1, 0.5, W, H)
    check(len(one) == 1, "only `count` entries are read (padding ignored)")
    check(len(decode(padded_boxes, padded_ids, padded_scores, 0, 0.5, W, H)) == 0,
          "count=0 yields no detections")


class _FakeInterpreter:
    """Stands in for LiteRT. Returns one high-confidence person, centre frame."""

    def __init__(self) -> None:
        self.invoked = 0

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": (1, MODEL_INPUT, MODEL_INPUT, 3), "dtype": np.uint8}]

    def get_output_details(self):
        return [{"index": i} for i in (1, 2, 3, 4)]

    def set_tensor(self, index, tensor) -> None:
        self.tensor = tensor

    def invoke(self) -> None:
        self.invoked += 1

    def get_tensor(self, index):
        return {
            1: np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32),
            2: np.array([[0.0]], dtype=np.float32),
            3: np.array([[0.88]], dtype=np.float32),
            4: np.array([1.0], dtype=np.float32),
        }[index]


def detector_contract() -> None:
    print("\nCoralDetector contract (no hardware, no litert)")
    det = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)

    # Before bootstrap it must be visibly NOT ready and refuse to infer, so the
    # ingest gate skips it rather than the app crashing.
    check(det.ready is False, "starts not-ready")
    raised = False
    try:
        det.detect(np.zeros((480, 704, 3), dtype=np.uint8), 704, 480)
    except RuntimeError:
        raised = True
    check(raised, "detect() on a not-ready detector raises RuntimeError")

    # Inject the fake interpreter the way a successful bootstrap would.
    fake = _FakeInterpreter()
    _inject(det, [fake])

    out = det.detect(np.zeros((480, 704, 3), dtype=np.uint8), 704, 480)
    check(fake.invoked == 1, "detect() invokes the interpreter exactly once")
    check(len(out) == 1, "detect() returns the decoded detection")
    check(COCO_LABELS[int(out.class_id[0])] == "person", "detect() yields a person")
    x1, _, x2, _ = out.xyxy[0]
    check(abs(x1 - 176.0) < 0.01 and abs(x2 - 528.0) < 0.01,
          "detect() boxes are in DETECT-STREAM pixels (0.25/0.75 of 704)")
    check(det.last_inference_ms is not None, "last_inference_ms is recorded")

    # Every member ingest + the self-heal supervisor drive must exist.
    for name in ("ready", "detect", "note_detect_ok", "note_detect_failure",
                 "needs_reinit", "reinit", "kind", "device", "model_key",
                 "confidence", "model_sha_ok", "last_inference_ms", "status",
                 "start", "stop", "reconfigure", "last_reinit_age_s"):
        check(hasattr(det, name), f"implements `{name}`")
    check(det.kind == "coral", "kind is 'coral'")

    # Self-heal hysteresis must match OnnxDetector's, or the shared supervisor
    # behaves differently depending on which backend is loaded.
    check(det.needs_reinit is False, "no reinit flag initially")
    det.note_detect_failure()
    det.note_detect_failure()
    check(det.needs_reinit is False, "2 failures do NOT trigger reinit")
    det.note_detect_failure()
    check(det.needs_reinit is True, "3 consecutive failures DO flag reinit")
    det.note_detect_ok()
    check(det.consecutive_failures == 0, "a success resets the failure count")

    st = det.status()
    for key in ("kind", "ready", "device", "model", "last_inference_ms",
                "consecutive_failures", "needs_reinit"):
        check(key in st, f"status() exposes `{key}`")


def factory_wiring() -> None:
    print("\nfactory + config gate")
    from app.config import VALID_DETECTORS
    from app.native.detector import build_detector

    check("coral" in VALID_DETECTORS,
          "VALID_DETECTORS accepts 'coral' (without this _env_detector silently "
          "falls back to onnx and the branch is unreachable)")

    class Cfg:
        detector = "coral"
        require_gpu = True

    det = build_detector(
        config=Cfg(), models_dir=Path("/nonexistent"), model_key="dfine_s", confidence=0.4
    )
    check(det.kind == "coral", "build_detector returns a CoralDetector for detector=coral")
    check(det.confidence == 0.4, "confidence is threaded through")
    check(det.ready is False, "returned not-ready until start() runs (no hardware here)")


def backend_setting() -> None:
    """settings.detection.backend is the user-facing switch; VIGILUME_DETECTOR
    stays an explicit override. Both must resolve to the right detector, and the
    precedence must not silently invert — an operator forcing a backend from the
    environment is usually doing it BECAUSE the stored choice broke detection."""
    print("\nbackend setting + env precedence")
    import os
    from app.config import BACKEND_TO_DETECTOR, DEFAULT_SETTINGS, VALID_BACKENDS
    from app.native.detector import build_detector

    check(DEFAULT_SETTINGS["detection"]["backend"] == "auto",
          "default backend is 'auto' — a fitted Coral is used without anyone "
          "having to find a setting, and a box without one still detects")
    check(set(VALID_BACKENDS) == {"auto", "gpu", "coral"},
          "the user-facing backend set is auto|gpu|coral (onnx_cpu stays a debug knob)")
    check(BACKEND_TO_DETECTOR["auto"] == "auto", "auto maps to the auto detector kind")

    class Cfg:
        detector = "onnx"
        require_gpu = True

    saved = os.environ.pop("VIGILUME_DETECTOR", None)
    try:
        for backend, want in (("gpu", "onnx"), ("coral", "coral")):
            d = build_detector(
                config=Cfg(), models_dir=Path("/nonexistent"), model_key="dfine_s",
                confidence=0.5, backend=BACKEND_TO_DETECTOR[backend],
            )
            check(d.kind == want, f"backend={backend} -> detector kind={want}")

        os.environ["VIGILUME_DETECTOR"] = "onnx"
        d = build_detector(
            config=Cfg(), models_dir=Path("/nonexistent"), model_key="dfine_s",
            confidence=0.5, backend="coral",
        )
        check(d.kind == "onnx",
              "an explicitly-set VIGILUME_DETECTOR OVERRIDES the stored backend")

        # REGRESSION GUARD. docker-compose.yml injects VIGILUME_DETECTOR on
        # every start. It used to default to "onnx", so the variable was NEVER
        # unset inside the container, the override branch fired every time, and
        # the stored setting could not take effect AT ALL — the picker silently
        # did nothing. The compose default is now empty; an EMPTY value must be
        # treated as absent.
        os.environ["VIGILUME_DETECTOR"] = ""
        d = build_detector(
            config=Cfg(), models_dir=Path("/nonexistent"), model_key="dfine_s",
            confidence=0.5, backend="coral",
        )
        check(d.kind == "coral",
              "an EMPTY VIGILUME_DETECTOR (what compose injects) does NOT "
              "override — the stored backend still wins")
    finally:
        os.environ.pop("VIGILUME_DETECTOR", None)
        if saved is not None:
            os.environ["VIGILUME_DETECTOR"] = saved


def settings_round_trip() -> None:
    """detection.backend must SURVIVE the settings PUT. The router's Pydantic
    models silently DROP unmodelled keys, so a backend field that is not declared
    there would vanish on every save and the picker would appear to do nothing."""
    print("\nsettings persistence")
    from app.routers.settings import DetectionSettings

    d = DetectionSettings(model="dfine_s", confidence=0.5, backend="coral")
    check(d.model_dump()["backend"] == "coral",
          "backend survives the settings model (not silently dropped on PUT)")
    check(DetectionSettings(model="dfine_s", confidence=0.5).backend == "auto",
          "omitted backend defaults to auto")
    bad = False
    try:
        DetectionSettings(model="dfine_s", confidence=0.5, backend="tpu")
    except Exception:  # noqa: BLE001 — pydantic ValidationError
        bad = True
    check(bad, "an unknown backend is REJECTED, not coerced")


def model_registry() -> None:
    """Every registry field was read off the downloaded artifact. These pin the
    facts that silently corrupt detections if wrong: the per-model INPUT SIZE
    (300/320/384/448/512 — never inferrable from the filename) and the fact that
    the settings Literal and the registry cannot drift apart."""
    print("\nEdge TPU model registry")
    from app.native.coral import CORAL_DEFAULT_MODEL, CORAL_MODELS, coral_model
    from app.routers.settings import DetectionSettings

    check(len(CORAL_MODELS) == 6, "six Edge TPU models offered")
    lit = set(DetectionSettings.model_fields["coral_model"].annotation.__args__)
    check(lit == set(CORAL_MODELS),
          "settings Literal matches the registry exactly (no unselectable or "
          "unstorable model)")
    check(CORAL_DEFAULT_MODEL in CORAL_MODELS, "the default key exists")

    # Verified against the real artifacts, not the filenames.
    for key, want in (
        ("ssd_mobilenet_v2", 300), ("ssdlite_mobiledet", 320),
        ("efficientdet_lite0", 320), ("efficientdet_lite1", 384),
        ("efficientdet_lite2", 448), ("efficientdet_lite3", 512),
    ):
        check(CORAL_MODELS[key]["input"] == want, f"{key} input is {want}")
    for key, spec in CORAL_MODELS.items():
        check(len(spec["sha256"]) == 64, f"{key} has a full sha256")
        check(spec["file"].endswith("_edgetpu.tflite"), f"{key} is an EDGETPU build")
    check(coral_model("nonsense-key")["file"] == CORAL_MODELS[CORAL_DEFAULT_MODEL]["file"],
          "an unknown model key falls back to the default rather than raising")


def box_clamping() -> None:
    """SSD 'normalised' boxes genuinely exceed 1.0 (measured 1.0032 on a real
    frame), so decode must clamp — downstream tracking, exempt-zone foot-point
    tests and annotation all assume in-frame coordinates."""
    print("\nbox clamping")
    W, H = 704, 480
    over = np.array([[[-0.01, -0.02, 1.0032, 1.05]]], dtype=np.float32)
    det = decode(over, np.array([[0.0]], dtype=np.float32),
                 np.array([[0.9]], dtype=np.float32), 1, 0.5, W, H)
    x1, y1, x2, y2 = det.xyxy[0]
    check(x1 >= 0 and y1 >= 0, f"negative coords clamped to 0 (got {x1}, {y1})")
    check(x2 <= W and y2 <= H, f"over-1.0 coords clamped to frame (got {x2}, {y2})")


def _inject(det, fakes) -> None:
    """Bind fake interpreters the way a successful bootstrap would.

    The detector holds a POOL of devices now, not one interpreter, so a test
    fake has to enter through the same door real hardware does: a `_Tpu` per
    device, all of them in the checkout queue.
    """
    import queue as _q
    from app.native.coral import _Tpu

    det._tpus = [
        _Tpu(spec=f"fake:{i}", interpreter=f, input_index=0, output_indices=[1, 2, 3, 4],
             priority=i)
        for i, f in enumerate(fakes)
    ]
    # Priority pool: entries are (priority, seq, tpu). Distinct priorities here
    # so the tests also exercise the PCIe-before-USB ordering.
    det._pool = _q.PriorityQueue()
    det._pool_seq = 0
    for t in det._tpus:
        det._release(t)
    det._ready = True
    det._device = "edgetpu"


def device_pool() -> None:
    """The multi-TPU pool: auto-detect, exclusive checkout, and the two ways it
    could silently stop detection forever."""
    print("\nEdge TPU pool (up to 2 devices)")
    import queue as _q
    import threading
    import time
    from app.native.coral import MAX_TPUS, _DEVICE_SPECS

    check(MAX_TPUS == 2, "binds at most 2 Edge TPUs")
    check(_DEVICE_SPECS[0].startswith("pci"),
          "PCIe is probed FIRST — a one-TPU box binds the fast device, not the USB stick")

    # --- auto-detect: bind exactly the devices that answer, capped at 2 ---
    det = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    present = {"pci:0", "usb:0"}          # a PCIe card + a USB stick
    attempted: list = []

    def fake_bind(_path, spec):
        attempted.append(spec)
        if spec is not None and spec not in present:
            raise OSError(f"no device at {spec}")
        from app.native.coral import _Tpu
        return _Tpu(spec=spec or "auto", interpreter=_FakeInterpreter(),
                    input_index=0, output_indices=[1, 2, 3, 4])

    det._bind_one = fake_bind  # type: ignore[method-assign]
    bound = det._bind_devices(Path("/nonexistent"))
    check([t.spec for t in bound] == ["pci:0", "usb:0"],
          "auto-detect binds BOTH present devices and skips the absent ones")
    check("pci:1" in attempted, "it probes past a gap rather than stopping at the first miss")
    check(len(bound) <= MAX_TPUS, "never binds more than the cap")

    # --- one device present: still works, no fallback needed ---
    det2 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    present = {"pci:0"}
    det2._bind_one = fake_bind  # type: ignore[method-assign]
    check([t.spec for t in det2._bind_devices(Path("/nonexistent"))] == ["pci:0"],
          "a single-TPU box binds exactly one device")

    # --- NO addressed device binds: fall back to the unaddressed delegate ---
    # This is what protects a working single-TPU box from an older libedgetpu
    # that rejects the options dict entirely.
    det3 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    present = set()
    det3._bind_one = fake_bind  # type: ignore[method-assign]
    fell_back = det3._bind_devices(Path("/nonexistent"))
    check([t.spec for t in fell_back] == ["auto"],
          "no addressed device -> falls back to the UNADDRESSED delegate (old behaviour)")
    check(attempted[-1] is None, "the fallback passes device_spec=None, not a spec string")

    # --- exclusive checkout: two threads never share an interpreter ---
    det4 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    a, b = _FakeInterpreter(), _FakeInterpreter()
    _inject(det4, [a, b])
    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    overlap = {"max": 0, "cur": 0}
    guard = threading.Lock()
    real_invoke = det4._invoke

    def counting_invoke(tpu, tensor):
        with guard:
            overlap["cur"] += 1
            overlap["max"] = max(overlap["max"], overlap["cur"])
        try:
            time.sleep(0.05)             # wide enough to observe reliably; 0.02 was not
            return real_invoke(tpu, tensor)
        finally:
            with guard:
                overlap["cur"] -= 1

    det4._invoke = counting_invoke  # type: ignore[method-assign]
    # BARRIER, not thread-start ordering. Without it the threads trickle in and
    # each can finish before the next arrives, so the observed overlap measures
    # the scheduler rather than the pool — the first version of this check read
    # max=1 against a pool that was working correctly.
    gate = threading.Barrier(4)

    def run() -> None:
        gate.wait()
        det4.detect(frame, 704, 480)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(overlap["max"] <= 2,
          f"at most 2 inferences run at once with 2 devices (saw {overlap['max']})")
    check(overlap["max"] == 2,
          "...and it DOES reach 2 — the pool genuinely parallelises rather than serialising")
    check(a.invoked + b.invoked == 4, "every frame ran exactly once across the pool")
    check(a.invoked > 0 and b.invoked > 0, "BOTH devices did work (not one starved)")
    check(det4._pool.qsize() == 2, "every device is returned to the pool")

    # --- a raising inference must NOT leak the device ---
    det5 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    _inject(det5, [_FakeInterpreter()])

    def boom(_tpu, _tensor):
        raise RuntimeError("simulated TPU fault")

    det5._invoke = boom  # type: ignore[method-assign]
    for _ in range(3):
        try:
            det5.detect(frame, 704, 480)
        except RuntimeError:
            pass
    check(det5._pool.qsize() == 1,
          "a failing invoke RETURNS its device — leaking would deadlock detection forever")

    # --- release drops the devices so a re-bind can claim them ---
    det6 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    _inject(det6, [_FakeInterpreter(), _FakeInterpreter()])
    det6._release_devices()
    check(det6._tpus == [] and det6._pool.qsize() == 0,
          "release frees every device (a delegate holds its TPU exclusively)")

    # --- PCIe IS PREFERRED, and USB is real overflow capacity ---
    det8 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    x, y = _FakeInterpreter(), _FakeInterpreter()
    _inject(det8, [x, y])          # fake:0 priority 0 (PCIe), fake:1 priority 1 (USB)
    for _ in range(40):
        det8.detect(frame, 704, 480)
    # 39/1, not 40/0: a freshly bound device has last_ok_mono == 0.0 and is
    # therefore overdue, so the FIRST frame is spent proving the USB device and
    # every frame after it goes to the preferred one. That single frame is the
    # whole liveness mechanism — see _checkout.
    check(y.invoked == 1,
          f"the idle probe spends exactly ONE frame proving the non-preferred "
          f"device (got {y.invoked}) — binding allocates tensors but never "
          "invokes, so without this a dead TPU looks identical to an idle one")
    check(x.invoked == 39,
          f"every OTHER frame goes to the preferred device ({x.invoked}/40) — "
          "PCIe is used over USB rather than alternating")
    stats = det8.status()["device_stats"]
    by_dev = {d["device"]: d for d in stats}
    check(by_dev["fake:0"]["share_pct"] == 97.5 and by_dev["fake:1"]["share_pct"] == 2.5,
          "device_stats shows the preference honestly (97.5/2.5), not a fictional average")
    check(all(d["last_ok_age_s"] is not None for d in stats),
          "BOTH devices report a proven-alive age — this is the answer to "
          "'are both Corals working', which share_pct alone cannot give")
    check(all(d["errors"] == 0 for d in stats), "a healthy device reports no errors")
    check(all("avg_ms" in d for d in stats), "each device reports its own average latency")
    check({d["device"] for d in stats} == {"fake:0", "fake:1"},
          "stats are keyed by device spec, so a lopsided TPU is identifiable")

    # --- the probe RE-fires once a device goes stale again ---
    det8b = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    xb, yb = _FakeInterpreter(), _FakeInterpreter()
    _inject(det8b, [xb, yb])
    for _ in range(10):
        det8b.detect(frame, 704, 480)
    check(yb.invoked == 1, "one probe so far")
    for tpu in det8b._tpus:          # pretend _IDLE_PROBE_S has elapsed
        tpu.last_ok_mono -= 3600.0
    det8b.detect(frame, 704, 480)
    check(yb.invoked == 2,
          "a device that goes quiet again is re-proven — liveness is continuous, "
          "not a one-off at startup")

    # --- a device failing every invoke is COUNTED, not silently idle ---
    det8c = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    _inject(det8c, [_FakeInterpreter(), _FakeInterpreter()])
    real8c = det8c._invoke

    def fail_second(tpu, tensor):
        if tpu.spec == "fake:1":
            raise RuntimeError("simulated dead USB TPU")
        return real8c(tpu, tensor)

    det8c._invoke = fail_second  # type: ignore[method-assign]
    for _ in range(6):
        try:
            det8c.detect(frame, 704, 480)
        except RuntimeError:
            pass
    dead = {d["device"]: d for d in det8c.status()["device_stats"]}["fake:1"]
    check(dead["errors"] >= 1 and dead["inferences"] == 0,
          f"a TPU that fails every invoke reports errors={dead['errors']} with zero "
          "inferences — distinguishable from one that is merely idle")
    check(dead["last_ok_age_s"] is None,
          "and it never reports a proven-alive time, because it was never proven")

    # --- a device released after the fleet was rebuilt is DROPPED ---
    det8d = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    _inject(det8d, [_FakeInterpreter(), _FakeInterpreter()])
    orphan = det8d._tpus[0]
    det8d._release_devices()          # simulates a reinit under a live detect()
    det8d._release(orphan)            # the in-flight thread's finally
    check(det8d._pool.qsize() == 0,
          "a device from a previous fleet is NOT resurrected into the new pool — "
          "it would outrank everything, serve every frame, and be invisible in stats")
    check(orphan.interpreter is None,
          "and its interpreter is dropped so libedgetpu can actually free the device — "
          "otherwise the rebind finds it still claimed and comes back one TPU short")

    # ...but the slower device MUST still absorb overflow, or the second TPU is
    # decorative. Saturate the preferred one and the other has to pick up work.
    det9 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    p0, p1 = _FakeInterpreter(), _FakeInterpreter()
    _inject(det9, [p0, p1])
    real9 = det9._invoke

    def slow_invoke(tpu, tensor):
        time.sleep(0.05)
        return real9(tpu, tensor)

    det9._invoke = slow_invoke  # type: ignore[method-assign]
    gate9 = threading.Barrier(6)

    def run9() -> None:
        gate9.wait()
        det9.detect(frame, 704, 480)

    threads9 = [threading.Thread(target=run9) for _ in range(6)]
    for t in threads9:
        t.start()
    for t in threads9:
        t.join()
    check(p1.invoked > 0,
          f"under concurrent load the SECOND device takes overflow "
          f"({p0.invoked}/{p1.invoked}) — it is capacity, not decoration")
    check(p0.invoked + p1.invoked == 6, "every frame ran exactly once across the pool")
    check(p0.invoked >= p1.invoked,
          "and the preferred device still carries at least its share")

    # --- status reports the real fleet ---
    det7 = CoralDetector(models_dir=Path("/nonexistent"), confidence=0.5)
    _inject(det7, [_FakeInterpreter(), _FakeInterpreter()])
    st = det7.status()
    check(st["device"] == "edgetpu",
          "`device` stays the single string existing clients decode")
    check(st["device_count"] == 2 and len(st["devices"]) == 2,
          "`device_count`/`devices` expose the pool without breaking those clients")



def main() -> None:
    label_remap()
    model_registry()
    box_clamping()
    preprocessing()
    decoding()
    detector_contract()
    device_pool()
    factory_wiring()
    backend_setting()
    settings_round_trip()
    print(f"\nALL {PASS} CHECKS PASSED (coral backend)")


if __name__ == "__main__":
    main()

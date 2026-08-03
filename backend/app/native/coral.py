"""Coral Edge TPU detector backend (SSD/MobileDet via LiteRT + libedgetpu).

WHY THIS EXISTS, AND THE RULE THAT GOVERNS IT
---------------------------------------------
An earlier Coral integration was ripped out because it had become a first-class
CONCEPT: it reached into config validation, the detector factory internals, the
ingest worker topology, docs tables and the web/iOS status decoders, so removing
it was a tree-wide excision.

This one is a DEPLOYMENT DETAIL. The rule:

    the rest of the backend must keep seeing exactly ONE detector object and
    ONE ingest worker.

Concretely, ``native/ingest.py`` is NOT TOUCHED. It calls six duck-typed members
(``ready``, ``detect``, ``note_detect_ok``, ``note_detect_failure``,
``needs_reinit``, ``reinit``) and already wraps ``detect`` in
``asyncio.to_thread`` under an 8s timeout. This class satisfies that contract,
so switching backends is one env var and deleting it again is one branch.

THE NUMPY QUESTION
------------------
The original removal was forced by a NumPy 1-vs-2 ABI wall: the backend is
locked to NumPy 2 (``trackers``/``scipy``), while the then-current Coral runtime
(``tflite_runtime``) was built against NumPy 1. ``ai-edge-litert`` >= 2.x
declares ``numpy>=1.23.2`` with NO upper bound, which is what makes an
IN-PROCESS backend possible again.

Whether libedgetpu's ``edgetpu-custom-op`` actually EXECUTES under LiteRT (as
opposed to the delegate merely loading) is hardware-verified per box, not
assumed. If it does not, only :meth:`_invoke` has to move behind an
out-of-process NumPy-1 sidecar — every other line here is transport-agnostic on
purpose. Note there is no silent-CPU-fallback risk to design around: no CPU
kernel implements that op, so ``allocate_tensors`` either succeeds or raises.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import supervision as sv

from .coco_labels import coco90_to_label, COCO90_TO_COCO80

log = logging.getLogger("app.native.coral")

# ---------------------------------------------------------------------------
# Edge TPU model registry
# ---------------------------------------------------------------------------
#
# Every field here was VERIFIED by downloading the artifact and inspecting it
# with LiteRT — input size and dtype read off the model, sha256 computed from
# the bytes. Nothing is inferred from the filename (the "_320_" in a name is not
# a contract) and nothing is copied from documentation.
#
# OUTPUT ORDER, verified by running the CPU twins on a real photo: BOTH families
# emit [0]=boxes, [1]=class_id, [2]=scores, [3]=count, even though their tensor
# NAMES differ completely (SSD: "TFLite_Detection_PostProcess:1,2,3" ascending;
# EfficientDet-Lite0/1/2: "StatefulPartitionedCall:3,2,1,0" DESCENDING; Lite3:
# ":31,:32,:33,:34"). Do not "fix" decode() to follow the name suffixes.
#
# latency_ms is Coral's published Edge TPU figure, EXCEPT ssdlite_mobiledet,
# which we measured at 9.64 ms on this actual hardware (published 9.5).
# NOTE the slow end: lite2/lite3 at ~105 ms sustain under 10 inferences/sec,
# which is at or below a 2-camera-at-5fps load. The UI warns about this.
CORAL_MODELS: dict[str, dict[str, Any]] = {
    "ssd_mobilenet_v2": {
        "file": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "sha256": "b94e2d58222c32f31062c7604e10488e2aba9259ab77462039476a3ba4597fef",
        "input": 300, "map": 22.4, "latency_ms": 7.6,
        "label": "SSD MobileNet V2", "note": "fastest, lowest accuracy",
    },
    "ssdlite_mobiledet": {
        "file": "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
        "sha256": "b69e508ef2a670e06b80bd3e5559a827d5cd8d557c95d5e332cbf1d31d434a2e",
        "input": 320, "map": 32.9, "latency_ms": 9.6,
        "label": "SSDLite MobileDet", "note": "best speed/accuracy balance — recommended",
    },
    "efficientdet_lite0": {
        "file": "efficientdet_lite0_320_ptq_edgetpu.tflite",
        "sha256": "4b17288c7988973274c63e10c914748363e7795ba715c6d8d7db1a2b909744a5",
        "input": 320, "map": 25.7, "latency_ms": 37.4,
        "label": "EfficientDet-Lite0", "note": "",
    },
    "efficientdet_lite1": {
        "file": "efficientdet_lite1_384_ptq_edgetpu.tflite",
        "sha256": "635dd33cf5be42c74996fd582b5dbb73ca5f448e607d8203d66bde1685fdd493",
        "input": 384, "map": 30.6, "latency_ms": 56.3,
        "label": "EfficientDet-Lite1", "note": "",
    },
    "efficientdet_lite2": {
        "file": "efficientdet_lite2_448_ptq_edgetpu.tflite",
        "sha256": "420a28a60cf7f9dfd24774aad361ece142a8acc326063c77abf3fdff07cf60cf",
        "input": 448, "map": 34.0, "latency_ms": 104.6,
        "label": "EfficientDet-Lite2", "note": "slow — may not keep up",
    },
    "efficientdet_lite3": {
        "file": "efficientdet_lite3_512_ptq_edgetpu.tflite",
        "sha256": "4f98f09872404d9e28744d3ff694d8427a968ddb467a9aec0ac861bd9f3dba14",
        "input": 512, "map": 37.7, "latency_ms": 107.6,
        "label": "EfficientDet-Lite3", "note": "highest accuracy, slowest",
    },
}
CORAL_DEFAULT_MODEL = "ssdlite_mobiledet"
MODEL_BASE_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/"


def coral_model(key: str) -> dict[str, Any]:
    """Registry entry for ``key``, falling back to the default on an unknown
    value rather than raising — an unrecognised stored model must never stop a
    security system from detecting."""
    return CORAL_MODELS.get(key) or CORAL_MODELS[CORAL_DEFAULT_MODEL]


DELEGATE_SO = "libedgetpu.so.1"

# How many Edge TPUs to bind at once. Two, per the deployment: a PCIe card plus
# a USB stick. Not unbounded — each device holds an interpreter with its own
# copy of the model, and the point is to use the hardware present, not to make
# the pool a tuning knob.
MAX_TPUS = 2

# Probe order for auto-detection. libedgetpu addresses devices as `<type>:<index>`
# and a delegate holds its device EXCLUSIVELY, so probing is also claiming: the
# first successful load owns pci:0, the next moves on to pci:1, and so on.
#
# PCIe FIRST, deliberately. A USB Coral pays a bus round-trip per inference and
# measures noticeably slower than the PCIe card, so a single-TPU box binds the
# fast one. On a two-TPU box the pool is a PriorityQueue keyed PCIe-first, so
# the PCIe card serves EVERY request that finds it idle — USB is handed out
# only once every PCIe device is already checked out, which requires two
# concurrent detect() calls. See the note on _pool for why that is rare.
_DEVICE_SPECS: tuple[str, ...] = ("pci:0", "pci:1", "usb:0", "usb:1")


class _Tpu:
    """One bound Edge TPU: its delegate-backed interpreter and tensor indices.

    Exists because a LiteRT ``Interpreter`` is NOT safe to ``invoke()`` from two
    threads at once. Rather than guard one interpreter with a lock (which would
    serialise the whole pool and defeat the point), each device carries its own
    interpreter and is handed to exactly one caller at a time via the pool
    queue — the queue IS the mutual exclusion.
    """

    __slots__ = ("spec", "priority", "interpreter", "input_index",
                 "output_indices", "inferences", "total_ms", "errors",
                 "last_ok_mono")

    def __init__(self, spec: str, interpreter: Any, input_index: int,
                 output_indices: list[int], priority: int = 0) -> None:
        self.spec = spec
        # Lower wins when more than one device is idle. PCIe outranks USB
        # because a USB Coral pays a bus round-trip per inference.
        self.priority = priority
        self.interpreter = interpreter
        self.input_index = input_index
        self.output_indices = output_indices
        # Per-device tally, so the actual split between TPUs is observable
        # rather than assumed. Only ever mutated by the thread currently
        # holding this device (the pool guarantees exclusivity), so no lock is
        # needed; status() reads them without one and a torn read costs at most
        # one miscounted frame in a statistic.
        self.inferences = 0
        self.total_ms = 0.0
        # Liveness, NOT load. Binding a device only allocates tensors — it never
        # invokes — so a wedged or every-invoke-raising TPU binds exactly like a
        # healthy one. Without these, "idle by design" and "dead" are byte
        # identical in device_stats, which is the question that actually gets
        # asked about a two-TPU box. 0.0 means never proven: a freshly bound
        # device is overdue on purpose, so the idle probe exercises it at once.
        self.errors = 0
        self.last_ok_mono = 0.0

# Reinit hysteresis, mirroring OnnxDetector so the shared self-heal supervisor
# behaves identically regardless of which backend is loaded.
_FAILURES_BEFORE_REINIT = 3

# Bounded pool checkout, under ingest's DETECT_TIMEOUT_S (8.0) so a wedged
# device is reported here, as a counted failure, rather than as a generic
# upstream timeout.
_POOL_WAIT_S = 5.0
# How long a device may go without a completed invoke before one frame is
# deliberately spent proving it. See _checkout.
_IDLE_PROBE_S = 60.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(models_dir: Path, model_key: str = "") -> Path:
    """Return the Edge-TPU model path, downloading + SHA-verifying if absent.

    A checksum mismatch DELETES the file and raises rather than loading it: a
    truncated or tampered detector model is the one thing that must never be
    silently tolerated on a security system.
    """
    spec = coral_model(model_key)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / spec["file"]
    if path.exists():
        if _sha256(path) == spec["sha256"]:
            return path
        log.warning("coral: %s failed checksum — re-downloading", path.name)
        path.unlink()
    tmp = path.with_suffix(".part")
    url = MODEL_BASE_URL + spec["file"]
    log.info("coral: downloading %s", spec["file"])
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # noqa: S310
        fh.write(resp.read())
    got = _sha256(tmp)
    if got != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"coral model {spec['file']} checksum mismatch: got {got}, want {spec['sha256']}"
        )
    tmp.rename(path)
    return path


def preprocess(frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """BGR frame -> 1 x N x N x 3 uint8 RGB, PLAIN resize (no letterbox).

    ``input_size`` is per-model and read off the artifact (300 for
    ssd_mobilenet_v2, 320/384/448/512 across the rest) — NEVER inferred from the
    filename.

    Matches the ONNX path's geometry deliberately: the SSD head returns
    NORMALISED box coordinates, so a plain resize means decoding is a straight
    multiply by the detect-stream size with no padding to undo. Letterboxing
    here would silently offset every box.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return resized[np.newaxis, ...].astype(np.uint8)


def decode(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    count: int,
    confidence: float,
    detect_width: int,
    detect_height: int,
) -> sv.Detections:
    """SSD postprocess outputs -> ``sv.Detections`` in detect-stream pixels.

    TWO TRAPS, both silent if got wrong:

    1. The boxes tensor is ``[ymin, xmin, ymax, xmax]`` NORMALISED 0..1 — NOT
       xyxy, and not pixels. Feeding it straight through transposes every box.
    2. ``class_ids`` are the SPARSE COCO-90 space, while the engine's label
       table is contiguous COCO-80. ``engine`` resolves labels via
       ``ID_TO_LABEL.get()``, which never raises, so an unmapped id does not
       error — it mislabels. Ids with no COCO-80 counterpart are DROPPED here
       rather than coerced (0 is "person"; inventing that is the worst possible
       failure on a camera).
    """
    n = int(count)
    if n <= 0:
        return sv.Detections.empty()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[:n]
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)[:n]
    raw_ids = np.asarray(class_ids).reshape(-1)[:n].astype(int)

    keep_xyxy: list[list[float]] = []
    keep_conf: list[float] = []
    keep_cls: list[int] = []
    dw, dh = float(detect_width), float(detect_height)
    for i in range(n):
        score = float(scores[i])
        if score < confidence:
            continue
        dense = COCO90_TO_COCO80.get(int(raw_ids[i]))
        if dense is None:
            continue  # unassigned COCO-90 id — drop, never guess
        ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
        # CLAMP. These are "normalised" but the SSD head genuinely returns
        # values slightly outside 0..1 (measured 1.0032 on a real frame), so a
        # raw multiply puts boxes a few pixels off-frame. Downstream (tracker,
        # exempt-zone foot-point tests, annotation) all assume in-frame
        # coordinates.
        keep_xyxy.append([
            max(0.0, min(xmin * dw, dw)), max(0.0, min(ymin * dh, dh)),
            max(0.0, min(xmax * dw, dw)), max(0.0, min(ymax * dh, dh)),
        ])
        keep_conf.append(score)
        keep_cls.append(dense)

    if not keep_xyxy:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(keep_xyxy, dtype=np.float32),
        confidence=np.asarray(keep_conf, dtype=np.float32),
        class_id=np.asarray(keep_cls, dtype=int),
    )


class CoralDetector:
    """Edge TPU detector satisfying the same duck-typed contract as
    ``OnnxDetector`` (see that class; ingest/self-heal drive both identically)."""

    def __init__(
        self,
        models_dir: Path,
        confidence: float = 0.5,
        model_key: str = CORAL_DEFAULT_MODEL,
        delegate_so: str = DELEGATE_SO,
    ):
        self._models_dir = models_dir
        self._confidence = float(confidence)
        self._model_key = model_key if model_key in CORAL_MODELS else CORAL_DEFAULT_MODEL
        self._input_size = coral_model(self._model_key)["input"]
        self._delegate_so = delegate_so
        # Bound devices, and the checkout queue that hands each to one caller at
        # a time. The queue is the concurrency control — see _Tpu.
        self._tpus: list[_Tpu] = []
        # PRIORITY pool, not FIFO. When several devices are idle the caller gets
        # the FASTEST one (PCIe before USB) instead of whichever happened to
        # finish least recently. Entries are (priority, seq, tpu) — `seq` breaks
        # ties FIFO within a priority and keeps _Tpu itself out of comparisons.
        #
        # WHAT THIS MEANS IN PRACTICE, because it surprises people: USB is only
        # ever handed out while EVERY PCIe device is already checked out, which
        # takes two concurrent detect() calls. ingest drives detection from a
        # single worker that awaits each frame to completion, so concurrency is
        # 1 and the realised split is 100/0 — by design, not because the USB
        # stick is broken. The second TPU is a hot spare and burst headroom.
        # Do NOT "fix" that by equalising the priorities: with one frame in
        # flight the invoke sits on the serial worker's critical path, so
        # sending frames to the slower device raises mean invoke time and
        # lowers the worker's frame ceiling for exactly zero extra inferences.
        self._pool: "queue.PriorityQueue[tuple[int, int, _Tpu]]" = queue.PriorityQueue()
        self._pool_seq = 0
        self._max_devices_seen = 0
        self._ready = False
        self._device: Optional[str] = None
        self._last_inference_ms: Optional[float] = None
        self._consecutive_failures = 0
        self._needs_reinit = False
        self._last_reinit_monotonic: Optional[float] = None
        self._model_sha_ok: Optional[bool] = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        import asyncio  # noqa: PLC0415 — local, keeps module import cheap

        await asyncio.to_thread(self._bootstrap_blocking)

    def _bootstrap_blocking(self) -> None:
        """Load the delegate + model. Never raises: a failed bootstrap leaves
        ``ready`` False, which the ingest gate already treats as 'run nothing
        on this backend' rather than crashing the app."""
        try:
            spec = coral_model(self._model_key)
            path = ensure_model(self._models_dir, self._model_key)
            self._input_size = spec["input"]
            self._model_sha_ok = True
            self._tpus = self._bind_devices(path)
            if not self._tpus:
                raise RuntimeError("no Edge TPU could be bound")
            # A rebind that finds FEWER devices than a previous one almost
            # always means a delegate from the old fleet is still holding a
            # device. That is invisible in the routine "CORAL OK" line, so say
            # it separately and greppably. Deliberately NOT self-healing: a
            # reinit loop against an exclusively-claimed device would spin
            # forever and be worse than the fault it is chasing.
            if len(self._tpus) < self._max_devices_seen:
                log.warning(
                    "coral: bound only %d of %d previously-seen Edge TPUs — a "
                    "device is still claimed; restart the container to recover it",
                    len(self._tpus), self._max_devices_seen,
                )
            self._max_devices_seen = max(self._max_devices_seen, len(self._tpus))
            self._pool = queue.PriorityQueue()
            self._pool_seq = 0
            for tpu in self._tpus:
                self._release(tpu)
            self._ready = True
            self._device = "edgetpu"
            log.info(
                "CORAL OK — delegate=%s model=%s devices=%d [%s] outputs=%d",
                self._delegate_so, spec["file"], len(self._tpus),
                ", ".join(t.spec for t in self._tpus),
                len(self._tpus[0].output_indices),
            )
        except Exception:  # noqa: BLE001 — bootstrap must never kill the app
            self._tpus = []
            self._pool = queue.Queue()
            self._ready = False
            self._device = None
            log.exception(
                "CORAL UNAVAILABLE — Edge TPU detector failed to initialise. "
                "Detection is OFF for this backend; set VIGILUME_DETECTOR=onnx "
                "to fall back to the GPU."
            )

    def _checkout(self, pool: "queue.PriorityQueue[tuple[int, int, _Tpu]]") -> _Tpu:
        """Take one device, occasionally forcing a stale one so it stays proven.

        Priority alone means a healthy PCIe card starves USB completely, and a
        never-exercised device cannot be distinguished from a dead one. So if
        some device has not completed an invoke in _IDLE_PROBE_S, spend ONE
        frame on it. At ~55 inferences/sec that is roughly one frame in 3,300 —
        about 0.3%, or +0.03 ms on mean invoke time — which buys a continuously
        verified answer to "are both TPUs alive".

        Done at CHECKOUT, not on release: at concurrency 1 an idle device is
        dequeued once at bootstrap and then sits in the queue as a frozen tuple
        that nothing re-evaluates, so a release-time probe would fire once and
        never again.
        """
        try:
            first = pool.get(timeout=_POOL_WAIT_S)
        except queue.Empty as exc:
            raise RuntimeError(
                f"no Edge TPU became free within {_POOL_WAIT_S:.0f}s "
                "(a device is wedged mid-inference)"
            ) from exc

        now = time.monotonic()
        stale = [t for t in self._tpus
                 if t is not first[2] and now - t.last_ok_mono > _IDLE_PROBE_S]
        if not stale:
            return first[2]

        # Pull entries looking for the stalest device, holding aside whatever
        # we pass. Never BLOCK for it: if it is busy it is self-evidently
        # alive, so fall back to the device we already hold.
        want = min(stale, key=lambda t: t.last_ok_mono)
        aside: list[tuple[int, int, _Tpu]] = []
        chosen = first
        try:
            while True:
                if chosen[2] is want:
                    break
                aside.append(chosen)
                try:
                    chosen = pool.get_nowait()
                except queue.Empty:
                    chosen = aside.pop()   # want is checked out — it is alive
                    break
        finally:
            for entry in aside:
                pool.put(entry)
        return chosen[2]

    def _release(self, tpu: _Tpu) -> None:
        """Return a device to the pool — unless the fleet was rebuilt under it.

        _release_devices() can clear ``_tpus`` and install a fresh pool while a
        detect() thread is still mid-invoke. That thread's ``finally`` then runs
        and, without this guard, puts the PREVIOUS generation's _Tpu into the
        NEW pool at priority 0 — where it outranks everything and serves every
        subsequent frame while being absent from ``_tpus`` and therefore
        invisible in device_stats. Worse, it keeps its interpreter (and so its
        delegate's EXCLUSIVE claim on, say, pci:0) alive, which makes the
        rebind that prompted the teardown come back with one device fewer and
        never recover it short of a container restart. Dropping the interpreter
        here is what lets libedgetpu actually free the device.
        """
        if tpu not in self._tpus:
            tpu.interpreter = None
            log.warning(
                "coral: dropped stale device %s (fleet rebuilt mid-inference)",
                tpu.spec,
            )
            return
        self._pool_seq += 1
        self._pool.put((tpu.priority, self._pool_seq, tpu))

    def _bind_devices(self, model_path: Path) -> list[_Tpu]:
        """Claim up to MAX_TPUS Edge TPUs, newest-first by probe order.

        AUTO-DETECTION IS BY ATTEMPT, not by asking a registry: libedgetpu gives
        a delegate its device EXCLUSIVELY, so a successful load both proves the
        device exists and claims it. A failure means "not present, or already
        taken" — either way, move on.

        FALLS BACK TO THE UNADDRESSED DELEGATE. If no explicit `<type>:<index>`
        spec binds — an older libedgetpu that rejects the options dict, or a
        device topology these four specs do not name — this retries
        `load_delegate(so)` with no options, which is exactly what this class did
        before it could use two. That fallback is the reason adding multi-TPU
        cannot regress a working single-TPU box.
        """
        bound: list[_Tpu] = []
        for spec in _DEVICE_SPECS:
            if len(bound) >= MAX_TPUS:
                break
            try:
                bound.append(self._bind_one(model_path, spec))
                log.info("coral: bound Edge TPU %s", spec)
            except Exception as exc:  # noqa: BLE001 — absent device is the normal case
                log.debug("coral: no Edge TPU at %s (%s)", spec, exc)
        if bound:
            return bound
        log.info(
            "coral: no addressed device bound — retrying the unaddressed delegate "
            "(single-TPU behaviour)"
        )
        return [self._bind_one(model_path, None)]

    def _bind_one(self, model_path: Path, device_spec: Optional[str]) -> _Tpu:
        """Build one interpreter on `device_spec` (None = first available)."""
        interp = self._make_interpreter(model_path, device_spec)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        return _Tpu(
            spec=device_spec or "auto",
            interpreter=interp,
            input_index=inp["index"],
            output_indices=[d["index"] for d in interp.get_output_details()],
            # PCIe (and the unaddressed single-device case) outrank USB.
            priority=1 if (device_spec or "").startswith("usb") else 0,
        )

    def _make_interpreter(self, model_path: Path, device_spec: Optional[str] = None) -> Any:
        """THE ONLY TRANSPORT-SPECIFIC METHOD.

        In-process LiteRT + libedgetpu external delegate. If a box turns out not
        to execute the Edge-TPU custom op in-process, this is the single method
        that moves behind a NumPy-1 sidecar — nothing else in this file assumes
        the interpreter is local.
        """
        from ai_edge_litert.interpreter import (  # noqa: PLC0415 — optional dep
            Interpreter,
            load_delegate,
        )

        # No options dict when unaddressed: an older libedgetpu that does not
        # understand `device` must still work, and that path is the fallback
        # _bind_devices relies on.
        delegate = (
            load_delegate(self._delegate_so, options={"device": device_spec})
            if device_spec
            else load_delegate(self._delegate_so)
        )
        return Interpreter(
            model_path=str(model_path), experimental_delegates=[delegate]
        )

    async def stop(self) -> None:
        self._ready = False
        self._release_devices()

    def _release_devices(self) -> None:
        """Drop every bound interpreter so libedgetpu frees the devices.

        Load-bearing on a model switch and on reinit: a delegate holds its Edge
        TPU EXCLUSIVELY, so re-binding before releasing would find every device
        already taken and fall through to zero TPUs — turning a routine model
        change into "Coral unavailable".
        """
        self._tpus = []
        self._pool = queue.PriorityQueue()
        self._pool_seq = 0

    async def reconfigure(self, model_key: str, confidence: float) -> None:
        """Apply a settings change. Confidence takes effect on the next detect().

        A MODEL change needs a full reload (different artifact, different input
        size), so it re-runs the bootstrap in the background exactly like the
        ONNX path — the swap is not instant and detection briefly reports
        not-ready, which is the same contract the UI already expects. An unknown
        key is ignored rather than raising: never let a bad setting stop
        detection."""
        import asyncio  # noqa: PLC0415

        self._confidence = float(confidence)
        if model_key not in CORAL_MODELS or model_key == self._model_key:
            return
        log.info("coral: switching model %s -> %s", self._model_key, model_key)
        self._model_key = model_key
        self._ready = False
        self._release_devices()
        await asyncio.to_thread(self._bootstrap_blocking)

    # ---------- self-heal hooks (same semantics as OnnxDetector) ----------

    def note_detect_ok(self) -> None:
        self._consecutive_failures = 0

    def note_detect_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _FAILURES_BEFORE_REINIT and not self._needs_reinit:
            log.warning(
                "coral: %d consecutive inference failures — flagging for reinit",
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
        import asyncio  # noqa: PLC0415

        if not force and not self._needs_reinit:
            return False
        self._ready = False
        self._release_devices()
        await asyncio.to_thread(self._bootstrap_blocking)
        self._needs_reinit = False
        self._consecutive_failures = 0
        self._last_reinit_monotonic = time.monotonic()
        return self._ready

    def last_reinit_age_s(self) -> Optional[float]:
        if self._last_reinit_monotonic is None:
            return None
        return round(time.monotonic() - self._last_reinit_monotonic, 1)

    # ---------- inference ----------

    def _invoke(self, tpu: _Tpu, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Run one frame on ONE checked-out TPU and return the four SSD outputs.

        The caller owns `tpu` exclusively for the duration — see detect().
        """
        interp = tpu.interpreter
        interp.set_tensor(tpu.input_index, tensor)
        interp.invoke()
        out = [interp.get_tensor(i) for i in tpu.output_indices]
        boxes, class_ids, scores, count = out[0], out[1], out[2], out[3]
        return boxes, class_ids, scores, int(np.asarray(count).reshape(-1)[0])

    def detect(
        self, frame_bgr: np.ndarray, detect_width: int, detect_height: int
    ) -> sv.Detections:
        """BLOCKING single-frame inference. Same contract as OnnxDetector.detect:
        float32 xyxy in detect-stream pixels, COCO-80 class ids, already
        thresholded at ``self.confidence``."""
        if not self._ready or not self._tpus:
            raise RuntimeError("detector not ready")
        tensor = preprocess(frame_bgr, self._input_size)
        # CHECK OUT one device for the whole invoke. Ingest calls detect() from
        # asyncio.to_thread, so several frames can be in flight at once; with two
        # TPUs bound, two run genuinely in parallel and a third waits here rather
        # than corrupting an interpreter that is mid-invoke.
        #
        # BOUNDED wait. ingest's 8 s asyncio timeout cancels the awaiting
        # coroutine but NEVER interrupts the thread already inside invoke(), so
        # an unbounded get() here would park threads forever behind a wedged
        # device. 5 s sits deliberately under that 8 s so this reports first,
        # as a counted failure that feeds the existing reinit hysteresis.
        #
        # Bind the pool ONCE: a concurrent reinit can swap self._pool, and a
        # thread blocked on the old object would never be woken.
        pool = self._pool
        tpu = self._checkout(pool)
        t0 = time.perf_counter()
        try:
            boxes, class_ids, scores, count = self._invoke(tpu, tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            tpu.inferences += 1
            tpu.total_ms += elapsed_ms
            # Proof of life for THIS device, set only on a completed invoke.
            tpu.last_ok_mono = time.monotonic()
        except Exception:
            # Counted per-device: a TPU that fails every invoke is otherwise
            # indistinguishable from one that is merely idle, since the tally
            # above only ever runs on success.
            tpu.errors += 1
            raise
        finally:
            # ALWAYS return the device, including on a raised inference. Leaking
            # one permanently shrinks the pool, and leaking both deadlocks every
            # subsequent detect() on the get() above — a silent, total stop of
            # detection, which is the one outcome this system must never have.
            self._release(tpu)
        self._last_inference_ms = round(elapsed_ms, 2)
        return decode(
            boxes, class_ids, scores, count,
            self._confidence, detect_width, detect_height,
        )

    # ---------- status surface ----------

    @property
    def kind(self) -> str:
        return "coral"

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

    def _device_stats(self) -> list[dict[str, Any]]:
        # No `or 1` on the denominator: before the first inference every share
        # would read 0.0%, which looks like "nothing is working" on the very
        # screen someone opens to check whether anything is working. null says
        # "not yet known", which is the truth.
        total = sum(t.inferences for t in self._tpus)
        now = time.monotonic()
        return [
            {
                "device": t.spec,
                "inferences": t.inferences,
                "avg_ms": round(t.total_ms / t.inferences, 2) if t.inferences else None,
                "share_pct": round(100.0 * t.inferences / total, 1) if total else None,
                # The two fields that separate "idle by design" from "dead".
                "errors": t.errors,
                "last_ok_age_s": (round(now - t.last_ok_mono, 1)
                                  if t.last_ok_mono else None),
            }
            for t in self._tpus
        ]

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ready": self._ready,
            # `device` stays the single string every existing consumer decodes
            # ("edgetpu"); `devices`/`device_count` carry the new detail so the
            # UI can say "x2" without any of them having to change first.
            "device": self._device,
            "devices": [t.spec for t in self._tpus],
            "device_count": len(self._tpus),
            # Per-device work split. READ THIS AS POLICY, NOT AS SPEED. The
            # pool is a PriorityQueue keyed PCIe-first, so a device is only
            # handed out once every higher-priority device is already checked
            # out. ingest awaits detect() serially from one worker, so
            # concurrency is 1 and the healthy, expected reading on a
            # PCIe+USB box is 100/0 — that is the design, not a dead TPU.
            # To tell idle from dead use `errors` and `last_ok_age_s`: the
            # idle probe in _checkout forces one frame onto any device that has
            # gone _IDLE_PROBE_S without a completed invoke, so a live device
            # never shows last_ok_age_s much above that. A device that is
            # genuinely gone shows a climbing age, or a climbing error count.
            "device_stats": self._device_stats(),
            "model": self.model_key,
            "model_sha_ok": self._model_sha_ok,
            "last_inference_ms": self._last_inference_ms,
            "consecutive_failures": self._consecutive_failures,
            "needs_reinit": self._needs_reinit,
            "last_reinit_age_s": self.last_reinit_age_s(),
        }

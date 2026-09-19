#!/usr/bin/env python3
"""Recognition follows the detector's silicon, automatically and safely.

THE CONTRACT
============
One hardware decision per box, not two that can disagree. Whatever the detector
RESOLVED to is what recognition targets:

    detector on CUDA   ->  CUDA where a stage can use it
    detector on CPU    ->  CPU
    detector on Coral  ->  CPU

THE PART THAT IS EASY TO GET WRONG
----------------------------------
Following the CONFIGURED backend instead of the RESOLVED device. Those differ
exactly when it matters: `backend="gpu"` on a box whose CUDA never came up
reports device="cpu", and `backend="auto"` can land anywhere. Reading the
setting would claim a card that is not there and then fail at session creation —
on a maintenance tick, where the exception costs the whole feature.

CORAL IS CPU, AND THAT IS AN ANSWER
-----------------------------------
An Edge TPU runs int8 graphs compiled for it; YuNet, SFace and the plate OCR are
float ONNX with no Edge TPU build. So the TPU keeps detection and recognition
runs on CPU. The status report must SAY that, because "cpu" with no reason reads
as an oversight rather than a decision.

Offline-runnable; no models, no GPU, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native import accel  # noqa: E402
from app.native.timing import TIMINGS, Stage  # noqa: E402

_failures: list[str] = []
_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


class FakeDetector:
    def __init__(self, device=None, kind="onnx"):
        self.device = device
        self.kind = kind


def resolve_checks() -> None:
    print("\nrecognition follows the detector")
    has_cuda = accel.cuda_available()
    print(f"  (this container reports CUDA available: {has_cuda})")

    gpu = accel.resolve(FakeDetector("cuda", "onnx"))
    if has_cuda:
        check(gpu.device == "cuda" and gpu.providers[0] == accel.CUDA,
              "detector on CUDA -> recognition on CUDA")
        check(accel.CPU in gpu.providers,
              "...with CPU still in the list, so a per-op gap in the CUDA "
              "provider degrades that op instead of failing the session")
    else:
        check(gpu.device == "cpu",
              "detector on CUDA but no CUDA provider in this build -> CPU, "
              "rather than claiming a card that is not there")
        check("no CUDA provider" in gpu.reason, "...and the reason says exactly that")

    cpu = accel.resolve(FakeDetector("cpu", "onnx"))
    check(cpu.device == "cpu" and cpu.providers == [accel.CPU],
          "detector on CPU -> recognition on CPU")

    coral = accel.resolve(FakeDetector("cpu", "coral"))
    check(coral.device == "cpu", "detector on Coral -> recognition on CPU")
    check("Edge TPU" in coral.reason and "float ONNX" in coral.reason,
          "...and says WHY — an Edge TPU cannot run a float ONNX graph, so this "
          "is a decision and not an oversight")

    # A Coral box whose detector somehow reports cuda must STILL be CPU: the
    # kind is the authoritative fact about what the graph can run on.
    odd = accel.resolve(FakeDetector("cuda", "coral"))
    check(odd.device == "cpu", "kind=coral wins over a stale device string")

    print("\nnothing raises on a box that has not started up")
    for detector in (None, FakeDetector(None, ""), FakeDetector("", ""), object()):
        got = accel.resolve(detector)
        check(got.providers == [accel.CPU] and got.device == "cpu",
              f"{type(detector).__name__} resolves to CPU rather than raising")


def report_checks() -> None:
    print("\nthe status report names every stage AND the reason")
    rep = accel.report(FakeDetector("cuda", "onnx"), plate_ocr_device="cpu")
    for stage in ("face_detect", "face_embed", "plate_localize", "plate_ocr"):
        check(stage in rep and "device" in rep[stage] and "why" in rep[stage],
              f"{stage} reports a device and a reason")
    check(rep["plate_ocr"]["device"] == "cpu",
          "plate_ocr reports what the SESSION ACTUALLY BOUND, not what was "
          "requested — asking for CUDA and silently getting CPU is exactly the "
          "case that must not be reported as a GPU stage")
    check(rep["follows_detector"]["device"] == "cuda",
          "and the report says which detector device it is following")
    check("without CUDA" in rep["face_detect"]["why"],
          "a CPU-bound stage explains the constraint (the pip OpenCV wheel), so "
          "the limitation stays visible instead of becoming folklore")


def timing_checks() -> None:
    print("\ntimings are measured, and absence differs from zero")
    stage = Stage("probe")
    check(stage.snapshot() is None,
          "a stage that never ran reports None, NOT zeroes — 'did not run' and "
          "'ran instantly' are different findings")
    for ms in (1.0, 2.0, 3.0, 100.0):
        stage.record(ms)
    snap = stage.snapshot()
    check(snap is not None and snap["calls"] == 4, "calls are counted")
    check(snap["max_ms"] == 100.0, "the worst case is kept")
    check(snap["p95_ms"] >= snap["mean_ms"],
          "p95 >= mean — the mean hides the case that hurts, which is why both "
          "are reported")

    print("\na raising stage is still timed")
    boom = Stage("boom")
    try:
        with boom.measure():
            raise RuntimeError("simulated")
    except RuntimeError:
        pass
    snap = boom.snapshot()
    check(snap is not None and snap["calls"] == 1,
          "a stage that failed slowly is precisely the one worth seeing in the "
          "numbers, so the timer records on the exception path too")

    check(isinstance(TIMINGS.report(), dict), "the process-wide report assembles")


def main() -> int:
    resolve_checks()
    report_checks()
    timing_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (automatic device selection + timings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

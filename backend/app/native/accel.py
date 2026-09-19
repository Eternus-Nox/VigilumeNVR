"""Which silicon each recognition stage runs on, decided automatically.

THE RULE
========
Recognition FOLLOWS THE DETECTOR. Whatever D-FINE actually resolved to at boot
is what recognition targets, so there is one hardware decision on the box rather
than two that can disagree:

    detector on CUDA   ->  recognition uses CUDA where it can
    detector on CPU    ->  recognition uses CPU
    detector on Coral  ->  recognition uses CPU (see below)

It follows the detector's RESOLVED device, never the configured setting. Those
differ exactly when it matters most: `backend="gpu"` on a box whose CUDA did not
come up reports `device="cpu"`, and `backend="auto"` could be any of the three.
Reading the setting would claim an accelerator that is not there and then fail
at session-creation time, on a maintenance tick, where the exception costs the
whole feature.

WHY CORAL IS ALWAYS CPU HERE, AND WHY THAT IS NOT A GAP
-------------------------------------------------------
An Edge TPU executes int8-quantized graphs compiled by the Edge TPU compiler.
YuNet, SFace and the plate OCR are float ONNX models from the permissive tier —
there is no Edge-TPU build of any of them, and quantizing a face embedding to
int8 without re-validating it against real enrollments is how you get a gallery
that silently matches the wrong people.

So on a Coral box the TPU keeps doing what it is good at (detection) and
recognition runs on CPU. That is the honest answer to "coral or CPU", and
`/api/recognition/status` says it in those words rather than leaving an operator
to infer it from a number that never changes.

WHAT CAN ACTUALLY MOVE
----------------------
Only the stages that go through onnxruntime, which today is the plate OCR.
YuNet and SFace are driven by `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`,
and the pip OpenCV wheel is built WITHOUT CUDA — `cv2.cuda.getCudaEnabledDeviceCount()`
is 0 — so they cannot reach the card without either a custom OpenCV build or
re-implementing YuNet's per-stride decode against a raw ORT session. Both are
real options; neither is free, and `report()` names CPU-bound stages with the
reason so the decision stays visible instead of becoming folklore.

NOTHING HERE RAISES
-------------------
Every function returns a usable answer for a box with no GPU, no onnxruntime
providers at all, or a detector that never came up. The fallback is always
CPU, because a slower recognition pass is a far better failure than none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


@dataclass(frozen=True)
class Accel:
    """The resolved answer for one onnxruntime stage."""

    #: Providers to hand InferenceSession, best first, CPU always last.
    providers: list[str]
    #: "cuda" | "cpu" — what the caller should REPORT, not what it asked for.
    device: str
    #: Why, in words an operator can act on. Surfaced by /api/recognition/status.
    reason: str

    @property
    def wants_cuda(self) -> bool:
        return self.device == "cuda"


def cuda_available() -> bool:
    """Is a CUDA execution provider actually registered in this build?

    A registered provider is necessary, not sufficient — a build can advertise
    CUDA while the driver or a cuDNN library is missing, and that only surfaces
    when a session is created. So callers still fall back on failure; this is
    the cheap pre-check that avoids attempting it at all on a CPU-only image.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415 — optional at import time
    except Exception:  # noqa: BLE001
        return False
    try:
        return CUDA in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        log.debug("could not list onnxruntime providers", exc_info=True)
        return False


def detector_device(detector: Any) -> tuple[str, str]:
    """(device, kind) the DETECTOR actually resolved to. ("", "") if unknown.

    `device` is "cuda" | "cpu" | "" and `kind` is "onnx" | "coral" | "". Read
    defensively: the engine may hold a skeleton detector during startup, and a
    detector that failed to load reports device=None rather than raising.
    """
    if detector is None:
        return "", ""
    device = getattr(detector, "device", None) or ""
    kind = getattr(detector, "kind", None) or ""
    return str(device).lower(), str(kind).lower()


def resolve(detector: Any) -> Accel:
    """Providers for an onnxruntime recognition stage, following the detector."""
    device, kind = detector_device(detector)

    if kind == "coral":
        # The TPU is busy with detection and cannot run these graphs anyway.
        return Accel(
            [CPU], "cpu",
            "the Edge TPU runs detection; recognition models are float ONNX "
            "with no Edge TPU build, so they run on CPU",
        )
    if device == "cuda":
        if cuda_available():
            # CPU stays in the list as ORT's own fallback: a per-op gap in the
            # CUDA provider degrades that op instead of failing the session.
            return Accel([CUDA, CPU], "cuda", "following the detector onto CUDA")
        return Accel(
            [CPU], "cpu",
            "the detector reports CUDA but this onnxruntime build registers no "
            "CUDA provider — recognition stays on CPU rather than failing",
        )
    if device == "cpu":
        return Accel([CPU], "cpu", "following the detector, which is on CPU")
    return Accel(
        [CPU], "cpu",
        "the detector has not reported a device yet — CPU until it does",
    )


def make_session(path: str, detector: Any, *, label: str) -> tuple[Any, str]:
    """Create an InferenceSession on the resolved device. Returns (session, device).

    FALLS BACK RATHER THAN RAISING. A CUDA provider can be registered and still
    fail to create a session — a missing cuDNN, a driver mismatch, or a card
    with no free memory because D-FINE took it. Every one of those is a reason
    to run this model on CPU, not a reason for recognition to be unavailable.
    """
    import onnxruntime as ort  # noqa: PLC0415

    accel = resolve(detector)
    if accel.wants_cuda:
        try:
            session = ort.InferenceSession(path, providers=accel.providers)
            # Trust what ORT actually bound, not what was requested: asking for
            # CUDA and silently getting CPU is exactly the case that would
            # otherwise be reported to the operator as a GPU stage.
            bound = list(session.get_providers())
            device = "cuda" if CUDA in bound else "cpu"
            log.info("%s: onnxruntime session on %s (%s)", label, device, ", ".join(bound))
            return session, device
        except Exception:  # noqa: BLE001
            log.warning(
                "%s: CUDA session failed, falling back to CPU", label, exc_info=True
            )
    session = ort.InferenceSession(path, providers=[CPU])
    log.info("%s: onnxruntime session on cpu — %s", label, accel.reason)
    return session, "cpu"


def report(detector: Any, *, plate_ocr_device: Optional[str] = None) -> dict[str, Any]:
    """Per-stage device report for /api/recognition/status.

    Names the CPU-BOUND stages and why, because "cpu" with no reason reads as an
    oversight. `plate_ocr_device` is what the live session actually bound, which
    can differ from what `resolve` wanted — that is the whole point of passing it
    rather than recomputing.
    """
    accel = resolve(detector)
    device, kind = detector_device(detector)
    return {
        "follows_detector": {"device": device or "unknown", "kind": kind or "unknown"},
        "face_detect": {
            "device": "cpu",
            "why": "cv2.FaceDetectorYN runs on OpenCV's DNN backend, and the pip "
                   "OpenCV wheel is built without CUDA",
        },
        "face_embed": {
            "device": "cpu",
            "why": "cv2.FaceRecognizerSF, same OpenCV DNN backend. alignCrop is "
                   "geometry rather than a network and gains nothing from a GPU",
        },
        "plate_localize": {
            "device": "cpu",
            "why": "classical CV (Sobel + morphology), not a model",
        },
        "plate_ocr": {
            "device": plate_ocr_device or accel.device,
            "why": accel.reason,
        },
    }

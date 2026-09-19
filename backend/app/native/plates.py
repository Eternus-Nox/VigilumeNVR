"""Plate localization and OCR.

NO LEARNED PLATE DETECTOR, DELIBERATELY
=======================================
Every high-accuracy license-plate detector available today descends from a
GPL-3.0 (YOLOv9) or AGPL-3.0 (Ultralytics) training codebase. AGPL in
particular would oblige VigilumeNVR to offer its own source to anyone using it
over a network, which is not a licence this project can absorb for one feature.
So plates are localized WITHOUT a dedicated model:

    D-FINE already found the vehicle  ->  crop its lower region
    classical CV finds plate-shaped high-contrast strips in that crop
    a permissively-licensed OCR (MIT) reads each strip
    multi-frame voting reconciles the reads

This is the pre-deep-learning ANPR pipeline, and it is honest about what it
buys: it reads a plate on a driveway well and a plate across the street badly.
That trade was made knowingly. If the licence position ever changes, only
`candidate_regions` has to be replaced — everything downstream takes boxes.

WHY A CRUDE LOCALIZER IS ACCEPTABLE HERE
----------------------------------------
Because the OCR is a good discriminator on its own. Fed pure noise the pinned
model returns an EMPTY string at zero confidence rather than inventing
characters (verified in plates_smoke). That means this stage can afford to be
GENEROUS — proposing several regions per vehicle and letting OCR confidence,
the aspect-ratio veto in bestshot, and multi-frame voting throw away the ones
that were never plates. A localizer that has to be precise would need the model
we are deliberately not shipping; one that only has to be *inclusive* does not.

THE OCR MODEL
-------------
`cct_xs_v2_global` from fast-plate-ocr (MIT, github.com/ankandrew/cnn-ocr-lp) —
3.3 MB, a Compact Convolutional Transformer trained in Keras. Nothing in its
lineage touches YOLO. It covers the Latin-alphabet regions including the United
States, takes a 128x64 RGB uint8 image, and emits 10 slots x 37 classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import accel
from .timing import TIMINGS
from .detector import ensure_model, model_path, sha256_file

log = logging.getLogger(__name__)

# Revision-pinned + SHA-256 verified, like every other artifact here. The hash
# was produced by downloading this exact release asset and hashing it.
PLATE_MODELS: dict[str, dict[str, Any]] = {
    "plate_ocr": {
        "url": (
            "https://github.com/ankandrew/cnn-ocr-lp/releases/download/"
            "arg-plates/cct_xs_v2_global.onnx"
        ),
        "bytes": 3_344_292,
        "sha256": "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44",
        "license": "MIT",
    },
}

#: Decode configuration for the pinned model. These come from its companion
#: `cct_xs_v2_global_plate_config.yaml` and are inlined rather than fetched
#: because the alphabet IS the decode: a model swap that changed it would need
#: a code change here anyway, and a silently mismatched alphabet would decode
#: every plate into confident nonsense.
OCR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
OCR_PAD_CHAR = "_"
OCR_INPUT_W = 128
OCR_INPUT_H = 64
OCR_MAX_SLOTS = 10

#: Per-character confidence below which a read is not worth voting with. The
#: model is well calibrated on this axis — it returns an empty string at zero
#: confidence for noise — so this mostly filters half-legible strips.
OCR_MIN_CONFIDENCE = 0.55

#: Plates live low on a vehicle. Searching the whole car wastes work and offers
#: the OCR grille slats and badges, which are the only things that look like
#: plates to a contrast-based localizer.
VEHICLE_LOWER_FRACTION = 0.55

#: Aspect band for a candidate strip.
#:
#: NOT the ~2:1 of a whole plate. A gradient localizer responds to the
#: CHARACTERS, not the plate's border, so what a contour actually bounds is the
#: TEXT ROW — which on a US plate is roughly 5-7:1 and on a European one
#: narrower still. Filtering these with the whole-plate band rejected a clean,
#: perfectly legible plate at 6.17:1, which is exactly the "too precise"
#: failure this localizer is supposed to avoid. The band is therefore wide, and
#: `_expand_to_plate` below reconstructs plate proportions afterwards.
MIN_ASPECT = 1.2
MAX_ASPECT = 9.0

#: Fraction of a plate's height its characters occupy. Used to grow a text-row
#: box back out to the whole plate before OCR, which was trained on plate
#: crops rather than bare text rows.
TEXT_HEIGHT_FRACTION = 0.6

#: A strip narrower than this carries too few pixels per character to read.
MIN_PLATE_W = 48
MIN_PLATE_H = 12

#: How many candidate strips to propose per vehicle, best-scoring first.
MAX_REGIONS = 4

#: Labels whose boxes are searched for plates.
VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle", "motorbike", "van")


# ---------------------------------------------------------------------------
# Localization (classical CV, no model)
# ---------------------------------------------------------------------------


def candidate_regions(
    vehicle_bgr: np.ndarray,
    *,
    max_regions: int = MAX_REGIONS,
) -> list[tuple[int, int, int, int]]:
    """Plate-shaped bright/dark strips in a vehicle crop, best first.

    Returns boxes as ``(x1, y1, x2, y2)`` in the CROP's coordinates — callers
    translate by the crop origin, exactly as the face pass does.

    The method is the classical one: a blackhat transform with a wide,
    short kernel responds to dark characters on a light background (and its
    tophat counterpart to the inverse), a Sobel-x pass emphasises the vertical
    strokes that letters are made of, and a wide morphological close merges the
    characters of one plate into a single blob. Contours over that give
    plate-shaped candidates.
    """
    with TIMINGS.plate_localize.measure():
        return _candidate_regions(vehicle_bgr, max_regions=max_regions)


def _candidate_regions(
    vehicle_bgr: np.ndarray,
    *,
    max_regions: int = MAX_REGIONS,
) -> list[tuple[int, int, int, int]]:
    """The real body. Split only so the timer wraps exactly the work."""
    if vehicle_bgr is None or vehicle_bgr.size == 0:
        return []
    h, w = vehicle_bgr.shape[:2]
    if w < MIN_PLATE_W or h < MIN_PLATE_H:
        return []

    # Only the lower part of the vehicle. `top` is kept so boxes can be
    # reported in the caller's crop coordinates rather than the sub-crop's.
    top = int(h * (1.0 - VEHICLE_LOWER_FRACTION))
    region = vehicle_bgr[top:, :]
    if region.size == 0:
        return []

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    gray = cv2.bilateralFilter(gray, 5, 40, 40)

    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    # Both polarities: dark-on-light (most plates) and light-on-dark (many
    # European and some US night captures under IR).
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect)
    combined = cv2.max(blackhat, tophat)

    grad = cv2.Sobel(combined, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.absolute(grad)
    lo, hi = float(grad.min()), float(grad.max())
    if hi - lo < 1e-6:
        return []
    grad = ((grad - lo) / (hi - lo) * 255).astype(np.uint8)

    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw < MIN_PLATE_W or ch < MIN_PLATE_H:
            continue
        aspect = cw / float(ch)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        # Prefer bigger strips and ones closest to a plate's text row, but keep
        # this a soft ranking — it decides ORDER, never inclusion.
        area = cw * ch
        aspect_fit = 1.0 / (1.0 + abs(aspect - 5.0))
        box = _expand_to_plate(cx, cy + top, cx + cw, cy + ch + top, w, h)
        scored.append((area * aspect_fit, box))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [box for _, box in scored[:max_regions]]


def _expand_to_plate(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """Grow a text-row box out to the whole plate, clamped to the crop.

    The contour bounds the characters; the OCR was trained on plates, which
    carry a border and some margin. Growing vertically by the character-height
    fraction (plus a little horizontally) hands it something closer to its
    training distribution. Clamped rather than centred-and-scaled so a plate at
    the very bottom of a vehicle crop does not get shifted off it.
    """
    th = y2 - y1
    pad_y = int(th * (1.0 / TEXT_HEIGHT_FRACTION - 1.0) / 2.0)
    pad_x = int((x2 - x1) * 0.04)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def deskew(strip_bgr: np.ndarray) -> np.ndarray:
    """Straighten a plate strip using the dominant text angle.

    A few degrees of rotation costs OCR real accuracy, and a driveway camera
    almost never sees a plate level. Returns the input unchanged when no
    confident angle is found — a wrong rotation is worse than none.
    """
    if strip_bgr is None or strip_bgr.size == 0:
        return strip_bgr
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY) if strip_bgr.ndim == 3 else strip_bgr
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(255 - thresh)
    if coords is None or len(coords) < 20:
        return strip_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    # Beyond this the estimate is usually the box, not the text.
    if abs(angle) < 0.5 or abs(angle) > 20:
        return strip_bgr
    h, w = strip_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(strip_bgr, m, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


class PlateReader:
    """The pinned OCR model. Loads lazily; never raises into the caller."""

    def __init__(self, models_dir: Path, detector: Any = None) -> None:
        # The DETECTOR, so this session can follow whatever silicon it
        # actually resolved to (native/accel.py). Optional so a test can
        # construct a reader with no engine around it; None resolves to CPU.
        self._detector = detector
        # What the session actually BOUND, which is not always what was
        # asked for — reported by /api/recognition/status.
        self.device = "cpu"
        self._models_dir = Path(models_dir)
        self._session: Any = None
        self._input_name = ""
        self._failed = False

    @property
    def ready(self) -> bool:
        return self._session is not None

    async def ensure_models(self) -> None:
        for key, pin in PLATE_MODELS.items():
            await ensure_model(self._models_dir, key, pin=pin)

    async def load(self) -> bool:
        """Download, verify and open the session. Never raises."""
        if self.ready:
            return True
        import asyncio

        try:
            await self.ensure_models()
            await asyncio.to_thread(self._build_blocking)
            self._failed = False
            log.info("plate OCR ready (cct_xs_v2_global)")
            return True
        except Exception:
            if not self._failed:
                log.exception("plate OCR unavailable — model not loaded")
            self._failed = True
            self._session = None
            return False

    def _build_blocking(self) -> None:
        import onnxruntime as ort

        path = model_path(self._models_dir, "plate_ocr")
        if not path.is_file():
            raise FileNotFoundError(f"plate OCR model missing at {path}")
        # Re-verify on load, not only on download: a model that rots on disk
        # would decode into confident nonsense rather than failing.
        digest = sha256_file(path)
        if digest != PLATE_MODELS["plate_ocr"]["sha256"]:
            raise ValueError(
                f"plate OCR on disk hashes to {digest[:12]}… — pin is "
                f"{PLATE_MODELS['plate_ocr']['sha256'][:12]}…"
            )
        # Device follows the DETECTOR automatically — CUDA when it is on
        # CUDA, CPU when it is on CPU or an Edge TPU (which cannot run this
        # float graph at all). make_session falls back to CPU rather than
        # raising if the CUDA session cannot be created: a missing cuDNN or a
        # card D-FINE has filled is a reason to run on CPU, not a reason for
        # plate reading to be unavailable.
        self._session, self.device = accel.make_session(
            str(path), self._detector, label="plate OCR"
        )
        self._input_name = self._session.get_inputs()[0].name

    def close(self) -> None:
        self._session = None

    def read_blocking(self, strip_bgr: np.ndarray) -> tuple[str, float]:
        """Read one plate strip. Returns ``(text, mean_confidence)``.

        ``("", 0.0)`` for an unreadable strip — including one that was never a
        plate. The model is well behaved here: fed noise it emits only padding
        rather than inventing characters, which is what lets the localizer
        upstream be generous.
        """
        if not self.ready or strip_bgr is None or strip_bgr.size == 0:
            return "", 0.0
        with TIMINGS.plate_ocr.measure():
            return self._read_blocking(strip_bgr)

    def _read_blocking(self, strip_bgr: np.ndarray) -> tuple[str, float]:
        """The real body. Split only so the timer wraps exactly the work."""
        if not self.ready or strip_bgr is None or strip_bgr.size == 0:
            return "", 0.0
        try:
            resized = cv2.resize(
                strip_bgr, (OCR_INPUT_W, OCR_INPUT_H), interpolation=cv2.INTER_LINEAR
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            outputs = self._session.run(None, {self._input_name: rgb[None].astype(np.uint8)})
        except Exception:
            log.exception("plate OCR inference failed")
            return "", 0.0

        slots = np.asarray(outputs[0])[0]  # (OCR_MAX_SLOTS, len(alphabet))
        chars: list[str] = []
        confs: list[float] = []
        for slot in range(min(OCR_MAX_SLOTS, slots.shape[0])):
            idx = int(slots[slot].argmax())
            if idx >= len(OCR_ALPHABET):
                continue
            ch = OCR_ALPHABET[idx]
            if ch == OCR_PAD_CHAR:
                continue
            chars.append(ch)
            confs.append(float(slots[slot][idx]))
        if not chars:
            return "", 0.0
        # MEAN over the characters actually emitted. The weakest character is
        # what `vote_plate` cares about and it computes that itself across
        # frames; here the useful number is how confident this whole read was.
        return "".join(chars), sum(confs) / len(confs)

    async def read(self, strip_bgr: np.ndarray) -> tuple[str, float]:
        import asyncio

        return await asyncio.to_thread(self.read_blocking, strip_bgr)

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "failed": self._failed,
            "models": {
                key: {
                    "present": model_path(self._models_dir, key).is_file(),
                    "bytes": pin["bytes"],
                    "license": pin["license"],
                }
                for key, pin in PLATE_MODELS.items()
            },
        }


def is_vehicle(label: str) -> bool:
    return (label or "").lower() in VEHICLE_LABELS

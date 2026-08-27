"""Night contrast boost for the DETECTOR's input frame.

WHAT THIS IS FOR. Running a camera without IR keeps the image in colour and
avoids washing out close subjects, but leaves the detector very little to work
with once the light goes. This lifts local contrast on the frame handed to
inference, so an object that is present-but-flat has a better chance of being
found.

WHAT IT IS NOT. It does not touch recordings, clips, live view, or the snapshot
saved with an event. The recorder stream-copies from the camera and never sees
this; ingest boosts a COPY and passes the original on to the engine. That
separation is deliberate: the archive is evidence, and evidence should be what
the camera saw, not what an algorithm inferred. It also means a bad boost can
degrade detection but can never corrupt footage.

TWO HONEST LIMITS, stated here because the settings UI cannot say them at
length:

1. CONTRAST CANNOT CREATE SIGNAL. On a frame with genuinely no photons — no
   moon, no streetlight, no porch light — there is nothing to amplify but
   sensor noise, and CLAHE will amplify it enthusiastically. This helps a dim
   scene; it does nothing for a black one.

2. THE MODEL WAS NOT TRAINED ON BOOSTED IMAGES. D-FINE learned on ordinary
   photographs, so changing the input distribution can help OR hurt, and which
   one depends on the camera, the scene and the light. That is why the default
   is off and why `auto` exists: the honest way to use this is to turn it on,
   watch the detection rate on the camera that bothers you, and turn it off
   again if it does not improve.

WHY CLAHE AND NOT A CONTRAST SLIDER. A global contrast/gamma stretch moves
every pixel by the same rule, so a scene with one bright porch light and a dark
lawn just clips the light and leaves the lawn dark. CLAHE equalises within
small tiles, so the lawn gets stretched on its own terms. `clipLimit` bounds
how far any tile may stretch, which is exactly the noise-amplification knob.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# off    — never boost (default; the model sees exactly what the camera sent)
# auto   — boost only frames measured darker than the threshold
# always — boost every frame, for A/B testing against `off`
VALID_MODES = ("off", "auto", "always")
DEFAULT_MODE = "off"

# Mean luma (0-255) below which `auto` considers a frame "night". 60 is well
# under a lit indoor scene and comfortably above a genuinely black frame, so it
# separates "dim" from "daylight" without firing on a dark-but-fine image.
DEFAULT_DARK_THRESHOLD = 60
# How far CLAHE may stretch one tile. 2.0 is the conventional starting point:
# visibly more local contrast, without the grain explosion that 4.0+ produces
# on an already-noisy high-gain night frame.
CLIP_LIMIT = 2.0
# Tile grid. 8x8 keeps each tile large enough to hold real scene statistics at
# detection resolution; finer grids start equalising individual objects.
TILE_GRID = (8, 8)


def mean_luma(frame_bgr: np.ndarray) -> float:
    """Average brightness 0-255.

    Measured on a strided sample rather than the whole frame: this runs per
    frame per camera, and brightness is a scene-wide property that a quarter of
    the pixels estimates just as well.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    sample = frame_bgr[::2, ::2]
    # Rec.601 luma weights on BGR, matching cv2's own COLOR_BGR2GRAY.
    return float(np.dot(sample.reshape(-1, 3).mean(axis=0), (0.114, 0.587, 0.299)))


def boost_contrast(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the LUMA channel only, returning a NEW frame.

    Luma only, via LAB: running CLAHE on B, G and R independently equalises
    each channel to its own histogram, which shifts the white balance and turns
    a night scene lurid. Separating lightness from colour changes what the
    detector can see without inventing colours that were never there.

    Never raises — a boost failure must degrade to the original frame, not take
    down the detection worker.
    """
    try:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
        return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)
    except cv2.error:
        log.warning("night boost failed on a frame — using it unmodified", exc_info=True)
        return frame_bgr


def maybe_boost(
    frame_bgr: np.ndarray,
    *,
    mode: str = DEFAULT_MODE,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
) -> tuple[np.ndarray, bool]:
    """``(frame_for_detection, was_boosted)``.

    The returned frame is a NEW array when boosted and the SAME OBJECT when
    not — callers rely on the un-boosted path costing nothing, and on the
    original never being modified in place (the engine keeps it as the event
    snapshot).
    """
    if mode not in VALID_MODES or mode == "off" or frame_bgr is None:
        return (frame_bgr, False)
    if mode == "auto" and mean_luma(frame_bgr) >= dark_threshold:
        return (frame_bgr, False)
    return (boost_contrast(frame_bgr), True)


def settings_for(detection: dict[str, Any]) -> tuple[str, int]:
    """``(mode, dark_threshold)`` from settings.detection, defensively.

    An unknown mode degrades to "off" rather than to a boost: a typo in a
    settings document must not silently start altering what the detector sees.
    """
    mode = str((detection or {}).get("night_boost") or DEFAULT_MODE).strip().lower()
    if mode not in VALID_MODES:
        mode = DEFAULT_MODE
    raw: Optional[Any] = (detection or {}).get("night_boost_threshold")
    try:
        threshold = int(raw) if raw is not None else DEFAULT_DARK_THRESHOLD
    except (TypeError, ValueError):
        threshold = DEFAULT_DARK_THRESHOLD
    return (mode, max(0, min(255, threshold)))

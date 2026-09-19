"""Face detection + embedding: the only module here that runs a model.

SPLIT OF RESPONSIBILITIES
=========================
    bestshot.py    — is this crop legible?          (OpenCV maths, no model)
    recognition.py — who does this vector belong to? (numpy, no model)
    recognizer.py  — turn a frame into faces and vectors.  <- you are here

Keeping the first two model-free is what lets their thresholds and tie-breaks
be tested against fixtures with no weights on disk. This module is the part
that cannot be tested that way, so it is kept as thin as possible: load two
pinned files, call into OpenCV, hand back plain data.

WHY OPENCV'S OWN IMPLEMENTATIONS, NOT A HAND-ROLLED ONNX GRAPH
---------------------------------------------------------------
cv2 ships ``FaceDetectorYN`` and ``FaceRecognizerSF``, which consume exactly
these two pinned files. Using them instead of driving the ONNX graphs directly
buys two things that are easy to get wrong and silent when wrong:

  * YuNet's decode. Its raw outputs are per-stride cls/obj/bbox/kps heads that
    need priors and NMS. A subtly wrong decode does not crash — it returns
    plausible boxes in the wrong places.
  * ``alignCrop``. SFace embeddings are only comparable when the face has been
    warped to a canonical 112x112 by a similarity transform from the 5
    landmarks. Skipping or approximating that alignment does not fail either;
    it just quietly costs accuracy, and it would cost it asymmetrically across
    poses, which is the worst possible failure for a recognition gallery.

The detector stays on onnxruntime (CUDA/Coral); this runs on OpenCV's CPU DNN
backend. That is the right trade here: a face crop pass runs on a handful of
small images per event, not on every frame of every camera, and keeping it off
the GPU leaves the whole card for D-FINE.

THE MODELS, AND WHY THESE TWO
-----------------------------
Both are pinned by revision + SHA-256 like every other artifact in this repo,
and both are permissive:

  * YuNet (MIT) — 233 KB. Emits the 5-point landmarks that bestshot's pose
    scoring wants, so the quality score is measured rather than guessed.
  * SFace (Apache-2.0) — 37 MB, 128-d embeddings.

They are deliberately NOT in detector.MODELS: they are not detector tiers, must
never show up in the model picker, and the ModelStore keeps no per-key state
for them. They reuse ``ensure_model``'s download-and-verify path via its
``pin=`` override instead.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .detector import ensure_model, model_path, sha256_file

from .timing import TIMINGS

log = logging.getLogger(__name__)

# Revision-pinned, SHA-256-verified. Both hashes were verified by downloading
# the pinned-revision file and hashing it; they match the repository's own
# git-lfs object ids. `media.githubusercontent.com/media/...` is the URL that
# serves the real binary — the ordinary raw.githubusercontent.com path returns
# a 131-byte LFS POINTER, which would download "successfully" and then fail
# every hash check.
_ZOO_REV = "47534e27c9851bb1128ccc0102f1145e27f23f98"
_ZOO_BASE = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{_ZOO_REV}/models"

FACE_MODELS: dict[str, dict[str, Any]] = {
    "yunet": {
        "url": f"{_ZOO_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "bytes": 232_589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "license": "MIT",
    },
    "sface": {
        "url": f"{_ZOO_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "bytes": 38_696_353,
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "license": "Apache-2.0",
    },
}

#: Identifies the embedding space in `profile_samples.model_key`. CHANGE THIS
#: WHENEVER THE EMBEDDING MODEL CHANGES — it is the only thing that stops
#: vectors from two models being compared, which produces meaningless numbers
#: in an entirely plausible range. Bumping it makes every existing sample
#: report as stale and prompts a re-enroll, which is the correct, visible
#: outcome (see Gallery.build and /api/recognition/status).
EMBEDDING_MODEL_KEY = "sface-2021dec"

#: SFace's own documented operating point is a cosine of 0.363. recognition.py
#: defaults slightly above it (0.38) on purpose: a missed match surfaces as
#: "unknown" and can be enrolled, while a false one silently attaches a
#: stranger to someone's name.
SFACE_REFERENCE_COSINE = 0.363

#: YuNet's detection confidence floor. Deliberately not low: a false face costs
#: a candidate row, a wasted embedding, and a stranger's crop in the reviewable
#: store, none of which are free.
FACE_SCORE_THRESHOLD = 0.7
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 50

#: Faces smaller than this in the source frame are not worth embedding. It
#: mirrors bestshot.FACE_MIN_PX; below it the aligned 112x112 crop is mostly
#: interpolation.
MIN_FACE_PX = 40


class FaceDetection:
    """One detected face, in the coordinate space of the frame passed in."""

    __slots__ = ("box", "landmarks", "score", "_raw")

    def __init__(self, raw: np.ndarray) -> None:
        # YuNet row: [x, y, w, h, 10 landmark coords, score]
        x, y, w, h = (float(v) for v in raw[:4])
        self.box: tuple[float, float, float, float] = (x, y, x + w, y + h)
        self.landmarks: list[list[float]] = raw[4:14].reshape(5, 2).astype(float).tolist()
        self.score: float = float(raw[14])
        # Kept verbatim because alignCrop consumes YuNet's own row format; the
        # parsed fields above are for everyone else.
        self._raw = raw

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"<FaceDetection {self.box} score={self.score:.2f}>"


class FaceRecognizer:
    """Loads YuNet + SFace and turns frames into faces and embeddings.

    Construction does NOT load anything: `ensure_models()` downloads and
    verifies, and the cv2 objects are built lazily on first use. A box that
    never enables recognition therefore pays nothing, and a box whose download
    failed degrades to `ready == False` rather than raising into the ingest
    loop.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)
        self._detector: Any = None
        self._embedder: Any = None
        self._input_size: tuple[int, int] = (0, 0)
        self._failed = False
        self._lock = asyncio.Lock()

    # -- lifecycle ------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._detector is not None and self._embedder is not None

    @property
    def model_key(self) -> str:
        """The embedding space id, or "" when nothing is loaded.

        Empty is meaningful: /api/recognition/status reports every enrolled
        sample as unusable, which is correct — with no model loaded nothing can
        be matched.
        """
        return EMBEDDING_MODEL_KEY if self.ready else ""

    def paths(self) -> dict[str, Path]:
        return {key: model_path(self._models_dir, key) for key in FACE_MODELS}

    async def ensure_models(self) -> None:
        """Download + verify both pinned files. Idempotent; safe to call often."""
        for key, pin in FACE_MODELS.items():
            await ensure_model(self._models_dir, key, pin=pin)

    async def load(self) -> bool:
        """Ensure the files are present and build the cv2 objects.

        Returns True when ready. Never raises: recognition is an enhancement,
        and a box that cannot fetch a model must keep recording and detecting.
        """
        async with self._lock:
            if self.ready:
                return True
            try:
                await self.ensure_models()
                await asyncio.to_thread(self._build_blocking)
                self._failed = False
                log.info(
                    "face recognition ready (yunet + sface, embedding space %r)",
                    EMBEDDING_MODEL_KEY,
                )
                return True
            except Exception:
                if not self._failed:
                    # Logged once per failure streak, not per retry — a box with
                    # no outbound network would otherwise fill its log.
                    log.exception("face recognition unavailable — models not loaded")
                self._failed = True
                self._detector = self._embedder = None
                return False

    def _build_blocking(self) -> None:
        import cv2  # local import keeps module import light

        paths = self.paths()
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{key} model missing at {path}")
            # Re-verify on load, not only on download. A model file that rots on
            # disk (bad RAM, a half-written upgrade, a truncated restore) would
            # otherwise load into a silently wrong embedding space, and every
            # enrolled face would drift without anything reporting an error.
            digest = sha256_file(path)
            if digest != FACE_MODELS[key]["sha256"]:
                raise ValueError(
                    f"{key} on disk hashes to {digest[:12]}… — pin is "
                    f"{FACE_MODELS[key]['sha256'][:12]}…"
                )

        self._input_size = (320, 320)
        self._detector = cv2.FaceDetectorYN.create(
            str(paths["yunet"]), "", self._input_size,
            FACE_SCORE_THRESHOLD, FACE_NMS_THRESHOLD, FACE_TOP_K,
        )
        self._embedder = cv2.FaceRecognizerSF.create(str(paths["sface"]), "")

    def close(self) -> None:
        self._detector = self._embedder = None

    # -- inference ------------------------------------------------------

    def detect_blocking(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Faces in a BGR frame. CPU-bound — call via `detect()` off the loop."""
        if not self.ready or frame_bgr is None or frame_bgr.size == 0:
            return []
        with TIMINGS.face_detect.measure():
            return self._detect_blocking(frame_bgr)

    def _detect_blocking(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """The real body. Split only so the timer wraps exactly the work."""
        h, w = frame_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return []
        # setInputSize is stateful on the cv2 object, so it must be set for
        # every frame shape — a stale size silently rescales the boxes.
        if (w, h) != self._input_size:
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        try:
            _, faces = self._detector.detect(frame_bgr)
        except Exception:
            log.exception("face detection failed on a %dx%d frame", w, h)
            return []
        if faces is None:
            return []
        out = [FaceDetection(row) for row in faces]
        return [f for f in out if min(f.width, f.height) >= MIN_FACE_PX]

    def align_blocking(
        self, frame_bgr: np.ndarray, face: FaceDetection
    ) -> Optional[np.ndarray]:
        """Warp one detected face to the canonical 112x112 SFace input.

        Split from `feature_blocking` deliberately. Alignment is cheap and is
        needed on EVERY pass (it is what the quality score is measured on and
        what the best-shot buffer keeps); the embedding is the expensive half
        and is needed once per track. Doing both together meant paying for an
        embedding on every frame and throwing it away.
        """
        with TIMINGS.face_align.measure():
            return self._align_blocking(frame_bgr, face)

    def _align_blocking(
        self, frame_bgr: np.ndarray, face: FaceDetection
    ) -> Optional[np.ndarray]:
        """The real body. Split only so the timer wraps exactly the work."""
        if not self.ready:
            return None
        try:
            return self._embedder.alignCrop(frame_bgr, face._raw)
        except Exception:
            log.exception("face alignment failed")
            return None

    def feature_blocking(self, aligned_bgr: np.ndarray) -> Optional[np.ndarray]:
        """128-d embedding for an ALREADY-ALIGNED 112x112 crop.

        Takes the aligned crop rather than (frame, face) so a shot kept in the
        best-shot buffer can be embedded later, long after its frame is gone.
        Passing anything not produced by `align_blocking` here is a bug: SFace
        embeddings are only comparable in the canonical geometry, and an
        unaligned crop does not fail, it just quietly lands in the wrong place.
        """
        with TIMINGS.face_embed.measure():
            return self._feature_blocking(aligned_bgr)

    def _feature_blocking(self, aligned_bgr: np.ndarray) -> Optional[np.ndarray]:
        """The real body. Split only so the timer wraps exactly the work."""
        if not self.ready or aligned_bgr is None or aligned_bgr.size == 0:
            return None
        try:
            feature = self._embedder.feature(aligned_bgr)
        except Exception:
            log.exception("face embedding failed")
            return None
        if feature is None:
            return None
        return np.asarray(feature, dtype=np.float32).ravel()

    def embed_blocking(
        self, frame_bgr: np.ndarray, face: FaceDetection
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(aligned 112x112 crop, 128-d embedding) — both halves at once."""
        aligned = self.align_blocking(frame_bgr, face)
        if aligned is None:
            return None, None
        return aligned, self.feature_blocking(aligned)

    # -- async wrappers -------------------------------------------------

    async def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        return await asyncio.to_thread(self.detect_blocking, frame_bgr)

    async def align(
        self, frame_bgr: np.ndarray, face: FaceDetection
    ) -> Optional[np.ndarray]:
        return await asyncio.to_thread(self.align_blocking, frame_bgr, face)

    async def feature(self, aligned_bgr: np.ndarray) -> Optional[np.ndarray]:
        return await asyncio.to_thread(self.feature_blocking, aligned_bgr)

    async def embed(
        self, frame_bgr: np.ndarray, face: FaceDetection
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return await asyncio.to_thread(self.embed_blocking, frame_bgr, face)

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "failed": self._failed,
            "model_key": self.model_key,
            "models": {
                key: {
                    "present": model_path(self._models_dir, key).is_file(),
                    "bytes": pin["bytes"],
                    "license": pin["license"],
                }
                for key, pin in FACE_MODELS.items()
            },
        }


def crop_with_origin(
    frame_bgr: np.ndarray,
    box: Sequence[float],
    *,
    pad: float = 0.0,
) -> Optional[tuple[np.ndarray, int, int]]:
    """Crop `box` from a frame, returning ``(crop, origin_x, origin_y)``.

    The ORIGIN is why this exists separately from `crop_box`. Anything detected
    inside the crop — a face found in a person box — comes back in the crop's
    own coordinates, and is meaningless anywhere else until it is translated by
    the offset the crop was taken at. Losing that offset is a silent bug: the
    coordinates stay plausible and simply point at the wrong part of the frame.

    Returns None for a degenerate result rather than a zero-sized array, so
    callers have one thing to check instead of two.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if pad > 0:
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        return None
    return frame_bgr[yi1:yi2, xi1:xi2], xi1, yi1


def crop_box(
    frame_bgr: np.ndarray,
    box: Sequence[float],
    *,
    pad: float = 0.0,
) -> Optional[np.ndarray]:
    """`crop_with_origin` without the offset, for callers that stay in-crop."""
    got = crop_with_origin(frame_bgr, box, pad=pad)
    return None if got is None else got[0]

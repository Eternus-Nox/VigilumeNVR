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

OPENCV'S OWN IMPLEMENTATIONS ARE THE REFERENCE
----------------------------------------------
cv2 ships ``FaceDetectorYN`` and ``FaceRecognizerSF``, which consume exactly
these two pinned files, and on a CPU box they do the work. They are also the
reference the GPU path below is measured against, because two things here are
easy to get wrong and silent when wrong:

  * YuNet's decode. Its raw outputs are per-stride cls/obj/bbox/kps heads that
    need priors and NMS. A subtly wrong decode does not crash — it returns
    plausible boxes in the wrong places.
  * ``alignCrop``. SFace embeddings are only comparable when the face has been
    warped to a canonical 112x112 by a similarity transform from the 5
    landmarks. Skipping or approximating that alignment does not fail either;
    it just quietly costs accuracy, and it would cost it asymmetrically across
    poses, which is the worst possible failure for a recognition gallery.

ON THE GPU WHEN THE DETECTOR IS
-------------------------------
OpenCV in the image is the pip wheel, built without CUDA, so the two cv2
classes above can only ever run on the CPU. When the detector resolved to CUDA
(native/accel.py — recognition follows the detector), both networks instead
run on onnxruntime's CUDA provider:

  * YuNet is decoded here, by a line-for-line port of OpenCV's own
    ``FaceDetectorYN`` post-processing (priors per stride, sqrt(cls*obj),
    exp() box sizes, NMS through cv2.dnn.NMSBoxes). Measured against cv2 on
    the same 640x640 input: identical faces, identical scores, 0.000 px apart.
    The graph DECLARES a fixed 640x640 input, but every reshape inside it is
    size-independent — which is how OpenCV runs it at any size. So the input
    (and output) dimensions are rewritten to symbolic ones as the file is
    loaded (``dynamic_yunet``, a few lines of protobuf wire format, no extra
    dependency), and each frame is fed exactly as OpenCV feeds it: padded
    with zeros to a multiple of 32. Same input, same network, same decode —
    the same faces, bit for bit. (Placing frames on a fixed 640 canvas was
    tried first; the extra padding changes the network's output near the
    frame's edge, and a face cut off by the crop edge then came out
    differently.)
  * SFace takes exactly the blob ``FaceRecognizerSF.feature`` builds (RGB,
    unscaled, 112x112). Measured: cosine 1.000000 against cv2's embedding on
    every face tried, so the embedding space — and every enrolled sample — is
    unchanged. EMBEDDING_MODEL_KEY does not move.
  * ``alignCrop`` stays OpenCV's: it is geometry, cheap on any CPU, and the one
    step whose exact reimplementation would buy nothing but risk.

On a CPU-only box the cv2 classes stay in use: on a CPU they are as fast or
faster (SFace measured 13 ms in cv2 vs 24 ms in onnxruntime), so "GPU when
there is one" is the whole rule.

BIG FRAMES ARE SEARCHED SMALL AND ALIGNED BIG
---------------------------------------------
A full-resolution person crop can be 1500 px tall. YuNet finds faces down to
~10 px, so searching it at 640 px loses nothing, while searching it at full
size costs ~10x on a CPU. The landmarks are scaled back up, and alignment reads
the FULL-resolution pixels — the 112x112 the embedding sees is cut from the
real detail, not from the shrunk copy.

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
import threading
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

#: YuNet's three detection heads.
_YUNET_STRIDES = (8, 16, 32)

#: Frames larger than this on their longest side are shrunk before face
#: DETECTION; alignment still reads the full-resolution pixels.
DETECT_MAX_SIDE = 640

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

    def __init__(
        self,
        models_dir: Path,
        *,
        detector: Any = None,
        runtime: str = "auto",
    ) -> None:
        """`detector` is the object detector, followed onto its silicon
        (native/accel.py). `runtime` is "auto" (onnxruntime on CUDA, OpenCV
        otherwise), or "onnxruntime" / "opencv" to force one — tests use the
        former to exercise the GPU code path on a CPU-only box."""
        self._models_dir = Path(models_dir)
        self._object_detector = detector
        self._runtime_pref = runtime
        self._detector: Any = None
        self._embedder: Any = None
        # onnxruntime sessions; None while the cv2 objects do the work.
        self._yunet: Any = None
        self._sface: Any = None
        self._yunet_outputs: list[str] = []
        #: What actually runs each network: "opencv" | "onnxruntime", and the
        #: silicon onnxruntime bound ("cuda" | "cpu").
        self.runtime = "opencv"
        self.device = "cpu"
        # The detector's (device, kind) when the models were built — see
        # stale_device().
        self._built_for: tuple[str, str] = ("", "")
        self._input_size: tuple[int, int] = (0, 0)
        self._failed = False
        self._lock = asyncio.Lock()
        # The cv2 face objects are NOT thread-safe — the detector's input size
        # is state, and OpenCV 5's graph engine asserts when two threads run
        # one net at once ("buf.shape() == m.shape()"). The live pass, a
        # snapshot look and a recorded burst can all reach them together, so
        # every cv2 call holds this. (onnxruntime sessions need no lock.)
        self._cv_lock = threading.Lock()

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
                if self._runtime_pref == "auto":
                    # Follow the detector's RESOLVED device, so wait for it to
                    # have one (bounded) — see accel.settle.
                    from . import accel  # noqa: PLC0415

                    await accel.settle(self._object_detector)
                await asyncio.to_thread(self._build_blocking)
                self._failed = False
                log.info(
                    "face recognition ready (yunet + sface on %s/%s, embedding space %r)",
                    self.runtime, self.device, EMBEDDING_MODEL_KEY,
                )
                return True
            except Exception:
                if not self._failed:
                    # Logged once per failure streak, not per retry — a box with
                    # no outbound network would otherwise fill its log.
                    log.exception("face recognition unavailable — models not loaded")
                self._failed = True
                self.close()
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
        # Built in both modes: alignCrop is OpenCV's, whichever runtime embeds.
        self._embedder = cv2.FaceRecognizerSF.create(str(paths["sface"]), "")
        self._yunet = self._sface = None
        self.runtime, self.device = "opencv", "cpu"
        from . import accel  # noqa: PLC0415

        self._built_for = accel.detector_device(self._object_detector)
        if self._wants_onnxruntime():
            self._build_onnxruntime(paths)

    def stale_device(self) -> bool:
        """The detector has moved since these models were built (typically:
        it was still warming up at boot and has since come up on CUDA), so a
        rebuild would put them somewhere else. Only for runtime "auto"."""
        if not self.ready or self._runtime_pref != "auto":
            return False
        from . import accel  # noqa: PLC0415

        return accel.stale(self._object_detector, self._built_for)

    def _wants_onnxruntime(self) -> bool:
        if self._runtime_pref == "opencv":
            return False
        if self._runtime_pref == "onnxruntime":
            return True
        from . import accel  # noqa: PLC0415

        return accel.resolve(self._object_detector).wants_cuda

    def _build_onnxruntime(self, paths: dict[str, Path]) -> None:  # noqa: C901
        """YuNet + SFace on onnxruntime, following the detector. On "auto", a
        session that could only bind the CPU is dropped again: there the cv2
        objects are as fast or faster, so the GPU path is used only where it is
        a GPU path."""
        from . import accel  # noqa: PLC0415

        try:
            yunet, ydev = accel.make_session(
                dynamic_yunet(paths["yunet"].read_bytes()), self._object_detector,
                label="face detector (YuNet)", quiet=True,
                # Every crop is its own size, so cuDNN must not benchmark
                # every algorithm for each new shape (EXHAUSTIVE, the default,
                # costs far more than the inference it is choosing for).
                cuda_options={"cudnn_conv_algo_search": "HEURISTIC"},
            )
            sface, sdev = accel.make_session(
                str(paths["sface"]), self._object_detector, label="face embedder (SFace)", quiet=True
            )
        except Exception:  # noqa: BLE001
            log.warning("face models: onnxruntime unavailable — using OpenCV", exc_info=True)
            return
        if self._runtime_pref == "auto" and "cpu" in (ydev, sdev):
            log.info("face models: CUDA did not bind — staying on OpenCV (faster on CPU)")
            return
        self._yunet, self._sface = yunet, sface
        self._yunet_outputs = [o.name for o in yunet.get_outputs()]
        self.runtime = "onnxruntime"
        self.device = "cuda" if (ydev, sdev) == ("cuda", "cuda") else "cpu"

    def close(self) -> None:
        self._detector = self._embedder = None
        self._yunet = self._sface = None
        self.runtime, self.device = "opencv", "cpu"

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
        # Search big frames small; the rows are scaled back below, so every
        # coordinate handed out — and alignCrop's input — is in FULL pixels.
        scale = min(1.0, DETECT_MAX_SIDE / float(max(w, h)))
        search = frame_bgr
        if scale < 1.0:
            import cv2  # noqa: PLC0415

            search = cv2.resize(
                frame_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        try:
            if self._yunet is not None:
                faces = self._yunet_detect(search)
            else:
                faces = self._cv2_detect(search)
        except Exception:
            log.exception("face detection failed on a %dx%d frame", w, h)
            return []
        if faces is None or len(faces) == 0:
            return []
        faces = np.asarray(faces, dtype=np.float32).copy()
        if scale < 1.0:
            faces[:, :14] /= scale
        out = [FaceDetection(row) for row in faces]
        return [f for f in out if min(f.width, f.height) >= MIN_FACE_PX]

    def _cv2_detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        with self._cv_lock:
            # setInputSize is stateful on the cv2 object, so it must be set for
            # every frame shape — a stale size silently rescales the boxes.
            if (w, h) != self._input_size:
                self._detector.setInputSize((w, h))
                self._input_size = (w, h)
            _, faces = self._detector.detect(frame_bgr)
            return faces

    def _yunet_detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """YuNet on onnxruntime: rows in OpenCV's format, in frame pixels."""
        blob, pw, ph = yunet_blob(frame_bgr)
        return yunet_decode(
            self._yunet.run(self._yunet_outputs, {"input": blob}),
            self._yunet_outputs, pw, ph,
        )

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
            with self._cv_lock:
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
            if self._sface is not None:
                import cv2  # noqa: PLC0415

                # Exactly the blob FaceRecognizerSF.feature builds: RGB,
                # unscaled, 112x112 — which is what keeps the embedding space
                # (and every enrolled sample) identical across runtimes.
                blob = cv2.dnn.blobFromImage(
                    aligned_bgr, 1.0, (112, 112), (0, 0, 0), True, False
                )
                feature = self._sface.run(None, {self._sface.get_inputs()[0].name: blob})[0]
            else:
                with self._cv_lock:
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
            # What runs the two networks: "onnxruntime" on the GPU when the
            # detector is on CUDA, "opencv" on the CPU otherwise.
            "runtime": self.runtime if self.ready else "",
            "device": self.device if self.ready else "",
            "models": {
                key: {
                    "present": model_path(self._models_dir, key).is_file(),
                    "bytes": pin["bytes"],
                    "license": pin["license"],
                }
                for key, pin in FACE_MODELS.items()
            },
        }


def yunet_blob(frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(NCHW float blob, padded width, padded height): the frame padded with
    zeros on the right and bottom to a multiple of 32 — BGR, unscaled —
    exactly the input OpenCV's FaceDetectorYN builds."""
    h, w = frame_bgr.shape[:2]
    pw, ph = ((w - 1) // 32 + 1) * 32, ((h - 1) // 32 + 1) * 32
    canvas = np.zeros((ph, pw, 3), np.float32)
    canvas[:h, :w] = frame_bgr
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), pw, ph


def yunet_decode(
    outputs: Sequence[np.ndarray],
    names: Sequence[str],
    pad_w: int,
    pad_h: int,
    *,
    score_threshold: float = FACE_SCORE_THRESHOLD,
    nms_threshold: float = FACE_NMS_THRESHOLD,
    top_k: int = FACE_TOP_K,
) -> np.ndarray:
    """YuNet's raw heads (for a ``pad_w`` x ``pad_h`` input) -> OpenCV-format
    rows ``[x, y, w, h, 10 landmark coords, score]`` in frame pixels.

    A port of OpenCV's FaceDetectorYN::postProcess: for each stride, every
    grid cell's score is sqrt(clamp(cls) * clamp(obj)); the box centre is
    (cell + offset) * stride and its size exp(raw) * stride; landmarks are
    (cell + offset) * stride. NMS runs on integer boxes, as OpenCV's does.
    """
    import cv2  # noqa: PLC0415

    heads = dict(zip(names, outputs))
    rows = []
    for st in _YUNET_STRIDES:
        cols, nrows = pad_w // st, pad_h // st
        n = cols * nrows
        cls = np.clip(np.asarray(heads[f"cls_{st}"]).reshape(n), 0.0, 1.0)
        obj = np.clip(np.asarray(heads[f"obj_{st}"]).reshape(n), 0.0, 1.0)
        score = np.sqrt(cls * obj)
        idx = np.nonzero(score >= score_threshold)[0]
        if idx.size == 0:
            continue
        r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        bb = np.asarray(heads[f"bbox_{st}"]).reshape(n, 4)[idx]
        kp = np.asarray(heads[f"kps_{st}"]).reshape(n, 10)[idx]
        cx, cy = (c + bb[:, 0]) * st, (r + bb[:, 1]) * st
        bw, bh = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
        lm = np.empty((idx.size, 10), np.float32)
        lm[:, 0::2] = (kp[:, 0::2] + c[:, None]) * st
        lm[:, 1::2] = (kp[:, 1::2] + r[:, None]) * st
        rows.append(np.column_stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh, lm, score[idx]]))
    if not rows:
        return np.zeros((0, 15), np.float32)
    det = np.vstack(rows).astype(np.float32)
    boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in det]
    keep = np.asarray(
        cv2.dnn.NMSBoxes(boxes, det[:, 14].tolist(), score_threshold, nms_threshold,
                         top_k=top_k)
    ).reshape(-1)
    return det[keep]


# ---------- YuNet with a dynamic input size ----------
#
# The pinned file declares input [1, 3, 640, 640] and outputs [1, 6400, 1] etc.
# The graph itself reshapes with [1, -1, k] throughout, so any multiple-of-32
# size runs; only the DECLARED shapes stop onnxruntime from accepting one.
# These rewrite exactly those dimensions to symbolic names in the serialized
# ModelProto — protobuf wire format, walked by hand so the image needs no
# `onnx` package. Field numbers are from onnx.proto3:
#   ModelProto.graph = 7; GraphProto.input = 11, output = 12, value_info = 13;
#   ValueInfoProto.name = 1, type = 2; TypeProto.tensor_type = 1;
#   TypeProto.Tensor.shape = 2; TensorShapeProto.dim = 1;
#   Dimension.dim_value = 1, dim_param = 2.

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def _map_len_fields(buf: bytes, field: int, fn: Any) -> bytes:
    """`buf` with every length-delimited `field` replaced by fn(payload, k)
    (k = its index among those fields; None drops the field); everything
    else copied verbatim."""
    out = bytearray()
    i = k = 0
    while i < len(buf):
        key, j = _read_varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            _, end = _read_varint(buf, j)
        elif wire == 1:
            end = j + 8
        elif wire == 5:
            end = j + 4
        elif wire == 2:
            length, start = _read_varint(buf, j)
            end = start + length
            if num == field:
                payload = fn(buf[start:end], k)
                k += 1
                if payload is not None:
                    out += _varint(key) + _varint(len(payload)) + payload
                i = end
                continue
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out += buf[i:end]
        i = end
    return bytes(out)


def _field_string(buf: bytes, field: int) -> str:
    found: list[str] = []
    _map_len_fields(buf, field, lambda p, k: found.append(p.decode()) or p)
    return found[0] if found else ""


def _symbolic_dims(value_info: bytes, dims: dict[int, str]) -> bytes:
    def shape(p: bytes, _k: int) -> bytes:
        return _map_len_fields(
            p, 1, lambda d, k: (_varint(2 << 3 | 2) + _varint(len(dims[k].encode()))
                                + dims[k].encode()) if k in dims else d,
        )

    def tensor(p: bytes, _k: int) -> bytes:
        return _map_len_fields(p, 2, shape)

    def type_(p: bytes, _k: int) -> bytes:
        return _map_len_fields(p, 1, tensor)

    return _map_len_fields(value_info, 2, type_)


def dynamic_yunet(model: bytes) -> bytes:
    """YuNet's serialized model with its input's H and W, and every output's
    anchor count, made symbolic. See the block comment above.

    The graph's value_info — shape hints for its 94 intermediate tensors, all
    recorded at 640x640 — is dropped: it is optional, and stale hints would
    only invite onnxruntime to plan for the wrong shapes."""
    def graph(g: bytes, _k: int) -> bytes:
        g = _map_len_fields(g, 13, lambda _vi, _i: None)
        g = _map_len_fields(
            g, 11, lambda vi, _i: _symbolic_dims(vi, {2: "height", 3: "width"})
            if _field_string(vi, 1) == "input" else vi,
        )
        return _map_len_fields(
            g, 12, lambda vi, _i: _symbolic_dims(vi, {1: "anchors_" + _field_string(vi, 1)}),
        )

    return _map_len_fields(model, 7, graph)


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

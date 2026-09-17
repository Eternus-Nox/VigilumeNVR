"""Best-shot selection — pick the frames worth recognizing from, not the frame
that happened to score highest for the OBJECT detector.

WHY THIS IS A SEPARATE IDEA FROM engine.py's `best_frame`
========================================================
The engine already keeps one "best" frame per event, chosen by the detector's
confidence that a person is present (``_adopt_best``). That is the right rule
for a notification thumbnail and the wrong rule for recognition. Detector
confidence answers "is this a person?", which a large, centred, motion-blurred
subject answers very well — and a blurred face is exactly the one a face
embedding cannot read. The two questions are close to uncorrelated:

    detector score   -> is the OBJECT there?
    shot quality     -> is the DETAIL legible?

So recognition gets its own selection pass over its own crops, with its own
score, and the event snapshot keeps using the detector's pick. Neither one
degrades the other.

WHAT "A FEW EXTRA SNAPSHOTS" BUYS
---------------------------------
A face crossing a doorway is sharp in maybe one frame in five: the rest are
mid-stride, mid-blink, or half-turned. Reading the first frame that clears the
detector threshold therefore throws away most of the information the camera
already captured. Keeping a small ranked buffer over the object's whole visit
and reading the best of it is a large accuracy win for a small constant memory
cost (``KEEP_SHOTS`` crops per track, each a fraction of a frame).

The same buffer is what the operator enrolls from: `shots()` returns the ranked
list, the app shows it, and a person picks the reference image. That is the
"which image should we train from" question, answered with real candidates
instead of one arbitrary grab.

DIVERSITY, NOT JUST TOP-N
-------------------------
Ranking by quality alone fills the buffer with five copies of the same instant:
consecutive frames of a slow-moving subject score almost identically, so the
top five are frames n..n+4 of one stride. That is five samples of one pose,
which is worth barely more than one — and it is actively harmful for
enrollment, where the whole point is covering several poses. So a new shot must
be ``MIN_GAP_S`` away in time from every shot it does not outright replace. The
buffer ends up holding the best shot from each of several distinct moments.

NO MODEL RUNS HERE
------------------
Everything in this module is OpenCV and numpy on a crop. It does not import
onnxruntime, does not touch the model store, and has no opinion about which
embedding model will eventually read the winner. That keeps it testable without
a single downloaded weight — and it is the reason the quality thresholds can be
tuned against real footage without re-running inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables. Every one of these is empirical; they are named and gathered here
# so they can be tuned against real footage instead of hunted through the code.
# ---------------------------------------------------------------------------

#: How many ranked shots to retain per track, per kind. Five is enough to give
#: an operator a real choice at enrollment and to let multi-frame plate voting
#: (see recognition.vote_plate) outvote a single misread, without holding a
#: meaningful amount of image memory per tracked object.
KEEP_SHOTS = 5

#: Minimum spacing between two RETAINED shots. Below this they are near-
#: duplicates of one instant (see DIVERSITY above). 0.4 s is ~2 frames at the
#: 5 fps detect stream these cameras run, which is the shortest gap that
#: reliably produces a different pose rather than a different noise pattern.
MIN_GAP_S = 0.4

#: A crop scoring below this is never retained at all. It exists to keep the
#: buffer from filling with garbage on a track that is never legible — at which
#: point `best()` correctly returns nothing and the event carries no
#: recognition, rather than a confident reading of a smear.
MIN_QUALITY = 0.25

#: Face embedding models in the permissive tier take a 112x112 aligned crop.
#: A face arriving at least this wide needs no upscaling, so it is the point
#: where the resolution term saturates.
FACE_TARGET_PX = 112
#: Below this a face crop carries less information than the embedding's input
#: layer consumes; upscaling invents detail and the embedding drifts. Scored 0.
FACE_MIN_PX = 40

#: Plate OCR needs horizontal pixels per character far more than it needs
#: height: ~100 px across a US plate is the usual floor for a clean read.
PLATE_TARGET_PX = 160
PLATE_MIN_PX = 64

#: North American plates are 12x6 inches — 2.0:1. Perspective on a driveway
#: camera legitimately squeezes that, so the acceptable band is wide; outside
#: it the "plate" is either not a plate or is being read at an angle no OCR
#: will survive.
PLATE_ASPECT_IDEAL = 2.0
PLATE_ASPECT_MIN = 1.3
PLATE_ASPECT_MAX = 4.0

#: Variance-of-Laplacian reference. The sharpness term is a saturating curve
#: rather than a threshold, so there is no cliff: `1 - exp(-var/REF)` reaches
#: ~0.63 at REF and ~0.86 at 2xREF. Calibrated on 112 px face crops off a
#: 704x480 detect stream; plates get a higher reference because their
#: high-contrast glyph edges produce more Laplacian energy for the same
#: perceived sharpness.
SHARPNESS_REF_FACE = 120.0
SHARPNESS_REF_PLATE = 260.0

#: Fraction of pixels at the very ends of the range before exposure is called
#: clipped. IR illuminators blow out a close face and headlights blow out a
#: plate, and in both cases the detail is gone, not merely dim.
CLIP_FRACTION_BAD = 0.08


@dataclass(frozen=True)
class Quality:
    """A crop's legibility, decomposed so the UI can say WHY a shot won.

    `total` is the weighted blend in 0..1. The components are kept because
    "0.31" tells an operator nothing, while "sharp enough, but only 44 px
    across" tells them to move the camera or draw a tighter plate zone.
    """

    total: float
    sharpness: float
    resolution: float
    exposure: float
    geometry: float
    reason: str

    def __bool__(self) -> bool:  # `if quality:` reads as "is it usable"
        return self.total >= MIN_QUALITY


@dataclass
class Shot:
    """One retained candidate crop and everything needed to use or show it."""

    crop: np.ndarray
    quality: Quality
    frame_time: float
    #: Box in DETECT-STREAM pixels, matching the rest of the pipeline's
    #: coordinate space (engine.py converts to normalized only at the API edge).
    box: tuple[float, float, float, float]
    tracker_id: int
    kind: str


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def _gray(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.ndim == 2:
        return crop_bgr
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)


def _sharpness(gray: np.ndarray, ref: float) -> float:
    """Variance of the Laplacian, mapped onto a saturating 0..1 curve.

    Variance-of-Laplacian is the standard cheap focus measure: it responds to
    edge energy, which is exactly what motion blur and defocus destroy. It is
    NOT scale-invariant — the same face at twice the size scores higher — which
    is fine here because bigger genuinely is more legible, and the resolution
    term below already accounts for size on its own axis.
    """
    if gray.size == 0:
        return 0.0
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return 1.0 - math.exp(-var / ref)


def _resolution(px: float, target: float, floor: float) -> float:
    """Ramp from `floor` (unusable, 0.0) to `target` (saturated, 1.0).

    Square-rooted so the curve is generous in the middle: a 70 px face is much
    more than half as useful as a 112 px one, because the embedding upscales
    gracefully until it runs out of real detail.
    """
    if px <= floor:
        return 0.0
    if px >= target:
        return 1.0
    return math.sqrt((px - floor) / (target - floor))


def _exposure(gray: np.ndarray) -> float:
    """Penalize crops whose detail has been clipped away at either end.

    Mean brightness is deliberately NOT the measure: a correctly exposed face
    at night is dark, and punishing that would make the buffer prefer the
    blown-out frame where the IR illuminator caught them. What actually costs
    information is pixels pinned at 0 or 255, so that is what is measured.
    """
    if gray.size == 0:
        return 0.0
    total = float(gray.size)
    crushed = float(np.count_nonzero(gray <= 2)) / total
    blown = float(np.count_nonzero(gray >= 253)) / total
    clipped = crushed + blown
    if clipped <= 0.0:
        return 1.0
    return max(0.0, 1.0 - clipped / CLIP_FRACTION_BAD)


def _face_geometry(
    crop_shape: tuple[int, ...], landmarks: Optional[Sequence[Sequence[float]]]
) -> tuple[float, str]:
    """How square-on the face is, from 5-point landmarks when we have them.

    Landmarks are the honest way to measure pose, and most face detectors in
    the permissive tier emit them (left eye, right eye, nose, left mouth
    corner, right mouth corner) for free alongside the box. When they are
    absent we return a NEUTRAL score and say so, rather than inventing a pose
    estimate out of the bounding box — box aspect ratio tracks how the detector
    was trained far more than it tracks which way someone is facing, and a
    confident wrong answer here silently poisons the ranking.
    """
    if not landmarks or len(landmarks) < 5:
        return 0.6, "pose unknown (no landmarks)"

    pts = np.asarray(landmarks, dtype=np.float64)[:5]
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    eye_span = float(np.linalg.norm(right_eye - left_eye))
    if eye_span < 1e-3:
        return 0.0, "degenerate landmarks"

    # YAW: a face turned away pushes the nose off the midpoint between the
    # eyes, measured in eye-spans so it is scale free. ~0.0 square on, ~0.5 is
    # roughly a three-quarter view, beyond which an embedding degrades sharply.
    eye_mid = (left_eye + right_eye) / 2.0
    yaw = abs(float(nose[0] - eye_mid[0])) / eye_span
    yaw_score = max(0.0, 1.0 - yaw / 0.5)

    # ROLL: a tilted head is much less damaging than a turned one (alignment
    # can rotate it back), so it is scored gently — 30 degrees costs half.
    dy, dx = float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0])
    roll_deg = abs(math.degrees(math.atan2(dy, dx)))
    roll_deg = min(roll_deg, 180.0 - roll_deg)
    roll_score = max(0.0, 1.0 - roll_deg / 60.0)

    score = 0.7 * yaw_score + 0.3 * roll_score
    if yaw_score < 0.4:
        return score, "turned away"
    if roll_score < 0.5:
        return score, "head tilted"
    return score, "square on"


def _plate_geometry(w: float, h: float) -> tuple[float, str]:
    """How close the crop is to a plate seen face-on.

    A plate read at a steep angle is a plate the OCR will hallucinate
    characters into, and its projected aspect ratio is the cheapest available
    signal for that. The band is wide because a driveway camera never sees a
    plate perfectly square on and should not be told it has failed.
    """
    if h <= 0:
        return 0.0, "degenerate box"
    aspect = w / h
    if aspect < PLATE_ASPECT_MIN:
        return 0.0, f"too square ({aspect:.1f}:1) — steep angle"
    if aspect > PLATE_ASPECT_MAX:
        return 0.0, f"too wide ({aspect:.1f}:1) — steep angle"
    # Triangular falloff either side of ideal, normalized by the distance to
    # whichever band edge lies on that side.
    edge = PLATE_ASPECT_MAX if aspect > PLATE_ASPECT_IDEAL else PLATE_ASPECT_MIN
    score = 1.0 - abs(aspect - PLATE_ASPECT_IDEAL) / abs(edge - PLATE_ASPECT_IDEAL)
    return max(0.0, score), "face on" if score > 0.6 else "angled"


def _blend(parts: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted blend, with DISQUALIFIERS applied as a veto rather than a
    deduction.

    Some components are not "worse", they are "no". A face below the embedding
    model's input size carries less information than the network consumes, and
    a crop whose aspect ratio is nowhere near a plate is not a plate seen at a
    bad angle — it is something else. Left as weighted terms, either one gets
    carried over the usability line by a high sharpness score: a razor-sharp
    24 px face blended to 0.65, which would have put an unreadable crop at the
    top of the enrollment list.

    So a zero in a vetoing component zeroes the total. Every other component
    still trades off against the rest, which is what a weighted score is for.
    """
    for key in ("res", "veto"):
        if parts.get(key, 1.0) <= 0.0:
            return 0.0
    return sum(parts[k] * weights[k] for k in weights)


def score_face(
    crop_bgr: np.ndarray,
    *,
    landmarks: Optional[Sequence[Sequence[float]]] = None,
) -> Quality:
    """Legibility of a face crop for an embedding model.

    Sharpness dominates because it is the failure an embedding cannot recover
    from: a small sharp face still embeds close to its enrolled self, while a
    large blurred one embeds close to nothing in particular.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return Quality(0.0, 0.0, 0.0, 0.0, 0.0, "empty crop")
    h, w = crop_bgr.shape[:2]
    gray = _gray(crop_bgr)

    sharp = _sharpness(gray, SHARPNESS_REF_FACE)
    res = _resolution(float(min(h, w)), FACE_TARGET_PX, FACE_MIN_PX)
    expo = _exposure(gray)
    geo, geo_reason = _face_geometry(crop_bgr.shape, landmarks)

    total = _blend(
        {"sharp": sharp, "res": res, "expo": expo, "geo": geo},
        {"sharp": 0.40, "res": 0.25, "geo": 0.25, "expo": 0.10},
    )
    return Quality(
        total=total,
        sharpness=sharp,
        resolution=res,
        exposure=expo,
        geometry=geo,
        reason=_face_reason(sharp, res, expo, geo_reason, min(h, w)),
    )


def _face_reason(
    sharp: float, res: float, expo: float, geo_reason: str, px: int
) -> str:
    """The single most useful sentence about this crop, worst problem first."""
    if res <= 0.0:
        return f"too small — {px} px across"
    if sharp < 0.35:
        return "soft — motion blur or out of focus"
    if expo < 0.4:
        return "clipped — blown highlights or crushed shadows"
    if geo_reason in ("turned away", "head tilted", "degenerate landmarks"):
        return geo_reason
    if res < 0.5:
        return f"usable but small — {px} px across"
    return "good"


def score_plate(crop_bgr: np.ndarray) -> Quality:
    """Legibility of a plate crop for OCR.

    Weighted harder toward sharpness and horizontal resolution than the face
    score is, because OCR fails character-by-character on exactly those two and
    is comparatively tolerant of odd exposure (plates are retroreflective and
    high contrast by design).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return Quality(0.0, 0.0, 0.0, 0.0, 0.0, "empty crop")
    h, w = crop_bgr.shape[:2]
    gray = _gray(crop_bgr)

    sharp = _sharpness(gray, SHARPNESS_REF_PLATE)
    # WIDTH, not min(w, h): OCR needs pixels per character along the string.
    res = _resolution(float(w), PLATE_TARGET_PX, PLATE_MIN_PX)
    expo = _exposure(gray)
    geo, geo_reason = _plate_geometry(float(w), float(h))

    total = _blend(
        # `veto` carries the geometry term for plates: an aspect ratio outside
        # the band means this is not a plate seen face-on, and OCR on it
        # produces invented characters rather than a poor read.
        {"sharp": sharp, "res": res, "expo": expo, "geo": geo, "veto": geo},
        {"sharp": 0.45, "res": 0.30, "geo": 0.15, "expo": 0.10},
    )
    return Quality(
        total=total,
        sharpness=sharp,
        resolution=res,
        exposure=expo,
        geometry=geo,
        reason=_plate_reason(sharp, res, expo, geo_reason, w),
    )


def _plate_reason(
    sharp: float, res: float, expo: float, geo_reason: str, px: int
) -> str:
    if res <= 0.0:
        return f"too small — {px} px wide"
    if geo_reason.startswith(("too square", "too wide", "degenerate")):
        return geo_reason
    if sharp < 0.35:
        return "soft — motion blur"
    if expo < 0.4:
        return "clipped — headlight glare or deep shadow"
    if res < 0.5:
        return f"usable but small — {px} px wide"
    return "good"


def score(kind: str, crop_bgr: np.ndarray, **kw) -> Quality:
    """Dispatch on kind, so callers can stay generic over faces and plates."""
    if kind == "face":
        return score_face(crop_bgr, landmarks=kw.get("landmarks"))
    if kind == "plate":
        return score_plate(crop_bgr)
    raise ValueError(f"unknown crop kind {kind!r}")


# ---------------------------------------------------------------------------
# The buffer
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    shots: list[Shot] = field(default_factory=list)


class BestShotBuffer:
    """Ranked, time-diverse candidate crops per (tracker_id, kind).

    Lifetime is the tracked object's, not the event's: a track that never opens
    an event still fills a slot, and `forget()` is what clears it. The engine
    already prunes tracker ids on its own schedule (see zones.forget_tracks),
    and `forget_all()` exists for a camera-level reset.
    """

    def __init__(
        self,
        *,
        keep: int = KEEP_SHOTS,
        min_gap_s: float = MIN_GAP_S,
        min_quality: float = MIN_QUALITY,
    ) -> None:
        self._keep = max(1, int(keep))
        self._min_gap = max(0.0, float(min_gap_s))
        self._min_quality = float(min_quality)
        self._slots: dict[tuple[int, str], _Slot] = {}

    # -- writing --------------------------------------------------------

    def offer(
        self,
        *,
        tracker_id: int,
        kind: str,
        crop_bgr: np.ndarray,
        box: tuple[float, float, float, float],
        frame_time: float,
        landmarks: Optional[Sequence[Sequence[float]]] = None,
        quality: Optional[Quality] = None,
    ) -> Optional[Shot]:
        """Offer one crop. Returns the retained Shot, or None if it was dropped.

        `quality` may be passed in when the caller has already scored the crop
        (the engine scores once and uses the number for both the buffer and the
        event row); otherwise it is computed here.
        """
        q = quality if quality is not None else score(kind, crop_bgr, landmarks=landmarks)
        if q.total < self._min_quality:
            return None

        slot = self._slots.setdefault((tracker_id, kind), _Slot())
        shot = Shot(
            # COPY. The caller's crop is a view into a frame buffer that the
            # ingest loop reuses; retaining it without copying would leave the
            # buffer full of whatever is on screen several seconds later.
            crop=np.ascontiguousarray(crop_bgr).copy(),
            quality=q,
            frame_time=float(frame_time),
            box=tuple(float(v) for v in box),  # type: ignore[assignment]
            tracker_id=int(tracker_id),
            kind=kind,
        )

        # A shot too close in time to one already held REPLACES it when better,
        # rather than taking a second slot. That is what keeps the retained set
        # spread across the visit instead of clustered on one instant.
        for i, held in enumerate(slot.shots):
            if abs(held.frame_time - shot.frame_time) < self._min_gap:
                if shot.quality.total > held.quality.total:
                    slot.shots[i] = shot
                    slot.shots.sort(key=lambda s: s.quality.total, reverse=True)
                    return shot
                return None

        slot.shots.append(shot)
        slot.shots.sort(key=lambda s: s.quality.total, reverse=True)
        if len(slot.shots) > self._keep:
            dropped = slot.shots[self._keep :]
            del slot.shots[self._keep :]
            if shot in dropped:
                return None
        return shot

    # -- reading --------------------------------------------------------

    def best(self, tracker_id: int, kind: str) -> Optional[Shot]:
        shots = self.shots(tracker_id, kind)
        return shots[0] if shots else None

    def shots(self, tracker_id: int, kind: str) -> list[Shot]:
        """Retained shots, best first. The list is a copy; the Shots are not."""
        slot = self._slots.get((tracker_id, kind))
        return list(slot.shots) if slot else []

    def tracks(self, kind: Optional[str] = None) -> list[int]:
        return sorted(
            {tid for (tid, k) in self._slots if kind is None or k == kind}
        )

    # -- lifetime -------------------------------------------------------

    def forget(self, tracker_id: int) -> None:
        for key in [k for k in self._slots if k[0] == tracker_id]:
            del self._slots[key]

    def forget_missing(self, live_ids: Iterable[int]) -> None:
        """Drop every track not in `live_ids` — the tracker's own pruning is
        the authority on which ids still exist, exactly as zones.forget_tracks
        keeps sv.LineZone's history bounded against the same set."""
        live = set(live_ids)
        for key in [k for k in self._slots if k[0] not in live]:
            del self._slots[key]

    def forget_all(self) -> None:
        self._slots.clear()

    def __len__(self) -> int:
        return sum(len(s.shots) for s in self._slots.values())

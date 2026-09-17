#!/usr/bin/env python3
"""Best-shot selection, the profile gallery, and multi-frame plate voting.

These two modules were written so that the part of recognition most likely to
be WRONG can be tested with no model weights and no GPU: thresholds, tie-breaks,
how disagreeing reads are reconciled, and what happens when a gallery is fed
garbage. Everything here is synthetic imagery and hand-built vectors.

What gets the most attention, because it would be quietly dangerous:

1. THE MARGIN TEST. Two enrolled people who look alike must resolve to UNKNOWN,
   not to a coin flip. A confident wrong name on a security event is worse than
   no name at all.
2. MODEL MIXING. Embeddings from two different models are not comparable, and
   their similarity is a plausible-looking number. The gallery must DROP the
   mismatched samples rather than score them.
3. TIME DIVERSITY. The shot buffer must not fill with five copies of one
   instant — that is the failure that makes "pick your best photo" useless.
4. PLATE VOTING across reads of different lengths, where naive per-position
   voting aligns the wrong characters.

Offline-runnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.bestshot import (  # noqa: E402
    KEEP_SHOTS,
    MIN_QUALITY,
    BestShotBuffer,
    score_face,
    score_plate,
)
from app.native.recognition import (  # noqa: E402
    FACE_THRESHOLD,
    Gallery,
    PlateRead,
    cosine,
    fold_plate,
    from_blob,
    normalize,
    normalize_plate,
    plate_distance,
    to_blob,
    vote_plate,
)

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


# ---------------------------------------------------------------------------
# Synthetic imagery
# ---------------------------------------------------------------------------


def sharp_crop(w: int = 120, h: int = 120, seed: int = 0) -> np.ndarray:
    """High-frequency detail — what a well-focused crop looks like to a
    Laplacian. Built from a checkerboard rather than noise so the result is
    deterministic across numpy versions."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = (((xx // 3) + (yy // 3)) % 2 * 200 + 27).astype(np.uint8)
    return np.dstack([base, base, base])


def blurred_crop(w: int = 120, h: int = 120) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(sharp_crop(w, h), (21, 21), 0)


def flat_crop(w: int = 120, h: int = 120, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def blown_crop(w: int = 120, h: int = 120) -> np.ndarray:
    """Sharp, but a third of it is pinned at 255 — an IR-flooded face."""
    c = sharp_crop(w, h)
    c[: h // 3, :, :] = 255
    return c


def quality_checks() -> None:
    print("\nCrop quality scoring")

    sharp = score_face(sharp_crop())
    blur = score_face(blurred_crop())
    check(sharp.total > blur.total, "a sharp face outscores the same face blurred")
    check(sharp.sharpness > 0.8, "checkerboard detail reads as sharp")
    check(blur.sharpness < 0.4, "a heavy Gaussian reads as soft")
    check("soft" in blur.reason, f"the blurred crop explains itself ({blur.reason!r})")

    tiny = score_face(sharp_crop(24, 24))
    check(tiny.resolution == 0.0, "a 24 px face scores zero on resolution")
    check(not tiny, "...and is therefore not usable at all (falsy Quality)")
    check("too small" in tiny.reason, f"and says why ({tiny.reason!r})")

    big = score_face(sharp_crop(160, 160))
    check(big.resolution == 1.0, "resolution saturates at the 112 px model input")

    check(score_face(np.zeros((0, 0, 3), np.uint8)).total == 0.0, "an empty crop scores 0")
    check(score_face(flat_crop()).sharpness < 0.05, "a flat grey crop has no detail")

    blown = score_face(blown_crop())
    check(blown.exposure < 0.5, "a third of the crop at 255 is scored as clipped")
    check(
        blown.exposure < sharp.exposure,
        "...and costs exposure relative to the same crop unclipped",
    )

    # Exposure must NOT just mean "dark": a correctly exposed night face is
    # dark, and preferring the blown-out frame would be exactly backwards.
    dark = (sharp_crop().astype(np.float32) * 0.35).astype(np.uint8)
    check(
        score_face(dark).exposure > 0.9,
        "a DARK but unclipped crop keeps full exposure credit (night != bad)",
    )

    print("\nFace pose from landmarks")
    # 5-point landmarks: L eye, R eye, nose, L mouth, R mouth.
    square_on = [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]]
    turned = [[40, 50], [80, 50], [44, 70], [45, 90], [75, 90]]
    tilted = [[40, 40], [80, 70], [60, 70], [45, 90], [75, 90]]

    q_square = score_face(sharp_crop(), landmarks=square_on)
    q_turned = score_face(sharp_crop(), landmarks=turned)
    q_tilted = score_face(sharp_crop(), landmarks=tilted)
    check(q_square.geometry > 0.9, "nose centred between the eyes reads as square on")
    check(q_turned.geometry < q_square.geometry, "an off-centre nose reads as turned away")
    check(q_tilted.geometry < q_square.geometry, "a sloped eye line reads as tilted")
    check(
        q_turned.geometry < q_tilted.geometry,
        "yaw is penalized harder than roll — alignment can undo a tilt, not a turn",
    )
    check(
        score_face(sharp_crop()).geometry == 0.6,
        "with NO landmarks pose is neutral, not invented",
    )
    check(
        "no landmarks" in score_face(sharp_crop(), landmarks=[[1, 1]]).reason
        or score_face(sharp_crop(), landmarks=[[1, 1]]).geometry == 0.6,
        "a short landmark list degrades to neutral rather than indexing off the end",
    )

    print("\nPlate crop scoring")
    plate_ok = score_plate(sharp_crop(200, 100))
    plate_square = score_plate(sharp_crop(100, 100))
    plate_narrow = score_plate(sharp_crop(500, 100))
    check(plate_ok.geometry > 0.9, "a 2:1 crop is the ideal plate aspect")
    check(plate_square.geometry == 0.0, "a 1:1 crop is not a plate seen face-on")
    check(plate_narrow.geometry == 0.0, "a 5:1 crop is too oblique to read")
    check("too square" in plate_square.reason, f"and says so ({plate_square.reason!r})")
    check(
        score_plate(sharp_crop(40, 20)).resolution == 0.0,
        "a 40 px-wide plate is below the OCR floor",
    )
    check(
        score_plate(sharp_crop(200, 100)).total > score_plate(blurred_crop(200, 100)).total,
        "sharpness dominates the plate score, as OCR requires",
    )


# ---------------------------------------------------------------------------
# The buffer
# ---------------------------------------------------------------------------


def buffer_checks() -> None:
    print("\nBest-shot buffer: ranking")
    buf = BestShotBuffer()
    box = (0.0, 0.0, 10.0, 10.0)

    buf.offer(tracker_id=1, kind="face", crop_bgr=blurred_crop(), box=box, frame_time=0.0)
    buf.offer(tracker_id=1, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=5.0)
    shots = buf.shots(1, "face")
    check(len(shots) == 2, "two well-separated offers are both retained")
    check(
        shots[0].quality.total > shots[1].quality.total,
        "shots() returns best-first",
    )
    check(shots[0].frame_time == 5.0, "...and the sharp one is the best")
    check(buf.best(1, "face") is shots[0], "best() agrees with shots()[0]")

    print("\nBest-shot buffer: time diversity")
    buf = BestShotBuffer(keep=5, min_gap_s=0.4)
    # Ten offers spanning 0.27 s — one continuous instant, inside a single gap
    # window. Naive top-N would keep five of these.
    for i in range(10):
        buf.offer(
            tracker_id=2, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=i * 0.03
        )
    check(
        len(buf.shots(2, "face")) == 1,
        "ten offers inside one 0.4 s window collapse to a single retained shot",
    )

    # Spread over 0.9 s the SAME ten offers are three distinct moments
    # (t=0.0, 0.4, 0.8), not one and not ten. This is the property that makes
    # the retained set worth showing an operator.
    buf = BestShotBuffer(keep=5, min_gap_s=0.4)
    for i in range(10):
        buf.offer(
            tracker_id=8, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=i * 0.1
        )
    check(
        sorted(s.frame_time for s in buf.shots(8, "face")) == [0.0, 0.4, 0.8],
        "the same ten offers over 0.9 s are kept as exactly the distinct moments",
    )

    buf = BestShotBuffer(keep=5, min_gap_s=0.4)
    for i in range(10):
        buf.offer(
            tracker_id=3, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=i * 1.0
        )
    check(
        len(buf.shots(3, "face")) == KEEP_SHOTS,
        "ten offers a second apart fill the buffer to its cap",
    )

    print("\nBest-shot buffer: a near-duplicate REPLACES rather than crowds")
    buf = BestShotBuffer(min_gap_s=1.0)
    buf.offer(tracker_id=4, kind="face", crop_bgr=blurred_crop(), box=box, frame_time=10.0)
    buf.offer(tracker_id=4, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=10.2)
    shots = buf.shots(4, "face")
    check(len(shots) == 1, "the near-duplicate did not take a second slot")
    check(shots[0].frame_time == 10.2, "...and the better of the two won the slot")

    buf.offer(tracker_id=4, kind="face", crop_bgr=blurred_crop(), box=box, frame_time=10.3)
    check(
        buf.shots(4, "face")[0].frame_time == 10.2,
        "a WORSE near-duplicate does not displace the shot already held",
    )

    print("\nBest-shot buffer: rejection and isolation")
    buf = BestShotBuffer()
    check(
        buf.offer(tracker_id=5, kind="face", crop_bgr=sharp_crop(20, 20), box=box, frame_time=0.0)
        is None,
        f"a crop below MIN_QUALITY ({MIN_QUALITY}) is refused outright",
    )
    check(buf.best(5, "face") is None, "...and leaves no slot behind")

    buf.offer(tracker_id=6, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=0.0)
    buf.offer(tracker_id=6, kind="plate", crop_bgr=sharp_crop(200, 100), box=box, frame_time=0.0)
    check(
        len(buf.shots(6, "face")) == 1 and len(buf.shots(6, "plate")) == 1,
        "faces and plates on ONE track are kept in separate slots",
    )

    print("\nBest-shot buffer: the crop is copied, not referenced")
    live = sharp_crop()
    buf = BestShotBuffer()
    shot = buf.offer(tracker_id=7, kind="face", crop_bgr=live, box=box, frame_time=0.0)
    live[:] = 0  # the ingest loop reuses its frame buffer
    check(
        shot is not None and shot.crop.any(),
        "overwriting the source frame does not blank the retained shot",
    )

    print("\nBest-shot buffer: lifetime")
    buf = BestShotBuffer()
    for tid in (10, 11, 12):
        buf.offer(tracker_id=tid, kind="face", crop_bgr=sharp_crop(), box=box, frame_time=0.0)
    check(buf.tracks() == [10, 11, 12], "tracks() lists what is held")
    buf.forget(11)
    check(buf.tracks() == [10, 12], "forget() drops one track")
    buf.forget_missing([12])
    check(buf.tracks() == [12], "forget_missing() drops everything the tracker retired")
    buf.forget_all()
    check(len(buf) == 0 and buf.tracks() == [], "forget_all() empties the buffer")


# ---------------------------------------------------------------------------
# Embedding plumbing + gallery
# ---------------------------------------------------------------------------


def unit(*values: float) -> np.ndarray:
    return normalize(np.array(values, dtype=np.float32))


def near(a: np.ndarray, nudge: float, seed: int = 1) -> np.ndarray:
    """A vector a small angle away from `a` — a second photo of one person."""
    rng = np.random.default_rng(seed)
    return normalize(a + rng.normal(0, nudge, size=a.shape).astype(np.float32))


def embedding_checks() -> None:
    print("\nEmbedding serialization")
    v = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    blob = to_blob(v)
    back = from_blob(blob, 4)
    check(len(blob) == 16, "four float32s serialize to 16 bytes, no header")
    check(back is not None and abs(float(np.linalg.norm(back)) - 1.0) < 1e-6,
          "what comes back out is UNIT length — to_blob normalizes on the way in")
    check(abs(cosine(back, normalize(v)) - 1.0) < 1e-6, "and round-trips to the same direction")

    check(from_blob(blob, 5) is None, "a dim that disagrees with the blob is refused, not read")
    check(from_blob(b"", 4) is None, "an empty blob is refused")
    check(from_blob(None, 4) is None, "a NULL embedding is refused")
    check(from_blob(blob, 0) is None, "dim 0 is refused")

    check(float(np.linalg.norm(normalize(np.zeros(4, np.float32)))) == 0.0,
          "normalizing a zero vector gives zero, not NaN")
    check(cosine(unit(1, 0, 0, 0), unit(1, 0, 0, 0)) <= 1.0,
          "a vector against itself is clipped to 1.0, never 1.0000001")
    check(cosine(unit(1, 0, 0, 0), np.zeros(3, np.float32)) == 0.0,
          "mismatched shapes score 0 instead of raising")


def gallery_checks() -> None:
    print("\nGallery: matching")
    adam = unit(1, 0, 0, 0)
    sam = unit(0, 1, 0, 0)

    prof_rows = [
        {"id": 1, "kind": "person", "name": "Adam", "enabled": 1, "threshold": None},
        {"id": 2, "kind": "person", "name": "Sam", "enabled": 1, "threshold": None},
    ]
    sample_rows = [
        {"id": 1, "profile_id": 1, "embedding": to_blob(adam), "dim": 4,
         "model_key": "m1", "plate": ""},
        {"id": 2, "profile_id": 2, "embedding": to_blob(sam), "dim": 4,
         "model_key": "m1", "plate": ""},
    ]
    g = Gallery.build(prof_rows, sample_rows, model_key="m1")
    check(len(g) == 2, "two enrolled people build a gallery of two")

    m = g.match_face(near(adam, 0.05))
    check(m.matched and m.name == "Adam", "a near-identical vector matches its person")
    check(m.score > FACE_THRESHOLD, "...above threshold")

    m = g.match_face(unit(0, 0, 1, 0))
    check(not m.matched, "an orthogonal vector matches nobody")
    check(m.reason == "below threshold", f"and says why ({m.reason!r})")

    print("\nGallery: max-over-samples, not centroid")
    # Two genuinely different poses of one person. Their MEAN sits between and
    # is close to neither; a centroid gallery would fail to match either pose.
    pose_a = unit(1, 0, 0, 0)
    pose_b = unit(0, 0, 1, 0)
    g2 = Gallery.build(
        [{"id": 1, "kind": "person", "name": "Adam", "enabled": 1, "threshold": None}],
        [
            {"id": 1, "profile_id": 1, "embedding": to_blob(pose_a), "dim": 4,
             "model_key": "m1", "plate": ""},
            {"id": 2, "profile_id": 1, "embedding": to_blob(pose_b), "dim": 4,
             "model_key": "m1", "plate": ""},
        ],
        model_key="m1",
    )
    centroid = normalize(pose_a + pose_b)
    check(
        g2.match_face(pose_a).matched and g2.match_face(pose_b).matched,
        "BOTH enrolled poses match — averaging them would have matched neither",
    )
    check(
        cosine(centroid, pose_a) < 0.72,
        "...and the centroid really is far from each pose (this is not a no-op test)",
    )

    print("\nGallery: the margin test")
    twin_a = unit(1, 0, 0, 0)
    twin_b = near(twin_a, 0.02, seed=7)  # two people who look alike
    g3 = Gallery.build(
        [
            {"id": 1, "kind": "person", "name": "Twin A", "enabled": 1, "threshold": None},
            {"id": 2, "kind": "person", "name": "Twin B", "enabled": 1, "threshold": None},
        ],
        [
            {"id": 1, "profile_id": 1, "embedding": to_blob(twin_a), "dim": 4,
             "model_key": "m1", "plate": ""},
            {"id": 2, "profile_id": 2, "embedding": to_blob(twin_b), "dim": 4,
             "model_key": "m1", "plate": ""},
        ],
        model_key="m1",
    )
    m = g3.match_face(near(twin_a, 0.01, seed=9))
    check(
        not m.matched,
        "a probe that clears threshold for TWO similar people resolves to unknown",
    )
    check("too close to call" in m.reason, f"and names the ambiguity ({m.reason!r})")
    check(m.score > FACE_THRESHOLD, "...even though the top score was well above threshold")

    print("\nGallery: model mixing is refused")
    g4 = Gallery.build(prof_rows, sample_rows, model_key="m2")
    check(len(g4) == 0, "samples embedded with m1 are dropped from an m2 gallery")
    check(
        not g4.match_face(adam).matched and g4.match_face(adam).reason == "no enrolled faces",
        "...so the gallery reports 'no enrolled faces' rather than scoring noise",
    )

    print("\nGallery: robustness")
    g5 = Gallery.build(
        prof_rows,
        sample_rows + [
            {"id": 3, "profile_id": 1, "embedding": b"\x00\x01", "dim": 4,
             "model_key": "m1", "plate": ""},
            {"id": 4, "profile_id": 99, "embedding": to_blob(adam), "dim": 4,
             "model_key": "m1", "plate": ""},
        ],
        model_key="m1",
    )
    check(len(g5) == 2, "a corrupt blob and an orphan sample do not take the gallery down")
    check(g5.match_face(near(adam, 0.05)).matched, "...and matching still works")

    g6 = Gallery.build(
        [{"id": 1, "kind": "person", "name": "Adam", "enabled": 0, "threshold": None}],
        sample_rows[:1],
        model_key="m1",
    )
    check(len(g6) == 0, "a disabled profile is excluded from the gallery entirely")

    print("\nGallery: per-profile threshold override")
    # A probe at a KNOWN angle from the enrolled vector, so the test turns on
    # the threshold and not on the draw of a random nudge: adam + 0.5*orthogonal
    # normalizes to cos = 1/sqrt(1.25) ~= 0.894.
    probe = normalize(adam + 0.5 * unit(0, 0, 1, 0))
    check(
        abs(cosine(probe, adam) - 0.894) < 0.01,
        "the probe sits at a known 0.89 similarity (this test is not a tautology)",
    )
    g_default = Gallery.build(
        [{"id": 1, "kind": "person", "name": "Adam", "enabled": 1, "threshold": None}],
        sample_rows[:1],
        model_key="m1",
    )
    check(
        g_default.match_face(probe).matched,
        "...which the DEFAULT threshold accepts",
    )
    g7 = Gallery.build(
        [{"id": 1, "kind": "person", "name": "Adam", "enabled": 1, "threshold": 0.95}],
        sample_rows[:1],
        model_key="m1",
    )
    check(
        not g7.match_face(probe).matched,
        "...and a profile with a stricter threshold rejects the very same probe",
    )


# ---------------------------------------------------------------------------
# Plates
# ---------------------------------------------------------------------------


def plate_checks() -> None:
    print("\nPlate normalization")
    check(normalize_plate(" 7abc-123 ") == "7ABC123", "case, spaces and dashes are stripped")
    check(normalize_plate("") == "", "an empty read normalizes to empty")
    check(fold_plate("AB0123") == fold_plate("ABO123"), "O and 0 fold together for comparison")
    check(fold_plate("1I5S") == "1155", "I/1 and S/5 fold too")
    check(
        normalize_plate("ABO123") == "ABO123",
        "...but normalize_plate NEVER rewrites the glyph — the raw read is preserved",
    )

    check(plate_distance("ABC123", "ABC123") == 0, "identical plates are 0 apart")
    check(plate_distance("ABC123", "ABC124") == 1, "one substitution is 1")
    check(plate_distance("ABC123", "ABC12") == 1, "one deletion is also 1")
    check(plate_distance("ABC123", "XYZ789") == 6, "a different plate is far")
    check(plate_distance("ABO123", "AB0123") == 0, "folding makes an O/0 misread free")
    check(plate_distance("", "ABC") == 3, "an empty read is len() away, not an error")

    print("\nPlate voting across frames")
    # Three reads, each wrong in a different position. No single frame is
    # right; the vote recovers ABC123.
    reads = [
        PlateRead("ABC173", confidence=0.9, quality=0.8),
        PlateRead("ABC123", confidence=0.8, quality=0.9),
        PlateRead("A8C123", confidence=0.9, quality=0.9),
    ]
    v = vote_plate(reads)
    check(v is not None and v.text == "ABC123", f"per-character vote recovers the true plate")
    check(v.reads == 3, "all three reads were in the winning cohort")
    check(len(v.agreement) == 6, "agreement is reported per character")
    check(
        min(v.agreement) < 1.0 and max(v.agreement) == 1.0,
        "positions that disagreed are visibly weaker than the ones that did not",
    )
    check(
        abs(v.confidence - min(v.agreement)) < 1e-9,
        "overall confidence is the WEAKEST character, not the mean",
    )

    print("\nPlate voting: length is decided before position")
    # A truncated read must not drag every later position out of alignment.
    v = vote_plate([
        PlateRead("ABC123", confidence=0.9, quality=0.9),
        PlateRead("ABC123", confidence=0.9, quality=0.9),
        PlateRead("BC123", confidence=0.9, quality=0.9),   # dropped leading char
    ])
    check(v is not None and v.text == "ABC123", "a short read loses the length vote")
    check(v.reads == 2, "...and is excluded from the cohort rather than padded")

    print("\nPlate voting: weighting")
    # One high-quality read against two poor ones that agree with each other.
    v = vote_plate([
        PlateRead("ABC123", confidence=0.99, quality=0.99),
        PlateRead("ABC124", confidence=0.30, quality=0.30),
        PlateRead("ABC124", confidence=0.30, quality=0.30),
    ])
    check(
        v is not None and v.text == "ABC123",
        "one sharp confident read outweighs two weak ones that agree",
    )
    check(
        vote_plate([PlateRead("ABC123", confidence=0.0, quality=1.0)]) is None,
        "a zero-confidence read carries no weight and yields no vote",
    )
    check(vote_plate([]) is None, "no reads yields no vote")
    check(vote_plate([PlateRead("", 1.0, 1.0)]) is None, "an empty read yields no vote")

    print("\nGallery: plate matching")
    g = Gallery.build(
        [{"id": 1, "kind": "vehicle", "name": "Adam's truck", "enabled": 1, "threshold": None}],
        [{"id": 1, "profile_id": 1, "embedding": None, "dim": 0,
          "model_key": "", "plate": "ABC123"}],
        model_key="m1",
    )
    check(g.match_plate("ABC123").matched, "an exact plate matches")
    check(g.match_plate("ABC123").score == 1.0, "...at a similarity of 1.0")
    check(g.match_plate("ABC124").matched, "a one-character misread still matches")
    check(g.match_plate("ABO123").matched, "an O/0 misread matches via folding")
    check(not g.match_plate("XYZ789").matched, "a different plate does not match")
    check(not g.match_plate("").matched, "an empty read matches nothing")
    check(
        g.match_plate("AB9924").reason.endswith("edits away"),
        "a near miss explains how far off it was",
    )
    check(
        g.counts() == {"vehicle": 1},
        "counts() reports the gallery by kind",
    )
    # A vehicle profile must not be reachable through the face matcher, and
    # vice versa: they are different spaces with different thresholds.
    check(
        not g.match_face(unit(1, 0, 0, 0)).matched,
        "a vehicle profile is invisible to face matching",
    )


def main() -> int:
    quality_checks()
    buffer_checks()
    embedding_checks()
    gallery_checks()
    plate_checks()

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (best-shot selection + recognition core)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

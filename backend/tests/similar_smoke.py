#!/usr/bin/env python3
""""Add the photos that look like this one" — suggested for faces, automatic
for plates, and the difference between those is the whole point.

THE ASYMMETRY
=============
Enrolling one shot of someone should not mean hunting their every other
sighting by eye in a list sorted by legibility. Nobody does that, so profiles
stay thin, and thin profiles are the ones that miss. So after an enrollment the
system offers the crops that look like what was just enrolled.

But the two kinds are NOT equally decidable, and treating them the same would be
a real bug:

    PLATE   "is this the same plate?" has an EXACT answer. After normalization
            and glyph folding, two reads of 7ABC123 are the same vehicle by
            definition. Acting on that without asking is safe, so the matching
            unread-plate rows are absorbed automatically.

    FACE    "is this the same face?" is a SCORE. Accept a wrong one and the
            profile matches a stranger from then on — silently, with nothing on
            screen ever pointing back at the moment it went wrong. So faces are
            OFFERED and never applied.

That is why the face suggestion threshold sits well above the MATCH threshold:
a wrong match mislabels one event and is corrected by looking at that event; a
wrong enrollment corrupts the gallery.

Offline-runnable; numpy only, no models, no network, no database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.recognition import (  # noqa: E402
    FACE_THRESHOLD, PLATE_MAX_DISTANCE, SUGGEST_FACE_COSINE, SUGGEST_LIMIT,
    cosine, normalize, plate_distance,
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


def threshold_checks() -> None:
    print("\nthe suggestion bar is HIGHER than the match bar")
    check(SUGGEST_FACE_COSINE > FACE_THRESHOLD,
          f"suggest ({SUGGEST_FACE_COSINE}) > match ({FACE_THRESHOLD}) — a wrong "
          "match mislabels one event and is corrected by looking at it; a wrong "
          "ENROLLMENT corrupts the gallery permanently and silently")
    check(SUGGEST_FACE_COSINE < 1.0,
          "but below 1.0, or nothing would ever be suggested and the feature "
          "would be decoration")
    check(0 < SUGGEST_LIMIT <= 50,
          "and the offer is bounded — past a screenful it stops being a review "
          "and becomes thumbnails nobody checks, which is how a stranger gets "
          "accepted")


def ranking_checks() -> None:
    """The scoring rule must be MAX-over-samples, matching the matcher."""
    print("\nsimilarity is max-over-samples, the same rule the matcher uses")
    rng = np.random.default_rng(7)

    # Two enrolled shots of one person: near-duplicates of two distinct poses.
    pose_a = normalize(rng.normal(size=128))
    pose_b = normalize(rng.normal(size=128))
    samples = [pose_a, pose_b]

    # A candidate that is a near-duplicate of pose_b ONLY.
    near_b = normalize(pose_b + 0.08 * rng.normal(size=128))
    by_max = max(cosine(near_b, v) for v in samples)
    centroid = normalize(pose_a + pose_b)
    by_centroid = cosine(near_b, centroid)
    check(by_max > by_centroid,
          f"max ({by_max:.3f}) beats centroid ({by_centroid:.3f}) — averaging "
          "unit vectors from different poses lands near NEITHER, so a centroid "
          "would miss the very shots this feature exists to find")
    check(by_max >= SUGGEST_FACE_COSINE,
          "a genuine near-duplicate clears the suggestion bar")

    stranger = normalize(rng.normal(size=128))
    check(max(cosine(stranger, v) for v in samples) < SUGGEST_FACE_COSINE,
          "an unrelated face does NOT — which is the failure that matters, "
          "because accepting it is irreversible in effect")


def plate_checks() -> None:
    print("\nplates: 'alike' is exact, so it can be acted on without asking")
    check(plate_distance("7ABC123", "7ABC123") == 0, "an identical read matches")
    check(plate_distance("7ABC123", "7A8C123") <= PLATE_MAX_DISTANCE,
          "and so does the single-character misread that survives voting "
          "(B/8 is a glyph confusion the folding covers)")
    check(plate_distance("7ABC123", "9XYZ789") > PLATE_MAX_DISTANCE,
          "a genuinely different plate does not")
    check(plate_distance("7ABC123", "7ABC12") <= PLATE_MAX_DISTANCE,
          "a dropped character is within range — edit distance, not positional "
          "compare, which would call this a total mismatch")
    check(plate_distance("AB1234", "CD5678") > PLATE_MAX_DISTANCE,
          "two unrelated plates of equal length stay apart")


def absorb_semantics_checks() -> None:
    """What absorbing a plate candidate must and must not do."""
    print("\nabsorbing a plate REMOVES the duplicate rather than enrolling it")
    # This is a semantic assertion about the design, pinned in source so the
    # reasoning survives: a second identical plate STRING is not a second piece
    # of evidence. Faces gain from more samples because each is a different
    # pose in a continuous space; a plate sample is an exact string, and two
    # copies of it match exactly what one copy matches.
    import ast
    import inspect
    import textwrap

    from app.routers import profiles as router

    src = inspect.getsource(router._absorb_matching_plates)
    tree = ast.parse(textwrap.dedent(src))
    body = tree.body[0].body
    code = "\n".join(
        ast.unparse(n) for n in body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    check("DELETE FROM recognition_candidates" in code,
          "it deletes the duplicate candidate rows")
    check("INSERT INTO profile_samples" not in code,
          "and does NOT enroll them — a second copy of the same string matches "
          "exactly what one copy matches, so it is noise, not evidence")
    check("_unlink" in code,
          "and removes their crops, since the rows are going away for good")

    # And the face path must NOT apply anything.
    sim = inspect.getsource(router.similar_to_profile)
    tree = ast.parse(textwrap.dedent(sim))
    body = tree.body[0].body
    code = "\n".join(
        ast.unparse(n) for n in body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    print("\nfaces are OFFERED, never applied")
    check("INSERT" not in code and "DELETE" not in code,
          "the similar-faces endpoint writes NOTHING — accepting a suggestion "
          "is the ordinary enroll call, so the confirmation is a real step and "
          "not a dialog someone dismisses without seeing what it did")


def main() -> int:
    threshold_checks()
    ranking_checks()
    plate_checks()
    absorb_semantics_checks()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (similar-shot suggestions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

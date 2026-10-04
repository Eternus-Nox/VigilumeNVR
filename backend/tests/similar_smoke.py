#!/usr/bin/env python3
""""Add the photos that look like this one" — suggested, never applied.

Enrolling one shot of someone should not mean hunting their every other
sighting by eye in a list sorted by legibility. Nobody does that, so profiles
stay thin, and thin profiles are the ones that miss. So after an enrollment the
system offers the crops that look like what was just enrolled.

"Is this the same face?" is a SCORE. Accept a wrong one and the profile matches
a stranger from then on — silently, with nothing on screen ever pointing back at
the moment it went wrong. So faces are OFFERED and never applied.

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
    FACE_THRESHOLD, SUGGEST_FACE_COSINE, SUGGEST_LIMIT, cosine, normalize,
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


def apply_semantics_checks() -> None:
    import ast
    import inspect
    import textwrap

    from app.routers import profiles as router

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
    apply_semantics_checks()
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

"""Recognition core — the gallery and the matcher.

NO MODEL RUNS HERE, deliberately, for the same reason bestshot.py runs none:
everything in this module is arithmetic over vectors that some model
produced elsewhere. That keeps the part of recognition most likely to be
WRONG — thresholds, tie-breaks, how several disagreeing reads are reconciled —
testable against fixtures, with no weights downloaded and no GPU.

WHAT A PROFILE IS
=================
A profile is a named identity with several enrolled SAMPLES:

    person  -> N face embeddings, each a unit vector from one enrolled crop

(Vehicle profiles enrolled by plate are still in older databases; licence
plate reading was removed, so they are never matched and the API hides them.)

Recognition is a nearest-neighbour lookup, not a classifier. This matters, and
it is the thing most people expect to be otherwise: there is no training step,
no weights that change when you enroll, and no retraining when you add the
eleventh person. Adding a sample appends a row; deleting one removes a row; the
model on disk never changes. Consequences worth stating plainly, because they
drive the whole UX:

  * Enrollment is INSTANT and reversible. A bad reference image is one DELETE
    away from gone, with no residue in a trained artifact.
  * Accuracy scales with sample DIVERSITY, not sample count. Five shots of one
    pose are worth about one; five shots across angles, lighting and seasons
    are worth five. This is why bestshot.py enforces time diversity and why
    the app should show the operator what they are picking between.
  * Every embedding in the gallery must come from the SAME model. Vectors from
    two models are not comparable — the similarity between them is noise that
    happens to be in 0..1. `model_key` is stored per sample for exactly this
    reason, and `Gallery.build` refuses to mix.

MAXIMUM-OVER-SAMPLES, NOT CENTROID
----------------------------------
A profile's score is the best similarity over its samples, not the similarity
to their mean. Averaging unit vectors from genuinely different poses produces a
vector that is close to none of them — the classic failure where enrolling more
photos makes recognition worse. Max-over-samples degrades gracefully instead:
an unhelpful sample is simply never the argmax, so a bad enrollment costs
nothing beyond the row it occupies.

MARGIN, NOT JUST THRESHOLD
--------------------------
A match is accepted only when the best profile clears its threshold AND beats
the runner-up by `MIN_MARGIN`. Two enrolled siblings sit close together in
embedding space; without a margin test the system will confidently pick one of
them at random and be wrong half the time. With it, an ambiguous read reports
UNKNOWN, which is both true and actionable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Cosine-similarity floor for calling two face embeddings the same person.
#: CALIBRATED PER MODEL — every embedding network has its own operating point,
#: so this default belongs to whichever model the permissive tier pins, and a
#: profile can override it (profiles.threshold) when one person keeps
#: false-matching. Erring high is the right direction: a missed recognition
#: shows up as "unknown" and can be enrolled, while a false one silently
#: attaches a stranger to someone's name.
FACE_THRESHOLD = 0.38

#: How far ahead of the runner-up the winner must be. See MARGIN above.
MIN_MARGIN = 0.06

#: Similarity at which an UNMATCHED crop is offered as "this looks like the
#: person you just enrolled".
#:
#: DELIBERATELY WELL ABOVE ``FACE_THRESHOLD``, and the asymmetry is the point.
#: A match decides what one event is called and is corrected by looking at that
#: event. An enrollment suggestion decides what the GALLERY contains: accept a
#: wrong one and the profile matches a stranger from then on, silently, with
#: nothing on screen ever pointing back at the moment it went wrong. So the bar
#: for "offer this" is much higher than the bar for "call this a match", and
#: the offer is still only an offer — see the suggestion endpoint.
SUGGEST_FACE_COSINE = 0.52

#: Most similar crops offered at once. Past this it stops being a review and
#: becomes a page of thumbnails nobody looks at properly, which is the failure
#: mode that gets a stranger accepted into a profile.
SUGGEST_LIMIT = 24

# ---------------------------------------------------------------------------
# Embedding plumbing
# ---------------------------------------------------------------------------


def to_blob(vec: np.ndarray) -> bytes:
    """Serialize an embedding for `profile_samples.embedding`.

    float32 little-endian, no header: `dim` is a column, so the blob stays the
    smallest thing that round-trips. Always normalized first, so every vector
    in the database is unit length and a dot product IS the cosine.
    """
    v = normalize(np.asarray(vec, dtype=np.float32))
    return v.astype("<f4").tobytes()


def from_blob(blob: Optional[bytes], dim: int) -> Optional[np.ndarray]:
    """Inverse of `to_blob`, tolerant of a short or corrupt row.

    Returns None rather than raising: one bad sample row must not take the
    whole gallery — and therefore every camera's recognition — down with it.
    """
    if not blob or dim <= 0:
        return None
    expected = dim * 4
    if len(blob) != expected:
        log.warning("embedding blob is %d bytes, expected %d — skipping", len(blob), expected)
        return None
    return np.frombuffer(blob, dtype="<f4").astype(np.float32)


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize, mapping a zero vector to zero rather than to NaN."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors that are already unit length.

    Clipped to [-1, 1] because float32 accumulation genuinely produces 1.0000001
    for a vector against itself, and a similarity above 1 breaks every downstream
    comparison that assumes the range.
    """
    if a is None or b is None or a.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# ---------------------------------------------------------------------------
# The gallery
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    sample_id: int
    profile_id: int
    vector: Optional[np.ndarray] = None


#: What a sighting of one profile does to notifications. Closed on purpose:
#: an unknown string must read as "default" rather than as a fourth policy
#: nobody implemented, so `normalize_alert_mode` funnels everything through it.
ALERT_MODES = ("default", "mute", "alert")


def normalize_alert_mode(value: Any) -> str:
    """A stored alert_mode as one of ALERT_MODES.

    Anything unrecognised — None, a typo, a value written by a newer build and
    read back by an older one — becomes "default". That is the only safe
    fallback in both directions: an unknown mode must never silently MUTE a
    profile (a missed alert) and must never silently escalate one either.
    """
    text = str(value or "").strip().lower()
    return text if text in ALERT_MODES else "default"


@dataclass
class Profile:
    profile_id: int
    kind: str
    name: str
    enabled: bool = True
    threshold: Optional[float] = None
    #: See ALERT_MODES. Carried on the profile so it can ride the Match out to
    #: the events pipeline, rather than costing a database lookup on the
    #: notification path for every sighting.
    alert_mode: str = "default"
    samples: list[Sample] = field(default_factory=list)


@dataclass(frozen=True)
class Match:
    """The outcome of one lookup. `profile_id is None` means UNKNOWN."""

    profile_id: Optional[int]
    name: str
    score: float
    #: Gap to the runner-up. Small means "these two look alike", which is worth
    #: surfacing even on an accepted match.
    margin: float
    #: Why an unknown is unknown — "below threshold" vs "too close to call"
    #: are different problems with different fixes.
    reason: str
    #: The matched profile's notification policy (ALERT_MODES). Always
    #: "default" on an unknown, which is what makes the gate's job simple:
    #: an unmatched sighting can never be muted by somebody else's setting.
    alert_mode: str = "default"

    @property
    def matched(self) -> bool:
        return self.profile_id is not None


class Gallery:
    """An immutable snapshot of the enrolled profiles, built per reload.

    Rebuilt wholesale when profiles change rather than mutated in place: a
    gallery is small (hundreds of vectors), the rebuild is microseconds, and a
    snapshot means the matcher never sees a half-applied enrollment.
    """

    def __init__(
        self,
        profiles: Sequence[Profile],
        *,
        model_key: str = "",
        face_threshold: float = FACE_THRESHOLD,
        min_margin: float = MIN_MARGIN,
    ) -> None:
        self.model_key = model_key
        self._face_threshold = float(face_threshold)
        self._min_margin = float(min_margin)
        self._profiles = [p for p in profiles if p.enabled and p.samples]

    # -- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        profile_rows: Iterable[dict[str, Any]],
        sample_rows: Iterable[dict[str, Any]],
        *,
        model_key: str,
        **kw: Any,
    ) -> "Gallery":
        """Assemble from raw DB rows, dropping what cannot be compared.

        Samples whose `model_key` differs from the active model are SKIPPED,
        not silently mixed in: their vectors live in a different space and
        their similarities would be meaningless numbers in a plausible range.
        A profile left with no usable sample simply stops matching, which is
        the honest outcome and is visible in the API as a sample count of 0.
        """
        by_id: dict[int, Profile] = {}
        for r in profile_rows:
            by_id[int(r["id"])] = Profile(
                profile_id=int(r["id"]),
                kind=str(r["kind"]),
                name=str(r["name"]),
                enabled=bool(r.get("enabled", 1)),
                threshold=r.get("threshold"),
                alert_mode=normalize_alert_mode(r.get("alert_mode")),
            )

        skipped = 0
        for r in sample_rows:
            prof = by_id.get(int(r["profile_id"]))
            if prof is None:
                continue
            if r.get("embedding") is None:
                continue  # a plate sample from before plates were removed
            if str(r.get("model_key") or "") != model_key:
                skipped += 1
                continue
            vector = from_blob(r["embedding"], int(r.get("dim") or 0))
            if vector is None:
                continue
            prof.samples.append(
                Sample(
                    sample_id=int(r["id"]),
                    profile_id=prof.profile_id,
                    vector=vector,
                )
            )
        if skipped:
            log.warning(
                "gallery: skipped %d sample(s) embedded with a different model "
                "(active=%r) — re-enroll them to restore those profiles",
                skipped,
                model_key,
            )
        return cls(list(by_id.values()), model_key=model_key, **kw)

    # -- lookup ---------------------------------------------------------

    def match_face(self, vector: np.ndarray) -> Match:
        """Nearest enrolled person, subject to threshold and margin."""
        v = normalize(vector)
        if v.size == 0 or not np.any(v):
            return Match(None, "", 0.0, 0.0, "empty embedding")

        scored: list[tuple[float, Profile]] = []
        for prof in self._profiles:
            if prof.kind != "person":
                continue
            # MAX over samples — see the module docstring.
            best = max(
                (cosine(v, s.vector) for s in prof.samples if s.vector is not None),
                default=None,
            )
            if best is not None:
                scored.append((best, prof))
        if not scored:
            return Match(None, "", 0.0, 0.0, "no enrolled faces")

        scored.sort(key=lambda t: t[0], reverse=True)
        top_score, top = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_score - runner
        threshold = top.threshold if top.threshold is not None else self._face_threshold

        if top_score < threshold:
            return Match(None, "", top_score, margin, "below threshold")
        if len(scored) > 1 and margin < self._min_margin:
            return Match(
                None, "", top_score, margin,
                f"too close to call ({top.name} vs {scored[1][1].name})",
            )
        return Match(top.profile_id, top.name, top_score, margin, "matched", top.alert_mode)

    # -- introspection --------------------------------------------------

    @property
    def profiles(self) -> list[Profile]:
        return list(self._profiles)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for p in self._profiles:
            out[p.kind] += 1
        return dict(out)

    def __len__(self) -> int:
        return len(self._profiles)

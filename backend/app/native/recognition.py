"""Recognition core — the gallery, the matcher, and multi-frame plate voting.

NO MODEL RUNS HERE, deliberately, for the same reason bestshot.py runs none:
everything in this module is arithmetic over vectors and strings that some
model produced elsewhere. That keeps the part of recognition most likely to be
WRONG — thresholds, tie-breaks, how several disagreeing reads are reconciled —
testable against fixtures, with no weights downloaded and no GPU.

WHAT A PROFILE IS
=================
A profile is a named identity with several enrolled SAMPLES:

    person  -> N face embeddings, each a unit vector from one enrolled crop
    vehicle -> N plate strings

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
import re
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

#: Plate strings are compared after normalization; this is the edit distance
#: within which two normalized plates are called the same vehicle. 1 covers the
#: single-character misread that survives voting; 2 starts matching genuinely
#: different plates in a small gallery.
PLATE_MAX_DISTANCE = 1

#: Characters an OCR confuses by glyph shape. Folded ONLY for comparison — the
#: read is always stored raw, because showing an operator a plate we quietly
#: rewrote is how you lose their trust in the whole feature.
_PLATE_CONFUSIONS = str.maketrans({"O": "0", "I": "1", "Q": "0", "S": "5", "Z": "2", "B": "8"})

_PLATE_STRIP = re.compile(r"[^A-Z0-9]")


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
# Plate strings
# ---------------------------------------------------------------------------


def normalize_plate(text: str) -> str:
    """Uppercase, strip non-alphanumerics. The form stored and indexed."""
    return _PLATE_STRIP.sub("", (text or "").upper())


def fold_plate(text: str) -> str:
    """Normalized AND glyph-folded, for comparison only (see _PLATE_CONFUSIONS)."""
    return normalize_plate(text).translate(_PLATE_CONFUSIONS)


def plate_distance(a: str, b: str) -> int:
    """Levenshtein distance between two folded plates.

    Full edit distance rather than positional mismatch count, because the
    common OCR errors are an inserted or dropped character as often as a
    substituted one, and a positional compare calls "ABC123" vs "AB C123" a
    total mismatch.
    """
    s, t = fold_plate(a), fold_plate(b)
    if s == t:
        return 0
    if not s:
        return len(t)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s, 1):
        cur = [i]
        for j, ct in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cs != ct)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class PlateRead:
    """One OCR pass over one crop."""

    text: str
    #: The OCR's own confidence in this string, 0..1.
    confidence: float = 1.0
    #: bestshot.Quality.total for the crop it was read from, 0..1.
    quality: float = 1.0

    @property
    def weight(self) -> float:
        """How much this read counts in a vote.

        Confidence and quality multiply because they fail independently: a
        confident read of a smeared crop and a hesitant read of a sharp one are
        both weak evidence, and neither should outvote a read that is good on
        both axes.
        """
        return max(0.0, float(self.confidence)) * max(0.0, float(self.quality))


@dataclass(frozen=True)
class PlateVote:
    text: str
    confidence: float
    reads: int
    #: Per-position agreement, so the UI can grey out the character the frames
    #: disagreed on instead of presenting a uniform-looking string.
    agreement: tuple[float, ...]


def vote_plate(reads: Sequence[PlateRead]) -> Optional[PlateVote]:
    """Reconcile several OCR reads of the same plate into one answer.

    THIS IS THE "several small models looking for the same indicator" idea,
    done where it actually pays. Running N different OCR networks over one
    frame mostly buys N correlated errors — they all fail on the same motion
    blur. Running ONE OCR over N frames of the same plate, taken from
    different instants by bestshot.BestShotBuffer, gives genuinely independent
    errors, and a per-character weighted vote then recovers the true string
    from reads where no single frame got it entirely right.

    Length is decided first, by weight, because voting per position across
    strings of different lengths aligns the wrong characters. Reads of the
    losing length are discarded rather than padded: a 6-character read of a
    7-character plate has one character missing SOMEWHERE, and guessing where
    corrupts every position after it.
    """
    usable = [r for r in reads if normalize_plate(r.text) and r.weight > 0]
    if not usable:
        return None

    by_length: dict[int, float] = defaultdict(float)
    for r in usable:
        by_length[len(normalize_plate(r.text))] += r.weight
    best_len = max(by_length, key=lambda k: (by_length[k], k))

    cohort = [r for r in usable if len(normalize_plate(r.text)) == best_len]
    total_weight = sum(r.weight for r in cohort)
    if total_weight <= 0:
        return None

    chars: list[str] = []
    agreement: list[float] = []
    for pos in range(best_len):
        tally: dict[str, float] = defaultdict(float)
        for r in cohort:
            tally[normalize_plate(r.text)[pos]] += r.weight
        winner = max(tally, key=lambda c: (tally[c], c))
        chars.append(winner)
        agreement.append(tally[winner] / total_weight)

    text = "".join(chars)
    # Overall confidence is the WEAKEST position, not the mean: a plate with
    # one coin-flip character is one wrong character, and averaging that away
    # would report 0.9 on a string we cannot actually stand behind.
    confidence = min(agreement) if agreement else 0.0
    return PlateVote(
        text=text,
        confidence=float(confidence),
        reads=len(cohort),
        agreement=tuple(agreement),
    )


# ---------------------------------------------------------------------------
# The gallery
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    sample_id: int
    profile_id: int
    vector: Optional[np.ndarray] = None
    plate: str = ""


@dataclass
class Profile:
    profile_id: int
    kind: str
    name: str
    enabled: bool = True
    threshold: Optional[float] = None
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
            )

        skipped = 0
        for r in sample_rows:
            prof = by_id.get(int(r["profile_id"]))
            if prof is None:
                continue
            plate = normalize_plate(str(r.get("plate") or ""))
            vector = None
            if r.get("embedding") is not None:
                if str(r.get("model_key") or "") != model_key:
                    skipped += 1
                    continue
                vector = from_blob(r["embedding"], int(r.get("dim") or 0))
                if vector is None:
                    continue
            if vector is None and not plate:
                continue
            prof.samples.append(
                Sample(
                    sample_id=int(r["id"]),
                    profile_id=prof.profile_id,
                    vector=vector,
                    plate=plate,
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
        return Match(top.profile_id, top.name, top_score, margin, "matched")

    def match_plate(self, text: str) -> Match:
        """Nearest enrolled vehicle by folded edit distance."""
        plate = normalize_plate(text)
        if not plate:
            return Match(None, "", 0.0, 0.0, "empty read")

        best: Optional[tuple[int, Profile]] = None
        for prof in self._profiles:
            if prof.kind != "vehicle":
                continue
            for s in prof.samples:
                if not s.plate:
                    continue
                d = plate_distance(plate, s.plate)
                if best is None or d < best[0]:
                    best = (d, prof)
        if best is None:
            return Match(None, "", 0.0, 0.0, "no enrolled plates")

        distance, prof = best
        # Score reported as a similarity so callers can treat faces and plates
        # uniformly: exact = 1.0, one edit on a 7-char plate ~= 0.86.
        score = max(0.0, 1.0 - distance / max(len(plate), 1))
        if distance > PLATE_MAX_DISTANCE:
            return Match(None, "", score, 0.0, f"nearest is {distance} edits away")
        return Match(prof.profile_id, prof.name, score, 0.0, "matched")

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

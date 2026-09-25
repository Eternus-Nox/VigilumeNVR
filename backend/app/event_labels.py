"""Which object type an event is named after, and alerts about first.

Events are one per camera: a car that pulls in and the person who gets out are
one event. When several types are in it, the event is NAMED after the most
important one — the person, not the car — and that same order decides which
type an alert leads with. Shared by the engine (naming, best-frame choice) and
the events pipeline (alert wording), so the two can never disagree.
"""
from __future__ import annotations

from typing import Sequence

_LABEL_RANK: dict[str, int] = {
    "person": 0,
    "car": 1, "truck": 1, "bus": 1, "motorcycle": 1, "motorbike": 1,
    "bicycle": 1, "van": 1,
}
_ANIMAL_RANK = 2
_OTHER_RANK = 3
_ANIMALS = frozenset({"dog", "cat", "bird", "horse", "sheep", "cow", "bear",
                      "deer", "fox", "raccoon", "squirrel", "rabbit"})


def label_rank(label: str) -> int:
    """Lower is more important. Unknown labels rank last, never first."""
    if label in _LABEL_RANK:
        return _LABEL_RANK[label]
    return _ANIMAL_RANK if label in _ANIMALS else _OTHER_RANK


def primary_label(labels: Sequence[str]) -> str:
    """The most important label, the earliest seen on a tie. '' for none."""
    best = ""
    for label in labels:
        if label and (not best or label_rank(label) < label_rank(best)):
            best = label
    return best

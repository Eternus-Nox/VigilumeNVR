"""Label tables for the D-FINE detector output spaces + the ACTIVE-model view.

Two vocabularies ship today (``LABELMAPS``):
- ``coco``   — contiguous COCO-80 (``dfine_n/s/m/l/x``); ids 0..79.
- ``obj365`` — Objects365 366-entry space (``dfine_*_obj365``); ids 0..365,
  index 0 a background ``"none"`` placeholder (see ``obj365_labels``).

COCO ids 0..79 are verified from the ustc-community/dfine-*-coco ``config.json``
``id2label`` (NOT the sparse 91-id COCO annotation space): 0=person, 1=bicycle,
2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat, 16=dog. Multi-word names use
underscores (``traffic_light``) so labels stay URL/UI-safe;
``annotate.plural_label`` renders underscores as spaces.

Active-model view (``ID_TO_LABEL``): the engine imports ``ID_TO_LABEL`` once at
module load and resolves ``class_id -> label`` via ``ID_TO_LABEL.get(...)``. It
is a LIVE view over whichever model is currently loaded: the detector calls
``set_active_labelmap(name)`` when it swaps models, so labels always track the
running model WITHOUT the engine needing to know which model is active. Object
identity is preserved (the view is mutated in place, never rebound), so the
engine's already-bound reference keeps working. Swaps are a single attribute
assignment — GIL-atomic, same eventual-consistency contract as the detector's
session swap.
"""
from __future__ import annotations

from typing import Optional

from .obj365_labels import OBJ365_LABELS

COCO_LABELS: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
)

assert len(COCO_LABELS) == 80

# COCO-80 lookups (static; used by tests + any COCO-only caller).
COCO_ID_TO_LABEL: dict[int, str] = dict(enumerate(COCO_LABELS))
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(COCO_LABELS)}

# ---------------------------------------------------------------------------
# COCO-90 -> COCO-80 (Edge TPU / TFLite detectors)
# ---------------------------------------------------------------------------
#
# Every detector shipped so far emits the CONTIGUOUS COCO-80 space, so class_id
# indexes COCO_LABELS directly. Edge-TPU-compiled SSD models (SSDLite MobileDet
# et al) do NOT: they emit the SPARSE COCO-90 annotation space, where ids run
# 0..89 with holes at the 11 category ids COCO never assigned.
#
# This matters more than it looks. ``engine`` resolves labels via
# ``ID_TO_LABEL.get(...)``, which NEVER RAISES — so feeding COCO-90 ids into the
# COCO-80 table does not fail loudly, it silently mislabels. id 16 is "bird" in
# COCO-90 but "dog" in COCO-80; id 3 is "car" vs "motorcycle". Every event would
# be wrong, per-camera detect_objects filtering would match the wrong classes,
# and nothing anywhere would error. Remap at the source.
#
# The holes are the ten "n/a" placeholder rows in Coral's own label file
# (google-coral/test_data/coco_labels.txt, 90 lines, 0-indexed) — VERIFIED
# against that file, not derived by hand. Getting this wrong is silent: an
# off-by-one in this set makes ~20 ids resolve to the neighbouring class, and
# because engine resolves via ID_TO_LABEL.get() nothing anywhere raises. Do not
# "tidy" these numbers; regenerate them from the file.
#
# Ids in a hole (and anything >= 90) map to None and MUST be dropped by the
# caller, never coerced to 0 — 0 is "person", the single worst class to invent
# on a security camera.
_COCO90_HOLES = frozenset({11, 25, 28, 29, 44, 65, 67, 68, 70, 82})


def _build_coco90_to_80() -> dict[int, int]:
    """COCO-90 id -> contiguous COCO-80 index, skipping the unassigned ids."""
    mapping: dict[int, int] = {}
    dense = 0
    for sparse in range(90):
        if sparse in _COCO90_HOLES:
            continue
        mapping[sparse] = dense
        dense += 1
    return mapping


COCO90_TO_COCO80: dict[int, int] = _build_coco90_to_80()
assert len(COCO90_TO_COCO80) == 80, "COCO-90 minus its 10 holes must yield exactly 80"

# Anchors from Coral's coco_labels.txt. These pin the ALIGNMENT, not just the
# count — a shifted hole set still yields 80 entries, so a length check alone
# would have let the original off-by-one through. Chosen around the holes and on
# the classes this product actually acts on (person/car/dog).
assert [COCO_LABELS[COCO90_TO_COCO80[i]] for i in (0, 2, 3, 5, 7, 9)] == [
    "person", "car", "motorcycle", "bus", "truck", "traffic_light"
]
assert [COCO_LABELS[COCO90_TO_COCO80[i]] for i in (12, 15, 16, 17, 18)] == [
    "stop_sign", "bird", "cat", "dog", "horse"
]
assert all(i not in COCO90_TO_COCO80 for i in _COCO90_HOLES)


def coco90_to_label(class_id: int) -> Optional[str]:
    """Label for a SPARSE COCO-90 class id, or None if it has no COCO-80
    counterpart (an unassigned id, or anything out of range).

    Returning None rather than a fallback label is deliberate: the caller must
    DROP an unmappable detection. Substituting a default would put a confident
    box on screen under a class the model never predicted.
    """
    dense = COCO90_TO_COCO80.get(class_id)
    return None if dense is None else COCO_LABELS[dense]


# ---------------------------------------------------------------------------
# Model output-space registry (name -> ordered label tuple). ``labelmap``
# fields in detector.MODELS reference these names; the API's vocabulary label
# is derived here too. Keep names stable — settings/UI surface them.
# ---------------------------------------------------------------------------
LABELMAPS: dict[str, tuple[str, ...]] = {
    "coco": COCO_LABELS,
    "obj365": OBJ365_LABELS,
}
DEFAULT_LABELMAP = "coco"

# Short machine vocabulary name per labelmap — the value the frontend reads and
# maps to a display name ("coco" -> "COCO", "objects365" -> "Objects365").
VOCABULARY_NAMES: dict[str, str] = {
    "coco": "coco",
    "obj365": "objects365",
}

# Leading ids to hide from the USER-FACING pick list: Objects365 reserves id 0
# as a background "none" placeholder (never a real detection to select), so the
# picker offers its 365 real categories. The full labelmap is still used for
# class_id -> label resolution (ID_TO_LABEL) — this only trims what's offered.
_BACKGROUND_PREFIX: dict[str, int] = {"coco": 0, "obj365": 1}


def labels_for(labelmap: str) -> tuple[str, ...]:
    """FULL ordered output space for a labelmap (indexes match model class ids;
    used by ``ID_TO_LABEL``). Raises ``KeyError`` if unknown."""
    return LABELMAPS[labelmap]


def selectable_labels(labelmap: str) -> tuple[str, ...]:
    """User-facing pick list — the real classes, minus any leading background
    placeholder (Objects365 id 0). This is what the object picker / labels API
    surface, so a user never selects the ``none`` background class."""
    return LABELMAPS[labelmap][_BACKGROUND_PREFIX.get(labelmap, 0):]


def num_classes(labelmap: str) -> int:
    """Count of user-selectable classes (80 for COCO, 365 for Objects365)."""
    return len(selectable_labels(labelmap))


def vocabulary_name(labelmap: str) -> str:
    """Short machine vocabulary name for the API (``"coco"``/``"objects365"``)."""
    return VOCABULARY_NAMES.get(labelmap, labelmap)


class _ActiveLabelMap:
    """Live ``class_id -> label`` view over the ACTIVE model's labelmap.

    Kept as a single long-lived object so the engine's import-time binding of
    ``ID_TO_LABEL`` never goes stale across model swaps. Duck-types the subset
    of ``dict`` the engine uses (``.get``) plus ``[]``/``in``/``len`` for
    convenience; out-of-range ids resolve to the default (never raise on
    ``.get``), so a stray background/high id from decode is dropped cleanly.
    """

    __slots__ = ("_name", "_map")

    def __init__(self, name: str) -> None:
        self._name = name
        self._map = LABELMAPS[name]

    def set(self, name: str) -> None:
        self._map = LABELMAPS[name]  # raises KeyError first -> _name unchanged
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get(self, class_id: int, default: Optional[str] = None) -> Optional[str]:
        m = self._map
        return m[class_id] if 0 <= class_id < len(m) else default

    def __getitem__(self, class_id: int) -> str:
        return self._map[class_id]

    def __contains__(self, class_id: int) -> bool:
        return 0 <= class_id < len(self._map)

    def __len__(self) -> int:
        return len(self._map)


# The engine imports THIS object once; the detector mutates it in place.
ID_TO_LABEL = _ActiveLabelMap(DEFAULT_LABELMAP)


def set_active_labelmap(labelmap: str) -> None:
    """Point the live ``ID_TO_LABEL`` view at ``labelmap`` (the detector calls
    this when a model becomes the loaded one). Unknown name raises ``KeyError``."""
    ID_TO_LABEL.set(labelmap)


def active_labelmap_name() -> str:
    return ID_TO_LABEL.name

"""DetectionEngine — tracked detections -> frigate-SHAPED event payloads.

FULLY IMPLEMENTED (pure logic; the detector/ingest pass calls into it).

The existing ``EventsPipeline`` consumes Frigate-shaped
``{type: new|update|end, after: {...}}`` dicts and a live in-frame count
cache. This engine synthesizes those payloads in-process from per-frame
tracked observations and feeds ``pipeline.update_count()`` — events,
enrichment, annotation, notifications, WS broadcast and the UI work
unchanged (design doc §4).

CALLER CONTRACT (the detector/ingest implementation pass)
==========================================================
Per enabled camera, an ffmpeg ingest loop feeds ONE inference worker
(latest-frame drop). For every frame that reaches inference::

    dets    = await asyncio.to_thread(detector.detect, frame, dw, dh)  # sv.Detections
    tracked = tracker.update(dets)          # trackers.ByteTrackTracker, one PER CAMERA
    obs     = observations_from_supervision(tracked)
    await engine.process(camera, frame_time, obs, frame_bgr=frame)

- ``engine.process`` MUST be awaited on the app event loop (it awaits
  ``pipeline.handle_event``); never call it from a bare thread.
- Call it on EVERY processed frame, including frames with zero detections —
  absence of a label is what ends events. (A 2 s housekeeping task also ends
  events on wall-clock silence, so a dead ingest can't wedge events open.)
- ``frame_bgr`` ownership passes to the engine (it keeps the reference as
  the camera's latest frame); pass a fresh buffer per frame.
- tracker_id values start at 0 — never treated as falsy anywhere here.

Event model: ONE open event per camera, holding every object type seen while
it is open (it was one per (camera, label) — see _EventState); a track is
confirmed after MIN_HITS frames carrying its tracker_id; "update" emits on
best-score +0.02 / active-count change / a type arriving or leaving / 10 s
heartbeat; "end" after ABSENCE_TIMEOUT_S with NOTHING confirmed in view,
end_time = last time anything was seen.
Native event ids use the ``native.`` prefix (must never collide with the
``doorbell.``/``audio.`` synthetic-no-media prefixes in routers/events.py).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import cv2
import numpy as np

from ..event_labels import label_rank, primary_label
from . import zones as zonelib
from .coco_labels import ID_TO_LABEL
from .stillness import STATIONARY_AFTER_S, Stillness, clamp_stationary_after

if TYPE_CHECKING:  # pragma: no cover
    import supervision as sv

    from ..config import Config
    from ..db import Database
    from ..events_pipeline import EventsPipeline
    from ..settings_store import SettingsStore
    from .detector import OnnxDetector
    from .recorder import Recorder

log = logging.getLogger(__name__)

MIN_HITS = 3                 # frames carrying a tracker_id before a track confirms
ABSENCE_TIMEOUT_S = 5.0      # DEFAULT label-unseen timeout before an event ends;
                             # overridden by settings.detection.absence_timeout_s
UPDATE_SCORE_DELTA = 0.02    # best-score improvement that forces an "update"
UPDATE_HEARTBEAT_S = 10.0    # max seconds between "update" emits while open
ENDED_FRAME_KEEP_S = 60.0    # keep an ended event's best frame for late enrichment
_TRACK_FORGET_S = 60.0       # drop hit-counters for tracker_ids unseen this long
_HOUSEKEEPING_S = 2.0
_JPEG_QUALITY = 80
# Reject-suppression match radius as a fraction of detect_width (pixels). A
# stationary phantom re-fires within a few→~20 px of tracker/box wobble; ~6% of
# width (~42 px at 704) covers that while sparing a distinct subject standing
# well clear. Same-label matching prevents cross-class over-suppression.
SUPPRESS_RADIUS_FRAC = 0.06

#: What a "package" looks like to a COCO detector. There is no `package` or
#: `box` class, so these are the carried containers the model DOES know, and
#: they are what a parcel on a doorstep is most often reported as. The
#: Objects365 tier knows more, but this list stays COCO-only on purpose: a
#: label that only exists on one model tier would make the feature silently
#: depend on which model is loaded.
PACKAGE_LABELS = ("backpack", "handbag", "suitcase")

#: How long a package-shaped object must sit motionless before it counts as
#: LEFT rather than as being carried. Long enough that someone standing with a
#: bag over their shoulder, or setting one down to find keys, does not trip it.
PACKAGE_SETTLE_S = 45.0

#: A package is only interesting if a PERSON was recently here — that is what
#: separates "a parcel was delivered" from "the detector has decided the
#: doormat is a handbag". Generous, because the person may leave frame before
#: the object has finished settling.
PACKAGE_PERSON_WINDOW_S = 180.0


@dataclass(frozen=True)
class Observation:
    """One tracked detection on one frame (detect-stream pixel space)."""

    label: str
    tracker_id: int
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


# --------------------------------------------------------------------------
# Exempt (privacy / ignore) detection zones
# --------------------------------------------------------------------------
# Each camera row carries ``exempt_zones``: a JSON list of polygons in
# NORMALIZED (0..1, resolution-independent) coords. The engine converts them to
# detect-stream pixels ONCE per camera-row change (not per detection) and drops
# any observation whose box FOOT-CENTER — the midpoint of the bottom edge,
# ((x1+x2)/2, y2), i.e. where a person/vehicle meets the ground — lies inside
# any polygon. Polygons with fewer than 3 points are ignored.

# A detect-space polygon: an ordered list of (x, y) pixel vertices.
DetectPolygon = list[tuple[float, float]]


def _zone_points(zone: Any) -> Optional[Sequence[Any]]:
    """Pull the point list out of a stored zone, tolerating either the
    ``{"points": [[x, y], ...], "name": ...}`` object form or a bare
    ``[[x, y], ...]`` list."""
    if isinstance(zone, dict):
        return zone.get("points")
    if isinstance(zone, (list, tuple)):
        return zone
    return None


def _zone_name(zone: Any, index: int) -> str:
    """Best-effort display name for a stored zone (object form carries a
    ``name``; the bare-list form has none -> a positional fallback)."""
    if isinstance(zone, dict):
        name = str(zone.get("name") or "").strip()
        if name:
            return name
    return f"zone#{index}"


def exempt_detect_zones(row: dict[str, Any]) -> list[tuple[str, DetectPolygon]]:
    """Convert a camera row's normalized ``exempt_zones`` into ``(name,
    detect-space polygon)`` pairs using the row's ``detect_width``/
    ``detect_height``. Zones with fewer than 3 points (or malformed points)
    are skipped. The name is only used for logging."""
    zones = row.get("exempt_zones") or []
    dw = float(row.get("detect_width") or 0.0)
    dh = float(row.get("detect_height") or 0.0)
    out: list[tuple[str, DetectPolygon]] = []
    if dw <= 0 or dh <= 0:
        return out
    for i, zone in enumerate(zones):
        pts = _zone_points(zone)
        if not pts or len(pts) < 3:
            continue
        try:
            poly = [(float(p[0]) * dw, float(p[1]) * dh) for p in pts]
        except (TypeError, ValueError, IndexError):
            continue
        if len(poly) >= 3:
            out.append((_zone_name(zone, i), poly))
    return out


def exempt_detect_polygons(row: dict[str, Any]) -> list[DetectPolygon]:
    """Detect-space exempt polygons for a camera row (names dropped). Thin
    wrapper over :func:`exempt_detect_zones` — behavior is identical to the
    original single-purpose implementation."""
    return [poly for _name, poly in exempt_detect_zones(row)]


def point_in_polygon(x: float, y: float, poly: Sequence[tuple[float, float]]) -> bool:
    """Standard even-odd ray-casting point-in-polygon test. A point on an edge
    may test either way (the usual ray-cast ambiguity) — good enough for
    masking. Polygons with < 3 vertices are never inside."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def box_foot_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Foot-center of a detection box: midpoint of the bottom edge."""
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


# An object is masked by an exempt zone when its GROUND-CONTACT line — the bottom
# edge of its box, where a person's feet / a vehicle's tyres sit — substantially
# falls inside the zones. Testing the bottom edge (NOT the whole box) keeps the
# "where the object stands" meaning: a tall FOREGROUND object is never masked by
# a zone drawn higher in the frame just because its body projects over it in 2D
# (masking that would drop a real person walking up to the camera — the worst
# outcome for an NVR). Samples are UNIONED across all zones, so a wide object, an
# imprecisely-drawn zone, or an object straddling two adjacent zones is caught
# even when its exact foot-center pixel lands just outside a single polygon.
_EXEMPT_FOOT_MIN = 0.5       # fraction of the bottom-edge samples inside the zones
# A detection whose WHOLE box is almost entirely inside the excluded area is
# masked too — this kills a localized FALSE POSITIVE (a phantom box confined to
# the zone) whose foot happens to fall just outside it. The threshold is high so
# a real foreground object that merely OVERLAPS a high zone (its box extends well
# below the zone) stays under this bar and is never masked.
_EXEMPT_CONTAINED_MIN = 0.8  # fraction of the WHOLE box inside the zones


def _box_grid(
    box: tuple[float, float, float, float], grid: int = 5
) -> list[tuple[float, float]]:
    """A grid×grid lattice of sample points across the whole box (area test)."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return [((x1 + x2) / 2.0, (y1 + y2) / 2.0)]
    return [
        (x1 + (x2 - x1) * (i + 0.5) / grid, y1 + (y2 - y1) * (j + 0.5) / grid)
        for i in range(grid)
        for j in range(grid)
    ]


def _first_zone_containing(
    px: float, py: float, polys: Sequence[DetectPolygon]
) -> Optional[int]:
    """Index of the first polygon containing (px, py), or None."""
    for i, poly in enumerate(polys):
        if point_in_polygon(px, py, poly):
            return i
    return None


def box_foot_line(
    box: tuple[float, float, float, float], samples: int = 5
) -> list[tuple[float, float]]:
    """Evenly-spaced points along the box's bottom edge (its ground-contact
    line). A zero-width/degenerate box collapses to just the foot-center."""
    x1, _y1, x2, y2 = box
    if x2 <= x1 or samples <= 1:
        return [((x1 + x2) / 2.0, y2)]
    step = (x2 - x1) / (samples - 1)
    return [(x1 + step * i, y2) for i in range(samples)]


def box_in_exempt_zones(
    box: tuple[float, float, float, float], polys: Sequence[DetectPolygon]
) -> bool:
    """True when ``box`` is masked by the exempt zones (see
    :func:`first_exempt_zone_index`)."""
    return first_exempt_zone_index(box, polys) is not None


def first_exempt_zone_index(
    box: tuple[float, float, float, float], polys: Sequence[DetectPolygon]
) -> Optional[int]:
    """Index of an exempt polygon that masks ``box``, or None. A box is masked by
    ANY of three rules (all unioned across zones):

    1. **foot-center inside** a zone — the object is standing in the area;
    2. **≥``_EXEMPT_FOOT_MIN`` of its bottom-edge samples** inside — a wide
       object, or one straddling two adjacent / imprecisely-drawn ground zones;
    3. **≥``_EXEMPT_CONTAINED_MIN`` of its whole box** inside — a detection
       confined to the excluded area (a localized false positive) whose foot
       happens to fall just outside.

    Only rule 3 looks above the feet, and its bar is high, so a real FOREGROUND
    object whose box merely projects over a zone drawn higher in the frame (box
    extends well below the zone) is never masked. The index just names a matching
    zone for logging."""
    if not polys:
        return None
    fx, fy = box_foot_center(box)
    zi = _first_zone_containing(fx, fy, polys)
    if zi is not None:
        return zi
    # Rule 2: ground-contact line substantially inside the union of zones.
    foot_pts = box_foot_line(box)
    foot_hits = [(px, py) for (px, py) in foot_pts if _first_zone_containing(px, py, polys) is not None]
    if len(foot_hits) >= _EXEMPT_FOOT_MIN * len(foot_pts):
        return _first_zone_containing(foot_hits[0][0], foot_hits[0][1], polys)
    # Rule 3: the whole box is almost entirely inside the excluded area.
    grid_pts = _box_grid(box)
    grid_hits = [(px, py) for (px, py) in grid_pts if _first_zone_containing(px, py, polys) is not None]
    if len(grid_hits) >= _EXEMPT_CONTAINED_MIN * len(grid_pts):
        return _first_zone_containing(grid_hits[0][0], grid_hits[0][1], polys)
    return None


def observations_from_supervision(detections: "sv.Detections") -> list[Observation]:
    """sv.Detections (post-tracker) -> [Observation]; detections without a
    tracker_id are dropped (ByteTrack hasn't activated them yet — trackers
    2.4.0 marks those with tracker_id **-1**, and real ids start at 0)."""
    out: list[Observation] = []
    if detections is None or detections.tracker_id is None:
        return out
    if detections.confidence is None:
        # sv.DetectionsSmoother drops confidence for the whole frame if any
        # track in its window lacks it. Our detectors always set it, so this is
        # defence rather than a live path — but zip() over None would raise on
        # every frame, and a detection worker must not be one None away from
        # logging an exception 5 times a second.
        log.warning("detections arrived without confidence — dropping the frame")
        return out
    for xyxy, conf, class_id, tid in zip(
        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
    ):
        if tid is None or int(tid) < 0:
            continue
        label = ID_TO_LABEL.get(int(class_id))
        if label is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        out.append(Observation(label, int(tid), float(conf), (x1, y1, x2, y2)))
    return out


@dataclass
class _EventState:
    """ONE event per camera, open while anything worth reporting is in view.

    It used to be one per (camera, label): a person and the car they arrived in
    were two events a second apart, and the event list showed two rows for one
    thing that happened. Now the event opens on the first confirmed object of
    any type, gathers every type that appears while it is open, and ends only
    when ALL of them have been gone for the absence timeout.

    `label` is the event's NAME — the most important type seen so far (see
    `primary_label`), so it can change from "car" to "person" mid-event.
    """

    fid: str
    camera: str
    label: str
    start_time: float
    record_enabled: bool
    last_seen: float
    best_score: float = 0.0
    best_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    best_frame: Optional[np.ndarray] = None
    best_frame_time: Optional[float] = None
    # Every confirmed/counted object present at the best-frame moment (all
    # labels), so the annotated snapshot can box the whole scene. Refreshed
    # in lockstep with best_frame.
    best_scene: list[Observation] = field(default_factory=list)
    count: int = 0
    #: Types the loitering alert has fired for in this event. ONE per type, not
    #: one per interval: "still there" said every minute is the noise the
    #: feature is meant to replace. Per TYPE because the event is one per
    #: camera, and the car on the drive and the person standing by it are two
    #: different things to have been told about.
    dwell_alerted: set[str] = field(default_factory=set)
    last_emit_time: float = 0.0
    last_emit_score: float = 0.0
    last_emit_count: int = 0
    # Include zones this event's objects have stood in, and crossing lines they
    # have crossed, accumulated over the event's whole life (not just the best
    # frame). Emitted as the payload's `entered_zones`, which EventsPipeline
    # already stores on the event row and the web UI already displays.
    zones: set[str] = field(default_factory=set)
    # Per-line crossing tally FOR THIS EVENT: {line name: [in, out]}. Deliberately
    # not the LineZone's own counters, which are cumulative since boot — on an
    # event snapshot "1 in" means this visit, and a running total since the
    # backend last restarted would be noise on a piece of evidence.
    line_counts: dict[str, list[int]] = field(default_factory=dict)
    # Ground path per tracker_id for the objects in best_scene, captured when
    # the best frame was adopted so the trace drawn on the snapshot ends where
    # the boxes are. Refreshed in lockstep with best_frame.
    best_traces: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    #: Label of the object the best frame was adopted for. The best frame
    #: follows the most IMPORTANT subject first and the highest score second,
    #: so the snapshot of a car-then-person event shows the person.
    best_label: str = ""
    #: Every type seen during the event, in the order it first appeared.
    labels: list[str] = field(default_factory=list)
    #: Types in view NOW (seen within the absence timeout) -> how many.
    label_counts: dict[str, int] = field(default_factory=dict)
    #: When each present type was last seen, and when its current continuous
    #: presence began. A type that leaves and comes back starts again — which
    #: is what the loitering clock needs.
    label_last_seen: dict[str, float] = field(default_factory=dict)
    label_first_seen: dict[str, float] = field(default_factory=dict)
    #: The present set as of the last emit, so a type arriving or leaving is
    #: reported promptly rather than on the next heartbeat.
    last_emit_present: tuple[str, ...] = ()


@dataclass
class _CameraState:
    row: dict[str, Any]
    # Detect-space exempt polygons, precomputed from row.exempt_zones on every
    # row change (NOT per detection). Empty => no masking.
    exempt_polys: list[DetectPolygon] = field(default_factory=list)
    # Display names aligned index-for-index with exempt_polys (logging only).
    # May be shorter than exempt_polys when a state is built directly with
    # only polygons (e.g. tests) — the log falls back to a positional name.
    exempt_names: list[str] = field(default_factory=list)
    # Running count of detections dropped by exempt-zone masking (debug/status).
    masked_dropped: int = 0
    # Detect-space INCLUDE zones (native.zones), precomputed on every row change.
    # Empty => the whole frame is watched, exactly as before this existed. Any
    # entry flips the camera to allow-list mode: an object whose foot-center is
    # outside every one of them is dropped before it can open an event.
    include_zones: list[tuple[str, Any]] = field(default_factory=list)
    include_dropped: int = 0
    # Detect-space CROSSING lines (sv.LineZone), also precomputed per row change.
    # These are stateful — each holds the per-track history that makes a crossing
    # detectable — so they are REUSED across frames and only rebuilt when the
    # camera's geometry actually changes (see reload()).
    cross_lines: list[tuple[str, Any]] = field(default_factory=list)
    # The stored (normalized) geometry the current zones/lines were built from,
    # so reload() can leave the stateful LineZones alone when nothing changed.
    geometry_key: Any = None
    # tracker_id -> recent ground positions (detect px), bounded per track.
    # Collected unconditionally: a trace is a few hundred bytes per live track,
    # and collecting only when the draw toggle is on would mean turning it on
    # gives you nothing until the NEXT event.
    traces: dict[int, deque] = field(default_factory=dict)
    # Detect-space reject-suppression samples, precomputed from
    # detection_suppressions on every reload (NOT per detection):
    # (label, foot_px, foot_py). Empty => no suppression.
    suppress_samples: list[tuple[str, float, float]] = field(default_factory=list)
    # Match radius in detect-space pixels (SUPPRESS_RADIUS_FRAC * detect_width).
    suppress_radius: float = 0.0
    # Running count of detections dropped by reject-suppression (debug/status).
    suppress_dropped: int = 0
    # Detect-space FACE regions of interest (native.zones.polygon_zones).
    # Unlike include_zones these do NOT filter detection — they only mark where
    # a face is legible enough to be worth a recognition pass. Empty => the
    # whole frame, which is correct and merely slower.
    face_zones: list[tuple[str, Any]] = field(default_factory=list)
    # Detect-space PLATE regions of interest. Same contract as face_zones.
    plate_zones: list[tuple[str, Any]] = field(default_factory=list)
    # WHETHER this camera recognizes at all, as opposed to where it looks.
    # The zones cannot express this: an empty zone list means WHOLE FRAME,
    # so without these a box with recognition on ran a face pass on every
    # camera it had. Default True, matching the column default, so a camera
    # row from before the switches existed behaves as it always did.
    face_recognition: bool = True
    plate_recognition: bool = True
    # Per-camera override for settings.detection.ignore_stationary. None (the
    # default) means follow the global setting — which is what keeps the
    # global control meaningful for every camera nobody has pinned.
    ignore_stationary: Optional[bool] = None
    # Per-camera override for settings.detection.dwell_alert_seconds. None
    # follows the global setting; 0 means off HERE specifically.
    dwell_seconds: Optional[int] = None
    #: Frame time a person was last confirmed here, and the tracks already
    #: reported as a left package. Both per camera: tracker ids are only
    #: unique within one, and "a person was here" is a fact about this view.
    last_person_at: float = 0.0
    packages_reported: set[int] = field(default_factory=set)
    # tracker_id -> (hit count, last seen epoch)
    hits: dict[int, tuple[int, float]] = field(default_factory=dict)
    # Per-track motion state: what is moving, what arrived and settled, and
    # what was already sitting there. Only `active` tracks reach the event
    # layer. Per camera because tracker_ids are only unique within one.
    stillness: Stillness = field(default_factory=Stillness)
    latest_frame: Optional[np.ndarray] = None
    latest_frame_time: Optional[float] = None
    frame_times: deque = field(default_factory=lambda: deque(maxlen=600))


def _make_fid(start_time: float) -> str:
    # "native." prefix per contract; must NOT start with doorbell./audio.
    return f"native.{int(start_time * 1000)}-{secrets.token_hex(3)}"


class DetectionEngine:
    """Owns per-camera track/event state; emits into the EventsPipeline."""

    def __init__(
        self,
        db: "Database",
        detector: "OnnxDetector",
        recorder: "Recorder",
        settings: "SettingsStore",
        config: "Config",
    ):
        self._db = db
        self._detector = detector
        self._recorder = recorder
        self._settings = settings
        self._config = config
        self._pipeline: Optional["EventsPipeline"] = None
        self._cameras: dict[str, _CameraState] = {}
        # Keyed by CAMERA: one open event per camera (see _EventState).
        self._events: dict[str, _EventState] = {}
        # fid -> (expires_at, best frame) for events that already ended
        self._ended_frames: dict[str, tuple[float, np.ndarray]] = {}
        self._tasks: list[asyncio.Task] = []
        self._ingest: Optional[Any] = None  # IngestManager (created in start)
        # Camera-AI event listener (amcrest.ai_events.AiEventListener), wired by
        # main.py. Forwarded to the ingest manager so the per-frame gate can ask
        # whether a "camera_ai" camera's on-board AI is currently active.
        self._ai_events: Optional[Any] = None
        # Smart-spotlight controller (native.spotlight.SpotlightController),
        # wired by main.py. Notified per frame that carries a confirmed person
        # so it can arm/hold a night spotlight on smart_spotlight cameras.
        self._spotlight: Optional[Any] = None
        # Face recognition (native.facepass.FacePass). None until enabled —
        # a box that never turns recognition on pays nothing for it, and every
        # call site below is guarded rather than assuming it exists.
        self._face: Optional[Any] = None
        # Plate reading (native.platepass.PlatePass). Same contract as _face:
        # None until enabled, every call site guarded.
        self._plates: Optional[Any] = None
        self.running = False

    def set_spotlight(self, spotlight: Optional[Any]) -> None:
        """Inject the smart-spotlight controller. Safe to call before or after
        start(); the per-frame path notifies it when a person is present."""
        self._spotlight = spotlight

    def set_ai_events(self, ai_events: Optional[Any]) -> None:
        """Inject the camera-AI event listener (used by the ingest gate). Safe
        to call before or after start(); forwarded to the ingest manager once
        it exists."""
        self._ai_events = ai_events
        if self._ingest is not None:
            self._ingest.set_ai_events(ai_events)

    def set_face_pass(self, face: Optional[Any]) -> None:
        """Inject the face pass. Safe before or after start()."""
        self._face = face
        self._wire_recognition_hook()

    def set_plate_pass(self, plates: Optional[Any]) -> None:
        """Inject the plate pass. Safe before or after start()."""
        self._plates = plates
        self._wire_recognition_hook()

    def _wire_recognition_hook(self) -> None:
        """Give the passes a way to announce a recognition on a LIVE event.

        Called from BOTH set_pipeline and the two pass setters, because the
        wiring order is not fixed and getting it wrong is silent: main.py
        happens to call set_pipeline first, so wiring only there would have
        found no passes and left every alert unnamed with nothing to show for
        it. Idempotent, so being called three times costs nothing.
        """
        note = getattr(self._pipeline, "note_recognition", None)
        if note is None:
            return
        for pass_ in (self._face, self._plates):
            if pass_ is not None:
                pass_.on_recognition = note

    @property
    def recognition_model_key(self) -> str:
        """Embedding space currently loaded, or "" — read by /api/recognition."""
        return self._face.model_key if self._face is not None else ""

    async def reload_gallery(self) -> None:
        """Rebuild the recognition gallery after a profile/sample change.

        Best-effort: recognition being off is the normal state on a box that
        never enabled it, and a profile edit must still succeed there.
        """
        if self._face is not None:
            await self._face.reload_gallery()
        if self._plates is not None:
            await self._plates.reload_gallery()

    def set_pipeline(self, pipeline: "EventsPipeline") -> None:
        """Late-bound: the pipeline needs the media provider, which needs
        this engine — main.py wires the cycle up in that order."""
        self._pipeline = pipeline
        self._wire_recognition_hook()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        # Ingest: per-camera ffmpeg FrameSource tasks + the single inference
        # worker (latest-frame drop). Created lazily to avoid an import
        # cycle (ingest.py imports observations_from_supervision from here).
        if self._ingest is None:
            from .ingest import IngestManager

            self._ingest = IngestManager(
                self, self._detector, self._config, settings=self._settings
            )
            self._ingest.set_ai_events(self._ai_events)
        await self._ingest.start()
        await self.reload()
        self.running = True
        self._tasks.append(asyncio.create_task(self._housekeeping(), name="engine-housekeeping"))
        self._tasks.append(asyncio.create_task(self._detector.start(), name="detector-start"))

    async def stop(self) -> None:
        self.running = False
        if self._ingest is not None:
            await self._ingest.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        now = time.time()
        for key in list(self._events):
            await self._end_event(key, end_time=now)
        await self._detector.stop()

    async def reload(self) -> None:
        """Re-read camera rows + detection settings. Called at start, after
        camera CRUD and after settings.detection changes."""
        rows = await self._db.list_cameras()
        # Reject-suppression samples for ALL cameras in one query, grouped by
        # camera; each is scaled to detect-space pixels below (like exempt zones)
        # so the per-frame path does zero rescaling.
        suppress_by_cam: dict[str, list[dict[str, Any]]] = {}
        for s in await self._db.list_suppressions():
            suppress_by_cam.setdefault(s["camera"], []).append(s)
        fresh: dict[str, _CameraState] = {}
        for row in rows:
            prev = self._cameras.get(row["name"])
            state = prev if prev is not None else _CameraState(row=row)
            state.row = row
            # Precompute detect-space exempt polygons (+ names) on the row
            # change so the per-frame path never re-parses/scales them.
            zones = exempt_detect_zones(row)
            state.exempt_polys = [poly for _n, poly in zones]
            state.exempt_names = [name for name, _p in zones]
            if state.exempt_polys:
                log.info(
                    "camera %s: %d exempt detection zone(s) active",
                    row["name"], len(state.exempt_polys),
                )
            # Include zones + crossing lines. Rebuilt ONLY when the stored
            # geometry (or the detect resolution it scales against) actually
            # changed: a LineZone carries the per-track crossing history, and
            # rebuilding it on every unrelated reload — a camera rename, a
            # privacy toggle — would forget which side of the line everyone was
            # on and silently drop the crossing in progress.
            # Assigned unconditionally, OUTSIDE the geometry_key guard
            # below: these are scalars, not geometry, and folding them into
            # that key would rebuild every zone polygon each time someone
            # ticked a checkbox. Leaving them out of the guard entirely is
            # what makes the switch take effect on the next reload.
            state.face_recognition = bool(row.get("face_recognition", True))
            state.plate_recognition = bool(row.get("plate_recognition", True))
            # THREE-STATE: None means "follow settings.detection". Read with a
            # default of None rather than coerced with bool(), which would turn
            # "inherit" into "off for this camera" the moment the row came from
            # a fixture or a backend that predates the column.
            stationary = row.get("ignore_stationary")
            state.ignore_stationary = None if stationary is None else bool(stationary)
            dwell = row.get("dwell_seconds")
            state.dwell_seconds = None if dwell is None else int(dwell)
            geometry_key = (
                repr(row.get("include_zones") or []),
                repr(row.get("cross_lines") or []),
                repr(row.get("face_zones") or []),
                repr(row.get("plate_zones") or []),
                row.get("detect_width"),
                row.get("detect_height"),
            )
            if state.geometry_key != geometry_key:
                state.geometry_key = geometry_key
                state.include_zones = zonelib.include_detect_zones(row)
                state.cross_lines = zonelib.cross_detect_lines(row)
                state.face_zones = zonelib.polygon_zones(row, "face_zones", "face")
                state.plate_zones = zonelib.polygon_zones(row, "plate_zones", "plate")
                if state.include_zones:
                    log.info(
                        "camera %s: %d include zone(s) active — detections outside "
                        "them are dropped",
                        row["name"], len(state.include_zones),
                    )
                if state.cross_lines:
                    log.info(
                        "camera %s: %d crossing line(s) active",
                        row["name"], len(state.cross_lines),
                    )
            # Precompute detect-space reject-suppression samples + match radius.
            dw = float(row.get("detect_width") or 0.0)
            dh = float(row.get("detect_height") or 0.0)
            samples: list[tuple[str, float, float]] = []
            if dw > 0 and dh > 0:
                for s in suppress_by_cam.get(row["name"], []):
                    samples.append(
                        (str(s["label"]), float(s["foot_x"]) * dw, float(s["foot_y"]) * dh)
                    )
            state.suppress_samples = samples
            state.suppress_radius = SUPPRESS_RADIUS_FRAC * dw if dw > 0 else 0.0
            if samples:
                log.info(
                    "camera %s: %d reject-suppression sample(s) active",
                    row["name"], len(samples),
                )
            # PRIVACY MODE (app/privacy.py): drop the cached last frame for a
            # camera that is now private. Stopping its ingest source is NOT
            # enough — latest_frame is only ever overwritten, never cleared, so
            # the snapshot/JPEG surface (latest_frame_jpeg) would keep serving
            # the last REAL frame indefinitely while the operator believes the
            # camera is blacked out. Clearing here runs on every privacy toggle,
            # because privacy.apply() calls engine.reload().
            if self._settings.is_private(row["name"]):
                state.latest_frame = None
                state.latest_frame_time = 0.0
            fresh[row["name"]] = state
        self._cameras = fresh
        # Events for deleted/disabled cameras are ended by housekeeping.
        detection = self._settings.detection
        await self._detector.reconfigure(
            str(detection.get("model") or "dfine_s"),
            float(detection.get("confidence", 0.5)),
        )
        if self._ingest is not None:
            # settings.detection.default_mode gates cameras whose detect_mode is
            # unset/NULL; passed through so the ingest gate resolves the same
            # effective mode the AI listener does.
            default_mode = str(detection.get("default_mode") or "always")
            await self._ingest.reload(rows, default_mode=default_mode)

    # ---------- per-frame entry point (see module docstring) ----------

    async def process(
        self,
        camera: str,
        frame_time: float,
        observations: Sequence[Observation],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> None:
        cam = self._cameras.get(camera)
        if cam is None:
            return
        if frame_bgr is not None:
            cam.latest_frame = frame_bgr
            cam.latest_frame_time = frame_time
            cam.frame_times.append(frame_time)

        # Use the stored list as-is: an empty detect_objects means an empty
        # wanted-set -> no labels tracked -> no events (record-only). A NULL/
        # unset list was already backfilled to the defaults by the DB
        # migration, so empty here only comes from an explicit user action.
        wanted = set(cam.row.get("detect_objects") or [])
        obs = [o for o in observations if o.label in wanted]

        # --- include ("only alert here") zone filtering ---
        # Runs FIRST, so exempt zones below read as holes punched in the
        # included area. With no include zones configured this is skipped
        # entirely and the whole frame is watched, exactly as before.
        #
        # An object is in a zone when its FOOT-CENTER is (sv.PolygonZone with
        # BOTTOM_CENTER — the same anchor the exempt rules use). Deliberately a
        # single rule, unlike the exempt path's three: an include zone decides
        # what gets ignored, and a filter nobody can predict is a filter nobody
        # trusts.
        zone_names: dict[int, list[str]] = {}
        if cam.include_zones:
            hits = zonelib.zone_hits(obs, cam.include_zones)
            kept_i: list[Observation] = []
            for o, names in zip(obs, hits):
                if names:
                    zone_names[o.tracker_id] = names
                    kept_i.append(o)
                    continue
                cam.include_dropped += 1
                fx, fy = box_foot_center(o.box)
                # DEBUG, not INFO like the exempt path: an include zone on a
                # driveway drops every car on the road behind it, several times
                # a second, all day. The running total is on camera_stats
                # (include_dropped) for anyone who wants to confirm it is working.
                log.debug(
                    "outside include zones: %s at foot=(%.0f,%.0f) on %s",
                    o.label, fx, fy, camera,
                )
            obs = kept_i

        # --- exempt (privacy / ignore) zone masking ---
        # Drop any observation whose box foot-center lies inside an exempt
        # polygon (detect-space, precomputed in reload()). Empty polys =>
        # unchanged behavior. This runs BEFORE track bookkeeping so a suppressed
        # object never confirms, counts, or opens an event.
        if cam.exempt_polys:
            kept: list[Observation] = []
            for o in obs:
                zi = first_exempt_zone_index(o.box, cam.exempt_polys)
                if zi is None:
                    kept.append(o)
                    continue
                # Suppressed: same drop decision as box_in_exempt_zones, plus a
                # confirmation log so a deployed instance proves masking ran.
                cam.masked_dropped += 1
                fx, fy = box_foot_center(o.box)
                zone_name = cam.exempt_names[zi] if zi < len(cam.exempt_names) else f"zone#{zi}"
                log.info(
                    "masked %s at foot=(%.0f,%.0f) by exempt zone %s on %s",
                    o.label, fx, fy, zone_name, camera,
                )
            obs = kept

        # --- reject-suppression masking ---
        # Drop any observation whose SAME-LABEL foot-center lands within the
        # reject radius of a learned suppression sample (detect-space,
        # precomputed in reload()). Runs after exempt masking, before track
        # bookkeeping, so a suppressed object never confirms/counts/opens an
        # event.
        if cam.suppress_samples and cam.suppress_radius > 0:
            r2 = cam.suppress_radius * cam.suppress_radius
            kept_s: list[Observation] = []
            for o in obs:
                fx, fy = box_foot_center(o.box)
                if any(
                    lbl == o.label and (fx - sx) ** 2 + (fy - sy) ** 2 <= r2
                    for (lbl, sx, sy) in cam.suppress_samples
                ):
                    cam.suppress_dropped += 1
                    log.info(
                        "suppressed %s at foot=(%.0f,%.0f) near reject sample on %s",
                        o.label, fx, fy, camera,
                    )
                else:
                    kept_s.append(o)
            obs = kept_s

        # --- track confirmation bookkeeping ---
        for o in obs:
            count, _ = cam.hits.get(o.tracker_id, (0, 0.0))
            cam.hits[o.tracker_id] = (count + 1, frame_time)
        forgotten = [tid for tid, (_, seen) in cam.hits.items() if frame_time - seen > _TRACK_FORGET_S]
        for tid in forgotten:
            del cam.hits[tid]
            cam.traces.pop(tid, None)
        if forgotten:
            # Motion state is per track and must retire with it, or a
            # road-facing camera accumulates a dict entry per passing car
            # forever — and a REUSED tracker id would inherit the previous
            # occupant's "has moved" history, which is how a parked car
            # inherits a pedestrian's right to open an event.
            cam.stillness.forget(forgotten)
        if forgotten and cam.cross_lines:
            # sv.LineZone never prunes its own per-track crossing history. Ours
            # is forgotten here, so theirs is too — otherwise a camera watching
            # a road accumulates a dict entry per passing car, forever.
            zonelib.forget_tracks(cam.cross_lines, cam.hits.keys())
        if forgotten and self._face is not None:
            # A retired track is a finished VISIT, and its best shot is only
            # knowable now — so this is where the face pass identifies from the
            # best frame of the whole visit and writes the answer. Awaited
            # rather than fired-and-forgotten so the write cannot race the next
            # frame's pass over a reused tracker id.
            for tid in forgotten:
                await self._face.finish(camera, tid)
        if forgotten and self._plates is not None:
            for tid in forgotten:
                await self._plates.finish(camera, tid)
        confirmed = [o for o in obs if cam.hits[o.tracker_id][0] >= MIN_HITS]

        # --- stillness: which of these are actually DOING anything ---
        #
        # A detector answers "is there a car here?" on every frame, so a parked
        # car is detected five times a second forever. In an event model keyed
        # on (camera, label) that does not merely make noise — it holds the
        # label's event open, which means the car that PULLS IN arrives as a
        # count change on a stale event instead of as a new event. The arrival
        # is the thing worth telling someone about and it was the least visible
        # thing on the screen. See native/stillness.py.
        #
        # `scene` below deliberately keeps EVERY confirmed object, dormant ones
        # included: it is what gets boxed on the saved snapshot, and a picture
        # that omits the parked car is a picture that lies about the frame.
        scene = confirmed

        # MOTION STATE IS ALWAYS KEPT, even when the filter below is off. It
        # costs two subtractions and a hypot per observation, and two other
        # features read it — the left-package pass here, and the loitering hold
        # in `_stationary_after`. Keeping it behind the filter's own switch
        # meant turning the filter off silently disabled them too, with nothing
        # anywhere saying why.
        #
        # FED FROM `obs`, FILTERED ON `confirmed`. Motion history has to start
        # when a track is first SEEN, not when it confirms: fed from `confirmed`
        # instead, a track arrives at the event layer having been watched for
        # zero frames, so it has by definition never moved and is held back
        # until its NEXT step — which delays every real subject's event by a
        # frame and moves its start time off the moment they actually arrived.
        cam.stillness.stationary_after_s = self._stationary_after(cam)
        for o in obs:
            cam.stillness.update(o.tracker_id, o.box, frame_time)

        if self._ignore_stationary(cam):
            active: list[Observation] = []
            for o in confirmed:
                if cam.stillness.is_active(o.tracker_id, frame_time):
                    active.append(o)
                else:
                    cam.stillness.count_drop(o.tracker_id)
            confirmed = active

        # Left packages read the motion state DIRECTLY rather than the filtered
        # `confirmed`, because a package that was set down and never moved
        # again is precisely what the filter hides — so this runs on the full
        # confirmed set, before any of it is dropped.
        self._maybe_package(cam, scene, frame_time)

        # --- traces + line crossings (confirmed objects only) ---
        # Both are fed from `confirmed`, not `obs`: a crossing is only meaningful
        # if it belongs to an object real enough to have opened an event, and a
        # trace drawn from unconfirmed flicker is a path nobody walked.
        for o in confirmed:
            zonelib.push_trace(cam.traces, o.tracker_id, o.box)
        crossed_by_label: dict[str, list[zonelib.Crossing]] = {}
        if cam.cross_lines:
            # Triggered on EVERY frame, including empty ones — sv.LineZone
            # decides a crossing by comparing sides across frames, so a skipped
            # frame is a hole in the evidence, not a saved cycle.
            for crossing in zonelib.crossings(confirmed, cam.cross_lines):
                crossed_by_label.setdefault(crossing.label, []).append(crossing)
                log.info(
                    "%s crossed %s (%s) on %s",
                    crossing.label, crossing.line, crossing.direction, camera,
                )

        # --- per-label event state ---
        by_label: dict[str, list[Observation]] = {}
        for o in confirmed:
            by_label.setdefault(o.label, []).append(o)

        # --- smart-spotlight hook ---
        # A confirmed person survived detect_objects + exempt-zone filtering:
        # notify the controller ONCE per person-frame (it debounces + decides
        # smart_spotlight/white_light/night internally). Best-effort — a bad
        # controller must never break the detection worker.
        if self._spotlight is not None and "person" in by_label:
            try:
                self._spotlight.notify_person(cam.row)
            except Exception:  # noqa: BLE001
                log.exception("smart-spotlight notify failed for %s", camera)

        # --- face recognition pass ---
        # Fed from `confirmed` for the same reason the traces are: a face found
        # on unconfirmed flicker belongs to nobody. FacePass throttles itself
        # per track and swallows its own errors — recognition is an enhancement
        # on top of detection and recording, and must never cost a frame.
        if self._face is not None:
            # Hand over people AND vehicles, and let FacePass decide which it
            # actually wants (its `_labels`, driven by
            # settings.recognition.face_on_vehicles).
            #
            # This used to pass by_label["person"] alone, which made
            # face_on_vehicles dead on arrival: the pass would happily accept a
            # car, but no car was ever offered to it. The filter belongs in ONE
            # place, and that place is the pass that owns the setting.
            faceable = [
                o for label in self._face.labels for o in by_label.get(label, ())
            ]
            if faceable:
                # The camera's one open event, whatever it is named after.
                open_ev = self._events.get(camera)
                await self._face.observe(cam, faceable, frame_bgr, frame_time,
                                         open_ev.fid if open_ev is not None else "")

        # --- plate reading pass ---
        # Fed the vehicle labels the camera is actually tracking. PlatePass
        # picks its own out of the set and throttles per track, so handing it
        # the whole confirmed scene costs nothing when there is no vehicle.
        if self._plates is not None and confirmed:
            open_ev = self._events.get(camera)
            await self._plates.observe(cam, confirmed, frame_bgr, frame_time,
                                       open_ev.fid if open_ev is not None else "")

        # The full confirmed set is the "scene" saved with the event's best
        # frame — every counted object, all labels.
        absence_timeout = self._absence_timeout()
        if by_label:
            crossed_all = [c for group in crossed_by_label.values() for c in group]
            await self._observe_scene(
                cam, by_label, frame_time, frame_bgr, scene, zone_names, crossed_all,
                absence_timeout,
            )

        # --- absence ---
        # A type that has gone quiet leaves the event; the EVENT ends only when
        # every type has — "keep it running until everything has left".
        st = self._events.get(camera)
        if st is not None:
            if not by_label and frame_time - st.last_seen >= absence_timeout:
                await self._end_event(camera)
            elif not by_label:
                await self._retire_labels(st, frame_time, absence_timeout)

    async def _observe_scene(
        self,
        cam: _CameraState,
        by_label: dict[str, list[Observation]],
        frame_time: float,
        frame_bgr: Optional[np.ndarray],
        scene: Sequence[Observation],
        zone_names: Optional[dict[int, list[str]]],
        crossed: Sequence["zonelib.Crossing"],
        absence_timeout: float,
    ) -> None:
        """Open or extend the camera's one event with this frame's objects."""
        camera = cam.row["name"]
        group = [o for objs in by_label.values() for o in objs]
        best = min(group, key=lambda o: (label_rank(o.label), -o.score))
        st = self._events.get(camera)

        if st is None:
            st = _EventState(
                fid=_make_fid(frame_time),
                camera=camera,
                label=primary_label(list(by_label)),
                start_time=frame_time,
                record_enabled=bool(cam.row.get("record_enabled", True)),
                last_seen=frame_time,
            )
            self._events[camera] = st
            self._track_labels(st, by_label, frame_time)
            self._note_geometry(st, group, zone_names, crossed)
            self._adopt_best(cam, st, best, frame_time, frame_bgr, scene)
            st.last_emit_present = tuple(sorted(st.label_counts))
            await self._emit("new", st, frame_time)
            return

        st.last_seen = frame_time
        self._track_labels(st, by_label, frame_time)
        self._drop_quiet_labels(st, frame_time, absence_timeout, keep=by_label)
        self._maybe_dwell(cam, st, frame_time)
        # Zones and crossings are recorded BEFORE the emit decision below, so a
        # crossing that happens on a quiet frame — no score improvement, no
        # count change — still reaches the event row on the next update.
        self._note_geometry(st, group, zone_names, crossed)
        # A more important subject replaces the best frame outright; among
        # equals, a higher score does. So a car's event that a person then
        # walks into ends up with a snapshot of the person.
        if (label_rank(best.label), -best.score) < (label_rank(st.best_label), -st.best_score):
            self._adopt_best(cam, st, best, frame_time, frame_bgr, scene)
        count = st.count
        present = tuple(sorted(st.label_counts))

        # The HEARTBEAT is held back while every subject of this event is
        # motionless: an event whose people are all standing still has nothing
        # new to say every 10 seconds, and on a camera watching a parked car
        # that heartbeat is the entire content of the event log.
        #
        # Only the heartbeat. A score improvement, a count change and a line
        # crossing are new information no matter who is moving, and each still
        # emits — this must not become "nothing is reported while someone
        # stands at the door".
        heartbeat_due = frame_time - st.last_emit_time >= UPDATE_HEARTBEAT_S
        if heartbeat_due and cam.stillness.all_still(o.tracker_id for o in group):
            heartbeat_due = False

        if (
            st.best_score - st.last_emit_score >= UPDATE_SCORE_DELTA
            or count != st.last_emit_count
            or present != st.last_emit_present  # a type arrived or left
            or crossed  # a line crossing is the sharpest signal here — don't sit
                        # on it for up to a heartbeat before the row records it
            or heartbeat_due
        ):
            await self._emit("update", st, frame_time)

    def _track_labels(
        self, st: _EventState, by_label: dict[str, list[Observation]], frame_time: float
    ) -> None:
        """Fold this frame's objects into the event's per-type bookkeeping."""
        for label, objs in by_label.items():
            if label not in st.labels:
                st.labels.append(label)
            st.label_first_seen.setdefault(label, frame_time)
            st.label_last_seen[label] = frame_time
            n = len({o.tracker_id for o in objs})
            if st.label_counts.get(label) != n:
                st.label_counts[label] = n
                self._update_count(st.camera, label, n)
        st.label = primary_label(st.labels)
        st.count = sum(st.label_counts.values())

    def _drop_quiet_labels(
        self, st: _EventState, frame_time: float, absence_timeout: float,
        keep: Sequence[str] = (),
    ) -> bool:
        """Remove types unseen for the absence timeout. True if any left."""
        gone = [
            label for label in st.label_counts
            if label not in keep
            and frame_time - st.label_last_seen.get(label, frame_time) >= absence_timeout
        ]
        for label in gone:
            del st.label_counts[label]
            st.label_first_seen.pop(label, None)
            self._update_count(st.camera, label, 0)
        if gone:
            st.count = sum(st.label_counts.values())
        return bool(gone)

    async def _retire_labels(
        self, st: _EventState, frame_time: float, absence_timeout: float
    ) -> None:
        """On an empty frame: let types that have gone quiet leave the event,
        and say so, while the event itself stays open for the rest."""
        if self._drop_quiet_labels(st, frame_time, absence_timeout):
            present = tuple(sorted(st.label_counts))
            if present != st.last_emit_present:
                await self._emit("update", st, frame_time)

    @staticmethod
    def _note_geometry(
        st: _EventState,
        group: Sequence[Observation],
        zone_names: Optional[dict[int, list[str]]],
        crossed: Sequence["zonelib.Crossing"],
    ) -> None:
        """Accumulate include-zone names and line crossings onto the event.

        Both accumulate over the event's WHOLE life rather than tracking the
        current frame: "this person walked up the drive and crossed the front
        line" stays true for the event even once they have moved on, and that
        is what the operator wants to read afterwards.
        """
        if zone_names:
            for o in group:
                st.zones.update(zone_names.get(o.tracker_id, ()))
        for crossing in crossed:
            st.zones.add(crossing.line)
            tally = st.line_counts.setdefault(crossing.line, [0, 0])
            tally[0 if crossing.direction == "in" else 1] += 1

    @staticmethod
    def _adopt_best(
        cam: _CameraState,
        st: _EventState,
        best: Observation,
        frame_time: float,
        frame_bgr: Optional[np.ndarray],
        scene: Sequence[Observation],
    ) -> None:
        st.best_score = best.score
        st.best_label = best.label
        st.best_box = best.box
        # Scene tracks the saved frame: refresh it alongside best_frame.
        st.best_scene = list(scene)
        # Snapshot the traces as they stand at this instant. Copied, not
        # referenced: cam.traces keeps mutating as the object walks on, and a
        # trace drawn on this frame must end where this frame's boxes are.
        st.best_traces = {
            o.tracker_id: list(cam.traces.get(o.tracker_id, ())) for o in scene
        }
        if frame_bgr is not None:
            st.best_frame = frame_bgr.copy()
            st.best_frame_time = frame_time

    async def _end_event(self, camera: str, end_time: Optional[float] = None) -> None:
        st = self._events.pop(camera, None)
        if st is None:
            return
        st.last_seen = end_time if end_time is not None else st.last_seen
        if st.best_frame is not None:
            self._ended_frames[st.fid] = (time.monotonic() + ENDED_FRAME_KEEP_S, st.best_frame)
        for label in set(st.labels) | set(st.label_counts):
            self._update_count(st.camera, label, 0)
        st.label_counts.clear()
        await self._emit("end", st, st.last_seen)
        # has_clip is written to the row by the recorder ONLY after the clip
        # file is actually assembled (recorder.extract_clip); the engine never
        # asserts clip availability at event end (see _payload).
        if st.record_enabled:
            await self._recorder.schedule_clip(st.camera, st.fid, st.start_time, st.last_seen)

    # ---------- payload synthesis (design doc §4.1) ----------

    def _payload(self, etype: str, st: _EventState) -> dict[str, Any]:
        cam = self._cameras.get(st.camera)
        after: dict[str, Any] = {
            "id": st.fid,
            "camera": st.camera,
            # The event's NAME: its most important type so far. May change
            # mid-event (car -> person); the pipeline updates the row.
            "label": st.label,
            # Every type seen during the event, and the ones in view right now.
            # The pipeline drives the per-type Home Assistant sensors from
            # `present_labels`, and alerts again when a new type joins.
            "labels": list(st.labels),
            "present_labels": sorted(st.label_counts),
            "top_score": st.best_score,
            "start_time": st.start_time,
            "snapshot": {
                "frame_time": st.best_frame_time,
                "score": st.best_score,
                "box": list(st.best_box),
            },
            "box": list(st.best_box),
            # Full scene: every counted object in the saved frame (all labels),
            # so EventsPipeline can box them all. snapshot.box / box above stay
            # for backward compatibility.
            "scene": [
                {
                    "box": list(o.box),
                    "label": o.label,
                    "score": o.score,
                    "tracker_id": o.tracker_id,
                    # The ground path this object had walked when the frame was
                    # saved (detect px). Drawn by annotate.py when
                    # notifications.draw_traces is on; ignored otherwise.
                    "trace": [list(p) for p in st.best_traces.get(o.tracker_id, ())],
                }
                for o in st.best_scene
            ],
            # NEVER assert has_clip optimistically: the clip file does not exist
            # yet (recorder.schedule_clip assembles it ~20 s after end, or fails).
            # The recorder flips the row's has_clip to true only once the file is
            # written and non-empty — so the API reflects reality, not intent.
            "has_clip": False,
            "has_snapshot": st.best_frame is not None,
            # Include zones stood in + crossing lines crossed, over the event's
            # whole life. EventsPipeline already folds this into the event row's
            # `zones`, which the web UI already shows — nothing downstream had
            # to change to surface it.
            "entered_zones": sorted(st.zones),
            "current_zones": [],
            # Detect-space geometry for the snapshot annotators, carried on the
            # payload rather than re-read from the camera row at enrichment time:
            # this is the geometry that actually did the filtering/counting for
            # THIS event, even if the operator has since redrawn it.
            "include_zones": zonelib.zone_geometry(cam.include_zones) if cam else [],
            # "Only alert on a crossing" for this camera, carried on the payload
            # rather than re-read by the pipeline: this is the setting that was
            # in force for THIS event, and it costs the notification path no
            # database round-trip per update.
            "notify_on_cross": bool(cam.row.get("notify_on_cross")) if cam else False,
            "lines": [
                {**line, "in": st.line_counts.get(line["name"], (0, 0))[0],
                 "out": st.line_counts.get(line["name"], (0, 0))[1]}
                for line in (zonelib.line_geometry(cam.cross_lines) if cam else [])
            ],
        }
        if etype == "end":
            after["end_time"] = st.last_seen
        return {"type": etype, "before": {}, "after": after}

    async def _emit(self, etype: str, st: _EventState, frame_time: float) -> None:
        st.last_emit_time = frame_time
        st.last_emit_score = st.best_score
        st.last_emit_count = st.count
        st.last_emit_present = tuple(sorted(st.label_counts))
        if self._pipeline is None:  # not wired yet (boot ordering bug guard)
            log.warning("engine emit before pipeline wiring: %s %s", etype, st.fid)
            return
        try:
            await self._pipeline.handle_event(self._payload(etype, st))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — pipeline trouble must not kill ingest
            log.exception("pipeline rejected native %s payload for %s", etype, st.fid)

    def _update_count(self, camera: str, label: str, count: int) -> None:
        if self._pipeline is not None:
            self._pipeline.update_count(camera, label, count)

    # ---------- housekeeping ----------

    def _absence_timeout(self) -> float:
        """How long a label may go unseen before its event ends.

        Read per sweep rather than cached: a settings change takes effect on the
        next frame, matching every other detection setting. Both callers resolve
        it ONCE per pass — the frame loop must not re-read it per open event, or
        a mid-loop settings write could end one event and spare the next on the
        same frame.
        """
        # None-safe: an engine can be built without a settings store (every
        # offline engine test does exactly that), and the absence sweep runs on
        # EVERY frame — so an unset store must read as "use the default", not
        # raise 5 times a second per camera.
        detection = self._settings.detection if self._settings is not None else {}
        configured = (detection or {}).get("absence_timeout_s")
        return ABSENCE_TIMEOUT_S if configured is None else max(0.5, float(configured))

    def _detection_setting(self, key: str, default: Any) -> Any:
        """One `settings.detection` value, None-safe.

        Same contract as `_absence_timeout`: read per frame so a change lands
        on the next one, and an engine built WITHOUT a settings store (every
        offline engine test) must read as "the default" rather than raise five
        times a second per camera.
        """
        if self._settings is None:
            return default
        detection = self._settings.detection or {}
        value = detection.get(key)
        return default if value is None else value

    def _maybe_package(
        self, cam: _CameraState, confirmed: Sequence[Observation], frame_time: float
    ) -> None:
        """Report a package-shaped object that someone left behind.

        THE WHOLE SIGNAL IS: something that can be carried is now sitting
        still, it was not sitting there before, and a person was recently here.
        Each of those three carries its weight:

        * still for `PACKAGE_SETTLE_S` — a bag over a shoulder, or one set down
          while someone finds their keys, is not a delivery;
        * younger than the person window — otherwise the doormat the detector
          has decided is a handbag would be reported every time the engine
          restarted;
        * a person seen recently — parcels do not arrive on their own, and this
          is what separates a delivery from a persistent misdetection.

        Deliberately NOT gated on the stillness filter being enabled: a left
        package is exactly the "never moved" case that filter hides from the
        event layer, so this reads the motion state directly. Reported once per
        track, and never raises.
        """
        if not self._package_alerts():
            return
        for o in confirmed:
            if o.label == "person":
                cam.last_person_at = frame_time
        if frame_time - cam.last_person_at > PACKAGE_PERSON_WINDOW_S:
            return
        for o in confirmed:
            if o.label not in PACKAGE_LABELS:
                continue
            if o.tracker_id in cam.packages_reported:
                continue
            if cam.stillness.still_for(o.tracker_id, frame_time) < PACKAGE_SETTLE_S:
                continue
            # Age is bounded by the person window for the reason above: a thing
            # that has been in view far longer than anyone has been here is
            # furniture, however package-shaped the detector finds it.
            if cam.stillness.age(o.tracker_id, frame_time) > PACKAGE_PERSON_WINDOW_S:
                continue
            cam.packages_reported.add(o.tracker_id)
            note = getattr(self._pipeline, "note_package", None)
            if note is None:
                return
            try:
                note(cam.row["name"], o.label, list(o.box))
            except Exception:
                log.exception("could not announce a package on %s", cam.row["name"])

    def _package_alerts(self) -> bool:
        return bool(self._detection_setting("package_alerts", False))

    def _dwell_seconds(self, cam: _CameraState) -> int:
        """How long a subject may be present here before the loitering alert.

        0 means off. The camera's own value wins when it has one — including a
        pinned 0, which is how a pavement-facing camera opts out of an alert
        the rest of the system wants. None falls through to the global setting.
        """
        if cam.dwell_seconds is not None:
            return max(0, int(cam.dwell_seconds))
        try:
            return max(0, int(self._detection_setting("dwell_alert_seconds", 0)))
        except (TypeError, ValueError):
            return 0

    def _maybe_dwell(self, cam: _CameraState, st: _EventState, frame_time: float) -> None:
        """Tell the pipeline once when a type has been here long enough.

        Measured from when that TYPE's continuous presence in the event began,
        not from a track's start: a subject whose track is lost behind a pillar
        and re-acquired has not just arrived, and restarting the clock there
        would let a loiterer avoid the alert by standing where tracking is
        poor. Not from the event's start either — the event is one per camera
        now, and a person who walks up to a car that has been on the drive for
        ten minutes has not been loitering for ten minutes.

        Best-effort and never raises — a loitering alert is an enhancement on
        top of detection and recording, and must not cost a frame.
        """
        threshold = self._dwell_seconds(cam)
        if threshold <= 0:
            return
        # The most important type that has stayed long enough and not yet been
        # reported.
        due = [
            label for label in st.label_counts
            if label not in st.dwell_alerted
            and frame_time - st.label_first_seen.get(label, frame_time) >= threshold
        ]
        if not due:
            return
        label = primary_label(due)
        stayed = frame_time - st.label_first_seen[label]
        st.dwell_alerted.add(label)
        pipeline = self._pipeline
        note = getattr(pipeline, "note_dwell", None)
        if note is None:
            return
        try:
            note(st.fid, label, int(stayed))
        except Exception:
            log.exception("could not announce a dwell on %s", st.camera)

    def _ignore_stationary(self, cam: _CameraState) -> bool:
        """Whether motionless objects are held back for THIS camera.

        The camera's own value wins when it has one; None — the default, and
        what every camera has until somebody pins it — falls through to
        settings.detection. That order is what lets a driveway ignore parked
        cars while a back gate reports every sighting, without either choice
        being disturbed when the global default is changed.
        """
        if cam.ignore_stationary is not None:
            return cam.ignore_stationary
        return bool(self._detection_setting("ignore_stationary", True))

    def _stationary_after(self, cam: _CameraState) -> float:
        """How long a subject that HAS moved may sit still before it stops
        sustaining its event.

        CAMERA-AWARE, because the two features would otherwise silently cancel
        each other out. Stationary suppression drops a settled subject after
        `stationary_after_s`, at which point its label goes absent and the
        event ends; loitering wants to report on a subject that arrived and did
        NOT leave. So on a camera with a dwell threshold at or above the
        dormancy window, the loitering alert could never fire — the subject
        would be dropped before the clock reached it, and the feature would
        appear to be broken with nothing in the logs.

        Holding the subject until a little past its dwell threshold makes them
        compose: the parked car is still furniture (it never moved, so it was
        never active and this number does not apply to it), while a person
        standing at a door stays counted long enough to be reported.
        """
        base = clamp_stationary_after(
            self._detection_setting("stationary_after_s", STATIONARY_AFTER_S)
        )
        dwell = self._dwell_seconds(cam)
        if dwell <= 0:
            return base
        # +30 s so the dwell pass has frames to fire on rather than racing the
        # dormancy sweep at the exact same instant.
        return max(base, float(dwell) + 30.0)

    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(_HOUSEKEEPING_S)
            try:
                now = time.time()
                absence_timeout = self._absence_timeout()
                for camera, st in list(self._events.items()):
                    camera_gone = camera not in self._cameras or not bool(
                        self._cameras[camera].row.get("detect_enabled", True)
                    )
                    if camera_gone or now - st.last_seen >= absence_timeout:
                        await self._end_event(camera)
                mono = time.monotonic()
                for fid, (expires, _) in list(self._ended_frames.items()):
                    if mono >= expires:
                        del self._ended_frames[fid]
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("engine housekeeping cycle failed")

    def open_event(self, camera: str, label: Optional[str] = None) -> Optional[_EventState]:
        """The camera's open event — and, given `label`, only if that type is
        in view in it right now. For callers and tests that used to look an
        event up by (camera, label)."""
        st = self._events.get(camera)
        if st is None or (label is not None and label not in st.label_counts):
            return None
        return st

    # ---------- media surface (consumed by NativeMediaProvider) ----------

    def event_best_jpeg(self, fid: str) -> Optional[bytes]:
        """JPEG of the event's best frame (open events + recently ended).
        Sync + CPU-bound: call via asyncio.to_thread."""
        frame: Optional[np.ndarray] = None
        for st in self._events.values():
            if st.fid == fid:
                frame = st.best_frame
                break
        if frame is None:
            entry = self._ended_frames.get(fid)
            frame = entry[1] if entry else None
        return _encode_jpeg(frame)

    def latest_frame_jpeg(self, camera: str, height: Optional[int] = None) -> Optional[bytes]:
        """JPEG of the camera's most recent decoded frame (downscaled to
        ``height`` if given). Sync + CPU-bound: call via asyncio.to_thread."""
        cam = self._cameras.get(camera)
        if cam is None or cam.latest_frame is None:
            return None
        frame = cam.latest_frame
        if height and frame.shape[0] > height:
            width = max(1, round(frame.shape[1] * height / frame.shape[0]))
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return _encode_jpeg(frame)

    # ---------- stats (health / detector endpoints) ----------

    def camera_stats(self) -> list[dict[str, Any]]:
        """[{name, ingest_ok, fps, last_frame_age_s}] for detect-enabled
        cameras (the /api/system/detector per_camera block)."""
        now = time.time()
        out: list[dict[str, Any]] = []
        for name, cam in self._cameras.items():
            if not bool(cam.row.get("detect_enabled", True)):
                continue
            age = (now - cam.latest_frame_time) if cam.latest_frame_time else None
            recent = [t for t in cam.frame_times if now - t <= 10.0]
            fps = round(len(recent) / 10.0, 2)
            out.append(
                {
                    "name": name,
                    "ingest_ok": age is not None and age < 15.0,
                    "fps": fps,
                    "last_frame_age_s": round(age, 2) if age is not None else None,
                    # Debug: detections dropped so far by exempt-zone masking.
                    "masked_dropped": cam.masked_dropped,
                    # Debug: detections dropped for falling OUTSIDE every include
                    # zone. Zero on a camera with no include zones configured.
                    "include_dropped": cam.include_dropped,
                    # Cumulative crossings per line since this camera's lines were
                    # last (re)built — the dashboard number, as opposed to the
                    # per-event tally that goes on a snapshot.
                    "line_counts": {
                        name: {"in": line.in_count, "out": line.out_count}
                        for name, line in cam.cross_lines
                    },
                }
            )
        return out


def _encode_jpeg(frame: Optional[np.ndarray]) -> Optional[bytes]:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    return buf.tobytes() if ok else None

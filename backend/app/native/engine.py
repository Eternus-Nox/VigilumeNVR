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

Event model (design doc §4.2): one open event per (camera, label); a track
is confirmed after MIN_HITS frames carrying its tracker_id; "update" emits
on best-score +0.02 / active-count change / 10 s heartbeat; "end" after
ABSENCE_TIMEOUT_S without the label, end_time = last time it was seen.
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

from .coco_labels import ID_TO_LABEL

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
ABSENCE_TIMEOUT_S = 5.0      # label unseen this long => event end
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
    last_emit_time: float = 0.0
    last_emit_score: float = 0.0
    last_emit_count: int = 0


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
    # Detect-space reject-suppression samples, precomputed from
    # detection_suppressions on every reload (NOT per detection):
    # (label, foot_px, foot_py). Empty => no suppression.
    suppress_samples: list[tuple[str, float, float]] = field(default_factory=list)
    # Match radius in detect-space pixels (SUPPRESS_RADIUS_FRAC * detect_width).
    suppress_radius: float = 0.0
    # Running count of detections dropped by reject-suppression (debug/status).
    suppress_dropped: int = 0
    # tracker_id -> (hit count, last seen epoch)
    hits: dict[int, tuple[int, float]] = field(default_factory=dict)
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
        self._events: dict[tuple[str, str], _EventState] = {}
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

    def set_pipeline(self, pipeline: "EventsPipeline") -> None:
        """Late-bound: the pipeline needs the media provider, which needs
        this engine — main.py wires the cycle up in that order."""
        self._pipeline = pipeline

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
        for tid, (_, seen) in list(cam.hits.items()):
            if frame_time - seen > _TRACK_FORGET_S:
                del cam.hits[tid]
        confirmed = [o for o in obs if cam.hits[o.tracker_id][0] >= MIN_HITS]

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

        # The full confirmed set is the "scene" saved with whichever frame each
        # label's event adopts as its best — every counted object, all labels.
        for label, group in by_label.items():
            await self._observe_label(cam, label, group, frame_time, frame_bgr, confirmed)

        # --- absence: end open events whose label went quiet ---
        for key, st in list(self._events.items()):
            if key[0] != camera or st.label in by_label:
                continue
            if frame_time - st.last_seen >= ABSENCE_TIMEOUT_S:
                await self._end_event(key)

    async def _observe_label(
        self,
        cam: _CameraState,
        label: str,
        group: list[Observation],
        frame_time: float,
        frame_bgr: Optional[np.ndarray],
        scene: Sequence[Observation],
    ) -> None:
        camera = cam.row["name"]
        key = (camera, label)
        best = max(group, key=lambda o: o.score)
        count = len({o.tracker_id for o in group})
        st = self._events.get(key)

        if st is None:
            st = _EventState(
                fid=_make_fid(frame_time),
                camera=camera,
                label=label,
                start_time=frame_time,
                record_enabled=bool(cam.row.get("record_enabled", True)),
                last_seen=frame_time,
            )
            self._events[key] = st
            self._adopt_best(st, best, frame_time, frame_bgr, scene)
            st.count = count
            self._update_count(camera, label, count)
            await self._emit("new", st, frame_time)
            return

        st.last_seen = frame_time
        if best.score > st.best_score:
            self._adopt_best(st, best, frame_time, frame_bgr, scene)
        count_changed = count != st.count
        st.count = count
        if count_changed:
            self._update_count(camera, label, count)

        if (
            st.best_score - st.last_emit_score >= UPDATE_SCORE_DELTA
            or count != st.last_emit_count
            or frame_time - st.last_emit_time >= UPDATE_HEARTBEAT_S
        ):
            await self._emit("update", st, frame_time)

    @staticmethod
    def _adopt_best(
        st: _EventState,
        best: Observation,
        frame_time: float,
        frame_bgr: Optional[np.ndarray],
        scene: Sequence[Observation],
    ) -> None:
        st.best_score = best.score
        st.best_box = best.box
        # Scene tracks the saved frame: refresh it alongside best_frame.
        st.best_scene = list(scene)
        if frame_bgr is not None:
            st.best_frame = frame_bgr.copy()
            st.best_frame_time = frame_time

    async def _end_event(self, key: tuple[str, str], end_time: Optional[float] = None) -> None:
        st = self._events.pop(key, None)
        if st is None:
            return
        st.last_seen = end_time if end_time is not None else st.last_seen
        if st.best_frame is not None:
            self._ended_frames[st.fid] = (time.monotonic() + ENDED_FRAME_KEEP_S, st.best_frame)
        self._update_count(st.camera, st.label, 0)
        await self._emit("end", st, st.last_seen)
        # has_clip is written to the row by the recorder ONLY after the clip
        # file is actually assembled (recorder.extract_clip); the engine never
        # asserts clip availability at event end (see _payload).
        if st.record_enabled:
            await self._recorder.schedule_clip(st.camera, st.fid, st.start_time, st.last_seen)

    # ---------- payload synthesis (design doc §4.1) ----------

    def _payload(self, etype: str, st: _EventState) -> dict[str, Any]:
        after: dict[str, Any] = {
            "id": st.fid,
            "camera": st.camera,
            "label": st.label,
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
                {"box": list(o.box), "label": o.label, "score": o.score}
                for o in st.best_scene
            ],
            # NEVER assert has_clip optimistically: the clip file does not exist
            # yet (recorder.schedule_clip assembles it ~20 s after end, or fails).
            # The recorder flips the row's has_clip to true only once the file is
            # written and non-empty — so the API reflects reality, not intent.
            "has_clip": False,
            "has_snapshot": st.best_frame is not None,
            "entered_zones": [],
            "current_zones": [],
        }
        if etype == "end":
            after["end_time"] = st.last_seen
        return {"type": etype, "before": {}, "after": after}

    async def _emit(self, etype: str, st: _EventState, frame_time: float) -> None:
        st.last_emit_time = frame_time
        st.last_emit_score = st.best_score
        st.last_emit_count = st.count
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

    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(_HOUSEKEEPING_S)
            try:
                now = time.time()
                for key, st in list(self._events.items()):
                    camera_gone = key[0] not in self._cameras or not bool(
                        self._cameras[key[0]].row.get("detect_enabled", True)
                    )
                    if camera_gone or now - st.last_seen >= ABSENCE_TIMEOUT_S:
                        await self._end_event(key)
                mono = time.monotonic()
                for fid, (expires, _) in list(self._ended_frames.items()):
                    if mono >= expires:
                        del self._ended_frames[fid]
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("engine housekeeping cycle failed")

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
                }
            )
        return out


def _encode_jpeg(frame: Optional[np.ndarray]) -> Optional[bytes]:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    return buf.tobytes() if ok else None

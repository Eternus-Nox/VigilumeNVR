"""The per-track face pass: watch a person, keep their best shots, identify
them once, and record the answer.

WHERE THIS SITS
===============
``engine.process`` calls :meth:`FacePass.observe` for every frame carrying
confirmed people, and :meth:`FacePass.finish` when a track goes away. Everything
else — the model, the quality score, the gallery — lives in the three modules
this one coordinates:

    recognizer.py  detect + align + embed   (the only module that runs a model)
    bestshot.py    is this crop legible?
    recognition.py who does this vector belong to?

THE COST MODEL, WHICH IS THE WHOLE DESIGN
=========================================
Running face detection on every frame of every camera would roughly double this
box's inference load for no benefit. Three things keep it cheap, and they are
the reason this class exists rather than a few lines inside ``process``:

1. FACE DETECTION RUNS ON THE PERSON CROP, not the frame. D-FINE has already
   said where the people are; searching the other 95% of a 704x480 frame for
   faces is work whose answer is already known. It also removes a class of
   false positive outright — a face found in foliage is not inside a person
   box.

2. THE PASS IS THROTTLED PER TRACK. A person crossing a driveway is on screen
   for tens of frames and does not change much between two of them, so a pass
   every ``PASS_INTERVAL_S`` collects the same information for a fraction of
   the cost. The best-shot buffer's own time-diversity rule wants spread-out
   samples anyway, so throttling and quality actively agree here.

3. EMBEDDING HAPPENS ONCE PER TRACK, not once per shot. Crops are cheap to
   keep and score; the 128-d vector is the expensive part. So the pass buffers
   shots continuously and only embeds when a shot is good enough to identify
   from — and again later only if a MUCH better shot turns up.

WHY THE ANSWER IS WRITTEN AT TRACK END
--------------------------------------
A track's best shot is not known until the track is over. Identifying at the
first usable frame would mean identifying from the worst shot that cleared the
bar, which is precisely the mistake bestshot.py exists to avoid. So the pass
identifies eagerly (so a notification can say "that's Adam" while he is still
on the doorstep) but RE-identifies from the final best shot when the track
ends, and that is the answer that reaches the database.

WHAT GETS STORED
----------------
    matched   -> event_recognitions row naming the profile
    unmatched -> event_recognitions row with profile_id NULL (a face WAS read
                 and belonged to nobody: that is the row an unknown-person
                 alert is built from) AND a recognition_candidates row with the
                 crop, so the person can be enrolled afterwards.

Candidates are deduplicated against the gallery's near-misses, and the store is
capped — see `_prune`.

FULL-RESOLUTION LOOKS
---------------------
The detect stream is ~704x480. A person at the door there has a face a few
dozen pixels wide; below ~40 px it is not even tried, and between 24 and 56 px
the embedding is measurably weaker (same face vs itself at full size: cosine
0.65 at 24 px, 0.88 at 40, 0.93 at 112), which is the margin a real-world
match — different day, light and angle — has to survive on. The camera's own
picture is three to six times wider.

So while a person is tracked, the pass also asks the camera for a
full-resolution snapshot about once a second (`SnapshotSource`, SHARED with the
plate reader: one request serves both, and every person and vehicle on the
camera at that moment), finds the person in it (platesnap.locate — the same
template match the plate reader uses, because the snapshot arrives a few
hundred milliseconds after the detect frame), and detects, aligns and scores
the face from those pixels. The shot joins the same best-shot buffer, where a
sharp 150 px face simply outscores a 35 px one. It runs in the background —
the engine's frame loop never waits on a camera — and stops once the person is
identified from a good shot, or after HIRES_MAX_PER_TRACK looks.

RECORDED BURSTS (native/burst.py)
---------------------------------
A face is often only frontal for a moment — as someone walks up, glances at
the camera, or turns at the door — and that moment is frequently BEFORE
detection has confirmed them, or between two snapshots. So the pass also
reads the camera's own recording: starting a second after the person is first
seen, from two seconds before that up to now, ten frames a second, cropped
around where they were in each frame; again every ~1.2 s while they are in
view; and once more just after they leave. Every frame's face (only one in the
upper part of THIS person's box at that moment) is scored into the same
best-shot buffer, so the shot identified from is the best of all of them.

Identification still uses the single best shot. Combining several was
measured on degraded copies of real faces: averaging the best three raised
the match rate on clean single-person crops (97% -> 98.5-100%), but on crops
with a second face nearby it doubled the wrong-person rate (3% -> 6%) — it
amplifies whichever face dominates the shots — and a wrong name is worse
than "unknown". A veto on a disagreeing second shot cost correct names
(91% -> 83%) without removing a single wrong one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import platesnap, trackpath
from . import zones as zonelib
from .burst import (
    BURST_FPS, MAX_BURSTS, TAIL_DELAY_S, TIME_TOLERANCE_S, BurstState,
)
from .bestshot import (
    KEEP_SHOTS, MIN_GAP_S, BestShotBuffer, Shot, clamp_setting, encode_frame_box,
    score_face, shot_params,
)
from .recognition import Gallery, Match, to_blob
from .heatmap import HeatmapAccumulator
from .recognizer import FaceRecognizer, crop_with_origin

log = logging.getLogger(__name__)

#: Minimum spacing between face passes on ONE track. At the 5 fps detect stream
#: these cameras run this is every ~3rd frame.
PASS_INTERVAL_S = 0.6

#: Quality at which a shot is good enough to identify from. Below it an
#: embedding is not worth computing — the vector would be dominated by blur and
#: would match either nobody or, worse, the wrong person.
IDENTIFY_QUALITY = 0.45

#: How much better a later shot must be before re-embedding a track we have
#: already identified. Small improvements do not change the answer and are not
#: worth the inference.
REIDENTIFY_IMPROVEMENT = 0.15

#: Padding around the person box before looking for a face in it. A person box
#: can clip the top of the head, and YuNet wants a little context.
PERSON_CROP_PAD = 0.08

#: Labels that always get a face pass.
FACE_LABELS = ("person",)

#: Labels that get one only when settings.recognition.face_on_vehicles is
#: set — the driver through a windscreen. Kept separate from FACE_LABELS
#: because it is a genuinely different trade: on a road-facing camera most
#: windscreens are glare, and a plate identifies a car better than a face
#: does. On a driveway or at a gate it is the only way to get the driver.
VEHICLE_FACE_LABELS = ("car", "truck", "bus", "motorcycle")

#: Hard cap on the rolling candidate store. Age-based purging is the primary
#: control (settings.recognition.candidate_retention_days); this is the backstop
#: that stops one busy night on a street-facing camera from filling the disk
#: before the daily purge ever runs.
MAX_CANDIDATES = 2000

#: How often the maintenance loop checks the enable flag and purges.
RUN_INTERVAL_S = 300.0

#: Fallback when settings.recognition.candidate_retention_days is missing or
#: unparseable. Matches the documented default.
DEFAULT_RETENTION_DAYS = 7.0

#: Full-resolution looks (module docstring): at most one per track this often,
#: and at most this many per track.
HIRES_INTERVAL_S = 1.0
HIRES_MAX_PER_TRACK = 6

#: A track identified from a shot at least this good needs no more looks.
HIRES_SETTLED_QUALITY = 0.7

#: When the person cannot be found in the snapshot (they moved), the face is
#: searched for this far around where they were, as a fraction of their box —
#: and only a face in the upper part of that box is taken as theirs.
HIRES_LOST_MARGIN = 0.25

#: A track that ends with a look in flight waits this long for it — in the
#: background — before its answer is written.
HIRES_FINISH_WAIT_S = 3.0

#: Margin around the person (fraction of their box) cropped from each
#: recorded frame, and the part of the box a face must sit in to be theirs:
#: the upper FACE_REGION of it, give or take a little.
BURST_LOOK_MARGIN = 0.12
FACE_REGION = 0.65

#: A new candidate whose best gallery score is within this of an existing
#: candidate's is treated as the same unknown person and skipped. Without it, one
#: stranger walking past twelve times produces twelve rows to wade through.
CANDIDATE_DEDUPE_COSINE = 0.62


@dataclass
class _TrackState:
    buffer: BestShotBuffer = field(default_factory=BestShotBuffer)
    last_pass: float = 0.0
    camera: str = ""
    #: Quality of the shot the current identification was made from; None until
    #: the track has been identified at all.
    identified_from: Optional[float] = None
    match: Optional[Match] = None
    embedding: Optional[np.ndarray] = None
    event_fid: str = ""
    # Full-resolution looks.
    hires_task: Optional["asyncio.Task[None]"] = None
    hires_last: float = 0.0
    hires_requests: int = 0
    # Serializes identification: the detect-frame pass and a full-resolution
    # look can both reach it for one track at once.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: Where the person was over time: (wall time, box as FRACTIONS of the
    #: frame), every sighting — what a recorded burst crops by.
    path: deque = field(default_factory=lambda: deque(maxlen=600))
    #: Recorded bursts (native/burst.py) for this person.
    burst: BurstState = field(default_factory=BurstState)


class FacePass:
    """Per-camera, per-track face recognition. One instance per engine."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        db: Any,
        images_dir: Path,
        on_recognition: Optional[Any] = None,
        snapshots: Optional[Any] = None,
        replay: Optional[Any] = None,
    ) -> None:
        # Called the moment a track is identified, NOT when the row is stored.
        # The stored row lands at track end, which for an alert is long after
        # the person has walked away — so a notification that wants to say
        # "Adam is at the door" has to hear about it here.
        self.on_recognition = on_recognition
        self._recognizer = recognizer
        # Full-resolution frames (platesnap.SnapshotSource, shared with the
        # plate pass). None disables full-resolution looks.
        self._snapshots = snapshots
        # The recording, read back in bursts (platereplay.RecordingReplay,
        # shared with the plate pass). None disables bursts.
        self._replay = replay
        # Recorded bursts, counted since boot, for the status screen.
        self.bursts: dict[str, int] = {
            "bursts": 0,        # bursts that came back with frames
            "frames": 0,        # recorded frames read
            "faces": 0,         # faces offered from them
            "early_faces": 0,   # ... from frames BEFORE the person was first seen
        }
        # The settings store, for the live `face_hires` switch. Set by run().
        self._settings: Any = None
        # Finishes deferred behind an in-flight look (see finish()).
        self._pending: set[asyncio.Task] = set()
        # Full-resolution looks, counted since boot, for the status screen.
        self.hires: dict[str, int] = {
            "requested": 0,   # looks started
            "frames": 0,      # a usable full-resolution frame arrived
            "faces": 0,       # a face was found and offered from it
            "lost": 0,        # the person could not be found in it
            "no_face": 0,     # the person was found, their face was not
        }
        self._db = db
        self._images_dir = Path(images_dir)
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        # Shot-buffer shape, refreshed from settings on the maintenance
        # tick. Held here rather than read per-frame because a settings
        # lookup in the per-observation path would cost more than the crop
        # it is sizing. Applied to tracks STARTED after a change; a track
        # already in flight keeps the buffer it was built with, which is
        # correct — resizing mid-visit would discard shots already chosen.
        self._shots = KEEP_SHOTS
        self._shot_gap = MIN_GAP_S
        self._pass_interval = PASS_INTERVAL_S
        self._identify_quality = IDENTIFY_QUALITY
        # Which labels this pass looks at. Recomputed on the tick so that
        # turning face_on_vehicles on takes effect without a restart.
        self._labels: tuple[str, ...] = FACE_LABELS
        # WHY EVERY FACE WAS DROPPED, counted.
        #
        # A face can fail to become a reviewable candidate at six separate
        # points, and every one of them was silent — so "unknown faces is
        # emptier than it should be" had no answer short of reading this file.
        # Each is a legitimate outcome; what was missing was the ability to
        # tell WHICH is happening, because the remedies are opposite. "too
        # small" means move the camera or raise the detect resolution; "below
        # quality" means lower the floor; "duplicate" means it is working as
        # intended and the person has already been seen.
        self.drops: dict[str, int] = {
            "no_face_found": 0,      # YuNet found nothing in the person crop
            "below_quality": 0,      # scored under the buffer's floor, incl. the
                                     # resolution veto for a face under FACE_MIN_PX
            "no_shot_at_end": 0,     # track ended with an empty buffer
            "embed_failed": 0,       # the embedder returned nothing
            "duplicate": 0,          # same stranger already in the store
            "crop_write_failed": 0,  # row stored, image did not — a placeholder
        }
        self.kept = 0
        # Where faces are actually legible on each camera — the map the ROI
        # editor draws over the live frame. Accumulated in memory and flushed
        # on the maintenance tick.
        self.heatmap = HeatmapAccumulator(db)
        self._gallery: Optional[Gallery] = None
        self._gallery_lock = asyncio.Lock()

    # ---------- gallery ----------

    @property
    def model_key(self) -> str:
        return self._recognizer.model_key

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels this pass wants offered to it.

        Read by the engine so the person/vehicle decision lives in ONE place —
        here, with the setting that drives it. The engine previously hardcoded
        "person", which made `face_on_vehicles` unreachable: the pass would
        accept a car and no car was ever handed to it.
        """
        return self._labels

    async def reload_gallery(self) -> None:
        """Rebuild the in-memory gallery from the database.

        Called on start and after any profile/sample change (the profiles
        router pokes this through engine.reload_gallery). Rebuilt wholesale
        rather than mutated: a gallery is hundreds of vectors, the rebuild is
        microseconds, and a snapshot means the matcher never sees a
        half-applied enrollment.
        """
        async with self._gallery_lock:
            try:
                profiles = await (
                    await self._db.conn.execute("SELECT * FROM profiles")
                ).fetchall()
                samples = await (
                    await self._db.conn.execute("SELECT * FROM profile_samples")
                ).fetchall()
            except Exception:
                log.exception("could not load the recognition gallery")
                return
            self._gallery = Gallery.build(
                [dict(r) for r in profiles],
                [dict(r) for r in samples],
                model_key=self.model_key,
            )
            counts = self._gallery.counts()
            log.info(
                "recognition gallery loaded: %d profile(s) %s",
                len(self._gallery), counts or "(none enrolled)",
            )

    # ---------- the per-frame pass ----------

    async def observe(
        self,
        cam: Any,
        observations: Sequence[Any],
        frame_bgr: Optional[np.ndarray],
        frame_time: float,
        event_fid: str = "",
    ) -> None:
        """Look for faces on the confirmed people in this frame.

        Never raises. Recognition is an enhancement on top of detection and
        recording; a failure here must not cost the frame.
        """
        if frame_bgr is None or not self._recognizer.ready:
            return
        # Per-camera opt-out, checked BEFORE any work. This is the cheapest
        # possible exit and the point of the switch: a camera watching a
        # driveway at 30 m cannot produce a legible face, so every millisecond
        # spent detecting one there is wasted and every crop it does produce is
        # a marginal one that can only make a FALSE MATCH more likely.
        if not getattr(cam, "face_recognition", True):
            return
        camera = cam.row.get("name", "")
        try:
            people = [o for o in observations if o.label in self._labels]
            if not people:
                return
            # ROI: only look where a face is actually legible. Empty means the
            # whole frame, which is correct and merely slower.
            if cam.face_zones:
                hits = zonelib.zone_hits(people, cam.face_zones)
                people = [o for o, names in zip(people, hits) if names]
                if not people:
                    return
            cam_row = dict(getattr(cam, "row", None) or {})
            cam_row.setdefault("name", camera)
            for obs in people:
                await self._observe_one(camera, cam_row, obs, frame_bgr, frame_time, event_fid)
        except Exception:
            log.exception("face pass failed on %s", camera)

    async def _observe_one(
        self, camera: str, cam_row: dict[str, Any], obs: Any, frame_bgr: np.ndarray,
        frame_time: float, event_fid: str,
    ) -> None:
        key = (camera, obs.tracker_id)
        st = self._tracks.get(key)
        if st is None:
            st = _TrackState(
                camera=camera,
                buffer=BestShotBuffer(keep=self._shots, min_gap_s=self._shot_gap),
            )
            self._tracks[key] = st
        if event_fid:
            st.event_fid = event_fid
        # Every sighting, before the throttle: a burst crops by the path.
        fh0, fw0 = frame_bgr.shape[:2]
        if fw0 > 0 and fh0 > 0:
            bx1, by1, bx2, by2 = (float(v) for v in obs.box[:4])
            st.path.append((frame_time, (bx1 / fw0, by1 / fh0, bx2 / fw0, by2 / fh0)))
            st.burst.note_seen(frame_time)
            self._maybe_burst(camera, st, obs.tracker_id, frame_time)
        if frame_time - st.last_pass < self._pass_interval:
            return
        st.last_pass = frame_time
        # Before the detect-frame look, and whatever it finds: the face that
        # is too small to be found HERE is exactly the one worth a snapshot.
        self._maybe_look_hires(cam_row, camera, st, obs, frame_bgr, frame_time)

        cropped = crop_with_origin(frame_bgr, obs.box, pad=PERSON_CROP_PAD)
        if cropped is None:
            return
        person, ox, oy = cropped
        faces = await self._recognizer.detect(person)
        if not faces:
            self.drops["no_face_found"] += 1
            return
        # One person box, one face: the biggest. A second face inside a person
        # box is someone standing behind them, and it belongs to THEIR track.
        face = max(faces, key=lambda f: f.width * f.height)

        # ALIGN ONLY. The embedding is the expensive half and is not needed
        # until this track has a shot worth identifying from.
        aligned = await self._recognizer.align(person, face)
        if aligned is None:
            return
        quality = score_face(aligned, landmarks=face.landmarks)

        # Heatmap. The face box is in the PERSON CROP's coordinates, so it is
        # meaningless until translated by the crop's origin — that translation
        # is the whole reason crop_with_origin exists. Getting it wrong would
        # not fail; it would paint a plausible map of the wrong places.
        fh, fw = frame_bgr.shape[:2]
        frame_box: Optional[tuple[float, float, float, float]] = None
        if fw > 0 and fh > 0:
            cx = (ox + (face.box[0] + face.box[2]) / 2.0) / fw
            cy = (oy + (face.box[1] + face.box[3]) / 2.0) / fh
            self.heatmap.record(camera, "face", cx, cy, quality.total)
            # Same translation, kept as a rectangle: this is what lets the
            # review UI ring THIS face in the event snapshot rather than
            # showing a 112px crop with no context. Normalized here, while the
            # frame dimensions are still in hand — `finish()` runs long after
            # the frame is gone.
            frame_box = (
                max(0.0, min(1.0, (ox + face.box[0]) / fw)),
                max(0.0, min(1.0, (oy + face.box[1]) / fh)),
                max(0.0, min(1.0, (ox + face.box[2]) / fw)),
                max(0.0, min(1.0, (oy + face.box[3]) / fh)),
            )
        if not st.buffer.offer(
            tracker_id=obs.tracker_id, kind="face", crop_bgr=aligned,
            box=face.box, frame_time=frame_time, quality=quality,
            frame_box=frame_box,
        ):
            # Refused: under the quality floor, or too close in time to a
            # shot already held. Only the first is worth counting as a
            # drop — the second is the diversity rule working.
            if not quality:
                self.drops["below_quality"] += 1
        await self._maybe_identify(st, obs.tracker_id)

    async def _maybe_identify(self, st: _TrackState, tracker_id: int) -> None:
        """Identify from the best shot so far, if it is good enough and
        meaningfully better than the one already identified from."""
        async with st.lock:
            best = st.buffer.best(tracker_id, "face")
            if best is None or best.quality.total < self._identify_quality:
                return
            already = st.identified_from
            if already is not None and best.quality.total - already < REIDENTIFY_IMPROVEMENT:
                return
            await self._identify(st, tracker_id, best)

    # ---------- recorded bursts (native/burst.py) ----------

    def _replay_on(self) -> bool:
        """Whether recorded bursts are wanted. Read live."""
        if self._replay is None or not getattr(self._replay, "available", False):
            return False
        if self._settings is None:
            return True
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
        except Exception:  # noqa: BLE001
            return True
        return bool(cfg.get("face_replay", True))

    @staticmethod
    def _settled(st: _TrackState) -> bool:
        return st.identified_from is not None and st.identified_from >= HIRES_SETTLED_QUALITY

    def _maybe_burst(self, camera: str, st: _TrackState, tracker_id: int, now: float) -> None:
        """Start a live burst if one is due, and (re)arm the one for after the
        person's last sighting. Never awaits."""
        if not self._replay_on() or self._settled(st) or len(st.path) < 2:
            return
        window = st.burst.due(now)
        if window is not None:
            self._launch_burst(camera, st, tracker_id, window, live=True)
        if st.burst.tail is not None:
            st.burst.tail.cancel()
        st.burst.tail = asyncio.get_running_loop().call_later(
            TAIL_DELAY_S, self._tail_burst, camera, tracker_id
        )

    def _tail_burst(self, camera: str, tracker_id: int) -> None:
        """The person has not been seen for TAIL_DELAY_S: read the seconds as
        they left, without waiting for the track to be retired."""
        st = self._tracks.get((camera, tracker_id))
        if st is None:
            return
        st.burst.tail = None
        if st.burst.busy:
            st.burst.tail = asyncio.get_running_loop().call_later(
                0.5, self._tail_burst, camera, tracker_id
            )
            return
        if not self._replay_on() or self._settled(st) or st.burst.count >= MAX_BURSTS + 1:
            return
        window = st.burst.window(final=True)
        if window is not None:
            self._launch_burst(camera, st, tracker_id, window, live=False)

    def _launch_burst(
        self, camera: str, st: _TrackState, tracker_id: int,
        window: tuple[float, float], *, live: bool,
    ) -> None:
        st.burst.started()
        st.burst.task = asyncio.create_task(
            self._run_burst(camera, st, tracker_id, window, live=live),
            name=f"face-burst-{camera}-{tracker_id}",
        )

    async def _run_burst(
        self, camera: str, st: _TrackState, tracker_id: int,
        window: tuple[float, float], *, live: bool,
    ) -> None:
        """Decode [start, end] of the recording around this person, offer the
        face in every frame, and identify from the best. Never raises."""
        start, end = window
        try:
            path = list(st.path)
            if not path:
                return
            region = trackpath.expand(
                trackpath.union_over(path, start - TIME_TOLERANCE_S, end + TIME_TOLERANCE_S),
                BURST_LOOK_MARGIN,
            )
            frames = await self._replay.frames(
                camera, start, end, region, fps=BURST_FPS, pad=False, live=live,
            )
            st.burst.finished(start, end, frames[-1].time if frames else None, BURST_FPS)
            if not frames:
                return
            self.bursts["bursts"] += 1
            self.bursts["frames"] += len(frames)
            found = await asyncio.to_thread(self._faces_in_frames_blocking, frames, path)
            first_seen = path[0][0]
            for fr_time, aligned, box, quality, frame_box in found:
                if st.buffer.offer(
                    tracker_id=tracker_id, kind="face", crop_bgr=aligned, box=box,
                    frame_time=fr_time, quality=quality, frame_box=frame_box,
                ):
                    self.bursts["faces"] += 1
                    if fr_time < first_seen:
                        self.bursts["early_faces"] += 1
            await self._maybe_identify(st, tracker_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("face burst failed on %s/%s", camera, tracker_id)

    def _faces_in_frames_blocking(self, frames: Sequence[Any], path: list) -> list:
        """For each recorded frame: crop around where the person was at that
        moment, find THEIR face (centre in the upper part of their box), align
        and score it. Runs in a thread."""
        out = []
        for fr in frames:
            near = trackpath.box_near(path, fr.time, TIME_TOLERANCE_S)
            ch, cw = fr.crop.shape[:2]
            look = trackpath.to_pixels(trackpath.expand(near, BURST_LOOK_MARGIN),
                                       fr.width, fr.height, fr.ox, fr.oy)
            lx1, ly1 = max(0, int(look[0])), max(0, int(look[1]))
            lx2, ly2 = min(cw, int(round(look[2]))), min(ch, int(round(look[3])))
            if lx2 - lx1 < 16 or ly2 - ly1 < 16:
                continue
            sub = fr.crop[ly1:ly2, lx1:lx2]
            faces = self._recognizer.detect_blocking(sub)
            if not faces:
                continue
            px1, py1, px2, py2 = trackpath.to_pixels(near, fr.width, fr.height,
                                                     fr.ox + lx1, fr.oy + ly1)
            pw, ph = px2 - px1, py2 - py1
            head = (px1 - 0.1 * pw, py1 - 0.1 * ph, px2 + 0.1 * pw, py1 + FACE_REGION * ph)
            mine = [f for f in faces if trackpath.contains(
                head, (f.box[0] + f.box[2]) / 2.0, (f.box[1] + f.box[3]) / 2.0)]
            if not mine:
                continue
            face = max(mine, key=lambda f: f.width * f.height)
            aligned = self._recognizer.align_blocking(sub, face)
            if aligned is None:
                continue
            quality = score_face(aligned, landmarks=face.landmarks)
            fx1, fy1 = fr.ox + lx1 + face.box[0], fr.oy + ly1 + face.box[1]
            fx2, fy2 = fr.ox + lx1 + face.box[2], fr.oy + ly1 + face.box[3]
            frame_box = (max(0.0, min(1.0, fx1 / fr.width)), max(0.0, min(1.0, fy1 / fr.height)),
                         max(0.0, min(1.0, fx2 / fr.width)), max(0.0, min(1.0, fy2 / fr.height)))
            out.append((fr.time, aligned, face.box, quality, frame_box))
        return out

    # ---------- full-resolution looks ----------

    def _hires_on(self) -> bool:
        """Whether full-resolution looks are wanted. Read live, so the switch
        takes effect at once."""
        if self._snapshots is None:
            return False
        if self._settings is None:
            return True
        try:
            cfg = (self._settings.current or {}).get("recognition") or {}
        except Exception:  # noqa: BLE001
            return True
        return bool(cfg.get("face_hires", True))

    def _maybe_look_hires(
        self, cam_row: dict[str, Any], camera: str, st: _TrackState, obs: Any,
        frame_bgr: np.ndarray, frame_time: float,
    ) -> None:
        """Start a full-resolution look at this person if one is due. Never awaits."""
        if not self._hires_on():
            return
        if st.hires_task is not None and not st.hires_task.done():
            return
        if st.hires_requests >= HIRES_MAX_PER_TRACK:
            return
        if frame_time - st.hires_last < HIRES_INTERVAL_S:
            return
        if st.identified_from is not None and st.identified_from >= HIRES_SETTLED_QUALITY:
            return
        if not self._snapshots.available(camera):
            return
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in obs.box[:4])
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(fw, int(round(x2))), min(fh, int(round(y2)))
        if xi2 - xi1 < platesnap.MIN_TEMPLATE_PX or yi2 - yi1 < platesnap.MIN_TEMPLATE_PX:
            return
        # The person only, copied: the engine reuses its frames.
        template = frame_bgr[yi1:yi2, xi1:xi2].copy()
        st.hires_last = frame_time
        st.hires_requests += 1
        self.hires["requested"] += 1
        st.hires_task = asyncio.create_task(
            self._look_hires(cam_row, camera, st, obs.tracker_id, template,
                             (xi1, yi1, xi2, yi2), (fh, fw)),
            name=f"face-hires-{camera}-{obs.tracker_id}",
        )

    async def _look_hires(
        self, cam_row: dict[str, Any], camera: str, st: _TrackState, tracker_id: int,
        template: np.ndarray, box: tuple[int, int, int, int],
        detect_shape: tuple[int, int],
    ) -> None:
        """Fetch a full-resolution frame, find the person in it, and offer
        their face from those pixels. Never raises."""
        try:
            got = await self._snapshots.fetch(cam_row)
            if got is None:
                return
            hires, taken = got
            if hires.shape[1] < detect_shape[1] * platesnap.MIN_GAIN:
                self._snapshots.note_no_gain(camera, hires.shape, detect_shape)
                return
            self._snapshots.note_gain(camera)
            self.hires["frames"] += 1
            hh, hw = hires.shape[:2]
            sx, sy = hw / float(detect_shape[1]), hh / float(detect_shape[0])
            found = await asyncio.to_thread(platesnap.locate, template, box, detect_shape, hires)
            if found is not None:
                pbox, pad, lost = found[0], PERSON_CROP_PAD, False
            else:
                # Not found with confidence (they moved, or turned): search
                # around where they were, and accept only a face where THEIR
                # head would be.
                pbox = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
                pad, lost = HIRES_LOST_MARGIN, True
            cropped = crop_with_origin(hires, pbox, pad=pad)
            if cropped is None:
                self.hires["lost"] += 1
                return
            person, ox, oy = cropped
            faces = await self._recognizer.detect(person)
            if lost:
                px1, py1, px2, py2 = pbox
                pw, ph = px2 - px1, py2 - py1
                faces = [
                    f for f in faces
                    if px1 - 0.15 * pw <= ox + (f.box[0] + f.box[2]) / 2.0 <= px2 + 0.15 * pw
                    and py1 - 0.15 * ph <= oy + (f.box[1] + f.box[3]) / 2.0 <= py1 + 0.6 * ph
                ]
            if not faces:
                self.hires["lost" if lost else "no_face"] += 1
                return
            face = max(faces, key=lambda f: f.width * f.height)
            aligned = await self._recognizer.align(person, face)
            if aligned is None:
                return
            quality = score_face(aligned, landmarks=face.landmarks)
            frame_box = (
                max(0.0, min(1.0, (ox + face.box[0]) / hw)),
                max(0.0, min(1.0, (oy + face.box[1]) / hh)),
                max(0.0, min(1.0, (ox + face.box[2]) / hw)),
                max(0.0, min(1.0, (oy + face.box[3]) / hh)),
            )
            self.heatmap.record(camera, "face", (frame_box[0] + frame_box[2]) / 2.0,
                                (frame_box[1] + frame_box[3]) / 2.0, quality.total)
            if st.buffer.offer(
                tracker_id=tracker_id, kind="face", crop_bgr=aligned, box=face.box,
                frame_time=taken, quality=quality, frame_box=frame_box,
            ):
                self.hires["faces"] += 1
            elif not quality:
                self.drops["below_quality"] += 1
            await self._maybe_identify(st, tracker_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("full-resolution face look failed on %s", camera)

    async def _identify(self, st: _TrackState, tracker_id: int, shot: Shot) -> None:
        """Embed the best shot so far and match it. Cheap enough to redo once."""
        # shot.crop came out of align(), so it is already canonical — embed the
        # pixels as they are. Re-aligning would warp an aligned crop twice.
        vector = await self._recognizer.feature(shot.crop)
        if vector is None:
            return
        st.embedding = vector
        st.identified_from = shot.quality.total
        gallery = self._gallery
        st.match = gallery.match_face(vector) if gallery is not None else None
        if st.match is not None and st.match.matched:
            log.info(
                "recognized %s on %s (score %.3f, margin %.3f, shot quality %.2f)",
                st.match.name, st.camera, st.match.score, st.match.margin,
                shot.quality.total,
            )
            self._announce(st)

    def _announce(self, st: _TrackState) -> None:
        """Tell the events pipeline who this is, while the event is still live.

        Best-effort and never raises: recognition is an enhancement, and a
        pipeline that has already closed the event (or a wiring that was never
        made) must not cost the recognition itself.
        """
        if self.on_recognition is None or not st.event_fid or st.match is None:
            return
        try:
            self.on_recognition(
                st.event_fid, "face",
                name=st.match.name,
                profile_id=st.match.profile_id,
                score=st.match.score,
                alert_mode=st.match.alert_mode,
            )
        except Exception:
            log.exception("could not announce a face recognition")

    # ---------- track end ----------

    async def finish(self, camera: str, tracker_id: int) -> None:
        """A track ended: identify from its FINAL best shot and store the answer.

        If a full-resolution look is still in flight — often the look with the
        best face of the visit — the answer waits for it IN THE BACKGROUND, up
        to HIRES_FINISH_WAIT_S: the engine awaits this inside its frame loop.
        """
        st = self._tracks.pop((camera, tracker_id), None)
        if st is None:
            return
        if st.burst.tail is not None:
            st.burst.tail.cancel()
            st.burst.tail = None
        task = st.hires_task
        hires_busy = task is not None and not task.done()
        final_burst = (
            self._replay_on() and not self._settled(st) and len(st.path) >= 2
            and st.burst.window(final=True) is not None
        )
        if hires_busy or st.burst.busy or final_burst:
            deferred = asyncio.create_task(
                self._finish_after(task, st, camera, tracker_id),
                name=f"face-finish-{camera}-{tracker_id}",
            )
            self._pending.add(deferred)
            deferred.add_done_callback(self._pending.discard)
            return
        await self._conclude(st, camera, tracker_id)

    async def _finish_after(
        self, task: Optional["asyncio.Task[None]"], st: _TrackState, camera: str, tracker_id: int,
    ) -> None:
        for pending, wait in ((task, HIRES_FINISH_WAIT_S), (st.burst.task, HIRES_FINISH_WAIT_S * 4)):
            if pending is None or pending.done():
                continue
            try:
                await asyncio.wait_for(asyncio.shield(pending), wait)
            except asyncio.TimeoutError:
                pending.cancel()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — each logs its own failures
                pass
        if self._replay_on() and not self._settled(st):
            # Whatever the live bursts did not cover, read now it is over.
            window = st.burst.window(final=True)
            if window is not None:
                st.burst.started()
                await self._run_burst(camera, st, tracker_id, window, live=False)
        await self._conclude(st, camera, tracker_id)

    async def wait_idle(self) -> None:
        """Wait for deferred finishes (tests, shutdown)."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def _cancel_hires(self, camera: Optional[str] = None) -> None:
        for (cam, _tid), st in self._tracks.items():
            if camera is not None and cam != camera:
                continue
            if st.hires_task is not None and not st.hires_task.done():
                st.hires_task.cancel()
            st.burst.cancel()

    async def _conclude(self, st: _TrackState, camera: str, tracker_id: int) -> None:
        try:
            shot = st.buffer.best(tracker_id, "face")
            if shot is None:
                self.drops["no_shot_at_end"] += 1
                return
            # Re-identify from the best shot of the WHOLE visit, which is only
            # knowable now. This is the answer that reaches the database.
            async with st.lock:
                if st.identified_from is None or shot.quality.total > st.identified_from:
                    await self._identify(st, tracker_id, shot)
            if st.embedding is None:
                self.drops["embed_failed"] += 1
                return
            await self._store(st, shot)
        except Exception:
            log.exception("could not finish face track %s/%s", camera, tracker_id)

    async def _store(self, st: _TrackState, shot: Shot) -> None:
        match = st.match
        matched = match is not None and match.matched
        now = time.time()

        await self._db.conn.execute(
            "INSERT INTO event_recognitions (event_fid, kind, profile_id, name, plate, "
            "score, quality, image_path, created_at) VALUES (?, 'face', ?, ?, '', ?, ?, '', ?)",
            (
                st.event_fid,
                match.profile_id if matched else None,
                match.name if matched else "",
                float(match.score) if match is not None else 0.0,
                float(shot.quality.total),
                now,
            ),
        )
        if not matched:
            await self._store_candidate(st, shot, now)
        await self._db.conn.commit()

    async def _store_candidate(self, st: _TrackState, shot: Shot, now: float) -> None:
        """Keep an unmatched face so it can be enrolled later."""
        if await self._is_duplicate(st.embedding):
            self.drops["duplicate"] += 1
            return
        match = st.match
        cur = await self._db.conn.execute(
            "INSERT INTO recognition_candidates (kind, camera, event_fid, embedding, dim, "
            "plate, model_key, image_path, quality, frame_box, best_score, best_profile_id, "
            "created_at) VALUES ('face', ?, ?, ?, ?, '', ?, '', ?, ?, ?, ?, ?)",
            (
                st.camera, st.event_fid, to_blob(st.embedding), int(st.embedding.shape[0]),
                self.model_key, float(shot.quality.total), encode_frame_box(shot.frame_box),
                float(match.score) if match is not None else 0.0,
                None, now,
            ),
        )
        candidate_id = cur.lastrowid
        name = await asyncio.to_thread(self._write_crop, candidate_id, shot.crop)
        if not name:
            # The row survives without its image — the embedding is what
            # matches — but the operator sees a placeholder and cannot
            # judge it, so this is a real failure and must be visible.
            self.drops["crop_write_failed"] += 1
            log.warning(
                "candidate %d stored WITHOUT its crop — %s is not writable?",
                candidate_id, self._images_dir,
            )
        else:
            self.kept += 1
        if name:
            await self._db.conn.execute(
                "UPDATE recognition_candidates SET image_path = ? WHERE id = ?",
                (name, candidate_id),
            )
        await self._prune()

    def _write_crop(self, candidate_id: int, crop: np.ndarray) -> str:
        import cv2

        try:
            self._images_dir.mkdir(parents=True, exist_ok=True)
            name = f"{candidate_id}.jpg"
            ok = cv2.imwrite(str(self._images_dir / name), crop,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return name if ok else ""
        except Exception:
            log.exception("could not write candidate crop %d", candidate_id)
            return ""

    async def _is_duplicate(self, vector: Optional[np.ndarray]) -> bool:
        """Has this same unknown person already been stored?

        One stranger walking past twelve times should produce one row to review,
        not twelve. Compared against the recent candidate embeddings rather than
        the gallery — by definition these matched nobody in the gallery.
        """
        if vector is None:
            return False
        try:
            rows = await (
                await self._db.conn.execute(
                    "SELECT embedding, dim FROM recognition_candidates "
                    "WHERE kind = 'face' AND model_key = ? "
                    "ORDER BY created_at DESC LIMIT 200",
                    (self.model_key,),
                )
            ).fetchall()
        except Exception:
            log.exception("candidate dedupe query failed")
            return False
        from .recognition import cosine, from_blob

        for r in rows:
            other = from_blob(r["embedding"], int(r["dim"] or 0))
            if other is None:
                continue
            if cosine(vector, other) >= CANDIDATE_DEDUPE_COSINE:
                return True
        return False

    async def _prune(self) -> None:
        """Backstop cap on the candidate store (age purging is elsewhere)."""
        try:
            row = await (
                await self._db.conn.execute(
                    "SELECT COUNT(*) AS n FROM recognition_candidates"
                )
            ).fetchone()
            excess = (row["n"] or 0) - MAX_CANDIDATES
            if excess <= 0:
                return
            # Drop the WORST-QUALITY rows, not the oldest: this store exists to
            # be enrolled from, so the shot worth keeping is the legible one.
            victims = await (
                await self._db.conn.execute(
                    "SELECT id, image_path FROM recognition_candidates "
                    "ORDER BY quality ASC, created_at ASC LIMIT ?",
                    (excess,),
                )
            ).fetchall()
            ids = [r["id"] for r in victims]
            await self._db.conn.execute(
                f"DELETE FROM recognition_candidates WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            for r in victims:
                if r["image_path"]:
                    (self._images_dir / r["image_path"]).unlink(missing_ok=True)
        except Exception:
            log.exception("candidate prune failed")

    # ---------- background loop ----------

    async def run(self, settings: Any) -> None:
        """Honour the enable flag and purge expired candidates.

        One loop rather than two because the two decisions are coupled: when
        recognition is switched OFF the models are released AND the rolling
        biometric store stops being topped up, and the operator who switched it
        off should not have to also remember to clear it.
        """
        while True:
            try:
                await self._tick(settings)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("recognition maintenance tick failed")
            await asyncio.sleep(RUN_INTERVAL_S)

    async def _tick(self, settings: Any) -> None:
        self._settings = settings
        cfg = (settings.get() or {}).get("recognition") or {}
        enabled = bool(cfg.get("enabled"))
        self._shots, self._shot_gap = shot_params(cfg)
        self._pass_interval = clamp_setting(
            cfg.get('pass_interval_seconds'), PASS_INTERVAL_S, 0.1, 5.0)
        self._identify_quality = clamp_setting(
            cfg.get('identify_quality'), IDENTIFY_QUALITY, 0.15, 0.9)
        self._labels = (
            FACE_LABELS + VEHICLE_FACE_LABELS
            if cfg.get('face_on_vehicles') else FACE_LABELS
        )

        if enabled and self._recognizer.ready and getattr(
                self._recognizer, "stale_device", lambda: False)():
            # Built while the detector was elsewhere (usually: still warming
            # up at boot). Rebuild once onto where it is now — the embeddings
            # are identical on either runtime, so the gallery stays valid.
            log.info("face models: the detector has moved — rebuilding to follow it")
            self._recognizer.close()
        if enabled and not self._recognizer.ready:
            # load() downloads on first use and never raises; a box with no
            # outbound network degrades to ready:false and keeps detecting.
            if await self._recognizer.load():
                await self.reload_gallery()
        elif not enabled and self._recognizer.ready:
            log.info("recognition disabled — releasing the face models")
            self._recognizer.close()
            self._cancel_hires()
            self._tracks.clear()

        await self.purge_expired(cfg.get("candidate_retention_days"))
        # The heatmap is flushed even when recognition has just been switched
        # off: those sightings already happened, and discarding them would lose
        # the map that tells the operator where to aim the camera.
        await self.heatmap.flush()
        await self.heatmap.maybe_decay()

    async def purge_expired(self, retention_days: Any) -> int:
        """Drop candidate crops past the retention window. Returns rows removed.

        This is the biometric retention control, so it is deliberately literal:
        0 (or anything non-positive) clears the store outright rather than
        meaning "keep forever". An operator setting a retention of zero is
        saying "do not keep strangers' faces", and the most dangerous possible
        reading of that is the permissive one.
        """
        try:
            days = float(retention_days)
        except (TypeError, ValueError):
            days = DEFAULT_RETENTION_DAYS
        cutoff = time.time() - max(0.0, days) * 86400.0
        try:
            victims = await (
                await self._db.conn.execute(
                    "SELECT id, image_path FROM recognition_candidates WHERE created_at < ?",
                    (cutoff,),
                )
            ).fetchall()
            if not victims:
                return 0
            await self._db.conn.execute(
                "DELETE FROM recognition_candidates WHERE created_at < ?", (cutoff,)
            )
            await self._db.conn.commit()
            for r in victims:
                if r["image_path"]:
                    (self._images_dir / r["image_path"]).unlink(missing_ok=True)
            log.info("purged %d expired face candidate(s) (retention %.1f days)",
                     len(victims), days)
            return len(victims)
        except Exception:
            log.exception("candidate purge failed")
            return 0

    # ---------- lifetime ----------

    def forget_missing(self, camera: str, live_ids: Sequence[int]) -> list[int]:
        """Return tracker ids this camera is holding that are no longer live.

        The caller passes each to `finish`; this does not drop them itself,
        because finishing is what writes the row.
        """
        live = set(live_ids)
        return [tid for (cam, tid) in self._tracks if cam == camera and tid not in live]

    def forget_camera(self, camera: str) -> None:
        self._cancel_hires(camera)
        for key in [k for k in self._tracks if k[0] == camera]:
            del self._tracks[key]

    def status(self) -> dict[str, Any]:
        return {
            "live_tracks": len(self._tracks),
            "gallery": (self._gallery.counts() if self._gallery is not None else {}),
            "model_key": self.model_key,
            "labels": list(self._labels),
            # Counted since boot. `drops` answers "why is Unknown Faces emptier
            # than I expected" without reading the source, and the remedies
            # differ per reason — see the comment on self.drops.
            "kept_candidates": self.kept,
            "drops": dict(self.drops),
            # Full-resolution looks (module docstring), and whether they are on.
            "hires_enabled": self._hires_on(),
            "hires": dict(self.hires),
            # Recorded bursts (module docstring), and whether they are on.
            "bursts_enabled": self._replay_on(),
            "bursts": dict(self.bursts),
            "tuning": {
                "shots_per_track": self._shots,
                "shot_min_gap_seconds": self._shot_gap,
                "pass_interval_seconds": self._pass_interval,
                "identify_quality": self._identify_quality,
            },
        }

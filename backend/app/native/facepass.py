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
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import zones as zonelib
from .bestshot import BestShotBuffer, Shot, score_face
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

#: Only these labels get a face pass.
FACE_LABELS = ("person",)

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


class FacePass:
    """Per-camera, per-track face recognition. One instance per engine."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        db: Any,
        images_dir: Path,
    ) -> None:
        self._recognizer = recognizer
        self._db = db
        self._images_dir = Path(images_dir)
        self._tracks: dict[tuple[str, int], _TrackState] = {}
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
        camera = cam.row.get("name", "")
        try:
            people = [o for o in observations if o.label in FACE_LABELS]
            if not people:
                return
            # ROI: only look where a face is actually legible. Empty means the
            # whole frame, which is correct and merely slower.
            if cam.face_zones:
                hits = zonelib.zone_hits(people, cam.face_zones)
                people = [o for o, names in zip(people, hits) if names]
                if not people:
                    return
            for obs in people:
                await self._observe_one(camera, obs, frame_bgr, frame_time, event_fid)
        except Exception:
            log.exception("face pass failed on %s", camera)

    async def _observe_one(
        self, camera: str, obs: Any, frame_bgr: np.ndarray,
        frame_time: float, event_fid: str,
    ) -> None:
        key = (camera, obs.tracker_id)
        st = self._tracks.get(key)
        if st is None:
            st = _TrackState(camera=camera)
            self._tracks[key] = st
        if event_fid:
            st.event_fid = event_fid
        if frame_time - st.last_pass < PASS_INTERVAL_S:
            return
        st.last_pass = frame_time

        cropped = crop_with_origin(frame_bgr, obs.box, pad=PERSON_CROP_PAD)
        if cropped is None:
            return
        person, ox, oy = cropped
        faces = await self._recognizer.detect(person)
        if not faces:
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
        if fw > 0 and fh > 0:
            cx = (ox + (face.box[0] + face.box[2]) / 2.0) / fw
            cy = (oy + (face.box[1] + face.box[3]) / 2.0) / fh
            self.heatmap.record(camera, "face", cx, cy, quality.total)
        st.buffer.offer(
            tracker_id=obs.tracker_id, kind="face", crop_bgr=aligned,
            box=face.box, frame_time=frame_time, quality=quality,
        )

        best = st.buffer.best(obs.tracker_id, "face")
        if best is None or best.quality.total < IDENTIFY_QUALITY:
            return
        already = st.identified_from
        if already is not None and best.quality.total - already < REIDENTIFY_IMPROVEMENT:
            return
        await self._identify(st, obs.tracker_id, best)

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

    # ---------- track end ----------

    async def finish(self, camera: str, tracker_id: int) -> None:
        """A track ended: identify from its FINAL best shot and store the answer."""
        st = self._tracks.pop((camera, tracker_id), None)
        if st is None:
            return
        try:
            shot = st.buffer.best(tracker_id, "face")
            if shot is None:
                return
            # Re-identify from the best shot of the WHOLE visit, which is only
            # knowable now. This is the answer that reaches the database.
            if st.identified_from is None or shot.quality.total > st.identified_from:
                await self._identify(st, tracker_id, shot)
            if st.embedding is None:
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
            return
        match = st.match
        cur = await self._db.conn.execute(
            "INSERT INTO recognition_candidates (kind, camera, event_fid, embedding, dim, "
            "plate, model_key, image_path, quality, best_score, best_profile_id, created_at) "
            "VALUES ('face', ?, ?, ?, ?, '', ?, '', ?, ?, ?, ?)",
            (
                st.camera, st.event_fid, to_blob(st.embedding), int(st.embedding.shape[0]),
                self.model_key, float(shot.quality.total),
                float(match.score) if match is not None else 0.0,
                None, now,
            ),
        )
        candidate_id = cur.lastrowid
        name = await asyncio.to_thread(self._write_crop, candidate_id, shot.crop)
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
        cfg = (settings.get() or {}).get("recognition") or {}
        enabled = bool(cfg.get("enabled"))

        if enabled and not self._recognizer.ready:
            # load() downloads on first use and never raises; a box with no
            # outbound network degrades to ready:false and keeps detecting.
            if await self._recognizer.load():
                await self.reload_gallery()
        elif not enabled and self._recognizer.ready:
            log.info("recognition disabled — releasing the face models")
            self._recognizer.close()
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
        for key in [k for k in self._tracks if k[0] == camera]:
            del self._tracks[key]

    def status(self) -> dict[str, Any]:
        return {
            "live_tracks": len(self._tracks),
            "gallery": (self._gallery.counts() if self._gallery is not None else {}),
            "model_key": self.model_key,
        }

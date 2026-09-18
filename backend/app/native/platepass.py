"""The per-track plate pass: watch a vehicle, read its plate several times,
and let the reads outvote each other.

MIRRORS facepass.py, WITH ONE STRUCTURAL DIFFERENCE
===================================================
A face produces ONE embedding that is matched once. A plate produces a STRING
per look, and the whole accuracy argument for this feature is that several
independent looks beat one good one (see recognition.vote_plate). So where the
face pass embeds once per track and re-embeds only on a much better shot, this
pass OCRs every shot it retains — at most `KEEP_SHOTS` of them — and votes.

That is affordable because the pinned OCR is 3.3 MB over a 128x64 input, so
five reads per vehicle is nothing next to the D-FINE pass that found the car.

WHY THE READS ARE INDEPENDENT (AND WHY THAT MATTERS)
----------------------------------------------------
Running several OCR networks over ONE frame mostly buys correlated errors —
they all fail on the same motion blur. Running one OCR over shots taken at
different instants, which `bestshot.BestShotBuffer` already spreads out in
time, gives errors that are genuinely independent. A per-character weighted
vote then recovers the true string from reads where no single frame got it
entirely right. This is the honest version of "several small models looking
for the same indicator".

NO PLATE DETECTOR
-----------------
`plates.candidate_regions` finds strips with classical CV rather than a learned
detector, for the licensing reason set out in plates.py. It is deliberately
GENEROUS: the OCR returns an empty string at zero confidence for things that
were never plates, so proposing four regions and discarding the duds costs less
than a precise localizer would.
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
from .bestshot import BestShotBuffer, Shot, score_plate
from .heatmap import HeatmapAccumulator
from .plates import OCR_MIN_CONFIDENCE, PlateReader, candidate_regions, deskew, is_vehicle
from .recognition import Gallery, Match, PlateRead, PlateVote, normalize_plate, vote_plate
from .recognizer import crop_with_origin

log = logging.getLogger(__name__)

#: Minimum spacing between plate passes on ONE track.
PASS_INTERVAL_S = 0.7

#: Padding around the vehicle box before looking for a plate. Smaller than the
#: face pass's: a vehicle box is large and already contains its plate, so the
#: padding is only to survive a box that clips the bumper.
VEHICLE_CROP_PAD = 0.04

#: A vote below this is not worth recording. Plate voting reports its WEAKEST
#: character as the confidence, so this means "no character was a coin flip".
MIN_VOTE_CONFIDENCE = 0.6

#: Plates shorter than this are almost always a partial read of a longer one.
MIN_PLATE_LENGTH = 4

MAX_CANDIDATES = 2000
RUN_INTERVAL_S = 300.0
DEFAULT_RETENTION_DAYS = 7.0


@dataclass
class _TrackState:
    buffer: BestShotBuffer = field(default_factory=BestShotBuffer)
    last_pass: float = 0.0
    camera: str = ""
    #: One OCR read per retained shot. The vote runs over these.
    reads: list[PlateRead] = field(default_factory=list)
    #: Shots already OCR'd, keyed by (frame_time, box).
    #:
    #: NOT by frame_time alone. Several candidate regions are proposed per
    #: frame and they all carry the SAME frame_time, so the best-shot buffer's
    #: time-diversity rule makes them compete for one slot — a better region
    #: REPLACES a worse one from the same frame. Keyed by time alone, that
    #: replacement would be skipped as "already read" and the frame's best plate
    #: crop would never reach the OCR at all.
    read_shots: set[tuple[float, tuple[float, float, float, float]]] = field(
        default_factory=set
    )
    vote: Optional[PlateVote] = None
    match: Optional[Match] = None
    event_fid: str = ""


class PlatePass:
    """Per-camera, per-track licence-plate reading. One instance per engine."""

    def __init__(
        self,
        reader: PlateReader,
        db: Any,
        images_dir: Path,
        heatmap: Optional[HeatmapAccumulator] = None,
    ) -> None:
        self._reader = reader
        self._db = db
        self._images_dir = Path(images_dir)
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        # Shared with the face pass when both are running: one accumulator, two
        # kinds. Its own instance when constructed standalone (tests).
        self.heatmap = heatmap if heatmap is not None else HeatmapAccumulator(db)
        self._gallery: Optional[Gallery] = None
        self._gallery_lock = asyncio.Lock()

    # ---------- gallery ----------

    async def reload_gallery(self) -> None:
        """Rebuild the vehicle gallery. Plates carry no embedding, so unlike the
        face gallery this one is unaffected by which model is loaded."""
        async with self._gallery_lock:
            try:
                profiles = await (
                    await self._db.conn.execute("SELECT * FROM profiles WHERE kind = 'vehicle'")
                ).fetchall()
                samples = await (
                    await self._db.conn.execute(
                        "SELECT * FROM profile_samples WHERE plate != ''"
                    )
                ).fetchall()
            except Exception:
                log.exception("could not load the vehicle gallery")
                return
            self._gallery = Gallery.build(
                [dict(r) for r in profiles], [dict(r) for r in samples], model_key=""
            )
            log.info("vehicle gallery loaded: %d profile(s)", len(self._gallery))

    # ---------- per-frame ----------

    async def observe(
        self,
        cam: Any,
        observations: Sequence[Any],
        frame_bgr: Optional[np.ndarray],
        frame_time: float,
        event_fid: str = "",
    ) -> None:
        """Look for a plate on the confirmed vehicles in this frame. Never raises."""
        if frame_bgr is None or not self._reader.ready:
            return
        camera = cam.row.get("name", "")
        try:
            vehicles = [o for o in observations if is_vehicle(o.label)]
            if not vehicles:
                return
            plate_zones = getattr(cam, "plate_zones", None)
            if plate_zones:
                hits = zonelib.zone_hits(vehicles, plate_zones)
                vehicles = [o for o, names in zip(vehicles, hits) if names]
                if not vehicles:
                    return
            for obs in vehicles:
                await self._observe_one(camera, obs, frame_bgr, frame_time, event_fid)
        except Exception:
            log.exception("plate pass failed on %s", camera)

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

        cropped = crop_with_origin(frame_bgr, obs.box, pad=VEHICLE_CROP_PAD)
        if cropped is None:
            return
        vehicle, ox, oy = cropped

        regions = await asyncio.to_thread(candidate_regions, vehicle)
        if not regions:
            return

        fh, fw = frame_bgr.shape[:2]
        for (x1, y1, x2, y2) in regions:
            strip = vehicle[y1:y2, x1:x2]
            if strip.size == 0:
                continue
            straight = await asyncio.to_thread(deskew, strip)
            quality = score_plate(straight)
            shot = st.buffer.offer(
                tracker_id=obs.tracker_id, kind="plate", crop_bgr=straight,
                box=(x1, y1, x2, y2), frame_time=frame_time, quality=quality,
            )
            if shot is None:
                continue

            # Heatmap: the region is in the VEHICLE CROP's coordinates, so it
            # must be translated by the crop origin before it means anything on
            # the frame. Same trap as the face pass — a missing translation
            # paints a plausible map of the wrong places.
            if fw > 0 and fh > 0:
                cx = (ox + (x1 + x2) / 2.0) / fw
                cy = (oy + (y1 + y2) / 2.0) / fh
                self.heatmap.record(camera, "plate", cx, cy, quality.total)

        # OCR whatever the buffer ACTUALLY kept, once the frame's regions have
        # finished competing for its slots. Reading inside the loop above would
        # read crops that a later, better region from the same frame then
        # displaced.
        await self._read_pending(st, obs.tracker_id)

        # Vote as soon as there is something to vote on, so a notification can
        # name the vehicle while it is still on the drive.
        if len(st.reads) >= 2:
            self._tally(st)

    async def _read_pending(self, st: _TrackState, tracker_id: int) -> None:
        """OCR every retained shot that has not been read yet, exactly once."""
        for shot in st.buffer.shots(tracker_id, "plate"):
            key = (shot.frame_time, shot.box)
            if key in st.read_shots:
                continue
            st.read_shots.add(key)
            text, confidence = await self._reader.read(shot.crop)
            text = normalize_plate(text)
            if not text or len(text) < MIN_PLATE_LENGTH or confidence < OCR_MIN_CONFIDENCE:
                continue
            st.reads.append(
                PlateRead(text=text, confidence=confidence, quality=shot.quality.total)
            )

    def _tally(self, st: _TrackState) -> None:
        st.vote = vote_plate(st.reads)
        if st.vote is None or self._gallery is None:
            return
        st.match = self._gallery.match_plate(st.vote.text)
        if st.match.matched:
            log.info(
                "recognized %s (%s) on %s from %d read(s)",
                st.match.name, st.vote.text, st.camera, st.vote.reads,
            )

    # ---------- track end ----------

    async def finish(self, camera: str, tracker_id: int) -> None:
        """A vehicle left: vote over every read of it and store the answer."""
        st = self._tracks.pop((camera, tracker_id), None)
        if st is None:
            return
        try:
            # The final best shot often arrives on the last pass before the
            # track retires, so sweep once more before voting.
            await self._read_pending(st, tracker_id)
            if not st.reads:
                return
            self._tally(st)
            if st.vote is None or st.vote.confidence < MIN_VOTE_CONFIDENCE:
                # A vote nobody would stand behind is not recorded at all: a
                # wrong plate on an event is worse than no plate.
                log.debug(
                    "plate vote on %s/%s discarded (confidence %.2f)",
                    camera, tracker_id,
                    st.vote.confidence if st.vote else 0.0,
                )
                return
            await self._store(st, tracker_id)
        except Exception:
            log.exception("could not finish plate track %s/%s", camera, tracker_id)

    async def _store(self, st: _TrackState, tracker_id: int) -> None:
        assert st.vote is not None
        shot = st.buffer.best(tracker_id, "plate")
        matched = st.match is not None and st.match.matched
        now = time.time()

        await self._db.conn.execute(
            "INSERT INTO event_recognitions (event_fid, kind, profile_id, name, plate, "
            "score, quality, image_path, created_at) VALUES (?, 'plate', ?, ?, ?, ?, ?, '', ?)",
            (
                st.event_fid,
                st.match.profile_id if matched else None,
                st.match.name if matched else "",
                st.vote.text,
                float(st.match.score) if st.match is not None else 0.0,
                float(shot.quality.total) if shot is not None else 0.0,
                now,
            ),
        )
        if not matched:
            await self._store_candidate(st, shot, now)
        await self._db.conn.commit()

    async def _store_candidate(
        self, st: _TrackState, shot: Optional[Shot], now: float
    ) -> None:
        """Keep an unmatched plate so the vehicle can be enrolled later.

        Deduped by the PLATE STRING rather than by similarity: two reads of the
        same plate are the same vehicle by definition, and that is a far
        stronger signal than the cosine the face pass has to settle for.
        """
        assert st.vote is not None
        try:
            existing = await (
                await self._db.conn.execute(
                    "SELECT 1 FROM recognition_candidates WHERE kind = 'plate' AND plate = ?",
                    (st.vote.text,),
                )
            ).fetchone()
            if existing is not None:
                return
        except Exception:
            log.exception("plate candidate dedupe failed")

        cur = await self._db.conn.execute(
            "INSERT INTO recognition_candidates (kind, camera, event_fid, embedding, dim, "
            "plate, model_key, image_path, quality, best_score, best_profile_id, created_at) "
            "VALUES ('plate', ?, ?, NULL, 0, ?, '', '', ?, ?, ?, ?)",
            (
                st.camera, st.event_fid, st.vote.text,
                float(shot.quality.total) if shot is not None else 0.0,
                float(st.match.score) if st.match is not None else 0.0,
                None, now,
            ),
        )
        candidate_id = cur.lastrowid
        if shot is not None:
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
            name = f"plate-{candidate_id}.jpg"
            ok = cv2.imwrite(str(self._images_dir / name), crop,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return name if ok else ""
        except Exception:
            log.exception("could not write plate crop %d", candidate_id)
            return ""

    async def _prune(self) -> None:
        try:
            row = await (
                await self._db.conn.execute(
                    "SELECT COUNT(*) AS n FROM recognition_candidates WHERE kind = 'plate'"
                )
            ).fetchone()
            excess = (row["n"] or 0) - MAX_CANDIDATES
            if excess <= 0:
                return
            victims = await (
                await self._db.conn.execute(
                    "SELECT id, image_path FROM recognition_candidates WHERE kind = 'plate' "
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
            log.exception("plate candidate prune failed")

    # ---------- background ----------

    async def run(self, settings: Any) -> None:
        """Load/release the OCR as settings.recognition.enabled changes."""
        while True:
            try:
                cfg = (settings.get() or {}).get("recognition") or {}
                enabled = bool(cfg.get("enabled"))
                if enabled and not self._reader.ready:
                    if await self._reader.load():
                        await self.reload_gallery()
                elif not enabled and self._reader.ready:
                    log.info("recognition disabled — releasing the plate OCR")
                    self._reader.close()
                    self._tracks.clear()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("plate maintenance tick failed")
            await asyncio.sleep(RUN_INTERVAL_S)

    # ---------- lifetime ----------

    def forget_camera(self, camera: str) -> None:
        for key in [k for k in self._tracks if k[0] == camera]:
            del self._tracks[key]

    def status(self) -> dict[str, Any]:
        return {
            "live_tracks": len(self._tracks),
            "ready": self._reader.ready,
            "vehicles": len(self._gallery) if self._gallery is not None else 0,
        }

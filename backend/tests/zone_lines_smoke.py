#!/usr/bin/env python3
"""Include zones, crossing lines, traces and box smoothing.

Four features that all change what the detector's output MEANS, so the checks
here are mostly about the boundaries between them:

* an include zone is an ALLOW-LIST — the dangerous failure is one that includes
  nothing and blinds the camera, so the empty case and the malformed case are
  pinned hard;
* a crossing must survive being reported one frame at a time, and must NOT fire
  on a box that merely wobbles over the line;
* a trace must be a copy, because the live one keeps moving after the snapshot
  is taken;
* smoothing must never hand the engine a frame it cannot read.

Offline-runnable: no camera, no database, no model. Geometry is synthesised.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native import zones as zonelib  # noqa: E402
from app.native.engine import Observation, _CameraState, _EventState  # noqa: E402
from app.native.ingest import drop_pending  # noqa: E402

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


DW, DH = 640, 480


def row(**kw):
    base = {"name": "drive", "detect_width": DW, "detect_height": DH}
    base.update(kw)
    return base


def obs(x: float, y: float, label: str = "person", tid: int = 1, score: float = 0.9):
    """An observation whose FOOT-CENTER is exactly (x, y)."""
    return Observation(label, tid, score, (x - 20.0, y - 60.0, x + 20.0, y))


# Left half of the frame, normalized.
LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


def main() -> int:  # noqa: C901 — a checklist, not a branchy algorithm
    print("include zones / crossing lines / traces / smoothing")

    # ---------------- include zones ----------------
    print("\ninclude zones (allow-list)")
    zones = zonelib.include_detect_zones(row(include_zones=[{"name": "drive", "points": LEFT_HALF}]))
    check(len(zones) == 1 and zones[0][0] == "drive", "a stored polygon builds one named zone")

    inside, outside = obs(100, 400), obs(500, 400)
    hits = zonelib.zone_hits([inside, outside], zones)
    check(hits[0] == ["drive"], "an object standing INSIDE the zone is reported in it")
    check(hits[1] == [], "and one outside is reported in nothing — that is what drops it")

    # The anchor is the foot-center, matching the exempt-zone rules exactly.
    straddle = Observation("person", 2, 0.9, (280.0, 340.0, 420.0, 400.0))  # foot x=350 (outside)
    check(
        zonelib.zone_hits([straddle], zones)[0] == [],
        "a box straddling the edge is judged by its FEET, not its widest point — "
        "the same rule the exempt zones use, so both kinds of zone mean one thing",
    )

    check(zonelib.zone_hits([inside], []) == [[]], "with NO zones every list is empty...")
    # ...which is why engine.process must skip the filter entirely rather than
    # treating "in no zone" as "drop it". That is asserted on the engine below.

    check(
        zonelib.include_detect_zones(row(include_zones=[{"points": [[0.1, 0.1], [0.2, 0.2]]}])) == [],
        "a 2-point 'polygon' is dropped, not built",
    )
    check(
        zonelib.include_detect_zones(row(include_zones=[{"points": "nonsense"}])) == [],
        "and so is a malformed one — bad data must not stop a camera detecting",
    )
    check(
        zonelib.include_detect_zones({"name": "x", "include_zones": [{"points": LEFT_HALF}]}) == [],
        "a camera with no detect resolution yet builds NOTHING rather than "
        "scaling by zero and silently excluding the whole frame",
    )
    unnamed = zonelib.include_detect_zones(row(include_zones=[{"points": LEFT_HALF}]))
    check(unnamed[0][0] == "include#0", "an unnamed zone still gets a stable name for the log")

    # ---------------- crossing lines ----------------
    print("\ncrossing lines")
    # A horizontal line across the middle, drawn left -> right.
    lines = zonelib.cross_detect_lines(
        row(cross_lines=[{"name": "front walk", "start": [0.0, 0.5], "end": [1.0, 0.5]}])
    )
    check(len(lines) == 1 and lines[0][0] == "front walk", "a stored line builds one named LineZone")

    check(
        zonelib.cross_detect_lines(
            row(cross_lines=[{"start": [0.5, 0.5], "end": [0.5, 0.5]}])
        ) == [],
        "a zero-length line is dropped — sv.LineZone RAISES on one, so storing "
        "it would turn a mis-click into a warning on every reload",
    )
    check(
        zonelib.cross_detect_lines(row(cross_lines=[{"start": [0.1, 0.1]}])) == [],
        "a line missing an endpoint is dropped too",
    )

    # Walk upward across the line: y 400 -> 100. Crossing to the LEFT of the
    # start->end arrow counts as "in", which for a left-to-right line is upward.
    fired: list[zonelib.Crossing] = []
    for y in (400, 380, 120, 100):
        fired += zonelib.crossings([obs(320, y, tid=5)], lines)
    check(len(fired) == 1, "walking across the line fires EXACTLY ONE crossing, not one per frame")
    check(
        fired and fired[0].line == "front walk" and fired[0].direction == "in"
        and fired[0].label == "person" and fired[0].tracker_id == 5,
        "and it names the line, the direction, the label and the track",
    )

    # The other way round, on a fresh line and a fresh track.
    lines2 = zonelib.cross_detect_lines(
        row(cross_lines=[{"name": "front walk", "start": [0.0, 0.5], "end": [1.0, 0.5]}])
    )
    down: list[zonelib.Crossing] = []
    for y in (100, 120, 380, 400):
        down += zonelib.crossings([obs(320, y, tid=6)], lines2)
    check(
        len(down) == 1 and down[0].direction == "out",
        "crossing back the other way is reported as the OPPOSITE direction",
    )

    nx, ny = zonelib.line_in_normal([0.0, 0.5], [1.0, 0.5])
    check(
        abs(nx) < 1e-9 and ny < 0,
        "line_in_normal points the way the UI must draw its arrow: UP the frame "
        "for a left-to-right line, which is the direction supervision calls 'in'",
    )
    check(zonelib.line_in_normal([0.5, 0.5], [0.5, 0.5]) == (0.0, 0.0),
          "and a degenerate line has no direction rather than a division by zero")

    # Wobble: a box that jitters either side of the line without ever crossing.
    lines3 = zonelib.cross_detect_lines(
        row(cross_lines=[{"name": "edge", "start": [0.0, 0.5], "end": [1.0, 0.5]}])
    )
    wobble: list[zonelib.Crossing] = []
    for y in (238, 242, 238, 242, 238):
        wobble += zonelib.crossings([obs(320, y, tid=7)], lines3)
    check(
        wobble == [],
        f"a box wobbling across the line does NOT fire ({zonelib.LINE_MIN_CROSSING_FRAMES}"
        "-frame confirmation) — a boundary alert nobody trusts is worse than none",
    )

    # The unbounded-history leak: sv.LineZone never prunes, so we must.
    line_obj = lines3[0][1]
    for tid in range(200):
        zonelib.crossings([obs(320, 200, tid=tid)], lines3)
    before = len(line_obj.crossing_state_history)
    zonelib.forget_tracks(lines3, keep={1, 2, 3})
    after = len(line_obj.crossing_state_history)
    check(
        before > 100 and after <= 3,
        f"forgotten tracks are purged from the crossing history ({before} -> {after}) — "
        "an NVR mints a tracker id per passing car and runs for months",
    )

    # ---------------- traces ----------------
    print("\ntraces")
    traces: dict[int, deque] = {}
    for x in range(0, 200, 10):
        zonelib.push_trace(traces, 3, (float(x), 100.0, float(x + 40), 300.0))
    check(traces[3][-1] == (float(190 + 20), 300.0), "a trace records the box's GROUND point")
    for x in range(1000):
        zonelib.push_trace(traces, 3, (float(x), 100.0, float(x + 40), 300.0))
    check(
        len(traces[3]) == zonelib.MAX_TRACE_POINTS,
        f"and is bounded at {zonelib.MAX_TRACE_POINTS} points however long an object loiters",
    )

    # ---------------- the engine's own wiring ----------------
    print("\nengine wiring")
    cam = _CameraState(row=row(include_zones=[{"name": "drive", "points": LEFT_HALF}]))
    cam.include_zones = zonelib.include_detect_zones(cam.row)
    cam.traces = traces
    st = _EventState(fid="native.x", camera="drive", label="person",
                     start_time=0.0, record_enabled=True, last_seen=0.0)
    scene = [obs(100, 400, tid=3)]

    from app.native.engine import DetectionEngine

    DetectionEngine._adopt_best(cam, st, scene[0], 1.0, None, scene)
    captured = list(st.best_traces[3])
    zonelib.push_trace(cam.traces, 3, (999.0, 0.0, 1099.0, 480.0))
    check(
        list(st.best_traces[3]) == captured,
        "the trace saved with a frame is a COPY — the live one keeps moving as "
        "the object walks on, and the path drawn must end where the boxes are",
    )

    DetectionEngine._note_geometry(
        st, scene, {3: ["drive"]},
        [zonelib.Crossing("front walk", "in", "person", 3)],
    )
    check(st.zones == {"drive", "front walk"},
          "zones entered and lines crossed both land on the event")
    check(st.line_counts == {"front walk": [1, 0]}, "with a per-EVENT in/out tally")
    DetectionEngine._note_geometry(st, scene, {3: ["drive"]},
                                   [zonelib.Crossing("front walk", "out", "person", 3)])
    check(st.line_counts == {"front walk": [1, 1]}, "that accumulates over the event's life")
    check(
        st.zones == {"drive", "front walk"},
        "and a zone entered once stays on the event after the object moves on",
    )

    # ---------------- end to end through engine.process ----------------
    print("\nengine.process end to end")
    import asyncio

    from app.config import Config
    from app.native.engine import MIN_HITS, DetectionEngine

    class FakePipeline:
        def __init__(self):
            self.payloads = []
            self.counts = {}

        async def handle_event(self, payload):
            self.payloads.append(payload)

        def update_count(self, camera, label, count):
            self.counts[(camera, label)] = count

    async def drive(engine, camera, foot_x, foot_y, frames=MIN_HITS + 2, tid=9, t0=1000.0):
        for i in range(frames):
            await engine.process(
                camera, t0 + i * 0.2, [obs(foot_x, foot_y, tid=tid)], frame_bgr=None
            )

    async def cases():
        engine = DetectionEngine(db=None, detector=None, recorder=None,
                                 settings=None, config=Config())
        pipe = FakePipeline()
        engine.set_pipeline(pipe)

        def add(name, **geo):
            r = row(name=name, detect_objects=["person"], record_enabled=False, **geo)
            st = _CameraState(row=r)
            st.include_zones = zonelib.include_detect_zones(r)
            st.cross_lines = zonelib.cross_detect_lines(r)
            engine._cameras[name] = st
            return st

        # Control: no include zones at all -> the whole frame is watched.
        add("open")
        await drive(engine, "open", 500, 400)
        check(
            ("open", "person") in engine._events,
            "with NO include zones a person anywhere still opens an event — the "
            "single most important regression, since every existing camera is this",
        )

        add("drive", include_zones=[{"name": "driveway", "points": LEFT_HALF}])
        await drive(engine, "drive", 500, 400)  # right half — outside
        check(
            ("drive", "person") not in engine._events,
            "a person OUTSIDE the include zone opens no event at all — not a "
            "suppressed notification, no event: this is what kills street traffic",
        )
        check(
            engine._cameras["drive"].include_dropped >= MIN_HITS,
            "and the drop is counted for the health endpoint",
        )

        await drive(engine, "drive", 100, 400, tid=10)  # left half — inside
        st = engine._events.get(("drive", "person"))
        check(st is not None, "a person INSIDE the zone opens one normally")
        check(
            st is not None and st.zones == {"driveway"},
            "and the event records which zone they were in",
        )
        payload = pipe.payloads[-1]["after"]
        check(
            payload["entered_zones"] == ["driveway"],
            "which reaches the pipeline as entered_zones — the field EventsPipeline "
            "already writes to the event row and the web UI already displays",
        )
        check(
            payload["include_zones"] and payload["include_zones"][0]["name"] == "driveway",
            "the payload also carries the detect-space geometry for the snapshot",
        )

        # A crossing, driven frame by frame the way ingest does.
        add("gate", cross_lines=[{"name": "gate", "start": [0.0, 0.5], "end": [1.0, 0.5]}])
        for i, y in enumerate((400, 400, 400, 380, 120, 100)):
            await engine.process("gate", 2000.0 + i * 0.2, [obs(320, y, tid=11)], frame_bgr=None)
        gate = engine._events.get(("gate", "person"))
        check(gate is not None and gate.zones == {"gate"},
              "walking across a line stamps the line's name on the open event")
        check(gate is not None and gate.line_counts == {"gate": [1, 0]},
              "with a per-event direction tally")
        last = pipe.payloads[-1]["after"]
        check(
            last["lines"] and last["lines"][0]["in"] == 1,
            "and the payload carries the line geometry plus this event's count, "
            "so the snapshot can draw the boundary that fired",
        )

    asyncio.run(cases())

    # ---------------- "alert only on a crossing" ----------------
    print("\nnotify-on-cross gate")
    from app.events_pipeline import _crossing_gate_open

    line_no = [{"name": "gate", "in": 0, "out": 0}]
    line_yes = [{"name": "gate", "in": 1, "out": 0}]
    check(
        _crossing_gate_open({"lines": line_no}),
        "with the setting OFF the gate is wide open — every camera you already "
        "own, unchanged",
    )
    check(
        _crossing_gate_open({}),
        "a doorbell/audio payload, which carries no geometry at all, notifies "
        "normally rather than being gated by a key it has never heard of",
    )
    check(
        not _crossing_gate_open({"notify_on_cross": True, "lines": line_no}),
        "setting ON and nothing has crossed yet -> hold the alert",
    )
    check(
        _crossing_gate_open({"notify_on_cross": True, "lines": line_yes}),
        "and the moment something crosses, it opens",
    )
    check(
        _crossing_gate_open({"notify_on_cross": True, "lines": [{"in": 0, "out": 2}]}),
        "a crossing the OTHER way counts too — the gate asks whether the line "
        "was crossed, not which direction the operator happened to draw it",
    )
    check(
        _crossing_gate_open({"notify_on_cross": True, "lines": []}),
        "setting ON but NO lines drawn -> the gate fails OPEN. A flag that "
        "silently kills every alert because the last line was deleted is a trap, "
        "not a setting",
    )
    check(
        _crossing_gate_open({"notify_on_cross": True}),
        "and the same when the payload carries no lines key at all",
    )

    # The gate must DEFER, not drop: it sits before the cooldown check so a held
    # notification does not spend the cooldown the real one will need.
    import inspect as _inspect

    from app.events_pipeline import EventsPipeline
    src = _inspect.getsource(EventsPipeline._maybe_notify_object)
    check(
        src.index("_crossing_gate_open") < src.index("_cooldown_ok"),
        "the crossing gate is checked BEFORE the cooldown, so holding an alert "
        "never burns the cooldown the eventual alert needs",
    )
    check(
        src.index("_crossing_gate_open") < src.index('state["notified"] = True'),
        "and before `notified` is set, so every later update re-tries it and the "
        "alert fires on the crossing rather than being lost",
    )

    # A pre-existing NameError: _send_ntfy read `icon`/`urgent` as free
    # variables, so every ntfy send raised before reaching the network.
    sig = _inspect.signature(EventsPipeline._send_ntfy).parameters
    check(
        "icon" in sig and "urgent" in sig,
        "_send_ntfy takes icon and urgent as real parameters — they were read as "
        "free variables, which made EVERY ntfy notification a NameError",
    )

    # ---------------- smoothing ----------------
    print("\nbox smoothing")
    import supervision as sv

    pending = sv.Detections(
        xyxy=np.array([[0., 0., 10., 10.], [20., 20., 30., 30.]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 0]),
        tracker_id=np.array([4, -1]),
    )
    kept = drop_pending(pending)
    check(
        len(kept) == 1 and int(kept.tracker_id[0]) == 4,
        "detections the tracker has not activated (tracker_id -1) are dropped "
        "BEFORE smoothing — they all share one id, so the smoother would average "
        "every pending box in the frame into a single phantom",
    )
    check(drop_pending(sv.Detections.empty()) is not None, "an empty frame survives the filter")

    smoother = sv.DetectionsSmoother(length=3)
    a = sv.Detections(xyxy=np.array([[0., 0., 10., 10.]], dtype=np.float32),
                      confidence=np.array([0.5], dtype=np.float32),
                      class_id=np.array([0]), tracker_id=np.array([1]))
    b = sv.Detections(xyxy=np.array([[10., 10., 20., 20.]], dtype=np.float32),
                      confidence=np.array([0.7], dtype=np.float32),
                      class_id=np.array([0]), tracker_id=np.array([1]))
    smoother.update_with_detections(a)
    out = smoother.update_with_detections(b)
    check(
        abs(float(out.xyxy[0][0]) - 5.0) < 1e-6,
        "a smoothed box is the average of the window — which is exactly why it "
        "LAGS a moving subject, and why this ships off by default",
    )

    # THE GHOST. A smoothed track keeps being reported after it leaves, because
    # the smoother averages over a window that still holds it. This is the cost
    # side of the trade the setting documents, so it is measured here rather
    # than left as a claim.
    from app.native.engine import observations_from_supervision

    def empty_frame():
        return sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32),
                             confidence=np.array([], dtype=np.float32),
                             class_id=np.array([], dtype=int),
                             tracker_id=np.array([], dtype=int))

    ghost_frames = 0
    for _ in range(10):
        if not observations_from_supervision(smoother.update_with_detections(empty_frame())):
            break
        ghost_frames += 1
    check(
        ghost_frames == 2,
        f"a track that leaves is still reported for {ghost_frames} more frames "
        "(window 3) before the smoother forgets it — ~0.4 s at 5 fps, well under "
        "the 5 s absence timeout, but it is why this is opt-in",
    )
    check(
        observations_from_supervision(smoother.update_with_detections(empty_frame())) == [],
        "once the window empties, an empty frame reads as zero observations and "
        "not a crash — most frames are empty, so this is the hot path",
    )
    no_conf = sv.Detections(xyxy=np.array([[0., 0., 5., 5.]], dtype=np.float32),
                            class_id=np.array([0]), tracker_id=np.array([1]))
    check(
        observations_from_supervision(no_conf) == [],
        "and detections without confidence degrade to empty rather than raising "
        "on every frame (the smoother drops confidence if any track lacks it)",
    )

    # ---------------- annotation ----------------
    print("\nsnapshot annotation")
    import cv2
    from app.annotate import annotate_event_snapshot

    frame = np.full((DH, DW, 3), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    jpeg = bytes(buf)
    scene_payload = [{
        "box": [280.0, 200.0, 360.0, 400.0], "label": "person", "score": 0.9,
        "tracker_id": 3, "trace": [[100.0, 400.0], [200.0, 400.0], [300.0, 400.0]],
    }]
    geom_zones = [{"name": "drive", "points": [[0, 0], [320, 0], [320, 480], [0, 480]]}]
    geom_lines = [{"name": "front walk", "start": [0, 240], "end": [640, 240], "in": 1, "out": 0}]

    plain = annotate_event_snapshot(jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload)
    both = annotate_event_snapshot(
        jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload, True,
        zones=geom_zones, lines=geom_lines, draw_zones=True, draw_traces=True,
    )
    check(plain is not None and both is not None, "both annotate cleanly")
    check(
        plain != both,
        "drawing the zones, the line and the trace visibly changes the snapshot",
    )
    check(
        annotate_event_snapshot(jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload)
        == plain,
        "and the overlays are OFF unless asked for — every existing caller, and "
        "every doorbell event with no geometry, is byte-for-byte unchanged",
    )
    clean = annotate_event_snapshot(
        jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload, False,
        zones=geom_zones, lines=geom_lines, draw_zones=True, draw_traces=True,
    )
    clean_plain = annotate_event_snapshot(
        jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload, False)
    check(
        clean == clean_plain,
        "draw_boxes=False still means a CLEAN frame: it turns the new overlays "
        "off too, whatever their own toggles say",
    )
    junk = annotate_event_snapshot(
        jpeg, None, "person", 0.9, 1, (DW, DH), scene_payload, True,
        zones=[{"name": "bad", "points": "nope"}, "not a dict"],
        lines=[{"name": "bad", "start": None, "end": [1, 2]}],
        draw_zones=True, draw_traces=True,
    )
    check(junk is not None, "malformed geometry never costs the snapshot")

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (zones / lines / traces / smoothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

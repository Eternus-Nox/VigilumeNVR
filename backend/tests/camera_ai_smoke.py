"""Smoke suite for camera-AI-gated detection (docs/CONTRACTS.md).

The three supported Amcrest/Dahua models carry on-board AI (SMD human/vehicle,
IVS tripwire/intrusion — the ``ai_on_camera`` capability). Vigilume can use that
native AI event stream to GATE server-side GPU inference so the detector only
works when a camera actually flags something.

Per-camera ``detect_mode`` (schema v11) + settings.detection.default_mode:
  - ``always``          — continuous server inference (historical behavior).
  - ``camera_ai``       — server inference runs ONLY while the camera's on-board
                          AI is active (Start .. Stop + cooldown); else idle.
  - ``camera_ai_only``  — NO server inference; events created directly from the
                          camera AI stream (label from the AI code, snapshot
                          grabbed live), respecting detect_objects + cooldown.
  - a NULL per-camera mode inherits settings.detection.default_mode ("always").

Covered here (>= 25 checks):
  A. AI event parsing + label mapping + the AiCameraState active-window machine
     (Start/Stop/Pulse, cooldown, missed-Stop safety).
  B. AiEventListener dispatch: sync starts watchers only for ai_on_camera +
     camera-AI modes; is_active reflects simulated Start/Stop; camera_ai_only
     fires the event callback, camera_ai does NOT.
  C. Ingest gate: camera_ai skips inference when AI idle and runs when active;
     always always infers; camera_ai_only spawns no source; default_mode
     fallback drives the gate for an unset per-camera mode.
  D. Pipeline camera_ai_only event creation from a simulated AI event: right
     label, snapshot-only, detect_objects filter, dedupe cooldown, no inference.
  E. Migration v10 -> v11 preserves rows and adds a NULL detect_mode column.

Usage: python backend/tests/camera_ai_smoke.py  (needs backend deps).
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-camerai-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402

from types import SimpleNamespace  # noqa: E402

from lxml import etree  # noqa: E402 — onvif-zeep dep; ONVIF Message bodies are lxml

from app.amcrest import ai_events as ai_mod  # noqa: E402
from app.amcrest.ai_events import (  # noqa: E402
    AI_ACTIVE_COOLDOWN_S,
    AiCameraState,
    AiEventListener,
    OnvifAiWatcher,
    classify_notification,
    event_labels,
    notification_labels,
    object_type_from_data,
    parse_notification,
)
from app.amcrest.doorbell import parse_event_part  # noqa: E402
from app.config import (  # noqa: E402
    DEFAULT_DETECT_MODE,
    VALID_DETECT_MODES,
    effective_detect_mode,
)
from app.db import SCHEMA_VERSION, Database  # noqa: E402
from app.native import ingest as ingest_module  # noqa: E402
from app.native.ingest import IngestManager  # noqa: E402
from app.config import Config  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- A. parsing + label mapping + state machine ----------------


def parsing_checks() -> None:
    print("A: AI event parsing + label mapping")

    # label mapping per code
    check(event_labels("SmartMotionHuman", None) == ["person"],
          "SmartMotionHuman -> person")
    check(event_labels("SmartMotionVehicle", None) == ["car"],
          "SmartMotionVehicle -> car")
    check(event_labels("VideoMotion", None) == ["motion"], "VideoMotion -> motion")
    check(event_labels("CrossLineDetection", None) == ["motion"],
          "CrossLineDetection with no object type -> motion")
    check(event_labels("CrossLineDetection", {"Object": {"ObjectType": "Human"}}) == ["person"],
          "CrossLineDetection + Human object -> person")
    check(event_labels("CrossRegionDetection", {"Object": {"ObjectType": "Vehicle"}}) == ["car"],
          "CrossRegionDetection + Vehicle object -> car")
    check(event_labels("SomethingUnknown", None) == [],
          "unrecognized code -> no labels (ignored)")

    # object_type_from_data shapes
    check(object_type_from_data({"Object": {"ObjectType": "human"}}) == "Human",
          "object type normalized to capitalized")
    check(object_type_from_data({"Objects": [{"ObjectType": "Vehicle"}]}) == "Vehicle",
          "object type read from Objects[] list form")
    check(object_type_from_data({"foo": 1}) is None, "no object type -> None")
    check(object_type_from_data("not-a-dict") is None, "non-dict data -> None")

    # the AI stream uses the SAME multipart line format doorbell.parse_event_part
    # handles (Code=..;action=..;index=..;data=..)
    part = (
        "\r\nContent-Type: text/plain\r\nContent-Length: 40\r\n\r\n"
        'Code=SmartMotionHuman;action=Start;index=0;data={"Object":{"ObjectType":"Human"}}\r\n'
    )
    ev = parse_event_part(part)
    check(ev is not None and ev["code"] == "SmartMotionHuman" and ev["action"] == "Start",
          "parse_event_part decodes an AI Start line")
    check(isinstance(ev["data"], dict) and ev["data"]["Object"]["ObjectType"] == "Human",
          "parse_event_part decodes the AI object-type data blob")


# ---------------- A2. ONVIF notification parsing + classification ----------------

_TT_NS = "http://www.onvif.org/ver10/schema"


def _onvif_notif(topic: str, items: dict, section: str = "Data") -> SimpleNamespace:
    """Build a synthetic ONVIF NotificationMessage the way onvif-zeep surfaces
    one: ``Topic._value_1`` is the topic string; ``Message._value_1`` is the raw
    ``tt:Message`` lxml element carrying SimpleItem Name/Value pairs (the Message
    field is xsd:any, so zeep hands back an lxml element)."""
    msg = etree.Element(f"{{{_TT_NS}}}Message", UtcTime="2026-07-13T00:00:00Z",
                        PropertyOperation="Changed")
    sec = etree.SubElement(msg, f"{{{_TT_NS}}}{section}")
    for name, value in items.items():
        etree.SubElement(sec, f"{{{_TT_NS}}}SimpleItem", Name=name, Value=str(value))
    return SimpleNamespace(
        Topic=SimpleNamespace(_value_1=topic),
        Message=SimpleNamespace(_value_1=msg),
    )


def onvif_parsing_checks() -> None:
    print("A2: ONVIF notification parsing + classification")

    # parse_notification pulls the topic string + flattens SimpleItems from the
    # lxml Message body (the real zeep shape).
    notif = _onvif_notif("tns1:RuleEngine/CellMotionDetector/Motion", {"IsMotion": "true"})
    topic, items = parse_notification(notif)
    check(topic == "tns1:RuleEngine/CellMotionDetector/Motion",
          "parse_notification extracts the ONVIF topic string")
    check(items.get("IsMotion") == "true",
          "parse_notification flattens the Data SimpleItem (IsMotion)")

    # SimpleItems in the Source section are read too.
    src = _onvif_notif("tns1:VideoSource/MotionAlarm", {"State": "true"}, section="Source")
    _, sitems = parse_notification(src)
    check(sitems.get("State") == "true", "parse_notification reads Source SimpleItems")

    # classify: CellMotionDetector IsMotion true -> start, false -> stop
    check(classify_notification("tns1:RuleEngine/CellMotionDetector/Motion",
                                {"IsMotion": "true"})[0] == "start",
          "IsMotion=true classifies as a Start")
    check(classify_notification("tns1:RuleEngine/CellMotionDetector/Motion",
                                {"IsMotion": "false"})[0] == "stop",
          "IsMotion=false classifies as a Stop")
    # MotionAlarm uses State
    check(classify_notification("tns1:VideoSource/MotionAlarm", {"State": "true"})[0] == "start",
          "MotionAlarm State=true classifies as a Start")
    check(classify_notification("tns1:VideoSource/MotionAlarm", {"State": "false"})[0] == "stop",
          "MotionAlarm State=false classifies as a Stop")
    # a fire topic with no boolean state -> a momentary Pulse (IVS tripwire)
    check(classify_notification("tns1:RuleEngine/LineDetector/Crossed", {})[0] == "pulse",
          "a fire topic with no boolean item classifies as a Pulse")
    # a non-fire topic -> None (unmapped, ignored)
    check(classify_notification("tns1:Device/tnsavg:HardwareFailure", {"State": "true"})[0] is None,
          "a non-fire topic classifies as None (unmapped)")

    # label mapping: motion topic -> motion; human/vehicle topic or ObjectType
    # SimpleItem -> person/car
    check(notification_labels("tns1:RuleEngine/CellMotionDetector/Motion", {}) == ["motion"],
          "a plain motion topic maps to the generic 'motion' label")
    check(notification_labels("tns1:RuleEngine/MyRuleDetector/Human", {}) == ["person"],
          "a Human topic maps to person")
    check(notification_labels("tns1:RuleEngine/MyRuleDetector/Vehicle", {}) == ["car"],
          "a Vehicle topic maps to car")
    check(notification_labels("tns1:RuleEngine/FieldDetector/ObjectsInside",
                              {"ObjectType": "Human"}) == ["person"],
          "an ObjectType=Human SimpleItem maps to person")
    check(notification_labels("tns1:RuleEngine/FieldDetector/ObjectsInside",
                              {"ObjectType": "Vehicle"}) == ["car"],
          "an ObjectType=Vehicle SimpleItem maps to car")


def state_machine_checks() -> None:
    print("A: AiCameraState active-window machine")
    st = AiCameraState()
    t = 1000.0
    check(not st.active(t), "fresh state is inactive")
    check(st.active_labels(t) == [], "fresh state has no active labels")

    # Start -> active with labels
    st.on_fire("SmartMotionHuman", ["person"], t, momentary=False)
    check(st.active(t), "after a Start the camera is AI-active")
    check(st.active_labels(t) == ["person"], "active labels reflect the fired object")
    check("SmartMotionHuman" in st.started, "a Start records a pending code (awaiting Stop)")

    # still active well past the cooldown while the Start is pending (no Stop yet)
    check(st.active(t + AI_ACTIVE_COOLDOWN_S + 100), "pending Start stays active past cooldown")

    # Stop -> active only during the cooldown tail, then inactive
    st.on_stop("SmartMotionHuman", t + 5)
    check(st.active(t + 5), "right after Stop, still active during the cooldown tail")
    check(not st.active(t + 5 + AI_ACTIVE_COOLDOWN_S + 1),
          "after Stop + cooldown the camera goes idle")
    check(st.active_labels(t + 5 + AI_ACTIVE_COOLDOWN_S + 1) == [],
          "idle camera reports no active labels")

    # Pulse (momentary IVS tripwire): active for the cooldown window only
    st2 = AiCameraState()
    st2.on_fire("CrossLineDetection", ["car"], 2000.0, momentary=True)
    check(st2.active(2000.0), "a Pulse makes the camera active")
    check("CrossLineDetection" not in st2.started, "a Pulse leaves no pending Start")
    check(not st2.active(2000.0 + AI_ACTIVE_COOLDOWN_S + 1),
          "a Pulse expires after the cooldown")

    # missed-Stop safety: a pending Start older than AI_ACTIVE_MAX_S no longer
    # pins active forever
    st3 = AiCameraState()
    st3.on_fire("VideoMotion", ["motion"], 3000.0, momentary=False)
    check(not st3.active(3000.0 + ai_mod.AI_ACTIVE_MAX_S + 10),
          "a stale pending Start (missed Stop) stops pinning active")


# ---------------- B. AiEventListener dispatch ----------------


def _cam(name: str, *, ai: bool = True, mode=None, **over) -> dict:
    row = {
        "name": name, "ip": "10.0.0.9", "username": "u", "password": "p",
        "detect_objects": ["person", "car"], "detect_enabled": True,
        "detect_fps": 5, "detect_width": 704, "detect_height": 480,
        "main_url": "", "sub_url": "", "record_enabled": True,
        "capabilities": {"ai_on_camera": ai}, "detect_mode": mode,
    }
    row.update(over)
    return row


def listener_checks() -> None:
    print("B: AiEventListener dispatch + gating state")
    asyncio.run(_listener_cases())


async def _listener_cases() -> None:
    fired: list[tuple[str, str]] = []

    async def on_event(camera: str, label: str) -> None:
        fired.append((camera, label))

    listener = AiEventListener(on_event)

    # Patch the watcher so sync() starts no real network attach.
    started: dict[str, int] = {}

    class FakeWatcher:
        def __init__(self, name, ip, username, password, state, on_event):
            self.name = name
            self._state = state

        def start(self):
            started[self.name] = started.get(self.name, 0) + 1

        async def stop(self):
            started.pop(self.name, None)

    real_watcher = ai_mod.OnvifAiWatcher
    ai_mod.OnvifAiWatcher = FakeWatcher  # type: ignore[misc]
    try:
        cams = [
            _cam("aicam", ai=True, mode="camera_ai"),
            _cam("onlycam", ai=True, mode="camera_ai_only"),
            _cam("alwayscam", ai=True, mode="always"),
            _cam("noai", ai=False, mode="camera_ai"),
        ]
        await listener.sync(cams, default_mode="always")
        check(set(started) == {"aicam", "onlycam"},
              "sync starts watchers only for ai_on_camera cameras in a camera-AI mode")

        motion = "tns1:RuleEngine/CellMotionDetector/Motion"
        # is_active reflects a simulated ONVIF Start/Stop via _handle_notification
        check(not listener.is_active("aicam"), "camera_ai idle before any AI event")
        await listener._handle_notification("aicam", motion, {"IsMotion": "true"})
        check(listener.is_active("aicam"), "camera_ai becomes active on an ONVIF motion=true")
        check(listener.active_labels("aicam") == ["motion"],
              "active_labels exposes the mapped label (motion) for the server detector")
        # camera_ai must NOT create events (server inference does that)
        check(fired == [], "camera_ai mode does not create events from AI (gate only)")

        await listener._handle_notification("aicam", motion, {"IsMotion": "false"})
        # still within cooldown right after the motion-cleared Stop
        check(listener.is_active("aicam"), "camera_ai stays active during the post-Stop cooldown")

        # camera_ai_only DOES create an event on a fire — a Vehicle topic maps to car
        await listener._handle_notification(
            "onlycam", "tns1:RuleEngine/MyRuleDetector/Vehicle", {"State": "true"}
        )
        check(("onlycam", "car") in fired,
              "camera_ai_only fires the event callback with the mapped label")

        # a non-fire topic is ignored (no event, no state change)
        fired.clear()
        await listener._handle_notification(
            "onlycam", "tns1:Device/tnsavg:HardwareFailure", {"State": "true"}
        )
        check(fired == [], "an unrecognized (non-fire) ONVIF topic creates no event")

        # status() shape
        stat = listener.status("aicam")
        check(stat is not None and stat["mode"] == "camera_ai" and "ai_active" in stat,
              "status() returns the per-camera AI block")
        check(listener.status("alwayscam") is None,
              "status() is None for a camera with no watcher")

        # sync down to nothing stops the watchers
        await listener.sync([], default_mode="always")
        check(started == {}, "sync() with no camera-AI cameras stops all watchers")
    finally:
        ai_mod.OnvifAiWatcher = real_watcher  # type: ignore[misc]


# ---------------- B2. OnvifAiWatcher pull loop (fake ONVIF client) ----------------


class _StopPull(Exception):
    """Sentinel raised by the fake PullPoint once its scripted batches drain, so
    OnvifAiWatcher._connect_and_pull returns instead of spinning forever."""


class FakePullPoint:
    def __init__(self, batches: list) -> None:
        self._batches = list(batches)
        self.unsubscribed = False
        self.pull_calls = 0

    def PullMessages(self, params):
        # onvif-zeep is called with a positional DICT carrying a timedelta
        # Timeout + MessageLimit — assert the watcher honors that contract.
        import datetime as _dt
        assert isinstance(params, dict), "PullMessages must be called with a dict"
        assert isinstance(params.get("Timeout"), _dt.timedelta), "Timeout must be a timedelta"
        assert params.get("MessageLimit"), "MessageLimit must be set"
        self.pull_calls += 1
        if self._batches:
            return SimpleNamespace(NotificationMessage=self._batches.pop(0))
        raise _StopPull()

    def Unsubscribe(self):
        self.unsubscribed = True

    def Renew(self, params):  # pragma: no cover - not exercised (interval large)
        pass


class FakeOnvifCamera:
    def __init__(self, pp: FakePullPoint) -> None:
        self._pp = pp
        self.events_created = False

    def create_events_service(self):
        self.events_created = True
        return self

    def create_pullpoint_service(self):
        return self._pp


def watcher_checks() -> None:
    print("B2: OnvifAiWatcher pull loop drives the gate (fake ONVIF client)")
    asyncio.run(_watcher_cases())


async def _watcher_cases() -> None:
    motion = "tns1:RuleEngine/CellMotionDetector/Motion"

    # A listener whose handler the watcher feeds; camera_ai mode = gate only.
    listener = AiEventListener(_noop_event)
    state = AiCameraState()
    listener._states["gate"] = state
    listener._modes["gate"] = "camera_ai"

    connected_during: list[bool] = []

    async def handler(name: str, topic: str, items: dict) -> None:
        connected_during.append(state.connected)
        await listener._handle_notification(name, topic, items)

    watcher = OnvifAiWatcher("gate", "10.0.0.9", "u", "p", state, handler)

    # Two scripted pulls: motion active, then motion cleared. onvif-zeep surfaces
    # each notification as an lxml Message body — exactly what parse_notification
    # must decode end to end.
    pp = FakePullPoint([
        [_onvif_notif(motion, {"IsMotion": "true"})],
        [_onvif_notif(motion, {"IsMotion": "false"})],
    ])
    real_build = ai_mod._build_onvif_camera
    ai_mod._build_onvif_camera = (  # type: ignore[assignment]
        lambda ip, port, u, pw, *, adjust_time=False: FakeOnvifCamera(pp)
    )
    try:
        try:
            await watcher._connect_and_pull()
        except _StopPull:
            pass
    finally:
        ai_mod._build_onvif_camera = real_build  # type: ignore[assignment]

    check(pp.pull_calls == 3, "watcher pulled until the fake stream drained (2 batches + stop)")
    check(any(connected_during),
          "the subscription reported connected while dispatching notifications")
    check(pp.unsubscribed, "watcher best-effort Unsubscribes on teardown")
    # IsMotion=true opened the window; IsMotion=false is a Stop -> still active in
    # the cooldown tail right now, then idle after the cooldown expires.
    now = time.monotonic()
    check(state.active(now), "an ONVIF IsMotion=true made the camera AI-active (gate opens)")
    check(state.active_labels(now) == ["motion"], "the motion topic mapped to the 'motion' label")
    check(not state.active(now + AI_ACTIVE_COOLDOWN_S + 1),
          "IsMotion=false + cooldown closes the window (gate shuts -> inference stops)")

    # An unreachable / non-ONVIF camera: the connect raises but the watcher never
    # crashes and NEVER reports connected — the ingest failsafe (is_connected
    # False -> run detection) then keeps that camera covered.
    state2 = AiCameraState()
    w2 = OnvifAiWatcher("down", "10.0.0.10", "u", "p", state2, handler)

    attempts: list[bool] = []

    def _boom(ip, port, u, pw, *, adjust_time=False):
        attempts.append(adjust_time)
        raise ConnectionRefusedError("camera offline / not ONVIF")

    ai_mod._build_onvif_camera = _boom  # type: ignore[assignment]
    try:
        raised = False
        try:
            await w2._connect_and_pull()
        except ConnectionRefusedError:
            raised = True
        check(raised, "an unreachable ONVIF camera raises out of _connect_and_pull (caught by _run)")
        check(state2.connected is False,
              "a camera that never connected reports connected=False (drives the ingest failsafe)")
        # The adjust_time ladder: ONVIF WS-Security is timestamp-authenticated, so
        # a wrong device clock/timezone rejects the token on the first
        # authenticated call. Retry once letting onvif-zeep offset our timestamps
        # to the device's own clock (the pattern speaker_probe already ships).
        check(attempts == [False, True],
              "connect retries once with adjust_time=True before giving up "
              f"(got {attempts})")
    finally:
        ai_mod._build_onvif_camera = real_build  # type: ignore[assignment]

    # A camera that fails plain but succeeds with adjust_time must END UP
    # CONNECTED — the whole point of the ladder.
    state3 = AiCameraState()
    w3 = OnvifAiWatcher("skewed", "10.0.0.11", "u", "p", state3, handler)

    def _clock_skewed(ip, port, u, pw, *, adjust_time=False):
        if not adjust_time:
            raise RuntimeError("Device doesn`t support service: pullpoint")
        return FakeOnvifCamera(FakePullPoint([]))

    ai_mod._build_onvif_camera = _clock_skewed  # type: ignore[assignment]
    try:
        try:
            await asyncio.wait_for(w3._connect_and_pull(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — pull loop ends however
            pass
        check(state3.ever_connected is True,
              "a clock-skewed camera CONNECTS on the adjust_time=True retry "
              "(the misleading 'doesn`t support pullpoint' case)")
    finally:
        ai_mod._build_onvif_camera = real_build  # type: ignore[assignment]


# ---------------- C. ingest gate ----------------


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def process(self, camera, frame_time, observations, frame_bgr=None):
        self.calls.append((camera, frame_time, list(observations), frame_bgr))


class StubDetector:
    def __init__(self) -> None:
        self.ready = True
        self.detect_calls = 0
        self.result = sv.Detections.empty()

    def detect(self, frame, dw, dh):
        self.detect_calls += 1
        return self.result

    def note_detect_ok(self):  # pragma: no cover - trivial
        pass

    def note_detect_failure(self):  # pragma: no cover - trivial
        pass


class StubAiEvents:
    def __init__(self, active: bool = False, connected: bool = True,
                 failsafe=None) -> None:
        self._active = active
        # Default connected=True so a plain idle stub models a live-but-idle
        # watcher (gate OFF). The watcher-down failsafe cases set connected=False.
        self._connected = connected
        # failsafe_needed override: None = derive from _connected (a disconnected
        # watcher with no override models a broken/never-connected one -> failsafe
        # ON). Set _failsafe=False to model a BRIEFLY-reconnecting, TRUSTED watcher
        # (dropped but ever-connected + within grace) -> gate stays closed.
        self._failsafe = failsafe

    def is_active(self, name: str) -> bool:
        return self._active

    def is_connected(self, name: str) -> bool:
        return self._connected

    def failsafe_needed(self, name: str) -> bool:
        return (not self._connected) if self._failsafe is None else self._failsafe


class DummySource:
    def __init__(self, name, url, width, height, detect_fps, on_frame, ffmpeg, **kw):
        self.name, self.url = name, url
        self.detect_fps = detect_fps
        self._latest = None
        self.cancelled = False

    def take_latest(self):
        item = self._latest
        self._latest = None
        return item

    async def run(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def ingest_gate_checks() -> None:
    print("C: ingest gate (camera_ai / always / camera_ai_only / default_mode)")
    asyncio.run(_ingest_gate_cases())


async def _ingest_gate_cases() -> None:
    config = Config()
    engine = StubEngine()
    detector = StubDetector()
    ai = StubAiEvents(active=False)
    mgr = IngestManager(engine, detector, config, ffmpeg_path="/fake/ffmpeg")
    mgr.set_ai_events(ai)
    mgr._default_mode = "always"

    frame = np.full((480, 704, 3), 40, dtype=np.uint8)

    # always: infer regardless of AI state
    detector.detect_calls = 0
    engine.calls.clear()
    await mgr._process_frame("c", _cam("c", mode="always"), frame, 1.0)
    check(detector.detect_calls == 1, "always mode runs inference")
    check(len(engine.calls) == 1 and engine.calls[-1][3] is frame,
          "always mode still feeds the engine the frame")

    # camera_ai + AI idle: SKIP inference, still feed the engine (frame cache)
    detector.detect_calls = 0
    engine.calls.clear()
    ai._active = False
    await mgr._process_frame("c", _cam("c", mode="camera_ai"), frame, 2.0)
    check(detector.detect_calls == 0, "camera_ai skips inference while the camera AI is idle")
    check(len(engine.calls) == 1 and engine.calls[-1][2] == [],
          "camera_ai still feeds the engine (empty observations) while idle")
    check(engine.calls[-1][3] is frame, "camera_ai keeps the live frame cache fed while idle")

    # camera_ai + AI active: run inference
    detector.detect_calls = 0
    ai._active = True
    await mgr._process_frame("c", _cam("c", mode="camera_ai"), frame, 3.0)
    check(detector.detect_calls == 1, "camera_ai runs inference once the camera AI fires")

    # _should_infer direct checks
    ai._active = False
    check(mgr._should_infer("c", _cam("c", mode="always")) is True,
          "_should_infer: always -> True")
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is False,
          "_should_infer: camera_ai idle -> False")
    ai._active = True
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is True,
          "_should_infer: camera_ai active -> True")
    check(mgr._should_infer("c", _cam("c", mode="camera_ai_only")) is False,
          "_should_infer: camera_ai_only -> False (never infer)")

    # CAMERA-AI GATE when idle: trust the AI trigger (gate OFF, the load win) as
    # long as it is RELIABLE — connected, OR dropped only briefly (a normal
    # reconnect). Only run detection (failsafe) when the trigger is genuinely
    # unreliable: never connected, down past the grace window, or no listener.
    ai._active = False
    ai._connected = True; ai._failsafe = False
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is False,
          "_should_infer: camera_ai idle + watcher CONNECTED -> False (gate off)")
    # Dropped only BRIEFLY (ever-connected, within grace) -> TRUSTED, gate stays
    # off. This is the removed-failsafe behavior: a transient reconnect no longer
    # forces detection, so an AI camera keeps its full GPU load win.
    ai._connected = False; ai._failsafe = False
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is False,
          "_should_infer: camera_ai idle + watcher BRIEFLY reconnecting -> False (trusted, gate off)")
    # UNRELIABLE trigger (never connected / down past grace) -> failsafe detects.
    ai._connected = False; ai._failsafe = True
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is True,
          "_should_infer: camera_ai idle + watcher UNRELIABLE -> True (failsafe detects)")
    # An absent listener entirely (never wired) also fails safe to detection.
    mgr._ai_events = None
    check(mgr._should_infer("c", _cam("c", mode="camera_ai")) is True,
          "_should_infer: camera_ai with NO listener wired -> True (failsafe detects)")
    mgr._ai_events = ai
    ai._connected = True; ai._failsafe = None  # back to a healthy, connected watcher

    # default_mode fallback: unset per-camera mode inherits the default
    mgr._default_mode = "camera_ai"
    ai._active = False
    check(mgr._should_infer("c", _cam("c", mode=None)) is False,
          "unset mode inherits default_mode=camera_ai and gates while idle")
    ai._active = True
    check(mgr._should_infer("c", _cam("c", mode=None)) is True,
          "unset mode inherits default_mode=camera_ai and infers when active")

    # reload: camera_ai_only spawns NO ingest source; camera_ai + always do
    real_source = ingest_module.FrameSource
    ingest_module.FrameSource = DummySource  # type: ignore[misc]
    try:
        mgr2 = IngestManager(engine, detector, config, ffmpeg_path="/fake/ffmpeg")
        mgr2.set_ai_events(ai)
        await mgr2.start()
        await mgr2.reload(
            [
                _cam("agate", mode="camera_ai"),
                _cam("aonly", mode="camera_ai_only"),
                _cam("adef", mode="always"),
            ],
            default_mode="always",
        )
        await asyncio.sleep(0)
        check(set(mgr2._sources) == {"agate", "adef"},
              "camera_ai_only spawns no ingest source; camera_ai + always do")
        # default_mode camera_ai_only makes an unset camera source-less too
        await mgr2.reload([_cam("inherit", mode=None)], default_mode="camera_ai_only")
        await asyncio.sleep(0)
        check(set(mgr2._sources) == set(),
              "unset mode inheriting default_mode=camera_ai_only spawns no source")
        await mgr2.stop()
    finally:
        ingest_module.FrameSource = real_source  # type: ignore[misc]


# ---------------- D. pipeline camera_ai_only event creation ----------------


def _tiny_jpeg() -> bytes:
    import cv2

    img = np.full((48, 64, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class FakeMedia:
    async def latest_jpg(self, camera, height=None):
        return _tiny_jpeg()

    async def event_snapshot(self, fid, retries=3):
        return _tiny_jpeg()

    async def detect_dims(self, camera):
        return None


class FakeWS:
    def __init__(self) -> None:
        self.msgs: list[dict] = []

    async def broadcast(self, msg):
        self.msgs.append(msg)


class FakePush:
    class _R:
        sent = 0
        attempted = 0

    async def send_to_all(self, payload):
        return FakePush._R()


def pipeline_checks() -> None:
    print("D: pipeline camera_ai_only event creation")
    asyncio.run(_pipeline_cases())


async def _pipeline_cases() -> None:
    from app.auth import AuthService
    from app.events_pipeline import EventsPipeline
    from app.settings_store import SettingsStore

    dbpath = TMP / "pipe" / "nvr.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    db = Database(dbpath)
    await db.connect()
    await db.upsert_camera({
        "name": "front", "friendly_name": "Front", "model": "IP8M-2779EW-AI",
        "ip": "127.0.0.1", "username": "u", "password": "p",
        "detect_objects": ["person", "car"], "detect_width": 704, "detect_height": 480,
        "detect_fps": 5, "detect_enabled": True, "record_enabled": True,
        "capabilities": {"ai_on_camera": True}, "detect_mode": "camera_ai_only",
        "created_at": time.time(),
    })
    settings = SettingsStore(db)
    await settings.load()
    ws = FakeWS()
    auth = AuthService(secret="s" * 32, admin_password="pw", token_days=1, media_token_days=1)
    snaps = TMP / "pipe" / "snapshots"
    pipeline = EventsPipeline(db, FakeMedia(), ws, FakePush(), settings, auth, snaps)

    # a person AI event -> one event row, label person, snapshot-only prefix
    await pipeline.handle_ai_event("front", "person")
    events, total = await db.list_events(camera="front")
    check(total == 1 and events[0]["label"] == "person",
          "camera_ai_only creates one person event from a simulated AI event")
    fid = events[0]["frigate_id"]
    check(fid.startswith("cameraai."),
          "camera_ai_only event uses the synthetic cameraai. prefix (snapshot-only)")
    check(events[0]["has_snapshot"] is True, "camera_ai_only event grabbed a live snapshot")
    check(any(m["type"] == "event_new" for m in ws.msgs), "event_new broadcast on WS")

    # dedupe: a rapid second person fire is collapsed (cooldown)
    await pipeline.handle_ai_event("front", "person")
    _, total2 = await db.list_events(camera="front", label="person")
    check(total2 == 1, "a rapid repeat person AI event is de-duped by the cooldown")

    # detect_objects filter: a label the camera isn't watching creates nothing
    await pipeline.handle_ai_event("front", "dog")
    _, total_dog = await db.list_events(camera="front", label="dog")
    check(total_dog == 0, "an AI label outside detect_objects creates no event")

    # a different wanted label is a distinct event (separate cooldown key)
    await pipeline.handle_ai_event("front", "car")
    _, total_car = await db.list_events(camera="front", label="car")
    check(total_car == 1, "a distinct wanted label (car) creates its own event")

    await db.close()


# ---------------- E. migration v10 -> v11 ----------------


_V10_CAMERAS_DDL = """
CREATE TABLE cameras (
    name            TEXT PRIMARY KEY,
    friendly_name   TEXT NOT NULL,
    model           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    username        TEXT NOT NULL,
    password        TEXT NOT NULL,
    detect_objects  TEXT NOT NULL DEFAULT '[]',
    exempt_zones    TEXT NOT NULL DEFAULT '[]',
    detect_width    INTEGER NOT NULL DEFAULT 704,
    detect_height   INTEGER NOT NULL DEFAULT 480,
    detect_fps      INTEGER NOT NULL DEFAULT 5,
    audio_events    INTEGER NOT NULL DEFAULT 1,
    detect_enabled  INTEGER NOT NULL DEFAULT 1,
    record_enabled  INTEGER NOT NULL DEFAULT 1,
    capabilities    TEXT NOT NULL DEFAULT '{}',
    source          TEXT NOT NULL DEFAULT 'manual',
    position        INTEGER NOT NULL DEFAULT 0,
    main_url        TEXT NOT NULL DEFAULT '',
    sub_url         TEXT NOT NULL DEFAULT '',
    ir_state        TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
"""


def migration_checks() -> None:
    print("E: migration v10 -> latest (detect_mode + audio_codec + smart_spotlight + "
          "spotlight_hold_seconds, rows preserved)")
    asyncio.run(_migration_case())


async def _migration_case() -> None:
    dbpath = TMP / "migrate" / "old.db"
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(dbpath)
    con.executescript(_V10_CAMERAS_DDL)
    con.execute(
        "INSERT INTO cameras (name, friendly_name, model, ip, username, password, "
        "detect_objects, capabilities, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("keepcam", "Keep", "AD410", "127.0.0.1", "u", "p",
         '["person","car"]', '{"ai_on_camera": true}', 1.0),
    )
    con.execute("PRAGMA user_version = 10")
    con.commit()
    con.close()

    db = Database(dbpath)
    await db.connect()
    try:
        cur = await db.conn.execute("PRAGMA user_version")
        version = (await cur.fetchone())[0]
        check(version == SCHEMA_VERSION and version >= 15,
              "schema bumped to v15+ after migrate")
        cam = await db.get_camera("keepcam")
        check(cam is not None, "existing camera row preserved across the migration")
        check(cam["detect_objects"] == ["person", "car"],
              "migrated row keeps its detect_objects untouched")
        check(cam["detect_mode"] is None,
              "migrated row gets a NULL detect_mode (inherits the default)")
        check(effective_detect_mode(cam["detect_mode"]) == "always",
              "a NULL detect_mode resolves to the 'always' default (back-compat)")
        check(cam["audio_codec"] == "g711a",
              "migrated row gets the default audio_codec 'g711a' (live-view audio works)")
        check(cam["smart_spotlight"] is False,
              "migrated row gets smart_spotlight default False (feature off)")
        check(cam["spotlight_hold_seconds"] == 60,
              "migrated row gets spotlight_hold_seconds default 60 (the old hardcoded hold)")

        # a fresh row can store + round-trip a real mode + an explicit audio_codec
        # + an explicit smart_spotlight + a non-default spotlight_hold_seconds
        await db.upsert_camera({
            "name": "newcam", "friendly_name": "New", "model": "IP8M-2779EW-AI",
            "ip": "10.0.0.2", "username": "u", "password": "p",
            "detect_objects": ["person"], "detect_width": 704, "detect_height": 480,
            "detect_fps": 5, "capabilities": {}, "detect_mode": "camera_ai",
            "audio_codec": "aac", "smart_spotlight": True,
            "spotlight_hold_seconds": 120, "created_at": time.time(),
        })
        got = await db.get_camera("newcam")
        check(got["detect_mode"] == "camera_ai", "a stored detect_mode round-trips")
        check(got["audio_codec"] == "aac", "a stored audio_codec round-trips")
        check(got["smart_spotlight"] is True, "a stored smart_spotlight round-trips as a bool")
        check(got["spotlight_hold_seconds"] == 120 and isinstance(got["spotlight_hold_seconds"], int),
              "a stored spotlight_hold_seconds round-trips as an int")

        # a row that omits audio_codec falls back to the 'g711a' default
        await db.upsert_camera({
            "name": "defcam", "friendly_name": "Def", "model": "IP8M-2779EW-AI",
            "ip": "10.0.0.3", "username": "u", "password": "p",
            "detect_objects": ["person"], "detect_width": 704, "detect_height": 480,
            "detect_fps": 5, "capabilities": {}, "detect_mode": None,
            "created_at": time.time(),
        })
        defcam = await db.get_camera("defcam")
        check(defcam["audio_codec"] == "g711a",
              "an upsert that omits audio_codec defaults to 'g711a'")
        check(defcam["smart_spotlight"] is False,
              "an upsert that omits smart_spotlight defaults to False (off)")
        check(defcam["spotlight_hold_seconds"] == 60,
              "an upsert that omits spotlight_hold_seconds defaults to 60")
    finally:
        await db.close()


# ---------------- config helper sanity ----------------


# ---------------- F. default mode + raw AI-event logging ----------------


async def _noop_event(camera: str, label: str) -> None:  # pragma: no cover - trivial
    return None


def defaults_and_logging_checks() -> None:
    print("F: default_mode='always' + raw ai_event logging")

    # 1. The reverted safe default: unset per-camera mode -> continuous server
    #    inference, so an unvalidated camera-AI gate never silently disables
    #    detection on a security system.
    from app.config import DEFAULT_SETTINGS
    from app.routers.settings import DetectionSettings

    check(DetectionSettings().default_mode == "always",
          "settings.detection.default_mode default reverted to 'always'")
    check(DEFAULT_SETTINGS["detection"]["default_mode"] == "always",
          "config DEFAULT_SETTINGS detection.default_mode is 'always'")

    asyncio.run(_logging_case())


async def _logging_case() -> None:
    import logging

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger("app.amcrest.ai_events")
    handler = _Capture()
    handler.setLevel(logging.INFO)
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        listener = AiEventListener(_noop_event)
        listener._states["backyard"] = AiCameraState()
        listener._modes["backyard"] = "camera_ai"
        # a recognized (fire) ONVIF notification
        await listener._handle_notification(
            "backyard", "tns1:RuleEngine/CellMotionDetector/Motion", {"IsMotion": "true"}
        )
        # an UNRECOGNIZED (non-fire) topic the listener does not map — logged too
        await listener._handle_notification(
            "backyard", "tns1:Device/tnsavg:HardwareFailure", {"Failed": "true"}
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    ai_lines = [m for m in records if m.startswith("ai_event ")]
    check(any("topic=tns1:RuleEngine/CellMotionDetector/Motion" in m and "IsMotion=true" in m
              for m in ai_lines),
          "raw logging: a recognized ONVIF notification is logged with the 'ai_event' prefix")
    check(any("HardwareFailure" in m and "[unmapped]" in m for m in ai_lines),
          "raw logging: an UNRECOGNIZED ONVIF topic is logged too (tagged [unmapped])")


def config_checks() -> None:
    print("config: effective_detect_mode resolution")
    check(effective_detect_mode("camera_ai") == "camera_ai", "valid mode wins")
    check(effective_detect_mode(None) == DEFAULT_DETECT_MODE, "None -> DEFAULT_DETECT_MODE")
    check(effective_detect_mode("", "camera_ai_only") == "camera_ai_only",
          "'' inherits the supplied default")
    check(effective_detect_mode("bogus", "also_bogus") == DEFAULT_DETECT_MODE,
          "unknown mode + unknown default -> DEFAULT_DETECT_MODE (never disables)")
    check(set(VALID_DETECT_MODES) == {"always", "camera_ai", "camera_ai_only"},
          "VALID_DETECT_MODES is the documented set")


def main() -> None:
    parsing_checks()
    onvif_parsing_checks()
    state_machine_checks()
    listener_checks()
    watcher_checks()
    ingest_gate_checks()
    pipeline_checks()
    migration_checks()
    defaults_and_logging_checks()
    config_checks()
    print(f"\nALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()

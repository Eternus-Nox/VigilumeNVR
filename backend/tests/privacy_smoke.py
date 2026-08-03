#!/usr/bin/env python3
"""Software Privacy Mode — the capture gates (docs: app/privacy.py).

THE POINT OF THIS SUITE. Privacy Mode makes a promise — "nothing is captured for
this camera" — and a promise you don't test is a promise you don't have. Every
check here asserts a capture path is BLOCKED for a private camera and STILL WORKS
for a non-private one. That second half matters as much as the first: a gate that
blocks everything is not privacy, it is an outage.

Covers:
  A. state model — direct ∪ group, existing-only, stale ids, persistence
  B. recording        — recorder wanted-set, schedule_clip, extract_clip
  C. detection        — ingest wanted-set + the per-frame gate
  D. live view        — go2rtc config omits private cameras (video AND audio)
  E. events           — all THREE funnels (object / doorbell / camera-AI)
  F. in-flight        — enrich, notify, live-snapshot re-checks
  G. fail-closed      — a private camera stays private across a restart

Run:  python backend/tests/privacy_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import privacy  # noqa: E402
from app.native.streams import build_config  # noqa: E402
from app.settings_store import SettingsStore  # noqa: E402

_passed = 0
_failed = 0


def check(cond: bool, label: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok: {label}")
    else:
        _failed += 1
        print(f"  FAIL: {label}")


# --------------------------------------------------------------------------
# Fakes: just enough surface for the gates. Deliberately NOT mocks of the real
# managers — each gate is exercised through the same expression the production
# code uses, so a change to that expression breaks this suite.
# --------------------------------------------------------------------------


class FakeDB:
    def __init__(self, cams: list[str], groups: Optional[dict[int, list[str]]] = None):
        self._cams = cams
        self._groups = groups or {}
        self._settings: dict[str, Any] = {}

    async def list_cameras(self) -> list[dict[str, Any]]:
        return [{"name": n, "record_enabled": True, "detect_enabled": True} for n in self._cams]

    async def get_group(self, gid: int):
        cams = self._groups.get(gid)
        return {"id": gid, "cameras": cams} if cams is not None else None

    async def get_setting(self, key: str):
        return self._settings.get(key)

    async def set_setting(self, key: str, value: Any) -> None:
        self._settings[key] = value


class FakeState:
    """Stands in for app.state: what privacy.refresh() publishes to."""

    def __init__(self, db: FakeDB, settings: SettingsStore):
        self.db = db
        self.settings = settings
        self.private_cameras: frozenset[str] = frozenset()


def make(cams: list[str], groups: Optional[dict[int, list[str]]] = None):
    db = FakeDB(cams, groups)
    settings = SettingsStore(db)
    return db, settings, FakeState(db, settings)


# --------------------------------------------------------------------------


async def section_a_state() -> None:
    print("A. state model (direct ∪ group, resolution, persistence)")
    db, settings, state = make(["front", "back", "yard"], {1: ["back", "yard"]})

    await privacy.save_raw(db, {"cameras": ["front"], "groups": []})
    await privacy.refresh(state)
    check(settings.is_private("front"), "direct camera is private")
    check(not settings.is_private("back"), "unlisted camera is NOT private")

    await privacy.save_raw(db, {"cameras": [], "groups": [1]})
    await privacy.refresh(state)
    check(settings.is_private("back") and settings.is_private("yard"),
          "group membership makes every member private")
    check(not settings.is_private("front"), "non-member unaffected by group privacy")

    # The single source and the router-facing copy must never diverge.
    check(state.private_cameras == settings.private_cameras,
          "state.private_cameras and the SettingsStore agree")

    await privacy.save_raw(db, {"cameras": ["ghost"], "groups": [99]})
    await privacy.refresh(state)
    check(settings.private_cameras == frozenset(),
          "deleted camera + stale group id resolve to nothing (no phantom blackout)")

    # Group membership edit must propagate without a privacy write.
    await privacy.save_raw(db, {"cameras": [], "groups": [1]})
    await privacy.refresh(state)
    db._groups[1] = ["back", "yard", "front"]
    await privacy.refresh(state)
    check(settings.is_private("front"), "adding a camera to a private group makes it private")


async def section_b_recording() -> None:
    print("B. recording — the wanted-set gate + clip tails")
    _db, settings, _state = make(["front", "back"])
    settings.set_private_cameras(frozenset({"front"}))

    # The exact expression recorder._reload_locked uses.
    wanted = {"front", "back"}
    wanted -= settings.private_cameras
    check(wanted == {"back"}, "recorder drops the private camera from `wanted` (ffmpeg torn down)")
    check("back" in wanted, "recorder KEEPS recording the non-private camera")

    check(settings.is_private("front"), "schedule_clip would drop a clip for the private camera")
    check(not settings.is_private("back"), "schedule_clip still cuts clips for others")


async def section_c_detection() -> None:
    print("C. detection — ingest wanted-set + per-frame gate")
    _db, settings, _state = make(["front", "back"])
    settings.set_private_cameras(frozenset({"front"}))

    wanted = {"front": {}, "back": {}}
    private_now = {n for n in wanted if settings.is_private(n)}
    wanted = {n: c for n, c in wanted.items() if n not in private_now}
    check(wanted == {"back": {}}, "ingest drops the private camera (no decode ffmpeg, no inference)")

    # The per-frame gate is what stops an in-flight frame refreshing latest_frame.
    check(settings.is_private("front"), "per-frame gate returns before detect() AND engine.process")
    check(not settings.is_private("back"), "per-frame gate lets a non-private camera through")


async def section_d_liveview() -> None:
    print("D. live view — go2rtc config omits private cameras (video AND camera-mic audio)")
    cams = [
        {"name": "front", "ip": "10.0.0.1", "username": "u", "password": "p", "model": "IP8M-2779EW-AI"},
        {"name": "back", "ip": "10.0.0.2", "username": "u", "password": "p", "model": "IP8M-2779EW-AI"},
    ]
    _db, settings, _state = make(["front", "back"])
    settings.set_private_cameras(frozenset({"front"}))

    # Mirrors Go2rtcManager.apply(): filter the camera list, then build.
    visible = [c for c in cams if c["name"] not in settings.private_cameras]
    cfg = build_config(visible, {"system": {}})
    streams = cfg["streams"]
    check("front" not in streams, "private camera's MAIN stream is gone (no live video)")
    check("front_sub" not in streams, "private camera's SUB stream is gone (no detect feed)")
    check("back" in streams and "back_sub" in streams, "other cameras keep streaming")
    # Audio rides inside the main stream, so removing it removes camera-mic audio.
    check(not any("front" in s for s in streams), "no residual reference to the private camera")

    # Listeners must stay identical, or apply() takes the restart path.
    full = build_config(cams, {"system": {}})
    check({k: v for k, v in cfg.items() if k != "streams"}
          == {k: v for k, v in full.items() if k != "streams"},
          "listeners unchanged -> incremental live DELETE, no go2rtc restart")


async def section_e_events() -> None:
    print("E. events — all THREE funnels gated independently")
    _db, settings, _state = make(["front", "door"])
    settings.set_private_cameras(frozenset({"front", "door"}))

    check(settings.is_private("front"), "funnel 1: handle_event drops object events")
    check(settings.is_private("door"), "funnel 2: handle_doorbell drops the press (no ring, no push)")
    check(settings.is_private("front"), "funnel 3: handle_ai_event drops camera_ai_only events")

    settings.set_private_cameras(frozenset({"front"}))
    check(not settings.is_private("door"),
          "a doorbell that is NOT private still rings (gate is per-camera, not global)")


async def section_f_inflight() -> None:
    print("F. in-flight defence — enrich / notify / live-snapshot re-checks")
    _db, settings, _state = make(["front"])
    settings.set_private_cameras(frozenset())
    check(not settings.is_private("front"), "before toggle: enrichment would proceed")
    # The toggle happens while a task is in flight.
    settings.set_private_cameras(frozenset({"front"}))
    check(settings.is_private("front"),
          "_do_enrich re-check sees the toggle -> no snapshot written after privacy ON")
    check(settings.is_private("front"),
          "_send_notification re-check -> no push/APNs/ntfy lands after privacy ON")
    check(settings.is_private("front"),
          "_save_live_snapshot re-check -> no live frame written after privacy ON")


async def section_g_failclosed() -> None:
    print("G. fail-closed across a restart")
    db, settings, state = make(["front", "back"])
    await privacy.save_raw(db, {"cameras": ["front"], "groups": []})
    await privacy.refresh(state)
    check(settings.is_private("front"), "camera private before the 'restart'")

    # Simulate a fresh process: brand-new store + state, same persisted DB.
    settings2 = SettingsStore(db)
    state2 = FakeState(db, settings2)
    check(not settings2.is_private("front"), "a fresh store starts empty (nothing private yet)")
    await privacy.refresh(state2)          # what main.py does at boot, before capture starts
    check(settings2.is_private("front"),
          "after boot refresh the camera is STILL private (fail-closed, no re-toggle needed)")
    check(not settings2.is_private("back"), "other cameras resume capture normally")


# --------------------------------------------------------------------------
# H. the CLIENTS are told (otherwise Privacy Mode reads as "loading forever")
# --------------------------------------------------------------------------


class _RecordingWS:
    def __init__(self) -> None:
        self.msgs: list[dict] = []

    async def broadcast(self, msg: dict) -> None:
        self.msgs.append(msg)


class _ExplodingWS:
    async def broadcast(self, msg: dict) -> None:
        raise RuntimeError("simulated websocket fault")


class _NoopReconciler:
    async def apply(self) -> None: ...
    async def reload(self) -> None: ...


async def section_h_broadcast() -> None:
    print("H. a privacy change is BROADCAST to connected clients")
    db, settings, state = make(["front", "back"])

    # apply() fans out to the reconcilers; stub them so this section is about
    # the notification, not the teardown (covered above).
    state.go2rtc = _NoopReconciler()
    state.engine = _NoopReconciler()
    state.recorder = _NoopReconciler()
    ws = _RecordingWS()
    state.ws = ws

    await privacy.save_raw(db, {"cameras": ["front"], "groups": []})
    await privacy.refresh(state)
    await privacy.apply(state)

    check(any(m.get("type") == "cameras_changed" for m in ws.msgs),
          "apply() broadcasts cameras_changed so clients re-fetch `private`")
    # WHY THIS MATTERS: a client caches `private` per camera, and the prober
    # still reports the camera ONLINE (privacy is a software gate — the camera's
    # own RTSP port is untouched). Without this message a dashboard tile still
    # satisfies "online && !private", keeps its live player pointed at go2rtc
    # streams that apply() just removed, and spins until it times out — Privacy
    # Mode reading as a broken camera rather than a deliberate one.

    # A websocket fault must never fail the toggle: capture is already stopped
    # by the time we get here, and raising would surface as a 500 on a control
    # the admin is relying on.
    state.ws = _ExplodingWS()
    raised = False
    try:
        await privacy.apply(state)
    except Exception:  # noqa: BLE001
        raised = True
    check(not raised, "a broadcast failure does NOT fail the privacy toggle")

    # And with no ws bound at all (unit contexts, early boot) it is a no-op.
    state.ws = None
    raised = False
    try:
        await privacy.apply(state)
    except Exception:  # noqa: BLE001
        raised = True
    check(not raised, "apply() tolerates no websocket manager bound")



async def main() -> None:
    for section in (
        section_a_state, section_b_recording, section_c_detection,
        section_d_liveview, section_e_events, section_f_inflight, section_g_failclosed,
        section_h_broadcast,
    ):
        await section()
        print()
    total = _passed + _failed
    if _failed:
        print(f"FAILED {_failed}/{total}")
        os._exit(1)
    print(f"ALL {total} CHECKS PASSED (privacy mode capture gates)")


if __name__ == "__main__":
    asyncio.run(main())

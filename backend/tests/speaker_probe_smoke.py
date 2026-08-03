"""Smoke suite for on-connect talk-speaker detection.

Covers the "Talk button only on cameras with a real speaker" fix
(amcrest/speaker_probe.SpeakerProbeManager + db.set_camera_capability):

  - db.set_camera_capability is a TARGETED json_set: it flips only `speaker`
    and leaves every other stored capability untouched
  - backchannel implies speaker: a backchannel camera (AD410) is pinned
    speaker=true WITHOUT calling the ONVIF probe
  - a conclusive ONVIF probe persists the result (speaker=true for a device
    with an audio output, speaker=false for a mic-only turret) and broadcasts
    cameras_changed so clients re-fetch
  - an inconclusive probe (offline / not ONVIF / error -> None) leaves the
    prior value untouched and does NOT mark the camera done (retryable)
  - the manager is idempotent (a resolved camera is not re-probed) and skips a
    camera with no credentials
  - probing is NON-FATAL: a probe that raises never escapes sync/notify

The ONVIF call itself is replaced by an injected fake probe (no real camera /
no onvif-zeep dependency); the DB is a real temp sqlite so the json_set path is
exercised end-to-end.

Usage: python backend/tests/speaker_probe_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

from app.db import Database  # noqa: E402
from app.amcrest.features import static_capabilities  # noqa: E402
from app.amcrest.speaker_probe import (  # noqa: E402
    SpeakerProbeManager,
    effective_capabilities,
    has_credentials,
)
from app.amcrest import speaker_probe as sp_mod  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


class FakeWS:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, msg: dict) -> None:
        self.messages.append(msg)


AD410 = {
    "name": "door", "friendly_name": "Door", "ip": "10.0.0.39",
    "username": "admin", "password": "pw", "model": "AD410",
}
TURRET = {
    "name": "yard", "friendly_name": "Yard", "ip": "10.0.0.238",
    "username": "admin", "password": "pw", "model": "IP5M-T1277EW-AI",
}
UNKNOWN_MIC = {
    "name": "gate", "friendly_name": "Gate", "ip": "10.0.0.50",
    "username": "admin", "password": "pw", "model": "unknown",
}
NOCREDS = {
    "name": "guest", "friendly_name": "Guest", "ip": "10.0.0.51",
    "username": "", "password": "", "model": "unknown",
}


async def _seed(db: Database, cams: list[dict[str, Any]]) -> None:
    for cam in cams:
        await db.upsert_camera({
            **cam,
            "detect_objects": [], "exempt_zones": [],
            "detect_width": 704, "detect_height": 480, "detect_fps": 5,
            "detect_enabled": True, "record_enabled": True,
            "capabilities": cam.get("capabilities") or static_capabilities(cam["model"]),
            "created_at": time.time(),
        })


async def _drain(mgr: SpeakerProbeManager) -> None:
    for _ in range(50):
        if not mgr._tasks:
            break
        await asyncio.gather(*list(mgr._tasks), return_exceptions=True)
        await asyncio.sleep(0)


async def _stored_speaker(db: Database, name: str) -> Optional[bool]:
    cam = await db.get_camera(name)
    return (cam or {}).get("capabilities", {}).get("speaker")


def pure_checks() -> None:
    check(has_credentials(AD410) and not has_credentials(NOCREDS),
          "has_credentials gates on stored credentials")
    check(effective_capabilities(AD410)["backchannel"] is True,
          "effective_capabilities: AD410 backchannel=true (static)")
    check(effective_capabilities(TURRET)["speaker"] is False,
          "effective_capabilities: turret speaker=false (static)")
    # Stored value wins over static.
    cam = {**TURRET, "capabilities": {"speaker": True}}
    check(effective_capabilities(cam)["speaker"] is True,
          "effective_capabilities: stored speaker overrides static")


async def db_targeted_update() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        await db.connect()
        try:
            await _seed(db, [TURRET])
            before = (await db.get_camera("yard"))["capabilities"]
            check(before.get("ir") is True and before.get("white_light") is True,
                  "seed: turret has ir + white_light before update")
            updated = await db.set_camera_capability("yard", "speaker", True)
            check(updated, "set_camera_capability reports a row updated")
            after = (await db.get_camera("yard"))["capabilities"]
            check(after.get("speaker") is True, "set_camera_capability flips speaker to true")
            check(after.get("ir") is True and after.get("white_light") is True
                  and after.get("mic") is True,
                  "set_camera_capability leaves other capabilities untouched")
            missing = await db.set_camera_capability("nope", "speaker", True)
            check(not missing, "set_camera_capability on a missing camera reports no update")
        finally:
            await db.close()


async def backchannel_implies_speaker() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        await db.connect()
        try:
            # A stale-stored AD410 speaker=false must be corrected WITHOUT the
            # ONVIF probe ever being consulted (backchannel implies speaker).
            ad410_stale = {**AD410, "capabilities": {
                **static_capabilities("AD410"), "speaker": False}}
            await _seed(db, [ad410_stale])
            calls: list[str] = []

            async def probe(cam: dict[str, Any]) -> Optional[bool]:
                calls.append(cam["name"])
                return None  # would fail if consulted

            ws = FakeWS()
            mgr = SpeakerProbeManager(db, ws, probe=probe)
            # The prober/boot-sweep always passes DB-loaded rows (stored caps).
            await mgr.notify_reachable(await db.get_camera("door"))
            await _drain(mgr)
            check(await _stored_speaker(db, "door") is True,
                  "AD410 speaker pinned true via backchannel-implies-speaker")
            check(calls == [], "ONVIF probe NOT called for a backchannel camera")
            check({"type": "cameras_changed"} in ws.messages,
                  "cameras_changed broadcast on a speaker change")
            check(mgr._done.get("door") is not None, "AD410 marked done")
            await mgr.stop_all()
        finally:
            await db.close()


async def onvif_conclusive() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        await db.connect()
        try:
            # A mic-only turret and an unknown camera that ONVIF says HAS output.
            await _seed(db, [TURRET, UNKNOWN_MIC])

            async def probe(cam: dict[str, Any]) -> Optional[bool]:
                # Emulate GetAudioOutputs: unknown gate has an output, turret 0.
                return cam["name"] == "gate"

            ws = FakeWS()
            mgr = SpeakerProbeManager(db, ws, probe=probe)
            await mgr.sync([TURRET, UNKNOWN_MIC])
            await _drain(mgr)
            check(await _stored_speaker(db, "gate") is True,
                  "unknown camera with an ONVIF audio output -> speaker=true")
            check(await _stored_speaker(db, "yard") is False,
                  "turret with no ONVIF audio output stays speaker=false")
            # Only the changed camera (gate) triggers a broadcast; the turret was
            # already false, so no redundant write/broadcast for it.
            check(ws.messages.count({"type": "cameras_changed"}) == 1,
                  "only the changed camera broadcasts cameras_changed (no-op skipped)")
            # Idempotent: a second sync must not re-probe either camera.
            probed_again: list[str] = []

            async def probe2(cam: dict[str, Any]) -> Optional[bool]:
                probed_again.append(cam["name"])
                return cam["name"] == "gate"

            mgr._probe = probe2  # type: ignore[assignment]
            await mgr.sync([TURRET, UNKNOWN_MIC])
            await _drain(mgr)
            check(probed_again == [], "idempotent: resolved cameras are not re-probed")
            await mgr.stop_all()
        finally:
            await db.close()


async def inconclusive_and_nonfatal() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        await db.connect()
        try:
            await _seed(db, [UNKNOWN_MIC, NOCREDS])

            async def probe_none(cam: dict[str, Any]) -> Optional[bool]:
                return None  # offline / not ONVIF -> inconclusive

            ws = FakeWS()
            mgr = SpeakerProbeManager(db, ws, probe=probe_none)
            await mgr.sync([UNKNOWN_MIC, NOCREDS])  # must not raise
            await _drain(mgr)
            check(await _stored_speaker(db, "gate") is False,
                  "inconclusive probe leaves prior speaker value (static false)")
            check("gate" not in mgr._done,
                  "inconclusive probe does NOT mark done (retryable)")
            check("guest" not in mgr._done and ws.messages == [],
                  "no-credential camera is skipped entirely")

            # A probe that RAISES must be swallowed (non-fatal) and left retryable.
            async def probe_boom(cam: dict[str, Any]) -> Optional[bool]:
                raise RuntimeError("boom")

            mgr._probe = probe_boom  # type: ignore[assignment]
            await mgr.notify_reachable(UNKNOWN_MIC)  # must not raise
            await _drain(mgr)
            check("gate" not in mgr._done, "a raising probe is non-fatal + retryable")

            # Recovery: once the probe resolves, the later transition applies it.
            async def probe_true(cam: dict[str, Any]) -> Optional[bool]:
                return True

            mgr._probe = probe_true  # type: ignore[assignment]
            await mgr.notify_reachable(UNKNOWN_MIC)
            await _drain(mgr)
            check(await _stored_speaker(db, "gate") is True,
                  "retry applies the result after the device answers")
            await mgr.stop_all()
        finally:
            await db.close()


def pullpoint_released() -> None:
    """The probe must RELEASE the PullPoint subscription that merely
    CONSTRUCTING an ONVIFCamera creates — on the success path AND on the raise
    path, and it must never turn a probe failure into a crash.

    onvif-zeep's ONVIFCamera.__init__ -> update_xaddrs() calls
    CreatePullPointSubscription on the device, so this probe registered an event
    subscription it never wanted and never freed. Dahua caps concurrent
    subscriptions; the retry ladder made it up to two per probe, and probes fire
    on every offline->online transition."""
    print("\nPullPoint subscription release")
    PP = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"

    class FakeService:
        def __init__(self, outputs, log):
            self._outputs, self._log = outputs, log

        def GetAudioOutputs(self):
            if isinstance(self._outputs, Exception):
                raise self._outputs
            return self._outputs

        def Unsubscribe(self):
            self._log.append("unsubscribed")

    class FakeCam:
        def __init__(self, outputs, log, *, has_sub=True):
            self.xaddrs = {PP: "http://cam/onvif/Subscription?Idx=1"} if has_sub else {}
            self._outputs, self._log = outputs, log

        def create_deviceio_service(self):
            return FakeService(self._outputs, self._log)

        def create_media_service(self):
            return FakeService(self._outputs, self._log)

        def create_pullpoint_service(self):
            return FakeService(None, self._log)

    # -- success path --
    log_ok: list[str] = []
    sp_mod._release_pullpoint(FakeCam([1], log_ok))
    check(log_ok == ["unsubscribed"], "release calls Unsubscribe when a subscription exists")

    # -- a camera that never registered one: no call, no error --
    log_none: list[str] = []
    sp_mod._release_pullpoint(FakeCam([1], log_none, has_sub=False))
    check(log_none == [], "no Unsubscribe when the device never registered a subscription")

    # -- Unsubscribe itself failing must be swallowed (hygiene, not correctness) --
    class Hostile(FakeCam):
        def create_pullpoint_service(self):
            raise RuntimeError("device refused Unsubscribe")

    sp_mod._release_pullpoint(Hostile([1], []))
    check(True, "a refused Unsubscribe is swallowed (never fails the probe)")

    # -- the finally really runs: released even when GetAudioOutputs RAISES --
    real_build = None
    try:
        import onvif  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    log_raise: list[str] = []
    boom = RuntimeError("ONVIF exploded")

    # Drive the real _get_audio_outputs with a patched ONVIFCamera factory.
    import types
    fake_onvif = types.ModuleType("onvif")
    fake_onvif.ONVIFCamera = lambda ip, port, u, p, adjust_time=False: FakeCam(boom, log_raise)
    sys.modules["onvif"] = fake_onvif
    try:
        raised = False
        try:
            sp_mod._get_audio_outputs("1.2.3.4", 80, "u", "p", adjust_time=False)
        except Exception:  # noqa: BLE001 — SpeakerProbeError expected
            raised = True
        check(raised, "a totally failing ONVIF read still raises to the caller")
        check(log_raise == ["unsubscribed"],
              "the subscription is released even when the probe FAILS (finally runs)")

        log_good: list[str] = []
        fake_onvif.ONVIFCamera = (
            lambda ip, port, u, p, adjust_time=False: FakeCam([1, 2], log_good)
        )
        n = sp_mod._get_audio_outputs("1.2.3.4", 80, "u", "p", adjust_time=False)
        check(n == 2, "a successful probe still returns the audio-output count")
        check(log_good == ["unsubscribed"], "the subscription is released on SUCCESS too")
    finally:
        sys.modules.pop("onvif", None)
        if real_build is not None:
            sys.modules["onvif"] = real_build


def main() -> None:
    pure_checks()
    pullpoint_released()
    asyncio.run(db_targeted_update())
    asyncio.run(backchannel_implies_speaker())
    asyncio.run(onvif_conclusive())
    asyncio.run(inconclusive_and_nonfatal())
    print(f"\nAll {PASS} speaker-probe checks passed.")


if __name__ == "__main__":
    main()

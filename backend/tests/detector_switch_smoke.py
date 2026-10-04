"""Smoke suite for switching the detection hardware live (settings.detection.
backend: auto / gpu / cpu / coral — native/detector.SwitchableDetector).

  A. SwitchableDetector: delegates the detector contract; a switch stops the
     old detector BEFORE the new one starts (an Edge TPU is claimed
     exclusively, a CUDA session holds GPU memory); same backend is a no-op
     unless forced; a build failure keeps the old detector running.
  B. The API: saving a new backend switches the running detector without a
     restart, "cpu" is accepted and maps to D-FINE on the CPU, an unknown
     value is refused.

Usage: python backend/tests/detector_switch_smoke.py  (needs backend deps)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ.pop("VIGILUME_DETECTOR", None)
TMP = Path(tempfile.mkdtemp(prefix="vigilume-detswitch-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from app.config import BACKEND_TO_DETECTOR  # noqa: E402
from app.native.detector import SwitchableDetector  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


class FakeDetector:
    def __init__(self, backend: str, log: list[str]) -> None:
        self.backend_name = backend
        self.log = log
        self.ready = False
        self.kind = "coral" if backend == "coral" else "onnx"
        self.device = {"gpu": "cuda", "cpu": "cpu", "coral": "edgetpu"}.get(backend, "cuda")

    async def start(self) -> None:
        self.log.append(f"start {self.backend_name}")
        self.ready = True

    async def stop(self) -> None:
        self.log.append(f"stop {self.backend_name}")
        self.ready = False

    def detect(self, frame, w, h):
        return f"detected by {self.backend_name}"


async def unit_checks() -> None:
    print("A: SwitchableDetector")
    log: list[str] = []
    builds: list[str] = []

    def build(backend: str) -> FakeDetector:
        builds.append(backend)
        if backend == "broken":
            raise RuntimeError("no such hardware")
        return FakeDetector(backend, log)

    det = SwitchableDetector(build, "gpu")
    await det.start()
    check(det.ready and det.device == "cuda" and det.detect(None, 0, 0) == "detected by gpu",
          "delegates ready / device / detect to the current detector")
    check(det.backend == "gpu", "reports the backend it was built for")

    changed = await det.switch("cpu")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    check(changed and det.backend == "cpu", "switch to cpu")
    check(log == ["start gpu", "stop gpu", "start cpu"],
          f"old stopped BEFORE the new one starts ({log})")
    check(det.device == "cpu" and det.detect(None, 0, 0) == "detected by cpu",
          "calls now reach the new detector")

    n = len(builds)
    check(not await det.switch("cpu") and len(builds) == n, "same backend: nothing rebuilt")
    check(await det.switch("cpu", force=True) and len(builds) == n + 1,
          "forced (a new Edge TPU model): rebuilt on the same backend")
    await asyncio.sleep(0)

    before = det.backend
    check(not await det.switch("broken") and det.backend == before and det.ready,
          "a build failure keeps the running detector")

    await det.switch("coral")
    await det.stop()
    check(not det.ready, "stop() stops the current detector")
    try:
        getattr(SwitchableDetector.__new__(SwitchableDetector), "_active")
        check(False, "own fields never recurse through delegation")
    except AttributeError:
        check(True, "own fields never recurse through delegation")

    check(BACKEND_TO_DETECTOR["cpu"] == "onnx_cpu", "cpu maps to D-FINE on the CPU")


def api_checks() -> None:
    print("B: API")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        det = app.state.detector
        check(isinstance(det, SwitchableDetector) and det.backend == "auto",
              "the app holds a switchable detector, on auto by default")
        r = client.patch("/api/settings", headers=h, json={"detection": {"backend": "cpu"}})
        check(r.status_code == 200 and r.json()["detection"]["backend"] == "cpu",
              f"cpu is a valid setting ({r.status_code})")
        check(det.backend == "cpu" and det.kind == "onnx",
              "saving switched the RUNNING detector — no restart")
        check(getattr(det, "_force_cpu", None) is True,
              "...to D-FINE forced onto the CPU")
        r = client.patch("/api/settings", headers=h, json={"detection": {"backend": "gpu"}})
        check(r.status_code == 200 and det.backend == "gpu", "and back to the GPU")
        r = client.patch("/api/settings", headers=h, json={"detection": {"backend": "tpu9000"}})
        check(r.status_code == 422 and det.backend == "gpu", "an unknown backend is refused")
        r = client.patch("/api/settings", headers=h, json={"detection": {"confidence": 0.55}})
        check(r.status_code == 200 and det.backend == "gpu", "other detection changes do not rebuild it")


def main() -> None:
    asyncio.run(unit_checks())
    api_checks()
    print(f"\nALL {PASS} CHECKS PASSED (detector switch)")


if __name__ == "__main__":
    main()

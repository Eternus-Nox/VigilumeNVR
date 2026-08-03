"""Doorbell IR + NIGHT-VISION re-assert (app/amcrest/ir_reassert.py).

The AD410 resets BOTH its IR illuminator and its day/night mode back to Auto
whenever an RTSP client connects, so anything the user pinned is silently lost
every time the recorder / go2rtc reconnects. This suite pins the re-assert
contract that restores it.

REGRESSION THIS GUARDS (reported 2026-07-23: "doorbell not sustaining save
settings, I have full color set and it keeps going to auto"): night vision was
broken in TWO places at once — `routers/cameras.py` applied the mode to the
camera but never persisted it into `ir_state`, and `ir_reassert` only ever
re-applied `mode`/`brightness`. So there was nothing stored AND nothing would
have restored it anyway.

cv2-free by construction (pure dict/async logic against a fake client), so it
runs anywhere.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.amcrest.ir_reassert import (  # noqa: E402
    apply_ir_state,
    desired_ir_from,
    desired_night_vision_from,
    model_reverts_ir,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


class _FakeClient:
    """Records the device calls a re-assert would make."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def set_night_vision_mode(self, mode: str) -> None:
        self.calls.append(("nv", mode))

    async def set_ir(self, mode=None, brightness=None) -> None:
        self.calls.append(("ir", mode, brightness))


def main() -> None:
    # ---- which cameras are re-asserted at all ----
    check(model_reverts_ir("AD410"), "the AD410 is re-asserted")
    check(model_reverts_ir("", {"doorbell": True}), "any doorbell-capable camera is re-asserted")
    check(not model_reverts_ir("IP5M-T1277EW-AI"),
          "IR-only turrets are NOT re-asserted (they hold IR across streaming)")

    # ---- night-vision extraction ----
    check(desired_night_vision_from({"night_vision_mode": "color"}) == "color",
          "full colour is extracted for re-assert")
    check(desired_night_vision_from({"night_vision_mode": "auto"}) == "auto",
          "auto is extracted too (an explicit choice, not a default)")
    check(desired_night_vision_from({"night_vision_mode": "bogus"}) is None,
          "an invalid mode is ignored rather than pushed to the device")
    check(desired_night_vision_from({"day_night": "color"}) is None,
          "day_night is NOT re-asserted — the IR-cut filter survives streaming")
    check(desired_night_vision_from(None) is None, "no stored state -> nothing to re-assert")

    # ---- IR extraction still works (unchanged behaviour) ----
    check(desired_ir_from({"mode": "on", "brightness": 60}) == {"mode": "on", "brightness": 60},
          "IR mode + brightness still extracted")
    check(desired_ir_from({"brightness": 999})["brightness"] == 100, "brightness is clamped to 100")
    check(desired_ir_from({}) is None, "an empty ir_state yields no IR desire")

    # ---- the actual re-assert ----
    client = _FakeClient()
    check(asyncio.run(apply_ir_state(client, {"night_vision_mode": "color"})) is True,
          "a night-vision-ONLY state triggers a re-assert (this was the bug)")
    check(client.calls == [("nv", "color")], f"…and calls set_night_vision_mode ({client.calls})")

    client = _FakeClient()
    asyncio.run(apply_ir_state(client, {"night_vision_mode": "color", "mode": "on", "brightness": 60}))
    check(client.calls == [("nv", "color"), ("ir", "on", 60)],
          "night vision is applied BEFORE IR, so a pinned IR survives the "
          f"illuminator coupling ({client.calls})")

    client = _FakeClient()
    check(asyncio.run(apply_ir_state(client, {})) is False and client.calls == [],
          "an empty state issues no device traffic at all")

    client = _FakeClient()
    check(asyncio.run(apply_ir_state(client, None)) is False,
          "a missing ir_state is a clean no-op")

    print(f"\nAll {PASS} ir/night-vision re-assert checks passed.")


if __name__ == "__main__":
    main()

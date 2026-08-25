#!/usr/bin/env python3
"""Doorbell event-stream liveness: keepalives, and what silence means.

The bug this pins: an idle `eventManager attach` sends nothing, so a stream that
has silently died looks exactly like a quiet doorstep. With a bare 300 s read
timeout the watcher sat deaf for up to five minutes per occurrence and every
press in that window was lost — the "sometimes I don't even get a call" failure.

Asking for `heartbeat=N` makes silence meaningful. These checks cover the parts
that decide whether a press is heard at all: the keepalive is requested, it is
recognised, it never reads as a press, and the read timeout only tightens once
the camera has PROVEN it sends keepalives (firmware that ignores the parameter
must not be put on a 75 s reconnect treadmill).

Offline-runnable — no camera, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.amcrest import doorbell  # noqa: E402
from app.amcrest.doorbell import (  # noqa: E402
    _HEARTBEAT_S,
    _NOISY_CODES,
    _READ_TIMEOUT_HEARTBEAT_S,
    _READ_TIMEOUT_SILENT_S,
    DoorbellWatcher,
    is_button_press,
    parse_event_part,
)

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


def _watcher() -> DoorbellWatcher:
    async def _noop(_name: str) -> None:  # pragma: no cover - never called here
        return None

    return DoorbellWatcher("front", "10.0.0.5", "admin", "pw", _noop)


# Real shapes, as the camera writes them into the multipart stream.
HEARTBEAT_PART = "\r\nCode=Heartbeat;action=Pulse;index=0\r\n"
PRESS_PART = '\r\nCode=_DoTalkAction_;action=Pulse;index=0;data={"Action":"Invite"}\r\n'
MOTION_PART = "\r\nCode=VideoMotion;action=Start;index=0\r\n"


def main() -> int:
    print("doorbell stream liveness")

    # --- the keepalive is actually requested -----------------------------
    src = Path(doorbell.__file__).read_text()
    check(
        f"heartbeat={{_HEARTBEAT_S}}" in src or f"heartbeat={_HEARTBEAT_S}" in src,
        "the attach URL asks the camera for keepalives",
    )
    check(
        1 <= _HEARTBEAT_S <= 60,
        f"heartbeat={_HEARTBEAT_S}s is inside Dahua's documented 1-60s range",
    )
    check(
        _READ_TIMEOUT_HEARTBEAT_S > _HEARTBEAT_S * 2,
        f"the read timeout ({_READ_TIMEOUT_HEARTBEAT_S:.0f}s) tolerates at least "
        f"two missed keepalives — one dropped packet must not force a reconnect",
    )
    check(
        _READ_TIMEOUT_HEARTBEAT_S < _READ_TIMEOUT_SILENT_S,
        "and is well short of the silent-firmware fallback it replaces",
    )

    # --- a keepalive is not a press --------------------------------------
    hb = parse_event_part(HEARTBEAT_PART)
    check(hb is not None and hb["code"] == "Heartbeat", "a keepalive part parses")
    check(not is_button_press(hb or {}), "a keepalive is NOT a button press")
    check("Heartbeat" in _NOISY_CODES, "and is not logged as a notable event")
    press = parse_event_part(PRESS_PART)
    check(press is not None and is_button_press(press), "a real press still reads as one")

    # --- the timeout only tightens on PROVEN keepalive support -----------
    w = _watcher()
    check(
        not w._heartbeat_seen,
        "a fresh watcher assumes nothing about the firmware",
    )
    check(
        _timeout_for(w) == _READ_TIMEOUT_SILENT_S,
        f"so its first attach waits the full {_READ_TIMEOUT_SILENT_S:.0f}s — "
        "silence from an unproven camera is not evidence of a dead stream",
    )

    import asyncio

    asyncio.run(w._handle_part(HEARTBEAT_PART))
    check(w._heartbeat_seen, "one keepalive proves the firmware supports them")
    check(
        _timeout_for(w) == _READ_TIMEOUT_HEARTBEAT_S,
        f"and the watcher drops to {_READ_TIMEOUT_HEARTBEAT_S:.0f}s — a dead "
        f"stream is now caught in about a minute, not five",
    )

    # Sticky: a camera that goes quiet for a while has not lost the capability,
    # and re-probing on every reconnect would reinstate the long stall forever.
    asyncio.run(w._handle_part(MOTION_PART))
    asyncio.run(w._handle_part(PRESS_PART))
    check(w._heartbeat_seen, "and that knowledge survives later traffic")

    # --- presses still reach the callback --------------------------------
    # The point of all of the above is that this keeps happening; assert it
    # rather than assuming it, since _handle_part now returns early on one code.
    seen: list[str] = []

    async def _record(name: str) -> None:
        seen.append(name)

    w2 = DoorbellWatcher("front", "10.0.0.5", "admin", "pw", _record)
    asyncio.run(w2._handle_part(HEARTBEAT_PART))
    check(seen == [], "a keepalive does not fire the press callback")
    asyncio.run(w2._handle_part(PRESS_PART))
    check(seen == ["front"], "a press does")

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (doorbell stream liveness)")
    return 0


def _timeout_for(w: DoorbellWatcher) -> float:
    """The read timeout _attach_once would choose for this watcher."""
    return _READ_TIMEOUT_HEARTBEAT_S if w._heartbeat_seen else _READ_TIMEOUT_SILENT_S


if __name__ == "__main__":
    raise SystemExit(main())

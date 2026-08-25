#!/usr/bin/env python3
"""Two-way talk transport fallback.

A camera's `backchannel` capability is a CLAIM, not a measurement — features.py
clones entries between models believed identical, and the IP4M-1041B was added
as a copy of the IP3M-941B on exactly that basis. When the claim is wrong, talk
used to fail outright with "the camera rejected the audio".

Now a NEGOTIATION failure falls back to CGI postAudio. The line these checks
defend is WHERE that fallback is allowed: negotiation happens before a single
PCM frame is read, so the mic iterator is untouched and retrying costs nothing.
Once streaming has begun those frames are consumed and gone — falling back then
would replay a truncated stream over itself, so a mid-stream failure must
propagate instead.

No camera and no network: the two transports are stubbed and the frames they
consume are counted.

Offline-runnable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.amcrest.backchannel import BackchannelUnsupported  # noqa: E402
from app.amcrest.client import AmcrestError  # noqa: E402

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


async def frames(n: int = 5) -> AsyncIterator[bytes]:
    for i in range(n):
        yield bytes([i])


class Transport:
    """Records what reached it, and can fail at a chosen point."""

    def __init__(self, fail: Exception | None = None, fail_after: int = 0) -> None:
        self.consumed: list[bytes] = []
        self.called = False
        self._fail = fail
        self._fail_after = fail_after

    async def run(self, chunks: AsyncIterator[bytes]) -> None:
        self.called = True
        if self._fail is not None and self._fail_after == 0:
            raise self._fail          # negotiation-time: nothing consumed
        async for chunk in chunks:
            self.consumed.append(chunk)
            if self._fail is not None and len(self.consumed) >= self._fail_after:
                raise self._fail      # mid-stream: frames already gone


async def deliver(backchannel: bool, bc: Transport, cgi: Transport,
                  chunks: AsyncIterator[bytes]) -> None:
    """The router's delivery decision, mirrored exactly (routers/talk.py)."""
    if not backchannel:
        await cgi.run(chunks)
        return
    try:
        await bc.run(chunks)
    except BackchannelUnsupported:
        await cgi.run(chunks)


async def main_async() -> None:
    # --- a camera correctly marked NOT backchannel -------------------------
    bc, cgi = Transport(), Transport()
    await deliver(False, bc, cgi, frames())
    check(not bc.called and len(cgi.consumed) == 5,
          "a non-backchannel camera goes straight to CGI postAudio, untouched")

    # --- backchannel works: CGI must never be involved ---------------------
    bc, cgi = Transport(), Transport()
    await deliver(True, bc, cgi, frames())
    check(
        len(bc.consumed) == 5 and not cgi.called,
        "a working backchannel keeps every frame — the AD410 path is unchanged, "
        "and CGI is never touched (that firmware silently DROPS postAudio, so a "
        "stray fallback would look fine and play nothing)",
    )

    # --- the wrong claim: negotiation fails, fall back ---------------------
    bc = Transport(fail=BackchannelUnsupported("no ONVIF backchannel"))
    cgi = Transport()
    await deliver(True, bc, cgi, frames())
    check(
        bc.called and not bc.consumed,
        "a negotiation failure consumes NO frames — which is what makes the "
        "fallback free",
    )
    check(
        len(cgi.consumed) == 5,
        "so CGI receives the COMPLETE mic stream, not a remnant — this is the "
        "IP4M-1041B case, mis-cloned from the 941B",
    )

    # --- mid-stream failure must NOT fall back -----------------------------
    bc = Transport(fail=AmcrestError("connection reset"), fail_after=2)
    cgi = Transport()
    raised = False
    try:
        await deliver(True, bc, cgi, frames())
    except AmcrestError:
        raised = True
    check(raised, "a mid-stream failure PROPAGATES (the session fences, 4502)")
    check(
        not cgi.called,
        "and does NOT fall back — those frames are already consumed, so a retry "
        "would send a truncated stream on top of what was already spoken",
    )

    # --- the exception has to be catchable as an Amcrest failure -----------
    check(
        issubclass(BackchannelUnsupported, AmcrestError),
        "BackchannelUnsupported is an AmcrestError, so an un-fallen-back "
        "negotiation failure still fences the session like any camera error",
    )
    generic = Transport(fail=AmcrestError("negotiation refused"))
    cgi = Transport()
    raised = False
    try:
        await deliver(True, generic, cgi, frames())
    except AmcrestError:
        raised = True
    check(
        raised and not cgi.called,
        "and a PLAIN AmcrestError does not trigger the fallback — only the "
        "specific negotiation signal does, so the safe window stays narrow",
    )


def main() -> int:
    print("talk transport fallback")
    asyncio.run(main_async())
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (talk transport fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

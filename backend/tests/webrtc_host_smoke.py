"""WebRTC host-candidate discovery (app/native/streams.py).

THE PROBLEM THIS GUARDS. The backend runs in a Docker BRIDGE network, so
`_auto_lan_ipv4()` can only ever see the bridge address and correctly rejects
it — meaning a default install advertised NO host candidate at all. go2rtc then
offers STUN only, WebRTC cannot connect even on the LAN, and every client
silently degrades to slow HLS after burning its connect timeout. The operator
had to know to set `VIGILUME_WEBRTC_HOST` by hand, which nobody does.

THE FIX. Learn the box's LAN address from the address a client actually reached
it on (the Host header), via `note_observed_host`, called from a middleware in
main.py. That header is caller-controlled, so the accept rule is deliberately
narrow — private IPv4 literals only, never a hostname or public address, and
learning is disabled outright when an explicit env host is configured.

cv2-free, so it runs anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.native.streams as streams  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


EMPTY = {"system": {"webrtc_candidates": []}}


def main() -> None:
    for var in ("VIGILUME_WEBRTC_HOST", "SENTINEL_WEBRTC_HOST"):
        os.environ.pop(var, None)

    # Simulate the real deployment. On bare metal (a dev Mac) the auto path
    # succeeds and would mask the bug this whole feature exists to fix.
    real_auto = streams._auto_lan_ipv4
    streams._auto_lan_ipv4 = lambda: None
    try:
        streams._observed_host = None
        check(streams.webrtc_status(EMPTY)["ready"] is False,
              "bridge container + nothing configured -> ready=False (the original bug)")

        check(streams.note_observed_host("192.168.1.253") is True,
              "a private IPv4 from a real request is learned")
        status = streams.webrtc_status(EMPTY)
        check(status["ready"] is True, "…and WebRTC becomes ready with NO manual setup")
        check("192.168.1.253:8555" in status["candidates"],
              f"the host candidate is advertised ({status['candidates']})")
        check(status["source"] == "observed", "the source is reported as 'observed'")
        check(streams.note_observed_host("192.168.1.253") is False,
              "the same address again is not a change (no go2rtc regen storm)")

        # Caller-controlled input: everything that is not a LAN IPv4 is refused.
        check(streams.note_observed_host("nvr.example.com") is False,
              "a HOSTNAME is refused — a tunnel/remote caller cannot inject a candidate")
        check(streams.note_observed_host("203.0.113.7") is False, "a PUBLIC IPv4 is refused")
        check(streams.note_observed_host("127.0.0.1") is False, "loopback is refused")
        check(streams.note_observed_host("172.17.0.5") is False,
              "a docker-bridge address is refused (same rule as the auto path)")
        check(streams.note_observed_host("") is False, "an empty Host is refused")
        check(streams.note_observed_host("192.0.2.5") is False,
              "a DOCUMENTATION net is refused (Python's is_private counts these as private)")
        check(streams.note_observed_host("169.254.1.1") is False, "link-local is refused")
        check(streams.webrtc_status(EMPTY)["detected_ip"] == "192.168.1.253",
              "…and none of those overwrote the good learned value")

        streams._observed_host = None
        check(streams.note_observed_host("100.101.102.103") is True,
              "a Tailscale/CGNAT (100.64/10) address IS accepted — a valid VPN path to us")
        streams._observed_host = "192.168.1.253"

        # An explicit configuration must never be influenced by a client.
        os.environ["VIGILUME_WEBRTC_HOST"] = "10.0.0.9"
        streams._observed_host = None
        check(streams.note_observed_host("192.168.1.253") is False,
              "VIGILUME_WEBRTC_HOST set -> learning is disabled entirely")
        check(streams.webrtc_status(EMPTY)["source"] == "env",
              "the explicit env host still wins")
        os.environ.pop("VIGILUME_WEBRTC_HOST")

        # A hand-entered candidate stays authoritative.
        streams._observed_host = "192.168.1.253"
        manual = streams.webrtc_status({"system": {"webrtc_candidates": ["10.9.9.9:8555"]}})
        check(manual["candidates"][0] == "10.9.9.9:8555",
              "a MANUAL candidate stays first — the operator override is respected")
        check("192.168.1.253:8555" in manual["candidates"],
              "…and the learned address is offered alongside it")
    finally:
        streams._auto_lan_ipv4 = real_auto
        streams._observed_host = None

    print(f"\nAll {PASS} webrtc host-candidate checks passed.")


if __name__ == "__main__":
    main()

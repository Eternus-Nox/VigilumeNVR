#!/usr/bin/env python3
"""No-internet operation: the LAN must not depend on the WAN.

An NVR's job is recording the house. That job has nothing to do with the
internet, and it must keep running when the internet is gone — cable unplugged,
ISP down, or a deliberately isolated network. Cloud features (the nightly
archive, push, model downloads) legitimately stop; LIVE VIEW, RECORDING,
DETECTION on an already-downloaded model, and the whole web UI must not.

These checks pin the parts of that promise which are testable without a network:
what the app would REACH for, and whether the paths that matter avoid it.

The failure mode being guarded is subtle. Nothing here crashes when offline —
the code is careful about that already. What goes wrong is WAITING: a live-view
setup that pauses on a STUN lookup, an event pipeline that stalls on a push
relay. An appliance that merely survives an outage is not the same as one that
keeps working through it.

Offline-runnable, obviously.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native import streams as streams_module  # noqa: E402
from app.native.streams import build_config, webrtc_status  # noqa: E402

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


def settings(candidates=None, public_url=""):
    return {"system": {"webrtc_candidates": list(candidates or []),
                       "public_url": public_url}}


CAM = {
    "name": "front", "ip": "192.168.1.60", "username": "admin", "password": "pw",
    "audio_codec": "g711a", "model": "IP4M-1041B",
}


def main() -> int:
    print("offline (LAN-only) operation")
    saved_auto = streams_module._auto_lan_ipv4
    try:
        streams_module._auto_lan_ipv4 = lambda: "192.168.1.45"

        # --- live view must not wait on the internet ----------------------
        lan = webrtc_status(settings())
        check(
            not any(c.startswith("stun:") for c in lan["candidates"]),
            "a LAN-only install advertises NO stun candidate, so live view "
            "never blocks on a public-address lookup that cannot be answered",
        )
        check(
            lan["ready"] is True and lan["candidates"] == ["192.168.1.45:8555"],
            "it advertises its LAN address instead — which is the one every "
            "viewer on the LAN can actually reach",
        )
        cfg = build_config([CAM], settings())
        check(
            not any(str(c).startswith("stun:") for c in cfg["webrtc"]["candidates"]),
            "and the go2rtc config written to disk carries no stun entry either",
        )

        # --- the video path itself is entirely local -----------------------
        blob = str(cfg)
        check(
            "http://" not in blob.replace("http://192.168.", "")
            and "https://" not in blob,
            "no go2rtc stream source points anywhere but the LAN",
        )
        sources = cfg["streams"]
        check(bool(sources), "streams are still configured with no internet")
        check(
            all("192.168.1.60" in str(v) for v in sources.values()),
            "every stream source is the camera's own LAN address (rtsp://), so "
            "recording and live view are pure LAN traffic",
        )

        # --- a remote install is deliberately unchanged --------------------
        remote = webrtc_status(settings(public_url="https://nvr.example.com"))
        check(
            any(c.startswith("stun:") for c in remote["candidates"]),
            "declaring a public_url restores stun — off-LAN viewers need the "
            "public address, and that install has internet by definition",
        )

        # --- nothing reaches the WAN to serve the UI -----------------------
        # A CDN font or script would make the settings page hang offline even
        # though every byte it needs is on the box.
        web = Path(__file__).resolve().parents[2] / "frontend"
        external: list[str] = []
        for pattern in ("index.html", "src/**/*.tsx", "src/**/*.ts", "src/**/*.css"):
            for f in web.glob(pattern):
                text = f.read_text(errors="ignore")
                for marker in ("fonts.googleapis.com", "cdn.jsdelivr.net",
                               "unpkg.com", "cdnjs.cloudflare.com"):
                    if marker in text:
                        external.append(f"{f.name}:{marker}")
        check(
            not external,
            f"the web UI loads no CDN fonts or scripts{'' if not external else ': ' + ', '.join(external)}"
            " — every asset is served by the box itself",
        )

        print()
        if _failures:
            print(f"{len(_failures)} of {_checks} CHECKS FAILED")
            for f in _failures:
                print(f"  - {f}")
            return 1
        print(f"ALL {_checks} CHECKS PASSED (offline LAN-only operation)")
        return 0
    finally:
        streams_module._auto_lan_ipv4 = saved_auto


if __name__ == "__main__":
    raise SystemExit(main())

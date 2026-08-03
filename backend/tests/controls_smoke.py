"""Smoke suite for Camera controls v2 (docs/CONTRACTS.md addendum):

  - capability map: white_light now TRUE for the two EW turrets, AD410
    unchanged (white_light false, speaker true)
  - POST /api/cameras/{name}/light {mode, brightness?}: exact Lighting_V2
    CGI strings asserted via a fake httpx transport (MockTransport);
    brightness bounds -> 422; firmware without the CGI -> 501 (not 500);
    transient setConfig rejection -> 502
  - GET/PUT /api/cameras/{name}/settings white_light passthrough
  - POST /api/cameras/{name}/probe: success + "unknown" model adoption,
    authentication failed, camera unreachable, and the on-demand time cap
    against a stalling (TCP-open, never-answering) device
  - siren regression: play_tone still posts the exact G.711A tone
  - WS /api/cameras/{name}/talk: the CGI postAudio happy-path runs against
    a REAL fake camera HTTP server (raw asyncio TCP) using a NON-backchannel
    speaker camera — A-law bytes arrive, busy 4009 (concurrent second talker),
    non-speaker 4003, media-scope/missing/garbage token rejected, camera-reject
    4502, stop-on-close releases the single-talker lock, 120 s cap path
    (shrunk). A separate check asserts a backchannel-capable camera (AD410)
    routes to talk_stream_backchannel (RTSP), with that sender stubbed so no
    real RTSP server is needed.
  - native regression guard: camera CRUD + go2rtc config regen, URL
    overrides + detect_fps round-trip

Runs the real FastAPI app via TestClient with AmcrestClient.__init__
patched so device HTTP goes to httpx.MockTransport (unit-ish tests) or to
the local fake camera TCP server (talk tests).

Usage: python backend/tests/controls_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Environment must be clean before app config is instantiated (lifespan).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
# Unroutable local port -> instant connection refusal, so go2rtc config
# syncs during camera CRUD never slow the suite down.
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-controls-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from app.amcrest import client as amcrest_client_module  # noqa: E402
from app.amcrest.client import (  # noqa: E402
    AmcrestClient,
    cgi_accepted,
    make_alarm_tone,
    pcm16le_to_alaw,
    white_light_control_for_model,
)
from app.amcrest.features import static_capabilities  # noqa: E402
from app.native import sun as sun_mod  # noqa: E402
from app.native.spotlight import SpotlightController  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import cameras as cameras_router  # noqa: E402
from app.routers import talk as talk_router  # noqa: E402

cameras_router._PROBE_TIMEOUT_S = 2.0  # POST/PUT background capability probe

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- fake Amcrest device (httpx.MockTransport) ----------------


class FakeDevice:
    """Swappable per-test device behavior + request log.

    behavior: "turret" | "ad410" | "auth_fail" | "unreachable"
              | "wl_reject_unsupported" (setConfig -> Error AND Lighting_V2
                readback -> Error: firmware without the white-LED slot)
              | "wl_reject_transient" (setConfig -> Error but Lighting_V2
                readback OK: flaky device, must stay a 502)
    base_url: when set, AmcrestClient talks REAL TCP to that URL instead of
    the MockTransport (used by the talk tests' fake camera server).
    """

    def __init__(self) -> None:
        self.behavior = "turret"
        self.base_url: str | None = None
        self.requests: list[httpx.Request] = []

    # -- MockTransport handler --

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.behavior == "unreachable":
            raise httpx.ConnectError("fake: no route to host")
        if self.behavior == "auth_fail":
            return httpx.Response(401)
        path, params = request.url.path, request.url.params
        if path == "/cgi-bin/magicBox.cgi" and params.get("action") == "getDeviceType":
            # NB: this fake reports a TRUNCATED turret id to exercise the
            # prefix path. Real hardware does NOT do this — the live fleet
            # audit (2026-07-16) shows the turrets report the full
            # "IP5M-T1277EW-AI" WITH the suffix, the AD410 a bare "AD410", and
            # the IP8M "IP8M-2779E-AI" (no W). The previous comment here
            # asserted real firmware drops the -AI, which is false.
            dt = "AD410" if self.behavior in ("ad410", "ad410_no_audiocollect") else "IP5M-T1277EW"
            return httpx.Response(200, text=f"type={dt}\r\n")
        if path == "/cgi-bin/configManager.cgi" and params.get("action") == "getConfig":
            return self._get_config(params.get("name", ""))
        if path == "/cgi-bin/configManager.cgi" and params.get("action") == "setConfig":
            if self.behavior in ("wl_reject_unsupported", "wl_reject_transient"):
                return httpx.Response(200, text="Error\r\n")
            return httpx.Response(200, text="OK\r\n")
        if path == "/cgi-bin/devAudioInput.cgi":
            # An AD410 firmware that doesn't expose the audio collect CGI.
            inp = "0" if self.behavior == "ad410_no_audiocollect" else "1"
            return httpx.Response(200, text=f"result={inp}\r\n")
        if path == "/cgi-bin/devAudioOutput.cgi":
            # "speaker_cgi": a plain (non-AD410) speaker camera — audio output
            # present, so the probe stores speaker=True while backchannel stays
            # False (only the AD410 branch sets backchannel). Drives the CGI
            # postAudio talk happy-path below.
            out = "1" if self.behavior in ("ad410", "speaker_cgi") else "0"
            return httpx.Response(200, text=f"result={out}\r\n")
        if path == "/cgi-bin/coaxialControlIO.cgi":
            # EW-turret white spotlight (Type=1) on/off + status.
            if self.behavior == "coax_reject_unsupported":
                return httpx.Response(400)  # firmware without the CGI -> 501
            if self.behavior == "coax_reject_transient":
                return httpx.Response(200, text="Error\r\n")  # flaky -> 502
            if params.get("action") == "getStatus":
                return httpx.Response(200, text="status.WhiteLight=On\r\n")
            return httpx.Response(200, text="OK\r\n")
        if path == "/cgi-bin/audio.cgi":
            return httpx.Response(200, text="OK\r\n")
        if path == "/cgi-bin/ptz.cgi":
            # PTZ dome (IP3M-941B): move/stop/preset all answer OK.
            if self.behavior == "ptz_reject_transient":
                return httpx.Response(200, text="Error\r\n")  # flaky -> 502
            return httpx.Response(200, text="OK\r\n")
        return httpx.Response(404)

    def _get_config(self, name: str) -> httpx.Response:
        tables = {
            # Four day/night profiles, like a real turret — the camera obeys
            # the ACTIVE one (often not [0][0]), so set_ir must write them all.
            "Lighting": (
                "table.Lighting[0][0].Mode=Auto\r\n"
                "table.Lighting[0][1].Mode=Auto\r\n"
                "table.Lighting[0][2].Mode=Auto\r\n"
                "table.Lighting[0][3].Mode=Auto\r\n"
            ),
            "Lighting_V2": (
                # slot [0] = IR, slot [1] = white LED (rroller/dahua layout)
                "table.Lighting_V2[0][0][0].Mode=Auto\r\n"
                "table.Lighting_V2[0][0][0].MiddleLight[0].Light=50\r\n"
                "table.Lighting_V2[0][0][1].Mode=Manual\r\n"
                "table.Lighting_V2[0][0][1].MiddleLight[0].Light=55\r\n"
            ),
            # Multiple day/night profiles like a real turret — night_vision_mode
            # must write EVERY exposed index, not just [0][0].
            "VideoInDayNight": (
                "table.VideoInDayNight[0][0].Mode=Color\r\n"
                "table.VideoInDayNight[0][1].Mode=Color\r\n"
                "table.VideoInDayNight[0][2].Mode=Color\r\n"
            ),
            "MotionDetect": "table.MotionDetect[0].Enable=true\r\n",
            "ChannelTitle": "table.ChannelTitle[0].Name=Cam\r\n",
            "VideoInOptions": "table.VideoInOptions[0].Flip=false\r\n",
            "AudioOutputVolume": "table.AudioOutputVolume[0]=80\r\n",
        }
        if self.behavior == "ad410" and name == "Lighting_V2":
            return httpx.Response(200, text="Error\r\n")  # doorbell: no table
        if self.behavior == "wl_reject_unsupported" and name == "Lighting_V2":
            return httpx.Response(200, text="Error\r\n")  # no white-LED slot
        body = tables.get(name)
        if body is None:
            return httpx.Response(200, text="Error\r\n")
        return httpx.Response(200, text=body)

    def set_config_requests(self) -> list[httpx.Request]:
        return [
            r for r in self.requests
            if r.url.path == "/cgi-bin/configManager.cgi"
            and r.url.params.get("action") == "setConfig"
        ]

    def coax_control_requests(self) -> list[httpx.Request]:
        return [
            r for r in self.requests
            if r.url.path == "/cgi-bin/coaxialControlIO.cgi"
            and r.url.params.get("action") == "control"
        ]


FAKE = FakeDevice()


def _fake_init(
    self, ip: str, username: str, password: str, timeout: float = 8.0, model: str = ""
) -> None:
    self.ip = ip
    self.model = (model or "").strip()
    if FAKE.base_url:
        # Talk tests: real TCP to the fake camera server below.
        self._client = httpx.AsyncClient(
            base_url=FAKE.base_url,
            auth=httpx.DigestAuth(username, password),
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
    else:
        self._client = httpx.AsyncClient(
            base_url=f"http://{ip}",
            auth=httpx.DigestAuth(username, password),
            transport=httpx.MockTransport(FAKE),
            timeout=httpx.Timeout(2.0, connect=1.0),
        )


AmcrestClient.__init__ = _fake_init  # type: ignore[method-assign]


# ---------------- fake camera HTTP server (talk tests) ----------------


class FakeCameraServer(threading.Thread):
    """Minimal raw-HTTP camera on 127.0.0.1: answers the digest-priming
    getDeviceType GET and captures audio.cgi postAudio body bytes. Sends the
    postAudio 200 EARLY so httpx resolves the request as soon as the audio
    stream ends (matching real firmwares that answer immediately)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.port: int | None = None
        self.ready = threading.Event()
        self.mode = "ok"  # "ok" | "reject" (401 everything) | "stall" (never answer)
        self.audio_bodies: list[bytearray] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        # NB: not named _stop — that shadows threading.Thread._stop, which
        # join() calls internally once the thread finishes (TypeError).
        self._stop_event: asyncio.Event | None = None

    def run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        self.ready.set()
        async with server:
            await self._stop_event.wait()

    def shutdown(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self.join(timeout=5)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = await reader.read(4096)
                if not chunk:
                    return
                head += chunk
            head, _, initial_body = head.partition(b"\r\n\r\n")
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            if self.mode == "stall":
                # Accept the request and never answer (dead device with an
                # open TCP port) — drain until the client gives up.
                while await reader.read(4096):
                    pass
                return
            if self.mode == "reject":
                writer.write(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                return
            if "magicBox.cgi" in request_line:
                body = b"type=AD410\r\n"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
                    % (len(body), body)
                )
                await writer.drain()
                return
            if "audio.cgi" in request_line:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                buf = bytearray(initial_body)
                self.audio_bodies.append(buf)
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        return
                    buf.extend(chunk)
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()


# ---------------- helpers ----------------


def login(client: TestClient) -> tuple[dict[str, str], str]:
    resp = client.post("/api/auth/login", json={"password": "test-password"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}, token


def add_camera(client: TestClient, headers: dict, name: str, model: str, ip: str) -> dict:
    FAKE.requests.clear()
    resp = client.post("/api/cameras", headers=headers, json={
        "name": name, "friendly_name": name.replace("_", " ").title(),
        "model": model, "ip": ip, "username": "admin", "password": "pw",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


def pcm_frame(n_samples: int = 160, seed: int = 1) -> bytes:
    """One 20 ms 8 kHz mono frame of deterministic Int16 LE PCM."""
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32767, n_samples, dtype=np.int16).astype("<i2").tobytes()


def ws_close_code(msg: dict) -> int | None:
    if msg.get("type") == "websocket.close":
        return msg.get("code")
    return None


# ---------------- unit checks ----------------


def unit_checks() -> None:
    print("unit: capability map + A-law transcode")
    for model in ("IP5M-T1277EW-AI", "IP8M-2779EW-AI"):
        caps = static_capabilities(model)
        check(caps["white_light"] is True, f"{model} white_light TRUE (EW illuminator)")
        check(caps["speaker"] is False and caps["siren"] is False,
              f"{model} still no speaker/siren")
    ad = static_capabilities("AD410")
    check(ad["white_light"] is False, "AD410 white_light stays false (status LED only)")
    check(ad["speaker"] is True and ad["siren"] is True and ad["doorbell"] is True,
          "AD410 keeps speaker/siren/doorbell")
    check(static_capabilities("unknown")["white_light"] is False,
          "model 'unknown' promises nothing")

    # New models: IP3M-941B (PTZ + backchannel talk) and IP4M-1056E (night vision).
    ptz_caps = static_capabilities("IP3M-941B")
    check(ptz_caps["ptz"] is True and ptz_caps["backchannel"] is True
          and ptz_caps["mic"] is True and ptz_caps["speaker"] is True,
          "IP3M-941B: ptz + backchannel + mic + speaker true")
    check(ptz_caps["white_light"] is False and ptz_caps["ai_on_camera"] is False,
          "IP3M-941B: white_light + ai_on_camera false")
    nv_caps = static_capabilities("IP4M-1056E")
    check(nv_caps["night_vision"] is True and nv_caps["ir"] is True,
          "IP4M-1056E: night_vision + ir true")
    check(nv_caps["ptz"] is False and nv_caps["mic"] is False
          and nv_caps["speaker"] is False and nv_caps["white_light"] is False,
          "IP4M-1056E: ptz/mic/speaker/white_light false")
    # night_vision is now TRUE on EVERY model — the night-vision Auto/Full-color/
    # IR control replaces the retired IR button everywhere; ptz stays per-model.
    for model in ("IP5M-T1277EW-AI", "IP8M-2779EW-AI", "AD410",
                  "IP3M-941B", "IP4M-1056E"):
        check(static_capabilities(model)["night_vision"] is True,
              f"{model} advertises night_vision=true")
    for model in ("IP5M-T1277EW-AI", "IP8M-2779EW-AI", "AD410"):
        check(static_capabilities(model)["ptz"] is False,
              f"{model} advertises ptz=false")

    # white-light control contract: coax turrets are on/off ONLY (no brightness,
    # no 'auto'); Lighting_V2 models keep brightness + auto.
    for model in ("IP5M-T1277EW-AI", "IP8M-2779EW-AI"):
        wlc = white_light_control_for_model(model)
        check(wlc == {"brightness": False, "modes": ["off", "on"]},
              f"{model} white_light_control drops brightness + 'auto' (coax on/off only)")
    wlc = white_light_control_for_model("GENERIC-WL-CAM")
    check(wlc == {"brightness": True, "modes": ["off", "on", "auto"]},
          "non-coax model white_light_control keeps brightness + off/on/auto")

    pcm = pcm_frame()
    alaw = pcm16le_to_alaw(pcm)
    check(len(alaw) == len(pcm) // 2, "A-law output is 1 byte per Int16 sample")
    check(pcm16le_to_alaw(pcm + b"\x00") == alaw, "odd trailing byte tolerated")
    check(pcm16le_to_alaw(b"") == b"", "empty PCM -> empty A-law")
    two = pcm16le_to_alaw(pcm[:100]) + pcm16le_to_alaw(pcm[100:])
    check(two == alaw, "chunked transcode == whole-buffer transcode (stateless)")


# ---------------- HTTP route checks ----------------


def light_checks(client: TestClient, headers: dict) -> None:
    # The EW turret (IP5M-T1277EW-AI) drives its white spotlight over
    # coaxialControlIO, NOT Lighting_V2 (user-verified). Type=1, IO=1 on / 0 off.
    print("light route: coaxialControlIO CGI strings (EW turret)")
    FAKE.behavior = "turret"
    add_camera(client, headers, "turret", "IP5M-T1277EW-AI", "192.0.2.80")

    # The camera response advertises the on/off-only contract so the UI hides
    # the (dead) brightness slider and the 'auto' option for coax turrets.
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["turret"]["white_light_control"] == {"brightness": False, "modes": ["off", "on"]},
          "turret camera response advertises white_light_control on/off, brightness:false")

    FAKE.requests.clear()
    # brightness is DROPPED for coax turrets: still 204, still exactly one coax
    # call, and no Lighting_V2 brightness setConfig leaks.
    resp = client.post("/api/cameras/turret/light", headers=headers,
                       json={"mode": "on", "brightness": 70})
    check(resp.status_code == 204, "POST light {mode:on, brightness:70} -> 204 (brightness dropped)")
    coax = FAKE.coax_control_requests()
    check(len(coax) == 1, "exactly one coaxialControlIO control call")
    check(not any("Light" in k or "brightness" in k.lower() for k in coax[0].url.params.keys()),
          "coaxialControlIO control carries NO brightness parameter")
    req = coax[0]
    check(req.url.path == "/cgi-bin/coaxialControlIO.cgi"
          and req.url.params.get("action") == "control"
          and req.url.params.get("channel") == "1",
          "coaxialControlIO control on channel 1")
    check(req.url.params.get("info[0].Type") == "1",
          "white spotlight -> info[0].Type=1")
    check(req.url.params.get("info[0].IO") == "1", "mode 'on' -> info[0].IO=1")
    check(not FAKE.set_config_requests(),
          "EW turret spotlight issues NO Lighting_V2 setConfig")

    FAKE.requests.clear()
    resp = client.post("/api/cameras/turret/light", headers=headers, json={"mode": "auto"})
    check(resp.status_code == 204, "POST light {mode:auto} -> 204")
    p = FAKE.coax_control_requests()[0].url.params
    check(p.get("info[0].Type") == "1" and p.get("info[0].IO") == "1",
          "mode 'auto' -> illuminate (Type=1, IO=1)")

    FAKE.requests.clear()
    resp = client.post("/api/cameras/turret/light", headers=headers, json={"mode": "off"})
    check(resp.status_code == 204, "POST light {mode:off} -> 204")
    p = FAKE.coax_control_requests()[0].url.params
    check(p.get("info[0].Type") == "1" and p.get("info[0].IO") == "0",
          "mode 'off' -> info[0].IO=0")

    resp = client.post("/api/cameras/turret/light", headers=headers, json={"on": True})
    check(resp.status_code == 422, "old {on: bool} body shape rejected (422)")

    resp = client.post("/api/cameras/turret/light", headers=headers,
                       json={"mode": "on", "brightness": 101})
    check(resp.status_code == 422, "brightness > 100 rejected (422)")

    resp = client.post("/api/cameras/turret/light", headers=headers,
                       json={"mode": "on", "brightness": -1})
    check(resp.status_code == 422, "brightness < 0 rejected (422)")

    # Firmware without coaxialControlIO (HTTP 400/404) -> clean 501, never 500.
    FAKE.behavior = "coax_reject_unsupported"
    resp = client.post("/api/cameras/turret/light", headers=headers, json={"mode": "on"})
    check(resp.status_code == 501, "firmware without coaxialControlIO -> 501 (not 500)")

    # Flaky device: coaxialControlIO answers 200 "Error" -> transient 502.
    FAKE.behavior = "coax_reject_transient"
    resp = client.post("/api/cameras/turret/light", headers=headers, json={"mode": "on"})
    check(resp.status_code == 502, "transient coaxialControlIO rejection -> 502")
    FAKE.behavior = "turret"

    # Digest: the EW-turret client is built with HTTP Digest auth (stored
    # per-camera creds), same as every other CGI call.
    from app.amcrest.client import AmcrestClient as _AC
    _c = _AC("192.0.2.80", "admin", "pw", model="IP5M-T1277EW-AI")
    check(isinstance(_c._client.auth, httpx.DigestAuth),
          "coax control uses HTTP Digest auth with stored creds")


def lighting_v2_regression_check() -> None:
    # Non-coax models still use the Lighting_V2 white-LED CGI unchanged.
    print("white light: Lighting_V2 path preserved for non-coax models")
    from app.amcrest.client import AmcrestClient as _AC

    FAKE.behavior = "turret"
    FAKE.requests.clear()

    async def run():
        c = _AC("192.0.2.70", "admin", "pw", model="GENERIC-WL-CAM")
        await c.set_white_light(mode="on", brightness=70)
        wl = await c.get_white_light()
        await c.aclose()
        return wl

    wl = asyncio.run(run())
    sets = FAKE.set_config_requests()
    check(len(sets) == 1, "non-coax model still issues one Lighting_V2 setConfig")
    p = sets[0].url.params
    check(p.get("Lighting_V2[0][0][1].Mode") == "Manual"
          and p.get("Lighting_V2[0][0][1].MiddleLight[0].Light") == "70",
          "Lighting_V2 white-LED CGI strings unchanged for non-coax models")
    check(wl == {"mode": "on", "brightness": 55},
          "Lighting_V2 get_white_light still reads white-LED slot [1]")
    check(not FAKE.coax_control_requests(),
          "non-coax model issues NO coaxialControlIO request")


def settings_checks(client: TestClient, headers: dict) -> None:
    print("device settings: white_light passthrough (EW turret -> coax)")
    FAKE.behavior = "turret"
    FAKE.requests.clear()
    resp = client.get("/api/cameras/turret/settings", headers=headers)
    check(resp.status_code == 200, "GET settings -> 200")
    body = resp.json()
    # coaxialControlIO getStatus reports WhiteLight=On; brightness isn't
    # reported over that CGI, so it's fixed at 100.
    check(body.get("white_light") == {"mode": "on", "brightness": 100},
          "GET settings returns white_light {mode:'on', brightness:100} from coaxialControlIO")
    check(body.get("ir_mode") == "auto", "ir_mode still present (Lighting[0][0].Mode)")

    FAKE.requests.clear()
    resp = client.put("/api/cameras/turret/settings", headers=headers,
                      json={"white_light": {"mode": "auto", "brightness": 80}})
    check(resp.status_code == 200, "PUT settings white_light patch -> 200")
    coax = FAKE.coax_control_requests()
    check(len(coax) == 1, "white_light patch -> one coaxialControlIO control call")
    params = coax[0].url.params
    check(params.get("info[0].Type") == "1" and params.get("info[0].IO") == "1",
          "PUT settings 'auto' illuminates spotlight (Type=1, IO=1)")
    check(not FAKE.set_config_requests(),
          "EW-turret white_light patch issues NO Lighting_V2 setConfig")


def cgi_accepted_checks() -> None:
    """cgi_accepted: the single gate every Dahua CGI success/failure passes.

    This bug class has bitten twice: _ptz/reboot accepted any body CONTAINING
    o-k, and the same trap in set_current_time helped hide the clock bug. The
    check must be safe in BOTH directions — no false success on an error body,
    no false failure on a decorated success."""
    print("cgi body check: whole-word 'ok', safe in both directions")

    # Successes — every shape a firmware might plausibly answer.
    for body in ("OK", "ok", "OK\r\n", "  OK  \r\n", "Preset OK\r\n", "OK: preset 1 set"):
        check(cgi_accepted(body) is True, f"success body {body!r} -> accepted")

    # Error bodies that CONTAIN the letters o-k. Each of these was reported as
    # SUCCESS by the old `"ok" in text.lower()` check.
    check(cgi_accepted("Error") is False, "'Error' -> rejected")
    check(cgi_accepted("Error\r\n") is False, "'Error\\r\\n' -> rejected")
    check(cgi_accepted("Broken") is False, "'Broken' (Br-OK-en) -> rejected")
    check(cgi_accepted("Error: Invalid token") is False,
          "'Error: Invalid token' (t-OK-en) -> rejected")
    check(cgi_accepted("<html>Set-Cookie: x</html>") is False,
          "'Set-Cookie' (co-OK-ie) -> rejected")
    # Case-sensitivity alone would MISS this one — TOKEN shouts an uppercase OK.
    check(cgi_accepted("Error: TOKEN invalid") is False,
          "'Error: TOKEN invalid' -> rejected (a case-sensitive test accepts it)")
    check(cgi_accepted("") is False, "empty body -> rejected")


def model_match_checks() -> None:
    """match_known_model: the gate every auto-detected model passes through.

    It must FAIL CLOSED. A wrong-but-confident model is worse than "unknown",
    because the capability keys it corrupts (white_light, ptz, ai_on_camera)
    are static-only — not in features.PROBE_KEYS — so no later probe can ever
    undo it. "unknown" merely promises nothing."""
    print("model auto-detect: match_known_model fails closed")
    match_known_model = cameras_router.match_known_model
    KNOWN_MODELS = cameras_router.KNOWN_MODELS

    # Exact device strings.
    for model in KNOWN_MODELS:
        check(match_known_model(model) == model, f"exact {model} -> itself")

    # A truncation that extends to exactly one known model resolves.
    check(match_known_model("IP5M-T1277EW") == "IP5M-T1277EW-AI",
          "truncated 'IP5M-T1277EW' (no -AI) -> IP5M-T1277EW-AI")
    check(match_known_model("IP8M-2779EW") == "IP8M-2779EW-AI",
          "truncated 'IP8M-2779EW' (no -AI) -> IP8M-2779EW-AI")
    check(match_known_model("IP3M-941") == "IP3M-941B", "truncated 'IP3M-941' -> IP3M-941B")

    # THE REAL FLEET (verified by the live audit 2026-07-16 — every string below
    # is what the hardware actually answered, not an assumption). Note the
    # turrets DO report the -AI suffix, contrary to what this suite's fake long
    # claimed; the truncation cases above are kept as defensive coverage.
    check(match_known_model("AD410") == "AD410",
          "live AD410 (192.168.1.39) reports a bare 'AD410'")
    check(match_known_model("IP3M-941B") == "IP3M-941B",
          "live IP3M-941B (192.168.1.135) answers getDeviceType exactly")
    check(match_known_model("IP5M-T1277EW-AI") == "IP5M-T1277EW-AI",
          "live turrets report the FULL string including -AI")
    check(match_known_model("IP4M-1056E") == "IP4M-1056E",
          "live IP4M-1056E reports exactly")
    # The alias: firmware drops the 'W' (white-housing suffix) from its own id.
    # Neither equal nor a prefix of the stored name, so only the alias saves it.
    check(match_known_model("IP8M-2779E-AI") == "IP8M-2779EW-AI",
          "live IP8M (192.168.1.87/.88) reports 'IP8M-2779E-AI' (no W) -> aliased")
    check(match_known_model("ip5m-t1277ew") == "IP5M-T1277EW-AI", "match is case-insensitive")
    check(match_known_model("  AD410  ") == "AD410", "surrounding whitespace tolerated")

    # A LONGER, genuinely different device string must not adopt a shorter
    # known model. 'IP4M-1056EW-AI' is not an 'IP4M-1056E' -- that model's map
    # pins mic/ai_on_camera False, which a probe can never restore.
    check(match_known_model("IP4M-1056EW-AI") is None,
          "'IP4M-1056EW-AI' does NOT pose as IP4M-1056E")
    check(match_known_model("AD410B") is None, "'AD410B' does NOT pose as AD410")
    check(match_known_model("IPC-HDW2231T") is None, "unlisted third-party model -> None")

    # Inconclusive input must NEVER produce a write. None is what an offline or
    # auth-failed camera yields; writing on it would cost a good camera its
    # model on the first network blip.
    check(match_known_model(None) is None, "None (offline/auth-fail) -> None, never a write")
    check(match_known_model("") is None, "empty device type -> None")
    check(match_known_model("IP5") is None, "junk shorter than 'AD410' -> None")

    # Ambiguity: only one IP5M model exists today, so 'IP5M-' resolves. The
    # rule must fail closed the moment a second one is added.
    check(match_known_model("IP5M-") == "IP5M-T1277EW-AI",
          "'IP5M-' resolves while it uniquely prefixes one known model")
    original = cameras_router.KNOWN_MODELS
    try:
        # match_known_model reads the module global at call time, so patching
        # it here genuinely exercises the ambiguity branch.
        cameras_router.KNOWN_MODELS = original + ("IP5M-X9999EW-AI",)
        check(cameras_router.match_known_model("IP5M-") is None,
              "'IP5M-' -> None once a 2nd IP5M model makes it ambiguous")
    finally:
        cameras_router.KNOWN_MODELS = original


def probe_checks(client: TestClient, headers: dict) -> None:
    print("probe route: success / auth-fail / unreachable / model adoption")
    FAKE.behavior = "unreachable"
    add_camera(client, headers, "mystery", "unknown", "192.0.2.90")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["mystery"]["model"] == "unknown", "camera starts with model 'unknown'")

    resp = client.post("/api/cameras/mystery/probe", headers=headers)
    check(resp.status_code == 200, "probe (unreachable) -> 200 envelope")
    body = resp.json()
    check(body["ok"] is False and body["detail"] == "camera unreachable"
          and body["model"] is None,
          "unreachable -> ok:false, detail 'camera unreachable'")

    FAKE.behavior = "auth_fail"
    body = client.post("/api/cameras/mystery/probe", headers=headers).json()
    check(body["ok"] is False and body["detail"] == "authentication failed",
          "401 from device -> detail 'authentication failed'")

    FAKE.behavior = "turret"
    FAKE.requests.clear()
    body = client.post("/api/cameras/mystery/probe", headers=headers).json()
    check(body["ok"] is True and body["detail"] is None, "reachable probe -> ok:true")
    check(body["model"] == "IP5M-T1277EW-AI",
          "getDeviceType 'IP5M-T1277EW' adopted as IP5M-T1277EW-AI")
    check(body["capabilities"]["white_light"] is True and body["capabilities"]["ir"] is True,
          "capabilities refreshed (white_light true for adopted turret)")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["mystery"]["model"] == "IP5M-T1277EW-AI", "adopted model persisted in DB")

    resp = client.post("/api/cameras/nope/probe", headers=headers)
    check(resp.status_code == 404, "probe unknown camera -> 404")


def doorbell_audio_caps_check(client: TestClient, headers: dict) -> None:
    # Regression for the two-way-talk fix: an AD410 whose devAudioInput/Output
    # collect CGIs report 0 (or are unsupported) must STILL advertise speaker +
    # mic, or the talk WS would reject with 4003 and two-way talk never works.
    print("doorbell mic: AD410 speaker/mic survive a zero audio-collect probe")
    FAKE.behavior = "ad410_no_audiocollect"
    add_camera(client, headers, "bell_probe", "unknown", "192.0.2.95")
    body = client.post("/api/cameras/bell_probe/probe", headers=headers).json()
    check(body["ok"] is True, "AD410 probe ok")
    check(body["model"] == "AD410", "getDeviceType 'AD410' adopted")
    check(body["capabilities"]["speaker"] is True,
          "AD410 keeps speaker=True despite devAudioOutput result=0")
    check(body["capabilities"]["mic"] is True,
          "AD410 keeps mic=True despite devAudioInput result=0")
    FAKE.behavior = "turret"


def siren_checks(client: TestClient, headers: dict) -> None:
    print("siren regression: shared postAudio path")
    FAKE.behavior = "ad410"
    add_camera(client, headers, "doorbell", "AD410", "127.0.0.1")
    FAKE.requests.clear()
    resp = client.post("/api/cameras/doorbell/siren", headers=headers, json={"duration_s": 1})
    check(resp.status_code == 204, "POST siren -> 204")
    posts = [r for r in FAKE.requests if r.url.path == "/cgi-bin/audio.cgi"]
    check(len(posts) == 1, "one audio.cgi call")
    req = posts[0]
    check(req.url.params.get("action") == "postAudio"
          and req.url.params.get("httptype") == "singlepart"
          and req.url.params.get("channel") == "1",
          "postAudio query params intact (python-amcrest shape)")
    check(req.headers.get("content-type") == "Audio/G.711A", "Content-Type Audio/G.711A")
    check(req.content == make_alarm_tone(1), "exact G.711A tone payload sent")


def go2rtc_regression_checks(client: TestClient, headers: dict) -> None:
    print("native regression: CRUD + go2rtc config regen")
    cfg_path = Path(os.environ["GO2RTC_CONFIG_DIR"]) / "go2rtc.yaml"
    text = cfg_path.read_text()
    check("turret:" in text and "doorbell:" in text,
          "go2rtc config regenerated with added cameras")
    check("turret_sub:" in text and "doorbell_sub:" in text,
          "each camera gets a _sub detect stream")

    FAKE.behavior = "turret"
    resp = client.put("/api/cameras/turret", headers=headers, json={
        "name": "turret", "friendly_name": "Turret Renamed",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.80",
        "username": "", "password": "",
    })
    check(resp.status_code == 200 and resp.json()["friendly_name"] == "Turret Renamed",
          "PUT camera edit still works (blank creds keep stored)")

    add_camera(client, headers, "temp_cam", "IP8M-2779EW-AI", "192.0.2.81")
    check("temp_cam:" in cfg_path.read_text(), "config regen after add")

    # Native additions: URL overrides + detect_fps round-trip into the
    # camera response and the generated go2rtc config.
    resp = client.put("/api/cameras/temp_cam", headers=headers, json={
        "name": "temp_cam", "friendly_name": "Temp Cam",
        "model": "IP8M-2779EW-AI", "ip": "192.0.2.81",
        "username": "", "password": "",
        "main_url": "rtsp://192.0.2.81:7447/main", "detect_fps": 7,
    })
    check(resp.status_code == 200
          and resp.json()["main_url"] == "rtsp://192.0.2.81:7447/main"
          and resp.json()["sub_url"] == ""
          and resp.json()["detect_fps"] == 7,
          "main_url/detect_fps stored and returned; sub_url stays default")
    check("rtsp://192.0.2.81:7447/main" in cfg_path.read_text(),
          "main_url override wins in the generated go2rtc config")
    resp = client.put("/api/cameras/temp_cam", headers=headers, json={
        "name": "temp_cam", "friendly_name": "Temp Cam",
        "model": "IP8M-2779EW-AI", "ip": "192.0.2.81",
        "username": "", "password": "",
        "main_url": "http://not-rtsp",
    })
    check(resp.status_code == 422, "non-rtsp URL override rejected (422)")

    resp = client.delete("/api/cameras/temp_cam", headers=headers)
    check(resp.status_code == 204, "DELETE camera -> 204")
    check("temp_cam:" not in cfg_path.read_text(), "config regen after delete")


def audio_codec_checks(client: TestClient, headers: dict) -> None:
    """Per-camera live-view audio codec preference: default 'g711a'
    (WebRTC-legal, live-view audio works) and an opt-in 'aac' (higher recording
    quality, no live-view audio). PUT accepts + persists it, GET round-trips it,
    an invalid codec -> 400. The on-change device re-provision is best-effort:
    the fake device has no Encode table, so provision_audio fails silently and
    the PUT still succeeds (never a 500)."""
    print("audio codec: live-view codec preference (g711a default | aac) round-trip")
    FAKE.behavior = "turret"
    add_camera(client, headers, "audiocam", "IP5M-T1277EW-AI", "192.0.2.90")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["audiocam"]["audio_codec"] == "g711a",
          "a new camera defaults to audio_codec=g711a (live-view audio works)")

    # PUT audio_codec=aac -> accepted, returned, and round-trips in GET.
    resp = client.put("/api/cameras/audiocam", headers=headers, json={
        "name": "audiocam", "friendly_name": "Audio Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.90",
        "username": "", "password": "", "audio_codec": "aac",
    })
    check(resp.status_code == 200 and resp.json()["audio_codec"] == "aac",
          "PUT audio_codec=aac accepted + returned in the camera response (never 500)")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["audiocam"]["audio_codec"] == "aac",
          "audio_codec=aac round-trips in GET /api/cameras")

    # An invalid codec is rejected with a clean 400 (not 422/500).
    resp = client.put("/api/cameras/audiocam", headers=headers, json={
        "name": "audiocam", "friendly_name": "Audio Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.90",
        "username": "", "password": "", "audio_codec": "opus",
    })
    check(resp.status_code == 400, "PUT with an invalid audio_codec -> 400")

    # Back to the g711a default, still round-tripping.
    resp = client.put("/api/cameras/audiocam", headers=headers, json={
        "name": "audiocam", "friendly_name": "Audio Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.90",
        "username": "", "password": "", "audio_codec": "g711a",
    })
    check(resp.status_code == 200 and resp.json()["audio_codec"] == "g711a",
          "PUT audio_codec back to g711a accepted + returned")

    client.delete("/api/cameras/audiocam", headers=headers)


def smart_spotlight_checks(client: TestClient, headers: dict) -> None:
    """Per-camera "Smart spotlight" flag: default false, PUT persists true/false,
    GET round-trips it, and the PUT is persist-only (NO device call — the
    controller reads the stored flag live on the next person-at-night)."""
    print("smart spotlight: per-camera flag (default false) round-trip")
    FAKE.behavior = "turret"
    add_camera(client, headers, "spotcam", "IP5M-T1277EW-AI", "192.0.2.95")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["spotcam"]["smart_spotlight"] is False,
          "a new camera defaults to smart_spotlight=false")
    check(cams["spotcam"]["spotlight_hold_seconds"] == 60,
          "a new camera defaults to spotlight_hold_seconds=60")
    check(cams["spotcam"]["capabilities"]["white_light"] is True,
          "spotcam (EW turret) has the white_light capability the feature needs")

    # PUT smart_spotlight=true -> accepted, returned, round-trips in GET.
    resp = client.put("/api/cameras/spotcam", headers=headers, json={
        "name": "spotcam", "friendly_name": "Spot Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
        "username": "", "password": "", "smart_spotlight": True,
    })
    check(resp.status_code == 200 and resp.json()["smart_spotlight"] is True,
          "PUT smart_spotlight=true accepted + returned in the camera response")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["spotcam"]["smart_spotlight"] is True,
          "smart_spotlight=true round-trips in GET /api/cameras")

    # Omitting smart_spotlight keeps the stored value (a settings-only PUT).
    resp = client.put("/api/cameras/spotcam", headers=headers, json={
        "name": "spotcam", "friendly_name": "Spot Cam Renamed",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
        "username": "", "password": "",
    })
    check(resp.status_code == 200 and resp.json()["smart_spotlight"] is True,
          "a PUT that omits smart_spotlight keeps the stored true")

    # PUT smart_spotlight=false -> turns it back off.
    resp = client.put("/api/cameras/spotcam", headers=headers, json={
        "name": "spotcam", "friendly_name": "Spot Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
        "username": "", "password": "", "smart_spotlight": False,
    })
    check(resp.status_code == 200 and resp.json()["smart_spotlight"] is False,
          "PUT smart_spotlight=false accepted + returned (toggled back off)")

    # --- spotlight_hold_seconds: validated int in [5, 600], round-trips in GET ---
    # PUT a valid non-default hold (30) -> accepted, returned, round-trips.
    resp = client.put("/api/cameras/spotcam", headers=headers, json={
        "name": "spotcam", "friendly_name": "Spot Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
        "username": "", "password": "", "spotlight_hold_seconds": 30,
    })
    check(resp.status_code == 200 and resp.json()["spotlight_hold_seconds"] == 30,
          "PUT spotlight_hold_seconds=30 accepted + returned in the camera response")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["spotcam"]["spotlight_hold_seconds"] == 30,
          "spotlight_hold_seconds=30 round-trips in GET /api/cameras")

    # Omitting spotlight_hold_seconds keeps the stored value (a settings-only PUT).
    resp = client.put("/api/cameras/spotcam", headers=headers, json={
        "name": "spotcam", "friendly_name": "Spot Cam",
        "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
        "username": "", "password": "", "smart_spotlight": True,
    })
    check(resp.status_code == 200 and resp.json()["spotlight_hold_seconds"] == 30,
          "a PUT that omits spotlight_hold_seconds keeps the stored 30")

    # Out-of-range values -> 400 (both below the 5 floor and above the 600 ceiling)
    # and the stored value is left untouched.
    for bad in (0, 9999):
        resp = client.put("/api/cameras/spotcam", headers=headers, json={
            "name": "spotcam", "friendly_name": "Spot Cam",
            "model": "IP5M-T1277EW-AI", "ip": "192.0.2.95",
            "username": "", "password": "", "spotlight_hold_seconds": bad,
        })
        check(resp.status_code == 400,
              f"PUT spotlight_hold_seconds={bad} (out of range) -> 400")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["spotcam"]["spotlight_hold_seconds"] == 30,
          "a rejected out-of-range hold leaves the stored value unchanged (still 30)")

    client.delete("/api/cameras/spotcam", headers=headers)


# ---------------- Smart spotlight: sun + controller (unit) ----------------


# NYC (config default lat/lon) — clear day/night instants far from the horizon.
_NYC_LAT, _NYC_LON = 40.71, -74.01
# 2026-07-15 16:00Z = 12:00 EDT (solar noon-ish) -> daytime.
_DAY_EPOCH = datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc).timestamp()
# 2026-07-15 05:00Z = 01:00 EDT -> the dead of night.
_NIGHT_EPOCH = datetime(2026, 7, 15, 5, 0, tzinfo=timezone.utc).timestamp()


def sun_checks() -> None:
    print("sun: is_night day vs night for a known lat/lon")
    check(sun_mod.is_night(_DAY_EPOCH, _NYC_LAT, _NYC_LON) is False,
          "is_night(NYC noon EDT) == False (daytime)")
    check(sun_mod.is_night(_NIGHT_EPOCH, _NYC_LAT, _NYC_LON) is True,
          "is_night(NYC 01:00 EDT) == True (night)")
    ev = sun_mod.sun_events(_DAY_EPOCH, _NYC_LAT, _NYC_LON)
    check(ev.sunrise is not None and ev.sunset is not None and ev.sunrise < ev.sunset,
          "sun_events yields a sunrise before sunset on a normal day")
    # Polar edge cases never crash and pick the right constant.
    check(sun_mod.is_night(datetime(2026, 6, 21, 12, tzinfo=timezone.utc).timestamp(),
                           78.0, 15.0) is False,
          "midnight-sun latitude in June -> is_night False (never crashes)")
    check(sun_mod.is_night(datetime(2026, 12, 21, 12, tzinfo=timezone.utc).timestamp(),
                           78.0, 15.0) is True,
          "polar-night latitude in December -> is_night True (never crashes)")


class _StubLightClient:
    """Records set_white_light(mode) calls into a shared log; no real device."""

    def __init__(self, cam: dict, log: list) -> None:
        self._cam = cam
        self._log = log

    async def set_white_light(self, mode=None, brightness=None) -> None:
        self._log.append((self._cam["name"], mode))

    async def aclose(self) -> None:
        pass


class _FakeConfig:
    latitude = _NYC_LAT
    longitude = _NYC_LON


_WL_CAM = {
    "name": "turret", "model": "IP5M-T1277EW-AI", "ip": "192.0.2.200",
    "username": "u", "password": "p",
    "smart_spotlight": True, "capabilities": {"white_light": True},
}


def _spotlight_controller(log: list, *, night: bool):
    def factory(cam):
        return _StubLightClient(cam, log)

    return SpotlightController(
        _FakeConfig(),
        client_factory=factory,
        hold_s=0.03,
        now=lambda: 1_000_000.0,
        is_night=lambda now, lat, lon: night,
    )


async def _drain(ctrl: SpotlightController) -> None:
    for _ in range(50):
        if not ctrl._tasks:
            break
        await asyncio.gather(*list(ctrl._tasks), return_exceptions=True)
        await asyncio.sleep(0)


def spotlight_controller_checks() -> None:
    print("spotlight controller: person@night arms once, holds, then turns off")

    # (1) person @ night, enabled, white_light -> set_white_light("on") ONCE;
    # a second person does NOT re-send "on"; after the hold with no person ->
    # set_white_light("off").
    async def on_hold_off() -> None:
        log: list = []
        ctrl = _spotlight_controller(log, night=True)
        ctrl.notify_person(_WL_CAM)
        await _drain(ctrl)
        check(log == [("turret", "on")],
              "person@night+enabled+white_light -> set_white_light('on') once")
        ctrl.notify_person(_WL_CAM)  # second person while already on
        await _drain(ctrl)
        check(log == [("turret", "on")],
              "a second person notify does NOT re-send 'on' (debounced)")
        await asyncio.sleep(0.06)  # exceed the 0.03 s hold with no new person
        await asyncio.sleep(0)
        check(log == [("turret", "on"), ("turret", "off")],
              "after the hold with no person -> set_white_light('off') once")
        await ctrl.stop_all()

    asyncio.run(on_hold_off())

    # (2) new person before the hold expires RESETS the timer (no early off).
    async def hold_reset() -> None:
        log: list = []
        ctrl = _spotlight_controller(log, night=True)
        ctrl.notify_person(_WL_CAM)
        await _drain(ctrl)
        await asyncio.sleep(0.02)          # < hold
        ctrl.notify_person(_WL_CAM)        # resets the 0.03 s timer
        await _drain(ctrl)
        await asyncio.sleep(0.02)          # 0.04 s total, but < 0.03 since reset
        check(log == [("turret", "on")],
              "a person before the hold expires resets the timer (light stays on)")
        await asyncio.sleep(0.04)          # now let the reset hold elapse
        await asyncio.sleep(0)
        check(log == [("turret", "on"), ("turret", "off")],
              "the light turns off 60 s after the LAST person (hold is trailing)")
        await ctrl.stop_all()

    asyncio.run(hold_reset())

    # (3) DAY -> never turns on.
    async def day_noop() -> None:
        log: list = []
        ctrl = _spotlight_controller(log, night=False)
        ctrl.notify_person(_WL_CAM)
        await _drain(ctrl)
        await asyncio.sleep(0.05)
        check(log == [], "DAY: a person never turns the spotlight on")
        await ctrl.stop_all()

    asyncio.run(day_noop())

    # (4) disabled (smart_spotlight false) -> never turns on, even at night.
    async def disabled_noop() -> None:
        log: list = []
        ctrl = _spotlight_controller(log, night=True)
        cam = {**_WL_CAM, "smart_spotlight": False}
        ctrl.notify_person(cam)
        await _drain(ctrl)
        await asyncio.sleep(0.05)
        check(log == [], "disabled camera (smart_spotlight false): never turns on")
        await ctrl.stop_all()

    asyncio.run(disabled_noop())

    # (5) non-white_light camera -> never turns on, even enabled + at night.
    async def no_white_light_noop() -> None:
        log: list = []
        ctrl = _spotlight_controller(log, night=True)
        cam = {**_WL_CAM, "model": "AD410", "capabilities": {"white_light": False}}
        ctrl.notify_person(cam)
        await _drain(ctrl)
        await asyncio.sleep(0.05)
        check(log == [], "non-white_light camera: never turns on")
        await ctrl.stop_all()

    asyncio.run(no_white_light_noop())

    # (6) per-camera hold: the trailing off fires after the camera's OWN
    # spotlight_hold_seconds, NOT the 60 default. Uses an injected sleep that
    # compresses seconds 500x (the tiny-interval approach) so a real hold value
    # stays fast to test: a 30 s hold -> 0.06 s wait, the 60 default -> 0.12 s.
    async def per_camera_hold() -> None:
        log: list = []
        ctrl = SpotlightController(
            _FakeConfig(),
            client_factory=lambda cam: _StubLightClient(cam, log),
            now=lambda: 1_000_000.0,
            is_night=lambda now, lat, lon: True,
            sleep=lambda s: asyncio.sleep(s * 0.002),
        )
        cam = {**_WL_CAM, "spotlight_hold_seconds": 30}  # 30 * 0.002 = 0.06 s
        ctrl.notify_person(cam)
        await _drain(ctrl)
        check(log == [("turret", "on")],
              "person@night -> on (per-camera hold path)")
        await asyncio.sleep(0.03)          # < 0.06 (the 30 s hold), > 0 -> still on
        await asyncio.sleep(0)
        check(log == [("turret", "on")],
              "still on before the per-camera hold elapses")
        await asyncio.sleep(0.06)          # total ~0.09: > 0.06 (30 s) but < 0.12 (60)
        await asyncio.sleep(0)
        check(log == [("turret", "on"), ("turret", "off")],
              "spotlight turns OFF after the per-camera hold (30 s), not the 60 default")
        await ctrl.stop_all()

    asyncio.run(per_camera_hold())

    # (7) _resolve_hold: the stored value is clamped to [5, 600]; missing / None /
    # non-numeric falls back to the 60 default (matches the API's 5..600 contract).
    def resolve_hold() -> None:
        ctrl = SpotlightController(_FakeConfig())  # default hold_s = 60
        check(ctrl._resolve_hold({"spotlight_hold_seconds": 30}) == 30.0,
              "an in-range hold (30) is used as-is")
        check(ctrl._resolve_hold({"spotlight_hold_seconds": 5}) == 5.0,
              "the floor (5) is used as-is")
        check(ctrl._resolve_hold({"spotlight_hold_seconds": 600}) == 600.0,
              "the ceiling (600) is used as-is")
        check(ctrl._resolve_hold({"spotlight_hold_seconds": 0}) == 5.0,
              "a below-range hold (0) clamps up to 5")
        check(ctrl._resolve_hold({"spotlight_hold_seconds": 9999}) == 600.0,
              "an above-range hold (9999) clamps down to 600")
        check(ctrl._resolve_hold({"spotlight_hold_seconds": None}) == 60.0,
              "a None hold falls back to the 60 default")
        check(ctrl._resolve_hold({}) == 60.0,
              "a missing hold falls back to the 60 default")

    resolve_hold()


# ---------------- PTZ ----------------


def _ptz_requests() -> list[httpx.Request]:
    return [r for r in FAKE.requests if r.url.path == "/cgi-bin/ptz.cgi"]


def ptz_checks(client: TestClient, headers: dict) -> None:
    """PTZ dome (IP3M-941B): exact ptz.cgi CGI strings for move / stop /
    preset, capability gating, and validation bounds."""
    print("PTZ: ptz.cgi move/stop/preset CGI strings + gating")
    FAKE.behavior = "turret"
    add_camera(client, headers, "ptzcam", "IP3M-941B", "192.0.2.86")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["ptzcam"]["capabilities"]["ptz"] is True,
          "IP3M-941B advertises ptz=True in the camera response")

    # -- continuous move: diagonal maps to Dahua's Left/Right-prefixed code --
    FAKE.requests.clear()
    resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                       json={"action": "move", "direction": "upleft", "speed": 5})
    check(resp.status_code == 204, "POST ptz move upleft speed 5 -> 204")
    ptz = _ptz_requests()
    check(len(ptz) == 1, "exactly one ptz.cgi call")
    p = ptz[0].url.params
    check(p.get("action") == "start" and p.get("channel") == "1"
          and p.get("code") == "LeftUp" and p.get("arg1") == "0"
          and p.get("arg2") == "5" and p.get("arg3") == "0",
          "move upleft -> action=start&channel=1&code=LeftUp&arg1=0&arg2=5&arg3=0")

    # -- stop uses the same code, action=stop --
    FAKE.requests.clear()
    resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                       json={"action": "stop", "direction": "upleft"})
    check(resp.status_code == 204, "POST ptz stop -> 204")
    p = _ptz_requests()[0].url.params
    check(p.get("action") == "stop" and p.get("code") == "LeftUp",
          "stop -> action=stop with the same direction code")

    # -- step: ONE bounded nudge = server-side start -> brief dwell -> stop
    #    (fixes the hold-to-move runaway). Exactly two ptz.cgi calls: a start
    #    at the small STEP_SPEED, then a stop with the SAME direction code. --
    FAKE.requests.clear()
    resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                       json={"action": "step", "direction": "right"})
    check(resp.status_code == 204, "POST ptz step right -> 204")
    ptz = _ptz_requests()
    check(len(ptz) == 2, "step issues exactly two ptz.cgi calls (start then stop)")
    start_p, stop_p = ptz[0].url.params, ptz[1].url.params
    check(start_p.get("action") == "start" and start_p.get("code") == "Right"
          and start_p.get("arg2") == "2",
          "step start -> action=start&code=Right&arg2=2 (STEP_SPEED default 2)")
    check(stop_p.get("action") == "stop" and stop_p.get("code") == "Right",
          "step stop -> action=stop&code=Right (same direction, server-issued)")
    check(client.post("/api/cameras/ptzcam/ptz", headers=headers,
                      json={"action": "step"}).status_code == 422,
          "step without a direction rejected (422)")

    # -- speed clamps to 1..8 at the client (Field bounds are 1..8, but the
    #    client clamp is the last line of defence) --
    FAKE.requests.clear()
    resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                       json={"action": "move", "direction": "right", "speed": 8})
    check(resp.status_code == 204 and _ptz_requests()[0].url.params.get("arg2") == "8",
          "move right speed 8 -> arg2=8 (max speed)")

    # -- presets 1..3: SetPreset / GotoPreset / ClearPreset, arg2=index --
    for action, code in (("preset_set", "SetPreset"),
                         ("preset_goto", "GotoPreset"),
                         ("preset_clear", "ClearPreset")):
        FAKE.requests.clear()
        resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                           json={"action": action, "index": 2})
        check(resp.status_code == 204, f"POST ptz {action} index 2 -> 204")
        p = _ptz_requests()[0].url.params
        check(p.get("action") == "start" and p.get("code") == code
              and p.get("arg2") == "2",
              f"{action} -> action=start&code={code}&arg2=2")

    # -- validation: bad direction, preset index out of range, missing fields --
    check(client.post("/api/cameras/ptzcam/ptz", headers=headers,
                      json={"action": "move", "direction": "sideways"}).status_code == 422,
          "invalid direction rejected (422)")
    check(client.post("/api/cameras/ptzcam/ptz", headers=headers,
                      json={"action": "preset_goto", "index": 4}).status_code == 422,
          "preset index > 3 rejected (422)")
    check(client.post("/api/cameras/ptzcam/ptz", headers=headers,
                      json={"action": "move"}).status_code == 422,
          "move without a direction rejected (422)")
    check(client.post("/api/cameras/ptzcam/ptz", headers=headers,
                      json={"action": "preset_goto"}).status_code == 422,
          "preset without an index rejected (422)")

    # -- capability gating: a non-PTZ camera -> 400 --
    check(client.post("/api/cameras/turret/ptz", headers=headers,
                      json={"action": "stop", "direction": "left"}).status_code == 400,
          "ptz on a non-PTZ camera (turret) -> 400")

    # -- transient device rejection -> 502 --
    FAKE.behavior = "ptz_reject_transient"
    resp = client.post("/api/cameras/ptzcam/ptz", headers=headers,
                       json={"action": "stop", "direction": "left"})
    check(resp.status_code == 502, "transient ptz.cgi rejection -> 502")
    FAKE.behavior = "turret"


def _nv_mode_params() -> dict[str, str]:
    """Every VideoInDayNight[0][N].Mode written across the captured setConfigs."""
    out: dict[str, str] = {}
    for s in FAKE.set_config_requests():
        for k in s.url.params.keys():
            if k.startswith("VideoInDayNight[0][") and k.endswith(".Mode"):
                out[k] = s.url.params.get(k)
    return out


def _lighting_mode_params() -> dict[str, str]:
    """Every Lighting[0][N].Mode written across the captured setConfigs."""
    out: dict[str, str] = {}
    for s in FAKE.set_config_requests():
        for k in s.url.params.keys():
            if k.startswith("Lighting[0][") and k.endswith(".Mode"):
                out[k] = s.url.params.get(k)
    return out


def night_vision_checks(client: TestClient, headers: dict) -> None:
    """Night-vision mode (now on ALL models — it replaces the retired IR
    button): night_vision_mode writes the VideoInDayNight Mode to EVERY exposed
    profile with the auto->Brightness / color->Color / bw->BlackWhite mapping,
    couples the IR Lighting Mode to Auto, GET returns it, and gating still 400s
    a camera that genuinely lacks the capability."""
    print("night vision: VideoInDayNight[0][N].Mode on ALL profiles + IR=Auto coupling")
    FAKE.behavior = "turret"
    add_camera(client, headers, "nightcam", "IP4M-1056E", "192.0.2.85")
    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["nightcam"]["capabilities"]["night_vision"] is True,
          "IP4M-1056E advertises night_vision=True")
    check(cams["nightcam"]["capabilities"]["ptz"] is False,
          "IP4M-1056E advertises ptz=False")
    # Every model now exposes night_vision (the turret included).
    check(cams["turret"]["capabilities"]["night_vision"] is True,
          "the EW turret now also advertises night_vision=True")

    # -- PUT night_vision_mode 'bw' writes Mode=BlackWhite on ALL three profiles
    #    AND couples the IR Lighting Mode to Auto on all Lighting profiles. --
    FAKE.requests.clear()
    resp = client.put("/api/cameras/nightcam/settings", headers=headers,
                      json={"night_vision_mode": "bw"})
    check(resp.status_code == 200, "PUT night_vision_mode 'bw' -> 200")
    nv = _nv_mode_params()
    check(all(nv.get(f"VideoInDayNight[0][{p}].Mode") == "BlackWhite" for p in range(3)),
          "night_vision_mode 'bw' -> Mode=BlackWhite on ALL three VideoInDayNight profiles")
    lit = _lighting_mode_params()
    check(bool(lit) and all(v == "Auto" for v in lit.values()),
          "night_vision set couples IR Lighting[0][N].Mode=Auto on every profile")

    # -- 'color' -> Mode=Color --
    FAKE.requests.clear()
    resp = client.put("/api/cameras/nightcam/settings", headers=headers,
                      json={"night_vision_mode": "color"})
    check(resp.status_code == 200, "PUT night_vision_mode 'color' -> 200")
    check(_nv_mode_params().get("VideoInDayNight[0][0].Mode") == "Color",
          "night_vision_mode 'color' -> Mode=Color")

    # -- 'auto' -> Mode=Brightness (NOT the Dahua 'Auto' enum the turret rejects) --
    FAKE.requests.clear()
    resp = client.put("/api/cameras/nightcam/settings", headers=headers,
                      json={"night_vision_mode": "auto"})
    check(resp.status_code == 200, "PUT night_vision_mode 'auto' -> 200")
    check(all(_nv_mode_params().get(f"VideoInDayNight[0][{p}].Mode") == "Brightness"
              for p in range(3)),
          "night_vision_mode 'auto' -> Mode=Brightness on ALL profiles (turret-safe)")

    # -- GET settings returns the flat night_vision_mode (device reports Color) --
    FAKE.requests.clear()
    body = client.get("/api/cameras/nightcam/settings", headers=headers).json()
    check(body.get("night_vision_mode") == "color",
          "GET settings returns night_vision_mode 'color' from VideoInDayNight")

    # -- night_vision is now UNIVERSAL: every model AND the unknown/offline
    #    fallback advertises it (the night-vision control replaces the retired IR
    #    button on all cameras), so even an unknown-model camera accepts the mode. --
    FAKE.behavior = "unreachable"
    add_camera(client, headers, "nonv", "unknown", "192.0.2.199")
    FAKE.behavior = "turret"
    nonv = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(nonv["nonv"]["capabilities"]["night_vision"] is True,
          "unknown-model camera now also advertises night_vision=True (universal)")
    check(client.put("/api/cameras/nonv/settings", headers=headers,
                     json={"night_vision_mode": "color"}).status_code == 200,
          "night_vision_mode accepted on every camera (universal night_vision)")


# ---------------- talk WS ----------------


def talk_checks(client: TestClient, headers: dict, token: str, server: FakeCameraServer) -> None:
    print("talk WS: alaw stream, busy, gating, close codes")
    # The CGI postAudio happy-path (A-law arrival, busy, stop-on-close, camera
    # reject, session cap) drives a NON-backchannel speaker camera so talk.py
    # routes to AmcrestClient.talk_stream (audio.cgi postAudio) against the fake
    # camera HTTP server — exactly as before the backchannel routing landed.
    # Backchannel routing is asserted separately in talk_backchannel_routing_check.
    url = f"/api/cameras/speaker_cam/talk?token={token}"
    amcrest_client_module.TALK_READ_TIMEOUT_S = 1.0  # fast settle in tests

    # -- media-scope token rejected (handshake 1008/403) --
    media_token = app.state.auth.create_media_token()
    rejected = False
    try:
        with client.websocket_connect(f"/api/cameras/doorbell/talk?token={media_token}") as ws:
            msg = ws.receive()
            rejected = msg.get("type") == "websocket.close"
    except WebSocketDisconnect:
        rejected = True
    check(rejected, "media-scope token rejected on talk WS")

    # -- missing token rejected the same way --
    rejected = False
    try:
        with client.websocket_connect("/api/cameras/doorbell/talk") as ws:
            msg = ws.receive()
            rejected = msg.get("type") == "websocket.close"
    except WebSocketDisconnect:
        rejected = True
    check(rejected, "missing token rejected on talk WS")

    # -- garbage token rejected --
    rejected = False
    try:
        with client.websocket_connect("/api/cameras/doorbell/talk?token=not-a-jwt") as ws:
            msg = ws.receive()
            rejected = msg.get("type") == "websocket.close"
    except WebSocketDisconnect:
        rejected = True
    check(rejected, "garbage token rejected on talk WS")

    # -- non-speaker camera -> 4003 --
    with client.websocket_connect(f"/api/cameras/turret/talk?token={token}") as ws:
        msg = ws.receive()
        check(ws_close_code(msg) == 4003, "turret (no speaker) -> close 4003")

    # -- happy path: frames arrive as A-law; garbage text ignored; stop works --
    server.mode = "ok"
    server.audio_bodies.clear()
    frames = [pcm_frame(seed=s) for s in (1, 2, 3)]
    expected = pcm16le_to_alaw(b"".join(frames))
    with client.websocket_connect(url) as ws:
        for frame in frames:
            ws.send_bytes(frame)
        ws.send_text("not json")                      # ignored
        ws.send_text(json.dumps({"type": "noise"}))   # ignored
        # wait for the bytes to land on the fake camera
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if server.audio_bodies and len(server.audio_bodies[0]) >= len(expected):
                break
            time.sleep(0.05)
        ws.send_text(json.dumps({"type": "stop"}))
        msg = ws.receive()
        check(ws_close_code(msg) == 1000, "{type:'stop'} -> clean close 1000")
    check(bool(server.audio_bodies) and bytes(server.audio_bodies[0]) == expected,
          "camera received the exact A-law transcode of the PCM frames")

    # -- busy: second talker while first holds the camera --
    server.audio_bodies.clear()
    with client.websocket_connect(url) as ws1:
        ws1.send_bytes(frames[0])
        with client.websocket_connect(url) as ws2:
            msg = ws2.receive()
            check(ws_close_code(msg) == 4009 and msg.get("reason") == "busy",
                  "second connection -> close 4009 'busy'")
        ws1.send_text(json.dumps({"type": "stop"}))
        msg = ws1.receive()
        check(ws_close_code(msg) == 1000, "first talker unaffected by busy reject")

    # -- stop on close (no stop message): lock must be released --
    with client.websocket_connect(url) as ws:
        ws.send_bytes(frames[0])
    # closing the WS ends the session; a new talker must not see "busy"
    deadline = time.monotonic() + 10
    reacquired = False
    while time.monotonic() < deadline and not reacquired:
        with client.websocket_connect(url) as ws:
            ws.send_text(json.dumps({"type": "stop"}))
            msg = ws.receive()
            code = ws_close_code(msg)
            if code == 1000:
                reacquired = True
            elif code == 4009:
                time.sleep(0.2)  # previous session still draining
            else:
                break
    check(reacquired, "WS close ends the session and releases the talk lock")

    # -- camera rejects the stream -> 4502 --
    server.mode = "reject"
    with client.websocket_connect(url) as ws:
        ws.send_bytes(frames[0])
        msg = ws.receive()
        check(ws_close_code(msg) == 4502 and msg.get("reason") == "camera rejected audio",
              "camera 401 -> close 4502 'camera rejected audio'")
    server.mode = "ok"

    # -- session cap (shrunk from 120 s) --
    old_cap = talk_router.TALK_MAX_S
    talk_router.TALK_MAX_S = 1.0
    try:
        with client.websocket_connect(url) as ws:
            ws.send_bytes(frames[0])
            msg = ws.receive()
            check(ws_close_code(msg) == 1000
                  and msg.get("reason") == "max talk duration reached",
                  "session capped -> clean close 1000 with reason")
    finally:
        talk_router.TALK_MAX_S = old_cap


def talk_backchannel_routing_check(client: TestClient, headers: dict, token: str) -> None:
    """Backchannel routing (task #8): a backchannel-capable camera (the AD410
    doorbell) must have its talk audio delivered over the RTSP audio
    backchannel (talk_stream_backchannel), NOT the CGI audio.cgi postAudio path.

    We stub talk_stream_backchannel so the test stays hermetic (no real RTSP
    server): the stub captures the RAW PCM frames it was handed and returns when
    the iterator ends, which drives the same clean 1000 close the CGI path does.
    If routing regressed to the CGI path the stub would never be called and the
    captured-args assertion below would fail."""
    print("talk WS: backchannel-capable camera routes to talk_stream_backchannel")
    url = f"/api/cameras/doorbell/talk?token={token}"

    cams = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
    check(cams["doorbell"]["capabilities"]["backchannel"] is True,
          "doorbell (AD410) advertises backchannel=True")

    frames = [pcm_frame(seed=s) for s in (4, 5, 6)]
    captured: dict = {"frames": [], "args": None}

    async def fake_backchannel(ip, username, password, pcm_chunks, gain=None):
        captured["args"] = (ip, username, password)
        captured["gain"] = gain
        async for chunk in pcm_chunks:
            captured["frames"].append(bytes(chunk))

    orig = talk_router.talk_stream_backchannel
    talk_router.talk_stream_backchannel = fake_backchannel  # type: ignore[assignment]
    try:
        with client.websocket_connect(url) as ws:
            for frame in frames:
                ws.send_bytes(frame)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and len(captured["frames"]) < len(frames):
                time.sleep(0.05)
            ws.send_text(json.dumps({"type": "stop"}))
            msg = ws.receive()
            check(ws_close_code(msg) == 1000,
                  "backchannel talk -> clean close 1000 (routed to RTSP backchannel)")
    finally:
        talk_router.talk_stream_backchannel = orig  # type: ignore[assignment]

    check(captured["args"] == ("127.0.0.1", "admin", "pw"),
          "talk_stream_backchannel called with the doorbell's ip + stored creds")
    check(captured["gain"] == 2.0,
          "talk_stream_backchannel gets the per-model talk gain (AD410 -> default 2.0)")
    check(captured["frames"] == frames,
          "backchannel path receives the RAW PCM frames (no CGI A-law transcode)")


def probe_timeout_checks(client: TestClient, headers: dict, server: FakeCameraServer) -> None:
    """On-demand probe cap: a device that accepts TCP but never answers must
    come back 'camera unreachable' within the (shrunk) cap, not hang."""
    print("probe route: on-demand cap honored against a stalling device")
    server.mode = "stall"
    old_cap = cameras_router._ONDEMAND_PROBE_CAP_S
    cameras_router._ONDEMAND_PROBE_CAP_S = 1.0
    try:
        t0 = time.monotonic()
        body = client.post("/api/cameras/doorbell/probe", headers=headers).json()
        elapsed = time.monotonic() - t0
        check(body["ok"] is False and body["detail"] == "camera unreachable",
              "stalled device -> ok:false, detail 'camera unreachable'")
        check(elapsed < 4.0, f"probe returned within the cap (took {elapsed:.2f}s)")
    finally:
        cameras_router._ONDEMAND_PROBE_CAP_S = old_cap
        server.mode = "ok"


def ir_control_checks(client: TestClient, headers: dict) -> None:
    """IR illuminator: Lighting[0][0].Mode + MiddleLight brightness, plus the
    optional VideoInDayNight IR-cut. Asserts the EXACT Lighting CGI strings the
    additive brightness contract builds, both at the client and the endpoint."""
    print("IR control: Lighting Mode + MiddleLight brightness + day/night CGI")
    from app.amcrest.client import AmcrestClient as _AC

    # -- direct client: set_ir(mode, brightness) writes Mode FIRST (its own
    #    setConfig) then the MiddleLight brightness SECOND (a separate setConfig),
    #    so the Manual switch lands before the strength that depends on it. --
    FAKE.behavior = "turret"
    FAKE.requests.clear()

    async def run_set():
        c = _AC("192.0.2.60", "admin", "pw", model="IP5M-T1277EW-AI")
        await c.set_ir(mode="on", brightness=60)
        await c.aclose()

    asyncio.run(run_set())
    sets = FAKE.set_config_requests()
    check(len(sets) == 2,
          "set_ir(mode, brightness) issues two setConfig calls (mode, then brightness)")
    check(all(sets[0].url.params.get(f"Lighting[0][{p}].Mode") == "Manual" for p in range(4)),
          "IR mode 'on' -> Mode=Manual on ALL four Lighting profiles, written FIRST")
    check(sets[0].url.params.get("Lighting[0][0].MiddleLight[0].Light") is None,
          "the mode setConfig carries no brightness param (separate CGI writes)")
    check(all(sets[1].url.params.get(f"Lighting[0][{p}].MiddleLight[0].Light") == "60"
              for p in range(4)),
          "IR brightness -> MiddleLight[0].Light=60 on ALL four profiles, written SECOND")

    # brightness is clamped to 0..100 at the client boundary.
    FAKE.requests.clear()

    async def run_clamp():
        c = _AC("192.0.2.60", "admin", "pw", model="IP5M-T1277EW-AI")
        await c.set_ir(brightness=250)
        await c.aclose()

    asyncio.run(run_clamp())
    sets = FAKE.set_config_requests()
    check(len(sets) == 1
          and sets[0].url.params.get("Lighting[0][0].MiddleLight[0].Light") == "100",
          "brightness > 100 clamped to 100 in the Lighting CGI")

    # -- day/night IR-cut builds the VideoInDayNight CGI. --
    FAKE.requests.clear()

    async def run_dn():
        c = _AC("192.0.2.60", "admin", "pw", model="IP5M-T1277EW-AI")
        await c.set_day_night("black_white")
        await c.aclose()

    asyncio.run(run_dn())
    sets = FAKE.set_config_requests()
    check(len(sets) == 1
          and sets[0].url.params.get("VideoInDayNight[0][0].Mode") == "BlackWhite",
          "day_night 'black_white' -> VideoInDayNight[0][0].Mode=BlackWhite")

    # -- endpoint: PUT settings ir_mode + day_night lands the setConfig CGIs and
    #    returns 200. IR is MODE-ONLY (no brightness): "On" = Manual at the
    #    camera's stored strength, written to every Lighting profile. --
    FAKE.behavior = "turret"
    FAKE.requests.clear()
    resp = client.put("/api/cameras/turret/settings", headers=headers,
                      json={"ir_mode": "on", "day_night": "black_white"})
    check(resp.status_code == 200, "PUT settings ir_mode + day_night -> 200")
    sets = FAKE.set_config_requests()

    def _has(key: str, val: str) -> bool:
        return any(s.url.params.get(key) == val for s in sets)

    check(_has("Lighting[0][0].Mode", "Manual"),
          "endpoint sets Lighting[0][0].Mode=Manual for ir_mode 'on'")
    check(not any(s.url.params.get("Lighting[0][0].MiddleLight[0].Light") is not None
                  for s in sets),
          "endpoint writes NO IR brightness param (mode-only IR)")
    check(_has("VideoInDayNight[0][0].Mode", "BlackWhite"),
          "endpoint sets VideoInDayNight[0][0].Mode=BlackWhite for day_night")


def ir_reassert_checks() -> None:
    """AD410 re-assert-after-stream: the reasserter re-applies a doorbell's
    STORED desired IR after a (re)connect, gated to IR-reverting models."""
    print("IR re-assert: AD410 doorbell re-applies stored desired IR on reconnect")
    from app.amcrest.ir_reassert import (
        IrReasserter, desired_ir_from, model_reverts_ir,
    )

    # gating: doorbells revert IR on RTSP connect; turrets keep it.
    check(model_reverts_ir("AD410") is True, "AD410 flagged as an IR-reverting model")
    check(model_reverts_ir("Foo", {"doorbell": True}) is True,
          "any doorbell-capable camera flagged as IR-reverting")
    check(model_reverts_ir("IP5M-T1277EW-AI") is False,
          "EW turret NOT flagged (keeps IR across streaming)")
    check(model_reverts_ir("IP8M-2779EW-AI", {"doorbell": False}) is False,
          "turret with doorbell:false NOT flagged")

    # desired-state extraction keeps mode+brightness, drops the IR-cut day_night
    # (streaming doesn't reset it) and clamps brightness.
    check(desired_ir_from({"mode": "on", "brightness": 60, "day_night": "color"})
          == {"mode": "on", "brightness": 60},
          "desired_ir_from keeps mode+brightness, drops day_night")
    check(desired_ir_from({}) is None and desired_ir_from(None) is None,
          "empty/absent ir_state -> nothing to re-assert")
    check(desired_ir_from({"brightness": 150}) == {"brightness": 100},
          "desired brightness clamped to 100")

    # end-to-end: a fake DB + client factory. reassert_soon must fire set_ir on
    # the AD410 with its stored ir_state, and be a no-op for a turret / an
    # unpinned doorbell.
    calls: list[tuple[str, dict]] = []

    class _FakeIrClient:
        def __init__(self, cam: dict) -> None:
            self._cam = cam

        async def set_ir(self, mode=None, brightness=None) -> None:
            calls.append((self._cam["name"], {"mode": mode, "brightness": brightness}))

        async def aclose(self) -> None:
            pass

    cameras = {
        "doorbell": {
            "name": "doorbell", "model": "AD410", "ip": "192.0.2.10",
            "username": "admin", "password": "pw",
            "capabilities": {"doorbell": True},
            "ir_state": {"mode": "on", "brightness": 60, "day_night": "color"},
        },
        "turret": {
            "name": "turret", "model": "IP5M-T1277EW-AI", "ip": "192.0.2.11",
            "username": "admin", "password": "pw",
            "capabilities": {"doorbell": False},
            "ir_state": {"mode": "on", "brightness": 60},
        },
        "doorbell_unpinned": {
            "name": "doorbell_unpinned", "model": "AD410", "ip": "192.0.2.12",
            "username": "admin", "password": "pw",
            "capabilities": {"doorbell": True}, "ir_state": {},
        },
    }

    class _FakeDb:
        async def get_camera(self, name: str):
            return cameras.get(name)

        async def list_cameras(self) -> list:
            return list(cameras.values())

    async def run():
        r = IrReasserter(_FakeDb(), client_factory=_FakeIrClient,
                         delay_s=0.0, interval_s=3600.0)
        await r.start()
        for name in ("doorbell", "turret", "doorbell_unpinned"):
            r.reassert_soon(name)
            task = r._pending.get(name)
            if task is not None:
                await task
        await r.stop()

    asyncio.run(run())
    check(calls == [("doorbell", {"mode": "on", "brightness": 60})],
          "reassert_soon fires set_ir(mode=on, brightness=60) for the AD410 ONLY")


def main() -> None:
    unit_checks()
    cgi_accepted_checks()
    model_match_checks()
    ir_reassert_checks()
    sun_checks()
    spotlight_controller_checks()

    server = FakeCameraServer()
    server.start()
    if not server.ready.wait(timeout=10):
        print("FAIL: fake camera server did not start")
        sys.exit(1)

    try:
        with TestClient(app) as client:
            headers, token = login(client)
            light_checks(client, headers)
            lighting_v2_regression_check()
            settings_checks(client, headers)
            ir_control_checks(client, headers)
            probe_checks(client, headers)
            doorbell_audio_caps_check(client, headers)
            siren_checks(client, headers)
            ptz_checks(client, headers)
            night_vision_checks(client, headers)
            go2rtc_regression_checks(client, headers)
            audio_codec_checks(client, headers)
            smart_spotlight_checks(client, headers)

            # A NON-backchannel speaker camera for the CGI postAudio talk
            # happy-path. Added while device HTTP still flows through the
            # MockTransport (base_url unset) so the registration probe stores
            # speaker=True, backchannel=False (its getDeviceType is non-AD410).
            FAKE.behavior = "speaker_cgi"
            add_camera(client, headers, "speaker_cam", "GENERIC-SPEAKER", "127.0.0.1")
            spk = {c["name"]: c for c in client.get("/api/cameras", headers=headers).json()}
            check(spk["speaker_cam"]["capabilities"]["speaker"] is True
                  and spk["speaker_cam"]["capabilities"]["backchannel"] is False,
                  "speaker_cam probes as a non-backchannel speaker (CGI postAudio talk path)")
            FAKE.behavior = "turret"

            # Talk + probe-timeout tests hit the real TCP fake camera.
            FAKE.base_url = f"http://127.0.0.1:{server.port}"
            try:
                talk_checks(client, headers, token, server)
                talk_backchannel_routing_check(client, headers, token)
                probe_timeout_checks(client, headers, server)
            finally:
                FAKE.base_url = None
    finally:
        server.shutdown()

    print(f"ALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()

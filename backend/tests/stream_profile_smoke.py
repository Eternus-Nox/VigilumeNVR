"""Smoke suite for main-stream encode profiles (docs/CONTRACTS.md, "Main-stream
encode profiles").

  A. Planning (amcrest/encode.py): resolution ceilings pick the camera's
     largest size under the cap and keep the picture's shape when they can;
     codec / keyframe / bitrate plans; "keep" is a no-op; every MainFormat is
     kept in step; a re-plan of the result is empty (idempotent).
  B. AmcrestClient.apply_main_stream against a stateful fake camera: a whole-
     set rejection falls back to one setting at a time; a value the camera
     answers OK to but does not keep is reported as not applied; a camera that
     will not list its resolutions gets a standard size, verified by read-back.
  C. StreamProfileManager: inherit vs own profile, no credentials, offline, the
     on_stream_changed hook fires only when size or codec moved.
  D. The API end to end: overview, live read, set for all cameras, per-camera
     pin + back to inherit, reset_cameras, re-apply, settings validation.

Usage: python backend/tests/stream_profile_smoke.py  (needs backend deps)
"""
from __future__ import annotations

import asyncio
import os
import re
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
TMP = Path(tempfile.mkdtemp(prefix="vigilume-streamprofile-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.amcrest import encode as E  # noqa: E402
from app.amcrest.client import AmcrestClient  # noqa: E402
from app.amcrest.stream_profiles import StreamProfileManager  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- fake cameras ----------------


class FakeCamera:
    """A Dahua Encode table that changes when written, with a few real-world
    quirks switchable per camera."""

    def __init__(self, *, width=3840, height=2160, codec="H.265", fps=15, gop=30,
                 bitrate=8192, caps=True, wxh_keys=True, max_keys=None,
                 bitrate_ceiling=None, h264_max_width=None, format3_720=False,
                 size_key=None, reject_custom_name=False, bitrate_per_mpx=None):
        self.formats = {n: {"Compression": codec, "FPS": str(fps), "GOP": str(gop),
                            "BitRate": str(bitrate), "BitRateControl": "VBR"}
                        for n in ((0, 1, 2, 3) if format3_720 else (0, 1))}
        for n, v in self.formats.items():
            w, h = (1280, 720) if (format3_720 and n == 3) else (width, height)
            if wxh_keys:
                v.update(Width=str(w), Height=str(h), CustomResolutionName=f"{w}x{h}")
            else:
                v["resolution"] = f"{w}x{h}"
        self.caps = caps
        self.max_keys = max_keys
        self.bitrate_ceiling = bitrate_ceiling
        self.h264_max_width = h264_max_width
        # Real-camera quirks seen in the field:
        #  format3_720       MainFormat[3] is a separate 720p-capped format and
        #                    refuses anything bigger (rejecting the WHOLE write)
        #  size_key          the only field that can change the size
        #                    ("resolution"); Width/Height writes are refused
        #  reject_custom_name  any write touching CustomResolutionName is refused
        #  bitrate_per_mpx   a size whose bitrate would exceed this many kbps
        #                    per megapixel is refused
        self.format3_720 = format3_720
        self.size_key = size_key
        self.reject_custom_name = reject_custom_name
        self.bitrate_per_mpx = bitrate_per_mpx
        self.sets: list[dict[str, str]] = []
        self.offline = False

    def encode_text(self) -> str:
        lines = ["table.Encode[0].ExtraFormat[0].Video.Compression=H.264",
                 "table.Encode[0].ExtraFormat[0].Video.GOP=15"]
        for n, v in self.formats.items():
            lines += [f"table.Encode[0].MainFormat[{n}].Video.{k}={val}" for k, val in v.items()]
        return "\r\n".join(lines) + "\r\n"

    def width(self, v: dict[str, str]) -> int:
        if "Width" in v:
            return int(v["Width"])
        return int(v["resolution"].split("x")[0])

    def height(self, v: dict[str, str]) -> int:
        if "Height" in v:
            return int(v["Height"])
        return int(v["resolution"].split("x")[1])

    def set_config(self, params: dict[str, str]) -> str:
        if self.max_keys is not None and len(params) > self.max_keys:
            return "Error\r\n"
        staged = {n: dict(v) for n, v in self.formats.items()}
        for key, value in params.items():
            m = re.match(r"^Encode\[0\]\.MainFormat\[(\d+)\]\.Video\.(\w+)$", key)
            if not m:
                return "Error\r\n"
            n, field = int(m.group(1)), m.group(2)
            if field == "BitRate" and self.bitrate_ceiling and int(value) > self.bitrate_ceiling:
                continue  # answers OK, keeps the old value
            if field == "CustomResolutionName" and self.reject_custom_name:
                return "Error\r\n"
            if self.size_key == "resolution" and field in ("Width", "Height"):
                return "Error\r\n"
            if field == "resolution" and "Width" in staged[n]:
                w, h = value.split("x")
                staged[n]["Width"], staged[n]["Height"] = w, h
                continue
            staged[n][field] = value
        for n, v in staged.items():
            w, h = self.width(v), self.height(v)
            if self.format3_720 and n == 3 and h > 720:
                return "Error\r\n"
            if self.bitrate_per_mpx and int(v["BitRate"]) > self.bitrate_per_mpx * w * h / 1e6:
                return "Error\r\n"
        if self.h264_max_width:
            for v in staged.values():
                if v["Compression"].startswith("H.264") and self.width(v) > self.h264_max_width:
                    return "Error\r\n"
        self.formats = staged
        self.sets.append(dict(params))
        return "OK\r\n"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.offline:
            raise httpx.ConnectError("fake: offline")
        path, params = request.url.path, dict(request.url.params)
        if path == "/cgi-bin/configManager.cgi" and params.get("action") == "getConfig":
            if params.get("name") == "Encode":
                return httpx.Response(200, text=self.encode_text())
            return httpx.Response(200, text="Error\r\n")
        if path == "/cgi-bin/configManager.cgi" and params.get("action") == "setConfig":
            params.pop("action")
            return httpx.Response(200, text=self.set_config(params))
        if path == "/cgi-bin/encode.cgi":
            if not self.caps:
                return httpx.Response(404)
            if params.get("action") != "getConfigCaps":
                return httpx.Response(200, text="Error\r\n")
            return httpx.Response(200, text=(
                "caps[0].MainFormat[0].Video.ResolutionTypes="
                "3840x2160,2688x1520,2304x1296,1920x1080,1280x720\r\n"
                "caps[0].MainFormat[0].Video.CompressionTypes=H.264,H.265,MJPG\r\n"
                "caps[0].MainFormat[0].Video.FPSMax=20\r\n"
                "caps[0].MainFormat[0].Video.BitRateOptions=512,16384\r\n"))
        if path == "/cgi-bin/magicBox.cgi":
            return httpx.Response(200, text="type=IP5M-T1277EW-AI\r\n")
        return httpx.Response(404)


CAMERAS: dict[str, FakeCamera] = {}


def _fake_init(self, ip, username, password, timeout=8.0, model=""):
    self.ip = ip
    self.model = (model or "").strip()

    def handler(request: httpx.Request) -> httpx.Response:
        cam = CAMERAS.get(ip)
        if cam is None:
            raise httpx.ConnectError("fake: no route to host")
        return cam(request)

    self._client = httpx.AsyncClient(
        base_url=f"http://{ip}", auth=httpx.DigestAuth(username, password),
        transport=httpx.MockTransport(handler), timeout=httpx.Timeout(2.0, connect=1.0),
    )


AmcrestClient.__init__ = _fake_init  # type: ignore[method-assign]


def _no_sleep_patch() -> None:
    """apply_main_stream waits for an encoder restart before re-reading; the
    fake never restarts, so make the waits free."""
    from app.amcrest import client as client_mod

    real_sleep = asyncio.sleep

    async def quick(delay, *a, **k):
        return await real_sleep(0)

    client_mod.asyncio.sleep = quick  # type: ignore[attr-defined]


# ---------------- A. planning ----------------


def planning_checks() -> None:
    print("A: planning")
    opts = [("3840x2160", 3840, 2160), ("2688x1520", 2688, 1520),
            ("1920x1080", 1920, 1080), ("1280x720", 1280, 720)]
    check(E.choose_resolution("1080p", opts, (3840, 2160))[0][0] == "1920x1080",
          "1080p on a 4K camera -> 1920x1080")
    check(E.choose_resolution("1440p", opts, (3840, 2160))[0][0] == "1920x1080",
          "1440p picks the largest at or under 1440 (2688x1520 is over)")
    check(E.choose_resolution("max", opts, (1920, 1080))[0][0] == "3840x2160", "max -> largest")
    check(E.choose_resolution("keep", opts, (3840, 2160)) == (None, None), "keep -> nothing")
    four3 = [("2592x1944", 2592, 1944), ("1600x1200", 1600, 1200),
             ("1920x1080", 1920, 1080), ("1280x960", 1280, 960)]
    check(E.choose_resolution("1440p", four3, (2592, 1944))[0][0] == "1600x1200",
          "a 4:3 camera keeps its 4:3 shape when a 4:3 size fits")
    check(E.choose_resolution("1080p", four3, (2592, 1944))[0][0] == "1920x1080",
          "...but not when the 4:3 option has far fewer pixels (1280x960)")
    res, note = E.choose_resolution("720p", [("1920x1080", 1920, 1080)], (1920, 1080))
    check(res is None and "smallest" in note, "a cap below every option is a note, not a write")
    res, note = E.choose_resolution("1080p", [], (2688, 1520))
    check(res[0] == "1920x1080" and note, "no caps -> a standard size, with a note")
    check(E.choose_resolution("1080p", [], (1280, 720))[0][0] == "1280x720",
          "no caps never upscales past the current size")
    check(E.choose_resolution("1920x1080", opts, (3840, 2160))[0][0] == "1920x1080", "exact WxH")
    check(E.choose_resolution("1600x900", opts, (3840, 2160))[0] is None,
          "an exact size the camera does not offer is refused")

    for bad in ({"resolution": "huge"}, {"codec": "vp9"}, {"keyframe_s": 0.1},
                {"bitrate_kbps": 10}):
        try:
            E.normalize_profile(bad)
            check(False, f"rejects {bad}")
        except ValueError:
            check(True, f"rejects {bad}")
    check(E.normalize_profile({"codec": "H.264"})["codec"] == "h264", "codec spelling normalized")
    check(E.is_noop(E.normalize_profile(None)), "an empty profile is a no-op")

    cam = FakeCamera()
    cfg = E.main_formats  # noqa: F841 (documentation)
    from app.amcrest.client import _parse_kv
    cfg = _parse_kv(cam.encode_text())
    caps = E.parse_caps({"caps[0].MainFormat[0].Video.ResolutionTypes": "3840x2160,1920x1080",
                         "caps[0].MainFormat[0].Video.CompressionTypes": "H.264,H.265",
                         "caps[0].MainFormat[0].Video.BitRateOptions": "512,16384"})
    prof = E.normalize_profile({"resolution": "1080p", "codec": "h264", "keyframe_s": 1,
                                "bitrate_kbps": 4096})
    plan = E.plan_main_stream(cfg, caps, prof)
    check(plan["primary"] == 0 and plan["resolution"] == ("1920x1080", 1920, 1080),
          "the regular stream is format 0; target 1920x1080")
    check(plan["resolution_formats"] == [0, 1], "every format with a different size, primary first")
    check(set(plan["settings"][0]) == {"codec", "keyframes", "bitrate"}, "the other three settings planned")
    check(plan["settings"][0]["keyframes"]["Encode[0].MainFormat[0].Video.GOP"] == "15",
          "1 s keyframes at 15 fps = GOP 15")
    flat = {k: v for g in plan["settings"].values() for grp in g.values() for k, v in grp.items()}
    check(not any("ExtraFormat" in k for k in flat), "the substream is never touched")
    check(len(plan["keys"]) == len(plan["changes"]) == 8, "one key per change")
    variants = E.resolution_variants(E.main_formats(cfg)[0], plan["resolution"], {"BitRate": "4096"})
    check(variants[0] == {"Width": "1920", "Height": "1080", "BitRate": "4096"},
          "first try: Width/Height alone, with the new bitrate")
    check({"resolution": "1920x1080", "BitRate": "4096"} in variants
          and all("BitRate" in v for v in variants),
          "then the other ways of naming the size, each with the bitrate")
    check(len(variants) == len({tuple(sorted(v.items())) for v in variants}), "no duplicate attempts")
    applied = dict(flat)
    applied = {**cfg, **flat}
    for n in plan["resolution_formats"]:
        p = f"Encode[0].MainFormat[{n}].Video."
        applied.update({p + "Width": "1920", p + "Height": "1080", p + "CustomResolutionName": "1920x1080"})
    check(E.plan_main_stream(applied, caps, prof)["changes"] == [],
          "re-planning the result changes nothing (idempotent)")
    cur = E.current_main(cfg)
    check(cur["codec"] == "h265" and cur["keyframe_s"] == 2.0 and cur["width"] == 3840,
          "current main stream read")


# ---------------- B. client ----------------


async def client_checks() -> None:
    print("B: client apply + read-back")
    CAMERAS["10.0.0.21"] = cam = FakeCamera(max_keys=3)
    client = AmcrestClient("10.0.0.21", "u", "p")
    res = await client.apply_main_stream({"resolution": "1080p", "codec": "h264", "keyframe_s": 1})
    check(cam.formats[0]["Width"] == "1920" and cam.formats[1]["Compression"] == "H.264",
          "a refused set of settings falls back to one setting and one format at a time")
    writes = len(cam.sets)
    check(res["after"]["width"] == 1920 and res["after"]["codec"] == "h264"
          and not res["not_applied"] and not res["rejected"],
          "result reports what the camera reports afterwards")
    check(all(c.startswith("main ") and not c.startswith("main #") for c in res["changed"]),
          "changed lists the regular stream only")
    again = await client.apply_main_stream({"resolution": "1080p", "codec": "h264", "keyframe_s": 1})
    check(again["changed"] == [] and len(cam.sets) == writes, "applying twice writes nothing")

    print("B2: the field report — 720p format #3, picky size keys, bitrate limits")
    CAMERAS["10.0.0.25"] = cam5 = FakeCamera(format3_720=True)
    res = await AmcrestClient("10.0.0.25", "u", "p").apply_main_stream(
        {"resolution": "1080p", "codec": "h264", "keyframe_s": 1})
    check(cam5.formats[0]["Width"] == "1920" and cam5.formats[0]["Compression"] == "H.264",
          "a 720p-capped format #3 no longer blocks the regular stream")
    check(cam5.formats[3]["Height"] == "720", "format #3 left at 720p")
    check(not res["rejected"] and not res["not_applied"], "...and the camera counts as applied")
    check(not any("#3" in n for n in res["notes"]),
          "format #3 already under the ceiling is not an error (1080p would upscale it — skipped)")

    CAMERAS["10.0.0.26"] = cam6 = FakeCamera(size_key="resolution")
    res = await AmcrestClient("10.0.0.26", "u", "p").apply_main_stream({"resolution": "1080p"})
    check(cam6.formats[0]["Width"] == "1920" and not res["rejected"],
          "a camera refusing Width/Height takes the size through `resolution`")

    CAMERAS["10.0.0.27"] = cam7 = FakeCamera(reject_custom_name=True)
    res = await AmcrestClient("10.0.0.27", "u", "p").apply_main_stream({"resolution": "1080p"})
    check(cam7.formats[0]["Width"] == "1920" and not res["rejected"],
          "a camera refusing CustomResolutionName takes Width/Height alone")

    CAMERAS["10.0.0.28"] = cam8 = FakeCamera(bitrate=8192, bitrate_per_mpx=2500)
    res = await AmcrestClient("10.0.0.28", "u", "p").apply_main_stream(
        {"resolution": "1080p", "bitrate_kbps": 4096})
    check(cam8.formats[0]["Width"] == "1920" and cam8.formats[0]["BitRate"] == "4096",
          "a size blocked by the old bitrate goes through with the new bitrate alongside")

    CAMERAS["10.0.0.29"] = cam9 = FakeCamera(size_key="resolution", reject_custom_name=True,
                                             bitrate_per_mpx=1)
    res = await AmcrestClient("10.0.0.29", "u", "p").apply_main_stream({"resolution": "1080p"})
    check(cam9.formats[0]["Width"] == "3840" and len(res["rejected"]) == 1
          and "resolution" in res["rejected"][0] and "refused" in res["rejected"][0],
          "a size refused every way is ONE readable line, not a dump of every format")

    CAMERAS["10.0.0.22"] = cam2 = FakeCamera(bitrate_ceiling=12000)
    res = await AmcrestClient("10.0.0.22", "u", "p").apply_main_stream({"bitrate_kbps": 16000})
    check(res["changed"] == [] and len(res["not_applied"]) == 1
          and "kept its old value" in res["not_applied"][0],
          "OK-but-not-kept is reported as not applied (regular stream)")
    check(any("#1" in n for n in res["notes"]), "...and the secondary format as a note")

    CAMERAS["10.0.0.23"] = cam3 = FakeCamera(h264_max_width=3000)
    res = await AmcrestClient("10.0.0.23", "u", "p").apply_main_stream({"codec": "h264"})
    check(cam3.formats[0]["Compression"] == "H.265" and res["rejected"],
          "a value the camera refuses (H.264 at 4K) is reported as rejected")

    CAMERAS["10.0.0.24"] = cam4 = FakeCamera(width=2688, height=1520, caps=False, wxh_keys=False)
    res = await AmcrestClient("10.0.0.24", "u", "p").apply_main_stream({"resolution": "1080p"})
    check(cam4.formats[0]["resolution"] == "1920x1080" and res["notes"],
          "no caps + a `resolution` key: a standard size is written and verified")
    live = await AmcrestClient("10.0.0.21", "u", "p").read_main_stream()
    check(live["caps"]["codecs"] == ["H.264", "H.265", "MJPG"] and live["current"]["fps"] == 15,
          "read_main_stream: current + caps")


# ---------------- C. manager ----------------


class Settings:
    def __init__(self, main=None):
        self.streams = {"main": main or {}}


async def manager_checks() -> None:
    print("C: manager")
    changed: list[str] = []
    CAMERAS["10.0.0.31"] = FakeCamera()
    CAMERAS["10.0.0.32"] = FakeCamera()
    m = StreamProfileManager(Settings({"resolution": "1080p"}), on_stream_changed=changed.append)
    a = {"name": "a", "ip": "10.0.0.31", "username": "u", "password": "p", "main_stream": None}
    b = {"name": "b", "ip": "10.0.0.32", "username": "u", "password": "p",
         "main_stream": {"codec": "h264", "resolution": "max"}}
    prof, inherited = m.effective(a)
    check(inherited and prof["resolution"] == "1080p", "NULL main_stream follows the global")
    prof, inherited = m.effective(b)
    check(not inherited and prof["codec"] == "h264", "a pinned profile wins")
    ra, rb = await m.apply_many([a, b])
    check(ra["ok"] and CAMERAS["10.0.0.31"].formats[0]["Width"] == "1920", "global applied to a")
    check(rb["ok"] and CAMERAS["10.0.0.32"].formats[0]["Compression"] == "H.264", "own applied to b")
    check(sorted(changed) == ["a", "b"], "on_stream_changed fired for both")
    changed.clear()
    await m.apply(a)
    check(changed == [], "...and not when nothing moved")
    r = await m.apply({**a, "name": "c", "username": ""})
    check(not r["ok"] and "credentials" in r["error"], "no credentials -> reported, not raised")
    r = await m.apply({**a, "name": "d", "ip": "10.0.0.99"})
    check(not r["ok"] and r["error"], "offline -> reported, not raised")
    noop = StreamProfileManager(Settings())
    r = await noop.apply(a)
    check(r["ok"] and r.get("skipped"), "an all-keep profile never contacts the camera")
    check(m.last("a") is not None, "last result kept for status")


# ---------------- D. API ----------------


def api_checks() -> None:
    print("D: API")
    from app.main import app

    CAMERAS["10.0.0.41"] = cam1 = FakeCamera()
    CAMERAS["10.0.0.42"] = cam2 = FakeCamera(width=2688, height=1520)
    with TestClient(app) as client:
        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        for name, ip in (("front", "10.0.0.41"), ("drive", "10.0.0.42"), ("gone", "10.0.0.49")):
            r = client.post("/api/cameras", headers=h, json={
                "name": name, "friendly_name": name.title(), "model": "IP5M-T1277EW-AI",
                "ip": ip, "username": "admin", "password": "pw"})
            assert r.status_code == 201, r.text

        r = client.get("/api/cameras/main-stream", headers=h)
        check(r.status_code == 200 and r.json()["profile"]["resolution"] == "keep",
              "overview: everything keep by default")
        check(all(c["inherited"] for c in r.json()["cameras"]), "every camera inherits by default")
        check(len(cam1.sets) == 0 and len(cam2.sets) == 0, "nothing written to any camera by default")

        r = client.get("/api/cameras/front/main-stream", headers=h)
        live = r.json()["live"]
        check(live["current"]["width"] == 3840 and live["resolutions"][0]["label"] == "3840x2160",
              "live read: current + offered resolutions, largest first")
        r = client.get("/api/cameras/gone/main-stream", headers=h)
        check(r.status_code == 200 and r.json()["live"] is None and r.json()["error"],
              "an unreachable camera reads as an error, not a 500")

        r = client.put("/api/cameras/main-stream", headers=h, json={
            "profile": {"resolution": "1080p", "codec": "h264", "keyframe_s": 1}})
        check(r.status_code == 200, f"set for all cameras ({r.status_code})")
        results = {x["camera"]: x for x in r.json()["results"]}
        check(results["front"]["ok"] and results["drive"]["ok"], "both reachable cameras applied")
        check(not results["gone"]["ok"], "the unreachable one is reported")
        check(cam1.formats[0]["Width"] == "1920" and cam2.formats[0]["Compression"] == "H.264",
              "the cameras changed")
        s = client.get("/api/settings", headers=h).json()
        check(s["streams"]["main"]["resolution"] == "1080p", "the all-cameras profile is stored in settings")

        r = client.put("/api/cameras/drive/main-stream", headers=h,
                       json={"profile": {"resolution": "max"}})
        check(r.status_code == 200 and not r.json()["inherited"]
              and cam2.formats[0]["Width"] == "3840", "a camera can pin its own (drive back to max)")
        r = client.put("/api/cameras/main-stream", headers=h, json={
            "profile": {"resolution": "1080p", "codec": "h264", "keyframe_s": 2}})
        check(cam1.formats[0]["GOP"] == "30" and cam2.formats[0]["Width"] == "3840",
              "a later all-cameras change skips the pinned camera")
        r = client.put("/api/cameras/drive/main-stream", headers=h, json={"profile": None})
        check(r.json()["inherited"] and cam2.formats[0]["Width"] == "1920",
              "profile: null puts it back on the all-cameras profile, applied now")

        client.put("/api/cameras/drive/main-stream", headers=h, json={"profile": {"codec": "h265"}})
        r = client.put("/api/cameras/main-stream", headers=h, json={
            "profile": {"resolution": "1080p", "codec": "h264", "keyframe_s": 2},
            "reset_cameras": True})
        ov = client.get("/api/cameras/main-stream", headers=h).json()
        check(all(c["inherited"] for c in ov["cameras"])
              and cam2.formats[0]["Compression"] == "H.264", "reset_cameras: all cameras follow again")

        writes = len(cam1.sets)
        r = client.post("/api/cameras/main-stream/apply", headers=h)
        check(r.status_code == 200 and len(cam1.sets) == writes, "re-apply with nothing to change writes nothing")

        r = client.patch("/api/settings", headers=h, json={"streams": {"main": {"codec": "vp9"}}})
        check(r.status_code == 422, f"an invalid profile in settings is refused ({r.status_code})")
        r = client.put("/api/cameras/front/main-stream", headers=h,
                       json={"profile": {"resolution": "8k"}})
        check(r.status_code == 422, "an invalid per-camera profile is refused")

        viewer = client.post("/api/users", headers=h, json={
            "username": "viewer1", "password": "viewerpass1", "role": "viewer"})
        if viewer.status_code in (200, 201):
            vt = client.post("/api/auth/login", json={"username": "viewer1",
                                                      "password": "viewerpass1"}).json()["token"]
            r = client.get("/api/cameras/main-stream", headers={"Authorization": f"Bearer {vt}"})
            check(r.status_code == 403, "viewers cannot read or change stream profiles")


def main() -> None:
    _no_sleep_patch()
    planning_checks()
    asyncio.run(client_checks())
    asyncio.run(manager_checks())
    api_checks()
    print(f"\nALL {PASS} CHECKS PASSED (main-stream profiles)")


if __name__ == "__main__":
    main()

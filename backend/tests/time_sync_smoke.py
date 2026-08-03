"""Smoke suite for automatic camera clock correction.

Covers the "camera clock drift" fix (amcrest/client.provision_time +
amcrest/client.camera_local_now + amcrest/time_sync.TimeSyncManager). The fix
NO LONGER trusts NTP or the Dahua timezone index (both proved unreliable on the
real cameras — NTP did not sync within 150s and a wrong index drifts the clock
hours off). Instead it pushes the correct LOCAL wall-clock time and disables the
device NTP client:

  - camera_local_now(tz) returns the wall-clock time for a configurable IANA
    zone via zoneinfo, INDEPENDENT of the container's own (usually UTC) clock;
    an unknown zone falls back to UTC without crashing
  - provision_time(tz) issues NTP.Enable=false + a single global.cgi
    setCurrentTime in the configured zone, and sends NO NTP.TimeZone /
    General.LocalNo / Locales.DST* (the retired timezone-index config)
  - disabling NTP is best-effort: if that setConfig is rejected the clock is
    still set (summary ntp_disabled=False); only a failed setCurrentTime raises
  - TimeSyncManager provisioning is NON-FATAL when the camera errors (sync/
    notify_reachable never raise; the camera is left unprovisioned for a retry),
    the CONNECT hook is idempotent (a provisioned camera is not re-hit), the
    PERIODIC re-push (resync_all / run) re-sets the clock unconditionally
    (clocks drift), and the whole thing is gated by settings.time_sync.auto_sync

The container timezone is forced to UTC via the TZ env var + time.tzset() so the
tests prove camera_local_now/provision_time use zoneinfo for the TARGET zone
rather than the container's local clock. Device HTTP is a fake
httpx.MockTransport (no real camera).

Usage: python backend/tests/time_sync_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

import httpx  # noqa: E402

from app.amcrest.client import (  # noqa: E402
    AmcrestClient,
    AmcrestError,
    camera_local_now,
)
from app.amcrest.time_sync import TimeSyncManager, is_amcrest_camera  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ---------------- fake Amcrest device (httpx.MockTransport) ----------------

# The Dahua stamp, exactly: a real space between date and time.
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def raw_query(request: httpx.Request) -> dict[str, str]:
    """Parse the query the way a percent-only CGI decoder does.

    CRITICAL: do NOT use ``request.url.params`` here. That runs urllib's
    ``unquote_plus``, which turns a ``+`` back into a space — i.e. it decodes
    with the same convention httpx used to encode, so the round-trip always
    agrees with itself and the fake can never see a malformed stamp. That blind
    spot let this suite pass green for months while every real camera received
    ``time=2026-07-16+11:54:26`` and silently ignored it.

    ``unquote`` decodes %XX only, leaving ``+`` a literal plus — which is what
    firmware following RFC 3986 (rather than the HTML form convention) sees."""
    out: dict[str, str] = {}
    for pair in request.url.query.decode().split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        out[unquote(key)] = unquote(value)
    return out


class FakeDevice:
    """Models a real camera's parser, not Python's. Records every CGI request;
    ``behavior`` toggles failure modes.

    Holds a device ``clock`` that only moves when a WELL-FORMED setCurrentTime
    lands, and answers getCurrentTime from it — so the read-back verification in
    provision_time is genuinely exercised rather than short-circuited.

    behavior: "ok"          — setConfig + setCurrentTime succeed
              "reject"      — setConfig returns "Error" (NTP.Enable=false
                              rejected) but setCurrentTime still succeeds
              "reject_time" — setCurrentTime returns "Error" (device rejects the
                              clock write -> provision_time raises)
              "ignore_time" — setCurrentTime answers "OK" but the clock does NOT
                              move (accept-and-discard firmware). The failure
                              mode that is INVISIBLE without a read-back.
              "no_getcurrenttime" — getCurrentTime 404s (firmware without it):
                              the set must still succeed, unverified.
              "down"        — every request raises ConnectError (offline)
    """

    def __init__(self) -> None:
        self.behavior = "ok"
        self.requests: list[httpx.Request] = []
        # The device's own wall-clock, stale until a well-formed set lands.
        self.clock = datetime(2001, 1, 1, 0, 0, 0)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.behavior == "down":
            raise httpx.ConnectError("fake: no route to host")
        path, params = request.url.path, request.url.params
        raw = raw_query(request)
        if path == "/cgi-bin/configManager.cgi" and params.get("action") == "setConfig":
            if self.behavior == "reject":
                return httpx.Response(200, text="Error\r\n")
            return httpx.Response(200, text="OK\r\n")
        if path == "/cgi-bin/global.cgi" and raw.get("action") == "setCurrentTime":
            if self.behavior == "reject_time":
                return httpx.Response(200, text="Error\r\n")
            # Validate the stamp AS IT ARRIVED ON THE WIRE. A '+' where the
            # space belongs is malformed and a percent-only decoder rejects it,
            # exactly as the fleet's firmware appears to.
            stamp = raw.get("time", "")
            if not _STAMP_RE.match(stamp):
                return httpx.Response(200, text="Error\r\n")
            if self.behavior != "ignore_time":
                self.clock = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            return httpx.Response(200, text="OK\r\n")
        if path == "/cgi-bin/global.cgi" and raw.get("action") == "getCurrentTime":
            if self.behavior == "no_getcurrenttime":
                return httpx.Response(404)
            return httpx.Response(200, text=f"result={self.clock:%Y-%m-%d %H:%M:%S}\r\n")
        return httpx.Response(404)

    def reset(self) -> None:
        self.requests.clear()
        self.behavior = "ok"
        # A stale free-running clock: what an un-synced camera actually has.
        # Any test asserting a small read-back delta must have MOVED this.
        self.clock = datetime(2001, 1, 1, 0, 0, 0)

    def set_config_params(self) -> dict[str, str]:
        """Merge every setConfig request's params (minus action) into one dict."""
        merged: dict[str, str] = {}
        for r in self.requests:
            if r.url.path == "/cgi-bin/configManager.cgi" \
                    and r.url.params.get("action") == "setConfig":
                for k, v in r.url.params.multi_items():
                    if k != "action":
                        merged[k] = v
        return merged

    def set_current_time_stamps(self) -> list[str]:
        """The stamp AS THE CAMERA SEES IT (percent-decoded only).

        Was ``r.url.params.get("time")`` — httpx's own decoder — which
        round-tripped its own '+' back to a space and asserted a string the
        camera never receives."""
        return [
            raw_query(r).get("time", "")
            for r in self.requests
            if r.url.path == "/cgi-bin/global.cgi"
            and raw_query(r).get("action") == "setCurrentTime"
        ]

    def set_current_time_raw_queries(self) -> list[str]:
        """The literal query string bytes, for asserting the wire encoding."""
        return [
            r.url.query.decode()
            for r in self.requests
            if r.url.path == "/cgi-bin/global.cgi"
            and raw_query(r).get("action") == "setCurrentTime"
        ]

    def get_current_time_count(self) -> int:
        return sum(
            1 for r in self.requests
            if r.url.path == "/cgi-bin/global.cgi"
            and raw_query(r).get("action") == "getCurrentTime"
        )


FAKE = FakeDevice()


def _fake_init(
    self, ip: str, username: str, password: str, timeout: float = 8.0, model: str = ""
) -> None:
    self.ip = ip
    self.model = (model or "").strip()
    self._client = httpx.AsyncClient(
        base_url=f"http://{ip}",
        auth=httpx.DigestAuth(username, password),
        transport=httpx.MockTransport(FAKE),
        timeout=httpx.Timeout(2.0, connect=1.0),
    )


AmcrestClient.__init__ = _fake_init  # type: ignore[method-assign]


def _set_tz(name: str) -> None:
    os.environ["TZ"] = name
    time.tzset()


def _utc_naive_now() -> datetime:
    return datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)


# ---------------- camera_local_now (zoneinfo, container-independent) --------


def local_now_checks() -> None:
    # Force the CONTAINER zone to UTC: camera_local_now must still return the
    # TARGET zone's wall-clock, proving it uses zoneinfo not the local clock.
    _set_tz("UTC")
    ny = camera_local_now("America/New_York")
    expected = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    check(abs((ny - expected).total_seconds()) < 5,
          "camera_local_now(America/New_York) == Eastern wall-clock (via zoneinfo)")
    check(abs((ny - _utc_naive_now()).total_seconds()) > 3000,
          "camera_local_now uses the target zone, NOT the container's UTC clock")

    # A different zone resolves differently (Asia/Shanghai = UTC+8).
    sh = camera_local_now("Asia/Shanghai")
    sh_expected = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    check(abs((sh - sh_expected).total_seconds()) < 5,
          "camera_local_now(Asia/Shanghai) == Shanghai wall-clock")

    # Unknown / malformed zone -> UTC fallback, never crashes.
    fb = camera_local_now("Not/AZone")
    check(abs((fb - _utc_naive_now()).total_seconds()) < 5,
          "camera_local_now(unknown zone) falls back to UTC (no crash)")


# ---------------- provision_time (client CGI) ----------------


def provision_time_checks() -> None:
    async def run_ny() -> dict:
        _set_tz("UTC")  # container UTC; target zone is Eastern
        FAKE.reset()
        client = AmcrestClient("10.0.0.11", "admin", "pw", model="AD410")
        try:
            return await client.provision_time("America/New_York")
        finally:
            await client.aclose()

    summary = asyncio.run(run_ny())
    params = FAKE.set_config_params()
    check(params.get("NTP.Enable") == "false", "provision_time issues NTP.Enable=false")
    check("NTP.TimeZone" not in params and "General.LocalNo" not in params,
          "provision_time sends NO device timezone index (retired)")
    check(not any(k.startswith("Locales.") for k in params),
          "provision_time sends NO Locales.DST* rules (retired)")
    stamps = FAKE.set_current_time_stamps()
    check(len(stamps) == 1 and len(stamps[0]) == 19 and stamps[0][4] == "-" and stamps[0][13] == ":",
          "provision_time issues setCurrentTime once as YYYY-MM-DD HH:MM:SS")

    # THE REGRESSION GUARD. httpx serializes a params dict via urlencode ->
    # quote_plus, so a space becomes '+' and the camera receives
    # "2026-07-16+11:54:26". '+'-as-space is an HTML form convention, not RFC
    # 3986; a percent-only decoder reads a literal plus. Assert the RAW wire —
    # asserting the decoded value (as this suite used to) can never catch it,
    # because httpx decodes with the same convention it encoded with.
    raw_qs = FAKE.set_current_time_raw_queries()
    check(len(raw_qs) == 1 and "%20" in raw_qs[0],
          "setCurrentTime encodes the space as %20 on the wire")
    check("+" not in raw_qs[0].split("time=")[1],
          "setCurrentTime does NOT plus-encode the space (params-dict regression)")

    pushed = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
    check(abs((pushed - camera_local_now("America/New_York")).total_seconds()) < 10,
          "pushed time is the configured-tz (Eastern) wall-clock")
    check(abs((pushed - _utc_naive_now()).total_seconds()) > 3000,
          "pushed time is NOT the container's UTC clock")
    check(summary["timezone"] == "America/New_York" and summary["ntp_disabled"] is True,
          "summary reports timezone + ntp_disabled=True")

    # The write must be VERIFIED, not assumed: an accepted request proves
    # nothing about the resulting clock.
    check(FAKE.get_current_time_count() == 1,
          "provision_time reads the clock BACK after setting it")
    check(summary.get("clock_delta_s") is not None
          and abs(summary["clock_delta_s"]) <= 10,
          "summary reports a MEASURED clock_delta_s (device clock actually moved)")
    check(abs((FAKE.clock - camera_local_now("America/New_York")).total_seconds()) < 10,
          "the fake device's clock genuinely advanced to the pushed time")

    # NTP disable rejected by the firmware -> clock is STILL set (best-effort),
    # summary reports ntp_disabled=False, never raises.
    async def run_reject() -> dict:
        _set_tz("UTC")
        FAKE.reset()
        FAKE.behavior = "reject"
        client = AmcrestClient("10.0.0.12", "admin", "pw", model="IP8M-2779EW-AI")
        try:
            return await client.provision_time("America/New_York")
        finally:
            await client.aclose()

    rsummary = asyncio.run(run_reject())
    check(len(FAKE.set_current_time_stamps()) == 1,
          "NTP-disable rejected: clock is still set (best-effort NTP disable)")
    check(rsummary["ntp_disabled"] is False,
          "NTP-disable rejected: summary ntp_disabled=False (non-fatal)")

    # setCurrentTime rejected -> the clock write is the load-bearing call, so it
    # DOES raise (the manager treats that as retryable).
    async def run_time_reject() -> bool:
        FAKE.reset()
        FAKE.behavior = "reject_time"
        client = AmcrestClient("10.0.0.13", "admin", "pw", model="AD410")
        try:
            await client.provision_time("America/New_York")
            return False
        except AmcrestError:
            return True
        finally:
            await client.aclose()

    check(asyncio.run(run_time_reject()),
          "provision_time raises AmcrestError when setCurrentTime is rejected")

    # THE BUG THIS FLEET ACTUALLY HAD, modelled: firmware that answers "OK" and
    # does not move its clock. Before the read-back this was INVISIBLE — every
    # camera logged "clock set" every 30 min while drifting 1-20 minutes.
    async def run_ignore() -> dict:
        _set_tz("UTC")
        FAKE.reset()
        FAKE.behavior = "ignore_time"
        client = AmcrestClient("10.0.0.14", "admin", "pw", model="AD410")
        try:
            return await client.provision_time("America/New_York")
        finally:
            await client.aclose()

    isummary = asyncio.run(run_ignore())
    check(isummary.get("clock_delta_s") is not None,
          "accept-and-discard: the clock is still read back")
    # Fake clock stays at 2001 -> ~25 years adrift.
    check(abs(isummary["clock_delta_s"]) > 60,
          "accept-and-discard is DETECTED: clock_delta_s exposes the stale clock")
    # Non-fatal on purpose: time_sync._provision piggybacks the G.711A audio
    # codec AFTER provision_time returns, so raising here would silently stop
    # audio provisioning fleet-wide (live-view audio needs G.711A, not AAC).
    check(isummary["ntp_disabled"] is True,
          "accept-and-discard does NOT raise (audio provisioning must still run)")

    # Firmware without getCurrentTime: the set must still succeed, reported
    # honestly as unverified rather than failed.
    async def run_no_readback() -> dict:
        _set_tz("UTC")
        FAKE.reset()
        FAKE.behavior = "no_getcurrenttime"
        client = AmcrestClient("10.0.0.15", "admin", "pw", model="IP4M-1056E")
        try:
            return await client.provision_time("America/New_York")
        finally:
            await client.aclose()

    nsummary = asyncio.run(run_no_readback())
    check(nsummary.get("clock_delta_s") is None,
          "firmware without getCurrentTime -> clock_delta_s None (unverified)")
    check(len(FAKE.set_current_time_stamps()) == 1,
          "firmware without getCurrentTime -> the clock write still happened")


# ---------------- TimeSyncManager (connect hook + periodic, non-fatal) ------


class FakeSettings:

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False
    def __init__(self, auto_sync: bool = True, timezone: str = "America/New_York") -> None:
        self._ts = {"auto_sync": auto_sync, "timezone": timezone}

    @property
    def time_sync(self) -> dict:
        return self._ts


CAM = {"name": "front", "ip": "10.0.0.20", "username": "admin", "password": "pw", "model": "AD410"}
NOCREDS = {"name": "guest", "ip": "10.0.0.21", "username": "", "password": "", "model": "unknown"}


async def _drain(mgr: TimeSyncManager) -> None:
    # Let scheduled provision tasks run to completion.
    for _ in range(50):
        if not mgr._tasks:
            break
        await asyncio.gather(*list(mgr._tasks), return_exceptions=True)
        await asyncio.sleep(0)


def manager_checks() -> None:
    check(is_amcrest_camera(CAM) and not is_amcrest_camera(NOCREDS),
          "is_amcrest_camera gates on stored credentials")

    # Happy path: provisions once, marks done, dedupes on a second connect sync.
    async def happy() -> None:
        _set_tz("UTC")
        FAKE.reset()
        mgr = TimeSyncManager(FakeSettings())
        await mgr.sync([CAM, NOCREDS])
        await _drain(mgr)
        check(mgr._done.get("front") is not None, "manager marks a provisioned camera done")
        check("guest" not in mgr._done, "manager skips a camera with no credentials")
        first_reqs = len(FAKE.requests)
        # A second connect sync must NOT re-hit the already-provisioned camera.
        await mgr.sync([CAM, NOCREDS])
        await _drain(mgr)
        check(len(FAKE.requests) == first_reqs, "connect-hook sync is idempotent (no re-provision)")
        await mgr.stop_all()

    asyncio.run(happy())

    # Non-fatal: the camera rejects the clock write -> sync never raises, camera
    # left unprovisioned for a later retry.
    async def rejected() -> None:
        _set_tz("UTC")
        FAKE.reset()
        FAKE.behavior = "reject_time"
        mgr = TimeSyncManager(FakeSettings())
        await mgr.sync([CAM])  # must not raise
        await _drain(mgr)
        check("front" not in mgr._done, "rejected camera is NOT marked done (retryable)")
        # A later reachable transition can retry once the device recovers.
        FAKE.behavior = "ok"
        FAKE.requests.clear()
        await mgr.notify_reachable(CAM)
        await _drain(mgr)
        check(mgr._done.get("front") is not None, "retry provisions after the device recovers")
        await mgr.stop_all()

    asyncio.run(rejected())

    # Offline device: ConnectError is swallowed, never fatal.
    async def offline() -> None:
        _set_tz("UTC")
        FAKE.reset()
        FAKE.behavior = "down"
        mgr = TimeSyncManager(FakeSettings())
        await mgr.notify_reachable(CAM)  # must not raise
        await _drain(mgr)
        check("front" not in mgr._done, "offline camera provisioning is non-fatal + retryable")
        await mgr.stop_all()

    asyncio.run(offline())

    # Disabled via settings: nothing is attempted.
    async def disabled() -> None:
        FAKE.reset()
        mgr = TimeSyncManager(FakeSettings(auto_sync=False))
        await mgr.sync([CAM])
        await mgr.notify_reachable(CAM)
        await mgr.resync_all([CAM])
        await _drain(mgr)
        check(len(FAKE.requests) == 0 and not mgr._done,
              "auto_sync=false disables all provisioning (connect + periodic)")
        await mgr.stop_all()

    asyncio.run(disabled())

    # Periodic re-push: resync_all re-sets the clock even on an already-done
    # camera (clocks drift -> force), while the connect hook stays idempotent.
    async def periodic() -> None:
        _set_tz("UTC")
        FAKE.reset()
        mgr = TimeSyncManager(FakeSettings())
        await mgr.sync([CAM])
        await _drain(mgr)
        check(mgr._done.get("front") is not None, "periodic: camera provisioned on the connect sweep")
        after_first = len(FAKE.requests)
        await mgr.sync([CAM])  # connect hook: idempotent
        await _drain(mgr)
        check(len(FAKE.requests) == after_first, "periodic: connect-hook sync stays idempotent")
        await mgr.resync_all([CAM])  # periodic loop: unconditional re-push
        await _drain(mgr)
        check(len(FAKE.requests) > after_first, "periodic: resync_all re-pushes the clock despite done-set")
        check(mgr._done.get("front") is not None, "periodic: re-push re-marks the camera done")
        await mgr.stop_all()

    asyncio.run(periodic())

    # run() with no cameras_provider is a no-op (returns immediately, no loop).
    async def no_provider() -> None:
        mgr = TimeSyncManager(FakeSettings())
        await asyncio.wait_for(mgr.run(), timeout=1.0)
        check(True, "run() is a no-op without a cameras_provider")
        await mgr.stop_all()

    asyncio.run(no_provider())

    # run() loop: periodically calls the cameras_provider + re-pushes.
    async def run_loop_cycle() -> None:
        _set_tz("UTC")
        FAKE.reset()
        calls = {"n": 0}

        async def provider() -> list[dict]:
            calls["n"] += 1
            return [CAM]

        mgr = TimeSyncManager(FakeSettings(), cameras_provider=provider, interval_s=0.01)
        task = asyncio.create_task(mgr.run())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if calls["n"] >= 1 and FAKE.requests:
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await _drain(mgr)
        await mgr.stop_all()
        check(calls["n"] >= 1 and len(FAKE.requests) > 0,
              "run() periodically calls the cameras_provider and re-pushes the clock")

    asyncio.run(run_loop_cycle())


def main() -> None:
    original_tz = os.environ.get("TZ")
    try:
        local_now_checks()
        provision_time_checks()
        manager_checks()
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()
    _check_substream_gop()
    print(f"\nAll {PASS} time-sync checks passed.")


def _check_substream_gop() -> None:
    """provision_substream_gop: shorten the SUB keyframe interval to ~1 s so live
    view paints sooner, WITHOUT touching MainFormat — the main stream feeds the
    24/7 recorder and extra keyframes there would inflate every recording."""

    class _Fake(AmcrestClient):
        def __init__(self, cfg: dict) -> None:  # no transport: pure config logic
            self._cfg = cfg
            self.written: dict | None = None

        async def get_config(self, name: str) -> dict:
            return self._cfg

        async def set_config(self, **params: str) -> None:
            self.written = params

    def run(cfg: dict):
        c = _Fake(cfg)
        result = asyncio.run(c.provision_substream_gop())
        return c.written, result

    written, _ = run({"Encode[0].ExtraFormat[0].Video.GOP": "10",
                      "Encode[0].ExtraFormat[0].Video.FPS": "5"})
    check(written == {"Encode[0].ExtraFormat[0].Video.GOP": "5"},
          "substream GOP 2xFPS is shortened to FPS (~1 s keyframes)")

    # THE load-bearing one: the recorder's stream must never grow.
    written, result = run({"Encode[0].MainFormat[0].Video.GOP": "40",
                           "Encode[0].MainFormat[0].Video.FPS": "20"})
    check(written is None and result["streams"] == 0,
          "MainFormat is NEVER touched (recording size unchanged)")

    written, _ = run({"Encode[0].ExtraFormat[0].Video.GOP": "5",
                      "Encode[0].ExtraFormat[0].Video.FPS": "5"})
    check(written is None, "already at ~1 s: no write (idempotent)")

    written, _ = run({"Encode[0].ExtraFormat[0].Video.GOP": "2",
                      "Encode[0].ExtraFormat[0].Video.FPS": "15"})
    check(written is None, "a shorter-than-FPS GOP is left alone (only ever shortens)")

    written, _ = run({"Encode[0].ExtraFormat[0].Video.GOP": "4",
                      "Encode[0].ExtraFormat[0].Video.FPS": "1"})
    check(written is None, "FPS < 2 skipped (GOP=1 would be all-intra)")

    written, _ = run({"Encode[0].ExtraFormat[0].Video.GOP": "abc",
                      "Encode[0].ExtraFormat[0].Video.FPS": "5"})
    check(written is None, "unparseable GOP/FPS is skipped, not crashed on")

    written, _ = run({
        "Encode[0].ExtraFormat[0].Video.GOP": "20", "Encode[0].ExtraFormat[0].Video.FPS": "10",
        "Encode[1].ExtraFormat[0].Video.GOP": "12", "Encode[1].ExtraFormat[0].Video.FPS": "6",
        "Encode[0].MainFormat[0].Video.GOP": "50", "Encode[0].MainFormat[0].Video.FPS": "25",
    })
    check(written == {"Encode[0].ExtraFormat[0].Video.GOP": "10",
                      "Encode[1].ExtraFormat[0].Video.GOP": "6"},
          "every substream channel/format handled, main excluded")


if __name__ == "__main__":
    main()

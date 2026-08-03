"""AD410 doorbell button watcher.

Maintains a persistent digest-auth attach to
`http://<ip>/cgi-bin/eventManager.cgi?action=attach&codes=[All]` (multipart
text/plain stream, parts delimited by `--myboundary`) with reconnect/backoff.

Button-press mapping (verified against dchesterton's community Amcrest
home-automation bridge source, 2026-07-05):
    Code == "_DoTalkAction_" with data.Action == "Invite"  -> pressed
That bridge handles only that code; per CONTRACTS.md we additionally accept
`Invite` and `CallNoAnswered` (action=Start), which some firmwares emit at
press time — a local dedupe window collapses multi-code presses into one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

log = logging.getLogger(__name__)

_BOUNDARY = "--myboundary"
_CODE_RE = re.compile(r"Code=(?P<code>[^;]+);action=(?P<action>[^;]+);index=(?P<index>[^;\s]+)")

# on_press(camera_name)
PressCallback = Callable[[str], Awaitable[None]]

_PRESS_DEDUPE_S = 5.0
_STREAM_STABLE_S = 60.0
_BACKOFF_MIN_S = 5.0
_BACKOFF_MAX_S = 120.0


def parse_event_part(part: str) -> Optional[dict[str, Any]]:
    """Parse one multipart segment into {code, action, index, data}."""
    match = _CODE_RE.search(part)
    if not match:
        return None
    event: dict[str, Any] = {
        "code": match.group("code").strip(),
        "action": match.group("action").strip(),
        "index": match.group("index").strip(),
        "data": None,
    }
    data_pos = part.find(";data=", match.end() - 1)
    if data_pos == -1:
        data_pos = part.find("data=", match.end())
        if data_pos != -1:
            data_pos -= 1  # align with the ';data=' slicing below
    if data_pos != -1:
        raw = part[data_pos + 6 :].strip()
        try:
            event["data"] = json.loads(raw)
        except json.JSONDecodeError:
            event["data"] = None
    return event


def is_button_press(event: dict[str, Any]) -> bool:
    code = event.get("code", "")
    action = (event.get("action") or "").lower()
    data = event.get("data") or {}
    if code == "_DoTalkAction_":
        return isinstance(data, dict) and data.get("Action") == "Invite"
    if code == "Invite":
        return action in ("start", "pulse")
    if code == "CallNoAnswered":
        return action == "start"
    return False


class DoorbellWatcher:
    def __init__(self, name: str, ip: str, username: str, password: str, on_press: PressCallback):
        self.name = name
        self._ip = ip
        self._username = username
        self._password = password
        self._on_press = on_press
        self._task: Optional[asyncio.Task] = None
        self._last_press = 0.0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"doorbell:{self.name}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        backoff = _BACKOFF_MIN_S
        while True:
            started = time.monotonic()
            try:
                await self._attach_once()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                log.info("doorbell %s: stream ended (%s); reconnecting", self.name, exc.__class__.__name__)
            except Exception:  # noqa: BLE001 — watcher must never die
                log.exception("doorbell %s: unexpected error; reconnecting", self.name)
            # Reset backoff if the stream was healthy for a while.
            if time.monotonic() - started > _STREAM_STABLE_S:
                backoff = _BACKOFF_MIN_S
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)

    async def _attach_once(self) -> None:
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(
            auth=httpx.DigestAuth(self._username, self._password), timeout=timeout
        ) as client:
            url = f"http://{self._ip}/cgi-bin/eventManager.cgi?action=attach&codes=[All]"
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    log.warning(
                        "doorbell %s: attach rejected with HTTP %s", self.name, resp.status_code
                    )
                    await resp.aread()
                    return
                log.info("doorbell %s: attached to event stream", self.name)
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    if len(buffer) > 262144:  # runaway safety
                        buffer = buffer[-65536:]
                    while _BOUNDARY in buffer:
                        part, _, rest = buffer.partition(_BOUNDARY)
                        # Only process once the *next* boundary arrived, i.e.
                        # `part` is a complete segment.
                        buffer = rest
                        if part.strip():
                            await self._handle_part(part)

    async def _handle_part(self, part: str) -> None:
        event = parse_event_part(part)
        if event is None:
            return
        if event["code"] not in ("VideoMotion", "AudioMutation", "TimeChange", "NTPAdjustTime"):
            log.debug("doorbell %s: event %s action=%s", self.name, event["code"], event["action"])
        if not is_button_press(event):
            return
        now = time.monotonic()
        if now - self._last_press < _PRESS_DEDUPE_S:
            return
        self._last_press = now
        log.info("doorbell %s: button press (code=%s)", self.name, event["code"])
        try:
            await self._on_press(self.name)
        except Exception:  # noqa: BLE001
            log.exception("doorbell %s: press handler failed", self.name)


class DoorbellManager:
    """Keeps one watcher per doorbell-capable camera; resync after CRUD."""

    def __init__(self, on_press: PressCallback):
        self._on_press = on_press
        self._watchers: dict[str, DoorbellWatcher] = {}

    async def sync(self, cameras: list[dict[str, Any]]) -> None:
        wanted: dict[str, dict[str, Any]] = {
            cam["name"]: cam
            for cam in cameras
            if cam.get("capabilities", {}).get("doorbell") or cam.get("model") == "AD410"
        }
        # stop removed/changed watchers
        for name in list(self._watchers):
            cam = wanted.get(name)
            watcher = self._watchers[name]
            if cam is None or cam["ip"] != watcher._ip or cam["username"] != watcher._username \
                    or cam["password"] != watcher._password:
                await watcher.stop()
                del self._watchers[name]
        # start new ones
        for name, cam in wanted.items():
            if name not in self._watchers:
                watcher = DoorbellWatcher(
                    name, cam["ip"], cam["username"], cam["password"], self._on_press
                )
                self._watchers[name] = watcher
                watcher.start()

    async def stop_all(self) -> None:
        for watcher in self._watchers.values():
            await watcher.stop()
        self._watchers.clear()

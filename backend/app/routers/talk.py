"""Two-way talk WebSocket (docs/CONTRACTS.md — Camera controls v2).

WS /api/cameras/{name}/talk (JWT via Sec-WebSocket-Protocol; media-scope rejected)

The client sends BINARY frames of raw little-endian Int16 PCM, MONO, 8 kHz
(the browser downsamples). Frames are forwarded as received (no re-pacing),
transcoded to G.711 A-law, and streamed to the camera. Delivery is per-camera:
a backchannel-capable camera (caps.backchannel — the AD410, whose firmware
silently drops audio.cgi postAudio) gets the audio streamed to its RTSP audio
backchannel (amcrest.backchannel.talk_stream_backchannel), the only path proven
to make its speaker play; every other speaker camera uses the audio.cgi
postAudio path the siren uses (AmcrestClient.talk_stream). Text frames are
ignored except {"type":"stop"}.

Close codes:
  1008 — bad/missing/media-scope token, or unknown camera (handshake reject)
  4003 — camera has no speaker capability
  4009 — another talk session is active for this camera ("busy")
  4502 — the camera rejected the audio stream / became unreachable
  1000 — clean end (client stop or close, or the 120 s session cap)

Camera errors never crash the app: the whole session is fenced, the
per-camera lock is always released, and the Amcrest client always closed.

AD410 caveat: a clean 1000 close here means the browser mic audio was
streamed to the doorbell's audio.cgi postAudio endpoint without a transport
error — it does NOT guarantee the doorbell actually played it. The AD410
firmware only honors POSTed talk audio when its own audio Encode is
G.711A/G.711Mu @ 8 kHz (default is 16 kHz AAC), and some firmwares refuse
third-party POSTed talk entirely (the in-app talk uses a proprietary
P2P/SIP path the HTTP CGI can't reach). AmcrestClient.talk_stream logs a
device-config diagnostic on each AD410 session; see amcrest/client.py for
the full write-up of what works and what doesn't.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..amcrest.backchannel import talk_gain_for, talk_stream_backchannel
from ..auth import ws_token
from ..amcrest.client import AmcrestError
from .. import privacy
from .cameras import _caps_of, _client_for

log = logging.getLogger(__name__)

router = APIRouter()

TALK_MAX_S = 120.0
# ~5 s of 20 ms PCM frames; newest frames are dropped beyond this so a slow
# camera bounds latency instead of building an ever-growing backlog.
_QUEUE_MAX = 256
# How long to let the postAudio request settle after the session ends
# before force-cancelling it.
_DRAIN_TIMEOUT_S = 12.0


def _is_stop(text: str) -> bool:
    try:
        return json.loads(text).get("type") == "stop"
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False


@router.websocket("/api/cameras/{name}/talk")
async def talk_ws(
    websocket: WebSocket, name: str, token: Optional[str] = Query(default=None)
) -> None:
    state = websocket.app.state
    # Prefer the Sec-WebSocket-Protocol bearer over ?token= — a query-string
    # token lands verbatim in nginx's error log. See auth.ws_token.
    raw, subprotocol = ws_token(websocket, token)
    claims = state.auth.decode(raw) if raw else None
    if claims is None or claims.get("scope") == "media":
        # Same policy-violation reject as /api/ws (pre-accept -> 403/1008).
        # MEDIA-scope tokens stay rejected: those are the long-lived, widely
        # shared image tokens in notifications/MQTT, and they must never be
        # able to open a live mic into the house.
        await websocket.close(code=1008)
        return
    # Viewers MAY talk. Two-way talk is a live-interaction feature (answer the
    # door, tell the dog off), not an admin configuration change — same class as
    # watching the stream, which viewers already have. A full session token is
    # still required (media tokens rejected above) and the single-talker lock
    # below is unchanged. Deliberate product decision; the RBAC matrix in
    # docs/CONTRACTS.md and tests/rbac_smoke.py assert this.
    cam = await state.db.get_camera(name)
    if cam is None:
        await websocket.close(code=1008)
        return
    # PRIVACY MODE GATE (app/privacy.py). Refuse BEFORE accept() and before the
    # talk lock is taken, so no Amcrest client is opened and no audio can reach
    # the camera. Both talk paths (RTSP backchannel and CGI postAudio) run
    # through _run_session below this point, so this single check covers them.
    if privacy.is_private(state, name):
        await websocket.close(code=1008)
        return
    if not _caps_of(cam).get("speaker"):
        await websocket.accept(subprotocol=subprotocol)
        await websocket.close(code=4003, reason="Camera has no speaker")
        return

    # ONE active talker per camera. The locked() check and the acquire run
    # in the same scheduling slice (Lock.acquire never suspends when
    # uncontended), so a second connection can't sneak in between them.
    locks: dict[str, asyncio.Lock] = state.talk_locks
    lock = locks.setdefault(name, asyncio.Lock())
    if lock.locked():
        await websocket.accept(subprotocol=subprotocol)
        await websocket.close(code=4009, reason="busy")
        return
    await lock.acquire()
    try:
        await websocket.accept(subprotocol=subprotocol)
        await _run_session(websocket, cam)
    except Exception:  # noqa: BLE001 — a talk session must never crash the app
        log.exception("talk session for camera %s failed", name)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
    finally:
        lock.release()


async def _run_session(websocket: WebSocket, cam: dict[str, Any]) -> None:
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=_QUEUE_MAX)

    async def pcm_frames() -> AsyncIterator[bytes]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    def signal_end() -> None:
        """Terminate pcm_frames() even when the queue is full."""
        while True:
            try:
                queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()

    log.info("talk: session started for camera %s (%s)", cam["name"], cam.get("model") or "?")
    client = _client_for(cam)
    # Delivery path is per-camera: backchannel-capable cameras (the AD410 —
    # its firmware silently drops audio.cgi postAudio) get the talk audio
    # streamed to their RTSP audio backchannel, the only path proven to make
    # the speaker actually play. Every other speaker camera keeps the CGI
    # postAudio path unchanged. Both consume the SAME pcm_frames() iterator and
    # raise AmcrestError on failure, so the session fencing below is identical.
    if _caps_of(cam).get("backchannel"):
        sender = asyncio.create_task(
            talk_stream_backchannel(
                cam["ip"], cam["username"], cam["password"], pcm_frames(),
                gain=talk_gain_for(cam.get("model")),
            )
        )
    else:
        sender = asyncio.create_task(client.talk_stream(pcm_frames()))
    close_code, close_reason = 1000, ""
    disconnected = False
    frames_in = 0
    receiver: Optional[asyncio.Task] = None
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TALK_MAX_S
        while True:
            if sender.done():
                break  # camera path ended (usually an error) — stop reading
            # Privacy Mode can be switched ON mid-session. The gate at connect
            # (see talk_ws) runs once, and privacy.apply() has no talk teardown,
            # so without this re-check a talker keeps speaking out of a camera
            # in a room an admin just marked private, for up to TALK_MAX_S.
            # Free: this loop already wakes at least once a second.
            if privacy.is_private(websocket.app.state, cam["name"]):
                close_code, close_reason = 1008, "privacy mode enabled"
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                close_reason = "max talk duration reached"
                break
            # Persistent receive task, NEVER cancelled between iterations:
            # wait_for(receive(), timeout) would cancel receive() on every
            # quiet tick, and a cancel that lands just as a message is
            # delivered silently discards it — a lost stop/disconnect would
            # keep the session (and the per-camera busy lock) alive until
            # the 120 s cap. asyncio.wait leaves the pending task untouched.
            if receiver is None:
                receiver = asyncio.create_task(websocket.receive())
            # Short timeout so sender failures and the deadline are noticed
            # even while the client is silent.
            await asyncio.wait(
                {receiver, sender},
                timeout=min(remaining, 1.0),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not receiver.done():
                continue  # tick or sender ended — loop re-checks both
            message = receiver.result()
            receiver = None
            if message["type"] == "websocket.disconnect":
                disconnected = True
                break
            data = message.get("bytes")
            if data:
                frames_in += 1
                if frames_in == 1:
                    log.info("talk: first mic frame from client for camera %s", cam["name"])
                if not queue.full():
                    queue.put_nowait(bytes(data))
                # else: drop the frame — bounded latency beats a backlog
                continue
            text = message.get("text")
            if text and _is_stop(text):
                break
    except WebSocketDisconnect:
        disconnected = True
    finally:
        if receiver is not None:
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver
        signal_end()
        try:
            await asyncio.wait_for(sender, timeout=_DRAIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            sender.cancel()
        except AmcrestError as exc:
            log.warning("camera %s rejected talk audio: %s", cam["name"], exc)
            close_code, close_reason = 4502, "camera rejected audio"
        except Exception:  # noqa: BLE001
            log.exception("talk stream to camera %s failed", cam["name"])
            close_code, close_reason = 4502, "camera rejected audio"
        with contextlib.suppress(BaseException):
            await sender  # reap cancellation / already-raised exception
        await client.aclose()
        log.info(
            "talk: session ended for camera %s (%d mic frames, close=%d %r)",
            cam["name"], frames_in, close_code, close_reason,
        )
        if not disconnected:
            with contextlib.suppress(Exception):
                await websocket.close(code=close_code, reason=close_reason)

"""RTSP audio-backchannel talk sender (AD410 doorbell and other
ONVIF-backchannel-capable cameras).

PROVEN, validated on real AD410 hardware (the user heard the beeps): the
doorbell speaker plays audio pushed to its RTSP audio BACKCHANNEL — NOT the
audio.cgi postAudio path (which the AD410 firmware silently drops) and NOT the
client-side WebRTC mic. This module ports the exact working flow from the
reference scripts (scratchpad/bcfinal.py + hs.py) to asyncio so the talk
WebSocket can deliver browser mic audio straight to the camera.

Flow (all over ONE TCP connection to <ip>:554):
  1. DESCRIBE rtsp://<ip>:554/cam/realmonitor?channel=1&subtype=0 with
     ``Accept: application/sdp`` + ``Require: www.onvif.org/ver20/backchannel``.
     Digest auth on the 401: parse ONLY the ``WWW-Authenticate: Digest`` line
     (the camera ALSO offers Basic — ignore it) for realm+nonce.
  2. From the 200 SDP: take Content-Base; find the media block flagged
     ``a=sendonly`` -> its ``a=control:`` track and RTP payload type (8 =
     PCMA/A-law @ 8 kHz). SETUP that track with
     ``Transport: RTP/AVP/TCP;unicast;interleaved=0-1``. CRITICAL: the camera
     REASSIGNS the interleaved channel — read the RTP channel back from the
     SETUP *response* Transport header, do not assume 0.
  3. PLAY the Content-Base with ``Range: npt=0.000-``.
  4. Stream: buffer incoming PCM to 20 ms (160-sample) chunks, A-law encode,
     RTP-packetize, and send each as an interleaved frame on the assigned RTP
     channel. A background task drains inbound RTCP so it never blocks sends.
     Pacing comes for free from the caller's mic-rate async iterator.
  5. On stop/error: TEARDOWN + close.

Any transport/protocol failure raises AmcrestError so the talk WS session
fences cleanly — the exact same contract as AmcrestClient.talk_stream.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import random
import re
import struct
from typing import AsyncIterator, Optional

from .client import AmcrestError


# Linear gain applied to each mic PCM sample before A-law encoding. The camera
# speaker is already driven at max hardware volume, so the only remaining way to
# make talk louder is to amplify the source PCM. 2.0 = +6 dB. Saturates cleanly
# at the Int16 rails (no wrap) so a hotter input clips rather than distorting via
# overflow. Some models have a quieter speaker at the same PCM level and need
# more gain — see TALK_GAIN_BY_MODEL.
TALK_GAIN = 2.0

# Per-model gain override (falls back to TALK_GAIN when a model isn't listed).
# The IP3M-941B PTZ dome plays talk audibly softer than the AD410 at 2.0, so it
# gets +12 dB.
TALK_GAIN_BY_MODEL = {
    "IP3M-941B": 4.0,
    # IP4M-1041B: same dome family as the 941B, assumed same quiet speaker →
    # same +12 dB. Adjust if the real unit is louder/quieter.
    "IP4M-1041B": 4.0,
}


def talk_gain_for(model: Optional[str]) -> float:
    """Talk PCM gain for a camera model (TALK_GAIN if it has no override)."""
    return TALK_GAIN_BY_MODEL.get((model or "").strip(), TALK_GAIN)


def _apply_gain(s: int, gain: float = TALK_GAIN) -> int:
    """Scale one Int16 sample by ``gain``, saturating to [-32768, 32767]."""
    v = int(s * gain)
    if v > 32767:
        return 32767
    if v < -32768:
        return -32768
    return v


# ITU-T G.711 A-law (piecewise-linear) — the EXACT encoder validated live against
# the AD410 backchannel (the beep test the user heard clearly). The camera decodes
# per the ITU standard, so we match it byte-for-byte here rather than the shared
# analog-companding approximation (pcm16le_to_alaw), which only ~36% overlaps and
# adds quantization hiss on this path.
def _alaw_sample(s: int) -> int:
    sign = 0x80 if s >= 0 else 0x00
    s = -s if s < 0 else s
    if s > 32767:
        s = 32767
    if s >= 256:
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (s & mask):
            exponent -= 1
            mask >>= 1
        a = (exponent << 4) | ((s >> (exponent + 3)) & 0x0F)
    else:
        a = s >> 4
    return (a | sign) ^ 0x55


def _pcm16le_to_alaw(pcm: bytes, gain: float = TALK_GAIN) -> bytes:
    n = len(pcm) // 2
    return bytes(_alaw_sample(_apply_gain(v, gain)) for v in struct.unpack("<%dh" % n, pcm))

log = logging.getLogger(__name__)

# 20 ms of 8 kHz mono PCM16 = 160 samples = 320 bytes per RTP packet.
_SAMPLES_PER_PACKET = 160
_BYTES_PER_PACKET = _SAMPLES_PER_PACKET * 2
# Default RTP payload type for the backchannel (8 = PCMA/G.711 A-law @ 8 kHz);
# overridden by whatever the SDP sendonly block advertises.
_DEFAULT_PAYLOAD_TYPE = 8
_CONNECT_TIMEOUT_S = 6.0
_RESPONSE_TIMEOUT_S = 8.0


class _RTSPBackchannel:
    """One RTSP audio-backchannel session over a single TCP connection."""

    def __init__(self, ip: str, username: str, password: str, port: int = 554,
                 gain: float = TALK_GAIN):
        self._ip = ip
        self._gain = gain
        self._user = username
        self._pw = password
        self._port = port
        self._base = f"rtsp://{ip}:{port}/cam/realmonitor?channel=1&subtype=0"
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._cseq = 0
        self._session: Optional[str] = None
        self._realm: Optional[str] = None
        self._nonce: Optional[str] = None
        self._content_base = self._base
        self._track_uri: Optional[str] = None
        self._payload_type = _DEFAULT_PAYLOAD_TYPE
        self._rtp_channel = 0

    # ---------- digest auth ----------

    def _digest_header(self, method: str, uri: str) -> str:
        if not self._nonce:
            return ""
        ha1 = hashlib.md5(f"{self._user}:{self._realm}:{self._pw}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        resp = hashlib.md5(f"{ha1}:{self._nonce}:{ha2}".encode()).hexdigest()
        return (
            f'Authorization: Digest username="{self._user}", realm="{self._realm}", '
            f'nonce="{self._nonce}", uri="{uri}", response="{resp}"\r\n'
        )

    @staticmethod
    def _parse_challenge(head: str) -> tuple[Optional[str], Optional[str]]:
        """Extract realm+nonce from the ``WWW-Authenticate: Digest`` line ONLY
        (the camera also sends a Basic line — never use it)."""
        for line in head.split("\r\n"):
            low = line.lower()
            if "digest" in low and "realm" in low:
                fields = dict(re.findall(r'(\w+)="([^"]+)"', line))
                return fields.get("realm"), fields.get("nonce")
        return None, None

    # ---------- RTSP request/response ----------

    async def _read_response(self) -> tuple[str, str]:
        assert self._reader is not None
        raw = await asyncio.wait_for(
            self._reader.readuntil(b"\r\n\r\n"), timeout=_RESPONSE_TIMEOUT_S
        )
        head = raw[:-4].decode(errors="replace")
        body = ""
        m = re.search(r"Content-Length:\s*(\d+)", head, re.I)
        if m:
            n = int(m.group(1))
            if n:
                data = await asyncio.wait_for(
                    self._reader.readexactly(n), timeout=_RESPONSE_TIMEOUT_S
                )
                body = data.decode(errors="replace")
        return head, body

    async def _request(self, method: str, uri: str, extra: str = "") -> tuple[str, str]:
        assert self._writer is not None

        async def once() -> tuple[str, str]:
            self._cseq += 1
            msg = f"{method} {uri} RTSP/1.0\r\nCSeq: {self._cseq}\r\n"
            msg += self._digest_header(method, uri)
            if self._session:
                msg += f"Session: {self._session}\r\n"
            msg += extra + "\r\n"
            self._writer.write(msg.encode())
            await self._writer.drain()
            return await self._read_response()

        head, body = await once()
        status = head.split("\r\n", 1)[0]
        if " 401" in status:
            realm, nonce = self._parse_challenge(head)
            if realm and nonce:
                self._realm, self._nonce = realm, nonce
                head, body = await once()
                status = head.split("\r\n", 1)[0]
        if not re.search(r"\b200\b", status):
            raise AmcrestError(
                f"camera {self._ip}: backchannel {method} failed ({status.strip()!r})"
            )
        return head, body

    # ---------- handshake ----------

    async def open(self) -> None:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._ip, self._port), timeout=_CONNECT_TIMEOUT_S
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise AmcrestError(
                f"camera {self._ip}: backchannel connect failed ({exc.__class__.__name__})"
            ) from exc

        try:
            head, body = await self._request(
                "DESCRIBE",
                self._base,
                "Accept: application/sdp\r\n"
                "Require: www.onvif.org/ver20/backchannel\r\n",
            )
        except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            raise AmcrestError(
                f"camera {self._ip}: backchannel DESCRIBE error ({exc.__class__.__name__})"
            ) from exc

        cb = re.search(r"Content-Base:\s*(\S+)", head, re.I)
        if cb:
            self._content_base = cb.group(1).strip()

        for block in re.split(r"(?=^m=)", body, flags=re.M):
            if "sendonly" in block:
                cm = re.search(r"a=control:(\S+)", block)
                pm = re.search(r"m=audio \d+ \S+ (\d+)", block)
                if cm:
                    self._track_uri = cm.group(1).strip()
                if pm:
                    self._payload_type = int(pm.group(1))
        if not self._track_uri:
            raise AmcrestError(
                f"camera {self._ip}: no ONVIF backchannel (sendonly) track in SDP"
            )

        setup_url = (
            self._track_uri
            if self._track_uri.startswith("rtsp://")
            else self._content_base.rstrip("/") + "/" + self._track_uri
        )
        try:
            head, _ = await self._request(
                "SETUP",
                setup_url,
                "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n",
            )
        except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            raise AmcrestError(
                f"camera {self._ip}: backchannel SETUP error ({exc.__class__.__name__})"
            ) from exc

        sm = re.search(r"Session:\s*([^\r\n;]+)", head)
        if sm:
            self._session = sm.group(1).strip()
        # CRITICAL: the camera reassigns the interleaved channel — read it back
        # from the SETUP response, do not assume the requested 0.
        ch = re.search(r"interleaved=(\d+)-(\d+)", head)
        self._rtp_channel = int(ch.group(1)) if ch else 0

        try:
            await self._request("PLAY", self._content_base, "Range: npt=0.000-\r\n")
        except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            raise AmcrestError(
                f"camera {self._ip}: backchannel PLAY error ({exc.__class__.__name__})"
            ) from exc

        log.info(
            "camera %s: RTSP backchannel ready (session=%s rtp_channel=%d payload=%d)",
            self._ip, self._session, self._rtp_channel, self._payload_type,
        )

    # ---------- streaming ----------

    async def stream(self, pcm_chunks: AsyncIterator[bytes]) -> None:
        """Consume mic PCM (LE Int16, mono, 8 kHz), A-law encode 20 ms chunks,
        and send each as an interleaved RTP frame on the assigned channel.

        Pacing comes from ``pcm_chunks`` — it yields at the mic rate, so the
        outbound RTP stream tracks real time without an artificial sleep.
        """
        assert self._writer is not None
        # A background reader drains inbound RTCP so it never fills the socket
        # buffer and stalls our writes (the sync reference recv()'d periodically).
        drain_task = asyncio.create_task(self._drain_inbound())
        seq = random.randint(0, 0xFFFF)
        timestamp = random.randint(0, 0xFFFFFFFF)
        ssrc = random.randint(0, 0xFFFFFFFF)
        buf = bytearray()
        packets = 0
        try:
            async for chunk in pcm_chunks:
                if drain_task.done():  # socket died under us
                    break
                buf.extend(chunk)
                while len(buf) >= _BYTES_PER_PACKET:
                    pcm = bytes(buf[:_BYTES_PER_PACKET])
                    del buf[:_BYTES_PER_PACKET]
                    payload = _pcm16le_to_alaw(pcm, self._gain)
                    rtp = struct.pack(
                        "!BBHII",
                        0x80,
                        self._payload_type & 0x7F,
                        seq & 0xFFFF,
                        timestamp & 0xFFFFFFFF,
                        ssrc,
                    ) + payload
                    frame = (
                        b"\x24"
                        + bytes([self._rtp_channel])
                        + struct.pack("!H", len(rtp))
                        + rtp
                    )
                    self._writer.write(frame)
                    await self._writer.drain()
                    seq += 1
                    timestamp = (timestamp + _SAMPLES_PER_PACKET) & 0xFFFFFFFF
                    packets += 1
                    if packets == 1:
                        log.info(
                            "camera %s: backchannel first RTP packet sent (channel %d)",
                            self._ip, self._rtp_channel,
                        )
        except (OSError, asyncio.TimeoutError) as exc:
            raise AmcrestError(
                f"camera {self._ip}: backchannel send error ({exc.__class__.__name__})"
            ) from exc
        finally:
            drain_task.cancel()
            with contextlib.suppress(BaseException):
                await drain_task
            log.info(
                "camera %s: backchannel stream ended (%d RTP packets, ~%.1fs)",
                self._ip, packets, packets * 0.02,
            )

    async def _drain_inbound(self) -> None:
        """Continuously read and discard inbound bytes (RTCP / RTSP keepalives)
        so they never block our interleaved sends. Ends when the peer closes."""
        assert self._reader is not None
        try:
            while True:
                data = await self._reader.read(4096)
                if not data:
                    return
        except (OSError, asyncio.IncompleteReadError):
            return

    # ---------- teardown ----------

    async def close(self) -> None:
        if self._writer is None:
            return
        if self._session:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._request("TEARDOWN", self._content_base), timeout=3.0
                )
        with contextlib.suppress(Exception):
            self._writer.close()
            await asyncio.wait_for(self._writer.wait_closed(), timeout=3.0)
        self._reader = None
        self._writer = None


async def talk_stream_backchannel(
    ip: str, username: str, password: str, pcm_chunks: AsyncIterator[bytes],
    gain: float = TALK_GAIN,
) -> None:
    """Deliver browser mic PCM to a camera's RTSP audio backchannel.

    ``pcm_chunks`` yields raw little-endian Int16 mono 8 kHz PCM (the same
    iterator the CGI ``AmcrestClient.talk_stream`` consumes). Opens the RTSP
    backchannel, streams the audio as interleaved G.711 A-law RTP, and tears the
    session down cleanly on end or error. Raises AmcrestError on any transport
    or protocol failure so the talk WS session fences the same way the CGI path
    does.
    """
    session = _RTSPBackchannel(ip, username, password, gain=gain)
    try:
        await session.open()
        await session.stream(pcm_chunks)
    finally:
        await session.close()

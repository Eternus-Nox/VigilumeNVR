"""RTSP stream URL resolution + go2rtc config generation/sync.

Vigilume owns a go2rtc 1.9.14 container. The backend writes its YAML config
(``$GO2RTC_CONFIG_DIR/go2rtc.yaml``, mounted at /config in the go2rtc
container) from the camera DB and syncs changes over the go2rtc HTTP API.

Per camera two streams are published:

    {name}:      [<main_url>, "ffmpeg:{name}#audio=aac"]   # record + live
    {name}_sub:  [<sub_url>]                               # detect ingest

``main_url``/``sub_url`` come from the per-camera override columns; empty
means "derive the Amcrest default from ip+credentials" (subtype=0/1,
credentials percent-encoded). The ``#audio=aac`` ffmpeg source transcodes
G.711 to AAC so recording (MPEG-TS), MSE and WebRTC all get legal audio.

WebRTC candidates: ``settings.system.webrtc_candidates`` entries (verbatim,
e.g. "192.168.1.10:8555", "100.64.0.7:8555") plus a best-effort auto-derived
host candidate (VIGILUME_WEBRTC_HOST env / a PUBLIC_URL IP literal / the host's
default-route LAN IPv4) plus the constant "stun:8555" — so LAN WebRTC connects
without the operator hand-entering an IP. See webrtc_status / docs/live-latency.md.

Change application: write the YAML only when content changed. Stream-only
changes (routine camera CRUD) are synced incrementally over
``PUT/DELETE /api/streams`` so live streams for the other cameras keep
running; listener/candidate changes (and the first sync after boot) require
``POST /api/restart``, whose failure falls back to per-stream PUTs.
Everything here is reconnect-tolerant and never raises into callers —
go2rtc being down is a normal boot state.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote, urlparse

import httpx
import yaml

from ..config import env_dual

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..db import Database
    from ..settings_store import SettingsStore

log = logging.getLogger(__name__)

WEBRTC_PORT = 8555
_HEALTH_CACHE_S = 5.0

# Optional operator override for the WebRTC host candidate: the LAN IP (or a
# hostname) the browser can reach this server's :8555 on. go2rtc accepts a bare
# IPv4 *or* a hostname candidate (resolved at negotiation time), so this may be
# either — set it in compose when the auto-derivation below can't see the host
# LAN IP (the normal case for a bridge-networked container). See
# docs/live-latency.md. (Legacy SENTINEL_WEBRTC_HOST is still honored via env_dual.)
WEBRTC_HOST_ENV = "VIGILUME_WEBRTC_HOST"
_PORT_SUFFIX_RE = re.compile(r":\d{1,5}$")


# ---------- WebRTC host-candidate derivation ----------
#
# Live view negotiates WebRTC first (sub-second) and falls back to MSE (slow)
# when ICE finds no reachable candidate. go2rtc can only advertise candidates
# it is told about; on a bridge-networked container it cannot discover the host
# LAN IP itself. These helpers derive a host candidate WITHOUT the operator
# hand-entering one, layered by trust: env override -> PUBLIC_URL IP literal ->
# auto-derived default-route LAN IPv4. All are best-effort and never raise; the
# manual settings.system.webrtc_candidates + a stun entry are always kept too.


def _is_docker_bridge_ipv4(ip: str) -> bool:
    """True when ``ip`` looks like a Docker/Compose bridge address rather than a
    real LAN IP. docker0 + Compose user networks live in 172.16.0.0/12; a
    published-port bridge container sees its OWN address there, and that address
    is unreachable from a LAN client — advertising it as a candidate just makes
    ICE waste time on a dead route. Home/office LANs almost always use
    192.168/16 or 10/8, so treating the whole 172.16/12 block as bridge-suspect
    is safe in practice (an operator on a genuine 172.16 LAN sets
    VIGILUME_WEBRTC_HOST explicitly)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and addr in ipaddress.ip_network("172.16.0.0/12")


def _auto_lan_ipv4() -> Optional[str]:
    """Best-effort host LAN IPv4 from the default route, via a connect-less UDP
    socket: 'connecting' a UDP socket toward an address makes the kernel pick
    the egress interface and its source IP WITHOUT sending any packet; we read
    that source IP back with getsockname(). The target is TEST-NET-1
    (192.0.2.1, RFC 5737 — never routed) so nothing leaves the host. Returns the
    address only when it is a private LAN IPv4 that is not a Docker bridge
    address; a public IP, a non-private/link-local result, or a docker-bridge
    address all yield None (inside a bridge container this normally returns the
    docker IP and is therefore correctly skipped). Never raises."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 9))  # RFC 5737 TEST-NET-1; no traffic sent
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if addr.version != 4 or not addr.is_private or addr.is_loopback:
        return None
    if _is_docker_bridge_ipv4(ip):
        return None
    return ip


def _public_url_ipv4(public_url: str) -> Optional[str]:
    """The host of ``public_url`` when it is a bare IPv4 literal (e.g.
    ``http://192.168.1.10:8443``) — a directly usable WebRTC host. A hostname
    PUBLIC_URL (the Tailscale/mDNS common case) yields None: a public-URL
    hostname is usually a reverse-proxy/tunnel name that does NOT front :8555,
    so we only trust an IP literal here. Never raises."""
    if not public_url:
        return None
    try:
        host = urlparse(public_url).hostname or ""
    except ValueError:
        return None
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return None
    return host if addr.version == 4 else None


def _host_to_candidate(host: str) -> str:
    """Form a go2rtc candidate string from a host: append ``:8555`` unless the
    operator already supplied an explicit ``host:port`` (a trailing ``:<port>``
    on a non-bracketed host). Bracketed IPv6 (``[::1]``) always gets the port."""
    host = host.strip()
    if _PORT_SUFFIX_RE.search(host) and not host.endswith("]"):
        return host
    return f"{host}:{WEBRTC_PORT}"


# The LAN address a client actually reached this server on, learned from the
# Host header of a real request (see main.py's _learn_webrtc_host middleware).
#
# WHY THIS EXISTS. `_auto_lan_ipv4()` cannot work in the normal deployment: the
# backend runs in a Docker BRIDGE network, so the only address it can see for
# itself is the docker-bridge one, which that function correctly rejects. The
# result was a stack with NO host candidate at all — WebRTC could not connect
# even on the LAN, every client silently fell back to slow HLS, and the operator
# had to know to set VIGILUME_WEBRTC_HOST by hand. The address a LAN client
# used to reach us IS the box's LAN address, so we simply remember it.
_LAN_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT — Tailscale et al.
)

_observed_host: Optional[str] = None


def note_observed_host(host: str) -> bool:
    """Remember a PRIVATE IPv4 a client reached us on. Returns True only when
    this changes the stored value (so the caller can regenerate go2rtc once,
    not on every request).

    Deliberately narrow, because the Host header is caller-controlled:
      * only a bare private IPv4 literal is accepted — a hostname (the tunnel,
        a public name) or a public address is ignored, so a remote caller can
        never inject a candidate;
      * a docker-bridge address is rejected, same as the auto path;
      * an explicit VIGILUME_WEBRTC_HOST always wins and disables learning
        entirely, so a configured deployment is never influenced by a client.
    Worst case for a hostile LAN client is that WebRTC fails to connect — which
    is exactly the state this feature exists to fix, never a data leak."""
    global _observed_host
    if env_dual("WEBRTC_HOST").strip():
        return False  # explicitly configured: never learn
    host = (host or "").strip()
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname, not an IP literal
    # NOT `addr.is_private`: Python counts the IANA special-purpose ranges as
    # private, including the DOCUMENTATION nets (192.0.2/24, 198.51.100/24,
    # 203.0.113/24) and link-local — none of which is a reachable LAN address.
    # Accept only genuine LAN/VPN space: RFC1918 plus 100.64/10, which is where
    # Tailscale and other CGNAT-range VPNs live and IS a valid way to reach us.
    if addr.version != 4 or not any(addr in net for net in _LAN_NETS):
        return False
    if _is_docker_bridge_ipv4(host):
        return False
    if _observed_host == host:
        return False
    _observed_host = host
    return True


def observed_host() -> Optional[str]:
    return _observed_host


def _derive_host(settings: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(host, source) for the best-effort auto WebRTC host candidate, or
    (None, None). Trust order: VIGILUME_WEBRTC_HOST env -> PUBLIC_URL IP literal
    -> auto-derived default-route LAN IPv4. ``host`` is a bare host (no port);
    ``source`` is "env" | "public_url" | "auto" | "observed". Never raises."""
    env_host = env_dual("WEBRTC_HOST").strip()
    if env_host:
        return env_host, "env"
    public_url = str((settings.get("system") or {}).get("public_url") or "").strip()
    public_ip = _public_url_ipv4(public_url)
    if public_ip:
        return public_ip, "public_url"
    auto_ip = _auto_lan_ipv4()
    if auto_ip:
        return auto_ip, "auto"
    # Last resort, and the one that actually fires in a bridge deployment: the
    # address a LAN client reached us on. See note_observed_host.
    if _observed_host:
        return _observed_host, "observed"
    return None, None


def webrtc_status(settings: dict[str, Any]) -> dict[str, Any]:
    """Compute the effective go2rtc WebRTC candidate list plus a readiness hint
    the settings UI uses to warn "live is on slow fallback — add your server IP".

    Returns:
      candidates   — full list handed to go2rtc: manual entries + the derived
                     host candidate + ``stun:<port>``, deduped, order-stable.
      ready        — True when at least one HOST candidate exists (manual OR
                     derived); False means WebRTC has only STUN and will almost
                     certainly fall back to MSE on the LAN.
      detected_ip  — best host to pre-fill the UI's "use detected IP" button
                     (bare host, no port), or None.
      source       — where detected_ip came from ("env"|"public_url"|"auto"|
                     "observed" = learned from a LAN client's Host header) or None.

    Never raises — a detection failure degrades to stun-only + ready:false."""
    manual = [
        str(c).strip()
        for c in ((settings.get("system") or {}).get("webrtc_candidates") or [])
        if str(c).strip()
    ]
    host, source = _derive_host(settings)

    candidates: list[str] = []
    for entry in manual:
        if entry not in candidates:
            candidates.append(entry)
    if host:
        host_candidate = _host_to_candidate(host)
        if host_candidate not in candidates:
            candidates.append(host_candidate)
    stun = f"stun:{WEBRTC_PORT}"
    if stun not in candidates:
        candidates.append(stun)

    return {
        "candidates": candidates,
        "ready": bool(manual or host),
        "detected_ip": host,
        "source": source,
    }


# ---------- URL resolution ----------


def default_stream_url(cam: dict[str, Any], subtype: int) -> str:
    """Amcrest RTSP URL from ip + credentials (docs/cameras-amcrest.md).

    Credentials are percent-encoded (safe="") so passwords with @ : / #
    survive; empty credentials produce a credential-less URL.
    """
    user = quote(cam.get("username") or "", safe="")
    password = quote(cam.get("password") or "", safe="")
    cred = f"{user}:{password}@" if (user or password) else ""
    return f"rtsp://{cred}{cam['ip']}:554/cam/realmonitor?channel=1&subtype={subtype}"


def resolve_urls(cam: dict[str, Any]) -> tuple[str, str]:
    """(main_url, sub_url) for a camera row: non-empty override columns win,
    blank falls back to the Amcrest defaults (subtype=0 main / 1 sub)."""
    main = (cam.get("main_url") or "").strip() or default_stream_url(cam, 0)
    sub = (cam.get("sub_url") or "").strip() or default_stream_url(cam, 1)
    return main, sub


def sub_stream_name(camera_name: str) -> str:
    return f"{camera_name}_sub"


def is_doorbell(cam: dict[str, Any]) -> bool:
    """Whether this camera is a doorbell (AD410). Robust to the exact stored
    model string (case-insensitive, whitespace-tolerant) and prefers the
    probed capability flag when present — mirrors amcrest/doorbell.py's own
    doorbell test so the two never disagree."""
    if (cam.get("capabilities") or {}).get("doorbell"):
        return True
    return (cam.get("model") or "").strip().upper() == "AD410"


def stream_sources(cam: dict[str, Any]) -> dict[str, list[str]]:
    """The two go2rtc streams for one camera row:
    ``{name: [main, ffmpeg-aac-audio], name_sub: [sub]}``.

    A DOORBELL points ``{name}_sub`` at its MAIN url, deliberately. Two reasons:
    (1) the AD410 ships ExtraFormat as MJPG, which go2rtc cannot restream, so
    ``subtype=1`` would give a black tile; and — the one that actually forces
    this — (2) the AD410 only offers its ONVIF two-way-TALK backchannel while a
    session slot is free, and it answers a contended backchannel DESCRIBE with a
    (non-standard) 404. Pulling a SEPARATE substream makes go2rtc hold a second
    RTSP session to the doorbell, which consumes that slot and silently breaks
    the mic. Pointing ``{name}_sub`` at the SAME url as the main stream keeps
    go2rtc's sessions deduped to one, leaving the backchannel available.

    This is a real trade: detect ingest then decodes the doorbell's full
    2560x1920 main stream (~6x the pixels of D1). We accept it, because two-way
    talk is a feature people use and the decode cost is one camera's worth. A
    genuine H.264 substream WAS provisioned and worked (subtype=1 -> 576x720),
    but using it cost the mic, so it was reverted. If the doorbell's decode load
    ever needs cutting, the correct fix is to route talk through go2rtc's own
    backchannel (one session, both features) — NOT a second camera pull."""
    main, sub = resolve_urls(cam)
    sub_src = main if is_doorbell(cam) else sub
    # WebRTC live audio requires a WebRTC-legal codec (G.711 PCMA/PCMU or opus).
    # go2rtc's ffmpeg transcode is NOT relied on here (the go2rtc image has no
    # working ffmpeg, so a `#audio=opus` track is advertised but never produced —
    # which actually BREAKS WebRTC audio, since go2rtc offers the dead opus track
    # instead of the camera's native codec). Instead the backend PROVISIONS each
    # camera's audio encoder to G.711A on connect (see amcrest.audio_provision),
    # so the camera's NATIVE audio is already WebRTC-legal and passes straight
    # through — the only path proven to work (the IP3M-941B ships as PCMA). The
    # `#audio=aac` source stays only to give MSE/recording consumers legal audio
    # where a go2rtc ffmpeg IS available; it never overrides the native passthrough.
    return {
        cam["name"]: [main, f"ffmpeg:{cam['name']}#audio=aac"],
        sub_stream_name(cam["name"]): [sub_src],
    }


# ---------- config generation ----------


def build_config(cameras: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    """go2rtc YAML config as a dict (api/rtsp/webrtc listeners + streams).

    WebRTC candidates are the manual settings entries + a best-effort derived
    host candidate (env / PUBLIC_URL / auto LAN IP) + stun — see webrtc_status.
    """
    candidates = webrtc_status(settings)["candidates"]

    streams: dict[str, Any] = {}
    for cam in cameras:
        streams.update(stream_sources(cam))

    return {
        "api": {"listen": ":1984"},
        "rtsp": {"listen": ":8554"},
        "webrtc": {"listen": f":{WEBRTC_PORT}", "candidates": candidates},
        "streams": streams,
    }


def render_yaml(cameras: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    return yaml.safe_dump(
        build_config(cameras, settings), sort_keys=False, default_flow_style=False
    )


def write_if_changed(path: Path, text: str) -> bool:
    """Idempotent config write; returns True when the file content changed."""
    try:
        if path.is_file() and path.read_text() == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


# ---------- go2rtc API sync ----------


class Go2rtcManager:
    """Owns the generated config file + the go2rtc HTTP API session.

    ``apply()`` is called at boot, after camera CRUD, and after settings
    changes that touch ``system.webrtc_candidates``. It never raises: go2rtc
    down/unreachable degrades to "config on disk is current; go2rtc picks it
    up on its next start".
    """

    def __init__(self, config: "Config", db: "Database", settings: "SettingsStore"):
        self._config = config
        self._db = db
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=config.go2rtc_api_url,
            timeout=httpx.Timeout(3.0, connect=1.5),
        )
        self._health_cache: tuple[float, bool] = (0.0, False)
        # Last-synced config, used by apply() to route stream-only changes
        # through incremental PUT/DELETE /api/streams instead of a full
        # restart (which tears down every live stream). None until the first
        # sync in this process — that one is always a full restart.
        self._prev_streams: Optional[dict[str, list[str]]] = None
        self._prev_listeners: Optional[dict[str, Any]] = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_healthy(self) -> bool:
        """Cheap cached liveness probe (GET /api, 5 s cache)."""
        now = time.monotonic()
        checked_at, healthy = self._health_cache
        if now - checked_at < _HEALTH_CACHE_S:
            return healthy
        ok = False
        try:
            resp = await self._client.get("/api")
            ok = resp.status_code == 200
        except httpx.HTTPError:
            ok = False
        self._health_cache = (now, ok)
        return ok

    async def apply(self) -> bool:
        """Regenerate the YAML from the camera DB + settings, write it if
        changed, and nudge go2rtc. Stream-only changes (routine camera CRUD)
        sync incrementally so live streams for other cameras keep running;
        listener/candidate changes (and the first sync after boot) fall back
        to a full restart — listeners can only be applied by restart.
        Returns whether the on-disk config changed. Never raises."""
        try:
            cameras = await self._db.list_cameras()
            # PRIVACY MODE GATE (app/privacy.py). Filter private cameras OUT of
            # the config entirely, so neither `{name}` (live view + camera-mic
            # audio) nor `{name}_sub` (detect ingest) is published. Filtering the
            # CAMERA LIST rather than adding a `private=` argument to
            # build_config() is deliberate: a parameter with an empty default is
            # a leak waiting to happen — any caller that forgot it would
            # republish a private camera's stream.
            #
            # The listeners (api/rtsp/webrtc) stay byte-identical, which keeps
            # apply() on the INCREMENTAL sync path below: privacy ON issues a
            # live DELETE for each of that camera's runtime streams, with no
            # go2rtc restart and no camera reconfiguration. Empty-vs-full is
            # always a content change, so write_if_changed always fires and the
            # diff-cache re-syncs correctly on toggle OFF.
            private = self._settings.private_cameras
            if private:
                cameras = [c for c in cameras if c["name"] not in private]
            cfg = build_config(cameras, self._settings.current)
            text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
            changed = write_if_changed(self._config.go2rtc_config_path, text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — config trouble must not break CRUD
            log.exception("go2rtc config generation failed")
            return False
        if changed:
            log.info("go2rtc config written (%d cameras)", len(cameras))
            new_streams: dict[str, list[str]] = cfg["streams"]
            new_listeners = {k: v for k, v in cfg.items() if k != "streams"}
            if self._prev_listeners is not None and new_listeners == self._prev_listeners:
                await self._sync_streams(self._prev_streams or {}, new_streams)
            else:
                await self._sync(cameras)
            self._prev_streams = new_streams
            self._prev_listeners = new_listeners
        return changed

    async def _sync_streams(
        self, prev: dict[str, list[str]], new: dict[str, list[str]]
    ) -> None:
        """Incrementally reconcile go2rtc's runtime streams with the new
        config: PUT added/changed streams (with their FULL source list),
        DELETE removed ones. Never touches unchanged streams, so their live
        viewers stay connected. Same tolerance as _sync: go2rtc down means
        the on-disk config applies on its next start; never raises."""
        try:
            for name, srcs in new.items():
                if prev.get(name) == srcs:
                    continue
                # go2rtc accepts repeated ?src= params — one PUT (re)creates
                # the stream with all sources (main + ffmpeg AAC audio).
                await self._client.put("/api/streams", params={"name": name, "src": srcs})
            for name in prev:
                if name not in new:
                    await self._client.delete("/api/streams", params={"name": name})
            log.info("go2rtc streams synced incrementally (no restart)")
        except httpx.TransportError:
            log.info("go2rtc unreachable — config applies on its next start")
        except httpx.HTTPError as exc:
            log.warning("go2rtc incremental stream sync failed: %s", exc)

    async def _sync(self, cameras: list[dict[str, Any]]) -> None:
        """POST /api/restart; on failure PUT each stream individually."""
        try:
            resp = await self._client.post("/api/restart")
            if resp.status_code < 400:
                log.info("go2rtc restart requested")
                return
            log.warning("go2rtc restart returned %s; trying per-stream sync", resp.status_code)
        except httpx.TransportError:
            # go2rtc down entirely — per-stream PUTs to the same host are
            # pointless; it reads the new file when it (re)starts.
            log.info("go2rtc unreachable — config applies on its next start")
            return
        except httpx.HTTPError as exc:
            log.warning("go2rtc restart failed (%s); trying per-stream sync", exc)
        for cam in cameras:
            for name, srcs in stream_sources(cam).items():
                try:
                    await self._client.put("/api/streams", params={"name": name, "src": srcs})
                except httpx.HTTPError as exc:
                    log.warning("go2rtc stream sync failed for %s: %s", name, exc)
                    return

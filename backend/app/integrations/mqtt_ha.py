"""Outbound MQTT + Home Assistant auto-discovery publisher.

Vigilume PUBLISHES to the operator's MQTT broker so cameras/detections show
up as Home Assistant entities automatically (HA MQTT discovery). This is a
one-way, opt-in bridge — Vigilume is the MQTT client, HA is unrelated to the
removed inbound Frigate-MQTT. Optional two-way control lets HA drive a
camera's IR / spotlight / siren over the same paths routers/cameras uses.

Design:
- ``aiomqtt`` is imported LAZILY (``_import_aiomqtt``) — importing this module,
  or ``app.main``, never requires aiomqtt or a broker. Mirrors the lazy
  onnxruntime import in the detector.
- A single background task (``_run``) owns one aiomqtt connection with
  auto-reconnect + capped backoff. It NEVER crashes the app: a down/unreachable
  broker is logged and retried; every publish is best-effort.
- Last-Will (LWT) on ``<base_topic>/status`` = ``"offline"``; a retained
  ``"online"`` birth is published on connect. All entity STATE topics are
  retained so HA restores them after a restart.
- Discovery payloads (``build_discovery``) are pure functions so they are unit
  testable without a broker. One HA "device" per camera, all grouped under a
  Vigilume bridge device via ``via_device``.

Topic conventions (``b`` = base_topic, ``c`` = camera, ``L`` = label):
    b/status                        availability (online/offline, retained, LWT)
    b/c/L/state                     per-label binary_sensor (ON while in-frame)
    b/c/connectivity/state          camera reachability binary_sensor
    b/c/last_event/state            last event label (sensor)
    b/c/last_event/attributes       last event JSON attributes
    b/c/image/url                   annotated snapshot URL (MQTT image, opt-in)
    b/c/{ir,spotlight}/state        two-way switch state echo
    b/c/{ir,spotlight}/set          two-way switch command (subscribed)
    b/c/siren/set                   two-way siren button command (subscribed)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..amcrest.features import CAPABILITY_KEYS, static_capabilities

log = logging.getLogger(__name__)

# Lazy aiomqtt handle — see _import_aiomqtt. Kept module-global so a missing
# dependency is logged once, not on every reconnect.
_aiomqtt = None
_aiomqtt_failed = False

# Reconnect backoff bounds (seconds).
_BACKOFF_START_S = 1.0
_BACKOFF_MAX_S = 30.0
# Bound the outgoing queue so a long broker outage can't grow it without limit;
# retained STATE is republished from the cache on reconnect anyway.
_QUEUE_MAX = 2000

# HA binary_sensor device_class per label. Living things read as "occupancy",
# everything else as generic "motion" (documented in docs/home-assistant.md).
_OCCUPANCY_LABELS = {"person", "cat", "dog"}
_DEFAULT_DEVICE_CLASS = "motion"

# Command kinds that carry an on/off switch state (siren is a stateless button).
_SWITCH_KINDS = ("ir", "spotlight")


def _import_aiomqtt():
    """Import aiomqtt on first use; return the module or None (logged once).

    Mirrors the detector's lazy onnxruntime import so importing app.main on a
    host without aiomqtt / a broker is always fine.
    """
    global _aiomqtt, _aiomqtt_failed
    if _aiomqtt is not None:
        return _aiomqtt
    if _aiomqtt_failed:
        return None
    try:
        import aiomqtt  # type: ignore
    except Exception as exc:  # noqa: BLE001 — optional dependency
        _aiomqtt_failed = True
        log.warning("aiomqtt is not installed — MQTT/Home Assistant integration disabled: %s", exc)
        return None
    _aiomqtt = aiomqtt
    return aiomqtt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MqttConfig:
    enabled: bool
    host: str
    port: int
    username: str
    password: str
    discovery_prefix: str
    base_topic: str

    @classmethod
    def from_settings(cls, mqtt: dict[str, Any]) -> "MqttConfig":
        return cls(
            enabled=bool(mqtt.get("enabled", False)),
            host=str(mqtt.get("host") or "").strip(),
            port=int(mqtt.get("port") or 1883),
            username=str(mqtt.get("username") or ""),
            password=str(mqtt.get("password") or ""),
            discovery_prefix=str(mqtt.get("discovery_prefix") or "homeassistant").strip("/"),
            base_topic=str(mqtt.get("base_topic") or "vigilume").strip("/"),
        )

    @property
    def runnable(self) -> bool:
        """Enabled AND a broker host is configured."""
        return self.enabled and bool(self.host)


# --------------------------------------------------------------------------- #
# Topic + discovery helpers (pure — unit tested by tests/mqtt_smoke.py)
# --------------------------------------------------------------------------- #


def status_topic(base: str) -> str:
    return f"{base}/status"


def label_state_topic(base: str, camera: str, label: str) -> str:
    return f"{base}/{camera}/{label}/state"


def connectivity_state_topic(base: str, camera: str) -> str:
    return f"{base}/{camera}/connectivity/state"


def last_event_state_topic(base: str, camera: str) -> str:
    return f"{base}/{camera}/last_event/state"


def last_event_attributes_topic(base: str, camera: str) -> str:
    return f"{base}/{camera}/last_event/attributes"


def image_url_topic(base: str, camera: str) -> str:
    return f"{base}/{camera}/image/url"


def command_topic(base: str, camera: str, kind: str) -> str:
    return f"{base}/{camera}/{kind}/set"


def command_state_topic(base: str, camera: str, kind: str) -> str:
    return f"{base}/{camera}/{kind}/state"


def label_device_class(label: str) -> str:
    return "occupancy" if label in _OCCUPANCY_LABELS else _DEFAULT_DEVICE_CLASS


def _caps_of(cam: dict[str, Any]) -> dict[str, bool]:
    """Static capability map for the model with the camera's stored overrides
    merged over it (same rule as routers/cameras._caps_of)."""
    stored = cam.get("capabilities") or {}
    caps = static_capabilities(cam.get("model") or "")
    for key in CAPABILITY_KEYS:
        if key in stored and isinstance(stored[key], bool):
            caps[key] = stored[key]
    return caps


def _bridge_device(base: str) -> dict[str, Any]:
    return {
        "identifiers": [f"{base}_bridge"],
        "name": "Vigilume NVR",
        "manufacturer": "Vigilume NVR",
        "model": "Vigilume NVR",
    }


def _camera_device(base: str, cam: dict[str, Any]) -> dict[str, Any]:
    return {
        "identifiers": [f"{base}_{cam['name']}"],
        "name": f"Vigilume {cam.get('friendly_name') or cam['name']}",
        "manufacturer": "Vigilume NVR",
        "model": cam.get("model") or "camera",
        # Nest each camera under the Vigilume bridge so HA groups them.
        "via_device": f"{base}_bridge",
    }


def _availability(base: str) -> dict[str, Any]:
    return {
        "availability_topic": status_topic(base),
        "payload_available": "online",
        "payload_not_available": "offline",
    }


def build_discovery(
    cfg: MqttConfig,
    cameras: list[dict[str, Any]],
    public_url: str = "",
    commands_enabled: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    """All HA discovery (topic, payload) tuples for the given cameras.

    Pure function: no I/O. unique_ids are stable (``<base>_<camera>_<slug>``)
    so HA keeps the same entities across restarts. Two-way command entities are
    emitted only for cameras whose capabilities include them.
    """
    base = cfg.base_topic
    prefix = cfg.discovery_prefix
    avail = _availability(base)
    out: list[tuple[str, dict[str, Any]]] = []

    for cam in cameras:
        camera = cam["name"]
        device = _camera_device(base, cam)
        friendly = cam.get("friendly_name") or camera

        # Per tracked-label motion/occupancy binary_sensor.
        for label in cam.get("detect_objects") or []:
            uid = f"{base}_{camera}_{label}"
            topic = f"{prefix}/binary_sensor/{uid}/config"
            out.append((topic, {
                "name": f"{label} detected",
                "unique_id": uid,
                "object_id": uid,
                "state_topic": label_state_topic(base, camera, label),
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": label_device_class(label),
                "device": device,
                **avail,
            }))

        # Camera reachability (connectivity) binary_sensor.
        uid = f"{base}_{camera}_connectivity"
        out.append((f"{prefix}/binary_sensor/{uid}/config", {
            "name": "Connectivity",
            "unique_id": uid,
            "object_id": uid,
            "state_topic": connectivity_state_topic(base, camera),
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "connectivity",
            "device": device,
            **avail,
        }))

        # Last-event sensor (state = last label; JSON attributes carry the rest).
        uid = f"{base}_{camera}_last_event"
        out.append((f"{prefix}/sensor/{uid}/config", {
            "name": "Last event",
            "unique_id": uid,
            "object_id": uid,
            "state_topic": last_event_state_topic(base, camera),
            "json_attributes_topic": last_event_attributes_topic(base, camera),
            "icon": "mdi:cctv",
            "device": device,
            **avail,
        }))

        # Annotated-snapshot MQTT Image entity — opt-in/lightweight: we publish
        # a URL (url_topic), not raw bytes, so nothing large crosses the broker.
        # Only useful when a public_url is set (HA must be able to fetch it).
        if public_url:
            uid = f"{base}_{camera}_snapshot"
            out.append((f"{prefix}/image/{uid}/config", {
                "name": "Last snapshot",
                "unique_id": uid,
                "object_id": uid,
                "url_topic": image_url_topic(base, camera),
                "device": device,
                **avail,
            }))

        if not commands_enabled:
            continue

        caps = _caps_of(cam)
        # Two-way IR switch.
        if caps.get("ir"):
            uid = f"{base}_{camera}_ir"
            out.append((f"{prefix}/switch/{uid}/config", {
                "name": "IR night vision",
                "unique_id": uid,
                "object_id": uid,
                "state_topic": command_state_topic(base, camera, "ir"),
                "command_topic": command_topic(base, camera, "ir"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:led-on",
                "device": device,
                **avail,
            }))
        # Two-way spotlight switch.
        if caps.get("white_light"):
            uid = f"{base}_{camera}_spotlight"
            out.append((f"{prefix}/switch/{uid}/config", {
                "name": "Spotlight",
                "unique_id": uid,
                "object_id": uid,
                "state_topic": command_state_topic(base, camera, "spotlight"),
                "command_topic": command_topic(base, camera, "spotlight"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:spotlight",
                "device": device,
                **avail,
            }))
        # Two-way siren button (stateless).
        if caps.get("siren"):
            uid = f"{base}_{camera}_siren"
            out.append((f"{prefix}/button/{uid}/config", {
                "name": "Siren",
                "unique_id": uid,
                "object_id": uid,
                "command_topic": command_topic(base, camera, "siren"),
                "payload_press": "PRESS",
                "icon": "mdi:alarm-light",
                "device": device,
                **avail,
            }))
        _ = friendly  # (kept for future name templating; silence linters)

    return out


# --------------------------------------------------------------------------- #
# Publisher
# --------------------------------------------------------------------------- #


class MqttPublisher:
    """Resilient outbound MQTT publisher + HA discovery + optional command sub.

    Lifecycle: ``start`` (spawn the background task), ``restart`` (reload config
    from settings and reconnect / stop), ``stop`` (cancel + disconnect). The
    event hooks (``publish_event`` / ``publish_connectivity``) are no-ops when
    the integration is disabled and are safe to call from the pipeline / prober
    on the app event loop.
    """

    def __init__(self, db, settings, prober, auth):
        self._db = db
        self._settings = settings
        self._prober = prober
        self._auth = auth

        self._cfg = MqttConfig.from_settings(settings.mqtt)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._connected = False
        # Retained STATE cache (topic -> payload) republished on every connect
        # so in-progress ON states / last events survive a broker restart.
        self._state_cache: dict[str, str] = {}
        # Outgoing publish queue used only while connected (retain flag per msg).
        self._queue: "asyncio.Queue[tuple[str, str, bool]]" = asyncio.Queue(maxsize=_QUEUE_MAX)

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._cfg = MqttConfig.from_settings(self._settings.mqtt)
        if not self._cfg.runnable:
            log.info("MQTT/Home Assistant publisher disabled (enabled=%s host=%r)",
                     self._cfg.enabled, self._cfg.host)
            return
        if _import_aiomqtt() is None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="mqtt-ha-publisher")
        log.info("MQTT/Home Assistant publisher started -> %s:%d", self._cfg.host, self._cfg.port)

    async def stop(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._connected = False
        # Drain any queued messages so a later restart starts clean.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def restart(self) -> None:
        """Reload config from settings and reconnect (or stop if now disabled).
        Never raises — a settings save must not fail on a down broker."""
        try:
            await self.stop()
        except Exception:  # noqa: BLE001
            log.exception("error stopping MQTT publisher during restart")
        # Fresh cache/state: the new broker/base_topic may differ entirely.
        self._state_cache.clear()
        await self.start()

    # ---------- connection loop ----------

    async def _run(self) -> None:
        aiomqtt = _import_aiomqtt()
        if aiomqtt is None:
            return
        backoff = _BACKOFF_START_S
        while not self._stop.is_set():
            cfg = self._cfg
            try:
                will = aiomqtt.Will(
                    topic=status_topic(cfg.base_topic),
                    payload="offline",
                    qos=1,
                    retain=True,
                )
                client_kwargs: dict[str, Any] = {
                    "hostname": cfg.host,
                    "port": cfg.port,
                    "will": will,
                }
                if cfg.username:
                    client_kwargs["username"] = cfg.username
                    client_kwargs["password"] = cfg.password
                async with aiomqtt.Client(**client_kwargs) as client:
                    self._connected = True
                    backoff = _BACKOFF_START_S
                    log.info("MQTT connected to %s:%d", cfg.host, cfg.port)
                    await self._on_connect(client)
                    await self._serve(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — MqttError + anything: never crash
                log.warning("MQTT connection to %s:%d failed/dropped: %s — retrying in %.0fs",
                            cfg.host, cfg.port, exc, backoff)
            finally:
                self._connected = False
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, _BACKOFF_MAX_S)

    async def _serve(self, client) -> None:
        """Run the writer (drains the queue) and reader (commands) until the
        connection drops or we're told to stop."""
        writer = asyncio.create_task(self._writer(client))
        try:
            async for message in client.messages:
                try:
                    await self._handle_command(client, message)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad command must not drop the link
                    log.exception("MQTT command handling failed")
        finally:
            writer.cancel()
            try:
                await writer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _writer(self, client) -> None:
        while True:
            topic, payload, retain = await self._queue.get()
            try:
                await client.publish(topic, payload=payload, qos=0, retain=retain)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — publish failure just drops one msg
                log.debug("MQTT publish to %s failed", topic)

    async def _on_connect(self, client) -> None:
        cfg = self._cfg
        # Birth: mark the bridge online (retained).
        await client.publish(status_topic(cfg.base_topic), payload="online", qos=1, retain=True)
        cameras = await self._db.list_cameras()
        # Discovery (retained) for every entity.
        public_url = self._public_url()
        for topic, payload in build_discovery(cfg, cameras, public_url=public_url):
            await client.publish(topic, payload=json.dumps(payload), qos=0, retain=True)
        # Seed a defined state for every entity so HA never shows "unknown".
        for cam in cameras:
            camera = cam["name"]
            for label in cam.get("detect_objects") or []:
                topic = label_state_topic(cfg.base_topic, camera, label)
                self._state_cache.setdefault(topic, "OFF")
            conn = connectivity_state_topic(cfg.base_topic, camera)
            self._state_cache[conn] = "ON" if self._prober.is_online(camera) else "OFF"
        # Republish the full retained-state cache (in-progress ON states, last
        # events, connectivity) so a broker restart is transparent.
        for topic, payload in list(self._state_cache.items()):
            await client.publish(topic, payload=payload, qos=0, retain=True)
        # Subscribe to every camera command topic for two-way control.
        await client.subscribe(f"{cfg.base_topic}/+/+/set", qos=0)

    # ---------- command handling (two-way control) ----------

    async def _handle_command(self, client, message) -> None:
        cfg = self._cfg
        topic = str(message.topic)
        prefix = f"{cfg.base_topic}/"
        if not topic.startswith(prefix) or not topic.endswith("/set"):
            return
        parts = topic[len(prefix):-len("/set")].split("/")
        if len(parts) != 2:
            return
        camera, kind = parts
        payload_raw = message.payload
        payload = payload_raw.decode() if isinstance(payload_raw, (bytes, bytearray)) else str(payload_raw)
        payload_on = payload.strip().upper() == "ON"

        cam = await self._db.get_camera(camera)
        if cam is None:
            log.info("MQTT command for unknown camera %r ignored", camera)
            return
        caps = _caps_of(cam)

        if kind == "ir" and caps.get("ir"):
            await self._control(cam, "set_ir_mode", "on" if payload_on else "off")
            await self._echo_state(client, command_state_topic(cfg.base_topic, camera, "ir"),
                                   "ON" if payload_on else "OFF")
        elif kind == "spotlight" and caps.get("white_light"):
            await self._control(cam, "set_white_light", mode="on" if payload_on else "off")
            await self._echo_state(client, command_state_topic(cfg.base_topic, camera, "spotlight"),
                                   "ON" if payload_on else "OFF")
        elif kind == "siren" and caps.get("siren"):
            await self._control(cam, "play_tone")
            # Siren is a stateless button — no state echo.
        else:
            log.info("MQTT command %s/%s unsupported for camera %r", camera, kind, camera)

    async def _control(self, cam: dict[str, Any], method: str, *args: Any, **kwargs: Any) -> None:
        """Invoke an Amcrest control via the SAME client path routers/cameras
        uses. Best-effort: device errors are logged, never raised."""
        from ..amcrest.client import AmcrestClient, AmcrestError

        # Pass the model so per-model control gating (e.g. coaxialControlIO
        # white light on the EW turrets) matches the routers/cameras path;
        # without it set_white_light would drive the inert Lighting_V2 CGI.
        client = AmcrestClient(
            cam["ip"], cam["username"], cam["password"], model=cam.get("model", "")
        )
        try:
            await getattr(client, method)(*args, **kwargs)
            log.info("MQTT command applied: %s.%s on camera %s", method, args or kwargs, cam["name"])
        except AmcrestError as exc:
            log.warning("MQTT command %s on %s failed: %s", method, cam["name"], exc)
        except Exception:  # noqa: BLE001 — never let a device error drop the link
            log.exception("MQTT command %s on %s raised", method, cam["name"])
        finally:
            await client.aclose()

    async def _echo_state(self, client, topic: str, payload: str) -> None:
        self._state_cache[topic] = payload
        try:
            await client.publish(topic, payload=payload, qos=0, retain=True)
        except Exception:  # noqa: BLE001
            log.debug("MQTT state echo to %s failed", topic)

    # ---------- event hooks (called from the pipeline / prober) ----------

    async def publish_event(self, camera: str, label: str, etype: str, event_row: Optional[dict[str, Any]]) -> None:
        """A detection event new/update/end — drive the label binary_sensor and
        the last-event sensor. No-op when the integration is disabled."""
        if not self._cfg.runnable:
            return
        base = self._cfg.base_topic
        active = etype in ("new", "update")
        self._set_state(label_state_topic(base, camera, label), "ON" if active else "OFF")
        if event_row is not None:
            attrs = self._event_attributes(event_row)
            self._set_state(last_event_state_topic(base, camera), label)
            self._set_state(last_event_attributes_topic(base, camera), json.dumps(attrs))
            snapshot_url = attrs.get("snapshot_url")
            if snapshot_url and event_row.get("has_snapshot"):
                self._set_state(image_url_topic(base, camera), snapshot_url)

    async def publish_connectivity(self, camera: str, online: bool) -> None:
        """Camera reachability changed (from the CameraProber). No-op when the
        integration is disabled."""
        if not self._cfg.runnable:
            return
        self._set_state(connectivity_state_topic(self._cfg.base_topic, camera), "ON" if online else "OFF")

    # ---------- internals ----------

    def _event_attributes(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "count": row.get("count"),
            "score": row.get("score"),
            "started": row.get("start_time"),
            "ended": row.get("end_time"),
            "snapshot_url": self._snapshot_url(int(row["id"])) if row.get("id") is not None else None,
        }

    def _public_url(self) -> str:
        return self._settings.public_url

    def _snapshot_url(self, event_id: int) -> str:
        base = self._public_url()
        token = self._auth.create_media_token(resource=f"event:{event_id}")
        return f"{base}/api/events/{event_id}/snapshot.jpg?token={token}"

    def _set_state(self, topic: str, payload: str) -> None:
        """Update the retained-state cache and, if connected, enqueue a publish.
        Cache-first means a reconnect republishes the latest value even if we
        were offline when it changed. Unchanged values are cached but not
        re-enqueued: the broker already retains them (and a reconnect
        republishes the whole cache), so e.g. detection 'update' ticks don't
        republish an identical label-sensor 'ON' every second."""
        old = self._state_cache.get(topic)
        self._state_cache[topic] = payload
        if old == payload:
            return
        if self._connected:
            try:
                self._queue.put_nowait((topic, payload, True))
            except asyncio.QueueFull:
                # Broker is badly backed up; the cache still holds the latest
                # value and will be republished on the next clean reconnect.
                log.debug("MQTT outgoing queue full — dropping live publish for %s", topic)

    # ---------- introspection (for the test route / diagnostics) ----------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def config(self) -> MqttConfig:
        return self._cfg


async def test_connection(mqtt: dict[str, Any], *, timeout: float = 6.0) -> dict[str, Any]:
    """Attempt a connect + a single publish with the given mqtt settings.

    Returns {ok: bool, detail: str} — used by POST /api/integrations/mqtt/test.
    Never raises. Distinguishes (best-effort) auth failure from an unreachable
    broker via the aiomqtt/paho error text.
    """
    cfg = MqttConfig.from_settings(mqtt)
    if not cfg.host:
        return {"ok": False, "detail": "No MQTT host configured"}
    aiomqtt = _import_aiomqtt()
    if aiomqtt is None:
        return {"ok": False, "detail": "aiomqtt is not installed on the server"}

    client_kwargs: dict[str, Any] = {"hostname": cfg.host, "port": cfg.port}
    if cfg.username:
        client_kwargs["username"] = cfg.username
        client_kwargs["password"] = cfg.password

    async def _attempt() -> None:
        async with aiomqtt.Client(**client_kwargs) as client:
            await client.publish(status_topic(cfg.base_topic), payload="online", qos=0, retain=True)

    try:
        await asyncio.wait_for(_attempt(), timeout=timeout)
        return {"ok": True, "detail": f"Connected to {cfg.host}:{cfg.port} and published a test message"}
    except asyncio.TimeoutError:
        return {"ok": False, "detail": f"Timed out connecting to {cfg.host}:{cfg.port}"}
    except Exception as exc:  # noqa: BLE001 — aiomqtt.MqttError etc.
        text = str(exc).lower()
        if any(w in text for w in ("not authoriz", "auth", "password", "credential", "bad user")):
            detail = "Authentication failed (check username/password)"
        elif any(w in text for w in ("refus", "unreach", "no route", "name or service", "timed out", "connection")):
            detail = f"Broker unreachable at {cfg.host}:{cfg.port}"
        else:
            detail = f"Connection failed: {exc}"
        return {"ok": False, "detail": detail}

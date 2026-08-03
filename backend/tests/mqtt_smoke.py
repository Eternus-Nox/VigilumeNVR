"""Smoke suite for the MQTT + Home Assistant integration
(app/integrations/mqtt_ha.py + routers/integrations.py + settings.mqtt).

Everything runs against a MOCK aiomqtt (no real broker) and a fake Amcrest
client — nothing here touches the network. Coverage:

  - pure discovery generation: per camera/label topics, HA device grouping
    (via_device), device_class (motion/occupancy/connectivity), STABLE
    unique_ids, capability-gated two-way command entities, image entity only
    when a public_url is set
  - availability: LWT (offline, retained) wired into the client + birth
    (online, retained) on connect
  - event new -> label binary_sensor ON, end -> OFF (retained), last-event
    sensor + attributes, snapshot image url
  - connectivity published from a camera_status (prober) hook
  - settings round-trip (MqttSettings / AppSettings) + publisher restart on a
    settings change reconnects to the NEW broker
  - two-way command: subscribe wildcard, a command message invokes the SAME
    Amcrest control path + echoes switch state; caps-gated (unsupported ignored)
  - resilience: an unreachable broker NEVER raises — _run retries with backoff,
    the app is unaffected, and hooks stay safe no-ops
  - lazy import: importing app.main / mqtt_ha needs neither aiomqtt nor a broker

Usage: python backend/tests/mqtt_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

from app.integrations import mqtt_ha  # noqa: E402
from app.integrations.mqtt_ha import (  # noqa: E402
    MqttConfig,
    MqttPublisher,
    build_discovery,
    command_state_topic,
    command_topic,
    connectivity_state_topic,
    label_state_topic,
    last_event_attributes_topic,
    last_event_state_topic,
    status_topic,
    test_connection,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# --------------------------------------------------------------------------- #
# Mock aiomqtt
# --------------------------------------------------------------------------- #


class MqttError(Exception):
    pass


class Will:
    def __init__(self, topic, payload, qos=0, retain=False):
        self.topic, self.payload, self.qos, self.retain = topic, payload, qos, retain


class _FakeMessages:
    def __init__(self, client):
        self._client = client

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._client._incoming.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


class _FakeMsg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload if isinstance(payload, bytes) else payload.encode()


class FakeClient:
    instances: list["FakeClient"] = []
    fail_connect = False

    def __init__(self, hostname, port, will=None, username=None, password=None, **kw):
        self.hostname, self.port, self.will = hostname, port, will
        self.username, self.password = username, password
        self.published: list[tuple[str, str, int, bool]] = []
        self.subscriptions: list[str] = []
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.messages = _FakeMessages(self)
        FakeClient.instances.append(self)

    async def __aenter__(self):
        if FakeClient.fail_connect:
            raise MqttError("[Errno 111] Connection refused")
        return self

    async def __aexit__(self, *a):
        return False

    async def publish(self, topic, payload=None, qos=0, retain=False):
        pl = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
        self.published.append((topic, pl, qos, retain))

    async def subscribe(self, topic, qos=0):
        self.subscriptions.append(topic)

    def feed(self, topic, payload):
        self._incoming.put_nowait(_FakeMsg(topic, payload))


class FakeAiomqtt:
    Client = FakeClient
    Will = Will
    MqttError = MqttError


# --------------------------------------------------------------------------- #
# Fake Amcrest client (records control calls)
# --------------------------------------------------------------------------- #


class FakeAmcrestError(Exception):
    pass


class FakeAmcrest:
    calls: list[tuple] = []
    fail = False

    def __init__(self, ip, username, password, timeout=8.0, model=""):
        self.ip = ip
        # The MQTT control path must forward the camera model so per-model
        # gating (e.g. coaxialControlIO white light on EW turrets) matches the
        # HTTP router path; record it so the smoke test can assert that.
        self.model = model
        FakeAmcrest.calls.append(("__init__", model))

    async def set_ir_mode(self, mode):
        if FakeAmcrest.fail:
            raise FakeAmcrestError("device unreachable")
        FakeAmcrest.calls.append(("ir", mode))

    async def set_white_light(self, mode=None, brightness=None):
        FakeAmcrest.calls.append(("spotlight", mode))

    async def play_tone(self, duration_s=10):
        FakeAmcrest.calls.append(("siren", duration_s))

    async def aclose(self):
        pass


# --------------------------------------------------------------------------- #
# Fakes for the publisher's collaborators
# --------------------------------------------------------------------------- #


TURRET = {
    "name": "front", "friendly_name": "Front Door", "model": "IP5M-T1277EW-AI",
    "ip": "192.0.2.10", "username": "admin", "password": "pw",
    "detect_objects": ["person", "car"],
    "capabilities": {"ir": True, "white_light": True, "siren": False},
}
DOORBELL = {
    "name": "porch", "friendly_name": "Porch", "model": "AD410",
    "ip": "192.0.2.11", "username": "admin", "password": "pw",
    "detect_objects": ["person"],
    "capabilities": {"ir": True, "white_light": False, "siren": True},
}


class FakeDB:
    def __init__(self, cameras):
        self._cams = {c["name"]: c for c in cameras}

    async def list_cameras(self):
        return list(self._cams.values())

    async def get_camera(self, name):
        return self._cams.get(name)


class FakeSettings:

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False
    def __init__(self, mqtt, public_url=""):
        self.mqtt = mqtt
        self._public_url = public_url

    @property
    def public_url(self):
        return self._public_url.rstrip("/")


class FakeProber:
    def __init__(self, online=None):
        self._online = online or {}

    def is_online(self, name):
        return self._online.get(name, False)


class FakeAuth:
    # Mirrors AuthService.create_media_token, INCLUDING `resource` — the
    # publisher binds each snapshot URL's token to its event so a token read off
    # a retained MQTT topic cannot open any other event.
    def create_media_token(self, username: str = "admin", role: str = "viewer",
                           resource: str = "") -> str:
        self.last_resource = resource
        return "MEDIATOKEN"


def base_mqtt(**over):
    cfg = {
        "enabled": True, "host": "192.168.1.10", "port": 1883,
        "username": "homeassistant", "password": "secret",
        "discovery_prefix": "homeassistant", "base_topic": "sentinel",
    }
    cfg.update(over)
    return cfg


async def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return False


def _find(published, topic):
    for t, payload, qos, retain in published:
        if t == topic:
            return payload, retain
    return None


# --------------------------------------------------------------------------- #
# Pure discovery tests
# --------------------------------------------------------------------------- #


def discovery_checks():
    print("discovery: topics / device grouping / device_class / unique_ids")
    cfg = MqttConfig.from_settings(base_mqtt())
    disco = build_discovery(cfg, [TURRET, DOORBELL], public_url="https://nvr.example")
    topics = {t: p for t, p in disco}

    # per-label binary_sensor
    t = "homeassistant/binary_sensor/sentinel_front_person/config"
    check(t in topics, "person binary_sensor discovery topic present")
    p = topics[t]
    check(p["state_topic"] == "sentinel/front/person/state", "person state_topic correct")
    check(p["device_class"] == "occupancy", "person -> device_class occupancy")
    check(p["unique_id"] == "sentinel_front_person", "person unique_id stable/derived")
    check(p["payload_on"] == "ON" and p["payload_off"] == "OFF", "binary payloads ON/OFF")
    check(p["availability_topic"] == "sentinel/status"
          and p["payload_not_available"] == "offline", "availability wired into entity")

    car = topics["homeassistant/binary_sensor/sentinel_front_car/config"]
    check(car["device_class"] == "motion", "car -> device_class motion")

    # device grouping: camera device nests under the bridge via via_device
    dev = p["device"]
    check(dev["identifiers"] == ["sentinel_front"], "camera device identifier")
    check(dev["via_device"] == "sentinel_bridge", "camera device nests under Vigilume bridge")
    check(dev["name"] == "Vigilume Front Door", "camera device uses friendly name")

    # connectivity sensor
    conn = topics["homeassistant/binary_sensor/sentinel_front_connectivity/config"]
    check(conn["device_class"] == "connectivity"
          and conn["state_topic"] == "sentinel/front/connectivity/state",
          "connectivity binary_sensor present with device_class connectivity")

    # last-event sensor + attributes topic
    le = topics["homeassistant/sensor/sentinel_front_last_event/config"]
    check(le["state_topic"] == "sentinel/front/last_event/state"
          and le["json_attributes_topic"] == "sentinel/front/last_event/attributes",
          "last-event sensor exposes state + json attributes topics")

    # image entity present because public_url is set (url_topic, not raw bytes)
    img = topics.get("homeassistant/image/sentinel_front_snapshot/config")
    check(img is not None and img["url_topic"] == "sentinel/front/image/url",
          "image entity uses url_topic (lightweight) when public_url set")

    # capability-gated two-way controls
    check("homeassistant/switch/sentinel_front_ir/config" in topics, "turret IR switch (caps.ir)")
    check("homeassistant/switch/sentinel_front_spotlight/config" in topics,
          "turret spotlight switch (caps.white_light)")
    check("homeassistant/button/sentinel_front_siren/config" not in topics,
          "turret has NO siren button (caps.siren false)")
    check("homeassistant/button/sentinel_porch_siren/config" in topics,
          "doorbell HAS siren button (caps.siren true)")
    check("homeassistant/switch/sentinel_porch_spotlight/config" not in topics,
          "doorbell has NO spotlight (caps.white_light false)")
    ir_sw = topics["homeassistant/switch/sentinel_front_ir/config"]
    check(ir_sw["command_topic"] == "sentinel/front/ir/set"
          and ir_sw["state_topic"] == "sentinel/front/ir/state",
          "IR switch has command + state topics")

    # unique_ids stable across identical calls
    disco2 = build_discovery(cfg, [TURRET, DOORBELL], public_url="https://nvr.example")
    check([t for t, _ in disco] == [t for t, _ in disco2], "discovery is deterministic (stable topics)")

    # no image entity when public_url empty
    disco_np = dict(build_discovery(cfg, [TURRET], public_url=""))
    check("homeassistant/image/sentinel_front_snapshot/config" not in disco_np,
          "image entity omitted when no public_url (HA couldn't fetch it)")

    # commands_enabled=False scaffolds read-only only
    disco_ro = dict(build_discovery(cfg, [TURRET], commands_enabled=False))
    check("homeassistant/switch/sentinel_front_ir/config" not in disco_ro,
          "commands_enabled=False drops two-way entities")


def config_checks():
    print("config: MqttConfig.from_settings + runnable gate")
    c = MqttConfig.from_settings(base_mqtt(base_topic="/sentinel/", discovery_prefix="ha/"))
    check(c.base_topic == "sentinel" and c.discovery_prefix == "ha",
          "base_topic/discovery_prefix stripped of slashes")
    check(MqttConfig.from_settings(base_mqtt(enabled=False)).runnable is False,
          "disabled -> not runnable")
    check(MqttConfig.from_settings(base_mqtt(host="")).runnable is False,
          "no host -> not runnable")
    check(MqttConfig.from_settings(base_mqtt()).runnable is True, "enabled + host -> runnable")


# --------------------------------------------------------------------------- #
# Settings model round-trip (Pydantic) + publisher restart
# --------------------------------------------------------------------------- #


def settings_model_checks():
    print("settings: MqttSettings / AppSettings round-trip + validation")
    from app.routers.settings import AppSettings, MqttSettings

    m = MqttSettings()
    check(m.enabled is False and m.port == 1883 and m.base_topic == "sentinel"
          and m.discovery_prefix == "homeassistant", "MqttSettings defaults match the contract")

    rt = MqttSettings(**base_mqtt(base_topic="  home/ ")).model_dump()
    check(rt["base_topic"] == "home", "base_topic trimmed/stripped on validation")

    bad = False
    try:
        MqttSettings(**base_mqtt(base_topic="a/#"))
    except Exception:
        bad = True
    check(bad, "wildcard in base_topic rejected (422-shaped)")

    app_defaults = AppSettings().model_dump()
    check("mqtt" in app_defaults and app_defaults["mqtt"]["enabled"] is False,
          "AppSettings includes the mqtt block (round-trips through PUT /api/settings)")


# --------------------------------------------------------------------------- #
# Live publisher tests (against the mock aiomqtt)
# --------------------------------------------------------------------------- #


def _new_publisher(settings, prober=None):
    return MqttPublisher(FakeDB([TURRET, DOORBELL]), settings, prober or FakeProber(), FakeAuth())


async def publisher_lifecycle_checks():
    print("publisher: birth/LWT, discovery, event ON/OFF, connectivity, commands")
    FakeClient.instances.clear()
    FakeClient.fail_connect = False
    settings = FakeSettings(base_mqtt(), public_url="https://nvr.example")
    pub = _new_publisher(settings, FakeProber({"front": True, "porch": False}))

    await pub.start()
    ok = await _wait_for(lambda: FakeClient.instances and pub.connected)
    check(ok, "publisher connected to the mock broker")
    client = FakeClient.instances[-1]

    # LWT wired into the client
    check(client.will is not None and client.will.topic == "sentinel/status"
          and client.will.payload == "offline" and client.will.retain is True,
          "Last-Will = sentinel/status 'offline' retained")
    check(client.username == "homeassistant" and client.password == "secret",
          "broker credentials passed to the client")

    # birth published (online, retained)
    ok = await _wait_for(lambda: _find(client.published, "sentinel/status") is not None)
    check(ok, "birth published")
    birth = _find(client.published, "sentinel/status")
    check(birth == ("online", True), "birth = 'online' retained")

    # discovery published + retained
    ok = await _wait_for(
        lambda: _find(client.published, "homeassistant/binary_sensor/sentinel_front_person/config") is not None)
    check(ok, "discovery config published")
    disc = _find(client.published, "homeassistant/binary_sensor/sentinel_front_person/config")
    check(disc[1] is True, "discovery config retained")
    check(json.loads(disc[0])["device"]["via_device"] == "sentinel_bridge",
          "published discovery keeps device grouping")

    # seeded initial state: connectivity from the prober, labels OFF
    ok = await _wait_for(lambda: _find(client.published, "sentinel/front/connectivity/state") is not None)
    check(ok, "connectivity seeded on connect")
    check(_find(client.published, "sentinel/front/connectivity/state") == ("ON", True),
          "front connectivity seeded ON (prober online) retained")
    check(_find(client.published, "sentinel/porch/connectivity/state") == ("OFF", True),
          "porch connectivity seeded OFF (prober offline)")
    check(_find(client.published, "sentinel/front/person/state") == ("OFF", True),
          "label sensor seeded OFF on connect")

    # command subscription (wildcard)
    check("sentinel/+/+/set" in client.subscriptions, "subscribed to the command wildcard")

    # ----- event new -> ON, last-event sensor + attributes -----
    client.published.clear()
    row = {"id": 42, "camera": "front", "label": "person", "count": 2, "score": 0.91,
           "start_time": 1000.0, "end_time": None, "has_snapshot": True}
    await pub.publish_event("front", "person", "new", row)
    ok = await _wait_for(lambda: _find(client.published, "sentinel/front/person/state") == ("ON", True))
    check(ok, "event new -> person binary_sensor ON (retained)")
    check(_find(client.published, last_event_state_topic("sentinel", "front")) == ("person", True),
          "last-event sensor state = label")
    attrs_pub = _find(client.published, last_event_attributes_topic("sentinel", "front"))
    attrs = json.loads(attrs_pub[0])
    check(attrs["count"] == 2 and attrs["score"] == 0.91 and attrs["started"] == 1000.0,
          "last-event attributes carry count/score/started")
    check(attrs["snapshot_url"] == "https://nvr.example/api/events/42/snapshot.jpg?token=MEDIATOKEN",
          "snapshot_url built from public_url + media token")
    check(_find(client.published, "sentinel/front/image/url") is not None,
          "image url published for the annotated snapshot")

    # ----- event end -> OFF -----
    client.published.clear()
    await pub.publish_event("front", "person", "end", {**row, "end_time": 1010.0})
    ok = await _wait_for(lambda: _find(client.published, "sentinel/front/person/state") == ("OFF", True))
    check(ok, "event end -> person binary_sensor OFF (retained)")

    # ----- connectivity hook (from a camera_status) -----
    client.published.clear()
    await pub.publish_connectivity("porch", True)
    ok = await _wait_for(lambda: _find(client.published, "sentinel/porch/connectivity/state") == ("ON", True))
    check(ok, "publish_connectivity -> retained ON")

    # ----- two-way command: IR switch on -> Amcrest set_ir_mode + state echo -----
    import app.amcrest.client as amcrest_mod
    orig_client, orig_err = amcrest_mod.AmcrestClient, amcrest_mod.AmcrestError
    amcrest_mod.AmcrestClient = FakeAmcrest
    amcrest_mod.AmcrestError = FakeAmcrestError
    try:
        FakeAmcrest.calls.clear()
        client.published.clear()
        client.feed(command_topic("sentinel", "front", "ir"), "ON")
        ok = await _wait_for(lambda: ("ir", "on") in FakeAmcrest.calls)
        check(ok, "IR command invoked AmcrestClient.set_ir_mode('on')")
        ok = await _wait_for(
            lambda: _find(client.published, command_state_topic("sentinel", "front", "ir")) == ("ON", True))
        check(ok, "IR switch state echoed ON (retained)")

        # spotlight command
        FakeAmcrest.calls.clear()
        client.feed(command_topic("sentinel", "front", "spotlight"), "OFF")
        ok = await _wait_for(lambda: ("spotlight", "off") in FakeAmcrest.calls)
        check(ok, "spotlight command invoked set_white_light(mode='off')")
        # The control client must be built WITH the camera model so EW-turret
        # coaxialControlIO gating fires instead of the inert Lighting_V2 CGI.
        check(("__init__", "IP5M-T1277EW-AI") in FakeAmcrest.calls,
              "MQTT control client constructed with the camera model (coax gating)")

        # siren button on the doorbell
        FakeAmcrest.calls.clear()
        client.feed(command_topic("sentinel", "porch", "siren"), "PRESS")
        ok = await _wait_for(lambda: any(c[0] == "siren" for c in FakeAmcrest.calls))
        check(ok, "siren command invoked play_tone")

        # capability-gated: siren on the turret (no siren cap) is ignored
        FakeAmcrest.calls.clear()
        client.feed(command_topic("sentinel", "front", "siren"), "PRESS")
        await asyncio.sleep(0.1)
        check(not any(c[0] == "siren" for c in FakeAmcrest.calls),
              "siren command on a non-siren camera is ignored (caps-gated)")

        # unknown camera command ignored (no crash)
        client.feed(command_topic("sentinel", "ghost", "ir"), "ON")
        await asyncio.sleep(0.1)
        check(pub.connected, "unknown-camera command did not drop the connection")
    finally:
        amcrest_mod.AmcrestClient = orig_client
        amcrest_mod.AmcrestError = orig_err

    await pub.stop()
    check(not pub.connected, "publisher stop() disconnects")


async def restart_checks():
    print("publisher: restart on settings change reconnects to the new broker")
    FakeClient.instances.clear()
    FakeClient.fail_connect = False
    settings = FakeSettings(base_mqtt(host="192.168.1.10"))
    pub = _new_publisher(settings)
    await pub.start()
    ok = await _wait_for(lambda: pub.connected)
    check(ok, "connected to first broker")
    check(FakeClient.instances[-1].hostname == "192.168.1.10", "first broker host")

    # operator edits settings.mqtt then the router calls restart()
    settings.mqtt = base_mqtt(host="10.0.0.5")
    await pub.restart()
    ok = await _wait_for(lambda: pub.connected and FakeClient.instances[-1].hostname == "10.0.0.5")
    check(ok, "restart() reconnected to the NEW broker host")

    # disabling then restart -> no connection, no error
    settings.mqtt = base_mqtt(enabled=False)
    await pub.restart()
    await asyncio.sleep(0.05)
    check(not pub.connected, "restart with enabled=false stops the publisher")
    await pub.stop()


async def resilience_checks():
    print("publisher: unreachable broker never raises (retries, app unaffected)")
    FakeClient.instances.clear()
    FakeClient.fail_connect = True
    old_start, old_max = mqtt_ha._BACKOFF_START_S, mqtt_ha._BACKOFF_MAX_S
    mqtt_ha._BACKOFF_START_S, mqtt_ha._BACKOFF_MAX_S = 0.01, 0.03
    try:
        settings = FakeSettings(base_mqtt())
        pub = _new_publisher(settings)
        await pub.start()  # must not raise
        # give the loop time to fail + retry several times
        await asyncio.sleep(0.15)
        check(not pub.connected, "stays disconnected while the broker is down")
        check(len(FakeClient.instances) >= 2, "reconnect loop retried (multiple connect attempts)")
        # hooks are safe no-ops while disconnected (cache-only, never raise)
        await pub.publish_event("front", "person", "new",
                                {"id": 1, "camera": "front", "label": "person", "has_snapshot": False})
        await pub.publish_connectivity("front", True)
        check(True, "hooks are safe no-ops while disconnected")
        # recovery: broker comes back -> next attempt connects and republishes ON
        FakeClient.fail_connect = False
        ok = await _wait_for(lambda: pub.connected, timeout=2.0)
        check(ok, "publisher recovers and connects once the broker is back")
        client = FakeClient.instances[-1]
        ok = await _wait_for(lambda: _find(client.published, "sentinel/front/person/state") == ("ON", True))
        check(ok, "cached in-progress ON state republished after recovery")
        await pub.stop()
    finally:
        mqtt_ha._BACKOFF_START_S, mqtt_ha._BACKOFF_MAX_S = old_start, old_max
        FakeClient.fail_connect = False


async def test_route_checks():
    print("test_connection: connected / unreachable / auth-failed / no-host")
    FakeClient.fail_connect = False
    res = await test_connection(base_mqtt())
    check(res["ok"] is True, "test_connection ok when broker reachable")

    res = await test_connection(base_mqtt(host=""))
    check(res["ok"] is False and "host" in res["detail"].lower(), "no host -> ok:false")

    FakeClient.fail_connect = True
    res = await test_connection(base_mqtt())
    check(res["ok"] is False and "unreachable" in res["detail"].lower(),
          "connection refused -> 'unreachable' detail")
    FakeClient.fail_connect = False


def disabled_noop_checks():
    print("disabled: hooks are cheap no-ops, start() connects nothing")

    async def run():
        FakeClient.instances.clear()
        settings = FakeSettings(base_mqtt(enabled=False))
        pub = _new_publisher(settings)
        await pub.start()
        await asyncio.sleep(0.02)
        check(len(FakeClient.instances) == 0, "disabled publisher opens no connection")
        await pub.publish_event("front", "person", "new", {"id": 1})
        await pub.publish_connectivity("front", True)
        check(True, "hooks no-op when disabled")
        await pub.stop()

    asyncio.run(run())


def main():
    # aiomqtt is imported lazily — importing the module needed neither aiomqtt
    # nor a broker. Now install the MOCK so the live tests have a client.
    mqtt_ha._aiomqtt = FakeAiomqtt
    mqtt_ha._aiomqtt_failed = False

    discovery_checks()
    config_checks()
    settings_model_checks()
    disabled_noop_checks()
    asyncio.run(publisher_lifecycle_checks())
    asyncio.run(restart_checks())
    asyncio.run(resilience_checks())
    asyncio.run(test_route_checks())

    print(f"ALL PASSED ({PASS} checks)")


if __name__ == "__main__":
    main()

# Home Assistant (MQTT auto-discovery)

Vigilume can **publish** to your MQTT broker so your cameras and detections show up
in Home Assistant automatically, with no manual YAML. This is a one-way, opt-in
bridge: Vigilume is an MQTT *client* that publishes state and HA MQTT
**auto-discovery** config. It is **unrelated** to the old inbound Frigate→MQTT
event feed that was removed in the standalone rewrite — nothing here makes Vigilume
depend on a broker; if the broker is down Vigilume keeps running normally.

## Point Vigilume at your broker

1. In Home Assistant, make sure the **MQTT integration** is set up (Settings →
   Devices & Services → MQTT) and pointed at your broker (e.g. Mosquitto at
   `192.168.1.10`, user `homeassistant`).
2. In Vigilume: **Settings → System → Home Assistant / MQTT** (or `PUT /api/settings`):

   | Field | Example | Meaning |
   |-------|---------|---------|
   | `enabled` | `true` | Turn the publisher on |
   | `host` | `192.168.1.10` | Broker hostname/IP |
   | `port` | `1883` | Broker port |
   | `username` | `homeassistant` | Broker user (blank = anonymous) |
   | `password` | `••••` | Broker password |
   | `discovery_prefix` | `homeassistant` | HA discovery prefix (match your MQTT integration) |
   | `base_topic` | `sentinel` | Root topic for all Vigilume state/command topics |

3. Save. The publisher connects immediately (no app restart) and HA discovers the
   entities within a few seconds. **Changing any MQTT field restarts the publisher**
   live. Use **Test connection** (`POST /api/integrations/mqtt/test`) to verify the
   broker/credentials before or after saving — it returns `{ok, detail}`
   (connected / authentication failed / unreachable).

## What appears in Home Assistant

Vigilume registers **one HA device per camera**, all grouped under a **"Vigilume NVR"**
bridge device (`via_device`). Per camera you get:

- **`binary_sensor` per tracked label** (from the camera's detect-objects list) — ON
  while that label is actively in frame (from the engine's live count / event
  new‥end), OFF once it clears. `device_class` is **`occupancy`** for person/cat/dog
  and **`motion`** for everything else.
- **`binary_sensor` "Connectivity"** (`device_class: connectivity`) — driven by
  Vigilume's camera reachability probe (the same `camera_status` the UI uses).
- **`sensor` "Last event"** — state is the last label; JSON attributes carry
  `count`, `score`, `started`, `ended`, and `snapshot_url`.
- **`image` "Last snapshot"** (only when a **Public URL** is configured in Settings →
  System) — HA fetches the annotated event snapshot from `snapshot_url`. We publish a
  **URL** (`url_topic`), **not** raw image bytes, so nothing large ever crosses the
  broker.

### Two-way controls (capability-gated)

For cameras whose capabilities include them, Vigilume also exposes controls that call
the **same** device paths the Vigilume UI uses:

- **IR night vision** — a `switch` (ON/OFF) — cameras with `ir`.
- **Spotlight** — a `switch` (ON/OFF) — cameras with `white_light` (the EW turrets).
- **Siren** — a `button` — cameras with `siren` (the AD410 doorbell; plays the
  generated alarm tone via `audio.cgi`).

Switch state is echoed back after a successful command. Device control is best-effort:
if a camera is unreachable the command is logged and dropped — it never affects the
MQTT connection.

## Topics & availability

All topics live under your `base_topic` (default `sentinel`):

```
sentinel/status                       availability (retained; "online" on connect,
                                      "offline" via Last-Will on disconnect/crash)
sentinel/<cam>/<label>/state          per-label binary_sensor  (ON/OFF, retained)
sentinel/<cam>/connectivity/state     camera reachability      (ON/OFF, retained)
sentinel/<cam>/last_event/state       last event label         (retained)
sentinel/<cam>/last_event/attributes  last event JSON attrs    (retained)
sentinel/<cam>/image/url              annotated snapshot URL   (retained)
sentinel/<cam>/{ir,spotlight}/state   two-way switch state echo (retained)
sentinel/<cam>/{ir,spotlight}/set     two-way switch command   (subscribed)
sentinel/<cam>/siren/set              two-way siren button      (subscribed)
```

Discovery config is published (retained) under
`<discovery_prefix>/<component>/<base_topic>_<cam>_<slug>/config` with **stable
`unique_id`s**, so HA keeps the same entities across restarts. State topics are
retained, so HA restores the last known state after an HA or broker restart. The
Last-Will means HA marks all Vigilume entities *unavailable* if Vigilume crashes or
loses the broker.

## Notes & limits

- **Resilience:** a down/unreachable broker never crashes Vigilume — the publisher
  logs and reconnects with capped backoff. Detection, recording, and the UI are
  unaffected.
- **Camera add/remove:** new or deleted cameras appear in HA after the next publisher
  reconnect (e.g. after saving MQTT settings, or an HA/broker restart).
- **Snapshots in HA** require **Settings → System → Public URL** to be set and
  reachable from your HA host (the snapshot URL carries a short-lived media token).
- **`aiomqtt`** is imported lazily; the integration is only active when enabled, and
  the rest of Vigilume runs fine without it.

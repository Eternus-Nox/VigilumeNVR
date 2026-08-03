# Amcrest camera onboarding

How to prepare each of the three supported cameras before adding them to Vigilume.
Menu names below are from current Amcrest firmware and can vary slightly between versions.

Supported models and what Vigilume uses on each (from the
[capability map](CONTRACTS.md#per-model-capability-map-backend-amcrestfeaturespy)):

| Capability | IP5M-T1277EW-AI (5MP turret) | IP8M-2779EW-AI (4K turret) | AD410 (doorbell) |
|---|---|---|---|
| IR night vision | yes | yes | yes |
| White light LED (spotlight) | yes | yes | no (status LED only) |
| Siren | no | no | yes (call tone / tamper) |
| Microphone (audio in RTSP) | yes | yes | yes |
| Speaker (two-way talk) | no | no | yes |
| Doorbell button | no | no | yes |
| On-camera AI | yes (unused) | yes (unused) | yes (unused) |

All detection runs in Vigilume's own engine on the GPU (D-FINE ONNX); the cameras' own AI
stays unused so behavior is consistent across models. The backend also probes each device
at registration and merges what it finds over this table, so newer models degrade
gracefully.

## 1. Give every camera a fixed IP

Vigilume addresses cameras by IP, so each one needs an address that never changes. Pick one:

- **DHCP reservation (recommended):** in your router, reserve the camera's MAC address to a
  fixed IP. Nothing to configure on the camera; survives factory resets.
- **Static IP on the camera:** turrets: web UI → **Setup → Network → TCP/IP**, switch from
  DHCP to Static, set an address outside your DHCP pool. AD410: has no local web UI for
  this — use a DHCP reservation.

Whatever you choose, the IPs must match what you enter in `.env` (first boot) or
**Settings → Cameras** in the app.

To find a new turret on the network, use the router's client list or Amcrest's "IP Config"
tool; turrets default to DHCP out of the box.

## 2. Create a dedicated user (turrets)

Don't hand Vigilume the built-in `admin` account. On each turret:

1. Web UI → **Setup → System → Account** (some firmware: **Settings → System → Account → User**).
2. Add a user, e.g. `vigilume`, in the **admin** group, with a strong password.
3. Use this account in Vigilume.

Why the admin group: Vigilume uses the same credentials for the RTSP streams *and* for
device control (IR mode, reboot, OSD, motion-detect toggles) over the Amcrest HTTP API —
config-write and reboot calls need admin-group rights. If you'd rather use a
least-privilege `user`-group account (Live + Playback only), streaming and recording will
work but the camera controls in the app will fail with permission errors.

**Password tip:** stick to letters and digits. Vigilume percent-encodes credentials into
the RTSP URLs it generates, but exotic characters remain a common source of "camera won't
connect" reports with third-party firmware.

The AD410 has no user management — it uses a single device password (see below).

## 3. Stream settings (both turrets)

RTSP is enabled by default on port 554 (verify under **Setup → Network → Port**, and that
no "RTSP disable" toggle got flipped). Vigilume consumes the standard Amcrest URLs
(all three models, including the AD410 on current firmware):

```
Main: rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=0
Sub:  rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=1
```

These are derived automatically from the camera's IP + credentials; per-camera
`main_url`/`sub_url` overrides exist in the camera edit form for non-standard setups.

**Concurrent RTSP clients:** Amcrest firmware limits simultaneous RTSP sessions per
stream. Vigilume opens exactly **one** session per stream — its go2rtc restream fans out
to live view, detection and recording — so avoid pointing extra NVR software at the
cameras at the same time.

Vigilume records the **main stream** and runs detection on the **sub stream**. Configure
both under **Setup → Camera → Video → Encode** (or **Settings → Camera → Encode**):

**Main stream** (recording):
- Compression: **H.264 or H.265 — both work.** H.265/HEVC roughly **halves disk usage**
  but browsers can't decode HEVC directly, so Vigilume **automatically transcodes HEVC to
  H.264 for browser playback** (timeline segments on the fly + cached; event clips once at
  extraction) using the GPU (NVENC) with a CPU fallback — the recordings stay HEVC on disk,
  keeping the storage saving. See
  [recordings.md → automatic H.264 transcoding](recordings.md#browser-playback-automatic-h264-transcoding-for-hevc-cameras).
  Pick **H.265** if you're storage-constrained (it costs a little GPU/CPU at review time);
  pick **H.264** if you'd rather the browser play segments untouched with zero transcode
  overhead. Either is a valid choice.
- Resolution/FPS: native is fine — 2592×1944 @ 15–20 fps (IP5M), 3840×2160 @ 15 fps (IP8M).
- Bitrate: VBR, ~4096 kbps (IP5M) / ~6144–8192 kbps (IP8M). This directly drives disk
  usage — see the [storage math](faq.md#how-much-storage-do-i-need).
- **Audio:** tick "Audio Enable" on the main stream if you want audio in recordings and
  live view. The turrets output G.711; Vigilume's go2rtc transcodes it to AAC
  automatically (the generated config handles it).

**Sub stream** (detection):
- Compression: H.264.
- Resolution: **1280×720 if your firmware offers it, otherwise 704×480 (D1)**. Detection
  does not benefit from more; the backend matches its detect dimensions to whatever the
  substream is set to.
- FPS: 5–10 is plenty (Vigilume samples detection at 5 fps by default; adjustable
  per camera in the edit form).

**Overlays (recommended: off):** **Setup → Camera → Overlay** — disable the time and
channel-title overlays; event timestamps come from Vigilume anyway. Skip this if you want
timestamps burned into the 24/7 recordings themselves.

**Verify** from any machine on the LAN before adding the camera:

```bash
ffprobe "rtsp://vigilume:PASS@192.168.1.101:554/cam/realmonitor?channel=1&subtype=1"
# or open the URL in VLC (Media → Open Network Stream)
```

## 4. Model notes

### IP5M-T1277EW-AI (5MP AI turret)

- PoE only. Built-in mic, dual illumination: IR plus a white-LED spotlight (the "EW" in
  the model name). **No speaker, no siren** — those controls simply don't appear for
  this camera in Vigilume.
- The white LEDs show up as the **Spotlight** card on the camera page: Off / On / Auto,
  with a brightness slider when forced on (Vigilume drives the camera's `Lighting_V2`
  white-light config directly).
- In the app you also get: IR mode (auto/on/off), flip/mirror, OSD name, motion-detect
  toggle, privacy mode, reboot, plus live view/recordings with audio.

### IP8M-2779EW-AI (4K NightColor turret)

- PoE only. Built-in mic, dual illumination: IR plus warm-white "NightColor" LEDs.
- The white NightColor LEDs double as an on-demand spotlight: the **Spotlight** card on
  the camera page drives them directly (Off / On / Auto + brightness) via the camera's
  `Lighting_V2` white-light config. **Auto** hands control back to the camera's own
  illumination logic (normal NightColor night-vision behavior); **On** forces the light
  regardless of ambient level — handy as a deterrent or to light a scene in color.
- No speaker, no siren.

### AD410 (2K video doorbell)

**Onboarding is different:** the AD410 has no local web UI for setup. You must onboard it
with the **Amcrest Smart Home** app (iOS/Android):

1. Set up the doorbell in the Amcrest Smart Home app (Wi-Fi join, firmware update).
2. **Update the firmware in the app before anything else.** RTSP on the AD410 requires
   current firmware; old builds had RTSP/ONVIF stream bugs.
3. The **device password** you set in the app is the RTSP/API password. Username is
   `admin`. There are no additional user accounts on this model.
4. Give it a DHCP reservation (step 1) and note the IP.

Streams: same Amcrest RTSP URLs as the turrets. Main stream is 2560×1920; the substream
(~720×576) is what Vigilume uses for detection. Stream settings are managed from the
Amcrest Smart Home app, not a web UI.

**Chime options** (configured in the Amcrest Smart Home app, not in Vigilume):
- Existing **mechanical or digital chime**: wire the included chime kit into the chime,
  then in the app: **Settings → Device Information → Amcrest Chimes → Link
  digital/Mechanical Chime** and pick the right type.
- Standalone (no chime): works fine; you'll rely on Vigilume's push + the phone app.
- Amcrest's AD1-CHIME wireless chime pairs with the AD410 but has been discontinued.

**What Vigilume does with it:**
- Button presses: the backend listens to the doorbell's event stream directly and sends a
  "Doorbell pressed at Front Door" push with a snapshot immediately, regardless of your
  notification label filters.
- Siren/call tone: exposed as the capability-gated **Siren** button on the camera page.
- Mic: audio in recordings and live view like the other cameras.
- Two-way talk: press-and-hold **Hold to talk** on the camera page streams your mic to
  the doorbell speaker (browser mic → backend → the doorbell's HTTP audio backchannel).

**Talk notes:**
- **HTTPS required.** Browsers only expose the microphone in a secure context, so the
  talk button is disabled over plain `http://` — serve Vigilume over HTTPS as described
  in [remote access & HTTPS](remote-access.md).
- **Expected latency:** roughly 0.5–2 seconds end-to-end (8 kHz mono audio buffered by
  the browser, the backend, and the doorbell itself). Treat it walkie-talkie style —
  hold, speak, release, listen via the live view's audio — rather than as a phone call.
- One talker at a time per camera; a second device gets a "busy" notice.
- The Amcrest Smart Home app remains a fine alternative for full phone-style calls;
  Vigilume keeps recording, detecting, and notifying alongside it either way.

## 5. Add the cameras to Vigilume

- First boot: fill `CAM1..3_*` in `.env` before `docker compose up` — the backend seeds
  its database from these once.
- Any time after: **Settings → Cameras → Add camera** (name, friendly name, model, IP,
  username, password). The backend regenerates the go2rtc streams and starts
  detection/recording for the camera; it appears on the dashboard within ~30 seconds.

If a camera shows offline: check the IP responds (`ping`), the RTSP URL plays in
VLC/ffprobe with the same credentials, and that the password is URL-safe (step 2).

See also: [FAQ — adding a 4th camera](faq.md#how-do-i-add-a-fourth-camera),
[remote access & HTTPS](remote-access.md).

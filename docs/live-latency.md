# Live view latency

Live tiles negotiate one of four protocols, in this order:

```
webrtc → mse → hls → mjpeg
```

The player (`LivePlayer.tsx` → go2rtc's `video-rtc` component) offers them in
that order over the same `/go2rtc/api/ws` signaling socket and uses the first
one that connects. Which one you land on decides whether live view is
sub-second or several seconds behind.

## Protocol map

| Protocol | Latency | How it works | Needs |
|----------|---------|--------------|-------|
| **WebRTC** | **< 1 s** | Peer-to-peer media over UDP/TCP `:8555`; the browser connects straight to go2rtc's media port | An ICE **candidate the browser can reach** — a `host:8555` address that resolves to this server. Audio can be the camera's native G.711 (WebRTC carries it directly) |
| **MSE** | ~2–5 s | Fragmented MP4 muxed by go2rtc and pushed over the WebSocket (same origin as the web UI, so it always works) | AAC audio (MSE/fMP4 can't carry G.711) — this is why the stream transcodes audio to AAC |
| **HLS** | ~5–15 s | Segmented MP4/TS playlist | AAC audio; last-resort compatibility |
| **MJPEG** | high, no audio | JPEG frames over HTTP | Nothing — always works, worst quality |

**MSE always works** because it rides the same WebSocket the UI is already
served over. So live view never *breaks* — but if WebRTC can't connect, every
tile silently drops to the multi-second MSE path. The whole point of the
candidate handling below is to keep you on WebRTC.

## Why WebRTC needs a candidate (and why the LAN case used to be slow)

go2rtc can only advertise ICE candidates it is *told about*. In Docker
Compose the go2rtc container publishes `8555:8555`, so from the LAN the browser
must reach **the host's LAN IP on 8555** — but go2rtc, inside a bridge network,
only sees its own container IP (a `172.x` Docker-bridge address) and cannot
discover the host LAN IP on its own. With no reachable candidate, ICE fails and
every tile falls back to MSE.

Historically you fixed this by hand-typing your server's IP under **Settings →
System → WebRTC addresses**. Now Vigilume derives it automatically.

### Zero-config candidate derivation

At every go2rtc config generation (`native/streams.py` → `webrtc_status`) the
candidate list is built as:

```
manual entries  +  one auto-derived host candidate  +  stun:8555
```

The auto-derived host candidate is found best-effort, in order of trust:

1. **`VIGILUME_WEBRTC_HOST` env** — an explicit override. May be an IPv4 **or a
   hostname** (go2rtc resolves hostname candidates at negotiation time). Set
   this in `.env`/compose when the server sits behind NAT or the auto-derivation
   below can't see the host LAN IP — the reliable choice for a bridge-networked
   deployment. Example: `VIGILUME_WEBRTC_HOST=192.168.1.10`.
2. **`PUBLIC_URL` when it is a bare IP literal** (e.g. `http://192.168.1.10:8443`).
   A *hostname* `PUBLIC_URL` (the Tailscale/mDNS case) is ignored here, because
   a public-URL hostname usually fronts a reverse proxy that does **not** expose
   `:8555`.
3. **Auto-derived default-route LAN IPv4** — read from a connect-less UDP socket
   (the kernel picks the egress interface's source IP; no packet is sent). Used
   only when it is a private LAN address and **not** a Docker-bridge address
   (`172.16.0.0/12`). Inside a bridge container this normally *is* the Docker IP,
   so it is correctly skipped — which is exactly why the env override exists.

`stun:8555` is always kept (helps external/NAT setups auto-detect a public
address), and your **manual** WebRTC addresses are always kept and take
priority — nothing you type is ever dropped.

Everything here is best-effort and never raises: if detection turns up nothing,
the config is simply `stun:8555` and live view uses MSE.

> **In a bridge container, set `VIGILUME_WEBRTC_HOST`.** Auto-derivation (step 3)
> sees the Docker IP and skips it, so on the stock Compose deployment the env
> override — or a manual entry in the UI — is what actually lands you on WebRTC.
> Host-network or bare-metal deployments get it from step 3 for free.

### The readiness warning

`GET /api/settings` returns a read-only `webrtc` block:

```json
"webrtc": {
  "ready": false,
  "detected_ip": "192.168.1.10",
  "source": "auto",
  "candidates": ["stun:8555"]
}
```

- `ready: false` means go2rtc has only STUN — live view will fall back to MSE
  on the LAN. **Settings → System → WebRTC addresses** shows a warning in that
  case.
- `detected_ip` is a best-effort guess at this server's own address. When
  present, the warning offers a one-click **"Use `<ip>:8555`"** button that
  fills it into the candidate list for you.
- `candidates` is the effective list go2rtc is actually running with, shown
  under the input so you can confirm what the browser will try.

## The startup tax (first-frame latency)

Two things gate *how fast the first frame paints* once a protocol connects:

### 1. The go2rtc stream sources

Each camera's main stream is:

```yaml
{cam}: [ <main rtsp>, "ffmpeg:{cam}#audio=aac" ]
```

The raw RTSP source is **first**, so go2rtc has the video track immediately; the
second `ffmpeg:{cam}#audio=aac` source only transcodes the camera's G.711 audio
to AAC, which **MSE, HLS and the MPEG-TS recorder** all require (WebRTC can use
the native G.711 directly and doesn't wait on it). Keeping the raw source first
is the correct low-latency ordering, and the AAC source must stay for recording
and MSE/HLS audio — so this structure is left as-is. The dominant first-frame
cost is not here; it's the camera keyframe interval below.

### 2. Camera I-frame (keyframe / GOP) interval — the big win

WebRTC and MSE can only **start** a stream on a keyframe (I-frame), and only
**recover** from a glitch on the next one. If the camera emits a keyframe every
2 seconds (a common default is a GOP of 2× the frame rate), the very first frame
can take up to that long to appear, and every reconnect stalls the same amount.

**Set the camera's I-frame interval equal to its frame rate** (a GOP of ~1
second). Streams then start and recover almost instantly, at a negligible
bitrate cost.

On Amcrest cameras (web UI):

> **Setup → Camera → Video → Encode → Main Stream** (and **Sub Stream**)
> → set **I Frame Interval** equal to the stream's **Frame Rate (FPS)**.

For example, at 15 FPS set the I Frame Interval to 15. Do it for **both** the
Main Stream (live/record) and the Sub Stream (detection/tiles). This is a
camera-side setting — it isn't managed by Vigilume — and it is the single
biggest improvement to how quickly live view starts and recovers.

## Quick checklist

- [ ] Live tiles feel multi-second? You're on MSE — check **Settings → System**
      for the WebRTC readiness warning.
- [ ] Bridge-networked (stock Compose)? Set `VIGILUME_WEBRTC_HOST` to the host
      LAN IP, or click **Use detected IP**, then save.
- [ ] Remote/Tailscale? Add the tailnet IP as a candidate too
      ([remote-access.md](remote-access.md)).
- [ ] Streams slow to start or recover? Set the camera **I Frame Interval = FPS**
      on both Main and Sub streams.

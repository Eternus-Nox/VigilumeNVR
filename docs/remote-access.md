# Remote access & HTTPS

Two things need HTTPS with a certificate your phone trusts:

1. Installing the PWA and receiving Web Push ([mobile-pwa.md](mobile-pwa.md)).
2. Viewing your cameras when you're away from home.

Everything in Vigilume runs behind one origin (nginx on port 8080: PWA, `/api`, live
video via `/go2rtc`), so exposing **one** HTTPS endpoint that proxies to port 8080 covers
the entire app — live view included.

## Do NOT port-forward

**Never forward router ports to the NVR — not 8080, not 8443, not 8554/8555.**

- NVRs and cameras are among the most attacked devices on the internet; exposed video
  stacks get scanned within minutes and have a long history of remote exploits.
- The go2rtc RTSP restream (8554) is **unauthenticated by design** — it's exposed on
  the assumption that the LAN is private (so VLC etc. can play streams).
- A single admin password on the web UI is not a sufficient barrier for a
  camera system reachable by the whole internet.

Both options below give you remote access with **zero open inbound ports**.

## Recommended: Tailscale

[Tailscale](https://tailscale.com) builds a private WireGuard network between your
devices. Your NVR gets a stable private hostname, reachable from your phone anywhere,
with a **publicly-trusted HTTPS certificate** — no certificate-trust fiddling on any
device, and nothing exposed to the internet. Free tier is plenty for this.

### Setup

On the NVR host:

```bash
# Install (see tailscale.com/download for distro specifics)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

In the [Tailscale admin console](https://login.tailscale.com/admin/dns): make sure
**MagicDNS** and **HTTPS Certificates** are enabled (DNS page). Your NVR is now e.g.
`nvr.tailnet-name.ts.net`.

On your phone: install the Tailscale app and log in to the same tailnet.

### Serve the app over HTTPS

```bash
sudo tailscale serve --bg --https=443 http://localhost:8080
```

This provisions a valid Let's Encrypt certificate for the machine's MagicDNS name and
proxies `https://nvr.tailnet-name.ts.net` → Vigilume, persistently (`--bg` survives
reboots). The service is reachable **only from devices on your tailnet** — it is not
public. Check with `tailscale serve status`.

Then set the public URL so notification links resolve:

```bash
# .env
PUBLIC_URL=https://nvr.tailnet-name.ts.net
```

(or **Settings → System → Public URL**), and install the PWA from that URL
([mobile-pwa.md](mobile-pwa.md)).

### Sub-second live view over the tailnet (WebRTC)

Live view works out of the box over Tailscale via MSE (a few seconds of
latency). For **sub-second WebRTC**, go2rtc needs an ICE candidate your phone
can reach: add the NVR's **tailnet IP** with port 8555 (e.g. `100.64.0.7:8555`)
under **Settings → System → WebRTC addresses** — add the LAN IP
(e.g. `192.168.1.10:8555`) there too for fast local playback. The browser
tries the candidates and picks whichever connects; if none is reachable, the
player falls back to MSE automatically, so live view never breaks — it just
isn't sub-second.

For the LAN, Vigilume now derives the local WebRTC candidate automatically (no
IP to type in), and Settings → System warns you when live view is stuck on the
slow MSE fallback. See [live-latency.md](live-latency.md) for the full protocol
map, the `VIGILUME_WEBRTC_HOST` override, and the camera-side keyframe fix that
makes streams start and recover instantly.

**Notes**
- Phone on the tailnet = access from anywhere (LTE, hotel Wi-Fi, …). Tailscale's iOS/
  Android apps are battery-friendly and can stay connected all the time.
- For household members, share the tailnet (or use node sharing) so their phones can
  reach the NVR too.

## LAN HTTPS: a secure context on your own network

The doorbell mic (`getUserMedia`), PWA install and Web Push only work in a
**secure context** — i.e. over HTTPS. Plain `http://<lan-ip>:8080` is not one,
so the talk button renders disabled there. The bundled Caddy profile fixes that
entirely on the LAN, with **no internet, no Cloudflare, no DNS**: Caddy is its
own certificate authority (`tls internal`).

```bash
# .env
COMPOSE_PROFILES=tls
TLS_HOSTNAME=192.168.1.253   # the box's LAN IP, or an mDNS name your LAN resolves
```

```bash
docker compose --profile tls up -d caddy
```

Caddy serves `https://192.168.1.253:8443` → the web container. The one cost of a
self-signed CA is trusting its root once per device.

### Trust the CA on your devices

Export Caddy's root CA once:

```bash
docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt ./vigilume-ca.crt
```

- **iOS:** get `vigilume-ca.crt` onto the phone (AirDrop/email/file share) → tap it →
  **Settings → Profile Downloaded → Install**. Then — easy to miss —
  **Settings → General → About → Certificate Trust Settings** and enable full trust for
  the certificate. Safari/Chrome will now treat the NVR as secure and the mic works.
- **Android:** **Settings → Security & privacy → More security → Encryption &
  credentials → Install a certificate → CA certificate** and pick the file.
- **Desktop browsers:** import into the OS/browser trust store, or accept the warning.

The CA root lives in the `caddy-data` volume, so it survives container recreation —
devices trust it once. Wiping that volume regenerates the CA and every device must
re-trust it.

**Note:** iOS trusts a *name* more smoothly than a bare IP. If the two-step trust
above is fiddly, set `TLS_HOSTNAME` to an mDNS name like `nvr.local` (that your LAN
resolves) and browse `https://nvr.local:8443` instead.

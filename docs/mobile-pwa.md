# Installing the mobile app (PWA)

Vigilume's phone app is a PWA — the web UI installed to your home screen. Once installed
it gets its own icon, runs full-screen, and receives push notifications.

**Prerequisite: HTTPS with a certificate your phone trusts.** Browsers only allow PWA
install + Web Push on secure origins. Plain `http://nvr:8080` will work as a website but
can't be installed properly or receive push. Set up HTTPS first —
[remote-access.md](remote-access.md) covers the two supported paths (Tailscale,
recommended; or LAN-only Caddy with a locally-trusted CA). Then make sure
`PUBLIC_URL` in `.env` (or **Settings → System → Public URL**) matches the HTTPS address
you'll use on the phone, so notification taps open the right URL.

## iOS (iPhone/iPad)

Requires **iOS/iPadOS 16.4 or newer**. Web Push on iOS only works for web apps that have
been added to the Home Screen — a plain Safari tab cannot receive push.

1. Use an HTTPS URL — the LAN Caddy profile
   ([instructions](remote-access.md#lan-https-a-secure-context-on-your-own-network)),
   the Cloudflare-tunnelled `https://` address, or Tailscale's `serve`. A plain
   `http://<lan-ip>` address is not a secure context, so PWA install and push
   will not offer themselves.
2. In **Safari**, open your HTTPS URL (e.g. `https://nvr.your-tailnet.ts.net`) and log in.
3. Tap **Share → Add to Home Screen → Add**. (It must be Safari; other iOS browsers can't
   install PWAs.)
4. Open Vigilume **from the new home-screen icon** — not the Safari tab.
5. Go to **Settings → Notifications** and enable push on this device. iOS shows the
   permission prompt; tap **Allow**.
6. Send a test from the same screen to confirm.

## Android

1. Open your HTTPS URL in **Chrome** and log in.
2. Chrome usually shows an **Install app** prompt; otherwise use **⋮ menu → Add to Home
   screen / Install app**.
3. Open the installed app, go to **Settings → Notifications**, enable push, and **Allow**
   the permission prompt.
4. Send a test notification.

## What notifications look like

- **Title:** `Person detected at Front Yard` (or `Doorbell pressed at Front Door`).
- **Body:** the current in-frame count, e.g. `2 in frame`.
- **Image:** the annotated snapshot — bounding boxes, labels, and a count banner like
  "2 people", drawn by the backend with Supervision.
- **Tap:** opens the event detail page with the recorded clip.

Which labels notify, the minimum score, and the per-camera cooldown are all under
**Settings → Notifications**. Doorbell presses bypass the label filter.

### iOS limitations

- The **snapshot image may not render** inside the iOS notification — iOS Web Push
  ignores the notification image on many versions and shows only the app icon and text.
  The tap-through always works: tap the notification and the event page opens with the
  annotated snapshot and clip.
- Push arrives only if the app was installed via **Add to Home Screen** and notifications
  were enabled from inside the installed app.
- iOS may silently expire a subscription if you repeatedly dismiss or never interact with
  notifications. If pushes stop, open the app and re-enable them in
  **Settings → Notifications**.

## Troubleshooting

| Symptom | Check |
|---|---|
| No "Add to Home Screen" / install option | You're on plain HTTP, or the cert isn't trusted. Fix HTTPS first ([remote-access.md](remote-access.md)) |
| Enable-push toggle does nothing on iOS | You're in a Safari tab, not the installed app; or iOS < 16.4 |
| Test push works, real events don't | Check **Settings → Notifications**: enabled labels, `min_score`, cooldown. Verify events appear in the Events page at all |
| Pushes stopped on iOS | Re-enable in Settings → Notifications (subscription may have expired) |
| Notification tap opens the wrong host | `PUBLIC_URL` / Settings → System → Public URL doesn't match the URL the phone uses |

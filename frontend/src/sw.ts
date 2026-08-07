/**
 * Vigilume NVR service worker (vite-plugin-pwa injectManifest).
 *
 * - Precaches the app shell from the injected build manifest (no workbox
 *   runtime — a versioned Cache API store keeps the bundle lean).
 * - Navigations: network-first, offline fallback to the cached shell.
 * - `push`: showNotification with annotated snapshot image; payload `data.url`
 *   is the click target (/events/{id}).
 * - `notificationclick`: focus an existing window and navigate it (postMessage
 *   to the SPA router), else openWindow(url).
 * - Never touches /api/ or /go2rtc/ requests.
 */
/// <reference lib="webworker" />

declare let self: ServiceWorkerGlobalScope & {
  __WB_MANIFEST: Array<{ url: string; revision: string | null } | string>;
};

export type {};

const manifest = self.__WB_MANIFEST || [];

const entries = manifest.map((e) =>
  typeof e === 'string' ? { url: e, revision: null } : e,
);

// Cache version derived from the manifest content: any changed asset hash
// produces a new cache and old ones are dropped on activate.
function hash(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}
const CACHE_PREFIX = 'vigilume-shell-';
// Pre-rename prefix (the app shipped as "Sentinel"). Kept ONLY so activation
// still deletes those caches — a renamed prefix alone would orphan them in the
// user's storage quota forever, since the sweep below matches by prefix.
const LEGACY_CACHE_PREFIX = 'sentinel-shell-';
const CACHE_NAME = CACHE_PREFIX + hash(JSON.stringify(entries));

const precacheUrls = entries.map((e) => new URL(e.url, self.location.origin).href);
const precacheSet = new Set(precacheUrls);

self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      // Fetch each entry individually so one miss doesn't fail the install.
      await Promise.allSettled(
        precacheUrls.map(async (url) => {
          const res = await fetch(url, { cache: 'no-cache' });
          if (res.ok) await cache.put(url, res);
        }),
      );
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((n) => (n.startsWith(CACHE_PREFIX) || n.startsWith(LEGACY_CACHE_PREFIX)) && n !== CACHE_NAME)
          .map((n) => caches.delete(n)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event: FetchEvent) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Live data, media and streams are never served from cache.
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/go2rtc/')) return;

  if (req.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          return await fetch(req);
        } catch {
          const cached =
            (await caches.match(new URL('index.html', self.location.origin).href)) ??
            (await caches.match(new URL('/', self.location.origin).href));
          return cached ?? Response.error();
        }
      })(),
    );
    return;
  }

  if (precacheSet.has(url.href)) {
    event.respondWith(
      (async () => {
        const cached = await caches.match(url.href);
        if (cached) return cached;
        const res = await fetch(req);
        if (res.ok) {
          const cache = await caches.open(CACHE_NAME);
          await cache.put(url.href, res.clone());
        }
        return res;
      })(),
    );
  }
});

// ---------- Web Push ----------

interface PushPayload {
  title?: string;
  body?: string;
  image?: string;
  icon?: string;
  badge?: string;
  tag?: string;
  url?: string;
  data?: { url?: string };
}

self.addEventListener('push', (event: PushEvent) => {
  let payload: PushPayload = {};
  try {
    payload = (event.data?.json() as PushPayload) ?? {};
  } catch {
    payload = { title: 'Vigilume NVR', body: event.data?.text() ?? '' };
  }
  const title = payload.title || 'Vigilume NVR';
  const url = payload.data?.url || payload.url || '/';
  const options: NotificationOptions & { image?: string } = {
    body: payload.body || '',
    icon: payload.icon || '/icons/icon-192.png',
    badge: payload.badge || '/icons/badge-96.png',
    tag: payload.tag,
    data: { url },
  };
  if (payload.image) options.image = payload.image;
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event: NotificationEvent) => {
  event.notification.close();
  const raw = (event.notification.data as { url?: string } | undefined)?.url || '/';
  event.waitUntil(
    (async () => {
      // Normalize: public_url click links may be absolute; navigate in-scope
      // when they point at this origin.
      let target = raw;
      try {
        const u = new URL(raw, self.location.origin);
        target = u.origin === self.location.origin ? u.pathname + u.search : u.href;
      } catch {
        target = '/';
      }
      const clientsList = await self.clients.matchAll({
        type: 'window',
        includeUncontrolled: true,
      });
      for (const client of clientsList) {
        try {
          await client.focus();
          client.postMessage({ type: 'navigate', url: target });
          return;
        } catch {
          // try the next client, or fall through to openWindow
        }
      }
      await self.clients.openWindow(target);
    })(),
  );
});

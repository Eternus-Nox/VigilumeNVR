/**
 * "You are looking at an old version of this page."
 *
 * Shown only when the server is demonstrably serving a different bundle from
 * the one this tab loaded — see lib/bundle.ts for why that comparison needs no
 * build-time plumbing, and why every uncertain case reports nothing at all.
 *
 * It offers a reload rather than performing one. A security console can be in
 * the middle of something (a clip playing, a face being reviewed, a form
 * half-filled) and taking that away to fetch a newer stylesheet is a worse
 * outcome than showing a slightly old UI for another minute.
 *
 * Dismissible, and the dismissal is remembered per bundle: dismissing it for
 * the build you are on must not silence it for the NEXT one, which is the
 * build you will actually want to hear about.
 */
import { useEffect, useState } from 'react';
import { checkBundle, runningBundleName } from '../lib/bundle';

const DISMISS_KEY = 'vigilume.staleBundleDismissed';

/** How often to re-check. Slow on purpose: this is a deploy, not a heartbeat. */
const CHECK_MS = 5 * 60 * 1000;

export default function StaleBundleBanner() {
  const [stale, setStale] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    const run = async () => {
      const state = await checkBundle(controller.signal);
      if (controller.signal.aborted) return;
      if (state === 'stale') {
        let dismissed = '';
        try {
          dismissed = window.localStorage.getItem(DISMISS_KEY) ?? '';
        } catch {
          // Private windows and blocked site data throw here. Treat it as
          // "not dismissed" — showing the banner twice is a smaller cost than
          // hiding it forever.
          dismissed = '';
        }
        setStale(dismissed !== runningBundleName());
      } else {
        setStale(false);
      }
      timer = window.setTimeout(() => void run(), CHECK_MS);
    };

    // Not on mount-tick: a hard reload has just fetched index.html, and asking
    // again in the same instant mostly measures the cache we just bypassed.
    timer = window.setTimeout(() => void run(), 4000);
    return () => {
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  if (!stale) return null;

  return (
    <div className="stale-bundle" role="status">
      <span>
        <strong>A newer version of this page is available.</strong> You are running an
        older build — reload to pick it up.
      </span>
      <span className="stale-bundle-actions">
        <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>
          Reload
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            try {
              // Keyed by the build being dismissed, so the NEXT deploy asks
              // again rather than inheriting this dismissal.
              window.localStorage.setItem(DISMISS_KEY, runningBundleName());
            } catch {
              // Nothing to do — it simply reappears on the next check.
            }
            setStale(false);
          }}
        >
          Not now
        </button>
      </span>
    </div>
  );
}

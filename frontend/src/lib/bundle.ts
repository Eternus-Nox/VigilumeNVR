/**
 * Is this tab running the bundle the server is currently serving?
 *
 * THE PROBLEM THIS SOLVES
 * =======================
 * The web app and the backend are separate containers, deployed by rsync +
 * `docker compose up -d --build`. Three things go wrong often enough to have
 * cost real time on this project, and all three look identical from the
 * browser — the UI simply behaves as though a change was never made:
 *
 *   1. the web image was rebuilt but the browser is holding a cached bundle;
 *   2. the web image was NOT rebuilt, so nginx is still serving an old one;
 *   3. the backend was rebuilt and the web image was not (or the reverse).
 *
 * NO BUILD PLUMBING. It would be tidier to bake a git SHA into both images and
 * compare them, but the documented deploy is a plain `docker compose up
 * --build` with no build args, so that field would be empty in exactly the
 * common case — a check that is blank when you need it is worse than none.
 *
 * Instead this compares what the running tab loaded against what the server
 * hands out RIGHT NOW: fetch `/index.html` with cache disabled, read the
 * hashed entry-script filename out of it, and compare with the hashed filename
 * this module was itself loaded from (`import.meta.url`). Vite content-hashes
 * those filenames, so a difference means the server has a build this tab is
 * not running — which covers (1) and (2) with no cooperation from the build at
 * all.
 *
 * (3) is answered separately, by the backend reporting its own version and the
 * schema version the DATABASE is actually on — see the System tab.
 *
 * Deliberately quiet on failure. A fetch that errors, an index.html that does
 * not parse, a dev server with unhashed names: all report "no opinion" rather
 * than nagging. This must never cry wolf, or it will be ignored on the day it
 * is right.
 */

/** A hashed asset name, e.g. "index-a1b2c3d4.js". Vite content-hashes these,
 *  which is the entire basis of the comparison; a dev server serves unhashed
 *  module paths, where this check means nothing and must stay silent. */
const HASHED = /-[A-Za-z0-9_-]{6,}\.js$/;

function hashedName(src: string): string | null {
  const name = src.split('?')[0].split('/').pop() ?? '';
  return HASHED.test(name) ? name : null;
}

/** The hashed entry script THIS TAB loaded.
 *
 * Read from the live document's own <script type="module"> rather than from
 * `import.meta.url`. That distinction matters: import.meta.url names whichever
 * CHUNK this module was bundled into, and Vite is free to split it into a
 * shared chunk the moment an import graph changes — at which point it would
 * never equal the entry script in index.html and the banner would claim every
 * build was stale, permanently. The document's script tag is by definition the
 * entry this tab is running, whatever the chunking does.
 */
function runningBundle(): string | null {
  try {
    const el = document.querySelector<HTMLScriptElement>('script[type="module"][src]');
    return el ? hashedName(el.getAttribute('src') ?? '') : null;
  } catch {
    return null;
  }
}

/** The hashed entry script the SERVER is handing out right now. */
async function servedBundle(signal?: AbortSignal): Promise<string | null> {
  try {
    const res = await fetch('/index.html', { cache: 'no-store', signal });
    if (!res.ok) return null;
    const html = await res.text();
    // Vite emits <script type="module" crossorigin src="/assets/index-HASH.js">
    const match = html.match(/<script[^>]+src="([^"]+\.js)"/);
    return match ? hashedName(match[1]) : null;
  } catch {
    return null;
  }
}

export type BundleState = 'unknown' | 'current' | 'stale';

/**
 * Compare the two. "unknown" whenever either side cannot be established —
 * see the note above about never crying wolf.
 */
export async function checkBundle(signal?: AbortSignal): Promise<BundleState> {
  const running = runningBundle();
  if (!running) return 'unknown';
  const served = await servedBundle(signal);
  if (!served) return 'unknown';
  return served === running ? 'current' : 'stale';
}

/** For the System tab, which shows the name rather than just the verdict. */
export function runningBundleName(): string {
  return runningBundle() ?? 'dev build';
}

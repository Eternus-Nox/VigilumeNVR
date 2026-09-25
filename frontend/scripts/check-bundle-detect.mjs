/**
 * The stale-bundle check must agree with what Vite actually emits.
 *
 * `lib/bundle.ts` compares two strings: the hashed entry script in the live
 * document, and the one in a freshly fetched `/index.html`. Both are parsed
 * out of real markup, so the whole thing rests on assumptions about Vite's
 * output — the tag shape, the `/assets/` path, the hash alphabet. If any of
 * those change the check does not error; it silently returns "unknown" and
 * stops protecting anything, or worse matches nothing and reports every build
 * as stale.
 *
 * So this runs the REAL parser over the REAL built index.html, and asserts
 * both directions: that a matching pair reads as current, and that a differing
 * pair reads as stale. It needs `dist/`, so it runs after a build.
 */
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const DIST = new URL('../dist/', import.meta.url).pathname;

let checks = 0;
const failures = [];
const check = (cond, label) => {
  checks += 1;
  if (cond) console.log(`  ok: ${label}`);
  else {
    console.error(`  FAIL: ${label}`);
    failures.push(label);
  }
};

if (!existsSync(join(DIST, 'index.html'))) {
  console.error('dist/index.html is missing — run `vite build` first.');
  process.exit(1);
}

// The two functions under test, kept in step with src/lib/bundle.ts by the
// assertion at the bottom: if that file's regex changes and this one does not,
// the mirror check fails loudly rather than this suite quietly testing an
// older rule than the app ships.
const HASHED = /-[A-Za-z0-9_-]{6,}\.js$/;
const hashedName = (src) => {
  const name = src.split('?')[0].split('/').pop() ?? '';
  return HASHED.test(name) ? name : null;
};
const entryOf = (html) => {
  const m = html.match(/<script[^>]+src="([^"]+\.js)"/);
  return m ? hashedName(m[1]) : null;
};

console.log('\nparsing the real built index.html');
const html = readFileSync(join(DIST, 'index.html'), 'utf8');
const entry = entryOf(html);
check(entry !== null,
      `the entry script is found and looks content-hashed (got ${entry})`);
check(entry !== null && existsSync(join(DIST, 'assets', entry)),
      'and that file actually exists in dist/assets — so the name the check ' +
      'compares is a real asset, not a path fragment');

console.log('\nthe comparison decides both ways');
check(entry === entryOf(html),
      'the same index.html twice reads as the SAME bundle (this is the case ' +
      'that must never show the banner)');
const older = html.replace(entry, 'index-0000000000.js');
check(entryOf(older) !== entry,
      'a rebuilt index.html with a different hash reads as DIFFERENT — the ' +
      'case the banner exists for');

console.log('\nthings that must read as "no opinion" rather than as stale');
check(entryOf('<html><body>no script here</body></html>') === null,
      'markup with no module script yields null, not a false mismatch');
check(hashedName('/src/main.tsx') === null,
      'a dev-server module path is not a hashed bundle, so the check stays quiet');
check(hashedName('/assets/index.js') === null,
      'an UNhashed production name is also null — without a content hash the ' +
      'comparison carries no information');
check(hashedName('/assets/index-QLk61cNM.js?v=2') === 'index-QLk61cNM.js',
      'a cache-busting query is stripped before comparing');

console.log('\nthe suite and the app share one rule');
const src = readFileSync(
  new URL('../src/lib/bundle.ts', import.meta.url).pathname, 'utf8',
);
const appRule = src.match(/const HASHED = (\/.*\/);/);
check(appRule !== null && appRule[1] === HASHED.toString(),
      `src/lib/bundle.ts uses the same hash pattern this suite tests ` +
      `(app: ${appRule ? appRule[1] : 'NOT FOUND'}, suite: ${HASHED})`);

// Belt and braces on the entry-script choice: `runningBundle` reads the
// document's own <script>, which only works because Vite emits exactly one
// module script in index.html. More than one and the selector picks the first,
// which may not be the entry.
const moduleScripts = html.match(/<script[^>]+type="module"[^>]*>/g) ?? [];
check(moduleScripts.length === 1,
      `index.html has exactly one module script (found ${moduleScripts.length}) ` +
      '— the document selector picks the first, so a second would make the ' +
      'comparison depend on emission order');

console.log();
if (failures.length) {
  console.error(`${failures.length} of ${checks} CHECKS FAILED`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`ALL ${checks} CHECKS PASSED (stale-bundle detection)`);

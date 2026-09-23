/**
 * The settings search index must point at real places.
 *
 * `searchIndex.ts` is hand-written — that is deliberate, because it carries the
 * words people actually type ("parked car", "too many alerts") rather than the
 * headings on screen. The cost of hand-writing it is drift: a tab renamed, a
 * card retitled, and an entry quietly becomes a result that lands nowhere.
 *
 * Nothing else catches that. The app compiles, the search still returns the
 * row, clicking it still switches tab — it just fails to scroll, which reads as
 * "the search is a bit broken" rather than as a bug anyone files.
 *
 * So this asserts three things:
 *   1. every `tab` is a real tab id in Settings.tsx;
 *   2. every `card` matches a heading that is actually rendered somewhere;
 *   3. no two entries point at the same destination with different labels,
 *      which produces two results that do the same thing.
 *
 * Run by `npm run check:settings` (and by `npm run build`).
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const SRC = new URL('../src/', import.meta.url).pathname;

const read = (p) => readFileSync(join(SRC, p), 'utf8');
const listTsx = (dir) =>
  readdirSync(join(SRC, dir))
    .filter((f) => f.endsWith('.tsx'))
    .map((f) => join(dir, f));

const shell = read('pages/Settings.tsx');
const tabs = new Set([...shell.matchAll(/\{ id: '([a-z]+)'/g)].map((m) => m[1]));

const index = read('pages/settings/searchIndex.ts');
const entries = [
  ...index.matchAll(/tab: '([a-z]+)', card: (?:'([^']*)'|"([^"]*)")/g),
].map((m) => ({ tab: m[1], card: m[2] ?? m[3] }));

// Every heading style settings cards actually use. All three are matched by
// SettingsSearch.revealCard, so all three are valid destinations here.
const headings = new Set();
for (const file of [...listTsx('pages/settings'), ...listTsx('components')]) {
  const text = read(file);
  for (const m of text.matchAll(/<h2>([^<{]+)<\/h2>/g)) {
    headings.add(m[1].trim().replace(/&amp;/g, '&'));
  }
  for (const m of text.matchAll(/title="([^"]+)"/g)) headings.add(m[1].trim());
  for (const m of text.matchAll(/-title">([^<]+)<\/span>/g)) headings.add(m[1].trim());
}

const problems = [];
if (entries.length === 0) {
  problems.push('parsed ZERO entries out of searchIndex.ts — this check has ' +
                'silently stopped checking anything, which is worse than a ' +
                'stale index');
}
for (const { tab, card } of entries) {
  if (!tabs.has(tab)) problems.push(`tab "${tab}" is not a settings tab (card "${card}")`);
  if (!headings.has(card)) {
    problems.push(`card "${card}" (tab "${tab}") matches no rendered heading — ` +
                  'the search will switch tab and then fail to scroll');
  }
}
const seen = new Map();
for (const { tab, card } of entries) {
  const key = `${tab} › ${card}`;
  seen.set(key, (seen.get(key) ?? 0) + 1);
}
for (const [key, n] of seen) {
  if (n > 1) problems.push(`${n} entries point at "${key}" — duplicate results`);
}

if (problems.length) {
  console.error('settings search index is out of date:\n');
  for (const p of problems) console.error(`  - ${p}`);
  console.error(`\n${problems.length} problem(s). Fix src/pages/settings/searchIndex.ts.`);
  process.exit(1);
}
console.log(`settings search index OK — ${entries.length} entries, all resolvable`);

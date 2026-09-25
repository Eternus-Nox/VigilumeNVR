/**
 * Event grouping must never lose a row, reorder history, or merge two visits.
 *
 * The grid collapses near-simultaneous detections on one camera into a single
 * row, because the server keys events `(camera, label)` — a person and the car
 * they arrived in are genuinely two events a second apart. That is a
 * PRESENTATION change over evidence, so the properties that matter are
 * conservation ones, and they are what this asserts:
 *
 *   - every input event appears in exactly one group (nothing hidden, nothing
 *     duplicated);
 *   - order is preserved, so turning grouping on never reshuffles history;
 *   - two cameras firing at once stay separate — the same person on two
 *     cameras is evidence about where they went;
 *   - a slow trickle cannot chain into one enormous group, because membership
 *     is measured against the group's START, not the previous member.
 *
 * Run by `npm run check:groups` (and by `npm run build`).
 */
import { readFileSync } from 'node:fs';

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

// The module is TypeScript; strip the types rather than adding a build step for
// one pure function. Only the two exports are needed.
const src = readFileSync(
  new URL('../src/lib/groupEvents.ts', import.meta.url).pathname,
  'utf8',
);
const js = src
  .replace(/^import[^;]+;$/gm, '')
  .replace(/^export interface [\s\S]*?^}$/gm, '')
  .replace(/: EventGroup\[\]/g, '')
  .replace(/: NvrEvent\[\]/g, '')
  .replace(/: NvrEvent/g, '')
  .replace(/: string\[\]/g, '')
  .replace(/: string/g, '')
  .replace(/: EventGroup/g, '')
  .replace(/new Map<[^>]+>\(\)/g, 'new Map()')
  .replace(/^export const/gm, 'const')
  .replace(/^export function/gm, 'function');
const mod = new Function(`${js}; return { groupEvents, GROUP_WINDOW_S };`)();
const { groupEvents, GROUP_WINDOW_S } = mod;

const ev = (id, camera, start, label = 'person', media = {}) => ({
  id, camera, start_time: start, label, labels: [label],
  has_clip: false, has_snapshot: false, ...media,
});

console.log('\nnothing is lost and nothing is reordered');
const page = [
  ev(1, 'drive', 1000, 'person'),
  ev(2, 'drive', 1002, 'car'),
  ev(3, 'porch', 1001, 'person'),
  ev(4, 'drive', 900, 'person'),
];
const groups = groupEvents(page);
const flat = groups.flatMap((g) => g.events.map((e) => e.id));
check(flat.length === page.length, `every event survives (${flat.length}/${page.length})`);
check(new Set(flat).size === flat.length, 'and none is duplicated');
check(
  JSON.stringify(groups.map((g) => g.lead.id)) === JSON.stringify([1, 3, 4]),
  `leads appear in the original order (got ${groups.map((g) => g.lead.id)})`,
);

console.log('\nwhat merges, and what must not');
check(groups[0].events.map((e) => e.id).join() === '1,2',
      'a person and a car two seconds apart on ONE camera group together');
check(groups[0].labels.join() === 'person,car',
      'and the group carries both labels for its chip');
check(groups.some((g) => g.lead.id === 3 && g.events.length === 1),
      'a different CAMERA at the same instant stays separate — the same person ' +
      'on two cameras is evidence about where they went');
check(groups.some((g) => g.lead.id === 4 && g.events.length === 1),
      'and the same camera 100 s earlier is a different visit');

console.log('\na trickle cannot chain into one enormous group');
const trickle = [];
for (let i = 0; i < 12; i += 1) {
  // Each one is within the window of the PREVIOUS, but not of the first.
  trickle.push(ev(100 + i, 'drive', 1000 - i * (GROUP_WINDOW_S - 1)));
}
const chained = groupEvents(trickle);
check(chained.length > 1,
      `a steady drip spanning ${11 * (GROUP_WINDOW_S - 1)}s becomes ` +
      `${chained.length} groups, not one — membership is measured against the ` +
      "group's start, not the previous member");
const widest = Math.max(
  ...chained.map((g) =>
    Math.abs(g.events[0].start_time - g.events[g.events.length - 1].start_time),
  ),
);
check(widest <= GROUP_WINDOW_S,
      `no group spans more than the window (widest ${widest}s)`);

console.log('\nthe row shows the member that was actually recorded');
const mixed = groupEvents([
  ev(1, 'drive', 1000, 'person'),
  ev(2, 'drive', 999, 'car', { has_snapshot: true }),
  ev(3, 'drive', 998, 'dog', { has_clip: true, has_snapshot: true }),
  ev(4, 'porch', 990, 'person'),
]);
check(mixed.length === 2 && mixed[0].lead.id === 3,
      `a sibling with a clip leads the row over a newer one without (got ${mixed[0].lead.id})`);
check(mixed[0].events.map((e) => e.id).join() === '1,2,3',
      'without reordering the members');
check(mixed[0].labels.join() === 'person,car,dog',
      'and the labels still read in page order');
check(mixed[1].lead.id === 4,
      'the next group keeps its place in the list');
const snapOnly = groupEvents([
  ev(1, 'drive', 1000),
  ev(2, 'drive', 999, 'car', { has_snapshot: true }),
]);
check(snapOnly[0].lead.id === 2, 'with no clip anywhere, a snapshot beats nothing');
const late = groupEvents([
  ev(1, 'drive', 1000),
  ev(2, 'drive', 1000 - GROUP_WINDOW_S, 'car', { has_clip: true }),
  ev(3, 'drive', 1000 - GROUP_WINDOW_S - 1, 'car'),
]);
check(late.length === 2,
      "re-picking the lead doesn't move the window: it is still measured from " +
      'the first member, so a clip at the far edge cannot stretch the group');

console.log('\nedges');
check(groupEvents([]).length === 0, 'an empty page yields no groups');
const one = groupEvents([ev(9, 'drive', 1000)]);
check(one.length === 1 && one[0].events.length === 1,
      'a single event is a group of one, which the UI renders exactly as before');
const exact = groupEvents([ev(1, 'd', 1000), ev(2, 'd', 1000 - GROUP_WINDOW_S)]);
check(exact.length === 1,
      'an event exactly at the window boundary is included (<=, not <)');
const past = groupEvents([ev(1, 'd', 1000), ev(2, 'd', 1000 - GROUP_WINDOW_S - 0.1)]);
check(past.length === 2, 'and one just past it is not');

console.log();
if (failures.length) {
  console.error(`${failures.length} of ${checks} CHECKS FAILED`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`ALL ${checks} CHECKS PASSED (event grouping)`);

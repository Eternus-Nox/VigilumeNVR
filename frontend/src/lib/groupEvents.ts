/**
 * Collapse near-simultaneous detections on one camera into a single row.
 *
 * WHY THE LIST LOOKS CLUTTERED
 * ============================
 * Events are keyed `(camera, label)` on the server — one open event per object
 * type per camera. So a person walking up a drive with a car behind them is
 * genuinely TWO events, opened within a second of each other, and the history
 * shows two rows for one thing that happened. Enabling plate recognition makes
 * this more common, because it adds `car` to cameras that previously detected
 * only `person`.
 *
 * That server-side model is right and is not changed here: each event keeps its
 * own snapshot, its own clip, its own box and its own recognitions, and losing
 * any of that to a merge would be a real cost on a system whose job is
 * evidence. This is PRESENTATION ONLY — the rows are still there, one click
 * away, and the grouping can be turned off.
 *
 * WHAT COUNTS AS ONE THING
 * ------------------------
 * Same camera, and starting within `WINDOW_S` of the group's first event.
 * Measured against the group's START rather than against the previous member,
 * so a slow trickle of detections cannot chain into one enormous group that
 * spans a minute — each group covers a bounded slice of time.
 *
 * Not merged across cameras: the same person crossing two cameras is two pieces
 * of evidence about where they went, which is exactly what you want to see
 * separately.
 */
import type { NvrEvent } from './api';

/** How close in time two detections on one camera have to be to read as one
 *  moment. Ten seconds: long enough to catch a person and the car they arrived
 *  in, short enough that two separate visits never merge. */
export const GROUP_WINDOW_S = 10;

export interface EventGroup {
  /**
   * The event the row shows. The one with a clip if any member has one, else
   * one with a snapshot, else the newest — the row is somebody's way into the
   * moment, and landing on "no clip was saved" when a sibling has one would
   * make the moment look worse-recorded than it was.
   */
  lead: NvrEvent;
  /** Every event in the group, in page order. Length 1 for an ungrouped row. */
  events: NvrEvent[];
  /** Distinct labels across the group, for the collapsed row's chip. */
  labels: string[];
}

function labelsOf(e: NvrEvent): string[] {
  return e.labels && e.labels.length > 0 ? e.labels : [e.label];
}

/**
 * Group an ALREADY-SORTED page of events (newest first, as the API returns).
 *
 * Order is preserved: the first member of each group is the one that would
 * have appeared first ungrouped, so turning grouping on never reshuffles the
 * history under you.
 */
export function groupEvents(events: NvrEvent[]): EventGroup[] {
  const groups: EventGroup[] = [];
  // Per camera, the group currently open for it. Keyed by camera because two
  // cameras firing at the same instant are two separate things, and a single
  // "current group" would let one camera's event close another's.
  // The window is measured from the group's FIRST event, which is kept apart
  // from `lead` because the lead is re-picked afterwards.
  const open = new Map<string, { group: EventGroup; start: number }>();

  for (const event of events) {
    const current = open.get(event.camera);
    const within =
      current !== undefined &&
      Math.abs(current.start - event.start_time) <= GROUP_WINDOW_S;

    if (current && within) {
      current.group.events.push(event);
      for (const l of labelsOf(event)) {
        if (!current.group.labels.includes(l)) current.group.labels.push(l);
      }
      continue;
    }
    const group: EventGroup = {
      lead: event,
      events: [event],
      labels: [...labelsOf(event)],
    };
    groups.push(group);
    open.set(event.camera, { group, start: event.start_time });
  }
  for (const group of groups) group.lead = pickLead(group.events);
  return groups;
}

/** Best member to stand for the moment: clip, then snapshot, then the first. */
function pickLead(events: NvrEvent[]): NvrEvent {
  return (
    events.find((e) => e.has_clip) ??
    events.find((e) => e.has_snapshot) ??
    events[0]
  );
}

/** A stable key for a group, for React. Uses the FIRST member's id rather than
 *  the lead's: the lead can change when a sibling's clip lands, and the row
 *  should update in place rather than remount. */
export function groupKey(group: EventGroup): string {
  return `g${group.events[0].id}`;
}

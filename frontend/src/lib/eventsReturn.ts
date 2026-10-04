/**
 * Where you were in the Events list, kept across a trip into an event and back.
 *
 * The list holds its pages and scroll position in component state, so opening
 * an event unmounted it and coming back started again at the top with only the
 * first 50 loaded — after scrolling a day back, you had to find your place
 * again for every event you opened. This module-level snapshot (filters,
 * loaded events, paging offset, scroll position) outlives the unmount; the
 * list restores it when it comes back with the same filters within a few
 * minutes.
 */
import type { NvrEvent } from './api';

export interface EventsReturnPoint {
  /** The list's filters, serialized — a different filter starts fresh. */
  key: string;
  events: NvrEvent[];
  total: number;
  offset: number;
  scrollY: number;
  at: number;
}

/** Older than this, the list reloads instead: it would be too stale to trust. */
export const RETURN_TTL_MS = 15 * 60_000;

let point: EventsReturnPoint | null = null;

export function saveEventsReturnPoint(p: EventsReturnPoint): void {
  point = p;
}

/** The saved point for these filters, if fresh; consumed by reading it. */
export function takeEventsReturnPoint(key: string): EventsReturnPoint | null {
  const p = point;
  point = null;
  if (!p || p.key !== key || Date.now() - p.at > RETURN_TTL_MS || p.events.length === 0) {
    return null;
  }
  return p;
}

/** An event was deleted or excluded from its detail page: do not bring it back. */
export function forgetEventInReturnPoint(id: number | string): void {
  if (!point) return;
  const before = point.events.length;
  point.events = point.events.filter((e) => String(e.id) !== String(id));
  if (point.events.length < before) {
    point.total = Math.max(0, point.total - 1);
    point.offset = Math.max(0, point.offset - 1);
  }
}

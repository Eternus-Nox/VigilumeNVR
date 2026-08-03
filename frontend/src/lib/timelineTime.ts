/**
 * Shared time helpers for the (multi-camera) Timeline: local-day <-> date-string
 * conversion, hour-window alignment, and the wall-clock <-> media-time mapping
 * through a camera's real segments (so coverage gaps map correctly). Extracted
 * from the original single-camera Timeline so every synchronized player + the
 * multi-lane bar share one implementation.
 */
import type { RecordingSegment } from './api';

export const DAY = 86_400;
export const HOUR = 3_600;

/**
 * Max span (seconds) the timeline range-export allows, matching the backend
 * cap on `GET /api/recordings/{camera}/export.mp4`. The drag clamps to this so
 * the request never exceeds the server limit.
 */
export const MAX_EXPORT_SECONDS = 30 * 60;

export type Window = { start: number; end: number };

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// ---- local-day <-> date-string helpers (recordings use local dates) ----
export function localDayStart(dateStr: string): number {
  const [y, m, d] = dateStr.split('-').map(Number);
  return Math.floor(new Date(y, (m ?? 1) - 1, d ?? 1, 0, 0, 0, 0).getTime() / 1000);
}
export function epochToDateStr(epoch: number): string {
  const d = new Date(epoch * 1000);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}
export function todayStr(): string {
  return epochToDateStr(Date.now() / 1000);
}
export function shiftDate(dateStr: string, deltaDays: number): string {
  const [y, m, d] = dateStr.split('-').map(Number);
  const dt = new Date(y, (m ?? 1) - 1, d ?? 1);
  dt.setDate(dt.getDate() + deltaDays);
  return epochToDateStr(dt.getTime() / 1000);
}

/** Hour window aligned to the local day (so it matches recorder hour dirs). */
export function hourWindow(t: number, dayStart: number, dayEnd: number): Window {
  const start = dayStart + Math.floor((t - dayStart) / HOUR) * HOUR;
  return { start, end: Math.min(start + HOUR, dayEnd) };
}

// ---- wall-clock <-> media-time mapping through real segments ----
export function segsInWindow(segments: RecordingSegment[], win: Window): RecordingSegment[] {
  return segments.filter((s) => s.start + s.duration > win.start && s.start < win.end);
}
/** media (playlist) time for a wall-clock instant; handles coverage gaps. */
export function mediaTimeForWall(segments: RecordingSegment[], t: number, win: Window): number {
  let acc = 0;
  for (const s of segsInWindow(segments, win)) {
    const end = s.start + s.duration;
    if (t >= end) {
      acc += s.duration;
      continue;
    }
    if (t <= s.start) return acc; // t sits in a gap before this segment
    return acc + (t - s.start);
  }
  return acc;
}
/** inverse: wall-clock for a media-time offset (drives the playhead). */
export function wallForMediaTime(segments: RecordingSegment[], m: number, win: Window): number {
  let acc = 0;
  const segs = segsInWindow(segments, win);
  for (const s of segs) {
    if (m < acc + s.duration) return s.start + (m - acc);
    acc += s.duration;
  }
  const last = segs[segs.length - 1];
  return last ? last.start + last.duration : win.start;
}

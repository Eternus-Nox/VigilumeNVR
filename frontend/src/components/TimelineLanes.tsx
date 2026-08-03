/**
 * Unified single-bar scrubber for the multi-camera Timeline. Renders, over the
 * shared window [viewStart, viewEnd] (epoch seconds), ONE time axis and ONE
 * draggable playhead across ALL selected cameras on a single track: the RECORDED
 * coverage as the UNION of every selected camera's ranges (a slice is shaded if
 * ANY camera has footage, with a subtle density/heat where more cameras overlap).
 *
 * This bar is a pure recorded-footage scrubber — it renders no events. It owns
 * no time state either: the parent passes `playhead` and reacts to the seek
 * callbacks, so the wall-clock <-> media-time mapping lives in one place (per
 * camera, in SyncPlayer). Scrubbing/keyboard moves the single shared playhead,
 * which the parent fans out to every attached player.
 */
import { useCallback, useMemo, useRef } from 'react';
import type { RecordingRange } from '../lib/api';

export interface Lane {
  camera: string;
  name: string;
  ranges: RecordingRange[];
  /** Whether this camera currently has a player attached in the grid. */
  onView: boolean;
  /** False when the camera has no footage for the loaded day. */
  hasFootage: boolean;
}

/** A start→end wall-clock span (epoch seconds) selected for export. */
export interface TimeRange {
  start: number;
  end: number;
}

interface TimelineLanesProps {
  viewStart: number;
  viewEnd: number;
  lanes: Lane[];
  playhead: number;
  onSeekStart?: () => void;
  onSeekMove?: (t: number) => void;
  onSeekEnd: (t: number) => void;
  /** When true, dragging the axis marks an export range instead of scrubbing. */
  rangeMode?: boolean;
  /** The current export selection (highlighted band), or null. */
  range?: TimeRange | null;
  /** Fired as the range drag updates; already ordered + capped to `maxRangeSpan`. */
  onRangeChange?: (r: TimeRange) => void;
  /** Max span (seconds) a drag may cover; clamps relative to the drag anchor. */
  maxRangeSpan?: number;
  /**
   * Event start times (epoch seconds) drawn as small marks on the track, purely
   * so you can SEE where events are and scrub to them yourself. Deliberately
   * inert: no hit target, no click/tap, no tooltip — `pointer-events: none` so
   * they never steal a scrub. Reviewing an event is the Events page's job.
   */
  eventTimes?: number[];
}

/** A union-coverage slice: [start,end) covered by `count` distinct cameras. */
interface CoverageSlice {
  start: number;
  end: number;
  count: number;
}

function fmtClock(epoch: number, withMinutes = true): string {
  const d = new Date(epoch * 1000);
  return d.toLocaleTimeString([], {
    hour: 'numeric',
    ...(withMinutes ? { minute: '2-digit' } : {}),
  });
}

/** Merge one camera's (possibly touching/overlapping) ranges into disjoint spans. */
function mergeRanges(ranges: RecordingRange[]): RecordingRange[] {
  const sorted = ranges.filter((r) => r.end > r.start).sort((a, b) => a.start - b.start);
  const out: RecordingRange[] = [];
  for (const r of sorted) {
    const last = out[out.length - 1];
    if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
    else out.push({ start: r.start, end: r.end });
  }
  return out;
}

/**
 * Sweep every camera's merged ranges into disjoint union slices carrying the
 * count of DISTINCT cameras covering each slice (merging per-camera first keeps
 * one camera's overlapping ranges from double-counting the heat).
 */
function unionCoverage(lanes: Lane[]): { slices: CoverageSlice[]; maxCount: number } {
  const deltas: { t: number; d: number }[] = [];
  for (const lane of lanes) {
    for (const r of mergeRanges(lane.ranges)) {
      deltas.push({ t: r.start, d: 1 });
      deltas.push({ t: r.end, d: -1 });
    }
  }
  deltas.sort((a, b) => a.t - b.t);
  const slices: CoverageSlice[] = [];
  let count = 0;
  let prevT: number | null = null;
  let maxCount = 0;
  let i = 0;
  while (i < deltas.length) {
    const t = deltas[i].t;
    if (prevT != null && t > prevT && count > 0) {
      slices.push({ start: prevT, end: t, count });
      if (count > maxCount) maxCount = count;
    }
    while (i < deltas.length && deltas[i].t === t) {
      count += deltas[i].d;
      i += 1;
    }
    prevT = t;
  }
  return { slices, maxCount };
}

export default function TimelineLanes({
  viewStart,
  viewEnd,
  lanes,
  playhead,
  onSeekStart,
  onSeekMove,
  onSeekEnd,
  rangeMode = false,
  range = null,
  onRangeChange,
  maxRangeSpan,
  eventTimes,
}: TimelineLanesProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  // null = idle; otherwise which gesture the current pointer drag is performing.
  const dragModeRef = useRef<'scrub' | 'range' | null>(null);
  const rangeAnchorRef = useRef(0);
  const span = Math.max(1, viewEnd - viewStart);

  const pct = useCallback((t: number) => ((t - viewStart) / span) * 100, [viewStart, span]);

  // Union coverage (with per-slice camera count) across all selected cameras.
  const { slices, maxCount } = useMemo(() => unionCoverage(lanes), [lanes]);

  // MEMOISED BECAUSE SCRUBBING RE-RENDERS THIS COMPONENT AT POINTER RATE.
  // Dragging the playhead fires setPlayhead on every pointermove — 60-120 Hz on
  // a trackpad — and none of the work below depends on the playhead at all. Left
  // inline, every one of those frames rebuilt the whole coverage strip AND one
  // <div> per event mark (up to ~4,000 across four lanes on a busy day), so the
  // scrub got heavier the more the cameras had actually seen. React bails out of
  // reconciling a subtree when the element reference is unchanged, so hoisting
  // these two arrays takes the per-move cost to nothing.
  //
  // `pct` is a useCallback keyed on [viewStart, span], and in hour zoom `view`
  // comes from the hour-aligned window — so it stays referentially stable across
  // a drag and does not defeat these deps.
  const coverageEls = useMemo(
    () =>
      slices.map((s, i) => {
        const left = Math.max(0, pct(s.start));
        const right = Math.min(100, pct(s.end));
        if (right <= 0 || left >= 100 || right <= left) return null;
        // Heat: one camera → base alpha; more overlapping cameras → denser.
        const heat = maxCount > 1 ? (s.count - 1) / (maxCount - 1) : 0;
        const opacity = 0.55 + 0.45 * heat;
        return (
          <div
            key={i}
            className="tll-coverage"
            style={{ left: `${left}%`, width: `${right - left}%`, opacity }}
            title={s.count > 1 ? `${s.count} cameras recording` : undefined}
          />
        );
      }),
    [slices, maxCount, pct],
  );

  // Event locations: inert marks (pointer-events: none) so you can spot them and
  // scrub there yourself. Drawn before the playhead so the playhead reads on top.
  const eventEls = useMemo(
    () =>
      eventTimes?.map((t, i) => {
        const left = pct(t);
        if (left < 0 || left > 100) return null;
        return (
          <div key={`${t}-${i}`} className="tll-event" style={{ left: `${left}%` }} aria-hidden="true" />
        );
      }),
    [eventTimes, pct],
  );

  const timeAt = useCallback(
    (clientX: number): number => {
      const el = trackRef.current;
      if (!el) return viewStart;
      const rect = el.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return viewStart + frac * span;
    },
    [viewStart, span],
  );

  // Ordered + anchor-clamped span for the current range drag.
  const rangeFromDrag = useCallback(
    (t: number): TimeRange => {
      const a = rangeAnchorRef.current;
      let start = Math.min(a, t);
      let end = Math.max(a, t);
      if (maxRangeSpan && end - start > maxRangeSpan) {
        // Keep the anchor fixed; cap the moving edge so the band never exceeds
        // the max, whichever direction the user drags.
        if (t >= a) end = a + maxRangeSpan;
        else start = a - maxRangeSpan;
      }
      return { start, end };
    },
    [maxRangeSpan],
  );

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    trackRef.current?.setPointerCapture(e.pointerId);
    const t = timeAt(e.clientX);
    if (rangeMode) {
      dragModeRef.current = 'range';
      rangeAnchorRef.current = t;
      onRangeChange?.({ start: t, end: t });
    } else {
      dragModeRef.current = 'scrub';
      onSeekStart?.();
      onSeekMove?.(t);
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const mode = dragModeRef.current;
    if (!mode) return;
    const t = timeAt(e.clientX);
    if (mode === 'range') onRangeChange?.(rangeFromDrag(t));
    else onSeekMove?.(t);
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    const mode = dragModeRef.current;
    if (!mode) return;
    dragModeRef.current = null;
    try {
      trackRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* pointer already released */
    }
    const t = timeAt(e.clientX);
    if (mode === 'range') onRangeChange?.(rangeFromDrag(t));
    else onSeekEnd(t);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = Math.max(5, Math.round(span / 100));
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      onSeekEnd(Math.max(viewStart, playhead - step));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      onSeekEnd(Math.min(viewEnd, playhead + step));
    }
  };

  // Axis ticks: aim for ~6 labels; snap the interval to a friendly value.
  const niceSteps = [300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200];
  const target = span / 6;
  const step = niceSteps.find((s) => s >= target) ?? 43200;
  const ticks: number[] = [];
  let first = Math.floor(viewStart / step) * step;
  if (first <= viewStart) first += step;
  for (let t = first; t < viewEnd; t += step) ticks.push(t);

  const playheadPct = Math.min(100, Math.max(0, pct(playhead)));
  const withMinutes = step < 3600;

  return (
    <div className="tll-wrap">
      <div className="tll-axis">
        {ticks.map((t) => (
          <span key={t} className="tll-tick" style={{ left: `${pct(t)}%` }}>
            {fmtClock(t, withMinutes)}
          </span>
        ))}
      </div>
      <div
        ref={trackRef}
        className={`tll-track${rangeMode ? ' tll-track-range' : ''}`}
        role="slider"
        tabIndex={0}
        aria-label="Recording timeline"
        aria-valuemin={viewStart}
        aria-valuemax={viewEnd}
        aria-valuenow={playhead}
        aria-valuetext={fmtClock(playhead)}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={onKeyDown}
      >
        {coverageEls}
        {eventEls}

        {range && range.end > range.start && (() => {
          const left = Math.max(0, pct(range.start));
          const right = Math.min(100, pct(range.end));
          if (right <= left) return null;
          return (
            <div
              className="tll-range"
              style={{ left: `${left}%`, width: `${right - left}%` }}
              aria-hidden="true"
            />
          );
        })()}

        <div className="tll-playhead" style={{ left: `${playheadPct}%` }}>
          <span className="tll-playhead-knob" />
        </div>
      </div>
    </div>
  );
}

/**
 * Horizontal scrubber for the Timeline page. Renders, over the window
 * [viewStart, viewEnd] (epoch seconds):
 * - shaded RECORDED-coverage ranges,
 * - EVENT markers colored by label (clickable, tooltip),
 * - a draggable PLAYHEAD; click / drag anywhere on the bar seeks.
 *
 * It owns no time state — the parent passes `playhead` and reacts to the seek
 * callbacks, so wall-clock ⇄ media-time mapping lives in one place.
 */
import { useCallback, useRef } from 'react';
import type { NvrEvent, RecordingRange } from '../lib/api';
import { titleCase } from '../lib/format';

interface TimelineBarProps {
  viewStart: number;
  viewEnd: number;
  ranges: RecordingRange[];
  events: NvrEvent[];
  playhead: number;
  /** Pointer went down — parent should pause playhead auto-follow. */
  onSeekStart?: () => void;
  /** Live drag / move to wall-clock `t`. */
  onSeekMove?: (t: number) => void;
  /** Pointer released at wall-clock `t` (also fires for a plain click). */
  onSeekEnd: (t: number) => void;
  onEventClick: (ev: NvrEvent) => void;
}

/** Stable, distinct hues per label; anything unknown falls back to slate. */
const LABEL_COLORS: Record<string, string> = {
  person: '#38bdf8',
  car: '#a78bfa',
  truck: '#c084fc',
  dog: '#fbbf24',
  cat: '#34d399',
  bicycle: '#f472b6',
  motorcycle: '#fb923c',
};

export function labelColor(label: string): string {
  return LABEL_COLORS[label] ?? '#94a3b8';
}

function fmtClock(epoch: number, withMinutes = true): string {
  const d = new Date(epoch * 1000);
  return d.toLocaleTimeString([], {
    hour: 'numeric',
    ...(withMinutes ? { minute: '2-digit' } : {}),
  });
}

export default function TimelineBar({
  viewStart,
  viewEnd,
  ranges,
  events,
  playhead,
  onSeekStart,
  onSeekMove,
  onSeekEnd,
  onEventClick,
}: TimelineBarProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const span = Math.max(1, viewEnd - viewStart);

  const pct = useCallback((t: number) => ((t - viewStart) / span) * 100, [viewStart, span]);

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

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Let event-marker buttons handle their own clicks.
    if ((e.target as HTMLElement).closest('.tl-event')) return;
    e.preventDefault();
    draggingRef.current = true;
    trackRef.current?.setPointerCapture(e.pointerId);
    onSeekStart?.();
    onSeekMove?.(timeAt(e.clientX));
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    onSeekMove?.(timeAt(e.clientX));
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    try {
      trackRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* pointer already released */
    }
    onSeekEnd(timeAt(e.clientX));
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    // Coarse step scales with zoom: ~1% of the window, min 5 s.
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
  const niceSteps = [
    300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200,
  ];
  const target = span / 6;
  const step = niceSteps.find((s) => s >= target) ?? 43200;
  const ticks: number[] = [];
  // Interior ticks only (strictly inside the window) so centered labels never
  // clip past the bar's edges / force the page to scroll.
  let first = Math.floor(viewStart / step) * step;
  if (first <= viewStart) first += step;
  for (let t = first; t < viewEnd; t += step) ticks.push(t);

  const playheadPct = Math.min(100, Math.max(0, pct(playhead)));
  const withMinutes = step < 3600;

  return (
    <div className="tl-wrap">
      <div className="tl-axis">
        {ticks.map((t) => (
          <span key={t} className="tl-tick" style={{ left: `${pct(t)}%` }}>
            {fmtClock(t, withMinutes)}
          </span>
        ))}
      </div>
      <div
        ref={trackRef}
        className="tl-track"
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
        {ranges.map((r, i) => {
          const left = Math.max(0, pct(r.start));
          const right = Math.min(100, pct(r.end));
          if (right <= 0 || left >= 100 || right <= left) return null;
          return (
            <div
              key={i}
              className="tl-coverage"
              style={{ left: `${left}%`, width: `${right - left}%` }}
            />
          );
        })}

        {events.map((ev) => {
          const end = ev.end_time ?? ev.start_time + 2;
          const left = pct(ev.start_time);
          if (left > 100 || pct(end) < 0) return null;
          const width = Math.max(0.5, pct(end) - left);
          const dur = Math.max(0, Math.round(end - ev.start_time));
          return (
            <button
              type="button"
              key={String(ev.id)}
              className="tl-event"
              style={{
                left: `${Math.max(0, left)}%`,
                width: `${Math.min(width, 100 - Math.max(0, left))}%`,
                background: labelColor(ev.label),
              }}
              title={`${titleCase(ev.label)}${ev.count > 1 ? ` ×${ev.count}` : ''} · ${fmtClock(
                ev.start_time,
              )}${dur ? ` · ${dur}s` : ''}`}
              aria-label={`${titleCase(ev.label)} event at ${fmtClock(ev.start_time)}`}
              onClick={(e) => {
                e.stopPropagation();
                onEventClick(ev);
              }}
            />
          );
        })}

        <div className="tl-playhead" style={{ left: `${playheadPct}%` }}>
          <span className="tl-playhead-knob" />
        </div>
      </div>
    </div>
  );
}

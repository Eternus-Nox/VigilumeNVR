/**
 * Timeline (/timeline) — multi-camera 24/7 continuous-recording scrubber.
 *
 * A pure RECORDED-FOOTAGE review page: no events here (the separate Events page
 * owns those). A simple multi-select picks 1–MAX_TIMELINE_CAMERAS cameras; the
 * default on first load is exactly the first recording camera. Every selected
 * camera shares ONE unified time axis + ONE draggable playhead (TimelineLanes):
 * the bar shows the UNION of every selected camera's recorded coverage (heat-
 * shaded where more cameras overlap). Scrubbing/clicking seeks EVERY on-screen
 * player to that wall-clock moment. Per-camera indexes are fetched in parallel
 * for the chosen day (defaulting on first load to the most recent day the
 * selected set recorded).
 *
 * Selection is capped at MAX_TIMELINE_CAMERAS (== the max simultaneously-attached
 * players; consumer NVIDIA NVENC concurrent-transcode headroom), so every
 * selected camera is on screen at once — each player owns the bounded (~1 h,
 * hour-aligned) VOD-window + wall-clock<->media-time mapping (SyncPlayer).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type RecordingCamera, type RecordingIndex } from '../lib/api';
import { downloadAttachment } from '../lib/download';
import { useAppState } from '../state/AppState';
import {
  DAY,
  HOUR,
  MAX_EXPORT_SECONDS,
  clamp,
  epochToDateStr,
  hourWindow,
  localDayStart,
  segsInWindow,
  shiftDate,
  todayStr,
  type Window,
} from '../lib/timelineTime';
import {
  MAX_TIMELINE_CAMERAS,
  readStoredCameras,
  storeCameras,
} from '../lib/timelineSelection';
import { titleCase } from '../lib/format';
import TimelineSelector from '../components/TimelineSelector';
import TimelineLanes, { type Lane, type TimeRange } from '../components/TimelineLanes';
import TimelineTransport from '../components/TimelineTransport';
import SyncPlaybackGrid, { type GridCamera } from '../components/SyncPlaybackGrid';
import type { SeekRequest } from '../components/SyncPlayer';

/** "1h 05m" / "12m 30s" / "45s" for a range duration (seconds). */
function formatSpan(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

/** Wall-clock label (h:mm:ss) for a range endpoint. */
function formatClock(epoch: number): string {
  return new Date(epoch * 1000).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  });
}

type DayData = Record<
  string,
  {
    index: RecordingIndex | null;
    /** Event start times (epoch s) for the loaded day — drawn as inert marks on
     *  the bar so you can see where events are and scrub to them. */
    eventTimes: number[];
  }
>;

export default function Timeline() {
  const { pushToast } = useAppState();
  const [recCameras, setRecCameras] = useState<RecordingCamera[] | null>(null);
  // null = nothing stored yet → default to the first recording camera. Once the
  // user touches the selection this becomes an explicit (possibly empty) list.
  const [stored, setStored] = useState<string[] | null>(() => readStoredCameras());
  const [date, setDate] = useState('');
  const [zoom, setZoom] = useState<'day' | 'hour'>('day');
  const [playhead, setPlayhead] = useState(0);
  const [seek, setSeek] = useState<SeekRequest>({ t: 0, seq: 0, play: false });
  // ---- shared transport state (drives ALL on-screen synced players) ----
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [muted, setMuted] = useState(true);
  const [dayData, setDayData] = useState<DayData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ---- range-export mode (drag a span → download MP4 per camera) ----
  const [rangeMode, setRangeMode] = useState(false);
  const [range, setRange] = useState<TimeRange | null>(null);
  const [exportCams, setExportCams] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState(false);

  const reqSeq = useRef(0);
  const seekSeqRef = useRef(1);
  const scrubbingRef = useRef(false);
  // Latest playing state for callbacks (requestSeek adopts it; onFollow gates on it).
  const playingRef = useRef(false);
  playingRef.current = playing;

  // ---- load recording-camera list ----
  useEffect(() => {
    let alive = true;
    api
      .recordingCameras()
      .then((cams) => alive && setRecCameras(cams))
      .catch((e) => alive && setError(e instanceof Error ? e.message : 'Failed to load cameras'));
    return () => {
      alive = false;
    };
  }, []);

  const camMap = useMemo(
    () => new Map((recCameras ?? []).map((c) => [c.camera, c] as const)),
    [recCameras],
  );

  // ---- resolve the (persisted) selection to an ordered camera-name list ----
  // Order follows the recordings list (stable lane/grid order); stale stored
  // names drop out automatically. Nothing stored yet → the first camera.
  const selectedCameras = useMemo(() => {
    if (!recCameras || recCameras.length === 0) return [];
    const order = recCameras.map((c) => c.camera);
    // First load → the first camera. Order is the API's (position, name), i.e.
    // the same first camera the dashboard shows. Prefer one that actually HAS
    // footage: /api/recordings/cameras returns cameras that have never
    // recorded too, and opening the timeline on one of those shows a
    // permanently empty bar, which reads as broken rather than as empty.
    const firstWithFootage = recCameras.find((c) => c.has_recordings)?.camera ?? order[0];
    const chosen =
      stored === null
        ? [firstWithFootage]
        : stored.filter((n) => order.includes(n)); // may be empty (user cleared)
    return order.filter((n) => chosen.includes(n)).slice(0, MAX_TIMELINE_CAMERAS);
  }, [recCameras, stored]);

  // ---- toggle a camera in/out of the selection (cap-enforced) ----
  const toggleCamera = useCallback(
    (name: string) => {
      setStored((prev) => {
        // Base off the current EFFECTIVE selection so the first toggle acts on
        // the defaulted-first camera, not an empty list.
        const base = prev ?? selectedCameras;
        let next: string[];
        if (base.includes(name)) {
          next = base.filter((n) => n !== name);
        } else {
          if (base.length >= MAX_TIMELINE_CAMERAS) return prev; // block a 5th
          next = [...base, name];
        }
        storeCameras(next);
        return next;
      });
    },
    [selectedCameras],
  );

  // ---- anchor the day ONCE (initial load) to the newest footage of the set ----
  useEffect(() => {
    if (date || selectedCameras.length === 0) return;
    let latest: number | null = null;
    for (const name of selectedCameras) {
      const c = camMap.get(name);
      if (c?.latest != null) latest = latest == null ? c.latest : Math.max(latest, c.latest);
    }
    setDate(latest != null ? epochToDateStr(latest) : todayStr());
  }, [date, selectedCameras, camMap]);

  // ---- issue an explicit seek that fans out to every attached player ----
  // Seeks adopt the current transport state: playing → stays playing through the
  // seek; paused → seeks and stays paused (resume on Play). The playhead is
  // parked at the target immediately so the readout/bar don't lag the seek.
  const requestSeek = useCallback((t: number) => {
    setPlayhead(t);
    setSeek({ t, seq: seekSeqRef.current++, play: playingRef.current });
  }, []);

  // ---- fetch each selected camera's index for the day (in parallel) ----
  useEffect(() => {
    if (!date || selectedCameras.length === 0) {
      setDayData({});
      return;
    }
    const seq = ++reqSeq.current;
    const ds = localDayStart(date);
    const de = ds + DAY;
    setLoading(true);
    setError(null);
    Promise.all(
      selectedCameras.map(async (name) => {
        // Coverage and event marks load independently: events are a cosmetic
        // overlay, so a failing events call must never cost us the footage lane.
        const [idx, eventTimes] = await Promise.all([
          api.recordingIndex(name, date).catch(() => null),
          api
            .events({ camera: name, after: ds, before: de, limit: 1000 })
            .then((page) => page.events.map((e) => e.start_time))
            .catch(() => [] as number[]),
        ]);
        return [name, { index: idx, eventTimes }] as const;
      }),
    )
      .then((results) => {
        if (seq !== reqSeq.current) return;
        const map: DayData = {};
        for (const [name, data] of results) map[name] = data;
        setDayData(map);
        // Default the playhead to the end of the day's coverage across the set.
        let end: number | null = null;
        for (const [, data] of results) {
          const last = data.index?.ranges[data.index.ranges.length - 1];
          if (last) end = end == null ? last.end : Math.max(end, last.end);
        }
        const ph = end != null ? clamp(end - 2, ds, de - 1) : ds;
        setPlayhead(ph);
        setSeek({ t: ph, seq: seekSeqRef.current++, play: false });
        setLoading(false);
      })
      .catch(() => {
        if (seq !== reqSeq.current) return;
        setError('Failed to load recordings');
        setLoading(false);
      });
  }, [date, selectedCameras]);

  // ---- leader players report playback position → shared playhead follows ----
  // The playhead only advances while playing; when paused it holds still (a
  // seek's currentTime change can still fire a stray timeupdate we ignore here).
  const onFollow = useCallback((t: number) => {
    if (scrubbingRef.current || !playingRef.current) return;
    setPlayhead(t);
  }, []);

  const dayStart = date ? localDayStart(date) : 0;
  const dayEnd = dayStart + DAY;

  // ---- transport controls (drive every on-screen synced player together) ----
  const togglePlay = useCallback(() => setPlaying((p) => !p), []);

  // Skip ±N seconds through the shared seek path (all players re-align).
  const skipBy = useCallback(
    (delta: number) => {
      if (!date) return;
      requestSeek(clamp(playhead + delta, dayStart, dayEnd - 1));
    },
    [date, playhead, dayStart, dayEnd, requestSeek],
  );

  // Event marks across the whole selected set, merged onto the one shared bar
  // (it renders a union, not per-camera lanes). Purely a "look here" hint.
  const eventTimes = useMemo(
    () => selectedCameras.flatMap((name) => dayData[name]?.eventTimes ?? []),
    [selectedCameras, dayData],
  );

  // Latest recorded coverage across the selected set (for "jump to newest").
  const newestCoverage = useMemo(() => {
    let end: number | null = null;
    for (const name of selectedCameras) {
      const ranges = dayData[name]?.index?.ranges;
      const last = ranges?.[ranges.length - 1];
      if (last) end = end == null ? last.end : Math.max(end, last.end);
    }
    return end;
  }, [selectedCameras, dayData]);

  const jumpToNewest = useCallback(() => {
    if (newestCoverage == null || !date) return;
    requestSeek(clamp(newestCoverage - 2, dayStart, dayEnd - 1));
  }, [newestCoverage, date, dayStart, dayEnd, requestSeek]);

  // A drag on the axis (in range mode) reports an ordered, cap-limited span; we
  // clamp it to the loaded day for safety and store it as the export selection.
  const onRangeChange = useCallback(
    (r: TimeRange) => {
      setRange({
        start: clamp(r.start, dayStart, dayEnd - 1),
        end: clamp(r.end, dayStart, dayEnd),
      });
    },
    [dayStart, dayEnd],
  );

  const toggleRangeMode = useCallback(() => {
    setRangeMode((on) => {
      if (on) {
        setRange(null); // leaving: drop the selection, restore normal scrub
      } else {
        // entering: default the export set to what's currently selected
        setExportCams(new Set(selectedCameras));
      }
      return !on;
    });
  }, [selectedCameras]);

  const toggleExportCam = useCallback((name: string) => {
    setExportCams((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  // A changed day / camera set invalidates any pending selection.
  useEffect(() => {
    setRange(null);
  }, [date, selectedCameras]);

  const rangeSecs = range ? Math.max(0, Math.round(range.end - range.start)) : 0;
  const atMaxRange = rangeSecs >= MAX_EXPORT_SECONDS;

  const doExport = useCallback(async () => {
    if (!range || exporting) return;
    const start = Math.floor(range.start);
    const end = Math.floor(range.end);
    if (end - start < 1) {
      pushToast({
        kind: 'error',
        title: 'Select a longer range',
        body: 'Drag on the timeline to mark at least a second of footage.',
      });
      return;
    }
    const cams = selectedCameras.filter((n) => exportCams.has(n));
    if (cams.length === 0) {
      pushToast({
        kind: 'error',
        title: 'No cameras chosen',
        body: 'Pick at least one camera to export.',
      });
      return;
    }
    setExporting(true);
    try {
      // Sequential: each export transcodes server-side, so we avoid firing
      // several heavy jobs at once.
      for (const cam of cams) {
        await downloadAttachment(api.recordingExportUrl(cam, start, end), `${cam}-${start}-${end}.mp4`);
      }
      pushToast({
        kind: 'info',
        title: 'Export ready',
        body: cams.length > 1 ? `${cams.length} clips downloaded` : '',
      });
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Export failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      setExporting(false);
    }
  }, [range, exporting, selectedCameras, exportCams, pushToast]);

  const view: Window = useMemo(() => {
    if (zoom === 'hour' && date) return hourWindow(playhead, dayStart, dayEnd);
    return { start: dayStart, end: dayEnd };
  }, [zoom, date, playhead, dayStart, dayEnd]);

  const lanes: Lane[] = useMemo(
    () =>
      selectedCameras.map((name) => {
        const c = camMap.get(name);
        const d = dayData[name];
        return {
          camera: name,
          name: c?.friendly_name || titleCase(name),
          ranges: d?.index?.ranges ?? [],
          onView: true, // selection is capped at MAX_TIMELINE_CAMERAS → all on screen
          hasFootage: (d?.index?.ranges.length ?? 0) > 0,
        };
      }),
    [selectedCameras, camMap, dayData],
  );

  const gridCameras: GridCamera[] = useMemo(
    () =>
      selectedCameras.map((name) => {
        const c = camMap.get(name);
        return {
          camera: name,
          friendlyName: c?.friendly_name || titleCase(name),
          segments: dayData[name]?.index?.segments ?? [],
        };
      }),
    [selectedCameras, camMap, dayData],
  );

  // The shared playhead follows the LEADER player's timeupdate. Lead with the
  // first on-screen camera that actually has footage in the playhead's hour —
  // a footage-less first tile has no playing <video> (no timeupdate), which
  // would freeze the playhead for the whole group while the others play.
  const leaderCamera = useMemo(() => {
    if (!date) return selectedCameras[0];
    const win = hourWindow(playhead, dayStart, dayEnd);
    return (
      selectedCameras.find((n) => segsInWindow(dayData[n]?.index?.segments ?? [], win).length > 0) ??
      selectedCameras[0]
    );
  }, [selectedCameras, date, playhead, dayStart, dayEnd, dayData]);

  const isToday = date >= todayStr();
  const hasAnyCoverage = lanes.some((l) => l.hasFootage);
  const nothingSelected = !!recCameras && selectedCameras.length === 0;

  // Spacebar toggles play/pause page-wide (a window listener — a handler on the
  // page div would miss keydowns while focus sits on <body>, the normal state
  // after load). Form controls, buttons, and editables keep their native space.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== ' ' && e.code !== 'Space') return;
      if (e.repeat) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        tag === 'BUTTON' ||
        el?.isContentEditable
      )
        return;
      if (!hasAnyCoverage) return; // transport is disabled — space matches it
      e.preventDefault();
      setPlaying((p) => !p);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hasAnyCoverage]);

  const playheadLabel = playhead
    ? new Date(playhead * 1000).toLocaleTimeString([], {
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
      })
    : '—';

  return (
    <div className="page timeline-page">
      <div className="page-head">
        <h1>Timeline</h1>
        <span className="muted">
          {selectedCameras.length
            ? `${selectedCameras.length} camera${selectedCameras.length > 1 ? 's' : ''}`
            : ''}
        </span>
      </div>

      <div className="tl-controls">
        {recCameras ? (
          <TimelineSelector
            recCameras={recCameras}
            selected={selectedCameras}
            max={MAX_TIMELINE_CAMERAS}
            onToggle={toggleCamera}
          />
        ) : (
          <span className="muted">Loading cameras…</span>
        )}

        <div className="tl-daynav">
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setDate((d) => shiftDate(d, -1))}
            aria-label="Previous day"
            disabled={!date}
          >
            ‹
          </button>
          <span className="tl-date">
            {date
              ? new Date(dayStart * 1000).toLocaleDateString([], {
                  weekday: 'short',
                  month: 'short',
                  day: 'numeric',
                })
              : '—'}
          </span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setDate((d) => shiftDate(d, 1))}
            aria-label="Next day"
            disabled={isToday || !date}
          >
            ›
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setDate(todayStr())}
            disabled={isToday || !date}
          >
            Today
          </button>
        </div>

        <div className="tl-zoom" role="group" aria-label="Zoom">
          <button
            type="button"
            className={`seg-btn ${zoom === 'day' ? 'seg-on' : ''}`}
            onClick={() => setZoom('day')}
          >
            Day
          </button>
          <button
            type="button"
            className={`seg-btn ${zoom === 'hour' ? 'seg-on' : ''}`}
            onClick={() => setZoom('hour')}
          >
            1 hour
          </button>
        </div>

        <button
          type="button"
          className={`btn btn-sm tl-range-toggle${rangeMode ? ' tl-range-toggle-on' : ''}`}
          onClick={toggleRangeMode}
          aria-pressed={rangeMode}
          disabled={nothingSelected}
        >
          {rangeMode ? 'Exit range select' : 'Select range'}
        </button>
      </div>

      {error ? (
        <div className="banner banner-error">
          <span>{error}</span>
        </div>
      ) : nothingSelected ? (
        <p className="muted tl-empty">No cameras selected — pick a camera above to scrub its footage.</p>
      ) : (
        <>
          <SyncPlaybackGrid
            cameras={gridCameras}
            date={date}
            seek={seek}
            playing={playing}
            rate={rate}
            muted={muted}
            leaderCamera={leaderCamera}
            onFollow={onFollow}
            onRemove={selectedCameras.length > 1 ? toggleCamera : undefined}
          />

          <div className="tl-readout">
            <span className="tl-readout-time">{playheadLabel}</span>
            {zoom === 'hour' && (
              <div className="tl-hour-nav">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => requestSeek(Math.max(dayStart, view.start - HOUR))}
                  disabled={view.start <= dayStart}
                  aria-label="Previous hour"
                >
                  ‹ hour
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => requestSeek(Math.min(dayEnd - 1, view.start + HOUR))}
                  disabled={view.end >= dayEnd}
                  aria-label="Next hour"
                >
                  hour ›
                </button>
              </div>
            )}
          </div>

          <TimelineLanes
            viewStart={view.start}
            viewEnd={view.end}
            lanes={lanes}
            eventTimes={eventTimes}
            playhead={playhead}
            onSeekStart={() => {
              scrubbingRef.current = true;
            }}
            onSeekMove={(t) => setPlayhead(clamp(t, dayStart, dayEnd - 1))}
            onSeekEnd={(t) => {
              scrubbingRef.current = false;
              requestSeek(clamp(t, dayStart, dayEnd - 1));
            }}
            rangeMode={rangeMode}
            range={range}
            onRangeChange={onRangeChange}
            maxRangeSpan={MAX_EXPORT_SECONDS}
          />

          <TimelineTransport
            playing={playing}
            onTogglePlay={togglePlay}
            playheadLabel={playheadLabel}
            onSkip={skipBy}
            rate={rate}
            onRateChange={setRate}
            muted={muted}
            onToggleMute={() => setMuted((m) => !m)}
            onJumpToNewest={jumpToNewest}
            canJumpToNewest={newestCoverage != null}
            disabled={!hasAnyCoverage}
          />

          {rangeMode &&
            (rangeSecs > 0 ? (
              <div className="tl-range-bar">
                <div className="tl-range-info">
                  <strong>Export {formatSpan(rangeSecs)}</strong>
                  <span className="muted small">
                    {formatClock(range!.start)} – {formatClock(range!.end)}
                    {atMaxRange ? ' · max 30 min' : ''}
                  </span>
                </div>
                <div className="tl-range-cams" role="group" aria-label="Cameras to export">
                  {selectedCameras.map((name) => {
                    const on = exportCams.has(name);
                    return (
                      <button
                        key={name}
                        type="button"
                        className={`tl-range-chip${on ? ' tl-range-chip-on' : ''}`}
                        onClick={() => toggleExportCam(name)}
                        aria-pressed={on}
                      >
                        {camMap.get(name)?.friendly_name || titleCase(name)}
                      </button>
                    );
                  })}
                </div>
                <div className="tl-range-actions">
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => setRange(null)}
                    disabled={exporting}
                  >
                    Clear
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    onClick={() => void doExport()}
                    disabled={exporting || exportCams.size === 0}
                  >
                    {exporting ? 'Preparing export…' : 'Export'}
                  </button>
                </div>
              </div>
            ) : (
              <p className="muted tl-range-hint">
                Drag across the timeline to select a span to export (up to 30 minutes).
              </p>
            ))}

          {!loading && !hasAnyCoverage && (
            <p className="muted tl-empty">
              No continuous recording saved for the selected cameras on this day.
            </p>
          )}
          {loading && <div className="page-loading">Loading recordings…</div>}
        </>
      )}
    </div>
  );
}

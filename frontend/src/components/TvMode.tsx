/**
 * TV mode: a chrome-free fullscreen wall of live tiles. The auto-fit grid
 * picks the column count maximizing total 16:9 tile area (contract formula),
 * recomputed on resize / orientation / fullscreen changes via a
 * ResizeObserver on the root. Every tile connects eagerly (a TV wall wants
 * all streams live), a screen wake lock is held while visible (re-acquired
 * on visibilitychange), and the cursor + controls auto-hide after 3 s idle.
 *
 * Dashboard path: mounted with `autoFullscreen` from the TV button click
 * (still inside the user gesture); leaving fullscreen (Esc) exits TV mode.
 * Standalone /tv path: the TV layout persists outside fullscreen and a
 * subtle "tap for fullscreen" hint is shown (browser gesture rules).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import LivePlayer from './LivePlayer';
import { useCameraLive } from '../state/AppState';
import { titleCase } from '../lib/format';
import type { Camera } from '../lib/api';

const IDLE_MS = 3000;

interface TvModeProps {
  /** Cameras to show, already filtered + ordered. */
  cameras: Camera[];
  /** Request fullscreen on mount (dashboard button — still a user gesture). */
  autoFullscreen?: boolean;
  /** /tv page: leaving fullscreen keeps the TV layout (kiosk bookmark). */
  standalone?: boolean;
  onExit: () => void;
}

/** Contract algorithm: pick cols maximizing N * tileW * tileH for 16:9 tiles. */
function bestLayout(n: number, w: number, h: number): { cols: number; tileW: number; tileH: number } {
  let best = { cols: 1, tileW: 0, tileH: 0 };
  let bestScore = -1;
  for (let cols = 1; cols <= n; cols += 1) {
    const rows = Math.ceil(n / cols);
    let tileW = w / cols;
    let tileH = Math.min((tileW * 9) / 16, h / rows);
    tileW = Math.min(tileW, (tileH * 16) / 9);
    const score = n * tileW * tileH;
    if (score > bestScore) {
      bestScore = score;
      best = { cols, tileW, tileH };
    }
  }
  return best;
}

export default function TvMode({ cameras, autoFullscreen, standalone, onExit }: TvModeProps) {
  const { isOnline } = useCameraLive();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [idle, setIdle] = useState(false);
  const [fullscreen, setFullscreen] = useState(() => !!document.fullscreenElement);
  const exitRef = useRef(onExit);
  exitRef.current = onExit;

  const requestFs = useCallback(() => {
    const root = rootRef.current;
    if (!root || document.fullscreenElement) return;
    // May reject (gesture expired, iOS Safari) — the fixed overlay still
    // covers all chrome, so TV mode works regardless.
    root.requestFullscreen?.().catch(() => {});
  }, []);

  // Track the root size (covers resize, orientation and fullscreen changes).
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const update = () => setSize({ w: root.clientWidth, h: root.clientHeight });
    update();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', update);
      return () => window.removeEventListener('resize', update);
    }
    const ro = new ResizeObserver(update);
    ro.observe(root);
    return () => ro.disconnect();
  }, []);

  // Fullscreen lifecycle. Exiting fullscreen (Esc) leaves TV mode unless
  // this is the standalone /tv page; unmount always releases fullscreen.
  useEffect(() => {
    if (autoFullscreen) requestFs();
    const onChange = () => {
      const fs = !!document.fullscreenElement;
      setFullscreen(fs);
      if (!fs && !standalone) exitRef.current();
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => {
      document.removeEventListener('fullscreenchange', onChange);
      if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
    };
  }, [autoFullscreen, standalone, requestFs]);

  // Esc exits the overlay even when the fullscreen request failed. (When
  // actually fullscreen, keydown fires before the browser exits fullscreen,
  // so this guard skips and fullscreenchange handles it.)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !document.fullscreenElement) exitRef.current();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  // Cursor + controls auto-hide after 3 s of no pointer/key activity.
  useEffect(() => {
    const root = rootRef.current;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poke = () => {
      setIdle(false);
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => setIdle(true), IDLE_MS);
    };
    poke();
    root?.addEventListener('pointermove', poke);
    root?.addEventListener('pointerdown', poke);
    document.addEventListener('keydown', poke);
    return () => {
      if (timer) clearTimeout(timer);
      root?.removeEventListener('pointermove', poke);
      root?.removeEventListener('pointerdown', poke);
      document.removeEventListener('keydown', poke);
    };
  }, []);

  // Screen wake lock: held while in TV mode, auto-released by the browser on
  // visibility loss and re-acquired when the page becomes visible again.
  useEffect(() => {
    let lock: WakeLockSentinel | null = null;
    let disposed = false;
    const acquire = async () => {
      if (!('wakeLock' in navigator) || document.visibilityState !== 'visible') return;
      try {
        const sentinel = await navigator.wakeLock.request('screen');
        if (disposed) {
          void sentinel.release().catch(() => {});
          return;
        }
        lock = sentinel;
      } catch {
        /* unsupported or denied (e.g. battery saver) — TV mode still works */
      }
    };
    const onVisibility = () => {
      if (document.visibilityState === 'visible') void acquire();
    };
    void acquire();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      disposed = true;
      document.removeEventListener('visibilitychange', onVisibility);
      void lock?.release().catch(() => {});
    };
  }, []);

  const layout = useMemo(
    () => (size && cameras.length > 0 ? bestLayout(cameras.length, size.w, size.h) : null),
    [size, cameras.length],
  );

  return (
    <div
      ref={rootRef}
      className={`tv-root${idle ? ' tv-idle' : ''}`}
      onClick={standalone && !fullscreen ? requestFs : undefined}
    >
      {cameras.length === 0 ? (
        <p className="muted">No cameras to show.</p>
      ) : (
        layout && (
          <div
            className="tv-grid"
            style={{
              gridTemplateColumns: `repeat(${layout.cols}, ${layout.tileW}px)`,
              gridAutoRows: `${layout.tileH}px`,
            }}
          >
            {cameras.map((cam) => (
              <div key={cam.name} className="tv-tile">
                {/* Privacy Mode wins over every other tile state, INCLUDING
                    offline — same rule as the Dashboard tile. The backend has
                    removed this camera's go2rtc streams, so mounting the player
                    would leave a permanently black tile on the wall that an
                    operator cannot tell apart from a failed camera. */}
                {cam.private ? (
                  <div className="camera-private">
                    <span className="camera-private-title">Privacy Mode</span>
                    <span className="camera-private-sub">nothing is being captured</span>
                  </div>
                ) : isOnline(cam) ? (
                  // Wall of >1 tile → substreams to keep total bitrate sane;
                  // a single-camera wall gets the full-res main stream.
                  <LivePlayer camera={cam.name} eager sub={cameras.length > 1} />
                ) : (
                  <div className="camera-offline">
                    <span>offline</span>
                  </div>
                )}
                <span className="tv-tile-name tv-chrome">
                  {cam.friendly_name || titleCase(cam.name)}
                </span>
              </div>
            ))}
          </div>
        )
      )}
      <button
        type="button"
        className="btn btn-sm tv-exit tv-chrome"
        aria-label="Exit TV mode"
        onClick={(e) => {
          e.stopPropagation();
          onExit();
        }}
      >
        ✕ Exit
      </button>
      {standalone && !fullscreen && (
        <button
          type="button"
          className="tv-hint tv-chrome"
          onClick={(e) => {
            e.stopPropagation();
            requestFs();
          }}
        >
          tap for fullscreen
        </button>
      )}
    </div>
  );
}

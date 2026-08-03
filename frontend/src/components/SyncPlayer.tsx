/**
 * One camera's synchronized recording player for the multi-camera Timeline.
 *
 * It wraps HlsPlayer and owns the same bounded (~1 h, hour-aligned) VOD-window
 * logic the single-camera Timeline used, but scoped to THIS camera's segments:
 * every explicit seek (`seek.seq` bumps) maps the shared wall-clock playhead to
 * this camera's media time and seeks its <video>, reloading the hour window when
 * the target falls outside it. Cameras with no footage for the window render an
 * empty player. The designated leader reports its playback position back up so
 * the shared playhead follows during play; followers just play their own footage
 * from the last seek (explicit seeks re-align everyone).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type RecordingSegment } from '../lib/api';
import {
  DAY,
  clamp,
  hourWindow,
  localDayStart,
  mediaTimeForWall,
  segsInWindow,
  wallForMediaTime,
  type Window,
} from '../lib/timelineTime';
import HlsPlayer from './HlsPlayer';
import { titleCase } from '../lib/format';

export interface SeekRequest {
  /** Wall-clock target (epoch seconds). */
  t: number;
  /** Bumped on every explicit user seek; drives the seek effect. */
  seq: number;
  /** Whether to auto-play once the seek lands. */
  play: boolean;
}

interface SyncPlayerProps {
  camera: string;
  friendlyName: string;
  /** This camera's segments for the loaded day. */
  segments: RecordingSegment[];
  /** Loaded day (local date string), for day-bounds + hour alignment. */
  date: string;
  seek: SeekRequest;
  /** Shared transport state — every attached player plays/pauses together. */
  playing: boolean;
  /** Shared playback rate, applied to this player's video. */
  rate: number;
  /** Shared mute state, applied to this player's video. */
  muted: boolean;
  /** Only the leader reports playback position back to the shared playhead. */
  isLeader?: boolean;
  onFollow?: (t: number) => void;
  /** Per-lane control to detach this player from the grid. */
  onRemove?: () => void;
}

export default function SyncPlayer({
  camera,
  friendlyName,
  segments,
  date,
  seek,
  playing,
  rate,
  muted,
  isLeader,
  onFollow,
  onRemove,
}: SyncPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pendingSeekRef = useRef<number | null>(null);
  const autoPlayRef = useRef(false);
  const lastSeqRef = useRef<number>(-1);
  const [playerWin, setPlayerWin] = useState<Window | null>(null);

  const onFollowRef = useRef(onFollow);
  onFollowRef.current = onFollow;

  // Latest shared-state snapshots for callbacks that must not re-create on every
  // change (handleReady is captured by HlsPlayer; the seek effect reads these).
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const mutedRef = useRef(muted);
  mutedRef.current = muted;

  const dayStart = date ? localDayStart(date) : 0;
  const dayEnd = dayStart + DAY;

  // ---- react to explicit seeks (scrub end / marker / keyboard / new day) ----
  useEffect(() => {
    if (seek.seq === lastSeqRef.current) return;
    lastSeqRef.current = seek.seq;
    if (!date) return;
    const at = clamp(seek.t, dayStart, dayEnd - 1);
    autoPlayRef.current = seek.play;
    const inWindow =
      playerWin &&
      at >= playerWin.start &&
      at < playerWin.end &&
      segsInWindow(segments, playerWin).length > 0;
    if (inWindow && playerWin) {
      const v = videoRef.current;
      if (v) {
        const mt = mediaTimeForWall(segments, at, playerWin);
        if (Number.isFinite(mt)) v.currentTime = mt;
        if (autoPlayRef.current) {
          void v.play().catch(() => {});
          autoPlayRef.current = false;
        }
      }
    } else {
      const win = hourWindow(at, dayStart, dayEnd);
      const covered = segsInWindow(segments, win).length > 0;
      if (covered) {
        pendingSeekRef.current = at;
        setPlayerWin(win);
      } else {
        // No footage in the target hour — empty player, nothing to seek/play.
        pendingSeekRef.current = null;
        autoPlayRef.current = false;
        setPlayerWin(null);
      }
    }
  }, [seek, date, dayStart, dayEnd, segments, playerWin]);

  // ---- playlist src for the loaded window (null when the window has no footage) ----
  const playlistSrc = useMemo(() => {
    if (!camera || !playerWin) return null;
    if (segsInWindow(segments, playerWin).length === 0) return null;
    return api.recordingPlaylistUrl(camera, playerWin.start, playerWin.end);
  }, [camera, playerWin, segments]);

  // ---- HlsPlayer seekable: apply the pending seek + adopt shared state ----
  // A window (re)load resets the element's playbackRate/muted, so re-apply the
  // shared transport state here, then resume if the group is playing (or an
  // explicit seek/auto-advance asked to autoplay).
  const handleReady = useCallback(() => {
    const v = videoRef.current;
    if (!v || !playerWin) return;
    const t = pendingSeekRef.current;
    if (t != null) {
      const mt = mediaTimeForWall(segments, t, playerWin);
      if (Number.isFinite(mt)) v.currentTime = mt;
      pendingSeekRef.current = null;
    }
    v.playbackRate = rateRef.current;
    v.muted = mutedRef.current;
    if (autoPlayRef.current || playingRef.current) {
      void v.play().catch(() => {});
      autoPlayRef.current = false;
    }
  }, [playerWin, segments]);

  // ---- advance to the next covered hour when the window finishes ----
  const handleEnded = useCallback(() => {
    if (!playerWin || !date) return;
    const next = segments.find((s) => s.start >= playerWin.end);
    if (!next || next.start >= dayEnd) return;
    autoPlayRef.current = true;
    pendingSeekRef.current = next.start;
    setPlayerWin(hourWindow(next.start, dayStart, dayEnd));
  }, [playerWin, segments, date, dayStart, dayEnd]);

  // ---- leader follows playback → shared playhead (guarded against fresh seeks) ----
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playerWin || !isLeader) return;
    const onTime = () => {
      if (pendingSeekRef.current != null) return;
      onFollowRef.current?.(wallForMediaTime(segments, v.currentTime, playerWin));
    };
    v.addEventListener('timeupdate', onTime);
    return () => v.removeEventListener('timeupdate', onTime);
  }, [playerWin, segments, isLeader]);

  // ---- shared play/pause: every attached player toggles together ----
  // Re-runs on (re)attach (playerWin change) so a freshly-mounted or reloaded
  // player adopts the group's playing state instead of autoplaying on its own.
  // If the media isn't seekable yet, play() rejects harmlessly and handleReady
  // resumes once ready.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playerWin) return; // detached/empty player → pause is a no-op
    if (playing) void v.play().catch(() => {});
    else v.pause();
  }, [playing, playerWin]);

  // ---- shared playback rate + mute, applied on change and on (re)attach ----
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
    v.muted = muted;
  }, [rate, muted, playerWin]);

  return (
    <div className="sp-tile">
      {/* controls={false}: the group is driven by the shared transport bar —
          native per-tile play/pause/seek would desync this tile from the rest. */}
      <HlsPlayer
        src={playlistSrc}
        videoRef={videoRef}
        onReady={handleReady}
        onEnded={handleEnded}
        controls={false}
      />
      <div className="sp-tile-bar">
        <span className="sp-tile-name">{friendlyName || titleCase(camera)}</span>
        {onRemove && (
          <button
            type="button"
            className="btn btn-sm sp-tile-remove"
            onClick={onRemove}
            aria-label={`Remove ${friendlyName || titleCase(camera)} from view`}
            title="Remove from view"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

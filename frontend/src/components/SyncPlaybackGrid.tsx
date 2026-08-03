/**
 * Responsive grid of synchronized recording players for the on-screen cameras.
 *
 * Only the cameras passed here get a live <video> (the parent caps this at
 * MAX_SYNC_PLAYERS and lazily attaches/detaches so off-screen hls.js instances
 * are released). One tile is the leader (parent-chosen; defaults to the first):
 * it reports its playback position up so the shared playhead follows during
 * play. A shared SeekRequest re-aligns every tile on an explicit seek.
 */
import SyncPlayer, { type SeekRequest } from './SyncPlayer';
import type { RecordingSegment } from '../lib/api';

export interface GridCamera {
  camera: string;
  friendlyName: string;
  segments: RecordingSegment[];
}

interface SyncPlaybackGridProps {
  cameras: GridCamera[];
  date: string;
  seek: SeekRequest;
  /** Shared transport state, fanned out to every player so they stay in sync. */
  playing: boolean;
  rate: number;
  muted: boolean;
  /**
   * Which camera reports its position to the shared playhead (the parent picks
   * one with footage at the playhead). Falls back to the first tile.
   */
  leaderCamera?: string;
  onFollow: (t: number) => void;
  onRemove?: (camera: string) => void;
}

export default function SyncPlaybackGrid({
  cameras,
  date,
  seek,
  playing,
  rate,
  muted,
  leaderCamera,
  onFollow,
  onRemove,
}: SyncPlaybackGridProps) {
  if (cameras.length === 0) return null;
  // Column count: 1 → 1, 2 → 2, 3+ → 2 columns (rows wrap). Capped small since
  // MAX_SYNC_PLAYERS is small; keeps tiles at a usable size on 380 px too.
  const cols = cameras.length === 1 ? 1 : 2;
  return (
    <div className="sp-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {cameras.map((c, i) => (
        <SyncPlayer
          key={c.camera}
          camera={c.camera}
          friendlyName={c.friendlyName}
          segments={c.segments}
          date={date}
          seek={seek}
          playing={playing}
          rate={rate}
          muted={muted}
          isLeader={leaderCamera != null ? c.camera === leaderCamera : i === 0}
          onFollow={onFollow}
          onRemove={onRemove ? () => onRemove(c.camera) : undefined}
        />
      ))}
    </div>
  );
}

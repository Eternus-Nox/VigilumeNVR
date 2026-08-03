/**
 * Transport controls for the multi-camera Timeline, mounted directly under the
 * unified scrub bar (TimelineLanes). This bar renders NO <video> of its own — it
 * only reflects and mutates the shared playback state that Timeline owns and
 * pushes down to every on-screen SyncPlayer, so play/pause, ±10 s skip, playback
 * speed, and mute all apply to every synchronized player at once. The time
 * readout mirrors the single shared playhead (advances only while playing).
 */
const RATES = [0.5, 1, 2, 4] as const;

interface TimelineTransportProps {
  /** Whether the shared players are playing (drives the primary toggle icon). */
  playing: boolean;
  onTogglePlay: () => void;
  /** Wall-clock HH:MM:SS of the shared playhead (formatted by the parent). */
  playheadLabel: string;
  /** Seek every player by a signed offset (seconds) via the shared seek path. */
  onSkip: (deltaSeconds: number) => void;
  /** Current shared playback rate (applied to every player's video). */
  rate: number;
  onRateChange: (rate: number) => void;
  /** Shared mute state (applied to every player's video). */
  muted: boolean;
  onToggleMute: () => void;
  /** Optional: seek to the newest recorded coverage across the selected set. */
  onJumpToNewest?: () => void;
  /** Whether any newest-coverage target exists (else the button is disabled). */
  canJumpToNewest?: boolean;
  /** Disable transport when nothing is playable (no coverage on the day). */
  disabled?: boolean;
}

export default function TimelineTransport({
  playing,
  onTogglePlay,
  playheadLabel,
  onSkip,
  rate,
  onRateChange,
  muted,
  onToggleMute,
  onJumpToNewest,
  canJumpToNewest = false,
  disabled = false,
}: TimelineTransportProps) {
  return (
    <div className="tl-transport" role="group" aria-label="Playback controls">
      <button
        type="button"
        className="btn btn-primary btn-sm tl-transport-play"
        onClick={onTogglePlay}
        aria-pressed={playing}
        aria-label={playing ? 'Pause' : 'Play'}
        disabled={disabled}
      >
        {playing ? (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
          </svg>
        ) : (
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      <div className="tl-transport-skip">
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onSkip(-10)}
          aria-label="Skip back 10 seconds"
          title="Back 10s"
          disabled={disabled}
        >
          « 10s
        </button>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onSkip(10)}
          aria-label="Skip forward 10 seconds"
          title="Forward 10s"
          disabled={disabled}
        >
          10s »
        </button>
      </div>

      <span className="tl-transport-time" aria-label="Playhead time">
        {playheadLabel}
      </span>

      <div className="tl-transport-speed" role="group" aria-label="Playback speed">
        {RATES.map((r) => (
          <button
            key={r}
            type="button"
            className={`seg-btn ${rate === r ? 'seg-on' : ''}`}
            onClick={() => onRateChange(r)}
            aria-pressed={rate === r}
            disabled={disabled}
          >
            {r}×
          </button>
        ))}
      </div>

      <button
        type="button"
        className="btn btn-sm tl-transport-mute"
        onClick={onToggleMute}
        aria-pressed={!muted}
        aria-label={muted ? 'Unmute' : 'Mute'}
        title={muted ? 'Unmute' : 'Mute'}
      >
        <span aria-hidden="true">{muted ? '🔇' : '🔊'}</span>
      </button>

      {onJumpToNewest && (
        <button
          type="button"
          className="btn btn-sm tl-transport-newest"
          onClick={onJumpToNewest}
          disabled={disabled || !canJumpToNewest}
          title="Jump to newest footage"
        >
          Newest
        </button>
      )}
    </div>
  );
}

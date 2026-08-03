/**
 * Concise status badge for a camera's *effective* detection-gating mode — i.e.
 * WHERE object detection runs for it. Derived from `Camera.detect_mode`
 * (missing = "always"):
 *   - always         → "Server"           (continuous server-side GPU)
 *   - camera_ai      → "Camera-triggered"  (server GPU gated to on-camera AI)
 *   - camera_ai_only → "On-camera"         (no server GPU; camera-AI events)
 * When `aiActive` is true (the camera object's live `ai_active`) a pulsing
 * "AI active" dot is appended — for camera_ai that means the server GPU is
 * firing right now, for camera_ai_only that the camera AI is currently
 * flagging. Shared by the Cameras settings tab and the dashboard tiles so the
 * label/colour stay identical everywhere.
 */
import type { DetectMode } from '../lib/api';

const LOCATION: Record<DetectMode, string> = {
  always: 'Server',
  camera_ai: 'Camera-triggered',
  camera_ai_only: 'On-camera',
};

const DESC: Record<DetectMode, string> = {
  always: 'Detection runs continuously on the server GPU',
  camera_ai: 'Server detection runs only when this camera’s on-board AI fires',
  camera_ai_only: 'No server detection — events come from the camera’s own AI',
};

/** Effective detection-location label for a (possibly missing) mode. */
export function detectModeLocation(mode: DetectMode | undefined): string {
  return LOCATION[mode ?? 'always'];
}

export default function DetectModeBadge({
  mode,
  aiActive,
  className,
}: {
  mode: DetectMode | undefined;
  /** Live camera `ai_active`; omit/undefined hides the pulsing dot. */
  aiActive?: boolean;
  className?: string;
}) {
  const m = mode ?? 'always';
  return (
    <span
      className={`pill detect-badge detect-badge-${m}${className ? ` ${className}` : ''}`}
      title={DESC[m]}
    >
      <span className="detect-badge-label">{LOCATION[m]}</span>
      {aiActive && (
        <span className="detect-badge-live" title="On-camera AI active now">
          <span className="detect-badge-live-dot" aria-hidden="true" />
          AI active
        </span>
      )}
    </span>
  );
}

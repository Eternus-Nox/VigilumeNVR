/**
 * Timeline camera selector: a straightforward multi-select of recording cameras
 * capped at `max` (default first-camera, up to 4 in sync). Each camera is a
 * toggle chip; once `max` are selected the remaining chips are disabled with a
 * hint so a 5th can't be added. The chosen set drives which cameras get a lane +
 * synchronized player. Selection persistence lives in the parent; this is a
 * controlled component.
 */
import type { RecordingCamera } from '../lib/api';
import { titleCase } from '../lib/format';

interface TimelineSelectorProps {
  recCameras: RecordingCamera[];
  /** Currently-selected camera names (ordered by the recordings list). */
  selected: string[];
  /** Max cameras selectable at once. */
  max: number;
  /** Toggle a camera in/out of the selection (parent enforces the cap). */
  onToggle: (camera: string) => void;
}

export default function TimelineSelector({
  recCameras,
  selected,
  max,
  onToggle,
}: TimelineSelectorProps) {
  const selectedSet = new Set(selected);
  const atMax = selected.length >= max;

  return (
    <div className="tl-selector" role="group" aria-label="Cameras to scrub">
      {recCameras.map((c) => {
        const on = selectedSet.has(c.camera);
        const blocked = !on && atMax; // can't add a 5th until one is removed
        return (
          <button
            key={c.camera}
            type="button"
            className={`tl-cam-chip${on ? ' tl-cam-chip-on' : ''}`}
            onClick={() => onToggle(c.camera)}
            aria-pressed={on}
            disabled={blocked}
            title={
              blocked
                ? `Up to ${max} cameras — remove one first`
                : on
                  ? 'Remove from timeline'
                  : 'Add to timeline'
            }
          >
            {c.friendly_name || titleCase(c.camera)}
            {!c.has_recordings && <span className="tl-cam-chip-note"> · no footage</span>}
          </button>
        );
      })}
      <span className="muted tl-selector-count">
        {selected.length}/{max}
      </span>
    </div>
  );
}

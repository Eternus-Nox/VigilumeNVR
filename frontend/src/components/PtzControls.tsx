/**
 * PTZ dome controls (capability-gated on `caps.ptz` — the IP3M-941B). A
 * tap-to-step directional pad (click a compass arrow to nudge the camera one
 * small increment in that direction) plus a 1–8 speed slider (step magnitude)
 * and three preset slots (click a slot to recall it; Set stores the current
 * position, Clear deletes it). Every action is a POST to
 * /api/cameras/{name}/ptz (api.ptz); failures surface via `onError`.
 */
import { useCallback, useState } from 'react';
import { api, type PtzDirection } from '../lib/api';

/** Compass layout: grid-area name === PtzDirection, so CSS places each arrow. */
const DIRECTIONS: { dir: PtzDirection; glyph: string; label: string }[] = [
  { dir: 'upleft', glyph: '↖', label: 'up-left' },
  { dir: 'up', glyph: '↑', label: 'up' },
  { dir: 'upright', glyph: '↗', label: 'up-right' },
  { dir: 'left', glyph: '←', label: 'left' },
  { dir: 'right', glyph: '→', label: 'right' },
  { dir: 'downleft', glyph: '↙', label: 'down-left' },
  { dir: 'down', glyph: '↓', label: 'down' },
  { dir: 'downright', glyph: '↘', label: 'down-right' },
];

const PRESETS = [1, 2, 3] as const;

interface PtzControlsProps {
  camera: string;
  onError: (title: string, body: string) => void;
}

export default function PtzControls({ camera, onError }: PtzControlsProps) {
  const [speed, setSpeed] = useState(4);

  const fail = useCallback(
    (title: string, e: unknown) => onError(title, e instanceof Error ? e.message : 'Request failed'),
    [onError],
  );

  // A single tap = one small step in `dir`. No hold-to-move / release-stop: the
  // camera nudges once per click, so the pad can never run away.
  const step = useCallback(
    (dir: PtzDirection) => {
      void api
        .ptz(camera, { action: 'step', direction: dir, speed })
        .catch((e) => fail('PTZ move failed', e));
    },
    [camera, speed, fail],
  );

  const preset = useCallback(
    (action: 'preset_goto' | 'preset_set' | 'preset_clear', index: number) => {
      const titles = {
        preset_goto: 'Go to preset failed',
        preset_set: 'Save preset failed',
        preset_clear: 'Clear preset failed',
      };
      void api.ptz(camera, { action, index }).catch((e) => fail(titles[action], e));
    },
    [camera, fail],
  );

  return (
    <div className="ptz">
      <div className="ptz-pad" role="group" aria-label="Pan and tilt">
        {DIRECTIONS.map(({ dir, glyph, label }) => (
          <button
            key={dir}
            type="button"
            className="ptz-dir"
            style={{ gridArea: dir }}
            aria-label={`Step ${label}`}
            onClick={() => step(dir)}
          >
            {glyph}
          </button>
        ))}
        <span className="ptz-hub" aria-hidden="true" />
      </div>

      <label className="ptz-speed">
        <span className="control-label">Speed {speed}</span>
        <input
          type="range"
          min={1}
          max={8}
          step={1}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        />
      </label>

      <div className="ptz-presets" role="group" aria-label="Presets">
        {PRESETS.map((i) => (
          <div className="ptz-preset" key={i}>
            <button
              type="button"
              className="ptz-preset-go"
              onClick={() => preset('preset_goto', i)}
              title={`Recall preset ${i}`}
            >
              P{i}
            </button>
            <div className="ptz-preset-actions">
              <button type="button" onClick={() => preset('preset_set', i)} title={`Save current position to preset ${i}`}>
                Set
              </button>
              <button type="button" onClick={() => preset('preset_clear', i)} title={`Clear preset ${i}`}>
                Clear
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

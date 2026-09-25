/**
 * Why plates are or are not being read — per camera, from the plate pass's
 * own counters (`GET /api/recognition/status` → `plates`).
 *
 * The counters are stages a plate has to get through: a look at the vehicle,
 * a plate-shaped region, a region big enough to read, a clear read, and reads
 * that agree. The first stage that stays at zero while the one before it
 * climbs is the answer, and each has a different fix — so this turns the
 * numbers into one sentence per camera saying which it is, with the numbers
 * underneath for anyone who wants them.
 */
import type {
  Camera,
  PlateCameraStats,
  PlateSnapshotHealth,
  RecognitionStatus,
} from '../../lib/api';
import { formatRelative, titleCase } from '../../lib/format';

type Tone = 'ok' | 'warn' | 'idle';

/** The plate reader's floor: a strip narrower than this is not read at all. */
const PLATE_MIN_PX = 64;

export function plateVerdict(
  c: PlateCameraStats | undefined,
  snap: PlateSnapshotHealth | undefined,
  hiresOn: boolean,
): { tone: Tone; text: string } {
  if (!c || c.passes === 0) {
    return { tone: 'idle', text: 'No vehicle seen here since the server started.' };
  }
  if (c.votes_stored > 0) {
    return {
      tone: 'ok',
      text: `Reading plates — last ${c.last_plate} ${formatRelative(c.last_plate_at)}.`,
    };
  }
  if (hiresOn && snap?.no_gain) {
    return { tone: 'warn', text: `Full-resolution snapshots don't help here: ${snap.no_gain}` };
  }
  if (hiresOn && snap && snap.ok === 0 && snap.failed > 0) {
    return {
      tone: 'warn',
      text: `Couldn't get a full-resolution snapshot from this camera: ${snap.last_error}.`,
    };
  }
  if (!hiresOn && c.reads === 0) {
    return {
      tone: 'warn',
      text:
        c.median_strip_px > 0
          ? `Plates on the detection stream are about ${c.median_strip_px} px wide, and ` +
            `${PLATE_MIN_PX} is the minimum to read one. Turn on “Read plates from ` +
            `full-resolution snapshots” under Detection → Faces & plates.`
          : 'The detection stream is too small to find a plate on. Turn on “Read ' +
            'plates from full-resolution snapshots” under Detection → Faces & plates.',
    };
  }
  if (hiresOn && c.hires_frames > 2 && c.hires_lost >= c.hires_frames * 0.8) {
    return {
      tone: 'warn',
      text:
        'Vehicles have usually moved on by the time the snapshot arrives, so they ' +
        "can't be found in it. A plate area drawn where cars slow down or stop helps.",
    };
  }
  if (c.reads > 0 && c.votes_discarded > 0) {
    return {
      tone: 'warn',
      text:
        "Plates are being read, but the reads disagree, so no answer was stored " +
        '(a wrong plate is worse than none). Usually glare or motion blur — a plate ' +
        'area where cars slow down helps.',
    };
  }
  if (c.regions + c.too_small === 0 && c.hires_frames > 0) {
    return {
      tone: 'warn',
      text:
        'No plate-shaped region found on the vehicles yet — the plate may be at too ' +
        'steep an angle, or out of view from where this camera sits.',
    };
  }
  if (c.regions > 0 && c.reads === 0 && c.rejected_reads > 0) {
    return {
      tone: 'warn',
      text: "Plate-shaped regions were found, but none read clearly enough to use yet.",
    };
  }
  return { tone: 'idle', text: 'Watching — no plate read yet.' };
}

function counters(c: PlateCameraStats, snap: PlateSnapshotHealth | undefined): string {
  const parts = [
    `${c.passes} look${c.passes === 1 ? '' : 's'}`,
    `${c.hires_frames}/${c.hires_requested} snapshots`,
    `${c.reads} read${c.reads === 1 ? '' : 's'}${c.hires_reads ? ` (${c.hires_reads} full-res)` : ''}`,
    `${c.votes_stored} plate${c.votes_stored === 1 ? '' : 's'} stored`,
  ];
  if (c.votes_discarded) parts.push(`${c.votes_discarded} discarded`);
  if (snap?.resolution) {
    parts.push(`camera snapshot ${snap.resolution}${snap.latency_ms ? `, ~${Math.round(snap.latency_ms)} ms` : ''}`);
  }
  if (snap?.backing_off_s) parts.push(`paused ${Math.ceil(snap.backing_off_s / 60)} min after failures`);
  return parts.join(' · ');
}

export default function PlateDiagnostics({
  plates,
  cameras,
  onRefresh,
  refreshing,
}: {
  plates: NonNullable<RecognitionStatus['plates']>;
  cameras: Camera[];
  onRefresh: () => void;
  refreshing: boolean;
}) {
  const hiresOn = plates.hires ?? false;
  const reading = cameras.filter((c) => c.plate_recognition ?? true);
  // The cameras reading plates NOW. Counters for a camera since deleted or
  // switched off stay in the backend's report until restart; listing them
  // here would be a row for something that is no longer happening.
  const names = new Set(
    cameras.length > 0 ? reading.map((c) => c.name) : Object.keys(plates.cameras ?? {}),
  );
  const label = (name: string) => {
    const cam = cameras.find((c) => c.name === name);
    return cam?.friendly_name || titleCase(name);
  };

  return (
    <section className="card">
      <h2>Plate reading by camera</h2>
      <p className="muted small">
        Counted since the server last started, for each camera with plate reading on.
        Full-resolution snapshots are{' '}
        <strong>{hiresOn ? 'on' : 'off'}</strong>
        {hiresOn
          ? ' — each tracked vehicle is also read from the camera’s full-resolution picture.'
          : ' — plates are read from the small detection stream only, where they are usually too small.'}
      </p>
      {names.size === 0 ? (
        <p className="control-hint">No camera has plate reading on.</p>
      ) : (
        <ul className="plate-diag-list">
          {[...names].sort().map((name) => {
            const c = plates.cameras?.[name];
            const snap = plates.snapshots?.[name];
            const v = plateVerdict(c, snap, hiresOn);
            return (
              <li key={name} className={`plate-diag plate-diag-${v.tone}`}>
                <strong>{label(name)}</strong>
                <span>{v.text}</span>
                {c && c.passes > 0 && (
                  <span className="control-hint mono">{counters(c, snap)}</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <button type="button" className="btn btn-sm" disabled={refreshing} onClick={onRefresh}>
        {refreshing ? 'Refreshing…' : 'Refresh'}
      </button>
    </section>
  );
}

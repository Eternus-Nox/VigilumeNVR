/**
 * Camera MAIN-stream quality: resolution, codec, keyframe interval, bitrate
 * (backend amcrest/encode.py + stream_profiles.py).
 *
 * Two pieces:
 *  - `MainStreamCard` (Settings → Cameras): the profile EVERY camera follows,
 *    applied to all of them at once, with what each camera actually did.
 *  - `CameraMainStreamPanel` (a camera's edit form): that camera's stream as it
 *    is now, read live, and either "follow all cameras" or its own profile.
 *
 * Both apply IMMEDIATELY and report the camera's read-back — not part of the
 * settings Save button. A camera answering OK and keeping the old value is a
 * real thing these cameras do, so "applied" here means "the camera now says so".
 * The backend re-applies on reconnect and every 30 min, so a camera reset or a
 * change made in its own web page drifts back.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  api,
  type CameraMainStreamDetail,
  type CameraMainStreamRow,
  type MainStreamProfile,
  type MainStreamResult,
} from '../lib/api';
import { useAppState } from '../state/AppState';

const KEEP: MainStreamProfile = { resolution: 'keep', codec: 'keep', keyframe_s: null, bitrate_kbps: null };

/** What the hint under the card recommends: decodes ~4-5x cheaper than 4K
 *  H.265, plays in live view everywhere, shorter keyframe wait. */
const RECOMMENDED: MainStreamProfile = { resolution: '1080p', codec: 'h264', keyframe_s: 1, bitrate_kbps: 4096 };

const RESOLUTION_OPTIONS: [string, string][] = [
  ['keep', 'Leave as the camera has it'],
  ['720p', 'Up to 720p'],
  ['1080p', 'Up to 1080p'],
  ['1440p', 'Up to 1440p'],
  ['4k', 'Up to 4K'],
  ['max', 'Camera maximum'],
];
const CODEC_OPTIONS: [MainStreamProfile['codec'], string][] = [
  ['keep', 'Leave as the camera has it'],
  ['h264', 'H.264 (plays everywhere, lighter to decode)'],
  ['h265', 'H.265 (smaller files, heavier to decode)'],
];
const KEYFRAME_OPTIONS: [string, string][] = [
  ['', 'Leave as the camera has it'],
  ['1', 'Every 1 second'],
  ['2', 'Every 2 seconds'],
  ['4', 'Every 4 seconds'],
];

function sameProfile(a: MainStreamProfile, b: MainStreamProfile): boolean {
  return a.resolution === b.resolution && a.codec === b.codec
    && a.keyframe_s === b.keyframe_s && a.bitrate_kbps === b.bitrate_kbps;
}

function describe(p: MainStreamProfile): string {
  const parts: string[] = [];
  if (p.resolution !== 'keep') {
    parts.push(RESOLUTION_OPTIONS.find(([v]) => v === p.resolution)?.[1] ?? p.resolution);
  }
  if (p.codec !== 'keep') parts.push(p.codec === 'h264' ? 'H.264' : 'H.265');
  if (p.keyframe_s != null) parts.push(`keyframe every ${p.keyframe_s} s`);
  if (p.bitrate_kbps != null) parts.push(`${p.bitrate_kbps} kbps`);
  return parts.length ? parts.join(' · ') : 'unchanged';
}

function ProfileFields({
  value,
  onChange,
  disabled,
  resolutions,
}: {
  value: MainStreamProfile;
  onChange: (p: MainStreamProfile) => void;
  disabled?: boolean;
  /** One camera's own list: offered as exact sizes after the ceilings. */
  resolutions?: { label: string; width: number; height: number }[];
}) {
  return (
    <div className="form-stack">
      <label>
        Resolution
        <select
          value={value.resolution}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, resolution: e.target.value })}
        >
          {RESOLUTION_OPTIONS.map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
          {resolutions && resolutions.length > 0 && (
            <optgroup label="This camera's sizes">
              {resolutions.map((r) => (
                <option key={r.label} value={`${r.width}x${r.height}`}>
                  {r.width}×{r.height}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <span className="control-hint">
          &ldquo;Up to&rdquo; picks each camera&rsquo;s largest size at or under that height,
          keeping the picture&rsquo;s shape where the camera offers it.
        </span>
      </label>
      <label>
        Codec
        <select
          value={value.codec}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, codec: e.target.value as MainStreamProfile['codec'] })}
        >
          {CODEC_OPTIONS.map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </select>
      </label>
      <label>
        Keyframe interval
        <select
          value={value.keyframe_s == null ? '' : String(value.keyframe_s)}
          disabled={disabled}
          onChange={(e) =>
            onChange({ ...value, keyframe_s: e.target.value === '' ? null : Number(e.target.value) })
          }
        >
          {KEYFRAME_OPTIONS.map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </select>
        <span className="control-hint">
          Live view and face/plate reads can only start on a keyframe; shorter starts sooner
          and makes files a little larger.
        </span>
      </label>
      <label>
        Bitrate (kbps)
        <input
          type="number"
          min={256}
          max={20480}
          step={256}
          placeholder="Leave as the camera has it"
          value={value.bitrate_kbps ?? ''}
          disabled={disabled}
          onChange={(e) =>
            onChange({
              ...value,
              bitrate_kbps: e.target.value === '' ? null : Math.round(Number(e.target.value)),
            })
          }
        />
        <span className="control-hint">
          Lowering the resolution does not lower the bitrate by itself. About 4096 is plenty
          for 1080p H.264; 2048 for H.265.
        </span>
      </label>
    </div>
  );
}

function ResultLine({ r, name }: { r: MainStreamResult; name?: string }) {
  const label = name ?? r.camera;
  if (r.error) {
    return <li><strong>{label}</strong>: <span className="form-error">{r.error}</span></li>;
  }
  if (r.skipped) return <li><strong>{label}</strong>: nothing to change</li>;
  const problems = [...(r.rejected ?? []), ...(r.not_applied ?? [])];
  const changed = (r.changed ?? []).filter((c) => c.startsWith('main ') && !c.startsWith('main #'));
  return (
    <li>
      <strong>{label}</strong>:{' '}
      {changed.length ? changed.map((c) => c.replace(/^main /, '')).join('; ') : 'already set'}
      {problems.length > 0 && (
        <span className="form-error"> — not applied: {problems.join('; ')}</span>
      )}
      {(r.notes ?? []).length > 0 && <span className="muted"> ({r.notes!.join('; ')})</span>}
    </li>
  );
}

export default function MainStreamCard() {
  const { pushToast, cameras } = useAppState();
  const [profile, setProfile] = useState<MainStreamProfile | null>(null);
  const [draft, setDraft] = useState<MainStreamProfile>(KEEP);
  const [rows, setRows] = useState<CameraMainStreamRow[]>([]);
  const [resetCameras, setResetCameras] = useState(false);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<MainStreamResult[] | null>(null);
  const [unsupported, setUnsupported] = useState(false);

  const load = useCallback(async () => {
    try {
      const o = await api.mainStreamOverview();
      setProfile(o.profile);
      setDraft(o.profile);
      setRows(o.cameras);
    } catch {
      setUnsupported(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (unsupported || !profile) return null;

  const friendly = (name: string) => (cameras ?? []).find((c) => c.name === name)?.friendly_name ?? name;
  const own = rows.filter((r) => !r.inherited);

  const run = async (fn: () => Promise<{ results: MainStreamResult[] }>) => {
    setBusy(true);
    setResults(null);
    try {
      const res = await fn();
      setResults(res.results);
      const failed = res.results.filter((r) => !r.ok && !r.skipped).length;
      pushToast({
        kind: failed ? 'error' : 'info',
        title: failed ? `${failed} camera${failed === 1 ? '' : 's'} not fully applied` : 'Cameras updated',
        body: failed ? 'See the results under Camera video quality.' : '',
      });
      await load();
    } catch (e) {
      pushToast({ kind: 'error', title: 'Could not apply', body: e instanceof Error ? e.message : '' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <details className="privacy-details">
        <summary>
          <span className="privacy-summary-title">Camera video quality</span>
          <span className="privacy-summary-badge">All cameras: {describe(profile)}</span>
        </summary>
        <p className="muted small">
          What every camera&rsquo;s <strong>main stream</strong> — the one that is recorded,
          read for faces and plates, and shown in fullscreen live view — is set to. Written to
          the cameras directly, kept that way on reconnect and every 30 minutes. A camera can
          have its own setting instead: edit the camera below.
        </p>
        <p className="muted small">
          Recommended for lighter decoding and reliable live view:{' '}
          <button type="button" className="btn btn-sm" disabled={busy} onClick={() => setDraft(RECOMMENDED)}>
            1080p · H.264 · 1 s keyframes · 4096 kbps
          </button>
        </p>
        <ProfileFields value={draft} onChange={setDraft} disabled={busy} />
        {own.length > 0 && (
          <label className="row-label">
            <input
              type="checkbox"
              checked={resetCameras}
              disabled={busy}
              onChange={(e) => setResetCameras(e.target.checked)}
            />
            Also apply to the {own.length} camera{own.length === 1 ? '' : 's'} with their own
            setting ({own.map((r) => friendly(r.camera)).join(', ')})
          </label>
        )}
        <p className="muted small">
          Changing resolution or codec restarts that camera&rsquo;s video for a few seconds;
          recording and live view reconnect on their own.
        </p>
        <div className="row-label">
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={busy || (sameProfile(draft, profile) && !resetCameras)}
            onClick={() => void run(() => api.setMainStreamForAll(draft, resetCameras))}
          >
            {busy ? 'Applying…' : 'Apply to all cameras'}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={busy}
            onClick={() => void run(() => api.reapplyMainStreams())}
          >
            Re-check cameras now
          </button>
        </div>
        {results && (
          <ul className="small">
            {results.map((r) => (
              <ResultLine key={r.camera} r={r} name={friendly(r.camera)} />
            ))}
          </ul>
        )}
      </details>
    </section>
  );
}

/** A camera's own main-stream setting, in its edit form. */
export function CameraMainStreamPanel({ name }: { name: string }) {
  const { pushToast } = useAppState();
  const [detail, setDetail] = useState<CameraMainStreamDetail | null>(null);
  const [follow, setFollow] = useState(true);
  const [draft, setDraft] = useState<MainStreamProfile>(KEEP);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<MainStreamResult | null>(null);
  const [unsupported, setUnsupported] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.cameraMainStream(name);
      setDetail(d);
      setFollow(d.inherited);
      setDraft(d.inherited ? d.global : { ...KEEP, ...d.own } as MainStreamProfile);
    } catch {
      setUnsupported(true);
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    void load();
  }, [load]);

  if (unsupported) return null;

  const save = async () => {
    setBusy(true);
    setResult(null);
    try {
      const res = await api.setCameraMainStream(name, follow ? null : draft);
      setResult(res.result);
      pushToast({
        kind: res.result.ok ? 'info' : 'error',
        title: res.result.ok ? 'Camera stream updated' : 'Not fully applied',
        body: res.result.error ?? '',
      });
      await load();
    } catch (e) {
      pushToast({ kind: 'error', title: 'Could not apply', body: e instanceof Error ? e.message : '' });
    } finally {
      setBusy(false);
    }
  };

  const now = detail?.live?.current;
  return (
    <div className="form-span">
      <span className="control-label">Video quality (main stream)</span>
      {loading ? (
        <p className="muted small">Reading the camera…</p>
      ) : (
        <>
          <p className="muted small">
            {now ? (
              <>
                Now: {now.width}×{now.height} · {now.codec_raw ?? '?'} · {now.fps ?? '?'} fps ·
                keyframe every {now.keyframe_s ?? '?'} s · {now.bitrate_kbps ?? '?'} kbps
                {now.bitrate_control ? ` ${now.bitrate_control}` : ''}
              </>
            ) : (
              <>Could not read the camera{detail?.error ? `: ${detail.error}` : ''}.</>
            )}
          </p>
          <label className="row-label">
            <input
              type="radio"
              checked={follow}
              disabled={busy}
              onChange={() => {
                setFollow(true);
                if (detail) setDraft(detail.global);
              }}
            />
            Same as all cameras ({detail ? describe(detail.global) : '…'})
          </label>
          <label className="row-label">
            <input type="radio" checked={!follow} disabled={busy} onChange={() => setFollow(false)} />
            This camera&rsquo;s own setting
          </label>
          {!follow && (
            <ProfileFields
              value={draft}
              onChange={setDraft}
              disabled={busy}
              resolutions={detail?.live?.resolutions}
            />
          )}
          <div className="row-label">
            <button type="button" className="btn btn-sm" disabled={busy} onClick={() => void save()}>
              {busy ? 'Applying…' : 'Apply to this camera'}
            </button>
          </div>
          {result && (
            <ul className="small">
              <ResultLine r={result} name="Result" />
            </ul>
          )}
          <span className="control-hint">
            Applied straight away — separate from the Save button. Use a higher resolution
            on a camera that reads plates from a distance.
          </span>
        </>
      )}
    </div>
  );
}

/**
 * Settings → Cameras → "Privacy Mode": the per-camera / per-group capture kill
 * switch (backend: app/privacy.py, GET/POST /api/privacy).
 *
 * NOT the hardware "Privacy mode (lens off)" toggle in a camera's own device
 * settings — that reconfigures the camera. This stops ALL Vigilume capture for
 * the selected cameras (recording, detection, events, live view, audio,
 * on-camera-AI) while touching nothing on the device itself.
 *
 * NOT batched behind the settings Save button, deliberately. Privacy is a
 * switch you reach for when you want capture to stop NOW; making you press Save
 * afterwards would leave a window where the UI says private and the cameras are
 * still recording. Each toggle applies immediately and the response carries the
 * authoritative resolved set.
 */
import { useCallback, useEffect, useState } from 'react';
import { api, type Camera, type CameraGroup, type PrivacyModeState } from '../lib/api';
import { useAppState } from '../state/AppState';

export default function PrivacyModeCard() {
  const { pushToast } = useAppState();
  const [state, setState] = useState<PrivacyModeState | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [groups, setGroups] = useState<CameraGroup[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [unsupported, setUnsupported] = useState(false);

  const load = useCallback(async () => {
    try {
      const [p, c, g] = await Promise.all([api.privacyMode(), api.cameras(), api.groups()]);
      setState(p);
      setCameras(c);
      setGroups(g);
    } catch {
      // Endpoint absent (backend not yet rebuilt) — hide the card rather than
      // showing a broken control that silently does nothing.
      setUnsupported(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const apply = async (body: { cameras?: string[]; groups?: number[] }, key: string) => {
    setBusy(key);
    try {
      // The response is the authoritative resolved set — adopt it rather than
      // optimistically toggling, so a camera that went private via a GROUP is
      // reflected correctly instead of the UI inventing its own answer.
      setState(await api.setPrivacyMode(body));
      // Camera rows carry `private`, so refresh them for the tile overlays.
      setCameras(await api.cameras());
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Privacy Mode failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      setBusy(null);
    }
  };

  if (unsupported || !state) return null;

  const priv = new Set(state.private_cameras);
  const directly = new Set(state.cameras);
  const groupIds = new Set(state.groups);

  const toggleCamera = (name: string) => {
    const next = new Set(directly);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    void apply({ cameras: [...next] }, `cam:${name}`);
  };

  const toggleGroup = (id: number) => {
    const next = new Set(groupIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    void apply({ groups: [...next] }, `grp:${id}`);
  };

  const clearAll = () => void apply({ cameras: [], groups: [] }, 'clear');

  return (
    <section className="card privacy-card">
      {/* Collapsed by default: this card lists every camera AND every group, so
          expanded it dominates the Cameras tab. It opens DOWNWARD as a normal
          disclosure (<details>, matching the "advanced-section" idiom used
          elsewhere in settings) rather than a modal or a drawer.

          It stays OPEN on its own whenever anything is private. Collapsing a
          capture kill switch must never hide the fact that it is armed — the
          summary below always states the count, and the card force-opens so a
          camera that is not recording can't sit silently behind a closed
          triangle. */}
      <details className="privacy-details" open={state.enabled}>
        <summary>
          <span className="privacy-summary-title">Privacy Mode</span>
          {state.enabled ? (
            <span className="privacy-summary-badge privacy-summary-badge-on">
              {priv.size} camera{priv.size === 1 ? '' : 's'} not recording
            </span>
          ) : (
            <span className="privacy-summary-badge">All cameras recording</span>
          )}
        </summary>
      <p className="muted small">
        Stops <strong>everything</strong> for the selected cameras — no recording, no
        detection, no events or notifications, no live view, no audio. Nothing is
        captured and nothing is deleted; the cameras themselves are not touched or
        reconfigured. Turn it off and capture resumes on its own.
      </p>
      <p className="muted small">
        Applies <strong>immediately</strong> — this is not part of the Save button below.
      </p>

      {state.enabled && (
        <div className="banner banner-warn">
          <span>
            <strong>
              {priv.size} camera{priv.size === 1 ? '' : 's'} currently capturing nothing
            </strong>
            {' — '}
            {[...priv].sort().join(', ')}
          </span>
          <button
            type="button"
            className="btn btn-sm"
            disabled={busy !== null}
            onClick={clearAll}
          >
            {busy === 'clear' ? 'Resuming…' : 'Resume all'}
          </button>
        </div>
      )}

      {groups.length > 0 && (
        <>
          <span className="control-label">Groups</span>
          <div className="form-stack">
            {groups.map((g) => (
              <label className="row-label" key={g.id}>
                <input
                  type="checkbox"
                  checked={groupIds.has(g.id)}
                  disabled={busy !== null}
                  onChange={() => toggleGroup(g.id)}
                />
                {g.name}
                <span className="control-hint">
                  {g.cameras.length} camera{g.cameras.length === 1 ? '' : 's'}
                </span>
              </label>
            ))}
          </div>
        </>
      )}

      <span className="control-label">Cameras</span>
      <div className="form-stack">
        {cameras.map((c) => {
          const viaGroup = priv.has(c.name) && !directly.has(c.name);
          return (
            <label className="row-label" key={c.name}>
              <input
                type="checkbox"
                checked={priv.has(c.name)}
                // A camera private via a GROUP can't be individually un-ticked —
                // that would silently contradict the group. Say so instead of
                // letting the click do nothing.
                disabled={busy !== null || viaGroup}
                onChange={() => toggleCamera(c.name)}
              />
              {c.friendly_name}
              {viaGroup && <span className="control-hint">private via its group</span>}
            </label>
          );
        })}
      </div>
      </details>
    </section>
  );
}

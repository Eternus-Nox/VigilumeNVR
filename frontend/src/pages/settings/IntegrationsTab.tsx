/**
 * Settings → Integrations (admin): Home Assistant over MQTT. Vigilume runs an
 * OUTBOUND MQTT publisher — when enabled it connects to the operator's broker
 * and announces each camera + its detections via Home Assistant's MQTT
 * auto-discovery, so they show up as HA devices/entities automatically (with
 * optional two-way control of camera features from HA).
 *
 * The broker config lives in the global settings document (`settings.mqtt`),
 * saved via the shared PUT /api/settings path. "Test connection" hits
 * POST /api/integrations/mqtt/test with the *draft* config so the operator can
 * verify credentials before saving. Both are admin-only (this tab lives under
 * admin Settings). The backend MQTT support ships in parallel: the settings
 * block is optional (falls back to DEFAULT_MQTT) and the test endpoint's
 * absence is caught and shown inline, so nothing breaks pre-deploy.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  api,
  type ArchiveSettings,
  type ArchiveStatus,
  type MqttSettings,
  type MqttTestStatus,
} from '../../lib/api';
import { useAdoptSaved, type TabProps } from '../Settings';
import SettingsDisclosure from '../../components/SettingsDisclosure';

const DEFAULT_MQTT: MqttSettings = {
  enabled: false,
  host: '',
  port: 1883,
  username: '',
  password: '',
  discovery_prefix: 'homeassistant',
  base_topic: 'vigilume',
};

const DEFAULT_ARCHIVE: ArchiveSettings = {
  enabled: false,
  remote: '',
  hour: 3,
  keep_days: 30,
  include_snapshots: true,
  bwlimit: '',
};

const TEST_LABEL: Record<MqttTestStatus, string> = {
  ok: 'Connected',
  auth_failed: 'Authentication failed',
  unreachable: 'Broker unreachable',
};

/** Inline "Test connection" state machine (distinct from the Save flow). */
type TestState =
  | { kind: 'idle' }
  | { kind: 'testing' }
  | { kind: 'result'; status: MqttTestStatus; detail?: string | null }
  // Endpoint absent (pre-deploy), network error, or an unexpected shape.
  | { kind: 'error'; message: string };

export default function IntegrationsTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft first, so switching tabs and back keeps
  // unsaved edits (one Save button now covers every settings tab).
  // Merge the pending PARTIAL over the full saved slice — SettingsPatch is a
  // deep-partial, so seeding straight from `pending` would drop fields.
  const [draft, setDraft] = useState<MqttSettings>({
    ...(settings.mqtt ?? DEFAULT_MQTT),
    ...(pending.mqtt ?? {}),
  });
  const [test, setTest] = useState<TestState>({ kind: 'idle' });
  const [archive, setArchive] = useState<ArchiveSettings>({
    ...(settings.archive ?? DEFAULT_ARCHIVE),
    ...(pending.archive ?? {}),
  });
  const [archiveStatus, setArchiveStatus] = useState<ArchiveStatus | null>(null);
  const [archiveRunning, setArchiveRunning] = useState(false);
  const [archiveError, setArchiveError] = useState<string | null>(null);

  // Re-sync from the server document (e.g. after a save echoes it back). A
  // backend without MQTT support omits the block — fall back to the defaults.
  useAdoptSaved(settings.mqtt ?? DEFAULT_MQTT, setDraft);
  useAdoptSaved(settings.archive ?? DEFAULT_ARCHIVE, setArchive);

  // Status is READ-ONLY server state, not part of the settings draft, so it is
  // fetched separately and re-fetched after a manual run.
  const loadArchiveStatus = useCallback(async () => {
    try {
      setArchiveStatus(await api.archiveStatus());
    } catch {
      // A backend predating the archive 404s here. Not an error worth showing —
      // the card's own copy already explains what is needed.
      setArchiveStatus(null);
    }
  }, []);
  useEffect(() => {
    void loadArchiveStatus();
  }, [loadArchiveStatus]);

  // Any edit invalidates a previous test result (it was for the old values).
  const patch = (p: Partial<MqttSettings>) => {
    setDraft((d) => ({ ...d, ...p }));
    setTest({ kind: 'idle' });
  };

  // Normalize before it leaves the form: trim the text fields, clamp the port.
  const normalized = (): MqttSettings => ({
    ...draft,
    host: draft.host.trim(),
    username: draft.username.trim(),
    discovery_prefix: draft.discovery_prefix.trim() || DEFAULT_MQTT.discovery_prefix,
    base_topic: draft.base_topic.trim() || DEFAULT_MQTT.base_topic,
    port: Math.min(65535, Math.max(1, Math.floor(draft.port) || DEFAULT_MQTT.port)),
  });

  // Report this tab's slice up on every edit; the shell's single Save button
  // persists it alongside every other tab's pending changes. Normalized here so
  // what is reported is exactly what would be stored.
  useEffect(() => {
    onDraftChange({ mqtt: normalized(), archive: normalizedArchive() });
    // `normalized` is derived from `draft`; re-report whenever the draft moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, archive, onDraftChange]);

  // Same contract as `normalized` above: what is reported up is exactly what
  // would be stored, so the form cannot save something it never showed.
  const normalizedArchive = (): ArchiveSettings => ({
    ...archive,
    remote: archive.remote.trim(),
    bwlimit: archive.bwlimit.trim(),
    hour: Math.min(23, Math.max(0, Math.floor(archive.hour) || 0)),
    keep_days: Math.min(3650, Math.max(0, Math.floor(archive.keep_days) || 0)),
  });

  const runArchiveNow = async () => {
    setArchiveRunning(true);
    setArchiveError(null);
    try {
      const res = await api.runArchive();
      if (!res.ok && res.detail) setArchiveError(res.detail);
      await loadArchiveStatus();
    } catch (e) {
      setArchiveError(e instanceof Error ? e.message : 'Archive run failed');
    } finally {
      setArchiveRunning(false);
    }
  };

  const runTest = async () => {
    setTest({ kind: 'testing' });
    try {
      const res = await api.testMqtt(normalized());
      const known =
        res.status === 'ok' || res.status === 'auth_failed' || res.status === 'unreachable';
      if (known) {
        setTest({ kind: 'result', status: res.status as MqttTestStatus, detail: res.detail ?? null });
      } else if (res.ok) {
        setTest({ kind: 'result', status: 'ok', detail: res.detail ?? null });
      } else {
        // !ok without a status: don't guess auth vs. reachability — surface the
        // backend's reason instead of mislabeling it.
        setTest({ kind: 'error', message: res.detail || 'Connection failed' });
      }
    } catch (e) {
      setTest({ kind: 'error', message: e instanceof Error ? e.message : 'Test failed' });
    }
  };

  const testing = test.kind === 'testing';
  const canTest = draft.host.trim().length > 0 && !testing;

  return (
    <div className="settings-section">
      <SettingsDisclosure
        title="Home Assistant (MQTT)"
        badge={draft.enabled ? draft.host.trim() || 'On' : 'Off'}
        tone={draft.enabled ? 'on' : 'muted'}
      >
        <p className="muted small">
          Vigilume publishes each camera and its detections to your MQTT broker using Home
          Assistant&rsquo;s MQTT auto-discovery, so they appear in Home Assistant automatically
          as devices and entities — with optional two-way control of camera features (IR,
          spotlight, siren). Point this at the same broker Home Assistant uses.
        </p>

        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(e) => patch({ enabled: e.target.checked })}
          />
          Publish to Home Assistant over MQTT
        </label>

        <div className="form-stack">
          <label>
            Broker host
            <input
              value={draft.host}
              onChange={(e) => patch({ host: e.target.value })}
              placeholder="192.168.1.5 or homeassistant.local"
              autoComplete="off"
            />
            <span className="control-hint">Hostname or IP of the MQTT broker — no scheme.</span>
          </label>
          <label>
            Port
            <input
              type="number"
              min={1}
              max={65535}
              step={1}
              value={draft.port}
              onChange={(e) => patch({ port: Number(e.target.value) })}
            />
            <span className="control-hint">MQTT default 1883 (8883 for TLS).</span>
          </label>
          <label>
            Username
            <input
              value={draft.username}
              onChange={(e) => patch({ username: e.target.value })}
              placeholder="optional"
              autoComplete="off"
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={draft.password}
              onChange={(e) => patch({ password: e.target.value })}
              placeholder="optional"
              autoComplete="new-password"
            />
          </label>
          <label>
            Discovery prefix
            <input
              value={draft.discovery_prefix}
              onChange={(e) => patch({ discovery_prefix: e.target.value })}
              placeholder={DEFAULT_MQTT.discovery_prefix}
              autoComplete="off"
            />
            <span className="control-hint">
              Must match Home Assistant&rsquo;s MQTT discovery prefix (default{' '}
              <code>homeassistant</code>).
            </span>
          </label>
          <label>
            Base topic
            <input
              value={draft.base_topic}
              onChange={(e) => patch({ base_topic: e.target.value })}
              placeholder={DEFAULT_MQTT.base_topic}
              autoComplete="off"
            />
            <span className="control-hint">
              Root topic Vigilume publishes its state and command topics under.
            </span>
          </label>
        </div>

        <div className="row-inline wrap" style={{ marginTop: '0.8rem' }}>
          <button type="button" className="btn btn-sm" disabled={!canTest} onClick={() => void runTest()}>
            {testing ? 'Testing…' : 'Test connection'}
          </button>
          {test.kind === 'result' && (
            <span className={`pill ${test.status === 'ok' ? 'pill-ok' : 'pill-down'}`}>
              {TEST_LABEL[test.status]}
            </span>
          )}
          {test.kind === 'result' && test.detail && (
            <span className="muted small">{test.detail}</span>
          )}
          {test.kind === 'error' && <span className="form-error">{test.message}</span>}
        </div>
      </SettingsDisclosure>

      <SettingsDisclosure
        title="Cloud archive (Dropbox, Drive, S3…)"
        badge={archive.enabled ? archive.remote.trim() || 'On' : 'Off'}
        tone={archive.enabled ? 'on' : 'muted'}
      >
        <p className="muted small">
          Copies each finished day of <strong>event clips and snapshots</strong> to cloud
          storage overnight, as one folder per day (<code>2026-08-25/</code>). 24/7 footage is
          never uploaded — it is far too large for any normal connection. This is a backup of
          the evidence, not of everything.
        </p>
        <p className="muted small">
          Set the destination up once on the server with{' '}
          <code>docker compose exec backend rclone config</code>, then put the remote name
          here. Anything rclone supports works: Dropbox, Google Drive, S3, Backblaze.
        </p>

        <label className="row-inline">
          <input
            type="checkbox"
            checked={archive.enabled}
            onChange={(e) => setArchive({ ...archive, enabled: e.target.checked })}
          />
          <span>Upload event media nightly</span>
        </label>

        <div className="form-grid">
          <label>
            Remote
            <input
              type="text"
              value={archive.remote}
              placeholder="dropbox:Vigilume"
              onChange={(e) => setArchive({ ...archive, remote: e.target.value })}
            />
            <span className="control-hint">
              The rclone remote and path, as <code>name:path</code>.
            </span>
          </label>
          <label>
            Run at (hour)
            <input
              type="number"
              min={0}
              max={23}
              value={archive.hour}
              onChange={(e) =>
                setArchive({ ...archive, hour: Math.max(0, Math.min(23, Number(e.target.value) || 0)) })
              }
            />
            <span className="control-hint">
              Local time. Each run uploads the PREVIOUS day, once it is complete.
            </span>
          </label>
          <label>
            Keep in cloud (days)
            <input
              type="number"
              min={0}
              max={3650}
              value={archive.keep_days}
              onChange={(e) =>
                setArchive({
                  ...archive,
                  keep_days: Math.max(0, Math.min(3650, Number(e.target.value) || 0)),
                })
              }
            />
            <span className="control-hint">
              {archive.keep_days === 0
                ? 'Never expires — the archive grows forever.'
                : `Older day folders are deleted from the cloud. Independent of local retention — outliving the local copy is the point.`}
            </span>
          </label>
          <label>
            Upload speed limit
            <input
              type="text"
              value={archive.bwlimit}
              placeholder="unlimited"
              onChange={(e) => setArchive({ ...archive, bwlimit: e.target.value })}
            />
            <span className="control-hint">
              e.g. <code>2M</code>. Worth setting on a thin uplink so the nightly run does not
              starve live view.
            </span>
          </label>
        </div>

        <label className="row-inline">
          <input
            type="checkbox"
            checked={archive.include_snapshots}
            onChange={(e) => setArchive({ ...archive, include_snapshots: e.target.checked })}
          />
          <span>Include event snapshots (small — clips dominate the size)</span>
        </label>

        <div className="row-inline wrap" style={{ marginTop: '0.8rem' }}>
          <button
            type="button"
            className="btn btn-sm"
            disabled={archiveRunning || !archive.enabled || !archive.remote.trim()}
            onClick={() => void runArchiveNow()}
          >
            {archiveRunning ? 'Running…' : 'Run now'}
          </button>
          <span className="muted small">
            Runs the real nightly pass, so a clean result proves the remote works. Save first —
            it uses the SAVED settings, not what is typed above. It can take a while.
          </span>
        </div>
        {archiveError && <div className="form-error">{archiveError}</div>}
        {archiveStatus?.available && (
          <div className="muted small" style={{ marginTop: '0.6rem' }}>
            <div>
              Archived through:{' '}
              <strong>{archiveStatus.last_uploaded_day ?? 'nothing yet'}</strong>
            </div>
            {archiveStatus.last_result?.at && (
              <div>
                Last run {archiveStatus.last_result.at} — uploaded{' '}
                {archiveStatus.last_result.uploaded_days?.join(', ') || 'nothing'}
                {typeof archiveStatus.last_result.files === 'number' &&
                  ` (${archiveStatus.last_result.files} files)`}
                {archiveStatus.last_result.pruned_days?.length
                  ? `, expired ${archiveStatus.last_result.pruned_days.join(', ')}`
                  : ''}
              </div>
            )}
            {archiveStatus.last_result?.errors?.length ? (
              <div className="form-error">{archiveStatus.last_result.errors.join(' · ')}</div>
            ) : null}
          </div>
        )}
      </SettingsDisclosure>

      {/* No Save button here by design — the shell owns the single Save for
          every settings tab. "Test connection" above is separate: it probes the
          broker with the CURRENT form values without persisting anything. */}
    </div>
  );
}

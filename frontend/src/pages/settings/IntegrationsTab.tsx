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
import { useEffect, useState } from 'react';
import { api, type MqttSettings, type MqttTestStatus } from '../../lib/api';
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

  // Re-sync from the server document (e.g. after a save echoes it back). A
  // backend without MQTT support omits the block — fall back to the defaults.
  useAdoptSaved(settings.mqtt ?? DEFAULT_MQTT, setDraft);

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
    onDraftChange({ mqtt: normalized() });
    // `normalized` is derived from `draft`; re-report whenever the draft moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, onDraftChange]);

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

      {/* No Save button here by design — the shell owns the single Save for
          every settings tab. "Test connection" above is separate: it probes the
          broker with the CURRENT form values without persisting anything. */}
    </div>
  );
}

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
  ApiError,
  type ArchiveSettings,
  type ArchiveStatus,
  type MqttSettings,
  type MqttTestStatus,
  type RcloneProvider,
  type RcloneRemote,
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
  // --- rclone remote setup (replaces `rclone config` over SSH) -------------
  const [providers, setProviders] = useState<RcloneProvider[]>([]);
  const [remotes, setRemotes] = useState<RcloneRemote[]>([]);
  // Why cloud storage is unavailable, if it is. Three states, because they need
  // three different actions and collapsing them into one "rebuild" message is
  // how someone rebuilds twice and is no wiser:
  //   null          — available, nothing to say
  //   'missing'     — the route 404s: this backend predates the feature
  //   <rclone text> — the route answered and rclone itself failed; that string
  //                   is rclone's own stderr, which names the real problem
  const [rcloneAvailable, setRcloneAvailable] = useState(true);
  const [rcloneWhy, setRcloneWhy] = useState<string | null>(null);
  const [newType, setNewType] = useState('');
  const [newName, setNewName] = useState('');
  const [newValues, setNewValues] = useState<Record<string, string>>({});
  const [creating, setCreating] = useState(false);
  const [setupMsg, setSetupMsg] = useState<
    { ok: boolean; text: string; hint?: string } | null
  >(null);
  const [busyRemote, setBusyRemote] = useState<string | null>(null);
  // OAuth providers offer two routes. 'browser' finishes the sign-in on this
  // server and needs the operator's own app credentials; 'token' is the older
  // paste-a-blob path, kept because it needs no app registration at all.
  const [authMode, setAuthMode] = useState<'browser' | 'token'>('browser');
  const [redirectUri, setRedirectUri] = useState('');
  // Non-empty when this page's origin cannot be used for browser sign-in at
  // all — providers refuse a plain-http non-localhost redirect URI when it is
  // REGISTERED, so this has to be said before an app is created.
  const [oauthBlocked, setOauthBlocked] = useState('');
  const [signingIn, setSigningIn] = useState(false);

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

  const loadRemotes = useCallback(async () => {
    try {
      const res = await api.rcloneRemotes();
      setRcloneAvailable(res.available);
      // The endpoint answered, so the code IS deployed. If it still says
      // unavailable, the reason is rclone's, and rclone already said what it
      // was — show that instead of guessing on its behalf.
      setRcloneWhy(res.available ? null : res.detail || 'rclone reported no reason.');
      setRemotes(res.remotes);
    } catch (err) {
      // A 404 is the only error that really means "this build predates the
      // feature". Anything else (500, a proxy error, a dropped connection) is
      // a live backend failing, and telling that operator to rebuild would send
      // them down the wrong path entirely.
      const missing = err instanceof ApiError && err.status === 404;
      setRcloneAvailable(false);
      setRcloneWhy(
        missing ? 'missing' : err instanceof Error ? err.message : 'Request failed',
      );
      setRemotes([]);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        setProviders((await api.rcloneProviders()).providers);
      } catch {
        setProviders([]);
      }
      await loadRemotes();
    })();
  }, [loadRemotes]);

  const selectedProvider = providers.find((p) => p.type === newType) ?? null;
  const browserAuth = !!selectedProvider?.oauth && authMode === 'browser';

  // BOTH or NEITHER: rclone needs the pair, and a half-supplied app would mint
  // a token bound to something the remote cannot reproduce.
  const ownAppComplete =
    !!newValues.client_id?.trim() && !!newValues.client_secret?.trim();
  /**
   * Mirrors rclone_config.authorize_command. Kept in sync deliberately rather
   * than fetched, because it has to update as the operator types — and the
   * whole point is that the command shown and the credentials stored are the
   * same app.
   */
  const authorizeCommand = selectedProvider
    ? `rclone authorize "${selectedProvider.type}"` +
      (ownAppComplete
        ? ` --client-id "${newValues.client_id.trim()}" --client-secret "${newValues.client_secret.trim()}"`
        : '')
    : '';

  // The redirect URI depends on the address THIS browser reached the server on
  // (192.168.1.45:8080, a hostname, a tunnel…), so it is resolved live rather
  // than configured — and it is what the operator registers on the app.
  useEffect(() => {
    if (!selectedProvider?.oauth) return;
    void (async () => {
      try {
        const res = await api.rcloneRedirectUri(window.location.origin);
        setRedirectUri(res.redirect_uri);
        setOauthBlocked(res.blocked_reason || '');
      } catch {
        setRedirectUri('');
        setOauthBlocked('');
      }
    })();
  }, [selectedProvider]);

  const startBrowserSignIn = async () => {
    if (!selectedProvider) return;
    setSigningIn(true);
    setSetupMsg(null);
    try {
      const res = await api.startRcloneOAuth(
        newName,
        selectedProvider.type,
        newValues.client_id ?? '',
        newValues.client_secret ?? '',
        window.location.origin,
      );
      // A new tab, not a redirect: the operator keeps this settings page (and
      // its unsaved state) while approving on the provider's site.
      window.open(res.auth_url, '_blank', 'noopener');
      setSetupMsg({
        ok: true,
        text: 'Approve the sign-in in the new tab, then come back and press Refresh below.',
      });
    } catch (e) {
      setSetupMsg({ ok: false, text: e instanceof Error ? e.message : 'Could not start sign-in' });
    } finally {
      setSigningIn(false);
    }
  };

  const canBrowserSignIn =
    !!selectedProvider &&
    newName.trim().length > 0 &&
    (newValues.client_id ?? '').trim().length > 0 &&
    (newValues.client_secret ?? '').trim().length > 0 &&
    !signingIn;

  // Switching provider clears the form: field keys are per-provider, so keeping
  // values would send another backend's keys and be refused.
  const pickProvider = (type: string) => {
    setNewType(type);
    setNewValues({});
    setSetupMsg(null);
    setAuthMode('browser');
    if (!newName.trim()) setNewName(type);
  };

  const createRemote = async () => {
    if (!selectedProvider) return;
    setCreating(true);
    setSetupMsg(null);
    try {
      const res = await api.createRcloneRemote(newName, selectedProvider.type, newValues);
      if (!res.ok) {
        setSetupMsg({ ok: false, text: res.detail || 'Could not save the remote.' });
      } else if (!res.reachable) {
        // Saved but unreachable is its OWN outcome, not a success and not a
        // failure to write — say which, or the operator retypes a correct token.
        setSetupMsg({
          ok: false,
          text: `Saved, but it did not answer: ${res.detail}. Check the token or keys.`,
        });
      } else {
        setSetupMsg({ ok: true, text: `Connected. Use "${res.suggested_remote}" below.` });
        setArchive((a) => (a.remote.trim() ? a : { ...a, remote: res.suggested_remote }));
        setNewValues({});
      }
      await loadRemotes();
    } catch (e) {
      setSetupMsg({ ok: false, text: e instanceof Error ? e.message : 'Setup failed' });
    } finally {
      setCreating(false);
    }
  };

  const testRemote = async (name: string) => {
    setBusyRemote(name);
    setSetupMsg(null);
    try {
      const res = await api.testRcloneRemote(name);
      setSetupMsg({
        ok: res.ok,
        text: res.ok
          ? `${name}: connected${res.folders.length ? ` — ${res.folders.slice(0, 5).join(', ')}` : ' (no folders yet)'}`
          : `${name}: ${res.detail}`,
        hint: res.hint,
      });
    } catch (e) {
      setSetupMsg({ ok: false, text: e instanceof Error ? e.message : 'Test failed' });
    } finally {
      setBusyRemote(null);
    }
  };

  const removeRemote = async (name: string) => {
    if (!window.confirm(`Forget "${name}"? Vigilume stops being able to upload to it. Nothing already in the cloud is deleted.`)) return;
    setBusyRemote(name);
    try {
      await api.deleteRcloneRemote(name);
      await loadRemotes();
    } catch (e) {
      setSetupMsg({ ok: false, text: e instanceof Error ? e.message : 'Remove failed' });
    } finally {
      setBusyRemote(null);
    }
  };

  const canCreate =
    !!selectedProvider &&
    newName.trim().length > 0 &&
    !creating &&
    selectedProvider.fields
      .filter((f) => f.required && !f.default)
      .every((f) => (newValues[f.key] ?? '').trim().length > 0);

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
        title="Cloud storage"
        badge={
          archive.enabled
            ? archive.remote.trim() || 'On'
            : remotes.length
              ? `${remotes.length} connected, off`
              : 'Off'
        }
        tone={archive.enabled ? 'on' : 'muted'}
      >
        <p className="muted small">
          Copies each finished day of <strong>event clips and snapshots</strong> to cloud
          storage overnight, as one folder per day (<code>2026-08-25/</code>). 24/7 footage is
          never uploaded — far too large for any normal connection. This backs up the
          evidence, not everything.
        </p>

        <h4 style={{ margin: '1rem 0 0.3rem' }}>1. Connect an account</h4>
        <p className="muted small">
          Everything happens here — no terminal on the server.
        </p>

        {!rcloneAvailable && (
          <div className="form-error">
            {rcloneWhy === 'missing' ? (
              <>
                This backend build predates cloud storage — the server has no{' '}
                <code>/api/integrations/rclone</code> routes yet. On the server:{' '}
                <code>git pull</code>, then{' '}
                <code>docker compose up -d --build backend</code>, then reload this page.
                Your settings and recordings are on the <code>/data</code> volume and are
                not touched by a rebuild.
              </>
            ) : (
              <>
                Cloud storage is deployed but not working: <em>{rcloneWhy}</em>
                {rcloneWhy?.includes('not installed') && (
                  <>
                    {' '}Rebuild the backend image (the Dockerfile installs rclone) rather
                    than restarting the container — a restart reuses the same image.
                  </>
                )}
              </>
            )}
          </div>
        )}

        {remotes.length > 0 && (
          <div className="form-stack" style={{ marginBottom: '0.9rem' }}>
            {remotes.map((r) => (
              <div key={r.name} className="row-inline wrap" style={{ gap: '0.5rem' }}>
                <strong>{r.name}</strong>
                <span className="pill">{r.label}</span>
                {/* WHICH APP this remote refreshes its sign-in against. The
                    backend already sent it (client_id is not a secret and is
                    not redacted) and nothing showed it — yet it is the single
                    fact that explains an invalid_grant: a token minted by one
                    app can never refresh against another, so a remote holding
                    your app's key needs a token from YOUR app, and one holding
                    none needs a token from plain `rclone authorize`. */}
                {r.oauth && (
                  <span className="pill" title={
                    r.details.client_id
                      ? `Signs in with your own app (key ${r.details.client_id}). Re-authorizing must use the SAME app: rclone authorize "${r.type}" --client-id <key> --client-secret <secret>`
                      : `Signs in with rclone's built-in app. Re-authorize with plain: rclone authorize "${r.type}"`
                  }>
                    {r.details.client_id ? 'own app' : "rclone's app"}
                  </span>
                )}
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={busyRemote === r.name}
                  onClick={() => void testRemote(r.name)}
                >
                  {busyRemote === r.name ? 'Testing…' : 'Test'}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-danger"
                  disabled={busyRemote === r.name}
                  onClick={() => void removeRemote(r.name)}
                >
                  Forget
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="form-grid">
          <label>
            Add storage
            <select value={newType} onChange={(e) => pickProvider(e.target.value)}>
              <option value="">Choose a provider…</option>
              {providers.map((p) => (
                <option key={p.type} value={p.type}>
                  {p.label}
                </option>
              ))}
            </select>
            {selectedProvider && (
              <span className="control-hint">{selectedProvider.blurb}</span>
            )}
          </label>
          {selectedProvider && (
            <label>
              Name it
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder={selectedProvider.type}
              />
              <span className="control-hint">
                A short label for this account, used as <code>name:folder</code> below.
              </span>
            </label>
          )}
        </div>

        {selectedProvider?.oauth && (
          <div className="muted small" style={{ margin: '0.7rem 0' }}>
            <div className="row-inline wrap" style={{ gap: '1rem', marginBottom: '0.5rem' }}>
              <label className="row-inline" style={{ gap: '0.35rem' }}>
                <input
                  type="radio"
                  checked={authMode === 'browser'}
                  onChange={() => setAuthMode('browser')}
                />
                <span>Sign in here (recommended)</span>
              </label>
              <label className="row-inline" style={{ gap: '0.35rem' }}>
                <input
                  type="radio"
                  checked={authMode === 'token'}
                  onChange={() => setAuthMode('token')}
                />
                <span>Paste a token</span>
              </label>
            </div>

            {authMode === 'browser' && oauthBlocked ? (
              <p className="form-error">{oauthBlocked}</p>
            ) : authMode === 'browser' ? (
              <>
                <p>
                  {selectedProvider.label} sign-in finishes on this server, so no terminal
                  is needed anywhere. It does need its own app on{' '}
                  {selectedProvider.console_url ? (
                    <a href={selectedProvider.console_url} target="_blank" rel="noreferrer">
                      the {selectedProvider.label} developer site
                    </a>
                  ) : (
                    "the provider's developer site"
                  )}{' '}
                  — free, and about two minutes. Create one, add this exact redirect URI to
                  it, then paste its app key and secret below.
                </p>
                {redirectUri && (
                  <div className="row-inline wrap" style={{ gap: '0.4rem' }}>
                    <code style={{ wordBreak: 'break-all' }}>{redirectUri}</code>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => void navigator.clipboard?.writeText(redirectUri)}
                    >
                      Copy
                    </button>
                  </div>
                )}
              </>
            ) : (
              <>
                <p>
                  Run this <strong>on your own computer</strong> (it needs rclone installed),
                  approve the sign-in, and paste what it prints below.
                </p>
                {/* Built from the App key/secret typed BELOW, not fixed, because
                    the two must match: a refresh token is bound to the app that
                    issued it. Leave those blank and this stays the plain
                    command, which pairs with a remote that stores no app and so
                    also refreshes against rclone's built-in one. */}
                <pre><code>{authorizeCommand}</code></pre>
                <p className="muted small">
                  {ownAppComplete
                    ? 'Using your own app — the key and secret are on the command because the token must be minted by the same app this remote will refresh with.'
                    : 'Using rclone\u2019s built-in app — no registration needed. If you have your own Dropbox/Drive app, fill in App key and App secret below and this command will update to match.'}
                </p>
              </>
            )}
          </div>
        )}

        {selectedProvider && (
          <div className="form-grid">
            {selectedProvider.fields
              // In BROWSER mode only the app credentials are collected (the
              // token is fetched for you). In TOKEN mode everything shows,
              // INCLUDING the app credentials — hiding them was the bug: an
              // operator with their own app had nowhere to put its key, so the
              // remote stored a token minted by that app while refreshing
              // against rclone's built-in one. Dropbox answers that with
              // invalid_grant, hours later, forever.
              .filter((f) =>
                browserAuth ? f.key === 'client_id' || f.key === 'client_secret' : true,
              )
              .map((f) => (
              <label key={f.key}>
                {f.label}
                {f.kind === 'select' ? (
                  <select
                    value={newValues[f.key] ?? f.default}
                    onChange={(e) => setNewValues({ ...newValues, [f.key]: e.target.value })}
                  >
                    {f.options.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                ) : f.kind === 'token' ? (
                  <textarea
                    rows={3}
                    value={newValues[f.key] ?? ''}
                    placeholder={f.placeholder}
                    onChange={(e) => setNewValues({ ...newValues, [f.key]: e.target.value })}
                  />
                ) : (
                  <input
                    type={f.kind === 'secret' ? 'password' : 'text'}
                    autoComplete="off"
                    value={newValues[f.key] ?? ''}
                    placeholder={f.placeholder}
                    onChange={(e) => setNewValues({ ...newValues, [f.key]: e.target.value })}
                  />
                )}
                {(f.help || !f.required) && (
                  <span className="control-hint">
                    {f.help}
                    {!f.required && !browserAuth && (f.help ? ' (optional)' : 'Optional.')}
                  </span>
                )}
              </label>
            ))}
          </div>
        )}

        {selectedProvider && (
          <div className="row-inline wrap" style={{ marginTop: '0.8rem' }}>
            {browserAuth ? (
              <>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={!canBrowserSignIn || !!oauthBlocked}
                  onClick={() => void startBrowserSignIn()}
                >
                  {signingIn ? 'Opening…' : `Sign in to ${selectedProvider.label}`}
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={busyRemote !== null}
                  onClick={() => void loadRemotes()}
                >
                  Refresh
                </button>
                <span className="muted small">
                  Opens in a new tab; the account appears above once approved.
                </span>
              </>
            ) : (
              <>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={!canCreate}
                  onClick={() => void createRemote()}
                >
                  {creating ? 'Connecting…' : 'Connect'}
                </button>
                <span className="muted small">
                  Saves the account and immediately checks it works.
                </span>
              </>
            )}
          </div>
        )}
        {setupMsg && (
          <div className={setupMsg.ok ? 'muted small' : 'form-error'} style={{ marginTop: '0.5rem' }}>
            {setupMsg.text}
            {/* The explanation sits UNDER the raw error, not instead of it:
                rclone's own text is the evidence, and a hint that turns out to
                be wrong must not be the only thing on screen. */}
            {setupMsg.hint && (
              <p className="muted small" style={{ marginTop: '0.4rem' }}>
                {setupMsg.hint}
              </p>
            )}
          </div>
        )}

        <h4 style={{ margin: '1.4rem 0 0.3rem' }}>2. Schedule the upload</h4>

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

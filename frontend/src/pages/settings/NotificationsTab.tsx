/**
 * Settings → Notifications (admin): enable Web Push on this device, notification
 * rules (labels / cooldown / min score / snapshot boxes), and the mobile-push
 * card where the two phone channels — APNs and ntfy — are configured behind a
 * segmented picker. The per-device push control is shared with viewers via
 * <DevicePushCard>; everything below it edits the global settings document and
 * is admin-only.
 *
 * THE PICKER IS A VIEW SWITCH, NOT A MODE SWITCH. All three channels are
 * independent and fire together off the same rules: web push (this browser),
 * APNs (the Vigilume iOS app), and ntfy (any phone, no Apple account). Picking
 * a tab only chooses which one you are editing — each panel carries its own
 * enable control, and the tab labels show both channels' live state so the
 * hidden one can never be silently off. Do not "simplify" this into one
 * three-way mode: it would make enabling ntfy disable APNs.
 *
 * Backend tolerance (pre-deploy, like the MQTT card): `draw_boxes`, `apns` and
 * `ntfy` are optional in the settings document — absence falls back to defaults
 * (draw_boxes=true, apns mode "off", ntfy disabled), and the registered-device
 * list is skipped when GET /api/notifications/apns/devices 404s. NB `ntfy` is
 * optional for a second reason: an older backend STRIPPED it as a legacy block
 * (ntfy support was removed once, then restored), so it can come back absent
 * from a server that hasn't been rebuilt.
 */
import { useEffect, useState } from 'react';
import ChipsInput from '../../components/ChipsInput';
import DevicePushCard from '../../components/DevicePushCard';
import SettingsDisclosure from '../../components/SettingsDisclosure';
import { api, type ApnsSettings, type NtfySettings } from '../../lib/api';
import { useAdoptSaved, type TabProps } from '../Settings';

const DEFAULT_LABELS = ['person', 'dog', 'cat', 'car'];

const DEFAULT_APNS: ApnsSettings = { mode: 'off', relay_url: '' };

/**
 * What to put in Relay URL when you run the bundled `push-relay` yourself — the
 * Docker-internal service name, NOT the public hostname you point a tunnel at.
 * Going out to your own public URL and back means push breaks whenever the
 * tunnel, DNS, or your internet does. See docs/push-architecture.md §4.
 */
const LOCAL_RELAY_URL = 'http://push-relay:8090';

const DEFAULT_NTFY: NtfySettings = {
  enabled: false,
  server: 'https://ntfy.sh',
  topic: '',
  auth_token: '',
  priority: 4,
  attach_snapshot: true,
};

/** Fill any missing pieces of a (possibly absent) apns block with defaults. */
function withApnsDefaults(apns: ApnsSettings | undefined): ApnsSettings {
  return { ...DEFAULT_APNS, ...apns };
}

function withNtfyDefaults(ntfy: NtfySettings | undefined): NtfySettings {
  return { ...DEFAULT_NTFY, ...ntfy };
}

/**
 * A fresh, unguessable topic.
 *
 * This is a SECURITY control, not a convenience. On a default-allow ntfy
 * server (ntfy.sh included) the topic is the only thing standing between a
 * stranger and every notification this NVR sends — camera names and timing map
 * when the house is empty, and an attached snapshot URL carries a media token.
 * A human-chosen topic ("vigilume", "home") is guessable in seconds, so the UI
 * never offers an empty box to type in: it generates one.
 *
 * 128 bits from the CSPRNG, hex, inside ntfy's ^[A-Za-z0-9_-]{1,64}$.
 */
function generateTopic(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `vigilume_${hex}`;
}

export default function NotificationsTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft so switching tabs keeps unsaved edits.
  // Merge the pending PARTIAL over the full saved slice (SettingsPatch is a
  // deep-partial, so `pending` alone would be missing fields).
  const [draft, setDraft] = useState({
    ...settings.notifications,
    ...(pending.notifications ?? {}),
  });
  const [apns, setApns] = useState<ApnsSettings>(() =>
    withApnsDefaults(pending.notifications?.apns ?? settings.notifications.apns),
  );
  const [ntfy, setNtfy] = useState<NtfySettings>(() =>
    withNtfyDefaults(pending.notifications?.ntfy ?? settings.notifications.ntfy),
  );
  // Reveal state for the ntfy access token: start revealed (editable) only when
  // nothing is stored yet; a stored token shows masked until asked for.
  const [showNtfyToken, setShowNtfyToken] = useState(
    () => !settings.notifications.ntfy?.auth_token,
  );
  const [topicCopied, setTopicCopied] = useState(false);
  // Which channel's panel is showing. Open on one that's already configured so
  // an admin coming back to check a setting lands on it; otherwise APNs, the
  // one that can actually ring the doorbell.
  const [channel, setChannel] = useState<'apns' | 'ntfy'>(() =>
    settings.notifications.apns?.mode === 'relay' || !settings.notifications.ntfy?.enabled
      ? 'apns'
      : 'ntfy',
  );
  // Registered iOS devices count; null = unknown (endpoint absent/pre-deploy).
  const [deviceCount, setDeviceCount] = useState<number | null>(null);

  useAdoptSaved(settings.notifications, (n) => {
    setDraft(n);
    setApns(withApnsDefaults(n.apns));
    setNtfy(withNtfyDefaults(n.ntfy));
  });

  useEffect(() => {
    let alive = true;
    void api
      .apnsDevices()
      .then((devices) => {
        if (alive && Array.isArray(devices)) setDeviceCount(devices.length);
      })
      .catch(() => {
        /* endpoint absent (pre-deploy) — skip the count gracefully */
      });
    return () => {
      alive = false;
    };
  }, []);

  const patchApns = (p: Partial<ApnsSettings>) => setApns((a) => ({ ...a, ...p }));

  /** Normalize before it leaves the form: trim identifiers/URLs. */
  const normalizedNtfy = (): NtfySettings => ({
    ...ntfy,
    server: ntfy.server.trim().replace(/\/+$/, ''),
    topic: ntfy.topic.trim(),
    auth_token: ntfy.auth_token.trim(),
  });

  const normalizedApns = (): ApnsSettings => ({
    ...apns,
    relay_url: (apns.relay_url ?? '').trim().replace(/\/+$/, ''),
  });

  // Only the notifications slice — no `...settings` spread, which would PATCH a
  // stale copy of every other block back over the stored document.
  // Report this tab's slice up on every edit; the shell's single Save button
  // persists it with every other tab's pending changes.
  useEffect(() => {
    onDraftChange({
      notifications: {
        ...draft,
        // Absent means "true" (historical behavior) — persist it explicitly.
        draw_boxes: draft.draw_boxes ?? true,
        // Same "absent means the default" rule as draw_boxes: persist it
        // explicitly so the stored document stops being ambiguous.
        draw_zones: draft.draw_zones ?? true,
        draw_traces: draft.draw_traces ?? true,
        apns: normalizedApns(),
        ntfy: normalizedNtfy(),
      },
    });
    // normalized* derive from apns/ntfy; re-report whenever any draft moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, apns, ntfy, onDraftChange]);

  // Always-visible state for each collapsed section. A closed section that does
  // not say whether it is on would be how someone finds out a week late that
  // their alerts were off.
  const rulesBadge = draft.enabled
    ? `${draft.labels.length} label${draft.labels.length === 1 ? '' : 's'}`
    : 'Off';
  const apnsOn = apns.mode === 'relay';
  const ntfyOn = !!ntfy.enabled;
  const phoneBadge = apnsOn && ntfyOn ? 'Relay + ntfy' : apnsOn ? 'Relay' : ntfyOn ? 'ntfy' : 'Off';
  const phoneTone: 'on' | 'muted' = apnsOn || ntfyOn ? 'on' : 'muted';

  return (
    <div className="settings-section">
      <SettingsDisclosure
        title="Rules"
        badge={rulesBadge}
        tone={draft.enabled ? 'on' : 'warn'}
        open
      >
        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
          />
          Notifications enabled
        </label>
        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.draw_boxes ?? true}
            onChange={(e) => setDraft({ ...draft, draw_boxes: e.target.checked })}
          />
          Draw detection boxes on snapshots
        </label>
        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.draw_zones ?? true}
            disabled={!(draft.draw_boxes ?? true)}
            onChange={(e) => setDraft({ ...draft, draw_zones: e.target.checked })}
          />
          Draw include zones and crossing lines
        </label>
        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.draw_traces ?? true}
            disabled={!(draft.draw_boxes ?? true)}
            onChange={(e) => setDraft({ ...draft, draw_traces: e.target.checked })}
          />
          Draw the path each object walked
        </label>
        {/* Both are disabled, not hidden, when boxes are off: "no boxes" means a
            CLEAN snapshot, so the backend skips these too — better to show them
            greyed out than to let someone tick a box that does nothing. */}
        {!(draft.draw_boxes ?? true) && (
          <p className="muted small">
            Overlays are off while &ldquo;Draw detection boxes&rdquo; is unchecked — that
            option keeps the snapshot completely clean.
          </p>
        )}
        <label className="row-label">
          <input
            type="checkbox"
            checked={draft.camera_down_alerts ?? false}
            onChange={(e) =>
              setDraft({ ...draft, camera_down_alerts: e.target.checked })
            }
          />
          Alert me when a camera goes offline
        </label>
        <div className="form-stack">
          <div>
            <span className="control-label">Notify for labels</span>
            <ChipsInput
              value={draft.labels}
              onChange={(labels) => setDraft({ ...draft, labels })}
              suggestions={DEFAULT_LABELS}
            />
          </div>
          <label>
            Cooldown per camera+label (seconds)
            <input
              type="number"
              min={0}
              step={5}
              value={draft.cooldown_seconds}
              onChange={(e) =>
                setDraft({ ...draft, cooldown_seconds: Math.max(0, Number(e.target.value) || 0) })
              }
            />
          </label>
          <label>
            Minimum score: {Math.round(draft.min_score * 100)}%
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={draft.min_score}
              onChange={(e) => setDraft({ ...draft, min_score: Number(e.target.value) })}
            />
          </label>
        </div>
      </SettingsDisclosure>

      <SettingsDisclosure title="Phone push" badge={phoneBadge} tone={phoneTone}>
        <p className="muted small">
          Two independent ways to reach a phone, both firing off the same rules and
          per-camera filters as web push. Switching tabs only changes which one
          you&rsquo;re editing — <strong>you can run both at once</strong>.
        </p>

        <div className="seg seg-full" role="tablist" aria-label="Phone push channel">
          <button
            type="button"
            role="tab"
            id="push-tab-apns"
            aria-selected={channel === 'apns'}
            aria-controls="push-panel-apns"
            className={`seg-btn${channel === 'apns' ? ' seg-on' : ''}`}
            onClick={() => setChannel('apns')}
          >
            Vigilume iOS app {apns.mode === 'relay' ? '· On' : '· Off'}
          </button>
          <button
            type="button"
            role="tab"
            id="push-tab-ntfy"
            aria-selected={channel === 'ntfy'}
            aria-controls="push-panel-ntfy"
            className={`seg-btn${channel === 'ntfy' ? ' seg-on' : ''}`}
            onClick={() => setChannel('ntfy')}
          >
            ntfy {ntfy.enabled ? '· On' : '· Off'}
          </button>
        </div>

        {channel === 'apns' ? (
          <div id="push-panel-apns" role="tabpanel" aria-labelledby="push-tab-apns">
            <p className="muted small">
              The real thing: a native notification and a <strong>doorbell that rings
              like a phone call</strong> on a locked screen. Delivery goes through a
              relay holding the app&rsquo;s Apple signing key — this NVR never needs an
              Apple developer account. Payloads are encrypted end-to-end, so the relay
              can&rsquo;t read them, and snapshots never pass through it.
            </p>
            {deviceCount !== null && (
              <p className="muted small">
                {deviceCount === 0
                  ? 'No registered iOS devices yet — open the Vigilume app and enable notifications.'
                  : `${deviceCount} registered iOS ${deviceCount === 1 ? 'device' : 'devices'}.`}
              </p>
            )}

            <label className="row-label">
              <input
                type="checkbox"
                checked={apns.mode === 'relay'}
                onChange={(e) => patchApns({ mode: e.target.checked ? 'relay' : 'off' })}
              />
              Send notifications to the Vigilume iOS app
            </label>

            {apns.mode === 'relay' && (
              <div className="form-stack">
                <label>
                  Relay URL
                  <div className="row-inline">
                    <input
                      value={apns.relay_url ?? ''}
                      onChange={(e) => patchApns({ relay_url: e.target.value })}
                      placeholder={LOCAL_RELAY_URL}
                      autoComplete="off"
                      spellCheck={false}
                      inputMode="url"
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => patchApns({ relay_url: LOCAL_RELAY_URL })}
                    >
                      Use local
                    </button>
                  </div>
                  <span className="control-hint">
                    Running the bundled relay yourself? Use <code>{LOCAL_RELAY_URL}</code> —
                    the container name, <strong>not</strong> your public hostname. Going out
                    to the internet and back means push breaks whenever your tunnel, DNS, or
                    connection does. Otherwise, the address of the relay you were given.
                  </span>
                </label>
                {!(apns.relay_url ?? '').trim() && (
                  <span className="control-hint">
                    <strong>No relay URL set</strong> — nothing will be delivered to the iOS
                    app until you fill this in.
                  </span>
                )}
              </div>
            )}
          </div>
        ) : (
          <div id="push-panel-ntfy" role="tabpanel" aria-labelledby="push-tab-ntfy">
            <p className="muted small">
              Push with <strong>no Apple developer account and no relay</strong> — install
              the{' '}
              <a href="https://ntfy.sh" target="_blank" rel="noreferrer noopener">
                ntfy
              </a>{' '}
              app, subscribe to the topic below, done. Use the public ntfy.sh or run your
              own server. The trade: alerts arrive in the ntfy app, so there&rsquo;s no
              doorbell ring and no Vigilume UI around them.
            </p>

            <label className="row-label">
              <input
                type="checkbox"
                checked={ntfy.enabled}
                onChange={(e) => setNtfy({ ...ntfy, enabled: e.target.checked })}
              />
              Send notifications to ntfy
            </label>

            <div className="form-stack">
              <label>
                Server
                <input
                  value={ntfy.server}
                  onChange={(e) => setNtfy({ ...ntfy, server: e.target.value })}
                  placeholder="https://ntfy.sh"
                  autoComplete="off"
                  inputMode="url"
                />
                <span className="control-hint">
                  Public ntfy.sh, or your own server. Self-hosting with{' '}
                  <code>auth-default-access: deny-all</code> plus an access token below is the
                  private option.
                </span>
              </label>

              <label>
                Topic
                <div className="row-inline">
                  <input
                    value={ntfy.topic}
                    onChange={(e) => setNtfy({ ...ntfy, topic: e.target.value })}
                    placeholder="click Generate"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => {
                      setNtfy({ ...ntfy, topic: generateTopic() });
                      setTopicCopied(false);
                    }}
                  >
                    Generate
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={!ntfy.topic}
                    onClick={() => {
                      void navigator.clipboard.writeText(ntfy.topic).then(() => {
                        setTopicCopied(true);
                        window.setTimeout(() => setTopicCopied(false), 2000);
                      });
                    }}
                  >
                    {topicCopied ? 'Copied' : 'Copy'}
                  </button>
                </div>
                {/* Not a nag: on a default-allow server this string is the ONLY
                    thing gating access to every notification, and an attached
                    snapshot URL carries a media token. */}
                <span className="control-hint">
                  <strong>Treat this like a password.</strong> On ntfy.sh (and any server with
                  default access) anyone who knows the topic receives every notification from
                  this NVR — including the snapshot links. Use Generate rather than a name
                  someone could guess, and only share it with your own devices.
                </span>
              </label>

              <label>
                Access token
                <div className="row-inline">
                  <input
                    type={showNtfyToken ? 'text' : 'password'}
                    value={ntfy.auth_token}
                    onChange={(e) => setNtfy({ ...ntfy, auth_token: e.target.value })}
                    placeholder="tk_… (optional)"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => setShowNtfyToken((s) => !s)}
                  >
                    {showNtfyToken ? 'Hide' : 'Reveal'}
                  </button>
                </div>
                <span className="control-hint">
                  Required by a self-hosted server with <code>deny-all</code> access, and for
                  reserved topics on ntfy.sh. Leave empty for an open server.
                </span>
              </label>

              <label>
                Priority
                <select
                  value={ntfy.priority}
                  onChange={(e) => setNtfy({ ...ntfy, priority: Number(e.target.value) })}
                  aria-label="ntfy priority"
                >
                  <option value={1}>1 — Min</option>
                  <option value={2}>2 — Low</option>
                  <option value={3}>3 — Default</option>
                  <option value={4}>4 — High</option>
                  <option value={5}>5 — Urgent</option>
                </select>
              </label>

              <label className="row-label">
                <input
                  type="checkbox"
                  checked={ntfy.attach_snapshot}
                  onChange={(e) => setNtfy({ ...ntfy, attach_snapshot: e.target.checked })}
                />
                Attach the event snapshot
              </label>
              <span className="control-hint">
                The image is <strong>linked, never uploaded</strong> — your phone fetches
                it straight from this NVR, so it never touches the ntfy server. Needs
                Settings → System → Public URL to be reachable from your phone. Turn this
                off for text-only notifications.
              </span>
            </div>
          </div>
        )}
      </SettingsDisclosure>

      {/* This browser's own web-push subscription. Last and closed: it is a
          per-device action you take once, not a setting you tune. */}
      <SettingsDisclosure title="This browser" badge="Web push" tone="muted">
        <DevicePushCard allowTest inline />
      </SettingsDisclosure>

      {/* No Save button here — the shell owns the single Save for every
          settings tab. This page reports its slice via onDraftChange. */}
    </div>
  );
}

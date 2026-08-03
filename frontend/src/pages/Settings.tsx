/**
 * Settings shell: tab navigation (/settings/:tab) and the shared AppSettings
 * document. The tab set and the settings fetch are role-gated — a viewer only
 * sees Groups (shared camera groups) and a per-device push toggle, and never
 * fetches the admin-only settings document. Admins get the full tab set,
 * including Users management. The backend is the real authorization gate; this
 * gating keeps viewers from firing admin-only requests (no 403 spray) and
 * hides admin surfaces from the UI.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { NavLink, useNavigate, useParams } from 'react-router-dom';
import { api, type AppSettings, type SettingsPatch } from '../lib/api';
import { useAppState } from '../state/AppState';
import DevicePushCard from '../components/DevicePushCard';
import ReportBugCard from '../components/ReportBugCard';
import CamerasTab from './settings/CamerasTab';
import ExcludedObjectsTab from './settings/ExcludedObjectsTab';
import GroupsTab from './settings/GroupsTab';
import IntegrationsTab from './settings/IntegrationsTab';
import NotificationsTab from './settings/NotificationsTab';
import RecordingTab from './settings/RecordingTab';
import SystemTab from './settings/SystemTab';
import UsersTab from './settings/UsersTab';

// Notifications and MQTT live on ONE tab, called "Integrations": both are
// "how Vigilume tells something else that something happened" (phones vs Home
// Assistant), and splitting them meant hunting two tabs to wire up alerting.
// They are separate CARDS within it, not one merged box.
const ADMIN_TABS = [
  { id: 'cameras', label: 'Cameras' },
  { id: 'groups', label: 'Groups' },
  { id: 'integrations', label: 'Integrations' },
  { id: 'recording', label: 'Recording' },
  { id: 'excluded', label: 'Excluded objects' },
  { id: 'users', label: 'Users' },
  { id: 'system', label: 'System' },
] as const;

// A viewer's version of this tab is ONLY their own phone-push toggle, so it
// keeps the name "Notifications" — calling that "Integrations" would be
// meaningless to someone who cannot see the MQTT or rules cards.
const VIEWER_TABS = [
  { id: 'groups', label: 'Groups' },
  { id: 'notifications', label: 'Notifications' },
] as const;

type TabId =
  | 'cameras'
  | 'groups'
  | 'notifications'
  | 'recording'
  | 'integrations'
  | 'system'
  | 'excluded'
  | 'users';

export default function Settings() {
  const { tab } = useParams();
  const navigate = useNavigate();
  const { pushToast, isAdmin } = useAppState();
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  /**
   * Unsaved edits accumulated across EVERY settings tab, so there is exactly
   * ONE Save button for the whole settings document.
   *
   * It lives here, in the shell, rather than in each tab because a tab
   * unmounts when you switch away from it — local draft state would be
   * silently discarded, which is the bug this replaces. Tabs report their
   * slice up via `onDraftChange` and re-mount FROM this, so edits survive
   * moving between tabs and are persisted together in a single PATCH.
   *
   * Only the settings-document tabs use it. Cameras / Groups / Users /
   * Excluded objects are CRUD on separate records and still apply
   * immediately — batching a camera deletion behind a Save button would be
   * both surprising and easy to lose.
   */
  const [pending, setPending] = useState<SettingsPatch>({});

  const tabs = isAdmin ? ADMIN_TABS : VIEWER_TABS;
  const defaultTab = isAdmin ? 'cameras' : 'groups';
  const isAllowed = (t: string | undefined): boolean => tabs.some((x) => x.id === t);
  // The admin tab was renamed notifications -> integrations. An existing
  // bookmark / open tab on the old URL should land on the same content, not be
  // bounced to Cameras.
  const canonical = (t: string | undefined): string | undefined =>
    isAdmin && t === 'notifications' ? 'integrations' : t;
  const activeTab: TabId = (isAllowed(canonical(tab)) ? canonical(tab) : defaultTab) as TabId;

  // No tab, or a tab this role may not see (defense in depth alongside the
  // backend): redirect to the role's default tab.
  useEffect(() => {
    const target = canonical(tab);
    if (!isAllowed(target)) navigate(`/settings/${defaultTab}`, { replace: true });
    else if (target !== tab) navigate(`/settings/${target}`, { replace: true });
    // isAllowed/defaultTab derive from isAdmin; re-run when tab or role changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, isAdmin, navigate]);

  // Only admins fetch the settings document (GET /api/settings is admin-only).
  const loadSettings = useCallback(() => {
    if (!isAdmin) return;
    setLoadError(null);
    api
      .settings()
      .then(setSettings)
      .catch((e) => setLoadError(e instanceof Error ? e.message : 'Failed to load settings'));
  }, [isAdmin]);

  useEffect(loadSettings, [loadSettings]);

  /**
   * Save ONE slice of the settings document.
   *
   * Takes a PATCH, not a whole document, and that is load-bearing: each page
   * models a subset, so under PUT it had to spread `{...settings, mySlice}`
   * from a snapshot fetched once when the shell mounted. Anything that changed
   * out-of-band since — a detector model activated from the Detection page,
   * another admin's save — was silently reverted by an unrelated page's Save
   * button. PATCH sends only what the page owns, so it cannot clobber.
   */
  const save = useCallback(
    async (patch: SettingsPatch): Promise<boolean> => {
      setSaving(true);
      try {
        // The response is the full merged document — adopt it so every page
        // re-renders against fresh state rather than the stale snapshot.
        const saved = await api.patchSettings(patch);
        setSettings(saved);
        pushToast({ kind: 'info', title: 'Settings saved', body: '' });
        return true;
      } catch (e) {
        pushToast({
          kind: 'error',
          title: 'Save failed',
          body: e instanceof Error ? e.message : '',
        });
        return false;
      } finally {
        setSaving(false);
      }
    },
    [pushToast],
  );

  /** A tab reports its current draft slice; merged into the shared pending patch. */
  const onDraftChange = useCallback((patch: SettingsPatch) => {
    setPending((prev) => ({ ...prev, ...patch }));
  }, []);

  /**
   * Dirty = a reported slice actually DIFFERS from what's saved. Computed by
   * comparison rather than "did a tab report something", because every tab
   * reports its slice on mount — treating that as an edit would light up the
   * Save button the moment you opened a tab and changed nothing.
   */
  const dirty =
    settings !== null &&
    (Object.keys(pending) as Array<keyof SettingsPatch>).some(
      (k) => patchDiffers(pending[k], (settings as unknown as Record<string, unknown>)[k as string]),
    );

  const saveAll = useCallback(async () => {
    if (!dirty) return;
    if (await save(pending)) setPending({});
  }, [dirty, pending, save]);

  const discardAll = useCallback(() => setPending({}), []);

  const renderTab = () => {
    // Tabs that need no settings document (work for viewers and admins).
    if (activeTab === 'groups') return <GroupsTab />;
    if (activeTab === 'notifications' && !isAdmin) {
      return (
        <div className="settings-section">
          <DevicePushCard />
        </div>
      );
    }
    // Admin-only tabs below.
    if (activeTab === 'cameras') return <CamerasTab />;
    if (activeTab === 'users') return <UsersTab />;
    if (activeTab === 'excluded') return <ExcludedObjectsTab />;

    // Remaining admin tabs need the settings document.
    if (loadError) {
      return (
        <div className="banner banner-error">
          <span>{loadError}</span>
          <button type="button" className="btn btn-sm" onClick={loadSettings}>
            Retry
          </button>
        </div>
      );
    }
    if (!settings) return <div className="page-loading">Loading settings…</div>;
    const shared = { settings, onDraftChange, pending, saving };
    if (activeTab === 'recording') return <RecordingTab {...shared} />;
    // Each renders its own .settings-section, so they stay visually distinct
    // cards (see the `.settings-section + .settings-section` rule) rather than
    // reading as one merged notifications+MQTT box.
    if (activeTab === 'integrations')
      return (
        <>
          <NotificationsTab {...shared} />
          <IntegrationsTab {...shared} />
        </>
      );
    return <SystemTab {...shared} />;
  };

  /** True on the tabs whose edits the single Save button owns. */
  const isSettingsTab =
    isAdmin && ['notifications', 'recording', 'integrations', 'system'].includes(activeTab);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Settings</h1>
      </div>

      <div className="tabs" role="tablist" aria-label="Settings sections">
        {tabs.map((t) => (
          <NavLink
            key={t.id}
            to={`/settings/${t.id}`}
            className={({ isActive }) => `tab ${isActive || activeTab === t.id ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === t.id}
          >
            {t.label}
          </NavLink>
        ))}
      </div>

      {renderTab()}

      {/* Anyone can hit a bug, but Settings → System (where the admin's copy of
          this card lives) is admin-only, so a viewer would have no way to file
          one. Give them the same card at the foot of whichever settings tab
          they're on. Not rendered for admins — they'd see it twice. */}
      {!isAdmin && (
        <div className="settings-section">
          <ReportBugCard />
        </div>
      )}

      {/* THE single Save for the whole settings document. Sticky, so it stays
          reachable on a long tab, and it reports how many sections are dirty —
          the point being that edits on OTHER tabs are still pending and will go
          out with this one press. */}
      {isSettingsTab && dirty && (
        <div className="settings-savebar" role="status">
          <span className="settings-savebar-text">
            Unsaved changes in{' '}
            {
              (Object.keys(pending) as Array<keyof SettingsPatch>).filter(
                (k) =>
                  patchDiffers(
                    pending[k],
                    (settings as unknown as Record<string, unknown>)[k as string],
                  ),
              ).length
            }{' '}
            section(s)
          </span>
          <div className="row-inline">
            <button type="button" className="btn btn-sm" disabled={saving} onClick={discardAll}>
              Discard
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={saving}
              onClick={saveAll}
            >
              {saving ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Structural equality for plain JSON settings values. Not JSON.stringify:
 * that is key-ORDER sensitive, and these objects are rebuilt with spreads, so
 * a reordered-but-identical slice would read as dirty and keep the Save button
 * permanently lit.
 */
/**
 * Is a reported draft slice actually different from what's saved?
 *
 * Compares ONLY the keys the patch names, recursively. A tab reports a PARTIAL
 * slice — RecordingTab sends `detection: {confidence, default_mode}` while the
 * saved document also holds `detection.model` — so a plain deep-equal would
 * always differ and the Save bar would sit there permanently lit on a page
 * nobody had touched.
 */
function patchDiffers(patch: unknown, saved: unknown): boolean {
  if (patch === saved) return false;
  const bothPlainObjects =
    typeof patch === 'object' && patch !== null && !Array.isArray(patch) &&
    typeof saved === 'object' && saved !== null && !Array.isArray(saved);
  if (!bothPlainObjects) return !deepEqual(patch, saved);
  const p = patch as Record<string, unknown>;
  const s = saved as Record<string, unknown>;
  return Object.keys(p).some((k) => patchDiffers(p[k], s[k]));
}

/**
 * Adopt a freshly SAVED value — and ONLY when it actually changes.
 *
 * A tab seeds its draft from the shell's pending patch on mount; if a plain
 * `useEffect(() => apply(saved), [saved])` also ran there it would immediately
 * overwrite that seed, so switching away from a tab and back would silently
 * discard your edits — the exact bug the single Save button exists to fix.
 *
 * Compares the VALUE rather than counting renders. A "skip the first run" ref
 * looks equivalent and is not: React StrictMode double-invokes effects in dev,
 * so the first pass consumed the skip and the second pass clobbered the draft
 * anyway. Value comparison is correct under any invocation pattern.
 */
export function useAdoptSaved<T>(saved: T, apply: (value: T) => void): void {
  const previous = useRef(saved);
  useEffect(() => {
    if (Object.is(previous.current, saved)) return;
    previous.current = saved;
    apply(saved);
    // `apply` is a setState fn (stable); re-run only when the saved value moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saved]);
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  if (typeof a !== 'object') return false;
  const ao = a as Record<string, unknown>;
  const bo = b as Record<string, unknown>;
  const ak = Object.keys(ao);
  const bk = Object.keys(bo);
  if (ak.length !== bk.length) return false;
  return ak.every((k) => Object.prototype.hasOwnProperty.call(bo, k) && deepEqual(ao[k], bo[k]));
}

export interface TabProps {
  /** The full saved document, for READING. Never spread it into a patch. */
  settings: AppSettings;
  /**
   * Report this tab's current draft slice, e.g.
   * `onDraftChange({ recording })`. The shell accumulates every tab's slice
   * and persists them together under ONE Save button, in a single PATCH —
   * omitted subtrees are left untouched by the backend's deep-merge.
   */
  onDraftChange: (patch: SettingsPatch) => void;
  /**
   * Unsaved edits accumulated so far, INCLUDING this tab's own if you have
   * been here before. Tabs must seed their local draft from this (falling back
   * to `settings`) or switching away and back silently discards your edits.
   */
  pending: SettingsPatch;
  saving: boolean;
}

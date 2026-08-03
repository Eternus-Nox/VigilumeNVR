/**
 * Settings → Users (admin only). List managed users, create new ones, reset a
 * password, change a role, and delete. The built-in env admin (username
 * "admin") is env-controlled and never returned here — it can't be created,
 * demoted or deleted. Backend guards (reserved name, last-admin, built-in
 * admin) are surfaced as toasts; the backend is the real authority.
 */
import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, type Role, type User } from '../../lib/api';
import { Modal, ConfirmDialog } from '../../components/Modal';
import { useAppState } from '../../state/AppState';

const MIN_PASSWORD = 8;
// Mirrors the backend's _USERNAME_RE (routers/users.py) EXACTLY. It previously
// read /^[a-z0-9_]{2,32}$/, which was wrong in both directions: it accepted
// 2-character names the backend 400s, and rejected the "." and "-" the backend
// allows — so valid usernames were blocked and invalid ones only failed after a
// round-trip. Keep this in lockstep with the backend.
const USERNAME_HINT = /^[a-z0-9][a-z0-9_.-]{2,31}$/;

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

function formatCreated(v: string | number): string {
  const d = typeof v === 'number' ? new Date(v * 1000) : new Date(v);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString();
}

export default function UsersTab() {
  const { pushToast, username: currentUser } = useAppState();
  const [users, setUsers] = useState<User[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [resetting, setResetting] = useState<User | null>(null);
  const [deleting, setDeleting] = useState<User | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setLoadError(null);
    api
      .users()
      .then(setUsers)
      .catch((e) => setLoadError(errMsg(e)));
  }, []);

  useEffect(load, [load]);

  const createUser = async (username: string, password: string, role: Role) => {
    setBusy(true);
    try {
      const created = await api.createUser({ username, password, role });
      setUsers((us) => (us ? [...us, created] : [created]));
      setCreating(false);
      pushToast({ kind: 'info', title: 'User created', body: username });
    } catch (e) {
      // 409 duplicate / 400 reserved name / validation surface via the detail.
      pushToast({ kind: 'error', title: 'Create user failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const resetPassword = async (user: User, password: string) => {
    setBusy(true);
    try {
      await api.updateUser(user.id, { password });
      setResetting(null);
      pushToast({ kind: 'info', title: 'Password reset', body: user.username });
    } catch (e) {
      pushToast({ kind: 'error', title: 'Password reset failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const changeRole = async (user: User, role: Role) => {
    if (role === user.role) return;
    const prev = users;
    // Optimistic; revert on error (e.g. "cannot demote the last admin").
    setUsers((us) => (us ? us.map((u) => (u.id === user.id ? { ...u, role } : u)) : us));
    try {
      const saved = await api.updateUser(user.id, { role });
      if (saved) setUsers((us) => (us ? us.map((u) => (u.id === user.id ? saved : u)) : us));
      pushToast({ kind: 'info', title: 'Role updated', body: `${user.username} → ${role}` });
    } catch (e) {
      setUsers(prev);
      pushToast({ kind: 'error', title: 'Role change failed', body: errMsg(e) });
    }
  };

  const confirmDelete = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteUser(deleting.id);
      setUsers((us) => (us ? us.filter((u) => u.id !== deleting.id) : us));
      setDeleting(null);
      pushToast({ kind: 'info', title: 'User deleted', body: deleting.username });
    } catch (e) {
      pushToast({ kind: 'error', title: 'Delete failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-section">
      <div className="section-head">
        <h2>Users</h2>
        <button type="button" className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
          + New user
        </button>
      </div>
      <p className="muted small">
        The built-in <strong>admin</strong> account is controlled by the server’s
        <code> ADMIN_PASSWORD</code> and isn’t listed here. Create additional accounts as
        admins (full access) or viewers (live, events, recordings and shared groups only).
      </p>

      {loadError ? (
        <div className="banner banner-error">
          <span>{loadError}</span>
          <button type="button" className="btn btn-sm" onClick={load}>
            Retry
          </button>
        </div>
      ) : users === null ? (
        <div className="page-loading">Loading users…</div>
      ) : users.length === 0 ? (
        <p className="muted">No additional users yet — create one to grant access.</p>
      ) : (
        users.map((user) => {
          // Defensive: the built-in admin should never appear here, but if a row
          // ever does, keep it non-destructible.
          const protectedBuiltin = user.username === 'admin';
          return (
            <div key={user.id} className="camera-row">
              <span className={`status-dot ${user.role === 'admin' ? 'ok' : ''}`} />
              <div className="camera-row-info">
                <strong>
                  {user.username}
                  {currentUser === user.username && (
                    <span className="attn-badge" title="This is you">
                      you
                    </span>
                  )}
                </strong>
                <span className="muted small">
                  {formatCreated(user.created_at)
                    ? `created ${formatCreated(user.created_at)}`
                    : `user #${user.id}`}
                </span>
              </div>
              <div className="camera-row-actions">
                <select
                  className="user-role-select"
                  value={user.role}
                  disabled={protectedBuiltin}
                  onChange={(e) => void changeRole(user, e.target.value as Role)}
                  aria-label={`Role for ${user.username}`}
                >
                  <option value="viewer">Viewer</option>
                  <option value="admin">Admin</option>
                </select>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setResetting(user)}
                >
                  Reset password
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-danger-ghost"
                  disabled={protectedBuiltin}
                  onClick={() => setDeleting(user)}
                >
                  Delete
                </button>
              </div>
            </div>
          );
        })
      )}

      {creating && (
        <CreateUserModal
          busy={busy}
          onCreate={(username, password, role) => void createUser(username, password, role)}
          onClose={() => setCreating(false)}
        />
      )}

      {resetting && (
        <ResetPasswordModal
          user={resetting}
          busy={busy}
          onReset={(password) => void resetPassword(resetting, password)}
          onClose={() => setResetting(null)}
        />
      )}

      {deleting && (
        <ConfirmDialog
          title="Delete user"
          message={`Delete the account “${deleting.username}”? They will lose access immediately.`}
          confirmLabel="Delete"
          danger
          busy={busy}
          onConfirm={() => void confirmDelete()}
          onCancel={() => setDeleting(null)}
        />
      )}
    </div>
  );
}

// ---------- modals ----------

function CreateUserModal({
  busy,
  onCreate,
  onClose,
}: {
  busy: boolean;
  onCreate: (username: string, password: string, role: Role) => void;
  onClose: () => void;
}) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('viewer');

  const nameOk = USERNAME_HINT.test(username.trim());
  const passOk = password.length >= MIN_PASSWORD;
  const reserved = username.trim().toLowerCase() === 'admin';
  const canSubmit = nameOk && passOk && !reserved && !busy;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onCreate(username.trim(), password, role);
  };

  return (
    <Modal title="New user" onClose={onClose}>
      <form onSubmit={submit} className="form-stack">
        <label>
          Username
          <input
            required
            autoComplete="off"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="jane"
          />
          <span className="control-hint">
            Lowercase letters, digits, and <code>_ . -</code> (3–32). Must start with a
            letter or digit. “admin” is reserved.
          </span>
        </label>
        <label>
          Password
          <input
            required
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={`at least ${MIN_PASSWORD} characters`}
          />
        </label>
        <label>
          Role
          <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
            <option value="viewer">Viewer — live, events, recordings, groups</option>
            <option value="admin">Admin — full access</option>
          </select>
        </label>
        {reserved && <p className="form-error">“admin” is reserved for the built-in account.</p>}
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!canSubmit}>
            {busy ? 'Creating…' : 'Create user'}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ResetPasswordModal({
  user,
  busy,
  onReset,
  onClose,
}: {
  user: User;
  busy: boolean;
  onReset: (password: string) => void;
  onClose: () => void;
}) {
  const [password, setPassword] = useState('');
  const passOk = password.length >= MIN_PASSWORD;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!passOk || busy) return;
    onReset(password);
  };

  return (
    <Modal title={`Reset password — ${user.username}`} onClose={onClose}>
      <form onSubmit={submit} className="form-stack">
        <label>
          New password
          <input
            required
            autoFocus
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={`at least ${MIN_PASSWORD} characters`}
          />
        </label>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={!passOk || busy}>
            {busy ? 'Saving…' : 'Reset password'}
          </button>
        </div>
      </form>
    </Modal>
  );
}

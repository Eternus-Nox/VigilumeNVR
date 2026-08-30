/**
 * Settings → Groups: create / rename / delete camera groups and manage each
 * group's ordered camera list (drag handle + up/down arrows, add/remove
 * members). Groups drive the dashboard selector bar and TV mode. Member
 * names that no longer match a camera are tolerated server-side and simply
 * hidden here.
 */
import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, type Camera, type CameraGroup } from '../../lib/api';
import { Modal, ConfirmDialog } from '../../components/Modal';
import ReorderList from '../../components/ReorderList';
import { useAppState } from '../../state/AppState';
import { pluralize, titleCase } from '../../lib/format';

function camLabel(cam: Camera): string {
  return cam.friendly_name || titleCase(cam.name);
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

export default function GroupsTab() {
  const { cameras, pushToast, isAdmin } = useAppState();
  const [groups, setGroups] = useState<CameraGroup[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [renaming, setRenaming] = useState<CameraGroup | null>(null);
  const [deleting, setDeleting] = useState<CameraGroup | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setLoadError(null);
    api
      .groups()
      .then(setGroups)
      .catch((e) => setLoadError(errMsg(e)));
  }, []);

  useEffect(load, [load]);

  const putLocal = (g: CameraGroup) =>
    setGroups((gs) => (gs ? gs.map((x) => (x.id === g.id ? g : x)) : gs));

  /** Replace a group's full ordered camera list (optimistic, reverts on error). */
  const setGroupCameras = async (group: CameraGroup, names: string[]) => {
    const prev = group.cameras;
    putLocal({ ...group, cameras: names });
    try {
      const saved = await api.updateGroup(group.id, { cameras: names });
      if (saved) putLocal(saved);
    } catch (e) {
      putLocal({ ...group, cameras: prev });
      pushToast({ kind: 'error', title: 'Group update failed', body: errMsg(e) });
    }
  };

  const createGroup = async (name: string, members: string[]) => {
    setBusy(true);
    try {
      const created = await api.addGroup(name, members);
      setGroups((gs) => (gs ? [...gs, created] : [created]));
      setCreating(false);
    } catch (e) {
      // 409 duplicate name surfaces via the error detail.
      pushToast({ kind: 'error', title: 'Create group failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const renameGroup = async (group: CameraGroup, name: string) => {
    setBusy(true);
    try {
      const saved = await api.updateGroup(group.id, { name });
      putLocal(saved ?? { ...group, name });
      setRenaming(null);
    } catch (e) {
      pushToast({ kind: 'error', title: 'Rename failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteGroup(deleting.id);
      setGroups((gs) => (gs ? gs.filter((g) => g.id !== deleting.id) : gs));
      setDeleting(null);
    } catch (e) {
      pushToast({ kind: 'error', title: 'Delete failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-section">
      <div className="section-head">
        <h2>Camera groups</h2>
        {/* Groups are SHARED configuration — one account renaming or deleting a
            group changes what everyone sees — so a viewer reads them and an
            admin edits them. The backend enforces this; hiding the controls
            just stops a viewer being offered a button that would 403. */}
        {isAdmin && (
          <button type="button" className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
            + New group
          </button>
        )}
      </div>
      <p className="muted small">
        Groups appear as chips on the dashboard and in TV mode.
        {isAdmin
          ? ' Drag the handle (or use the arrows) to set each group\u2019s display order.'
          : ' Only an administrator can add, rename or reorder them.'}
      </p>

      {loadError ? (
        <div className="banner banner-error">
          <span>{loadError}</span>
          <button type="button" className="btn btn-sm" onClick={load}>
            Retry
          </button>
        </div>
      ) : groups === null ? (
        <div className="page-loading">Loading groups…</div>
      ) : groups.length === 0 ? (
        <p className="muted">No groups yet — create one to organize the dashboard.</p>
      ) : (
        groups.map((group) => {
          const byName = new Map((cameras ?? []).map((c) => [c.name, c]));
          const members = group.cameras
            .map((n) => byName.get(n))
            .filter((c): c is Camera => c !== undefined);
          const available = (cameras ?? []).filter((c) => !group.cameras.includes(c.name));
          return (
            <section key={group.id} className="card" aria-label={`Group ${group.name}`}>
              <div className="group-head">
                <h2>{group.name}</h2>
                <span className="muted small">
                  {members.length} {pluralize('camera', members.length)}
                </span>
                {isAdmin && (
                  <div className="group-actions">
                    <button type="button" className="btn btn-sm" onClick={() => setRenaming(group)}>
                      Rename
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-danger-ghost"
                      onClick={() => setDeleting(group)}
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>

              {members.length === 0 ? (
                <p className="muted small">No cameras in this group yet.</p>
              ) : !isAdmin ? (
                // Read-only for a viewer. Rendering the ReorderList would offer
                // drag handles and remove buttons whose every commit 403s — a
                // control that looks live and cannot work is worse than none.
                <ul className="group-cam-list">
                  {members.map((c) => (
                    <li key={c.name} className="group-cam-row">
                      <span className="group-cam-name">{camLabel(c)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <ReorderList
                  items={members}
                  itemKey={(c) => c.name}
                  onCommit={(next) =>
                    void setGroupCameras(
                      group,
                      next.map((c) => c.name),
                    )
                  }
                  itemClassName="group-cam-row"
                  ariaLabel={`Cameras in ${group.name}`}
                  renderItem={(c) => (
                    <>
                      <span className="group-cam-name">{camLabel(c)}</span>
                      <button
                        type="button"
                        className="icon-btn"
                        aria-label={`Remove ${camLabel(c)} from ${group.name}`}
                        onClick={() =>
                          void setGroupCameras(
                            group,
                            group.cameras.filter((n) => n !== c.name),
                          )
                        }
                      >
                        ✕
                      </button>
                    </>
                  )}
                />
              )}

              {isAdmin && available.length > 0 && (
                <div className="group-add">
                  <span className="control-label">Add camera</span>
                  <div className="chips">
                    {available.map((c) => (
                      <button
                        key={c.name}
                        type="button"
                        className="chip"
                        onClick={() => void setGroupCameras(group, [...group.cameras, c.name])}
                      >
                        + {camLabel(c)}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>
          );
        })
      )}

      {creating && (
        <CreateGroupModal
          cameras={cameras ?? []}
          busy={busy}
          onCreate={(name, members) => void createGroup(name, members)}
          onClose={() => setCreating(false)}
        />
      )}

      {renaming && (
        <RenameGroupModal
          group={renaming}
          busy={busy}
          onRename={(name) => void renameGroup(renaming, name)}
          onClose={() => setRenaming(null)}
        />
      )}

      {deleting && (
        <ConfirmDialog
          title="Delete group"
          message={`Delete the group “${deleting.name}”? The cameras themselves are not affected.`}
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

function CreateGroupModal({
  cameras,
  busy,
  onCreate,
  onClose,
}: {
  cameras: Camera[];
  busy: boolean;
  onCreate: (name: string, members: string[]) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState('');
  // Selection order = initial display order (reorder afterwards).
  const [members, setMembers] = useState<string[]>([]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    onCreate(name.trim(), members);
  };

  return (
    <Modal title="New group" onClose={onClose}>
      <form onSubmit={submit} className="form-stack">
        <label>
          Name
          <input
            required
            maxLength={64}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Front yard"
          />
        </label>
        <div>
          <span className="control-label">Cameras (tap to add — order follows your taps)</span>
          {cameras.length === 0 ? (
            <p className="muted small">No cameras available yet.</p>
          ) : (
            <div className="chips">
              {cameras.map((c) => {
                const on = members.includes(c.name);
                return (
                  <button
                    key={c.name}
                    type="button"
                    className={`chip ${on ? 'chip-on' : ''}`}
                    aria-pressed={on}
                    onClick={() =>
                      setMembers(
                        on ? members.filter((n) => n !== c.name) : [...members, c.name],
                      )
                    }
                  >
                    {camLabel(c)}
                  </button>
                );
              })}
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={busy || !name.trim()}>
            {busy ? 'Creating…' : 'Create group'}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function RenameGroupModal({
  group,
  busy,
  onRename,
  onClose,
}: {
  group: CameraGroup;
  busy: boolean;
  onRename: (name: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState(group.name);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    onRename(name.trim());
  };

  return (
    <Modal title={`Rename ${group.name}`} onClose={onClose}>
      <form onSubmit={submit} className="form-stack">
        <label>
          Name
          <input
            required
            maxLength={64}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={busy || !name.trim()}>
            {busy ? 'Saving…' : 'Rename'}
          </button>
        </div>
      </form>
    </Modal>
  );
}

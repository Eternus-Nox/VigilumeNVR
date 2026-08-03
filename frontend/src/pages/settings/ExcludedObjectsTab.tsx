/**
 * Settings → Excluded objects (admin only). The management surface for
 * reject-to-suppress: each time an event is marked a false detection ("Not a …"
 * on an event), the backend learns a suppression that stops alerting on that
 * kind of object at that spot on that camera. This tab lists them as a
 * thumbnail grid and lets an admin remove one to start alerting again. Mirrors
 * UsersTab's load/error/empty/remove shape.
 */
import { useCallback, useEffect, useState } from 'react';
import { api, type Suppression } from '../../lib/api';
import { ConfirmDialog } from '../../components/Modal';
import AuthImage from '../../components/AuthImage';
import { useAppState } from '../../state/AppState';
import { formatDateTime, titleCase } from '../../lib/format';

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

export default function ExcludedObjectsTab() {
  const { pushToast, cameras } = useAppState();
  const [items, setItems] = useState<Suppression[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [removing, setRemoving] = useState<Suppression | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setLoadError(null);
    api
      .listSuppressions()
      .then(setItems)
      .catch((e) => setLoadError(errMsg(e)));
  }, []);

  useEffect(load, [load]);

  // Prefer the friendly name for the camera; fall back to a title-cased key.
  const cameraName = (name: string) =>
    cameras?.find((c) => c.name === name)?.friendly_name || titleCase(name);

  const confirmRemove = async () => {
    if (!removing) return;
    setBusy(true);
    try {
      await api.deleteSuppression(removing.id);
      setItems((xs) => (xs ? xs.filter((x) => x.id !== removing.id) : xs));
      setRemoving(null);
      pushToast({ kind: 'info', title: 'Exclusion removed', body: titleCase(removing.label) });
    } catch (e) {
      pushToast({ kind: 'error', title: 'Remove failed', body: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-section">
      <div className="section-head">
        <h2>Excluded objects</h2>
      </div>
      <p className="muted small">
        When you mark a detection as a false alarm (the “Not a …” button on an event),
        Vigilume learns to ignore that kind of object at that spot on that camera — no more
        events or notifications for it. Remove an exclusion here to start alerting on it again.
      </p>

      {loadError ? (
        <div className="banner banner-error">
          <span>{loadError}</span>
          <button type="button" className="btn btn-sm" onClick={load}>
            Retry
          </button>
        </div>
      ) : items === null ? (
        <div className="page-loading">Loading excluded objects…</div>
      ) : items.length === 0 ? (
        <p className="muted">Nothing excluded yet.</p>
      ) : (
        <div className="suppression-grid">
          {items.map((s) => (
            <div key={s.id} className="card suppression-card">
              <div className="event-thumb">
                <AuthImage
                  src={api.suppressionThumbPath(s.id)}
                  alt={`${titleCase(s.label)} at ${cameraName(s.camera)}`}
                />
                <span className="event-label-chip" title={titleCase(s.label)}>
                  {titleCase(s.label)}
                </span>
              </div>
              <div className="event-meta">
                <span className="event-camera" title={cameraName(s.camera)}>
                  {cameraName(s.camera)}
                </span>
                <span className="event-time">{formatDateTime(s.created_at)}</span>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-danger-ghost"
                onClick={() => setRemoving(s)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      {removing && (
        <ConfirmDialog
          title="Remove exclusion"
          message={`Start alerting on ${titleCase(removing.label)} at this spot on ${cameraName(
            removing.camera,
          )} again?`}
          confirmLabel="Remove"
          danger
          busy={busy}
          onConfirm={() => void confirmRemove()}
          onCancel={() => setRemoving(null)}
        />
      )}
    </div>
  );
}

/**
 * Settings → Faces & plates: the enrolled people and vehicles, and the
 * unmatched sightings you enroll them from.
 *
 * ADMIN-ONLY, INCLUDING JUST LOOKING AT IT. Every other settings tab is gated
 * because it configures the system; this one is gated because of what it
 * CONTAINS — a named register of who comes to this address and which cars they
 * drive, plus a rolling gallery of strangers' faces. The server enforces it
 * (`require_admin` on the whole router, reads included) and the shell never
 * routes a viewer here, so the 403 never has to be explained.
 *
 * THERE IS NO TRAINING STEP, and the copy on this page leans on that. A profile
 * is a nearest-neighbour gallery entry: enrolling appends a reference, deleting
 * one removes it, and the model on disk never changes. So enrollment is instant
 * and fully reversible, and accuracy comes from sample DIVERSITY — different
 * angles, light and times of day — rather than from piling on more shots of one
 * moment.
 *
 * This tab does NOT report a draft to the settings shell. Everything here is an
 * immediate API call against /api/recognition, not a slice of the settings
 * document, so it must never call onDraftChange — doing so would light the
 * shell's Save bar for edits that were already saved. The enable/alert-mode
 * controls, which ARE settings, live on the Recording tab.
 */
import { useCallback, useEffect, useState } from 'react';
import {
  api,
  type CandidateKind,
  type ProfileKind,
  type RecognitionCandidate,
  type RecognitionProfile,
  type RecognitionProfileDetail,
  type RecognitionStatus,
} from '../../lib/api';
import AuthImage from '../../components/AuthImage';
import { ConfirmDialog } from '../../components/Modal';
import { useAppState } from '../../state/AppState';
import { formatDateTime, titleCase } from '../../lib/format';

/** A profile kind and the candidate kind you enroll into it, in one place. */
const CANDIDATE_OF: Record<ProfileKind, CandidateKind> = {
  person: 'face',
  vehicle: 'plate',
};

export default function RecognitionTab() {
  const { pushToast } = useAppState();
  const [kind, setKind] = useState<ProfileKind>('person');
  const [profiles, setProfiles] = useState<RecognitionProfile[]>([]);
  const [status, setStatus] = useState<RecognitionStatus | null>(null);
  const [candidates, setCandidates] = useState<RecognitionCandidate[]>([]);
  const [openProfile, setOpenProfile] = useState<RecognitionProfileDetail | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [newName, setNewName] = useState('');
  const [confirm, setConfirm] = useState<
    | { kind: 'profile'; id: number; name: string }
    | { kind: 'sample'; id: number }
    | { kind: 'candidates' }
    | null
  >(null);

  const candidateKind = CANDIDATE_OF[kind];
  const isPerson = kind === 'person';

  const fail = useCallback(
    (e: unknown, title: string) =>
      pushToast({ kind: 'error', title, body: e instanceof Error ? e.message : '' }),
    [pushToast],
  );

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const [p, s, c] = await Promise.all([
        api.recognitionProfiles(kind),
        api.recognitionStatus(),
        api.recognitionCandidates({ kind: CANDIDATE_OF[kind], limit: 120 }),
      ]);
      setProfiles(p);
      setStatus(s);
      setCandidates(c);
      // Drop selections for crops that no longer exist, or the enroll bar
      // could name rows that have since been enrolled or purged.
      setSelected((prev) => {
        const live = new Set(c.map((x) => x.id));
        return new Set([...prev].filter((id) => live.has(id)));
      });
    } catch (e) {
      fail(e, 'Could not load faces & plates');
    } finally {
      setLoading(false);
    }
  }, [kind, fail]);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Keep the open profile's samples in step after an enroll or a delete.
  const refreshOpen = useCallback(async (id: number) => {
    try {
      setOpenProfile(await api.recognitionProfile(id));
    } catch {
      setOpenProfile(null);
    }
  }, []);

  const create = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    try {
      await api.createRecognitionProfile({ kind, name });
      setNewName('');
      await reload();
    } catch (e) {
      fail(e, 'Could not add');
    } finally {
      setBusy(false);
    }
  };

  const enroll = async (profileId: number) => {
    if (selected.size === 0) return;
    setBusy(true);
    try {
      const { enrolled } = await api.enrollRecognitionCandidates(profileId, [...selected]);
      pushToast({
        kind: 'info',
        title: `Enrolled ${enrolled} ${enrolled === 1 ? 'shot' : 'shots'}`,
        body: '',
      });
      setSelected(new Set());
      await reload();
      if (openProfile?.id === profileId) await refreshOpen(profileId);
    } catch (e) {
      fail(e, 'Could not enroll');
    } finally {
      setBusy(false);
    }
  };

  const setPlate = async (profileId: number, plate: string) => {
    setBusy(true);
    try {
      await api.addRecognitionPlate(profileId, plate);
      await Promise.all([reload(), refreshOpen(profileId)]);
    } catch (e) {
      fail(e, 'Could not add the plate');
    } finally {
      setBusy(false);
    }
  };

  const toggleEnabled = async (p: RecognitionProfile) => {
    setBusy(true);
    try {
      await api.updateRecognitionProfile(p.id, { enabled: !p.enabled });
      await reload();
    } catch (e) {
      fail(e, 'Could not update');
    } finally {
      setBusy(false);
    }
  };

  const doConfirm = async () => {
    if (!confirm) return;
    setBusy(true);
    try {
      if (confirm.kind === 'profile') {
        await api.deleteRecognitionProfile(confirm.id);
        if (openProfile?.id === confirm.id) setOpenProfile(null);
      } else if (confirm.kind === 'sample') {
        await api.deleteRecognitionSample(confirm.id);
        if (openProfile) await refreshOpen(openProfile.id);
      } else {
        await api.clearRecognitionCandidates(candidateKind);
        setSelected(new Set());
      }
      setConfirm(null);
      await reload();
    } catch (e) {
      fail(e, 'That did not work');
    } finally {
      setBusy(false);
    }
  };

  const confirmCopy = (): { title: string; message: string; label: string } => {
    if (confirm?.kind === 'profile') {
      return {
        title: `Delete ${confirm.name}?`,
        message:
          'The profile and every reference enrolled into it are deleted, including the stored images. Past events keep the name they were already labelled with.',
        label: 'Delete',
      };
    }
    if (confirm?.kind === 'sample') {
      return {
        title: 'Remove this reference?',
        message:
          'It stops being matched against immediately and its image is deleted. The profile keeps its other references.',
        label: 'Remove',
      };
    }
    return {
      title: isPerson ? 'Clear unknown faces?' : 'Clear unread plates?',
      message:
        'Every unmatched crop shown here is deleted from the server. Anything already enrolled into a profile is kept.',
      label: 'Clear',
    };
  };

  return (
    <div className="settings-section">
      <section className="card">
        <h2>Faces &amp; plates</h2>
        <p className="muted small">
          People and vehicles you want recognized by name. There is{' '}
          <strong>no training step</strong> — a profile is a set of reference shots, so
          enrolling takes effect at once and removing a reference undoes it completely.
          What helps accuracy is <em>variety</em>: the same face at different angles, in
          daylight and at night, rather than five frames of one moment.
        </p>
        <p className="muted small">
          Recognition itself is switched on under <strong>Recording → Faces &amp; plates</strong>.
          Profiles can be set up either way; they start matching once it is on.
        </p>

        <div className="tabs" role="tablist" aria-label="Profile kind">
          {(['person', 'vehicle'] as ProfileKind[]).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={kind === k}
              className={`tab ${kind === k ? 'tab-active' : ''}`}
              onClick={() => {
                setKind(k);
                setOpenProfile(null);
                setSelected(new Set());
              }}
            >
              {k === 'person' ? 'People' : 'Vehicles'}
            </button>
          ))}
        </div>

        {status && !status.ready && (
          <p className="control-hint">
            The recognition models are still loading on the server. Enrollment works now;
            matching starts by itself once they are ready.
          </p>
        )}
        {status && status.stale_samples > 0 && (
          <p className="control-hint">
            <strong>{status.stale_samples}</strong>{' '}
            {status.stale_samples === 1 ? 'reference needs' : 'references need'} re-enrolling
            — they were captured with a different recognition model and can no longer be
            compared, so the profiles holding them have quietly stopped matching.
          </p>
        )}
      </section>

      <section className="card">
        <h2>{isPerson ? 'People' : 'Vehicles'}</h2>
        <div className="form-stack">
          <label>
            {isPerson ? 'Add a person' : 'Add a vehicle'}
            <div className="inline-form">
              <input
                type="text"
                value={newName}
                placeholder={isPerson ? 'Name' : 'Vehicle name'}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    void create();
                  }
                }}
              />
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={busy || !newName.trim()}
                onClick={() => void create()}
              >
                Add
              </button>
            </div>
            <span className="control-hint">
              {isPerson
                ? "Create the person first, then enroll their face from real sightings below."
                : 'Create the vehicle, then type its plate or enroll a sighting below.'}
            </span>
          </label>
        </div>

        {loading && profiles.length === 0 ? (
          <p className="muted small">Loading…</p>
        ) : profiles.length === 0 ? (
          <p className="empty-state">
            {isPerson
              ? 'Nobody enrolled yet.'
              : 'No vehicles enrolled yet.'}
          </p>
        ) : (
          <ul className="recog-profile-list">
            {profiles.map((p) => (
              <li key={p.id} className="recog-profile-row">
                <button
                  type="button"
                  className="recog-profile-open"
                  onClick={() =>
                    openProfile?.id === p.id ? setOpenProfile(null) : void refreshOpen(p.id)
                  }
                  aria-expanded={openProfile?.id === p.id}
                >
                  <span className="recog-profile-name">{p.name}</span>
                  <span className="muted small">{subtitle(p)}</span>
                </button>
                <div className="recog-profile-actions">
                  {selected.size > 0 && (
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      disabled={busy}
                      onClick={() => void enroll(p.id)}
                    >
                      Enroll {selected.size}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy}
                    onClick={() => void toggleEnabled(p)}
                    title={p.enabled ? 'Stop matching this profile' : 'Match this profile again'}
                  >
                    {p.enabled ? 'On' : 'Off'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger-ghost btn-sm"
                    disabled={busy}
                    onClick={() => setConfirm({ kind: 'profile', id: p.id, name: p.name })}
                  >
                    Delete
                  </button>
                </div>

                {openProfile?.id === p.id && (
                  <ProfileDetail
                    profile={openProfile}
                    busy={busy}
                    onAddPlate={(plate) => void setPlate(p.id, plate)}
                    onDeleteSample={(id) => setConfirm({ kind: 'sample', id })}
                  />
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>{isPerson ? 'Unknown faces' : 'Unread plates'}</h2>
          {candidates.length > 0 && (
            <button
              type="button"
              className="btn btn-danger-ghost btn-sm"
              disabled={busy}
              onClick={() => setConfirm({ kind: 'candidates' })}
            >
              Clear all
            </button>
          )}
        </div>
        <p className="muted small">
          Sightings that matched nobody. Pick the clearest ones and enroll them into a
          profile above. They are ordered by how <em>legible</em> they are, not by how
          recent — the shot worth enrolling is the one you can actually make out.
        </p>

        {candidates.length === 0 ? (
          <p className="empty-state">
            {loading
              ? 'Loading…'
              : status && !status.ready
                ? 'Nothing yet — recognition is still starting up.'
                : 'Nothing unmatched right now.'}
          </p>
        ) : (
          <>
            {selected.size > 0 && (
              <p className="control-hint">
                <strong>{selected.size} selected.</strong> Use <em>Enroll</em> on a profile
                above to add {selected.size === 1 ? 'it' : 'them'}.{' '}
                <button
                  type="button"
                  className="linklike"
                  onClick={() => setSelected(new Set())}
                >
                  Deselect all
                </button>
              </p>
            )}
            <ul className="recog-candidate-grid">
              {candidates.map((c) => {
                const on = selected.has(c.id);
                return (
                  <li key={c.id}>
                    <button
                      type="button"
                      className={`recog-candidate ${on ? 'is-selected' : ''}`}
                      aria-pressed={on}
                      onClick={() =>
                        setSelected((prev) => {
                          const next = new Set(prev);
                          if (next.has(c.id)) next.delete(c.id);
                          else next.add(c.id);
                          return next;
                        })
                      }
                    >
                      {c.has_image && c.image_url ? (
                        <AuthImage src={c.image_url} alt="" loading="lazy" />
                      ) : (
                        <span className="recog-candidate-noimg">
                          {c.plate || 'no image'}
                        </span>
                      )}
                      <span className="recog-candidate-meta">
                        {c.plate && <strong className="mono">{c.plate}</strong>}
                        <span>{titleCase(c.camera)}</span>
                        <span>{formatDateTime(c.created_at)}</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </section>

      {confirm && (
        <ConfirmDialog
          title={confirmCopy().title}
          message={confirmCopy().message}
          confirmLabel={confirmCopy().label}
          danger
          busy={busy}
          onConfirm={() => void doConfirm()}
          onCancel={() => setConfirm(null)}
        />
      )}
    </div>
  );
}

function subtitle(p: RecognitionProfile): string {
  // "Stopped matching" is the state worth naming: sample_count looks healthy
  // while nothing can actually be compared against it.
  if (p.sample_count > 0 && p.usable_sample_count === 0) return 'Needs re-enrolling';
  if (p.sample_count === 0) return p.kind === 'person' ? 'No faces enrolled' : 'No plate set';
  const unit = p.kind === 'person' ? 'face' : 'plate';
  return `${p.sample_count} ${unit}${p.sample_count === 1 ? '' : 's'}`;
}

function ProfileDetail({
  profile,
  busy,
  onAddPlate,
  onDeleteSample,
}: {
  profile: RecognitionProfileDetail;
  busy: boolean;
  onAddPlate: (plate: string) => void;
  onDeleteSample: (id: number) => void;
}) {
  const [plate, setPlate] = useState('');
  const isVehicle = profile.kind === 'vehicle';

  return (
    <div className="recog-profile-detail">
      {isVehicle && (
        <div className="form-stack">
          <label>
            Add a plate
            <div className="inline-form">
              <input
                type="text"
                value={plate}
                placeholder="7ABC123"
                className="mono"
                onChange={(e) => setPlate(e.target.value.toUpperCase())}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && plate.trim()) {
                    e.preventDefault();
                    onAddPlate(plate.trim());
                    setPlate('');
                  }
                }}
              />
              <button
                type="button"
                className="btn btn-sm"
                disabled={busy || !plate.trim()}
                onClick={() => {
                  onAddPlate(plate.trim());
                  setPlate('');
                }}
              >
                Add
              </button>
            </div>
            <span className="control-hint">
              Spaces and dashes are ignored — the plate is stored normalized, so it matches
              however the camera happens to read it.
            </span>
          </label>
        </div>
      )}

      {profile.samples.length === 0 ? (
        <p className="empty-state">
          No references yet. Select shots below and enroll them.
        </p>
      ) : (
        <ul className="recog-sample-grid">
          {profile.samples.map((s) => (
            <li key={s.id} className="recog-sample">
              {s.has_image && s.image_url ? (
                <AuthImage src={s.image_url} alt="" loading="lazy" />
              ) : (
                <span className="recog-candidate-noimg mono">{s.plate || '—'}</span>
              )}
              <button
                type="button"
                className="recog-sample-remove"
                disabled={busy}
                aria-label="Remove this reference"
                title="Remove this reference"
                onClick={() => onDeleteSample(s.id)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

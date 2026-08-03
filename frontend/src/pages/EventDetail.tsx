/**
 * Event detail (/events/:id) — the push-notification click target.
 * Media area is driven by the backend's `clip_state` so we never show a
 * silently broken player: a ready clip plays; a "processing" clip shows a
 * spinner and auto-refetches with backoff; a disabled/unavailable clip states
 * plainly why there is no video.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api, type NvrEventDetail } from '../lib/api';
import { downloadAttachment } from '../lib/download';
import AuthImage from '../components/AuthImage';
import AuthVideo from '../components/AuthVideo';
import { ConfirmDialog } from '../components/Modal';
import { useAppState } from '../state/AppState';
import {
  formatDateTime,
  formatDuration,
  formatScore,
  pluralize,
  titleCase,
} from '../lib/format';

// Backoff schedule for polling a clip that is still being cut (~20 s after the
// event ends, per the recorder). Caps out so we stop hammering the API.
const REFETCH_DELAYS_MS = [2500, 3500, 5000, 7000, 10000, 15000];

// Tray-with-down-arrow download glyph, inheriting the button's text colour.
function DownloadIcon() {
  return (
    <svg
      className="btn-icon"
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v12" />
      <path d="m7 11 5 5 5-5" />
      <path d="M5 21h14" />
    </svg>
  );
}

export default function EventDetail() {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const { pushToast, isAdmin } = useAppState();
  const [event, setEvent] = useState<NvrEventDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmReject, setConfirmReject] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [stalled, setStalled] = useState(false);
  const [downloading, setDownloading] = useState<'clip' | 'snapshot' | null>(null);
  const attemptsRef = useRef(0);

  useEffect(() => {
    setEvent(null);
    setError(null);
    setStalled(false);
    attemptsRef.current = 0;
    let cancelled = false;
    api
      .event(id)
      .then((e) => !cancelled && setEvent(e))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : 'Failed to load event'));
    return () => {
      cancelled = true;
    };
  }, [id]);

  const refreshEvent = useCallback(() => {
    let cancelled = false;
    api
      .event(id)
      .then((e) => !cancelled && setEvent(e))
      .catch(() => {
        /* keep the current view; a transient refetch failure is not fatal */
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Auto-poll while the clip is still processing, then give up gracefully.
  useEffect(() => {
    if (!event || event.clip_state !== 'processing') {
      attemptsRef.current = 0;
      return;
    }
    if (attemptsRef.current >= REFETCH_DELAYS_MS.length) {
      setStalled(true);
      return;
    }
    const delay = REFETCH_DELAYS_MS[attemptsRef.current];
    const timer = setTimeout(() => {
      attemptsRef.current += 1;
      refreshEvent();
    }, delay);
    return () => clearTimeout(timer);
  }, [event, refreshEvent]);

  const doDelete = async () => {
    setDeleting(true);
    try {
      await api.deleteEvent(id);
      pushToast({ kind: 'info', title: 'Event deleted', body: '' });
      navigate('/events', { replace: true });
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Delete failed',
        body: e instanceof Error ? e.message : '',
      });
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  const doReject = async () => {
    setRejecting(true);
    try {
      await api.rejectEvent(id);
      pushToast({
        kind: 'info',
        title: 'Detection excluded',
        body: 'This kind of detection is now ignored at this spot.',
      });
      navigate('/events', { replace: true });
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Exclude failed',
        body: e instanceof Error ? e.message : '',
      });
      setRejecting(false);
      setConfirmReject(false);
    }
  };

  const manualRefresh = () => {
    attemptsRef.current = 0;
    setStalled(false);
    refreshEvent();
  };

  const download = async (kind: 'clip' | 'snapshot') => {
    if (!event || downloading) return;
    setDownloading(kind);
    try {
      await downloadAttachment(
        api.eventDownloadUrl(event.id, kind),
        kind === 'clip' ? `event-${event.id}.mp4` : `event-${event.id}.jpg`,
      );
    } catch (e) {
      pushToast({
        kind: 'error',
        title: kind === 'clip' ? 'Clip download failed' : 'Snapshot download failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      setDownloading(null);
    }
  };

  if (error) {
    return (
      <div className="page">
        <div className="empty-state">
          <h2>Event unavailable</h2>
          <p className="muted">{error}</p>
          <Link to="/events" className="btn">
            Back to events
          </Link>
        </div>
      </div>
    );
  }
  if (!event) return <div className="page-loading">Loading event…</div>;

  const countText =
    event.count > 0 ? `${event.count} ${pluralize(event.label, event.count)} in frame` : null;
  // Multi-object events list every detected class; older events only `label`.
  const eventLabels = event.labels && event.labels.length > 0 ? event.labels : [event.label];
  const clipReady = event.clip_state === 'ready' && event.has_clip;

  return (
    <div className="page event-detail">
      <div className="page-head">
        <div>
          <h1>
            {titleCase(event.label)}
            {event.count > 1 ? ` ×${event.count}` : ''} · {titleCase(event.camera)}
          </h1>
          <p className="muted">{formatDateTime(event.start_time)}</p>
        </div>
        <div className="event-actions">
          {clipReady && (
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={downloading !== null}
              onClick={() => void download('clip')}
              aria-label="Download clip"
            >
              <DownloadIcon />
              {downloading === 'clip' ? 'Downloading…' : 'Download clip'}
            </button>
          )}
          {event.has_snapshot && (
            <button
              type="button"
              className="btn btn-sm"
              disabled={downloading !== null}
              onClick={() => void download('snapshot')}
              aria-label="Download snapshot"
            >
              <DownloadIcon />
              {downloading === 'snapshot' ? 'Downloading…' : 'Download snapshot'}
            </button>
          )}
          {isAdmin && (
            <button
              type="button"
              className="btn btn-sm btn-danger-ghost"
              onClick={() => setConfirmReject(true)}
            >
              {event.label ? `Not a ${titleCase(event.label)}` : 'Exclude this detection'}
            </button>
          )}
          {isAdmin && (
            <button type="button" className="btn btn-sm btn-danger-ghost" onClick={() => setConfirmDelete(true)}>
              Delete
            </button>
          )}
        </div>
      </div>

      <div className="event-media">
        {clipReady ? (
          <AuthVideo src={api.eventClipPath(event.id)} autoPlay muted />
        ) : event.has_snapshot ? (
          <AuthImage src={api.eventSnapshotPath(event.id)} alt="Annotated snapshot" eager />
        ) : (
          <div className="video-fallback">
            <p className="muted">No media captured for this event.</p>
          </div>
        )}
      </div>

      {/* Clip lifecycle: only when the clip isn't ready to play. */}
      {!clipReady && event.clip_state === 'processing' && (
        <div className="clip-status clip-status-processing">
          <span className="live-player-spinner" aria-hidden="true" />
          <div className="clip-status-body">
            <strong>{stalled ? 'Still processing recording…' : 'Processing recording…'}</strong>
            <p className="muted small">
              {stalled
                ? 'The clip is taking longer than usual to cut from continuous recording.'
                : 'The clip is being cut from continuous recording — this usually takes about half a minute.'}
            </p>
          </div>
          <button type="button" className="btn btn-sm" onClick={manualRefresh}>
            Refresh
          </button>
        </div>
      )}
      {!clipReady && event.clip_state === 'recording_disabled' && (
        <div className="clip-status">
          <p className="muted">
            Recording is off for this camera, so no clip was saved
            {event.has_snapshot ? ' — the annotated snapshot is shown above.' : '.'}
          </p>
        </div>
      )}
      {!clipReady && event.clip_state === 'unavailable' && (
        <div className="clip-status">
          <p className="muted">No recording was saved for this event.</p>
        </div>
      )}

      {clipReady && event.has_snapshot && (
        <details className="snapshot-details">
          <summary>Annotated snapshot</summary>
          <AuthImage src={api.eventSnapshotPath(event.id)} alt="Annotated snapshot" />
        </details>
      )}

      <dl className="meta-grid">
        <div>
          <dt>Camera</dt>
          <dd>
            <Link to={`/cameras/${encodeURIComponent(event.camera)}`} className="link">
              {titleCase(event.camera)}
            </Link>
          </dd>
        </div>
        <div>
          <dt>{eventLabels.length > 1 ? 'Labels' : 'Label'}</dt>
          <dd>{eventLabels.map(titleCase).join(', ')}</dd>
        </div>
        {countText && (
          <div>
            <dt>Count</dt>
            <dd>{countText}</dd>
          </div>
        )}
        <div>
          <dt>Score</dt>
          <dd>{formatScore(event.score)}</dd>
        </div>
        <div>
          <dt>Start</dt>
          <dd>{formatDateTime(event.start_time)}</dd>
        </div>
        <div>
          <dt>End</dt>
          <dd>{event.end_time ? formatDateTime(event.end_time) : 'in progress'}</dd>
        </div>
        {event.end_time && (
          <div>
            <dt>Duration</dt>
            <dd>{formatDuration(event.start_time, event.end_time)}</dd>
          </div>
        )}
        {event.zones.length > 0 && (
          <div>
            <dt>Zones</dt>
            <dd>{event.zones.map(titleCase).join(', ')}</dd>
          </div>
        )}
        <div>
          <dt>Event ID</dt>
          <dd className="mono">{event.id}</dd>
        </div>
      </dl>

      {confirmDelete && (
        <ConfirmDialog
          title="Delete event"
          message="Delete this event and its media? This cannot be undone."
          confirmLabel="Delete"
          danger
          busy={deleting}
          onConfirm={() => void doDelete()}
          onCancel={() => setConfirmDelete(false)}
        />
      )}

      {confirmReject && (
        <ConfirmDialog
          title="Exclude this object"
          message={`Stop alerting on this kind of detection at this spot on ${titleCase(
            event.camera,
          )}? Vigilume learns to ignore it and removes this event.`}
          confirmLabel="Exclude"
          danger
          busy={rejecting}
          onConfirm={() => void doReject()}
          onCancel={() => setConfirmReject(false)}
        />
      )}
    </div>
  );
}

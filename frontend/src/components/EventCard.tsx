/** Event thumbnail card used on the Events page and camera-detail strip. */
import { memo, useState } from 'react';
import { Link } from 'react-router-dom';
import type { NvrEvent } from '../lib/api';
import { api } from '../lib/api';
import { downloadAttachment } from '../lib/download';
import { useAppState } from '../state/AppState';
import { formatDateTime, formatScore, titleCase } from '../lib/format';
import AuthImage from './AuthImage';

// memo: event objects are identity-stable across list appends (pagination)
// and parent re-renders, and `pushToast` comes from the STABLE context slice,
// so a page of 50+ cards no longer re-renders on live WS traffic or when the
// parent list grows.
function EventCard({
  event,
  compact = false,
}: {
  event: NvrEvent;
  compact?: boolean;
}) {
  const { pushToast, cameras } = useAppState();
  const [busy, setBusy] = useState(false);
  // A ready clip is the best download; otherwise offer the annotated snapshot.
  const kind: 'clip' | 'snapshot' | null = event.has_clip
    ? 'clip'
    : event.has_snapshot
      ? 'snapshot'
      : null;

  const onDownload = async (e: React.MouseEvent) => {
    // The card is a <Link>; keep the click from navigating to the detail page.
    e.preventDefault();
    e.stopPropagation();
    if (!kind || busy) return;
    setBusy(true);
    try {
      await downloadAttachment(
        api.eventDownloadUrl(event.id, kind),
        kind === 'clip' ? `event-${event.id}.mp4` : `event-${event.id}.jpg`,
      );
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Download failed',
        body: err instanceof Error ? err.message : '',
      });
    } finally {
      setBusy(false);
    }
  };

  // Prefer the friendly name for the camera; fall back to a title-cased key.
  const cameraName =
    cameras?.find((c) => c.name === event.camera)?.friendly_name ||
    titleCase(event.camera);

  // Multi-object events carry every distinct class; older events only `label`.
  const labels = (event.labels && event.labels.length > 0 ? event.labels : [event.label]).map(
    titleCase,
  );
  const labelText = labels.join(', ');

  return (
    <>
      <Link to={`/events/${event.id}`} className={`event-card ${compact ? 'event-card-compact' : ''}`}>
        <div className="event-thumb">
          {event.has_snapshot ? (
            <AuthImage src={api.eventSnapshotPath(event.id)} alt={`${labelText} at ${cameraName}`} loading="lazy" />
          ) : (
            <div className="img-fallback" aria-hidden="true" />
          )}
          <span className="event-label-chip" title={labelText}>
            {labelText}
            {event.count > 1 ? ` ×${event.count}` : ''}
          </span>
          {kind && (
            <button
              type="button"
              className="event-download-btn"
              onClick={onDownload}
              disabled={busy}
              aria-label={kind === 'clip' ? 'Download clip' : 'Download snapshot'}
              title={kind === 'clip' ? 'Download clip' : 'Download snapshot'}
            >
              {busy ? (
                <span className="event-download-spin" aria-hidden="true" />
              ) : (
                <svg
                  width="14"
                  height="14"
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
              )}
            </button>
          )}
        </div>
        <div className="event-meta">
          <span className="event-camera" title={cameraName}>{cameraName}</span>
          <span className="event-time">
            {formatDateTime(event.start_time)}
            {!compact && event.score > 0 && <em> · {formatScore(event.score)}</em>}
          </span>
        </div>
      </Link>
    </>
  );
}

export default memo(EventCard);

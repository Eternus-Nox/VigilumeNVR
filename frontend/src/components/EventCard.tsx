/** Event thumbnail card used on the Events page and camera-detail strip. */
import { memo, useState } from 'react';
import { Link } from 'react-router-dom';
import type { NvrEvent } from '../lib/api';
import { api, headlineRecognition } from '../lib/api';
import RecognitionChip from './RecognitionChip';
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
  groupLabels,
  groupCount = 1,
}: {
  event: NvrEvent;
  compact?: boolean;
  /**
   * Every label seen across the MOMENT this card stands for, when the list is
   * grouping. The server opens one event per object type, so a person and the
   * car they arrived in are two events; the list shows one row, and that row
   * has to name both or it would be quietly lying about what was detected.
   */
  groupLabels?: string[];
  /** How many events this row stands for. 1 = an ordinary, ungrouped card. */
  groupCount?: number;
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
  const labels = (
    groupLabels && groupLabels.length > 0
      ? groupLabels
      : event.labels && event.labels.length > 0
        ? event.labels
        : [event.label]
  ).map(titleCase);
  const labelText = labels.join(', ');

  // At most one recognition on the thumbnail: the card has room for a single
  // line, and a stack of chips over a 160px image would bury the image the
  // card exists to show. The rest are on the detail page.
  const headline = headlineRecognition(event.recognitions);

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
          {/* Not a button: the list shows ONE row for the moment, and the
              other detections are on the detail page rather than expanding
              here. This only says the row stands for more than one record, so
              nobody thinks something was lost. */}
          {groupCount > 1 && (
            <span
              className="event-group-count"
              title={`${groupCount} separate detections within a few seconds`}
            >
              {groupCount} detections
            </span>
          )}
          {headline && <RecognitionChip recognition={headline} className="event-recog-chip" />}
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

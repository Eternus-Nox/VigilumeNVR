/**
 * Events timeline: filter by camera / label / date range, with incremental
 * offset pagination (auto-load sentinel + explicit Load-more button).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api, type NvrEvent } from '../lib/api';
import EventCard from '../components/EventCard';
import { useAppState } from '../state/AppState';
import { localInputToEpochSeconds, titleCase } from '../lib/format';

const PAGE_SIZE = 50;
const DEFAULT_LABELS = ['person', 'dog', 'cat', 'car'];

export default function Events() {
  const { cameras } = useAppState();
  const [params, setParams] = useSearchParams();

  const camera = params.get('camera') ?? '';
  const label = params.get('label') ?? '';
  const afterStr = params.get('after') ?? '';
  const beforeStr = params.get('before') ?? '';

  const [events, setEvents] = useState<NvrEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const offsetRef = useRef(0);
  const requestSeq = useRef(0);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const query = useMemo(
    () => ({
      camera: camera || undefined,
      label: label || undefined,
      after: localInputToEpochSeconds(afterStr),
      before: localInputToEpochSeconds(beforeStr),
    }),
    [camera, label, afterStr, beforeStr],
  );

  const load = useCallback(
    async (reset: boolean) => {
      const seq = ++requestSeq.current;
      setLoading(true);
      setError(null);
      const offset = reset ? 0 : offsetRef.current;
      try {
        const page = await api.events({ ...query, limit: PAGE_SIZE, offset });
        if (seq !== requestSeq.current) return; // stale response
        offsetRef.current = offset + page.events.length;
        setTotal(page.total);
        setEvents((prev) => (reset ? page.events : [...prev, ...page.events]));
      } catch (e) {
        if (seq !== requestSeq.current) return;
        setError(e instanceof Error ? e.message : 'Failed to load events');
      } finally {
        if (seq === requestSeq.current) setLoading(false);
      }
    },
    [query],
  );

  useEffect(() => {
    setEvents([]);
    offsetRef.current = 0;
    void load(true);
  }, [load]);

  const hasMore = events.length < total;

  // Auto-load next page when the sentinel scrolls into view.
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore || loading) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) void load(false);
      },
      { rootMargin: '400px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [hasMore, loading, load]);

  const setFilter = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  const knownLabels = useMemo(() => {
    const set = new Set(DEFAULT_LABELS);
    for (const ev of events) set.add(ev.label);
    if (label) set.add(label);
    return [...set].sort();
  }, [events, label]);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Events</h1>
        <span className="muted">{total ? `${total} total` : ''}</span>
      </div>

      <div className="filters" role="search" aria-label="Event filters">
        <select value={camera} onChange={(e) => setFilter('camera', e.target.value)} aria-label="Camera">
          <option value="">All cameras</option>
          {(cameras ?? []).map((c) => (
            <option key={c.name} value={c.name}>
              {c.friendly_name || titleCase(c.name)}
            </option>
          ))}
        </select>
        <select value={label} onChange={(e) => setFilter('label', e.target.value)} aria-label="Label">
          <option value="">All labels</option>
          {knownLabels.map((l) => (
            <option key={l} value={l}>
              {titleCase(l)}
            </option>
          ))}
        </select>
        <input
          type="datetime-local"
          value={afterStr}
          onChange={(e) => setFilter('after', e.target.value)}
          aria-label="After"
        />
        <input
          type="datetime-local"
          value={beforeStr}
          onChange={(e) => setFilter('before', e.target.value)}
          aria-label="Before"
        />
        {(camera || label || afterStr || beforeStr) && (
          <button type="button" className="btn btn-sm" onClick={() => setParams({}, { replace: true })}>
            Clear
          </button>
        )}
      </div>

      {error && (
        <div className="banner banner-error">
          <span>{error}</span>
          <button type="button" className="btn btn-sm" onClick={() => void load(events.length === 0)}>
            Retry
          </button>
        </div>
      )}

      {events.length === 0 && !loading && !error ? (
        <div className="empty-state">
          <h2>No events</h2>
          <p className="muted">Nothing matches these filters yet.</p>
        </div>
      ) : (
        <div className="event-grid">
          {events.map((ev) => (
            <EventCard key={String(ev.id)} event={ev} />
          ))}
        </div>
      )}

      <div ref={sentinelRef} />
      {loading && <div className="page-loading">Loading…</div>}
      {!loading && hasMore && (
        <div className="load-more">
          <button type="button" className="btn" onClick={() => void load(false)}>
            Load more ({total - events.length} remaining)
          </button>
        </div>
      )}
    </div>
  );
}

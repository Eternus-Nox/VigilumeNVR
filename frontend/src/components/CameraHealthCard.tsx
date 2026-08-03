/**
 * Camera reachability card for the System settings tab.
 *
 * Self-contained: fetches GET /api/system/camera-health itself and holds its
 * own state, so it can be dropped into SystemTab without touching that tab's
 * (large) state machine. "Uptime" here is RTSP-port reachability — what the
 * prober measures — and the copy says so rather than implying footage was
 * written.
 */
import { useCallback, useEffect, useState } from 'react';

import { api, type CameraHealthReport } from '../lib/api';

const RANGES = [
  { hours: 24, label: '24h' },
  { hours: 24 * 7, label: '7d' },
  { hours: 24 * 30, label: '30d' },
] as const;

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

export default function CameraHealthCard() {
  const [hours, setHours] = useState<number>(24);
  const [report, setReport] = useState<CameraHealthReport | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  const load = useCallback(async (h: number) => {
    setLoading(true);
    try {
      setReport(await api.cameraHealth(h));
    } catch {
      // Endpoint absent (backend not yet rebuilt) — hide the card rather than
      // show a broken control, exactly like PrivacyModeCard.
      setUnsupported(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(hours);
  }, [hours, load]);

  // Cameras reported offline RIGHT NOW (online === false; null is "unknown").
  const offline = report?.cameras.filter((c) => c.online === false).length ?? 0;
  // Force the card open when a camera is down, so a problem is never hidden
  // behind a collapsed triangle. Fires only on the transition into "any down".
  useEffect(() => {
    if (offline > 0) setOpen(true);
  }, [offline > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  if (unsupported) return null;

  return (
    <section className="card">
      <details
        className="card-disclosure"
        open={open}
        onToggle={(e) => setOpen(e.currentTarget.open)}
      >
        <summary>
          <span className="card-disclosure-title">Camera health</span>
          {offline > 0 ? (
            <span className="card-disclosure-badge card-disclosure-badge-warn">
              {offline} offline
            </span>
          ) : (
            <span className="card-disclosure-badge">{report ? 'All online' : '—'}</span>
          )}
        </summary>

        <div className="seg health-range">
          {RANGES.map((r) => (
            <button
              key={r.hours}
              type="button"
              className={`seg-btn ${hours === r.hours ? 'seg-on' : ''}`}
              onClick={() => setHours(r.hours)}
            >
              {r.label}
            </button>
          ))}
        </div>
      <p className="muted small">
        Reachability of each camera's stream port over the selected window.
        Uptime is connectivity, not a guarantee footage was recorded.
      </p>

      {loading && !report && <p className="muted small">Loading…</p>}

      {report && (
        <table className="health-table">
          <thead>
            <tr>
              <th>Camera</th>
              <th>Now</th>
              <th>Uptime</th>
              <th>Outages</th>
              <th>Downtime</th>
            </tr>
          </thead>
          <tbody>
            {report.cameras.map((c) => (
              <tr key={c.camera}>
                <td>{c.camera}</td>
                <td>
                  <span
                    className={`dot ${
                      c.online === null ? 'dot-unknown' : c.online ? 'dot-up' : 'dot-down'
                    }`}
                    title={c.online === null ? 'unknown' : c.online ? 'online' : 'offline'}
                  />
                </td>
                <td>{c.uptime_pct === null ? '—' : `${c.uptime_pct}%`}</td>
                <td>{c.down_count}</td>
                <td>{c.down_seconds > 0 ? fmtDuration(c.down_seconds) : '—'}</td>
              </tr>
            ))}
            {report.cameras.length === 0 && (
              <tr>
                <td colSpan={5} className="muted small">
                  No cameras.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      </details>
    </section>
  );
}

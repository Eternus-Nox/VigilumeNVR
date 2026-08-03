/** Time/label formatting helpers. Tolerant of epoch-seconds, epoch-ms, or ISO strings. */

export function toDate(value: number | string | null | undefined): Date | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'number') {
    // Heuristic: epoch seconds vs milliseconds.
    return new Date(value > 1e12 ? value : value * 1000);
  }
  const num = Number(value);
  if (!Number.isNaN(num) && value.trim() !== '') return toDate(num);
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function formatTime(value: number | string | null | undefined): string {
  const d = toDate(value);
  if (!d) return '—';
  return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

export function formatDateTime(value: number | string | null | undefined): string {
  const d = toDate(value);
  if (!d) return '—';
  return d.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function formatRelative(value: number | string | null | undefined): string {
  const d = toDate(value);
  if (!d) return '—';
  const diff = Date.now() - d.getTime();
  if (diff < 45_000) return 'just now';
  const mins = Math.round(diff / 60_000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export function formatDuration(
  start: number | string | null | undefined,
  end: number | string | null | undefined,
): string {
  const s = toDate(start);
  const e = toDate(end);
  if (!s || !e) return '—';
  const secs = Math.max(0, Math.round((e.getTime() - s.getTime()) / 1000));
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

const PLURALS: Record<string, string> = { person: 'people' };

export function pluralize(label: string, count: number): string {
  if (count === 1) return label;
  return PLURALS[label] ?? `${label}s`;
}

export function titleCase(s: string): string {
  return s.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/** datetime-local input value -> epoch seconds (backend convention). */
export function localInputToEpochSeconds(value: string): number | undefined {
  if (!value) return undefined;
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? undefined : Math.floor(ms / 1000);
}

/**
 * The Amcrest RTSP URL the backend derives when a camera has no stream
 * override — shown as override-field placeholders (credentials elided).
 */
export function amcrestDefaultUrl(ip: string, username: string, subtype: 0 | 1): string {
  const host = ip.trim() || '<camera-ip>';
  const user = username.trim() || 'user';
  return `rtsp://${user}:•••@${host}:554/cam/realmonitor?channel=1&subtype=${subtype}`;
}

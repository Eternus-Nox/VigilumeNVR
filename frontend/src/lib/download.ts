/**
 * Trigger a browser "save file" for a protected attachment URL (an
 * `api.mediaUrl(...)` path that carries the media token inline and that the
 * backend serves with `Content-Disposition: attachment`).
 *
 * We fetch the bytes ourselves rather than pointing a bare <a download> at the
 * URL so we can (a) surface a real error toast on failure — an over-cap 400, a
 * missing clip 404 — instead of the browser silently saving an error page, and
 * (b) honor the backend-provided filename. The body is buffered into a Blob;
 * event clips are short and the timeline export is capped server-side (~30 min),
 * so this stays bounded.
 */
import { ApiError } from './api';

/** Pull the filename out of a Content-Disposition header (RFC 5987 aware). */
function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  // `filename*=UTF-8''foo%20bar.mp4` takes precedence over plain `filename=`.
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header);
  if (star?.[1]) {
    const raw = star[1].trim().replace(/^"|"$/g, '');
    try {
      return decodeURIComponent(raw);
    } catch {
      return raw;
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(header);
  return plain?.[1]?.trim() ?? null;
}

/**
 * Download `path` to disk, saving with the server's filename when provided and
 * `fallbackName` otherwise. Rejects (with an {@link ApiError}) on a non-OK
 * response so callers can toast the reason; resolves once the save has started.
 */
export async function downloadAttachment(
  path: string,
  fallbackName: string,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    // The token rides in the query (mediaUrl), so no auth header is needed.
    res = await fetch(path, { signal });
  } catch {
    throw new ApiError(0, 'Network error — is the NVR reachable?');
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(res.status, detail);
  }
  const name = filenameFromDisposition(res.headers.get('Content-Disposition')) ?? fallbackName;
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = name;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Revoke after the browser has had a chance to start the save.
    setTimeout(() => URL.revokeObjectURL(objectUrl), 10_000);
  }
}

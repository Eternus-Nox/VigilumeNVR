/**
 * Timeline camera-selection persistence.
 *
 * The recorded-footage timeline scrubs 1–MAX_TIMELINE_CAMERAS cameras in sync.
 * The chosen camera names persist in localStorage; on first load (nothing
 * stored yet) the page defaults to the FIRST recording camera. Stored names are
 * always intersected with the live recording-camera list where they're read, so
 * a deleted/renamed camera silently drops out.
 */

export const TIMELINE_CAMERAS_KEY = 'sentinel.timeline.cameras';

/**
 * Max cameras scrubbed in sync. Bounded by consumer NVIDIA NVENC concurrent
 * transcode headroom (each on-screen player owns one hls.js instance).
 */
export const MAX_TIMELINE_CAMERAS = 4;

/**
 * The persisted camera list, or `null` when nothing has ever been stored (the
 * caller then defaults to the first recording camera). Names are capped to
 * MAX_TIMELINE_CAMERAS defensively.
 */
export function readStoredCameras(): string[] | null {
  try {
    const raw = localStorage.getItem(TIMELINE_CAMERAS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      const names = parsed.filter((c): c is string => typeof c === 'string');
      return names.slice(0, MAX_TIMELINE_CAMERAS);
    }
  } catch {
    /* malformed / private mode — treated as "no stored selection" */
  }
  return null;
}

export function storeCameras(names: string[]): void {
  try {
    localStorage.setItem(
      TIMELINE_CAMERAS_KEY,
      JSON.stringify(names.slice(0, MAX_TIMELINE_CAMERAS)),
    );
  } catch {
    /* private mode — selection just won't persist across reloads */
  }
}

/**
 * Whether event lists collapse near-simultaneous detections into one card.
 *
 * ON by default — the clutter is the common case — and remembered per browser,
 * because somebody reviewing an incident wants every row exactly as recorded.
 * Shared by the Events page (which has the toggle) and the camera page's recent
 * strip (which follows it), so turning it off means off everywhere.
 */
const KEY = 'vigilume.groupEvents';

export function groupingEnabled(): boolean {
  try {
    return window.localStorage.getItem(KEY) !== 'off';
  } catch {
    return true;
  }
}

export function setGroupingEnabled(on: boolean): void {
  try {
    window.localStorage.setItem(KEY, on ? 'on' : 'off');
  } catch {
    // A private window just forgets the preference next time.
  }
}

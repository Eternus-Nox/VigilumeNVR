/**
 * What lives where in Settings, for the search box.
 *
 * WHY A HAND-WRITTEN INDEX
 * ========================
 * Settings are spread over nine tabs and roughly fifty cards, and the names
 * people search by are mostly NOT the names on screen: "parked car" is the
 * stationary card, "blurry" is the detector model, "don't tell me about the
 * cat" is the notification labels. Filtering the rendered DOM would only ever
 * match the words already visible, which are exactly the words someone who
 * cannot find the setting does not know.
 *
 * So each entry carries the heading it lives under plus `terms`: the words
 * someone would actually type, including the symptom rather than the feature.
 * That is the whole value of the thing, and it is why this is worth
 * maintaining by hand.
 *
 * KEEPING IT HONEST: `card` must match the card's <h2> text exactly — the
 * search scrolls by finding that heading after switching tabs, so a rename
 * that misses this file degrades to "switches to the right tab, does not
 * scroll". A settings-search smoke test asserts every `tab` here is a real
 * tab id; the heading text is checked by eye, and a miss is not fatal.
 */

export interface SettingsEntry {
  /** Tab id, as used by the router and ADMIN_TABS. */
  tab: string;
  /** The card's <h2>, verbatim — used to scroll to it. */
  card: string;
  /** What the setting is, in the words of someone looking for it. */
  label: string;
  /** Extra search terms: symptoms, old names, and what people call it. */
  terms: string[];
  /** Admin-only entries are hidden from a viewer's search. */
  admin?: boolean;
}

export const SETTINGS_INDEX: SettingsEntry[] = [
  // ---- Detection -------------------------------------------------------
  {
    tab: 'detection', card: 'Detection model', admin: true,
    label: 'Which AI model detects objects',
    terms: ['model', 'ai', 'dfine', 'd-fine', 'accuracy', 'tier', 'gpu', 'slow',
            'missing things', 'not detecting', 'coral', 'download'],
  },
  {
    tab: 'detection', card: 'Detection hardware', admin: true,
    label: 'GPU, Coral or CPU',
    terms: ['gpu', 'coral', 'cpu', 'tpu', 'edge tpu', 'nvidia', 'cuda',
            'hardware', 'backend', 'slow', 'high cpu'],
  },
  {
    tab: 'detection', card: 'Confidence', admin: true,
    label: 'How sure the detector must be',
    terms: ['confidence', 'threshold', 'false alarm', 'false positive',
            'too many events', 'sensitivity', 'score'],
  },
  {
    tab: 'detection', card: 'Default detection mode', admin: true,
    label: 'Whether the server always runs detection',
    terms: ['detect mode', 'camera ai', 'always', 'load', 'gpu load'],
  },
  {
    tab: 'detection', card: 'Event grouping', admin: true,
    label: 'How long before an event closes',
    terms: ['absence', 'timeout', 'split events', 'one long event',
            'merged', 'duplicate events', 'event length', 'clip late'],
  },
  {
    tab: 'detection', card: "Things that aren't doing anything", admin: true,
    label: 'Ignore parked cars and other motionless objects',
    terms: ['parked car', 'parked', 'stationary', 'not moving', 'motionless',
            'furniture', 'bin', 'wheelie bin', 'statue', 'same car',
            'constant events', 'repeat events', 'loitering'],
  },
  {
    tab: 'detection', card: 'Something left behind', admin: true,
    label: 'Alert when a parcel or bag is left',
    terms: ['package', 'parcel', 'delivery', 'left behind', 'dropped off',
            'amazon', 'courier', 'box', 'bag', 'abandoned', 'doorstep'],
  },
  {
    tab: 'detection', card: "Someone who doesn't leave", admin: true,
    label: 'Alert when someone arrives and stays',
    terms: ['loiter', 'loitering', 'still there', 'hanging around', 'waiting',
            'dwell', 'lingering', 'standing', 'prowler', 'casing',
            'someone at the door too long', 'second alert'],
  },
  {
    tab: 'detection', card: 'Detector input', admin: true,
    label: 'Night contrast boost and box smoothing',
    terms: ['night', 'dark', 'ir', 'infrared', 'contrast', 'boost',
            'smoothing', 'jittery boxes', 'wobble'],
  },
  {
    tab: 'detection', card: 'Faces & plates', admin: true,
    label: 'How hard recognition looks (tuning)',
    terms: ['recognition', 'face tuning', 'shots per track', 'quality',
            'plate tuning', 'retention', 'unknown faces', 'aggressive',
            'full resolution', 'snapshot', 'plates not reading',
            'licence plate not working', 'license plate not working'],
  },

  // ---- Recording -------------------------------------------------------
  {
    tab: 'recording', card: 'Retention', admin: true,
    label: 'How many days of footage to keep',
    terms: ['retention', 'days', 'keep', 'delete', 'old footage', 'history',
            'disk full', 'storage', 'how long'],
  },
  {
    tab: 'recording', card: 'Storage limits', admin: true,
    label: 'Disk caps and free space',
    terms: ['storage', 'disk', 'gb', 'space', 'full', 'cap', 'quota', 'free'],
  },
  {
    tab: 'recording', card: 'Clip padding', admin: true,
    label: 'Extra footage either side of an event',
    terms: ['clip', 'padding', 'pre-roll', 'lead in', 'run on', 'post roll',
            'starts too late', 'cut off', 'misses the start'],
  },

  // ---- Faces & plates --------------------------------------------------
  {
    tab: 'faces', card: 'Faces & plates', admin: true,
    label: 'Enrolled people and vehicles, and who to be alerted about',
    terms: ['face', 'person', 'people', 'name', 'enroll', 'recognise',
            'recognize', 'who', 'unknown face', 'identify', 'plate', 'licence',
            'license', 'number plate', 'vehicle', 'anpr', 'lpr',
            // The per-profile alert policy lives inside a profile, so these
            // are the words that lead to it.
            'mute', 'silence', 'stop notifying me about me', 'watchlist',
            'always alert', 'my own door', 'household', 'family',
            'notify for one person'],
  },
  {
    tab: 'faces', card: 'Plate reading by camera', admin: true,
    label: 'Why plates are or are not being read',
    terms: ['plate', 'licence plate', 'license plate', 'not reading', 'not working',
            'no plates', 'anpr', 'lpr', 'plate diagnostics', 'too small',
            'snapshot', 'full resolution'],
  },
  {
    tab: 'faces', card: 'Which cameras', admin: true,
    label: 'Turn face or plate recognition off per camera',
    terms: ['which cameras', 'per camera', 'enable recognition', 'turn off',
            'driveway', 'door'],
  },

  // ---- Cameras ---------------------------------------------------------
  {
    tab: 'cameras', card: 'Cameras', admin: true,
    label: 'Add, edit and remove cameras',
    terms: ['camera', 'add camera', 'ip', 'rtsp', 'password', 'stream',
            'url', 'offline', 'zones', 'exempt', 'include zone', 'line',
            'crossing', 'mask', 'ignore area'],
  },
  {
    tab: 'excluded', card: 'Excluded objects', admin: true,
    label: 'Object types to ignore entirely',
    terms: ['exclude', 'ignore', 'object', 'label', 'cat', 'dog', 'bird',
            'no more dogs', 'class'],
  },

  // ---- Alerting --------------------------------------------------------
  {
    tab: 'integrations', card: 'Rules', admin: true,
    label: 'What raises an alert, and how loudly',
    terms: ['notification', 'alert', 'push', 'phone', 'cooldown', 'quiet',
            'too many', 'spam', 'labels', 'min score', 'snapshot', 'ntfy',
            'apns', 'ios push'],
  },
  {
    tab: 'integrations', card: 'Home Assistant (MQTT)', admin: true,
    label: 'MQTT and Home Assistant discovery',
    terms: ['mqtt', 'home assistant', 'hass', 'broker', 'discovery',
            'automation', 'integration'],
  },

  // ---- System ----------------------------------------------------------
  {
    tab: 'system', card: 'Cloud storage', admin: true,
    label: 'Copy footage to cloud storage',
    terms: ['archive', 'backup', 'offsite', 'cloud', 's3', 'rclone',
            'google drive', 'dropbox'],
  },
  {
    tab: 'system', card: 'Server', admin: true,
    label: 'Camera clocks and timezone',
    terms: ['time', 'clock', 'timezone', 'wrong date', 'ntp', 'drift'],
  },
  {
    tab: 'users', card: 'Users', admin: true,
    label: 'Accounts, roles and passwords',
    terms: ['user', 'account', 'password', 'role', 'viewer', 'admin',
            'login', 'permission', 'share access'],
  },
  {
    tab: 'groups', card: 'Camera groups',
    label: 'Group cameras for the dashboard',
    terms: ['group', 'dashboard', 'wall', 'order', 'arrange', 'tv'],
  },
  {
    tab: 'cameras', card: 'Privacy Mode', admin: true,
    label: 'Stop recording and detecting',
    terms: ['privacy', 'pause', 'stop recording', 'off', 'disable',
            'home', 'away', 'turn cameras off'],
  },
];

/**
 * Entries matching `query`, best first.
 *
 * Scored rather than merely filtered so "car" puts the parked-car card above
 * the vehicle profiles it also legitimately matches. A prefix hit on the label
 * beats a mid-word hit on a search term, and a card whose own heading matches
 * beats one that only matched a synonym — otherwise the ranking reads as
 * random to anyone who typed an obvious word.
 */
export function searchSettings(
  query: string,
  { isAdmin }: { isAdmin: boolean },
): SettingsEntry[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];
  const scored: Array<{ entry: SettingsEntry; score: number }> = [];

  for (const entry of SETTINGS_INDEX) {
    if (entry.admin && !isAdmin) continue;
    const card = entry.card.toLowerCase();
    const label = entry.label.toLowerCase();
    let score = 0;
    if (card.startsWith(q)) score = 100;
    else if (card.includes(q)) score = 80;
    else if (label.startsWith(q)) score = 70;
    else if (label.includes(q)) score = 60;
    else {
      for (const term of entry.terms) {
        const t = term.toLowerCase();
        if (t === q) { score = Math.max(score, 55); continue; }
        if (t.startsWith(q)) { score = Math.max(score, 40); continue; }
        if (t.includes(q)) score = Math.max(score, 20);
      }
    }
    if (score > 0) scored.push({ entry, score });
  }

  scored.sort((a, b) => b.score - a.score || a.entry.card.localeCompare(b.entry.card));
  return scored.slice(0, 8).map((s) => s.entry);
}

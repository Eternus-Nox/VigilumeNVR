/**
 * Typed API client for the Vigilume NVR backend.
 * Shapes match docs/CONTRACTS.md exactly. JWT lives in localStorage;
 * any 401 clears it and redirects to /login (preserving the return path).
 */

import { migrateKey } from './legacyStorage';

const TOKEN_KEY = 'vigilume_token';
const ROLE_KEY = 'vigilume_role';
const USERNAME_KEY = 'vigilume_username';
// Pre-rename keys (the app shipped as "Sentinel"). Read-through migrated on
// first access so an existing session is preserved rather than signed out.
const LEGACY_TOKEN_KEY = 'sentinel_token';
const LEGACY_ROLE_KEY = 'sentinel_role';
const LEGACY_USERNAME_KEY = 'sentinel_username';

/**
 * RBAC roles. `admin` has full access (everything the app did before roles
 * existed); `viewer` is a restricted, read-mostly role that may view live /
 * events / recordings, manage shared camera groups, and toggle push on its own
 * device — nothing else.
 */
export type Role = 'admin' | 'viewer';

export function getToken(): string | null {
  return migrateKey(TOKEN_KEY, LEGACY_TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

/**
 * Cached role from the login response / GET /api/auth/me, used to gate the UI
 * and (crucially) to avoid firing admin-only requests as a viewer. The backend
 * is the real authorization gate — this only shapes the client. A legacy
 * session that predates roles has no cached role; callers treat that as admin
 * (matching the backend's legacy-token compatibility).
 */
export function getRole(): Role | null {
  const r = migrateKey(ROLE_KEY, LEGACY_ROLE_KEY);
  return r === 'admin' || r === 'viewer' ? r : null;
}

export function setRole(role: Role): void {
  try {
    localStorage.setItem(ROLE_KEY, role);
  } catch {
    /* private mode — role just won't persist across reloads */
  }
}

export function getUsername(): string | null {
  return migrateKey(USERNAME_KEY, LEGACY_USERNAME_KEY);
}

export function setUsername(username: string): void {
  try {
    localStorage.setItem(USERNAME_KEY, username);
  } catch {
    /* private mode — username just won't persist across reloads */
  }
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  try {
    localStorage.removeItem(ROLE_KEY);
    localStorage.removeItem(USERNAME_KEY);
    // Sign-out must clear the pre-rename copies too, or a stale legacy token
    // would be migrated straight back in on the next read.
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    localStorage.removeItem(LEGACY_ROLE_KEY);
    localStorage.removeItem(LEGACY_USERNAME_KEY);
  } catch {
    /* ignore */
  }
}

export function isAuthed(): boolean {
  return getToken() !== null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function redirectToLogin(): void {
  clearToken();
  const here = window.location.pathname + window.location.search;
  const next = here && here !== '/login' ? `?next=${encodeURIComponent(here)}` : '';
  // Hard navigation: wipes all in-memory state and sockets.
  window.location.assign(`/login${next}`);
}

async function request<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (auth) {
    const token = getToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
  }
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(0, 'Network error — is the NVR reachable?');
  }
  if (res.status === 401 && auth) {
    redirectToLogin();
    throw new ApiError(401, 'Session expired');
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown; message?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
      else if (typeof body.message === 'string') detail = body.message;
      else if (Array.isArray(body.detail)) {
        // FastAPI validation errors: [{loc:[...], msg, type}]. Surface the
        // real reason ("ip: must be a bare IP…") instead of a blank "422".
        const parts = (body.detail as Array<{ loc?: unknown[]; msg?: unknown }>)
          .map((e) => {
            const field = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : undefined;
            const msg = typeof e.msg === 'string' ? e.msg.replace(/^Value error,\s*/, '') : '';
            return field && msg ? `${field}: ${msg}` : msg;
          })
          .filter(Boolean);
        if (parts.length) detail = parts.join('; ');
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

/**
 * Append the JWT as `?token=` to a media-route path. Used for URLs that the
 * browser/media stack fetches on its own (HLS playlist + segments, native
 * <video src>) where an Authorization header cannot be attached. The
 * snapshot/clip routes accept a session OR media-scope token this way
 * (docs/CONTRACTS.md “Media routes … additionally accept ?token=”).
 *
 * The RECORDINGS routes accept a session token only — a media-scope token is
 * refused there. Those tokens are minted into notification bodies and retained
 * MQTT messages, so one being seen must not unlock archive playback and export
 * for every camera. This helper always sends the session token, so nothing here
 * changes.
 */
/**
 * Subprotocol list carrying the session JWT for a WebSocket handshake.
 * `new WebSocket(url, wsSubprotocols())`. The server echoes "bearer" back on
 * accept; a browser aborts the connection if it offers subprotocols and the
 * server selects none.
 */
export function wsSubprotocols(): string[] {
  const token = getToken();
  return token ? ['bearer', token] : [];
}

export function mediaUrl(path: string): string {
  const token = getToken();
  if (!token) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}token=${encodeURIComponent(token)}`;
}

/** Fetch a protected binary resource (snapshot/clip) as an object URL. */
export async function fetchBlobUrl(path: string, signal?: AbortSignal): Promise<string> {
  const token = getToken();
  const res = await fetch(path, {
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    signal,
  });
  if (res.status === 401) {
    redirectToLogin();
    throw new ApiError(401, 'Session expired');
  }
  if (!res.ok) throw new ApiError(res.status, `Failed to load ${path}`);
  return URL.createObjectURL(await res.blob());
}

// ---------- Types (per CONTRACTS.md) ----------

export interface CameraCapabilities {
  ir: boolean;
  white_light: boolean;
  siren: boolean;
  mic: boolean;
  speaker: boolean;
  doorbell: boolean;
  ai_on_camera: boolean;
  /**
   * Camera accepts an RTSP/ONVIF audio *backchannel* (a `sendonly` audio SDP
   * media go2rtc auto-negotiates on the RTSP source), so two-way talk is done
   * by pushing mic audio over the live WebRTC connection instead of the HTTP
   * CGI postAudio path (lib/talk.ts). True for the AD410 doorbell, whose CGI
   * talk does not work. Optional so a pre-backchannel backend reads as false.
   */
  backchannel?: boolean;
  /**
   * Pan/tilt/preset dome (the IP3M-941B). Gates the PTZ UI and
   * `POST /api/cameras/{name}/ptz`. Optional so a pre-PTZ backend reads false.
   */
  ptz?: boolean;
  /**
   * Full-color (white-LED) night vision selectable alongside IR — the
   * IP4M-1056E. Gates the `night_vision_mode` settings control. When true the
   * `night_vision_mode` field is present in device settings. Optional so a
   * backend that predates the capability reads false.
   */
  night_vision?: boolean;
}

/**
 * Per-camera detection scheduling mode — how the server-side detector (GPU) is
 * gated for this camera. The tradeoff is GPU load vs. coverage:
 * - `always`       — run detection continuously on every sampled frame (the
 *                    historical behavior; heaviest GPU load).
 * - `camera_ai`      — run detection only while the camera's OWN on-camera AI
 *                      reports activity (SMD human/vehicle, IVS tripwire /
 *                      intrusion), then idle when it clears — big GPU savings.
 * - `camera_ai_only` — never run server-side detection; surface the camera's own
 *                      AI events directly (lightest, but only what the camera AI
 *                      sees).
 * `camera_ai` / `camera_ai_only` require `capabilities.ai_on_camera === true`;
 * the UI only offers them for such cameras (a camera without a running on-camera
 * AI watcher would never detect). Absent on a backend that predates the feature
 * — the UI treats a missing value as `always`. These string values are the
 * canonical backend names (config.VALID_DETECT_MODES) and are sent verbatim.
 */
export type DetectMode = 'always' | 'camera_ai' | 'camera_ai_only';

/**
 * A per-camera exempt (privacy / ignore) detection zone. `points` are `[x, y]`
 * pairs in NORMALIZED (0..1, resolution-independent) coords with the origin at
 * the top-left. Any detected object whose box foot-center (bottom-edge
 * midpoint) falls inside the polygon is suppressed — no event, notification or
 * annotation. A polygon needs at least 3 points.
 */
export interface ExemptZone {
  name: string;
  points: [number, number][];
}

/**
 * An "only alert here" polygon — the allow-list counterpart of {@link ExemptZone},
 * in the same normalized 0..1 coords. With none configured the whole frame is
 * watched; configure one and every detection whose box foot-center falls
 * OUTSIDE all of them is dropped before an event can open.
 *
 * The two compose in that order: include zones decide what is watched, exempt
 * zones then punch holes in it.
 */
export interface IncludeZone {
  name: string;
  points: [number, number][];
}

/**
 * A boundary whose crossings are counted. `start`/`end` are normalized 0..1.
 *
 * Direction matters and is visible in the editor: a crossing to the LEFT of the
 * start->end arrow counts as "in", the other way as "out". Drawing the line the
 * other way round swaps the two.
 */
export interface CrossLine {
  name: string;
  start: [number, number];
  end: [number, number];
}

/**
 * Per-camera live-view audio codec preference (PUT /api/cameras/{name}).
 * - `g711a` (DEFAULT): the backend forces the camera's audio encoder to
 *   G.711A. Native G.711/PCMA is WebRTC-legal, so it passes through go2rtc and
 *   live-view (WebRTC) audio WORKS. This is the current/historical behavior.
 * - `aac`: the backend forces the encoder to AAC — higher recording quality but
 *   NO live-view audio (go2rtc/WebRTC cannot carry AAC).
 * Absent on a backend that predates the feature — the UI treats a missing value
 * as `"g711a"`.
 */
export type AudioCodec = 'g711a' | 'aac';

export interface Camera {
  name: string;
  friendly_name: string;
  model: string;
  ip: string;
  online: boolean;
  /**
   * Software Privacy Mode: this camera is capturing NOTHING right now — no
   * recording, detection, events, live view or audio. Carried on the camera row
   * so a tile renders the overlay from the list it already has. Optional so a
   * backend that predates the feature reads as false (never a phantom blackout).
   */
  private?: boolean;
  capabilities: CameraCapabilities;
  /** Effective tracked objects (server returns defaults when unset). */
  detect_objects: string[];
  /** Privacy/ignore polygons; objects inside are dropped for detection. */
  exempt_zones: ExemptZone[];
  /**
   * Allow-list polygons; when non-empty, objects OUTSIDE them all are dropped.
   * Optional so a backend predating the feature reads as "watch everything".
   */
  include_zones?: IncludeZone[];
  /** Boundaries whose crossings are counted. Optional for the same reason. */
  cross_lines?: CrossLine[];
  /**
   * Only send a notification once something crosses one of this camera's lines.
   * Gates the ALERT only — the event, its clip and its snapshot are recorded
   * either way — and it is ignored entirely when no lines are drawn, so it can
   * never silently mute a camera.
   */
  notify_on_cross?: boolean;
  detect: { enabled: boolean };
  record: { enabled: boolean };
  /**
   * RTSP override for the main (record/live) stream. Empty string = the
   * derived Amcrest default built from ip + stored credentials.
   */
  main_url: string;
  /** RTSP override for the substream (detection). Empty = Amcrest default. */
  sub_url: string;
  /** Detection frame rate (frames/second sampled from the substream). */
  detect_fps: number;
  /**
   * How server-side detection is gated for this camera. Absent on a backend
   * that predates the feature — treat a missing value as `"always"`.
   */
  detect_mode?: DetectMode;
  /**
   * Live indicator: `true` while the camera's own on-camera AI is currently
   * flagging activity (so a `detect_mode: "camera_ai"` camera is actively
   * running the GPU right now). **Optional** — only present when the backend
   * exposes the live AI-event state; the UI simply hides the indicator when it
   * is absent.
   */
  ai_active?: boolean;
  /**
   * True when the stored username or password is empty. Device controls,
   * stream URLs and model probing need real credentials.
   */
  needs_credentials: boolean;
  /**
   * Live-view audio codec preference. Absent on a backend that predates the
   * feature — treat a missing value as `"g711a"` (the default; live-view audio
   * works). See {@link AudioCodec}.
   */
  audio_codec?: AudioCodec;
  /**
   * "Smart spotlight": when true, the backend turns this camera's white-light
   * spotlight ON while a PERSON is detected at NIGHT (local sunset..sunrise) and
   * keeps it on until 60 s after the last person detection, then off. Only
   * meaningful for `capabilities.white_light` cameras (the on-demand spotlight
   * driven by `AmcrestClient.set_white_light`); ignored otherwise. Absent on a
   * backend that predates the feature — treat a missing value as `false`.
   */
  smart_spotlight?: boolean;
  /**
   * How long (seconds) the smart spotlight stays ON after the LAST person
   * detection before switching off — the hold that replaces the old hardcoded
   * 60 s. Integer in 5..600; DEFAULT 60. Only meaningful alongside
   * {@link smart_spotlight} on a `white_light` camera (the controller reads it
   * live). Absent on a backend that predates the feature — treat a missing
   * value as `60`.
   */
  spotlight_hold_seconds?: number;
}

export interface CameraInput {
  name: string;
  friendly_name: string;
  model: string;
  ip: string;
  username: string;
  password: string;
  detect_objects?: string[];
  /**
   * Exempt (privacy/ignore) detection zones. Omitted = keep server value on
   * update / none on create; an explicit `[]` clears all zones. Points are
   * normalized 0..1; zones with < 3 points are dropped server-side.
   */
  exempt_zones?: ExemptZone[];
  /**
   * Include zones and crossing lines. Same contract as `exempt_zones`: omitted
   * keeps the server value, an explicit `[]` clears them. An include zone with
   * < 3 points, or a line whose ends coincide, is dropped server-side.
   */
  include_zones?: IncludeZone[];
  cross_lines?: CrossLine[];
  /** Alert only on a line crossing; omitted = keep server value. */
  notify_on_cross?: boolean;
  /** Optional per-camera engine toggles; omitted = keep server value. */
  detect_enabled?: boolean;
  record_enabled?: boolean;
  /** Optional RTSP overrides; empty string = derived Amcrest default. */
  main_url?: string;
  sub_url?: string;
  /** Detection frame rate (1–10, backend-enforced); omitted = keep server value. */
  detect_fps?: number;
  /**
   * How server-side detection is gated for this camera; omitted = keep server
   * value. Sent verbatim as one of config.VALID_DETECT_MODES; an unknown value
   * is rejected server-side (the validator never silently disables detection).
   */
  detect_mode?: DetectMode;
  /**
   * Live-view audio codec preference; omitted = keep server value. Validated
   * server-side to exactly `"g711a"` or `"aac"`. When it changes the backend
   * persists it AND re-provisions the camera's audio encoder to the chosen
   * codec (best-effort, never fatal). See {@link AudioCodec}.
   */
  audio_codec?: AudioCodec;
  /**
   * "Smart spotlight" toggle; omitted = keep server value. Persist-only — the
   * backend stores the flag and the night/person spotlight controller reads it
   * live (no device call on the PUT). Only meaningful for `white_light`
   * cameras; ignored otherwise. See {@link Camera.smart_spotlight}.
   */
  smart_spotlight?: boolean;
  /**
   * Spotlight hold in seconds — how long the smart spotlight stays on after the
   * last person detection; omitted = keep server value. Integer validated
   * server-side to 5..600 (out-of-range → 400); the controller also clamps it
   * defensively. Persist-only. See {@link Camera.spotlight_hold_seconds}.
   */
  spotlight_hold_seconds?: number;
}

/** Spotlight (white-light LED) mode, per the camera-controls-v2 addendum. */
export type WhiteLightMode = 'off' | 'on' | 'auto';

export interface WhiteLightState {
  mode: WhiteLightMode;
  /** 0–100; only meaningful when mode is "on". */
  brightness: number;
}

/**
 * Sensor day/night night-vision mode for a `night_vision`-capable camera
 * (IP4M-1056E). `auto` lets the camera decide, `color` forces full-color night
 * vision (the white LED), `bw` forces IR black-and-white. Written to the Dahua
 * day/night table by PUT settings — distinct from `ir_mode` (the IR
 * illuminator). Present in device settings only when `capabilities.night_vision`.
 */
export type NightVisionMode = 'auto' | 'color' | 'bw';

export interface DeviceSettings {
  ir_mode?: 'auto' | 'on' | 'off';
  night_vision_mode?: NightVisionMode;
  white_light?: WhiteLightState;
  flip?: boolean;
  osd_name?: string;
  motion_detect?: boolean;
  volume?: { mic?: number; speaker?: number };
}

/**
 * PTZ control (POST /api/cameras/{name}/ptz), capability-gated on `ptz`.
 * `step` nudges one small increment in `direction` (the d-pad is tap-to-step —
 * no continuous move/stop); `preset_set`/`preset_goto`/`preset_clear` need an
 * `index` (1–3). `speed` (1–8, default 4) sets the step magnitude. `move`/`stop`
 * remain for backends that still drive a continuous move.
 */
export type PtzAction = 'step' | 'move' | 'stop' | 'preset_set' | 'preset_goto' | 'preset_clear';
export type PtzDirection =
  | 'up'
  | 'down'
  | 'left'
  | 'right'
  | 'upleft'
  | 'upright'
  | 'downleft'
  | 'downright';

export interface PtzRequest {
  action: PtzAction;
  /** Required for `move`/`stop`. */
  direction?: PtzDirection;
  /** 1–8; only meaningful for `move` (backend default 4). */
  speed?: number;
  /** Preset slot 1–3; required for the `preset_*` actions. */
  index?: number;
}

/** Result of POST /api/cameras/{name}/probe (getDeviceType + capability probe). */
export interface ProbeResult {
  ok: boolean;
  model: string | null;
  capabilities: CameraCapabilities;
  /** Human-readable failure reason ("authentication failed" / "camera unreachable"). */
  detail: string | null;
}

export interface NvrEvent {
  id: number | string;
  camera: string;
  /** Primary detected class (kept for back-compat). */
  label: string;
  /**
   * All distinct detected classes on the camera's detect list for this event
   * (multi-object events trip several, e.g. person + car). Optional: a pre-
   * multi-label backend omits it, so clients fall back to `[label]`.
   */
  labels?: string[];
  count: number;
  score: number;
  start_time: number;
  end_time: number | null;
  has_clip: boolean;
  has_snapshot: boolean;
  zones: string[];
}

/**
 * Lifecycle of an event's recorded clip, derived server-side so the UI can
 * tell "the clip is still coming" from "it is never coming":
 * - `ready` — the clip file exists and plays;
 * - `processing` — recording on, event ended recently, clip not written yet
 *   (the recorder cuts it ~20 s after the event ends);
 * - `recording_disabled` — the camera isn't recording (also synthetic
 *   doorbell/audio events, which never get a clip);
 * - `unavailable` — recording was on but the clip never landed (recorder was
 *   down for the window / extraction failed).
 */
export type ClipState = 'ready' | 'processing' | 'recording_disabled' | 'unavailable';

export interface NvrEventDetail extends NvrEvent {
  clip_url: string;
  snapshot_url: string;
  /** Whether the event's camera currently records 24/7. */
  record_enabled: boolean;
  /** Where the clip is in its lifecycle — drives the media UX. */
  clip_state: ClipState;
}

export interface EventsPage {
  events: NvrEvent[];
  total: number;
}

export interface EventsQuery {
  camera?: string;
  label?: string;
  after?: number;
  before?: number;
  limit?: number;
  offset?: number;
}

/**
 * Detector model key — the backend ModelStore is the source of truth for which
 * keys exist. Historically the COCO D-FINE sizes (`dfine_n`/`dfine_s`/
 * `dfine_m`); the expanded manager also handles higher-accuracy COCO tiers
 * (`dfine_l`/`dfine_x`) and larger-vocabulary models (e.g. an Objects365 model).
 * Kept as a string so a new key the backend adds flows through the model
 * manager and settings without a frontend change.
 */
export type DetectionModel = string;

/**
 * Night contrast boost mode (settings.detection.night_boost). `auto` boosts
 * only frames measured darker than the threshold; `always` exists so a boost
 * can be A/B'd against `off` on one camera.
 */
export type NightBoostMode = 'off' | 'auto' | 'always';

export interface AppSettings {
  notifications: {
    enabled: boolean;
    labels: string[];
    cooldown_seconds: number;
    min_score: number;
    /**
     * Draw detection bounding boxes on notification/event snapshots.
     * **Optional** in the document: a backend that predates the option omits
     * it — treat absence as `true` (the historical behavior).
     */
    draw_boxes?: boolean;
    /**
     * Draw the camera's include zones and crossing lines on event snapshots,
     * and each object's recent ground path. Both are pure annotation — they
     * never change what was detected. Optional: absent on an older backend —
     * treat absence as `true` (the shipped default).
     */
    draw_zones?: boolean;
    draw_traces?: boolean;
    /**
     * Push a system alert when a camera stops responding (debounced). Optional:
     * absent on a backend that predates it — treat absence as `false` (off).
     */
    camera_down_alerts?: boolean;
    /**
     * iOS push (APNs) transport config per docs/push-architecture.md §4.
     * **Optional**: a backend that predates APNs omits the block and (per the
     * settings legacy-drop rule) silently drops an unknown `apns` sent on
     * PUT, so the UI must fall back to defaults when it is absent.
     */
    apns?: ApnsSettings;
    /**
     * ntfy push — no Apple account needed (docs/push-architecture.md §7).
     * **Optional**: absent on a backend that predates it, and also on one old
     * enough to have stripped it as a legacy block (ntfy support was removed
     * once and later restored) — so fall back to defaults when absent.
     */
    ntfy?: NtfySettings;
  };
  recording: {
    continuous_days: number;
    event_days: number;
    snapshot_days: number;
    /**
     * Space-based rotation, applied on top of the day cutoffs above —
     * whichever frees a recording first wins. Only 24/7 footage rotates;
     * event clips expire by `event_days` alone and are never deleted for space.
     *
     * `max_storage_gb` 0 = uncapped. `min_free_gb` is the free-space floor kept
     * on the recordings filesystem.
     */
    max_storage_gb: number;
    min_free_gb: number;
    /**
     * Seconds of footage kept either side of a detected event in its clip.
     *
     * Measured from DETECTION, not from the subject entering frame — the
     * tracker needs a few frames on an object big enough to hit on, so pre-roll
     * absorbs that lag before it buys any lead-in. Post-roll is capped by the
     * backend (`MAX_CLIP_POST_S`) because later footage is not yet on disk when
     * the clip is cut.
     */
    clip_pre_s: number;
    clip_post_s: number;
    /**
     * Seconds after an event ends before its clip is cut. Raising it is the only
     * way to buy post-roll past the default ceiling, since a segment is not on
     * disk until it closes: reachable post-roll is `clip_delay_s - 10`. The
     * backend rejects a `clip_post_s` larger than that.
     */
    clip_delay_s: number;
  };
  detection: {
    model: DetectionModel;
    /**
     * Which silicon runs inference. `gpu` = D-FINE ONNX on CUDA (default,
     * highest accuracy). `coral` = SSDLite MobileDet on an Edge TPU — far lower
     * power, lower accuracy, and it needs the Coral hardware fitted plus a
     * backend restart to take effect. Optional: a backend that predates the
     * feature omits it; treat a missing value as `gpu`.
     */
    backend?: DetectionBackend;
    /**
     * Edge TPU model key. SEPARATE from `model` (the D-FINE tier) because the
     * two lists are disjoint — a single field would make an invalid pair
     * reachable the moment `backend` flips. Ignored while backend is `gpu`.
     */
    coral_model?: CoralModel;
    /** Decode confidence threshold (0.2–0.9). */
    confidence: number;
    /**
     * Default detection-gating mode applied to newly added cameras (and to any
     * camera left on the default). See {@link DetectMode}. **Optional**: a
     * backend that predates the feature omits it and (per the settings
     * legacy-drop rule) drops an unknown `default_mode` on PUT — the UI falls
     * back to `"always"` and only sends it when the backend round-trips it.
     */
    default_mode?: DetectMode;
    /**
     * Seconds a label may go unseen before its event is ended. Does not extend
     * the event — `end_time` is the last frame the label was actually seen —
     * but it decides whether a subject that pauses or is briefly hidden counts
     * as one event or several, and the clip is only cut once the event ends.
     */
    absence_timeout_s: number;
    /**
     * Night contrast boost on the DETECTOR's input frame only — never on
     * recordings, clips, live view or the saved snapshot. `auto` boosts only
     * frames darker than `night_boost_threshold` (mean luma 0–255).
     * Optional: absent on a backend that predates it — treat as `"off"`.
     */
    night_boost?: NightBoostMode;
    night_boost_threshold?: number;
    /**
     * Average each tracked object's box over the last N frames. Steadier boxes,
     * at the cost of the box lagging a moving subject and a track lingering a
     * few frames after it leaves. Optional; absent = off.
     */
    smoothing?: boolean;
    smoothing_frames?: number;
  };
  system: {
    public_url: string;
    /**
     * Extra WebRTC ICE host candidates ("ip:8555" entries — the server's LAN
     * and, if used, Tailscale IPs). Empty = go2rtc defaults.
     */
    webrtc_candidates: string[];
    /**
     * Optional nightly restart of the backend. `time` is local 24h "HH:MM" on
     * the NVR. Optional so a backend that predates the feature still parses —
     * treat an absent block as disabled.
     */
    auto_restart?: { enabled: boolean; time: string };
  };
  /**
   * Read-only computed WebRTC readiness (GET/PUT /api/settings). The backend
   * derives it each response from the manual candidates + a best-effort host
   * candidate (env / PUBLIC_URL / auto LAN IP); it is NOT sent back on PUT (the
   * backend ignores it). Absent on a backend that predates the feature — the UI
   * treats a missing block as "unknown" and simply shows no readiness banner.
   */
  webrtc?: WebrtcStatus;
  /**
   * Home Assistant MQTT integration. **Optional** in the document: a backend
   * that predates the integration omits the block entirely, and — per the
   * settings legacy-drop rule — an unknown `mqtt` block sent on PUT is silently
   * dropped. The UI falls back to `DEFAULT_MQTT` when it is absent, so the
   * Integrations tab renders and saves cleanly against either backend.
   */
  mqtt?: MqttSettings;
  /**
   * Optional like `mqtt`, and for the same reason: a backend that predates the
   * cloud archive omits the block, and the Integrations tab must render and
   * save cleanly against either.
   */
  archive?: ArchiveSettings;
}

/**
 * PATCH /api/settings body — a PARTIAL document the backend deep-merges over
 * the stored one. Every subtree is optional; omit what you did not edit.
 *
 * This type is why settings pages may model a subset of the document safely.
 * Under PUT they could not: each page spread a whole snapshot taken once when
 * the shell loaded, so anything changed out-of-band since (a detector model
 * activated via POST /api/detection/models/{key}/activate, another admin's
 * save) got silently reverted by an unrelated page's Save.
 *
 * Deliberately has NO `webrtc` key: that block is computed read-only by the
 * backend (GET injects it) and is discarded on write.
 *
 * Mirrors the iOS `SettingsPatch` (ios/.../Models/Models.swift) — keep the two
 * in step. Omit a subtree to leave it alone; never send an explicit `null`,
 * which `_deep_merge` would write straight through and blank the block.
 */
export type SettingsPatch = {
  notifications?: Partial<AppSettings['notifications']>;
  recording?: Partial<AppSettings['recording']>;
  detection?: Partial<AppSettings['detection']>;
  system?: Partial<AppSettings['system']>;
  mqtt?: Partial<MqttSettings>;
  archive?: Partial<ArchiveSettings>;
};

/**
 * Read-only WebRTC readiness computed by the backend (settings.webrtc).
 * `ready` is false when go2rtc has only a STUN candidate — live view will fall
 * back to slow MSE on the LAN. `detected_ip` (when present) is the server's own
 * detected host to pre-fill the candidate list; `source` says where it came
 * from. `candidates` is the effective list go2rtc is configured with.
 */
export interface WebrtcStatus {
  ready: boolean;
  detected_ip: string | null;
  source: 'env' | 'public_url' | 'auto' | null;
  candidates: string[];
}

/**
 * How Vigilume delivers iOS pushes (docs/push-architecture.md):
 * - `relay` — E2E-encrypted payloads via a public relay holding the app's Apple
 *             key. The ONLY way to a native notification + a CallKit doorbell
 *             ring from a self-hosted server. Needs `relay_url`.
 * - `off`   — no APNs delivery (default; web push and ntfy are unaffected).
 *
 * `direct` (this server holding its own .p8) is RETIRED. A stored `"direct"` is
 * migrated to `off` by the backend before it can reach the pydantic Literal.
 */
export type ApnsMode = 'relay' | 'off';

/**
 * Software Privacy Mode — a per-camera / per-group capture kill switch
 * (`GET`/`POST /api/privacy`): Vigilume stops recording, detecting, streaming
 * and serving snapshots for the camera, while touching nothing on the device
 * itself. This replaced a hardware lens-mask control that drove the camera's
 * own LeLensMask; that is gone, because reconfiguring the camera left state
 * behind that only the camera's own web UI could undo.
 */
/** settings.detection.backend — see AppSettings['detection'].backend. */
/** `auto` picks an Edge TPU when one is fitted, else the GPU. */
export type DetectionBackend = 'auto' | 'gpu' | 'coral';

/** Edge TPU models. Keys match backend app/native/coral.py CORAL_MODELS. */
export type CoralModel =
  | 'ssd_mobilenet_v2'
  | 'ssdlite_mobiledet'
  | 'efficientdet_lite0'
  | 'efficientdet_lite1'
  | 'efficientdet_lite2'
  | 'efficientdet_lite3';

/**
 * Display metadata for the Edge TPU picker. mAP and latency are Coral's
 * published Edge TPU figures, EXCEPT ssdlite_mobiledet, whose 9.6 ms was
 * measured on real hardware. `slow` marks models that cannot sustain ~10
 * inferences/sec — enough to fall behind two cameras at 5 fps.
 */
export const CORAL_MODELS: {
  key: CoralModel;
  label: string;
  map: number;
  latencyMs: number;
  /** Input square, read off the artifact — NEVER inferred from the filename. */
  inputSize: number;
  /** One-line card blurb, matching the GPU tier cards' voice. */
  blurb: string;
  note: string;
  slow?: boolean;
}[] = [
  { key: 'ssd_mobilenet_v2', label: 'SSD MobileNet V2', map: 22.4, latencyMs: 7.6,
    inputSize: 300,
    blurb: 'The fastest option. Lowest accuracy — misses small or partly hidden objects.',
    note: 'fastest, lowest accuracy' },
  { key: 'ssdlite_mobiledet', label: 'SSDLite MobileDet', map: 32.9, latencyMs: 9.6,
    inputSize: 320,
    blurb: 'Near the accuracy of models 4x slower, at almost the speed of the fastest. '
      + 'The right default for a multi-camera box.',
    note: 'best balance — recommended' },
  { key: 'efficientdet_lite0', label: 'EfficientDet-Lite0', map: 25.7, latencyMs: 37.4,
    inputSize: 320,
    blurb: 'A different architecture at the same input size. Slower than MobileDet '
      + 'for less accuracy — mainly useful for comparison.',
    note: '' },
  { key: 'efficientdet_lite1', label: 'EfficientDet-Lite1', map: 30.6, latencyMs: 56.3,
    inputSize: 384,
    blurb: 'A larger input helps distant objects, at roughly 6x MobileDet\'s inference time.',
    note: '' },
  { key: 'efficientdet_lite2', label: 'EfficientDet-Lite2', map: 34.0, latencyMs: 104.6,
    inputSize: 448,
    blurb: 'More accurate, but over 100 ms per frame — under 10 inferences/sec total.',
    note: 'may not keep up', slow: true },
  { key: 'efficientdet_lite3', label: 'EfficientDet-Lite3', map: 37.7, latencyMs: 107.6,
    inputSize: 512,
    blurb: 'The most accurate Edge TPU model offered, and the slowest.',
    note: 'highest accuracy, slowest', slow: true },
];

export interface PrivacyModeState {
  /** Cameras selected directly. */
  cameras: string[];
  /** Camera-group ids selected; every member goes private. */
  groups: number[];
  /** The RESOLVED effective set the backend gates on (direct ∪ group members). */
  private_cameras: string[];
  /** Convenience: is anything private at all. */
  enabled: boolean;
}

/** `settings.notifications.apns` — see docs/push-architecture.md §4. */
export interface ApnsSettings {
  mode: ApnsMode;
  /**
   * Required when mode == "relay". If you run the bundled `push-relay`, this is
   * `http://push-relay:8090` — the Docker-internal name, NOT your public
   * hostname, so push never depends on your tunnel/DNS/internet being up.
   */
  relay_url?: string;
}

/**
 * `settings.notifications.ntfy` — push with NO Apple developer account, via
 * ntfy.sh or a self-hosted ntfy. See docs/push-architecture.md §7. This is
 * a channel BESIDE the APNs relay, not a replacement (no CallKit ring).
 *
 * SECURITY: `topic` is a shared secret, not a name. On a default-allow server
 * (ntfy.sh included) ANYONE who knows the topic receives every notification —
 * and `attach_snapshot` puts a media-token URL in them. Generate an
 * unguessable topic; never let the user pick something like "vigilume".
 */
export interface NtfySettings {
  enabled: boolean;
  /** Base URL, e.g. https://ntfy.sh or your own server. */
  server: string;
  /** `^[A-Za-z0-9_-]{1,64}$` — one path segment. Empty = not configured. */
  topic: string;
  /** ntfy access token (`tk_...`) -> `Authorization: Bearer`. */
  auth_token: string;
  /** ntfy scale: 1 (min) .. 5 (urgent). */
  priority: number;
  /**
   * Link the event snapshot via ntfy's `Attach` header. The PHONE fetches it
   * from this NVR — the image never touches the ntfy server. Requires
   * system.public_url to be reachable from the phone.
   */
  attach_snapshot: boolean;
}

/**
 * One registered iOS device from GET /api/notifications/apns/devices.
 * `device_token_prefix` = first 8 hex chars of the token — enough to
 * disambiguate devices without exposing the full capability.
 */
export interface ApnsDevice {
  device_token_prefix: string;
  device_name: string;
  /** Epoch seconds or ISO string, depending on the backend; render-tolerant. */
  created_at: string | number;
}

/**
 * `settings.mqtt` — Vigilume's outbound MQTT publisher for Home Assistant.
 * When enabled, Vigilume connects to the operator's broker and publishes each
 * camera + its detections with HA MQTT auto-discovery, so they appear in Home
 * Assistant automatically (with optional two-way control of camera features).
 * The whole block round-trips through GET/PUT /api/settings (admin-only).
 */
/**
 * Nightly cloud archive of EVENT media (clips + snapshots) to an rclone remote,
 * one folder per local day. Never touches 24/7 footage.
 */
export interface ArchiveSettings {
  enabled: boolean;
  /** An rclone destination, e.g. "dropbox:Vigilume" or "b2:bucket/events". */
  remote: string;
  /** Local hour (0–23) the nightly pass runs; it uploads the PREVIOUS day. */
  hour: number;
  /** Day folders kept in the cloud. 0 = never expire. Independent of event_days. */
  keep_days: number;
  include_snapshots: boolean;
  /** rclone --bwlimit, e.g. "2M". Empty = unlimited. */
  bwlimit: string;
}

/** One field on an rclone provider's setup form (the catalogue is server-driven). */
export interface RcloneField {
  key: string;
  label: string;
  /** text | secret | token | select — how the UI renders it. */
  kind: string;
  required: boolean;
  help: string;
  placeholder: string;
  options: string[];
  default: string;
}

/** A storage backend the server will configure. */
export interface RcloneProvider {
  /** rclone's own backend name, sent back verbatim on create. */
  type: string;
  label: string;
  blurb: string;
  /**
   * True when the only way in is a browser sign-in. A headless NVR cannot
   * complete that itself, so these show the `authorize_command` to run on the
   * operator's own desktop plus a box to paste the resulting token.
   */
  oauth: boolean;
  authorize_command: string;
  /** Where to create the OAuth app whose key/secret the browser flow needs. */
  console_url: string;
  fields: RcloneField[];
}

/** A configured destination. Secrets are redacted server-side and never sent. */
export interface RcloneRemote {
  name: string;
  type: string;
  label: string;
  oauth: boolean;
  /** Non-secret settings verbatim; secret ones as a fixed marker. */
  details: Record<string, string>;
}

/** What the archive has actually done — GET /api/integrations/archive/status. */
export interface ArchiveStatus {
  /** False on a backend that predates the feature. */
  available: boolean;
  enabled: boolean;
  /**
   * Outcome of the last pass IN THIS PROCESS — cleared by a restart, so an
   * empty object means "nothing has run since boot", not "nothing has ever run".
   */
  last_result: {
    at?: string;
    uploaded_days?: string[];
    files?: number;
    pruned_days?: string[];
    errors?: string[];
  };
  /** The durable watermark: the last day successfully uploaded, or null. */
  last_uploaded_day: string | null;
}

export interface MqttSettings {
  enabled: boolean;
  /** Broker hostname or IP (no scheme), e.g. "192.168.1.5" or "homeassistant.local". */
  host: string;
  /** Broker TCP port (1–65535; MQTT default 1883, TLS 8883). */
  port: number;
  username: string;
  password: string;
  /** HA MQTT discovery topic prefix (Home Assistant's default is "homeassistant"). */
  discovery_prefix: string;
  /** Root topic Vigilume publishes its own state/command topics under (default "vigilume"). */
  base_topic: string;
}

/**
 * The three outcomes POST /api/integrations/mqtt/test distinguishes so the UI
 * can render each inline: connected, broker reachable but credentials rejected,
 * or the broker could not be reached at all.
 */
export type MqttTestStatus = 'ok' | 'auth_failed' | 'unreachable';

/**
 * Result of a live broker connect attempt (POST /api/integrations/mqtt/test).
 * `status` is authoritative; `ok` is a convenience mirror (`status === "ok"`);
 * `detail` is an optional human-readable reason. Tolerant of a minimal backend
 * that returns only `{ok, detail}` — the client derives the status from `ok`.
 */
export interface MqttTestResult {
  ok: boolean;
  status?: MqttTestStatus;
  detail?: string | null;
}

/**
 * Inference backend kind reported by the detector endpoints. `onnx` is the
 * D-FINE ONNX detector (tiered COCO/Objects365 models).
 */
export type DetectorKind = 'onnx' | 'coral';

/**
 * Execution device the active detector runs on. `cuda`/`cpu` belong to the
 * ONNX (D-FINE) backend. `null` while the detector is not loaded or the
 * hardware is unavailable.
 */
export type DetectorDevice = 'cuda' | 'cpu' | 'edgetpu' | null;

/** Summary detector state embedded in /api/system/health. */
export interface DetectorSummary {
  /**
   * Inference backend. Absent on an older backend — callers treat a missing
   * `kind` as `"onnx"` (the historical default).
   */
  kind?: DetectorKind;
  ready: boolean;
  /** Execution device; null while the detector is not loaded. */
  device: DetectorDevice;
  model: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  detector: DetectorSummary;
  go2rtc: boolean;
  cameras_online: number;
}

/** One down window from GET /api/system/camera-health. */
export interface CameraDownWindow {
  start: number;
  end: number;
  seconds: number;
}

/** Per-camera reachability from GET /api/system/camera-health. */
export interface CameraHealthRow {
  camera: string;
  /** RTSP-port uptime % over the window; null when never observed in it. */
  uptime_pct: number | null;
  /** Live reachability right now (null when the prober has no reading). */
  online: boolean | null;
  down_count: number;
  down_seconds: number;
  downs: CameraDownWindow[];
}

export interface CameraHealthReport {
  window: { since: number; until: number; hours: number };
  cameras: CameraHealthRow[];
}

/** Per-camera ingest health from GET /api/system/detector. */
export interface DetectorCameraStatus {
  name: string;
  ingest_ok: boolean;
  fps: number;
  last_frame_age_s: number | null;
}

/** Full detector self-test (GET /api/system/detector). */
export interface DetectorStatus {
  /** Inference backend; absent on an older backend (treat as "onnx"). */
  kind?: DetectorKind;
  ready: boolean;
  device: DetectorDevice;
  model: string;
  /** true = verified+loaded, false = definitive checksum failure, null =
   *  unknown / model (re)loading (SHA not yet confirmed — don't flag as broken). */
  model_sha_ok: boolean | null;
  last_inference_ms: number | null;
  per_camera: DetectorCameraStatus[];
  /**
   * What is ENCODING video, which is a different GPU question from what runs
   * inference. Detection is CUDA-only, so an AMD/Intel iGPU can never do it —
   * but it can do the HEVC→H.264 transcode, and this is the only place that
   * says whether it is. Optional: absent on a backend predating the field.
   */
  transcode?: TranscodeStatus;
}

/** GET /api/system/detector → `transcode`. */
export interface TranscodeStatus {
  /** ffmpeg + ffprobe both present. False means nothing can transcode at all. */
  enabled: boolean;
  /** `h264_nvenc` | `h264_vaapi` | `libx264`, or null when unavailable. */
  encoder: string | null;
  encoder_label: string;
  /** The headline: is a GPU encoding, or is this the CPU? */
  hardware: boolean;
  /**
   * The DRI render node in use. `null` on an AMD/Intel box almost always means
   * `VAAPI_DEVICE` is unset in the server's `.env`, so the container cannot see
   * the iGPU — not that the box has no GPU.
   */
  vaapi_device: string | null;
  nvidia: boolean;
  /** Encoders that failed at RUNTIME and were permanently demoted. */
  failed: string[];
  /** Completed transcodes per encoder — the evidence behind `hardware`. */
  runs: Record<string, { ok: number; failed: number }>;
}

// ---------- Recordings (24/7 continuous footage — the timeline source) ----------

/** One camera's recording availability from GET /api/recordings/cameras. */
export interface RecordingCamera {
  camera: string;
  friendly_name: string;
  has_recordings: boolean;
  /** First segment start, epoch seconds; null when the camera has no footage. */
  earliest: number | null;
  /** Last segment start + segment length, epoch seconds; null when none. */
  latest: number | null;
}

/** A single 10 s recording segment (epoch-second start + nominal duration). */
export interface RecordingSegment {
  start: number;
  duration: number;
}

/** A merged contiguous coverage span (a gap > one segment splits ranges). */
export interface RecordingRange {
  start: number;
  end: number;
}

/**
 * A local day's recording index (GET /api/recordings/{camera}/index?date=).
 * `segments` are that day's 10 s segments (sorted); `ranges` are the merged
 * contiguous coverage spans used to shade the timeline.
 */
export interface RecordingIndex {
  date: string;
  /** Local UTC offset in seconds for the day (informational). */
  tz_offset: number;
  segments: RecordingSegment[];
  ranges: RecordingRange[];
}

// ---------- Detection model manager (tiered download/activate) ----------

/**
 * User-facing grouping/label the backend assigns a model (title-cased for
 * display). Was a fixed `lightweight | balanced | heavy` set; kept open as a
 * string so new tiers (e.g. an extra-heavy COCO tier or an Objects365 model)
 * render without a frontend change. Prefer `DetectionModelInfo.label` for the
 * card title — `tier` is only a coarse grouping hint now.
 */
export type ModelTier = string;

/** Per-model download/verify lifecycle owned by the backend ModelStore. */
export type ModelState = 'absent' | 'downloading' | 'verifying' | 'ready' | 'error';

/**
 * One tier's static metadata + live download/activation state, from
 * GET /api/detection/models. `active` = it is the persisted
 * settings.detection.model; `loaded` = it is the model the detector has
 * actually loaded right now (the two differ mid-switch).
 */
export interface DetectionModelInfo {
  key: DetectionModel;
  tier: ModelTier;
  label: string;
  /** One-line speed/accuracy tradeoff. */
  blurb: string;
  size_bytes: number;
  /** Square inference input edge in pixels (e.g. 640). */
  input_size: number;
  /** Approximate COCO mAP from the design doc. */
  approx_map: number;
  recommended_for: string;
  /**
   * Label vocabulary the model detects. `vocabulary` is a short machine name
   * ("coco" | "objects365"); `num_classes` its class count (80 | 365). Both are
   * optional for forward/backward-compat — the UI derives a sensible label from
   * the model key when the backend omits them.
   */
  vocabulary?: string;
  num_classes?: number;
  state: ModelState;
  /** 0–100; meaningful while state is "downloading"/"verifying". */
  progress_pct: number;
  active: boolean;
  loaded: boolean;
  /** On-disk SHA-256 matches the pin. */
  sha_ok: boolean;
  /** Short failure reason when state === "error". */
  detail?: string | null;
}

export interface DetectionModelsResponse {
  active: DetectionModel;
  /** Execution device the detector is on; null when unloaded / GPU-gated. */
  device: DetectorDevice;
  models: DetectionModelInfo[];
}

/**
 * GET /api/detection/labels — the ACTIVE detector model's label vocabulary, so
 * the per-camera object picker can offer exactly the classes the running model
 * can detect (80 for COCO, 365 for Objects365). Admin-gated like the rest of
 * /api/detection. The UI falls back to a bundled COCO-80 list when this
 * endpoint is unavailable, so nothing breaks if it ships later.
 */
export interface ActiveLabelsResponse {
  /** Active model key the labels belong to. */
  model: string;
  /** Short vocabulary name ("coco" | "objects365"). */
  vocabulary: string;
  /** Number of classes (== labels.length). */
  count: number;
  /** Underscore-safe class names (e.g. "traffic_light"), in model output order. */
  labels: string[];
}

/** 202 body from POST /api/detection/models/{key}/download. */
export interface ModelDownloadAck {
  key: DetectionModel;
  state: ModelState;
  progress_pct: number;
}

/** 202 body from POST /api/detection/models/{key}/activate. */
export interface ModelActivateAck {
  key: DetectionModel;
  state: ModelState;
  active: boolean;
  loaded: boolean;
}

/** Body from DELETE /api/detection/models/{key}. */
export interface ModelDeleteAck {
  key: DetectionModel;
  state: ModelState;
}

/**
 * Live per-model state/progress push on /api/ws (throttled to ~1/sec while
 * downloading). Carries only the dynamic fields; static metadata comes from
 * GET /api/detection/models.
 */
export interface ModelStatusMessage {
  type: 'model_status';
  key: DetectionModel;
  tier: ModelTier;
  state: ModelState;
  progress_pct: number;
  active: boolean;
  loaded: boolean;
}

/** Camera group (dashboard selector bar / TV mode). */
export interface CameraGroup {
  id: number;
  name: string;
  /**
   * Camera names in display order. May reference deleted cameras (the
   * backend tolerates them); filter against the live camera list at render.
   */
  cameras: string[];
  position: number;
}

// ---------- Detection suppressions (reject-to-suppress) ----------

/**
 * A learned suppression from reject-to-suppress: marking an event a false
 * detection teaches the backend to stop alerting on this kind of object at this
 * spot on this camera. Created by POST /api/events/{id}/reject and managed under
 * Settings → Excluded objects. `foot_x`/`foot_y` are the normalized (0..1)
 * foot-center of the rejected box; `has_thumb` says whether a cropped thumbnail
 * was saved (the thumb route 404s when false).
 */
export interface Suppression {
  id: number;
  camera: string;
  label: string;
  foot_x: number;
  foot_y: number;
  has_thumb: boolean;
  /** Epoch seconds. */
  created_at: number;
  thumbnail_url?: string;
}

// ---------- Auth & users (RBAC) ----------

/** Response from POST /api/auth/login. `role`/`username` are present once the
 *  RBAC backend is deployed; a legacy backend returns only `token`. */
export interface LoginResponse {
  token: string;
  role?: Role;
  username?: string;
}

/** GET /api/auth/me — the current session's identity + role. */
export interface MeResponse {
  username: string;
  role: Role;
}

/** A managed (DB-backed) user. The built-in env admin is NOT listed here. */
export interface User {
  id: number;
  username: string;
  role: Role;
  /** Epoch seconds or ISO string, depending on the backend; render-tolerant. */
  created_at: string | number;
}

export type ServerMessage =
  | { type: 'event_new' | 'event_update' | 'event_end' | 'doorbell'; event: NvrEvent }
  | ({ type: 'camera_status' } & Record<string, unknown>)
  /** Camera rows changed server-side — refetch the list. */
  | { type: 'cameras_changed' }
  /** Admin wiped the event log — clear cached event state. */
  | { type: 'events_cleared' }
  /** Detection-model download/activation progress. */
  | ModelStatusMessage;

// ---------- Endpoints ----------

export const api = {
  // Auth
  login: (username: string, password: string) =>
    request<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }, false),
  /** Current identity + role. 404 on a pre-RBAC backend — callers tolerate it. */
  me: () => request<MeResponse>('/api/auth/me'),

  // Users (admin only). Never returns password hashes; the built-in admin is
  // env-controlled and not part of this list.
  users: () => request<User[]>('/api/users'),
  createUser: (body: { username: string; password: string; role?: Role }) =>
    request<User>('/api/users', { method: 'POST', body: JSON.stringify(body) }),
  /** Reset password and/or change role. Cannot target the built-in admin. */
  updateUser: (id: number, patch: { password?: string; role?: Role }) =>
    request<User>(`/api/users/${id}`, { method: 'PUT', body: JSON.stringify(patch) }),
  deleteUser: (id: number) => request<void>(`/api/users/${id}`, { method: 'DELETE' }),

  // Both status endpoints are bounded: a wedged backend must surface as the
  // status cards' error state, not an indefinite "Checking…" spinner.
  health: () =>
    request<HealthStatus>(
      '/api/system/health',
      { signal: AbortSignal.timeout(10_000) },
      false,
    ),
  /** Detector self-test: device, model integrity, per-camera ingest health. */
  detector: () =>
    request<DetectorStatus>('/api/system/detector', { signal: AbortSignal.timeout(10_000) }),

  /** Per-camera reachability history over the last `hours` (default 24). */
  cameraHealth: (hours = 24) =>
    request<CameraHealthReport>(
      `/api/system/camera-health?hours=${hours}`,
      { signal: AbortSignal.timeout(10_000) },
    ),

  // Cameras
  cameras: () => request<Camera[]>('/api/cameras'),
  addCamera: (cam: CameraInput) =>
    request<Camera>('/api/cameras', { method: 'POST', body: JSON.stringify(cam) }),
  updateCamera: (name: string, cam: CameraInput) =>
    request<Camera>(`/api/cameras/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(cam),
    }),
  deleteCamera: (name: string) =>
    request<void>(`/api/cameras/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  /**
   * Persist the global "All cameras" dashboard order. Names not listed keep
   * their relative order after the listed ones (server-side rule).
   */
  setCameraOrder: (names: string[]) =>
    request<void>('/api/cameras/order', { method: 'PUT', body: JSON.stringify({ names }) }),

  /**
   * Restart the backend process (admin). Returns 202 — the restart is
   * SCHEDULED, not done: the API drops for ~15 s while the container's restart
   * policy brings it back. Expect the next few requests to fail; that is the
   * restart working, not an error.
   */
  restartServer: () =>
    request<{ restarting: boolean; detail: string }>('/api/system/restart', { method: 'POST' }),

  /**
   * Software Privacy Mode state. **ADMIN-ONLY** — a viewer gets 403. Only call
   * this from an admin-gated surface (today: PrivacyModeCard, which mounts on
   * the Cameras settings tab, and that tab is absent from VIEWER_TABS).
   *
   * A viewer does NOT need this to render the "Privacy Mode" overlay: the
   * dashboard reads the per-camera `private` flag on {@link Camera} from
   * `GET /api/cameras`, which is any-authenticated. That flag is the RESOLVED
   * effect ("this camera is not being captured") and deliberately does not
   * reveal the camera/group selection behind it.
   *
   * `cameras`/`groups` are what an admin selected; `private_cameras` is the
   * RESOLVED effective set (direct ∪ every member of a selected group), which
   * is what the gates actually enforce and what the UI should reflect.
   */
  privacyMode: () => request<PrivacyModeState>('/api/privacy'),

  /**
   * Set which cameras/groups are in Privacy Mode (admin). Partial: omit a field
   * to leave it unchanged. Returns the new resolved state.
   */
  setPrivacyMode: (body: { cameras?: string[]; groups?: number[] }) =>
    request<PrivacyModeState>('/api/privacy', { method: 'POST', body: JSON.stringify(body) }),

  // Groups (ordered camera sets for the dashboard selector / TV mode)
  groups: () => request<CameraGroup[]>('/api/groups'),
  addGroup: (name: string, cameras: string[] = []) =>
    request<CameraGroup>('/api/groups', {
      method: 'POST',
      body: JSON.stringify({ name, cameras }),
    }),
  /** `cameras` REPLACES the full ordered list (reorder = PUT the new order). */
  updateGroup: (id: number, patch: { name?: string; cameras?: string[]; position?: number }) =>
    request<CameraGroup>(`/api/groups/${id}`, { method: 'PUT', body: JSON.stringify(patch) }),
  deleteGroup: (id: number) => request<void>(`/api/groups/${id}`, { method: 'DELETE' }),
  cameraSettings: (name: string) =>
    request<DeviceSettings>(`/api/cameras/${encodeURIComponent(name)}/settings`),
  updateCameraSettings: (name: string, patch: DeviceSettings) =>
    request<DeviceSettings>(`/api/cameras/${encodeURIComponent(name)}/settings`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),
  /** Spotlight control: {mode, brightness?} per the camera-controls-v2 addendum. */
  light: (name: string, mode: WhiteLightMode, brightness?: number) =>
    request<void>(`/api/cameras/${encodeURIComponent(name)}/light`, {
      method: 'POST',
      body: JSON.stringify(brightness !== undefined ? { mode, brightness } : { mode }),
    }),
  /** Re-probe the camera with stored creds: model detection + capability refresh. */
  probe: (name: string) =>
    request<ProbeResult>(`/api/cameras/${encodeURIComponent(name)}/probe`, { method: 'POST' }),
  siren: (name: string, duration_s = 10) =>
    request<void>(`/api/cameras/${encodeURIComponent(name)}/siren`, {
      method: 'POST',
      body: JSON.stringify({ duration_s }),
    }),
  reboot: (name: string) =>
    request<void>(`/api/cameras/${encodeURIComponent(name)}/reboot`, { method: 'POST' }),
  /**
   * Pan/tilt + preset control for PTZ domes (capability-gated on `ptz`). See
   * {@link PtzRequest}: `move`/`stop` carry a `direction` (+ `speed` for move),
   * the `preset_*` actions carry an `index` (1–3). Returns 204.
   */
  ptz: (name: string, body: PtzRequest) =>
    request<void>(`/api/cameras/${encodeURIComponent(name)}/ptz`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  cameraSnapshotPath: (name: string) =>
    `/api/cameras/${encodeURIComponent(name)}/snapshot.jpg`,
  /**
   * Push-to-talk WebSocket URL. NO ?token= — the JWT rides
   * Sec-WebSocket-Protocol instead (see talkSubprotocols); a query-string token
   * ends up in nginx's error log. Media-scope tokens are rejected server-side.
   * ws/wss follows the page scheme.
   */
  talkUrl: (name: string) => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.host}/api/cameras/${encodeURIComponent(
      name,
    )}/talk`;
  },

  // Events
  events: (q: EventsQuery = {}) => {
    const params = new URLSearchParams();
    if (q.camera) params.set('camera', q.camera);
    if (q.label) params.set('label', q.label);
    if (q.after !== undefined) params.set('after', String(q.after));
    if (q.before !== undefined) params.set('before', String(q.before));
    params.set('limit', String(q.limit ?? 50));
    params.set('offset', String(q.offset ?? 0));
    return request<EventsPage>(`/api/events?${params.toString()}`);
  },
  event: (id: string | number) => request<NvrEventDetail>(`/api/events/${id}`),
  deleteEvent: (id: string | number) =>
    request<void>(`/api/events/${id}`, { method: 'DELETE' }),
  /**
   * ADMIN: mark an event a false detection (reject-to-suppress). Learns a
   * suppression so this kind of object stops alerting at this spot on this
   * camera, then deletes the event. 201 with the created suppression (ignored).
   */
  rejectEvent: (id: string | number) =>
    request<void>(`/api/events/${id}/reject`, { method: 'POST' }),
  /** ADMIN: permanently delete ALL events + their snapshots/clips. Irreversible. */
  deleteAllEvents: () =>
    request<{ deleted: number; files_removed: number }>('/api/events', { method: 'DELETE' }),
  eventSnapshotPath: (id: string | number) => `/api/events/${id}/snapshot.jpg`,
  eventClipPath: (id: string | number) => `/api/events/${id}/clip.mp4`,
  /**
   * Tokened attachment URL for saving an event's clip / annotated snapshot to
   * disk. `?download=1` makes the backend send `Content-Disposition: attachment`
   * with a friendly filename; the media token is inline so the browser can fetch
   * it without headers. Pair with `downloadAttachment` (lib/download.ts).
   */
  eventDownloadUrl: (id: string | number, kind: 'clip' | 'snapshot') =>
    mediaUrl(
      kind === 'clip'
        ? `/api/events/${id}/clip.mp4?download=1`
        : `/api/events/${id}/snapshot.jpg?download=1`,
    ),

  // Recordings (continuous footage — the timeline/scrubber source)
  /** Per-camera recording availability (earliest/latest footage). */
  recordingCameras: () => request<RecordingCamera[]>('/api/recordings/cameras'),
  /** ADMIN: permanently delete ALL continuous recorded footage for every camera. Irreversible. */
  deleteAllRecordings: () =>
    request<{ purged: boolean; camera_dirs_removed: number; recorders_restarted: number }>(
      '/api/recordings',
      { method: 'DELETE' },
    ),
  /** A local day's segments + merged coverage ranges for the timeline bar. */
  recordingIndex: (camera: string, date: string) =>
    request<RecordingIndex>(
      `/api/recordings/${encodeURIComponent(camera)}/index?date=${encodeURIComponent(date)}`,
    ),
  /**
   * HLS VOD playlist URL for the window [start, end] (epoch seconds), with the
   * media token inline so hls.js / a native <video> can fetch it directly; the
   * segment URIs inherit the token server-side. The window is capped to 6 h
   * server-side, so callers should pass a bounded window (we load ~1 h around
   * the playhead).
   */
  recordingPlaylistUrl: (camera: string, start: number, end: number) =>
    mediaUrl(
      `/api/recordings/${encodeURIComponent(camera)}/playlist.m3u8` +
        `?start=${Math.floor(start)}&end=${Math.floor(end)}`,
    ),
  /**
   * Tokened attachment URL that exports the continuous footage in
   * [start, end] (epoch seconds) as one downloadable MP4. The backend
   * transcodes the span, caps it (~30 min) and returns
   * `Content-Disposition: attachment`; the media token is inline. Pair with
   * `downloadAttachment` (lib/download.ts).
   */
  recordingExportUrl: (camera: string, start: number, end: number) =>
    mediaUrl(
      `/api/recordings/${encodeURIComponent(camera)}/export.mp4` +
        `?start=${Math.floor(start)}&end=${Math.floor(end)}`,
    ),

  // Notifications
  vapidPublicKey: () =>
    request<{ key: string }>('/api/notifications/vapid-public-key', {}, false),
  subscribePush: (subscription: PushSubscriptionJSON) =>
    request<void>('/api/notifications/subscribe', {
      method: 'POST',
      body: JSON.stringify(subscription),
    }),
  unsubscribePush: (endpoint: string) =>
    request<void>('/api/notifications/unsubscribe', {
      method: 'POST',
      body: JSON.stringify({ endpoint }),
    }),
  testNotification: () =>
    request<{ push_sent: number }>('/api/notifications/test', {
      method: 'POST',
    }),
  /**
   * Registered iOS (APNs) devices. 404 on a backend that predates APNs —
   * callers skip the device list gracefully.
   */
  apnsDevices: () => request<ApnsDevice[]>('/api/notifications/apns/devices'),

  // Detection models (tiered in-app download/activate manager)
  /** Tier list + active/device; state & progress from the ModelStore. */
  detectionModels: () => request<DetectionModelsResponse>('/api/detection/models'),
  /**
   * The active model's label vocabulary (COCO-80 vs Objects365-365), for the
   * per-camera object picker. Admin-gated; callers fall back to the bundled
   * COCO-80 list when it 404s / errors (see lib/labels.ts).
   */
  detectionLabels: (model?: string) =>
    request<ActiveLabelsResponse>(
      '/api/detection/labels' + (model ? `?model=${encodeURIComponent(model)}` : ''),
    ),
  /** Start (or no-op) a background download; 202 with current state. */
  downloadModel: (key: DetectionModel) =>
    request<ModelDownloadAck>(
      `/api/detection/models/${encodeURIComponent(key)}/download`,
      { method: 'POST' },
    ),
  /**
   * Make {key} the active model (persists settings.detection.model and
   * reconfigures the detector; downloads first if absent). 202.
   */
  activateModel: (key: DetectionModel) =>
    request<ModelActivateAck>(
      `/api/detection/models/${encodeURIComponent(key)}/activate`,
      { method: 'POST' },
    ),
  /** Delete a downloaded, non-active model file (409 if active). */
  deleteModel: (key: DetectionModel) =>
    request<ModelDeleteAck>(`/api/detection/models/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    }),

  // Detection suppressions (reject-to-suppress; Settings → Excluded objects)
  /** ADMIN: learned suppressions, newest first. */
  listSuppressions: () => request<Suppression[]>('/api/detection/suppressions'),
  /** ADMIN: remove a suppression — this kind of object starts alerting again. */
  deleteSuppression: (id: number) =>
    request<void>(`/api/detection/suppressions/${id}`, { method: 'DELETE' }),
  /**
   * Bare path to a suppression's cropped thumbnail for {@link AuthImage} (media
   * auth: Bearer or ?token=). 404s when the suppression has no thumb — AuthImage
   * shows its built-in fallback.
   */
  suppressionThumbPath: (id: number) => `/api/detection/suppressions/${id}/thumb.jpg`,

  // Settings
  settings: () => request<AppSettings>('/api/settings'),
  /**
   * FULL REPLACE. Every omitted field reverts to its pydantic default — an
   * omitted `mqtt.password` or `ntfy.auth_token` is DESTROYED, not preserved.
   * (This is not hypothetical: it is how a stored APNs signing key was lost
   * back when `notifications.apns.direct.p8` existed.)
   *
   * NOT for the UI — use `patchSettings`. A settings page models a SUBSET of
   * the document, so it can only PUT safely by spreading a whole, fresh
   * snapshot, and any snapshot it holds is stale the moment anything changes
   * out-of-band (activating a detector model, another admin saving). Kept
   * because rbac_smoke exercises PUT /api/settings.
   */
  updateSettings: (settings: AppSettings) =>
    request<AppSettings>('/api/settings', { method: 'PUT', body: JSON.stringify(settings) }),
  /**
   * Deep-merge a PARTIAL document (backend `_deep_merge` + the same validation
   * and side-effects as PUT; 422 on invalid). Send ONLY the slice you edited —
   * an omitted subtree is left untouched.
   *
   * Mirrors the iOS `SettingsPatch` (ios/.../Models/Models.swift) so both
   * clients speak one shape. Never send an explicit `null` subtree: it would
   * reach `_deep_merge` and blank that whole block — the same data-destroying
   * shape this exists to avoid. Omit the key instead.
   */
  patchSettings: (patch: SettingsPatch) =>
    request<AppSettings>('/api/settings', { method: 'PATCH', body: JSON.stringify(patch) }),

  // Integrations
  /**
   * Test a live connection to the MQTT broker with the supplied config (the
   * draft the operator is editing — lets them verify before saving). Returns
   * ok / auth_failed / unreachable; a pre-deploy backend without the route
   * surfaces as an ApiError the caller renders inline.
   */
  testMqtt: (mqtt: MqttSettings) =>
    request<MqttTestResult>('/api/integrations/mqtt/test', {
      method: 'POST',
      // The route body is a MqttTestRequest ({ mqtt } — the draft config to
      // test); a bare MqttSettings would be parsed with mqtt=None and silently
      // fall back to the SAVED settings, so the draft must be wrapped.
      body: JSON.stringify({ mqtt }),
    }),

  /**
   * The exact redirect URI to register on the provider's app, derived from the
   * address this browser actually reached the server on. Asked of the server
   * rather than built here so the callback path has one owner.
   */
  rcloneRedirectUri: (origin: string) =>
    request<{ redirect_uri: string; blocked_reason: string }>(
      `/api/integrations/rclone/oauth/redirect-uri?origin=${encodeURIComponent(origin)}`,
    ),

  /**
   * Begin a browser sign-in. Returns the provider URL to send the operator to;
   * the provider then redirects back to this server, which finishes the
   * handshake and writes the remote.
   */
  startRcloneOAuth: (
    name: string,
    type: string,
    clientId: string,
    clientSecret: string,
    origin: string,
  ) =>
    request<{ auth_url: string; redirect_uri: string }>(
      '/api/integrations/rclone/oauth/start',
      {
        method: 'POST',
        body: JSON.stringify({
          name,
          type,
          client_id: clientId,
          client_secret: clientSecret,
          origin,
        }),
      },
    ),

  /** The storage backends the server can configure, with their form fields. */
  rcloneProviders: () =>
    request<{ providers: RcloneProvider[] }>('/api/integrations/rclone/providers'),

  /**
   * Configured remotes. `available: false` means rclone itself could not run —
   * usually a backend image built before the archive feature.
   */
  rcloneRemotes: () =>
    request<{ available: boolean; remotes: RcloneRemote[]; detail: string }>(
      '/api/integrations/rclone/remotes',
    ),

  /**
   * Create (or replace) a remote, then immediately probe it. `ok` means the
   * config was written; `reachable` means the credentials actually work — a
   * remote can save fine and still be wrong, and the UI should say which.
   */
  createRcloneRemote: (name: string, type: string, values: Record<string, string>) =>
    request<{ ok: boolean; reachable: boolean; detail: string; suggested_remote: string }>(
      '/api/integrations/rclone/remotes',
      { method: 'POST', body: JSON.stringify({ name, type, values }) },
    ),

  /** Forget a remote's credentials. Deletes NOTHING in the cloud. */
  deleteRcloneRemote: (name: string) =>
    request<{ ok: boolean; detail: string }>(
      `/api/integrations/rclone/remotes/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    ),

  /** List the remote's top level — the cheapest proof the credentials work. */
  /**
   * Probe a remote. On failure, `detail` is rclone's raw stderr and `hint` is
   * an optional plain-English explanation with the fix — additional to
   * `detail`, never a replacement, so a wrong hint can't hide the real error.
   * `hint` is absent on a backend predating it.
   */
  testRcloneRemote: (name: string) =>
    request<{ ok: boolean; detail: string; hint?: string; folders: string[] }>(
      `/api/integrations/rclone/remotes/${encodeURIComponent(name)}/test`,
      { method: 'POST' },
    ),

  /** What the nightly cloud archive has done. */
  archiveStatus: () => request<ArchiveStatus>('/api/integrations/archive/status'),

  /**
   * Run an archive pass NOW instead of waiting for the configured hour.
   *
   * Slow by design — it is the real pass, not a separate connectivity probe, so
   * a green result is evidence the remote actually works. Callers should show a
   * pending state rather than assuming it hung.
   */
  runArchive: () =>
    request<{ ok: boolean; detail: string; result?: ArchiveStatus['last_result'] }>(
      '/api/integrations/archive/run',
      { method: 'POST' },
    ),
};

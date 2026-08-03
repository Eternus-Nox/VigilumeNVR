/**
 * Global app state for the authenticated shell: camera list + live status,
 * last event per camera, toast notifications, and the unseen-events badge.
 * Fed by REST on mount and by the /api/ws live socket afterwards.
 *
 * Split into a STABLE context plus per-message-type HOT contexts so a WS
 * tick only re-renders the components that display that kind of data:
 * - `useAppState()` — stable state (role, cameras list, stable callbacks).
 *   Its value only changes on login/camera-list changes, so components that
 *   just need `pushToast`/`cameras`/`isAdmin` (Settings tabs, Events page,
 *   every EventCard) do NOT re-render on live socket traffic.
 * - `useCameraLive()` — camera_status ticks only (online/ingest maps).
 * - `useLastEvents()` — event_new/update/end ticks (per-camera last event).
 * - `useModelStatuses()` — model_status ticks (download/activation progress).
 *   Dashboard tiles and camera lists do NOT re-render on these.
 * - `useUiLive()` — toast stack, socket status, unseen-events badge (shell
 *   chrome: Layout + Toasts).
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  api,
  getRole,
  getUsername,
  setRole as persistRole,
  setUsername as persistUsername,
  type Camera,
  type ModelStatusMessage,
  type NvrEvent,
  type Role,
  type ServerMessage,
} from '../lib/api';
import { LiveSocket, type SocketStatus } from '../lib/ws';
import { pluralize, titleCase } from '../lib/format';

export interface Toast {
  id: number;
  title: string;
  body: string;
  url?: string;
  kind: 'event' | 'doorbell' | 'error' | 'info';
}

/**
 * Stable slice: changes only on login/logout, camera CRUD or a cameras
 * refetch. All callbacks here are identity-stable (useCallback with stable
 * deps), so consumers can safely list them in effect/memo deps and hot WS
 * traffic never re-renders a stable-only consumer.
 */
interface StableAppState {
  /** Current session role. Defaults to `admin` for legacy (pre-RBAC) sessions
   *  that carry no cached role, matching the backend's legacy-token handling. */
  role: Role;
  /** Convenience flag: role === 'admin'. Gate admin-only UI + requests on this. */
  isAdmin: boolean;
  /** Logged-in username ("admin" for the built-in admin); null if unknown. */
  username: string | null;
  cameras: Camera[] | null;
  camerasError: string | null;
  refreshCameras: () => Promise<void>;
  pushToast: (t: Omit<Toast, 'id'>) => void;
  dismissToast: (id: number) => void;
  clearUnseen: () => void;
}

/**
 * Hot slice, camera_status ticks only: online/ingest maps. Subscribers
 * (camera tiles, camera lists, TV wall) re-render on status flips but NOT
 * on event or model_status traffic.
 */
interface CameraLiveState {
  /** live overrides from camera_status WS messages, keyed by camera name */
  onlineOverrides: Record<string, boolean>;
  isOnline: (cam: Camera) => boolean;
  /**
   * Live ingest health per camera (detector frame flow), keyed by camera
   * name. Fed by camera_status WS messages; a camera missing here has no
   * live ingest reading yet.
   */
  ingestHealth: Record<string, boolean>;
}

/** Hot slice, event ticks only: the latest event per camera. */
interface EventsLiveState {
  lastEvents: Record<string, NvrEvent>;
}

/** Hot slice, model_status ticks only (Settings → detection models UI). */
interface ModelLiveState {
  /**
   * Latest detection-model download/activation status, keyed by model key.
   * Fed by `model_status` WS messages; `receivedAt` lets consumers ignore a
   * push that predates a fresher REST snapshot.
   */
  modelStatuses: Record<string, ModelStatusMessage & { receivedAt: number }>;
}

/** Hot slice for the app shell: toast stack, conn dot, unseen badge. */
interface UiLiveState {
  socketStatus: SocketStatus;
  toasts: Toast[];
  unseenEvents: number;
}

const StableCtx = createContext<StableAppState | null>(null);
const CameraLiveCtx = createContext<CameraLiveState | null>(null);
const EventsLiveCtx = createContext<EventsLiveState | null>(null);
const ModelLiveCtx = createContext<ModelLiveState | null>(null);
const UiLiveCtx = createContext<UiLiveState | null>(null);

let toastSeq = 1;

interface CameraStatusUpdate {
  online: Record<string, boolean>;
  ingest: Record<string, boolean>;
}

function parseCameraStatus(msg: Record<string, unknown>): CameraStatusUpdate {
  const out: CameraStatusUpdate = { online: {}, ingest: {} };
  const readIngest = (name: string, v: Record<string, unknown>) => {
    if (typeof v.ingest_ok === 'boolean') out.ingest[name] = v.ingest_ok;
  };
  const name = msg.camera ?? msg.name;
  if (typeof name === 'string' && typeof msg.online === 'boolean') {
    out.online[name] = msg.online;
    readIngest(name, msg);
    return out;
  }
  // Alternate shape: {cameras: {front_yard: true, ...}} or top-level map
  const map = (msg.cameras ?? msg.status) as unknown;
  if (map && typeof map === 'object') {
    for (const [k, v] of Object.entries(map as Record<string, unknown>)) {
      if (typeof v === 'boolean') out.online[k] = v;
      else if (v && typeof v === 'object') {
        const obj = v as Record<string, unknown>;
        if (typeof obj.online === 'boolean') out.online[k] = obj.online;
        readIngest(k, obj);
      }
    }
  }
  return out;
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  // Seed role synchronously from the last login so the very first render already
  // knows not to fire admin-only requests as a viewer (no 403 spray); GET
  // /api/auth/me then confirms/refreshes it below.
  const [role, setRole] = useState<Role>(() => getRole() ?? 'admin');
  const [username, setUsername] = useState<string | null>(() => getUsername());
  const [cameras, setCameras] = useState<Camera[] | null>(null);
  const [camerasError, setCamerasError] = useState<string | null>(null);
  const [onlineOverrides, setOnlineOverrides] = useState<Record<string, boolean>>({});
  const [ingestHealth, setIngestHealth] = useState<Record<string, boolean>>({});
  const [modelStatuses, setModelStatuses] = useState<
    Record<string, ModelStatusMessage & { receivedAt: number }>
  >({});
  const [lastEvents, setLastEvents] = useState<Record<string, NvrEvent>>({});
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('closed');
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [unseenEvents, setUnseenEvents] = useState(0);
  const socketRef = useRef<LiveSocket | null>(null);
  const timersRef = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  const dismissToast = useCallback((id: number) => {
    setToasts((ts) => ts.filter((t) => t.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const pushToast = useCallback(
    (t: Omit<Toast, 'id'>) => {
      const id = toastSeq++;
      setToasts((ts) => [...ts.slice(-3), { ...t, id }]);
      timersRef.current.set(
        id,
        setTimeout(() => dismissToast(id), t.kind === 'error' ? 8000 : 6000),
      );
    },
    [dismissToast],
  );

  const refreshCameras = useCallback(async () => {
    try {
      const cams = await api.cameras();
      setCameras(cams);
      setCamerasError(null);
    } catch (e) {
      setCamerasError(e instanceof Error ? e.message : 'Failed to load cameras');
      setCameras((prev) => prev ?? []);
    }
  }, []);

  // Confirm/refresh the session role from the server. Tolerates a pre-RBAC
  // backend (GET /api/auth/me 404s) by keeping the cached/default role. A 401
  // is handled inside the api layer (redirect to login).
  useEffect(() => {
    let alive = true;
    void api
      .me()
      .then((me) => {
        if (!alive) return;
        setRole(me.role);
        setUsername(me.username);
        persistRole(me.role);
        persistUsername(me.username);
      })
      .catch(() => {
        /* endpoint absent (pre-deploy) or transient — keep the cached role */
      });
    return () => {
      alive = false;
    };
  }, []);

  // Initial data: cameras + latest events (to seed per-camera "last event")
  useEffect(() => {
    void refreshCameras();
    void api
      .events({ limit: 50 })
      .then((page) => {
        setLastEvents((prev) => {
          const next = { ...prev };
          // events come newest-first per convention; keep the first seen per camera
          for (const ev of page.events) {
            if (!(ev.camera in next)) next[ev.camera] = ev;
          }
          return next;
        });
      })
      .catch(() => {
        /* dashboard still renders without seeds */
      });
  }, [refreshCameras]);

  // Live socket
  useEffect(() => {
    const socket = new LiveSocket();
    socketRef.current = socket;
    const offStatus = socket.onStatus(setSocketStatus);
    const offMsg = socket.onMessage((msg: ServerMessage) => {
      switch (msg.type) {
        case 'event_new':
        case 'event_update':
        case 'event_end': {
          const ev = msg.event;
          if (!ev || typeof ev.camera !== 'string') return;
          setLastEvents((prev) => ({ ...prev, [ev.camera]: ev }));
          if (msg.type === 'event_new') {
            setUnseenEvents((n) => n + 1);
            pushToast({
              kind: 'event',
              title: `${titleCase(ev.label)} detected`,
              body: `${ev.camera ? titleCase(ev.camera) : ''}${
                ev.count > 0 ? ` — ${ev.count} ${pluralize(ev.label, ev.count)} in frame` : ''
              }`,
              url: `/events/${ev.id}`,
            });
          }
          break;
        }
        case 'doorbell': {
          const ev = msg.event;
          pushToast({
            kind: 'doorbell',
            title: 'Doorbell pressed',
            body: ev?.camera ? titleCase(ev.camera) : '',
            url: ev?.id !== undefined ? `/events/${ev.id}` : undefined,
          });
          break;
        }
        case 'camera_status': {
          const updates = parseCameraStatus(msg as Record<string, unknown>);
          if (Object.keys(updates.online).length > 0) {
            setOnlineOverrides((prev) => ({ ...prev, ...updates.online }));
          }
          if (Object.keys(updates.ingest).length > 0) {
            setIngestHealth((prev) => ({ ...prev, ...updates.ingest }));
          }
          break;
        }
        case 'cameras_changed': {
          // Server-side camera rows changed: refetch so Dashboard tiles and
          // Settings lists pick up new cameras.
          void refreshCameras();
          break;
        }
        case 'events_cleared': {
          // Admin wiped the whole event log: drop the per-camera last-event
          // chips and the unseen badge so live UI matches the now-empty log.
          setLastEvents({});
          setUnseenEvents(0);
          break;
        }
        case 'model_status': {
          setModelStatuses((prev) => ({
            ...prev,
            [msg.key]: { ...msg, receivedAt: Date.now() },
          }));
          break;
        }
      }
    });
    socket.start();
    const timers = timersRef.current;
    return () => {
      offStatus();
      offMsg();
      socket.stop();
      socketRef.current = null;
      for (const t of timers.values()) clearTimeout(t);
      timers.clear();
    };
  }, [pushToast, refreshCameras]);

  const isOnline = useCallback(
    (cam: Camera) => onlineOverrides[cam.name] ?? cam.online,
    [onlineOverrides],
  );

  const clearUnseen = useCallback(() => setUnseenEvents(0), []);

  // Memoized per-slice context values: each WS message type only invalidates
  // its own slice, so e.g. a model_status tick never re-renders camera tiles
  // and an event tick never re-renders Settings pages.
  const stable = useMemo<StableAppState>(
    () => ({
      role,
      isAdmin: role === 'admin',
      username,
      cameras,
      camerasError,
      refreshCameras,
      pushToast,
      dismissToast,
      clearUnseen,
    }),
    [role, username, cameras, camerasError, refreshCameras, pushToast, dismissToast, clearUnseen],
  );

  const cameraLive = useMemo<CameraLiveState>(
    () => ({ onlineOverrides, isOnline, ingestHealth }),
    [onlineOverrides, isOnline, ingestHealth],
  );

  const eventsLive = useMemo<EventsLiveState>(() => ({ lastEvents }), [lastEvents]);

  const modelLive = useMemo<ModelLiveState>(() => ({ modelStatuses }), [modelStatuses]);

  const uiLive = useMemo<UiLiveState>(
    () => ({ socketStatus, toasts, unseenEvents }),
    [socketStatus, toasts, unseenEvents],
  );

  return (
    <StableCtx.Provider value={stable}>
      <CameraLiveCtx.Provider value={cameraLive}>
        <EventsLiveCtx.Provider value={eventsLive}>
          <ModelLiveCtx.Provider value={modelLive}>
            <UiLiveCtx.Provider value={uiLive}>{children}</UiLiveCtx.Provider>
          </ModelLiveCtx.Provider>
        </EventsLiveCtx.Provider>
      </CameraLiveCtx.Provider>
    </StableCtx.Provider>
  );
}

/** Stable app state (role, cameras, stable callbacks). Safe for hot lists. */
export function useAppState(): StableAppState {
  const ctx = useContext(StableCtx);
  if (!ctx) throw new Error('useAppState must be used inside AppStateProvider');
  return ctx;
}

/**
 * Live camera online/ingest status (camera_status ticks). Re-renders the
 * subscriber on status flips only — not on event or model_status traffic.
 */
export function useCameraLive(): CameraLiveState {
  const ctx = useContext(CameraLiveCtx);
  if (!ctx) throw new Error('useCameraLive must be used inside AppStateProvider');
  return ctx;
}

/** Latest event per camera (event_new/update/end ticks). */
export function useLastEvents(): EventsLiveState {
  const ctx = useContext(EventsLiveCtx);
  if (!ctx) throw new Error('useLastEvents must be used inside AppStateProvider');
  return ctx;
}

/** Detection-model download/activation pushes (model_status ticks). */
export function useModelStatuses(): ModelLiveState {
  const ctx = useContext(ModelLiveCtx);
  if (!ctx) throw new Error('useModelStatuses must be used inside AppStateProvider');
  return ctx;
}

/** Shell chrome slice: toasts, socket status, unseen-events badge. */
export function useUiLive(): UiLiveState {
  const ctx = useContext(UiLiveCtx);
  if (!ctx) throw new Error('useUiLive must be used inside AppStateProvider');
  return ctx;
}

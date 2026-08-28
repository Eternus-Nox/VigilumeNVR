/**
 * Settings → Cameras: add/edit/delete cameras (the backend regenerates the
 * go2rtc stream config and reloads the detection/recording engines), the
 * global dashboard camera order (drag handle + arrows, persisted via
 * PUT /api/cameras/order) and per-camera Amcrest device settings.
 * `?device=<name>` deep-links straight into a device settings panel.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  api,
  fetchBlobUrl,
  type AudioCodec,
  type Camera,
  type CameraInput,
  type DetectMode,
  type DeviceSettings,
  type CrossLine,
  type ExemptZone,
  type IncludeZone,
  type NightVisionMode,
  type WhiteLightMode,
} from '../../lib/api';
import { Modal, ConfirmDialog } from '../../components/Modal';
import DetectModeBadge from '../../components/DetectModeBadge';
import ObjectPicker from '../../components/ObjectPicker';
import ReorderList from '../../components/ReorderList';
import { useAppState, useCameraLive } from '../../state/AppState';
import CameraHealthCard from '../../components/CameraHealthCard';
import PrivacyModeCard from '../../components/PrivacyModeCard';
import { amcrestDefaultUrl, titleCase } from '../../lib/format';

const KNOWN_MODELS = ['IP5M-T1277EW-AI', 'IP8M-2779EW-AI', 'AD410', 'IP3M-941B', 'IP4M-1041B', 'IP4M-1056E'];
const DEFAULT_OBJECTS = ['person', 'dog', 'cat', 'car'];
const DEFAULT_DETECT_FPS = 5;
const DEFAULT_SPOTLIGHT_HOLD = 60;
const MIN_SPOTLIGHT_HOLD = 5;
const MAX_SPOTLIGHT_HOLD = 600;

/** Short labels for the per-camera detection-gating modes. */
const DETECT_MODE_LABELS: Record<DetectMode, string> = {
  always: 'Always',
  camera_ai: 'Camera-AI triggered',
  camera_ai_only: 'Camera-AI only',
};

const EMPTY_FORM: CameraInput = {
  name: '',
  friendly_name: '',
  // Auto-detect by default: the backend asks the camera its own model
  // (getDeviceType) and adopts it. Defaulting to a concrete model instead
  // would ASSERT that model and suppress detection entirely.
  model: 'unknown',
  ip: '',
  username: 'admin',
  password: '',
  detect_objects: [...DEFAULT_OBJECTS],
  exempt_zones: [],
  include_zones: [],
  cross_lines: [],
  notify_on_cross: false,
  detect_fps: DEFAULT_DETECT_FPS,
  // Omitted (undefined) so a new camera inherits the global default_mode
  // (Camera-triggered) instead of being pinned to Server. The mode control
  // still lets you override per camera before saving.
  detect_mode: undefined,
  main_url: '',
  sub_url: '',
  // Default codec keeps live-view (WebRTC) audio working; AAC trades it for
  // higher recording quality.
  audio_codec: 'g711a',
  // Off by default; the toggle only surfaces for white-light cameras when
  // editing an existing camera (capabilities are known post-add).
  smart_spotlight: false,
  // Spotlight hold defaults to 60 s; only surfaces alongside Smart spotlight on
  // white-light cameras (same gate).
  spotlight_hold_seconds: DEFAULT_SPOTLIGHT_HOLD,
};

export default function CamerasTab() {
  const { cameras, refreshCameras, pushToast } = useAppState();
  const { isOnline } = useCameraLive();
  const [params, setParams] = useSearchParams();
  const [editing, setEditing] = useState<CameraInput | null>(null);
  const [editingExisting, setEditingExisting] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Camera | null>(null);
  const [busy, setBusy] = useState(false);
  // Optimistic display order between a reorder commit and the refetch.
  const [pendingOrder, setPendingOrder] = useState<string[] | null>(null);
  // Server order (position) unless a just-committed reorder is in flight.
  const displayCams = useMemo(() => {
    if (!cameras) return null;
    if (!pendingOrder) return cameras;
    const byName = new Map(cameras.map((c) => [c.name, c]));
    const ordered = pendingOrder
      .map((n) => byName.get(n))
      .filter((c): c is Camera => c !== undefined);
    for (const cam of cameras) {
      if (!pendingOrder.includes(cam.name)) ordered.push(cam);
    }
    return ordered;
  }, [cameras, pendingOrder]);

  const commitOrder = async (next: Camera[]) => {
    const names = next.map((c) => c.name);
    setPendingOrder(names);
    try {
      await api.setCameraOrder(names);
      await refreshCameras();
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Reorder failed',
        body: err instanceof Error ? err.message : '',
      });
    } finally {
      setPendingOrder(null);
    }
  };

  const deviceName = params.get('device');
  const closeDevice = () => {
    const next = new URLSearchParams(params);
    next.delete('device');
    setParams(next, { replace: true });
  };

  const openAdd = () => {
    setEditingExisting(null);
    setEditing({ ...EMPTY_FORM, detect_objects: [...DEFAULT_OBJECTS] });
  };

  // Stored model of the camera being edited — kept selectable verbatim even
  // when it's not in the known-model list (e.g. probe-adopted "IP5M-T1277EW").
  const storedModel = editingExisting
    ? (cameras?.find((c) => c.name === editingExisting)?.model ?? '')
    : '';

  // The camera being edited (null for the add form). On-camera AI capability
  // gates the Camera-AI detection modes; `ai_active` (when the backend exposes
  // it) drives the live "AI seeing motion now" indicator.
  const editingCam = editingExisting
    ? (cameras?.find((c) => c.name === editingExisting) ?? null)
    : null;
  const aiOnCamera = editingCam?.capabilities.ai_on_camera ?? false;
  // On-demand white-light spotlight gates the "Smart spotlight" toggle — the
  // feature drives AmcrestClient.set_white_light, so it's meaningless without it.
  const hasSpotlight = editingCam?.capabilities.white_light ?? false;

  const openEdit = (cam: Camera) => {
    setEditingExisting(cam.name);
    setEditing({
      name: cam.name,
      friendly_name: cam.friendly_name,
      // The model select has no free-text entry; an empty stored model
      // (shouldn't happen, but tolerate old rows) maps to "unknown".
      model: cam.model || 'unknown',
      ip: cam.ip,
      username: '',
      password: '',
      // Prefill with the camera's stored objects so saving an unrelated edit
      // never silently resets them (DEFAULT_OBJECTS is only for the add form).
      detect_objects: [...(cam.detect_objects ?? DEFAULT_OBJECTS)],
      // Deep-copy so edits to points don't mutate the cached camera list.
      exempt_zones: (cam.exempt_zones ?? []).map((z) => ({
        name: z.name,
        points: z.points.map((p) => [p[0], p[1]] as [number, number]),
      })),
      // Deep-copied for the same reason as exempt_zones above.
      include_zones: (cam.include_zones ?? []).map((z) => ({
        name: z.name,
        points: z.points.map((p) => [p[0], p[1]] as [number, number]),
      })),
      cross_lines: (cam.cross_lines ?? []).map((l) => ({
        name: l.name,
        start: [l.start[0], l.start[1]] as [number, number],
        end: [l.end[0], l.end[1]] as [number, number],
      })),
      notify_on_cross: cam.notify_on_cross ?? false,
      detect_fps: cam.detect_fps ?? DEFAULT_DETECT_FPS,
      detect_mode: cam.detect_mode ?? 'always',
      main_url: cam.main_url ?? '',
      sub_url: cam.sub_url ?? '',
      // Default to G.711 (live-view audio works) when the backend hasn't set it.
      audio_codec: cam.audio_codec ?? 'g711a',
      // Default off when the backend hasn't set it; only editable for
      // white-light cameras (gated in the form below).
      smart_spotlight: cam.smart_spotlight ?? false,
      // Default 60 s when the backend hasn't set it; editable next to Smart
      // spotlight (same white-light gate).
      spotlight_hold_seconds: cam.spotlight_hold_seconds ?? DEFAULT_SPOTLIGHT_HOLD,
    });
  };

  const submitForm = async (e: FormEvent) => {
    e.preventDefault();
    if (!editing || busy) return;
    setBusy(true);
    try {
      if (editingExisting) {
        await api.updateCamera(editingExisting, editing);
      } else {
        await api.addCamera(editing);
      }
      pushToast({
        kind: 'info',
        title: editingExisting ? 'Camera updated' : 'Camera added',
        body: 'Streams are reloading with the new config.',
      });
      setEditing(null);
      await refreshCameras();
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Camera save failed',
        body: err instanceof Error ? err.message : '',
      });
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteCamera(deleting.name);
      pushToast({
        kind: 'info',
        title: 'Camera removed',
        body: 'Streams are reloading.',
      });
      setDeleting(null);
      await refreshCameras();
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Delete failed',
        body: err instanceof Error ? err.message : '',
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-section">
      {/* Camera health above Privacy: an at-a-glance up/down + uptime board,
          same self-contained-card pattern as Privacy Mode. */}
      <CameraHealthCard />

      {/* Privacy Mode: the switch you reach for when you want capture to stop
          NOW, so it stays near the top. */}
      <PrivacyModeCard />

      <div className="section-head">
        <h2>Cameras</h2>
        <button type="button" className="btn btn-primary btn-sm" onClick={openAdd}>
          + Add camera
        </button>
      </div>
      <p className="muted small">
        Adding, editing or removing a camera reloads the stream, detection and recording
        config — live streams drop for a few seconds.
      </p>

      {displayCams === null ? (
        <div className="page-loading">Loading…</div>
      ) : displayCams.length === 0 ? (
        <p className="muted">No cameras configured.</p>
      ) : (
        <>
          {displayCams.length > 1 && (
            <p className="muted small">
              Drag the handle (or use the arrows) to set the dashboard camera order.
            </p>
          )}
          <ReorderList
            items={displayCams}
            itemKey={(c) => c.name}
            onCommit={(next) => void commitOrder(next)}
            itemClassName="camera-row"
            ariaLabel="Cameras in dashboard order"
            renderItem={(cam) => (
              <>
                <span className={`status-dot ${isOnline(cam) ? 'ok' : 'down'}`} />
                <div className="camera-row-info">
                  <strong>
                    {cam.friendly_name || titleCase(cam.name)}
                    {cam.needs_credentials && (
                      <span
                        className="attn-badge"
                        title="No camera credentials stored — edit the camera and enter its own username/password to enable streaming, IR control, siren, doorbell alerts and model detection."
                      >
                        needs password
                      </span>
                    )}
                  </strong>
                  <span className="muted small">
                    {cam.name} · {cam.model} · {cam.ip}
                  </span>
                  <DetectModeBadge
                    mode={cam.detect_mode}
                    aiActive={cam.ai_active}
                    className="detect-badge-tile"
                  />
                </div>
                <div className="camera-row-actions">
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => {
                      const next = new URLSearchParams(params);
                      next.set('device', cam.name);
                      setParams(next, { replace: true });
                    }}
                  >
                    Device
                  </button>
                  <button type="button" className="btn btn-sm" onClick={() => openEdit(cam)}>
                    Edit
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger-ghost"
                    onClick={() => setDeleting(cam)}
                  >
                    Delete
                  </button>
                </div>
              </>
            )}
          />
        </>
      )}

      {editing && (
        <Modal
          title={editingExisting ? `Edit ${editingExisting}` : 'Add camera'}
          onClose={() => setEditing(null)}
          wide
        >
          <form onSubmit={submitForm} className="form-grid">
            <label>
              Name (URL-safe)
              <input
                required
                pattern="[a-z0-9_]+"
                title="lowercase letters, digits and underscores"
                value={editing.name}
                disabled={!!editingExisting}
                onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                placeholder="front_yard"
              />
            </label>
            <label>
              Friendly name
              <input
                required
                value={editing.friendly_name}
                onChange={(e) => setEditing({ ...editing, friendly_name: e.target.value })}
                placeholder="Front Yard"
              />
            </label>
            <label>
              Model
              <select
                required
                value={editing.model}
                onChange={(e) => setEditing({ ...editing, model: e.target.value })}
              >
                {/* Selecting this IS the re-detect action: the backend probes
                    getDeviceType on save and adopts the matching model. */}
                <option value="unknown">Auto-detect (recommended)</option>
                {KNOWN_MODELS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
                {/* Preserve a probe-adopted / legacy stored model verbatim. */}
                {storedModel &&
                  storedModel !== 'unknown' &&
                  !KNOWN_MODELS.includes(storedModel) && (
                    <option value={storedModel}>{storedModel}</option>
                  )}
              </select>
            </label>
            <label>
              IP address
              <input
                required
                value={editing.ip}
                onChange={(e) => setEditing({ ...editing, ip: e.target.value })}
                placeholder="192.168.1.101"
                inputMode="numeric"
              />
            </label>
            <label>
              Username
              <input
                required={!editingExisting}
                autoComplete="off"
                value={editing.username}
                onChange={(e) => setEditing({ ...editing, username: e.target.value })}
                placeholder={editingExisting ? 'unchanged if blank' : 'admin'}
              />
            </label>
            <label>
              Password
              <input
                required={!editingExisting}
                type="password"
                autoComplete="new-password"
                value={editing.password}
                onChange={(e) => setEditing({ ...editing, password: e.target.value })}
                placeholder={editingExisting ? 'unchanged if blank' : ''}
              />
            </label>
            {editingExisting &&
              cameras?.find((c) => c.name === editingExisting)?.needs_credentials && (
                <p className="muted small form-span">
                  Enter the camera&rsquo;s own username/password to enable streaming, IR
                  control, siren, doorbell alerts and model detection.
                </p>
              )}
            <div className="form-span">
              <span className="control-label">Detect objects</span>
              <ObjectPicker
                value={editing.detect_objects ?? []}
                onChange={(v) => setEditing({ ...editing, detect_objects: v })}
              />
              {(editing.detect_objects ?? []).length === 0 && (
                <p className="muted small">
                  No objects selected — this camera records only (no detection).
                </p>
              )}
            </div>
            <div className="form-span">
              <span className="control-label">Camera AI detection</span>
              {(() => {
                // Primary control: a switch. ON = "camera_ai" (server GPU gated
                // to the camera's own AI), OFF = "always" (continuous server
                // GPU). The advanced segmented control below reveals the third
                // mode, "On-camera only" (camera_ai_only). All three write the
                // same `detect_mode` field, so switch + segmented stay in sync.
                const mode: DetectMode = editing.detect_mode ?? 'always';
                const usesCameraAi = mode !== 'always';
                const setMode = (m: DetectMode) =>
                  setEditing({ ...editing, detect_mode: m });
                return (
                  <>
                    <div className="switch-row">
                      <button
                        type="button"
                        role="switch"
                        aria-checked={usesCameraAi}
                        aria-label="Camera AI detection"
                        className={`switch ${usesCameraAi ? 'switch-on' : ''}`}
                        disabled={!aiOnCamera}
                        onClick={() => setMode(usesCameraAi ? 'always' : 'camera_ai')}
                      >
                        <span className="switch-knob" />
                      </button>
                      <span className="switch-label">
                        {usesCameraAi
                          ? 'Server detection runs only when the camera AI fires'
                          : 'Server detection runs continuously'}
                      </span>
                      <DetectModeBadge mode={mode} aiActive={editingCam?.ai_active} />
                    </div>
                    {aiOnCamera ? (
                      <details className="advanced-section" open={mode === 'camera_ai_only'}>
                        <summary>Advanced — where detection runs</summary>
                        <div className="seg" role="group" aria-label="Detection mode">
                          {(
                            [
                              ['always', 'Server'],
                              ['camera_ai', 'Camera-triggered'],
                              ['camera_ai_only', 'On-camera only'],
                            ] as [DetectMode, string][]
                          ).map(([value, label]) => (
                            <button
                              key={value}
                              type="button"
                              className={`seg-btn ${mode === value ? 'seg-on' : ''}`}
                              aria-pressed={mode === value}
                              onClick={() => setMode(value)}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                        <span className="control-hint">
                          <strong>{DETECT_MODE_LABELS.camera_ai}</strong> runs the GPU only
                          when this camera&rsquo;s own AI sees motion — big GPU savings; may
                          miss what the camera AI misses.{' '}
                          <strong>{DETECT_MODE_LABELS.camera_ai_only}</strong> never runs the
                          GPU and surfaces the camera&rsquo;s own AI events directly.
                        </span>
                      </details>
                    ) : (
                      <span className="control-hint">
                        {editingExisting
                          ? 'This camera has no on-board AI, so detection always runs on the server. Re-probe it from the Device panel if it should support SMD / IVS.'
                          : 'On-camera AI capability is detected after the camera is added — add it first, then enable Camera AI detection when editing.'}
                      </span>
                    )}
                  </>
                );
              })()}
            </div>
            <label className="form-span">
              Live-view audio
              <select
                value={editing.audio_codec ?? 'g711a'}
                onChange={(e) =>
                  setEditing({ ...editing, audio_codec: e.target.value as AudioCodec })
                }
              >
                <option value="g711a">G.711 (works in live view)</option>
                <option value="aac">AAC (higher quality, no live audio)</option>
              </select>
              <span className="control-hint">
                <strong>G.711</strong> audio passes through to live view (WebRTC), so you hear
                the camera while watching live. <strong>AAC</strong> records at higher quality
                but can&rsquo;t play in live view.
              </span>
            </label>
            {hasSpotlight && (
              <div className="form-span">
                <label className="row-label">
                  <input
                    type="checkbox"
                    checked={editing.smart_spotlight ?? false}
                    onChange={(e) =>
                      setEditing({ ...editing, smart_spotlight: e.target.checked })
                    }
                  />
                  Smart spotlight
                </label>
                <span className="control-hint">
                  Turn the spotlight on when a person is seen at night; off after the hold below.
                </span>
                <label>
                  Spotlight hold (seconds)
                  <input
                    type="number"
                    min={MIN_SPOTLIGHT_HOLD}
                    max={MAX_SPOTLIGHT_HOLD}
                    step={5}
                    value={editing.spotlight_hold_seconds ?? DEFAULT_SPOTLIGHT_HOLD}
                    onChange={(e) =>
                      setEditing({
                        ...editing,
                        spotlight_hold_seconds: Math.min(
                          MAX_SPOTLIGHT_HOLD,
                          Math.max(
                            MIN_SPOTLIGHT_HOLD,
                            Math.floor(Number(e.target.value) || DEFAULT_SPOTLIGHT_HOLD),
                          ),
                        ),
                      })
                    }
                  />
                  <span className="control-hint">
                    How long the spotlight stays on after the last person.
                  </span>
                </label>
              </div>
            )}
            {editingExisting && (
              <div className="form-span">
                <span className="control-label">Detection zones &amp; lines</span>
                <DetectionGeometryEditor
                  cameraName={editingExisting}
                  exempt={editing.exempt_zones ?? []}
                  include={editing.include_zones ?? []}
                  lines={editing.cross_lines ?? []}
                  notifyOnCross={editing.notify_on_cross ?? false}
                  onChange={(next) => setEditing({ ...editing, ...next })}
                />
                {/* Explainer sits UNDER the picture — keeps the frame the first
                    thing you see instead of pushing it down behind a paragraph. */}
                <p className="muted small">
                  Everything is judged by where an object&rsquo;s <strong>feet</strong> land.
                  &ldquo;Only alert here&rdquo; is an allow-list: draw one and everything outside
                  it stops producing events — the sharp way to ignore a street. &ldquo;Ignore
                  here&rdquo; then punches holes in what is left. A crossing line counts objects
                  passing it in each direction.
                </p>
              </div>
            )}
            <details className="form-span advanced-section">
              <summary>Advanced — streams &amp; detection rate</summary>
              <div className="form-stack">
                <label>
                  Detection frame rate (fps)
                  <input
                    type="number"
                    min={1}
                    max={10}
                    step={1}
                    value={editing.detect_fps ?? DEFAULT_DETECT_FPS}
                    onChange={(e) =>
                      setEditing({
                        ...editing,
                        detect_fps: Math.min(10, Math.max(1, Math.floor(Number(e.target.value) || DEFAULT_DETECT_FPS))),
                      })
                    }
                  />
                  <span className="control-hint">
                    Frames per second analyzed by the detector (1–10; 5 is plenty).
                  </span>
                </label>
                <label>
                  Main stream URL override
                  <input
                    value={editing.main_url ?? ''}
                    onChange={(e) => setEditing({ ...editing, main_url: e.target.value })}
                    placeholder={amcrestDefaultUrl(editing.ip, editing.username, 0)}
                  />
                  <span className="control-hint">
                    Recording &amp; live view. Leave empty to use the Amcrest default shown.
                  </span>
                </label>
                <label>
                  Substream URL override
                  <input
                    value={editing.sub_url ?? ''}
                    onChange={(e) => setEditing({ ...editing, sub_url: e.target.value })}
                    placeholder={amcrestDefaultUrl(editing.ip, editing.username, 1)}
                  />
                  <span className="control-hint">
                    Detection. Leave empty to use the Amcrest default shown.
                  </span>
                </label>
              </div>
            </details>
            <div className="modal-actions form-span">
              <button type="button" className="btn" onClick={() => setEditing(null)} disabled={busy}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? 'Saving…' : editingExisting ? 'Save changes' : 'Add camera'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {deleting && (
        <ConfirmDialog
          title="Remove camera"
          message={`Remove ${deleting.friendly_name || deleting.name}? Recordings already on disk are kept until retention cleanup.`}
          confirmLabel="Remove"
          danger
          busy={busy}
          onConfirm={() => void confirmDelete()}
          onCancel={() => setDeleting(null)}
        />
      )}

      {deviceName && <DeviceSettingsModal name={deviceName} onClose={closeDevice} />}
    </div>
  );
}

// ---------- Detection geometry: exempt zones, include zones, crossing lines ----------

/**
 * Three shapes, one picture. Colour carries the meaning so the frame reads at a
 * glance without consulting the lists: WARM = ignored, COOL = watched, YELLOW =
 * a boundary. Within a kind the palette cycles so two adjacent zones of the
 * same kind stay tellable apart.
 */
const EXEMPT_COLORS = ['#ef4444', '#f97316', '#ec4899'];
const INCLUDE_COLORS = ['#10b981', '#22d3ee', '#3b82f6'];
const LINE_COLORS = ['#facc15', '#a3e635', '#fb923c'];

type GeometryMode = 'exempt' | 'include' | 'line';

const MODE_LABEL: Record<GeometryMode, string> = {
  exempt: 'Ignore here',
  include: 'Only alert here',
  line: 'Crossing line',
};

/**
 * Unit vector pointing to the side a crossing counts as "in".
 *
 * MUST match native/zones.py:line_in_normal. Supervision counts a crossing to
 * the LEFT of the start→end arrow as "in"; in screen coords (y down) that is
 * `(dy, -dx)`, i.e. upward for a line drawn left-to-right. The editor draws
 * this arrow so the direction is a visible property of the line rather than a
 * rule buried in a tooltip.
 */
function inNormal(start: [number, number], end: [number, number]): [number, number] {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const mag = Math.hypot(dx, dy);
  if (mag === 0) return [0, 0];
  return [dy / mag, -dx / mag];
}

/**
 * Draw/edit a camera's detection geometry over a live snapshot. Everything is
 * kept NORMALIZED (0..1) so it survives a resolution change; the SVG overlay
 * renders against the displayed image box. Mouse and touch (pointer events).
 *
 * All three kinds are always VISIBLE, but only the active one takes clicks —
 * an include zone and the exempt zones punched out of it only make sense seen
 * together, and hiding the others while you draw would make composing them
 * guesswork.
 */
function DetectionGeometryEditor({
  cameraName,
  exempt,
  include,
  lines,
  notifyOnCross,
  onChange,
}: {
  cameraName: string;
  exempt: ExemptZone[];
  include: IncludeZone[];
  lines: CrossLine[];
  notifyOnCross: boolean;
  onChange: (next: {
    exempt_zones?: ExemptZone[];
    include_zones?: IncludeZone[];
    cross_lines?: CrossLine[];
    notify_on_cross?: boolean;
  }) => void;
}) {
  const [snapUrl, setSnapUrl] = useState<string | null>(null);
  const [snapError, setSnapError] = useState<string | null>(null);
  const [mode, setMode] = useState<GeometryMode>('exempt');
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [lineStart, setLineStart] = useState<[number, number] | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);

  // Load an authed snapshot as an object URL; revoke it on cleanup.
  useEffect(() => {
    let url: string | null = null;
    let alive = true;
    setSnapUrl(null);
    setSnapError(null);
    fetchBlobUrl(api.cameraSnapshotPath(cameraName))
      .then((u) => {
        if (!alive) {
          URL.revokeObjectURL(u);
          return;
        }
        url = u;
        setSnapUrl(u);
      })
      .catch((e) => {
        if (alive) setSnapError(e instanceof Error ? e.message : 'Snapshot unavailable');
      });
    return () => {
      alive = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [cameraName]);

  // Switching mode abandons whatever was half-drawn: a polygon and a line are
  // not the same shape, so carrying points across would produce nonsense.
  const switchMode = (next: GeometryMode) => {
    setMode(next);
    setDraft([]);
    setLineStart(null);
  };

  const pointAt = useCallback((e: ReactPointerEvent<SVGSVGElement>): [number, number] | null => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return null;
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const y = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    return [Number(x.toFixed(4)), Number(y.toFixed(4))];
  }, []);

  const addPoint = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const pt = pointAt(e);
      if (!pt) return;
      if (mode !== 'line') {
        setDraft((d) => [...d, pt]);
        return;
      }
      // A line is two clicks: the first plants the start, the second finishes
      // it. Finishing immediately (rather than needing a Finish button) keeps
      // the two-click gesture obvious.
      if (lineStart === null) {
        setLineStart(pt);
        return;
      }
      if (lineStart[0] === pt[0] && lineStart[1] === pt[1]) return; // zero-length
      onChange({
        cross_lines: [...lines, { name: `Line ${lines.length + 1}`, start: lineStart, end: pt }],
      });
      setLineStart(null);
    },
    [mode, lineStart, lines, onChange, pointAt],
  );

  const finishZone = () => {
    if (draft.length < 3) return;
    if (mode === 'exempt') {
      onChange({ exempt_zones: [...exempt, { name: `Zone ${exempt.length + 1}`, points: draft }] });
    } else {
      onChange({ include_zones: [...include, { name: `Area ${include.length + 1}`, points: draft }] });
    }
    setDraft([]);
  };

  const toPointsAttr = (pts: [number, number][]) =>
    pts.map(([x, y]) => `${x * 1000},${y * 1000}`).join(' ');

  const polygonList = mode === 'exempt' ? exempt : include;
  const colors = mode === 'exempt' ? EXEMPT_COLORS : INCLUDE_COLORS;
  const replacePolygons = (next: (ExemptZone | IncludeZone)[]) =>
    onChange(mode === 'exempt' ? { exempt_zones: next } : { include_zones: next });

  return (
    <div className="zone-editor">
      <div className="geometry-modes" role="tablist" aria-label="Shape to draw">
        {(['exempt', 'include', 'line'] as GeometryMode[]).map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={mode === m}
            className={`btn btn-sm${mode === m ? ' btn-primary' : ''}`}
            onClick={() => switchMode(m)}
          >
            {MODE_LABEL[m]}
          </button>
        ))}
      </div>

      <div className="zone-frame" ref={frameRef}>
        {snapError ? (
          <div className="zone-frame-empty muted small">Live view unavailable: {snapError}</div>
        ) : !snapUrl ? (
          <div className="zone-frame-empty muted small">Loading live view…</div>
        ) : (
          <img src={snapUrl} alt={`${cameraName} snapshot`} draggable={false} />
        )}
        {snapUrl && (
          <svg
            className="zone-svg"
            viewBox="0 0 1000 1000"
            preserveAspectRatio="none"
            onPointerDown={addPoint}
          >
            {/* Include zones sit UNDER the exempt ones, matching how they
                compose: the watched area first, then the holes cut out of it. */}
            {include.map((z, i) => (
              <polygon
                key={`inc${i}`}
                points={toPointsAttr(z.points)}
                fill={INCLUDE_COLORS[i % INCLUDE_COLORS.length]}
                fillOpacity={mode === 'include' ? 0.28 : 0.14}
                stroke={INCLUDE_COLORS[i % INCLUDE_COLORS.length]}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {exempt.map((z, i) => (
              <polygon
                key={`exc${i}`}
                points={toPointsAttr(z.points)}
                fill={EXEMPT_COLORS[i % EXEMPT_COLORS.length]}
                fillOpacity={mode === 'exempt' ? 0.28 : 0.14}
                stroke={EXEMPT_COLORS[i % EXEMPT_COLORS.length]}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {lines.map((ln, i) => {
              const color = LINE_COLORS[i % LINE_COLORS.length];
              const [nx, ny] = inNormal(ln.start, ln.end);
              const mx = ((ln.start[0] + ln.end[0]) / 2) * 1000;
              const my = ((ln.start[1] + ln.end[1]) / 2) * 1000;
              return (
                <g key={`ln${i}`} opacity={mode === 'line' ? 1 : 0.55}>
                  <line
                    x1={ln.start[0] * 1000}
                    y1={ln.start[1] * 1000}
                    x2={ln.end[0] * 1000}
                    y2={ln.end[1] * 1000}
                    stroke={color}
                    strokeWidth={3}
                    vectorEffect="non-scaling-stroke"
                  />
                  {/* The "in" arrow: which way a crossing counts as coming IN.
                      Same normal the backend uses, so what you see is what it
                      counts. */}
                  <line
                    x1={mx}
                    y1={my}
                    x2={mx + nx * 70}
                    y2={my + ny * 70}
                    stroke={color}
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    vectorEffect="non-scaling-stroke"
                  />
                  <circle cx={mx + nx * 70} cy={my + ny * 70} r={7} fill={color} />
                </g>
              );
            })}
            {draft.length > 0 && (
              <polyline
                points={toPointsAttr(draft)}
                fill="none"
                stroke="#e2e8f0"
                strokeWidth={2}
                strokeDasharray="6 4"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {draft.map(([x, y], i) => (
              <circle key={i} cx={x * 1000} cy={y * 1000} r={7} fill="#e2e8f0" />
            ))}
            {lineStart && (
              <circle cx={lineStart[0] * 1000} cy={lineStart[1] * 1000} r={8} fill="#e2e8f0" />
            )}
          </svg>
        )}
      </div>

      {mode === 'line' ? (
        <>
          <div className="zone-toolbar">
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setLineStart(null)}
              disabled={lineStart === null}
            >
              Cancel line
            </button>
          </div>
          <p className="muted small">
            {lineStart
              ? 'Click again to place the far end of the line.'
              : 'Click once to start the line, again to finish it. The dashed arrow shows which way counts as “in” — draw it the other way round to swap in and out.'}
          </p>
          {lines.length === 0 ? (
            <p className="muted small">No crossing lines.</p>
          ) : (
            <ul className="zone-list">
              {lines.map((ln, i) => (
                <li key={i} className="zone-list-row">
                  <span
                    className="zone-swatch"
                    style={{ background: LINE_COLORS[i % LINE_COLORS.length] }}
                    aria-hidden
                  />
                  <input
                    aria-label={`Line ${i + 1} name`}
                    value={ln.name}
                    placeholder={`Line ${i + 1}`}
                    onChange={(e) =>
                      onChange({
                        cross_lines: lines.map((l, j) =>
                          j === i ? { ...l, name: e.target.value } : l,
                        ),
                      })
                    }
                  />
                  <button
                    type="button"
                    className="btn btn-sm"
                    title="Swap which direction counts as in"
                    onClick={() =>
                      onChange({
                        cross_lines: lines.map((l, j) =>
                          j === i ? { ...l, start: l.end, end: l.start } : l,
                        ),
                      })
                    }
                  >
                    Flip
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger-ghost"
                    onClick={() => onChange({ cross_lines: lines.filter((_, j) => j !== i) })}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
          {/* Only offered once a line exists. The backend ignores this flag on a
              camera with no lines (it fails open rather than muting the camera),
              and showing a toggle that does nothing would be a worse way to say
              the same thing. */}
          {lines.length > 0 && (
            <>
              <label className="row-label">
                <input
                  type="checkbox"
                  checked={notifyOnCross}
                  onChange={(e) => onChange({ notify_on_cross: e.target.checked })}
                />
                Only notify me when a line is crossed
              </label>
              <p className="muted small">
                A much sharper trigger than &ldquo;a person appeared in frame&rdquo;. The
                event, its clip and its snapshot are still recorded either way — this only
                holds back the alert until something actually crosses.
              </p>
            </>
          )}
        </>
      ) : (
        <>
          <div className="zone-toolbar">
            <button
              type="button"
              className="btn btn-sm"
              onClick={finishZone}
              disabled={draft.length < 3}
              title={draft.length < 3 ? 'Add at least 3 points first' : 'Save this polygon'}
            >
              Finish shape ({draft.length})
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setDraft((d) => d.slice(0, -1))}
              disabled={draft.length === 0}
            >
              Undo point
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setDraft([])}
              disabled={draft.length === 0}
            >
              Clear draft
            </button>
          </div>
          {polygonList.length === 0 ? (
            <p className="muted small">
              {mode === 'exempt'
                ? 'No exempt zones — the whole frame is watched.'
                : 'No include zones — the whole frame is watched. Draw one and everything outside it stops producing events.'}
            </p>
          ) : (
            <ul className="zone-list">
              {polygonList.map((z, i) => (
                <li key={i} className="zone-list-row">
                  <span
                    className="zone-swatch"
                    style={{ background: colors[i % colors.length] }}
                    aria-hidden
                  />
                  <input
                    aria-label={`${MODE_LABEL[mode]} ${i + 1} name`}
                    value={z.name}
                    placeholder={`Zone ${i + 1}`}
                    onChange={(e) =>
                      replacePolygons(
                        polygonList.map((p, j) => (j === i ? { ...p, name: e.target.value } : p)),
                      )
                    }
                  />
                  <span className="muted small">{z.points.length} pts</span>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger-ghost"
                    onClick={() => replacePolygons(polygonList.filter((_, j) => j !== i))}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

// ---------- Amcrest device settings ----------

function DeviceSettingsModal({ name, onClose }: { name: string; onClose: () => void }) {
  const { pushToast } = useAppState();
  const [device, setDevice] = useState<DeviceSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setDevice(null);
    setError(null);
    api
      .cameraSettings(name)
      .then(setDevice)
      .catch((e) =>
        setError(e instanceof Error ? e.message : 'Device unreachable'),
      );
  }, [name]);

  const saveDevice = async (e: FormEvent) => {
    e.preventDefault();
    if (!device || busy) return;
    setBusy(true);
    try {
      const saved = await api.updateCameraSettings(name, device);
      setDevice(saved ?? device);
      pushToast({ kind: 'info', title: 'Device settings applied', body: name });
      onClose();
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Device update failed',
        body: err instanceof Error ? err.message : '',
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={`Device settings — ${titleCase(name)}`} onClose={onClose}>
      {error ? (
        <p className="form-error">{error}</p>
      ) : !device ? (
        <div className="page-loading">Contacting camera…</div>
      ) : (
        <form onSubmit={saveDevice} className="form-stack">
          {/* Night vision (auto|color|bw) is the single day/night control on
              every camera — the old IR Auto/On/Off select is retired (both drove
              the same Dahua day/night table). */}
          <label>
            Night vision
            <select
              value={device.night_vision_mode ?? 'auto'}
              onChange={(e) =>
                setDevice({
                  ...device,
                  night_vision_mode: e.target.value as NightVisionMode,
                })
              }
            >
              <option value="auto">Auto</option>
              <option value="color">Full-color</option>
              <option value="bw">IR</option>
            </select>
          </label>
          {device.white_light !== undefined && (
            <label>
              Spotlight
              <select
                value={device.white_light.mode}
                onChange={(e) =>
                  setDevice({
                    ...device,
                    // Guarded by the surrounding `!== undefined` render check.
                    white_light: {
                      ...device.white_light!,
                      mode: e.target.value as WhiteLightMode,
                    },
                  })
                }
              >
                <option value="off">Off</option>
                <option value="on">On</option>
                {/* The LED is an on/off relay (no brightness); "Auto" only when
                    the device actually reports it. */}
                {device.white_light.mode === 'auto' && <option value="auto">Auto</option>}
              </select>
            </label>
          )}
          {device.flip !== undefined && (
            <label className="row-label">
              <input
                type="checkbox"
                checked={device.flip}
                onChange={(e) => setDevice({ ...device, flip: e.target.checked })}
              />
              Flip image 180°
            </label>
          )}
          {device.osd_name !== undefined && (
            <label>
              On-screen display name
              <input
                value={device.osd_name}
                onChange={(e) => setDevice({ ...device, osd_name: e.target.value })}
              />
            </label>
          )}
          {device.motion_detect !== undefined && (
            <label className="row-label">
              <input
                type="checkbox"
                checked={device.motion_detect}
                onChange={(e) => setDevice({ ...device, motion_detect: e.target.checked })}
              />
              On-camera motion detection
            </label>
          )}
          {device.volume?.mic !== undefined && (
            <label>
              Microphone volume: {device.volume.mic}
              <input
                type="range"
                min={0}
                max={100}
                value={device.volume.mic}
                onChange={(e) =>
                  setDevice({
                    ...device,
                    volume: { ...device.volume, mic: Number(e.target.value) },
                  })
                }
              />
            </label>
          )}
          {device.volume?.speaker !== undefined && (
            <label>
              Speaker volume: {device.volume.speaker}
              <input
                type="range"
                min={0}
                max={100}
                value={device.volume.speaker}
                onChange={(e) =>
                  setDevice({
                    ...device,
                    volume: { ...device.volume, speaker: Number(e.target.value) },
                  })
                }
              />
            </label>
          )}
          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? 'Applying…' : 'Apply to device'}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}

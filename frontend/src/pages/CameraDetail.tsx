/**
 * Camera detail: big live view + controls (night vision, siren
 * hold-to-confirm, reboot) plus three capability-aware cards — Spotlight
 * (white_light: mode + debounced brightness), two-way Talk (speaker:
 * press-and-hold PTT over the /talk WebSocket) and Credentials (Save & Test
 * with inline probe result) — and a recent events strip for this camera.
 */
import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  api,
  type DeviceSettings,
  type NightVisionMode,
  type NvrEvent,
  type ProbeResult,
  type WhiteLightMode,
  type WhiteLightState,
} from '../lib/api';
import { TalkSession, type TalkState } from '../lib/talk';
import LivePlayer from '../components/LivePlayer';
import HoldButton from '../components/HoldButton';
import PtzControls from '../components/PtzControls';
import EventCard from '../components/EventCard';
import { ConfirmDialog } from '../components/Modal';
import { useAppState, useCameraLive } from '../state/AppState';
import { amcrestDefaultUrl, titleCase } from '../lib/format';

/**
 * Spotlight modes the web UI offers — Auto/On/Off (matching the iOS app). These
 * EW turrets drive the white LED via coaxialControlIO (a plain on/off relay —
 * no brightness), so the brightness control is gone; the backend maps Auto to
 * the camera's own day/night logic. Order: Auto, On, Off.
 */
const LIGHT_MODES: WhiteLightMode[] = ['auto', 'on', 'off'];
const DEFAULT_LIGHT: WhiteLightState = { mode: 'off', brightness: 0 };

/**
 * Night-vision sensor modes shown on every camera: Auto (camera decides),
 * Full-color (white-LED colour night vision) and IR (black-and-white). Written
 * via `night_vision_mode` on the settings PUT. This is the day/night sensor
 * mode and is now the only day/night control (the IR Auto/On/Off select is
 * retired).
 */
const NIGHT_VISION_MODES: { mode: NightVisionMode; label: string }[] = [
  { mode: 'auto', label: 'Auto' },
  { mode: 'color', label: 'Full-color' },
  { mode: 'bw', label: 'IR' },
];

const TALK_HINTS: Record<TalkState, string> = {
  idle: 'hold to talk',
  connecting: 'connecting…',
  live: 'live — release to stop',
  error: 'release to reset',
};

export default function CameraDetail() {
  const { name = '' } = useParams();
  const navigate = useNavigate();
  const { cameras, pushToast, refreshCameras, isAdmin } = useAppState();
  const { isOnline } = useCameraLive();
  const cam = cameras?.find((c) => c.name === name) ?? null;

  const [device, setDevice] = useState<DeviceSettings | null>(null);
  const [deviceError, setDeviceError] = useState(false);
  const [events, setEvents] = useState<NvrEvent[] | null>(null);
  const [confirmReboot, setConfirmReboot] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  // Spotlight (white_light capability)
  const [light, setLight] = useState<WhiteLightState>(DEFAULT_LIGHT);

  // Push-to-talk (speaker capability): mic audio is streamed over the /talk
  // WebSocket and the backend delivers it to the camera — RTSP backchannel for
  // backchannel-capable cameras (e.g. the AD410 doorbell), CGI postAudio for the
  // rest. The live player stays receive-only; talk never touches its WebRTC PC.
  const [talkState, setTalkState] = useState<TalkState>('idle');
  const talkRef = useRef<TalkSession | null>(null);
  const secure = window.isSecureContext;

  // Credentials card
  const [credUser, setCredUser] = useState('');
  const [credPass, setCredPass] = useState('');
  const [testing, setTesting] = useState(false);
  const [probe, setProbe] = useState<ProbeResult | null>(null);

  // Stream & detection card (detect_fps + RTSP overrides)
  const [advFps, setAdvFps] = useState(5);
  const [advMain, setAdvMain] = useState('');
  const [advSub, setAdvSub] = useState('');
  const [advSaving, setAdvSaving] = useState(false);

  const caps = cam?.capabilities;

  // GET /api/cameras/{name}/settings is admin-only — viewers must never call it.
  const loadDevice = useCallback(() => {
    if (!name || !isAdmin) return;
    api
      .cameraSettings(name)
      .then((d) => {
        setDevice(d);
        setDeviceError(false);
        if (d.white_light) setLight(d.white_light);
      })
      .catch(() => setDeviceError(true));
  }, [name, isAdmin]);

  // Seed the stream/detection form once the camera row is known (and reseed
  // after a save refetch so the form tracks the stored values).
  const advSeed = cam ? `${cam.name}|${cam.detect_fps}|${cam.main_url}|${cam.sub_url}` : null;
  useEffect(() => {
    if (!cam) return;
    setAdvFps(cam.detect_fps ?? 5);
    setAdvMain(cam.main_url ?? '');
    setAdvSub(cam.sub_url ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reseed only when stored values change
  }, [advSeed]);

  useEffect(() => {
    setDevice(null);
    setEvents(null);
    setLight(DEFAULT_LIGHT);
    setCredUser('');
    setCredPass('');
    setProbe(null);
    setTalkState('idle');
    loadDevice();
    api
      .events({ camera: name, limit: 12 })
      .then((page) => setEvents(page.events))
      .catch(() => setEvents([]));
    return () => {
      // Leaving the page (or switching cameras) must stop the mic + socket.
      talkRef.current?.stop();
      talkRef.current = null;
    };
  }, [name, loadDevice]);

  if (cameras !== null && !cam) {
    return (
      <div className="page">
        <div className="empty-state">
          <h2>Camera not found</h2>
          <Link to="/" className="btn">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }
  if (!cam) return <div className="page-loading">Loading…</div>;

  const online = isOnline(cam);

  // Spotlight always offers all three modes (Auto/On/Off) so web matches iOS and
  // the IR control. The backend maps Auto to the camera's own day/night logic.
  const lightModes: WhiteLightMode[] = LIGHT_MODES;

  // Night-vision sensor mode — the single day/night control on every camera
  // (the old IR Auto/On/Off select is retired; both drove the same Dahua
  // day/night table). Optimistic; reverts on failure. Writes the day/night
  // table via the settings PUT.
  const setNightVision = async (mode: NightVisionMode) => {
    const prev = device;
    setDevice((d) => (d ? { ...d, night_vision_mode: mode } : d));
    try {
      await api.updateCameraSettings(cam.name, { night_vision_mode: mode });
    } catch (e) {
      setDevice(prev);
      pushToast({ kind: 'error', title: 'Night vision failed', body: errMsg(e) });
    }
  };

  // ----- Spotlight -----

  // On/off (and Auto when supported) only — the LED is a coaxialControlIO relay
  // with no brightness, so the /light call never carries a brightness value.
  const setLightMode = (mode: WhiteLightMode) => {
    setLight((l) => ({ ...l, mode }));
    void api.light(cam.name, mode).catch((e) => {
      pushToast({ kind: 'error', title: 'Spotlight failed', body: errMsg(e) });
    });
  };

  // ----- Push-to-talk -----

  const startTalk = () => {
    if (talkRef.current) return;
    const session = new TalkSession(api.talkUrl(cam.name), {
      onState: setTalkState,
      onFault: (f) => pushToast({ kind: 'error', title: f.title, body: f.body }),
    });
    talkRef.current = session;
    void session.start();
  };

  const stopTalk = () => {
    talkRef.current?.stop();
    talkRef.current = null;
    setTalkState('idle');
  };

  // ----- Credentials -----

  const saveAndTest = async (e: FormEvent) => {
    e.preventDefault();
    if (testing) return;
    setTesting(true);
    setProbe(null);
    try {
      // Blank username/password keep the stored credentials (PUT semantics).
      await api.updateCamera(cam.name, {
        name: cam.name,
        friendly_name: cam.friendly_name,
        model: cam.model,
        ip: cam.ip,
        username: credUser,
        password: credPass,
      });
      const result = await api.probe(cam.name);
      setProbe(result);
      if (result.ok) {
        setCredPass('');
        // Pick up adopted model / refreshed capabilities / cleared
        // needs_credentials without a reload.
        await refreshCameras();
        loadDevice();
      }
    } catch (err) {
      pushToast({ kind: 'error', title: 'Credentials save failed', body: errMsg(err) });
    } finally {
      setTesting(false);
    }
  };

  // ----- Stream & detection -----

  const saveAdvanced = async (e: FormEvent) => {
    e.preventDefault();
    if (advSaving) return;
    setAdvSaving(true);
    try {
      // Blank username/password keep the stored credentials (PUT semantics).
      await api.updateCamera(cam.name, {
        name: cam.name,
        friendly_name: cam.friendly_name,
        model: cam.model,
        ip: cam.ip,
        username: '',
        password: '',
        detect_fps: advFps,
        main_url: advMain.trim(),
        sub_url: advSub.trim(),
      });
      pushToast({
        kind: 'info',
        title: 'Stream settings saved',
        body: 'Streams are reloading with the new config.',
      });
      await refreshCameras();
    } catch (err) {
      pushToast({ kind: 'error', title: 'Stream settings failed', body: errMsg(err) });
    } finally {
      setAdvSaving(false);
    }
  };

  const fireSiren = async () => {
    setBusy('siren');
    try {
      await api.siren(cam.name, 10);
      pushToast({ kind: 'info', title: 'Siren triggered', body: '10 seconds' });
    } catch (e) {
      pushToast({ kind: 'error', title: 'Siren failed', body: errMsg(e) });
    } finally {
      setBusy(null);
    }
  };

  const doReboot = async () => {
    setBusy('reboot');
    try {
      await api.reboot(cam.name);
      pushToast({ kind: 'info', title: 'Rebooting camera', body: 'It will be back in a minute or two.' });
      setConfirmReboot(false);
    } catch (e) {
      pushToast({ kind: 'error', title: 'Reboot failed', body: errMsg(e) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="page camera-detail">
      <div className="page-head">
        <div>
          <h1>{cam.friendly_name || titleCase(cam.name)}</h1>
          <p className="muted">
            <span className={`status-dot ${online ? 'ok' : 'down'}`} /> {online ? 'Online' : 'Offline'} ·{' '}
            {cam.model} · {cam.ip}
          </p>
        </div>
        {isAdmin && (
          <Link className="btn btn-sm" to={`/settings/cameras?device=${encodeURIComponent(cam.name)}`}>
            Device settings
          </Link>
        )}
      </div>

      <div className="camera-detail-video">
        {/* Privacy Mode wins over every other state, INCLUDING offline — same
            rule as the Dashboard tile and TV mode. The backend has removed this
            camera's go2rtc streams, so the player would spin forever and read
            as a fault rather than as the deliberate choice it is. */}
        {cam.private ? (
          <div className="camera-private camera-private-big">
            <span className="camera-private-title">Privacy Mode</span>
            <span className="camera-private-sub">
              not recording, detecting or streaming
            </span>
          </div>
        ) : online ? (
          <LivePlayer camera={cam.name} controls />
        ) : (
          <div className="camera-offline camera-offline-big">
            <span>Camera offline</span>
          </div>
        )}
      </div>

      {/* All device controls + settings cards are admin-only. A viewer sees
          just the live view and the recent-events strip below. The cards are
          stacked in the same order as the iOS camera-detail screen: PTZ, talk,
          the grouped Controls card, then the device-settings cards. */}
      {isAdmin && (
      <div className="detail-cards">
        {caps?.ptz && (
          <section className="card" aria-label="Pan and tilt">
            <div className="card-head">
              <span className="card-icon" aria-hidden="true">🎮</span>
              <h2>Pan / tilt</h2>
            </div>
            <PtzControls
              camera={cam.name}
              onError={(title, body) => pushToast({ kind: 'error', title, body })}
            />
          </section>
        )}

        {/* Two-way talk is gated on `caps.speaker`: only cameras with an
            actual talk speaker (e.g. AD410 doorbell) get a Talk control —
            speakerless cameras (the EW turrets) render nothing here. Every
            speaker camera uses the same client pipeline: mic -> /talk WS
            (lib/talk.ts). The backend picks delivery per camera (RTSP
            backchannel for backchannel-capable cameras, CGI postAudio for the
            rest); the client is identical either way. */}
        {caps?.speaker ? (
          <section className="card" aria-label="Two-way talk">
            <div className="card-head">
              <span className="card-icon" aria-hidden="true">🎙️</span>
              <h2>Talk</h2>
            </div>
            {secure ? (
              <>
                <HoldButton
                  className={`talk-btn talk-${talkState}`}
                  onPressStart={startTalk}
                  onPressEnd={stopTalk}
                  hint={TALK_HINTS[talkState]}
                >
                  🎙 Hold to talk
                </HoldButton>
                <p className="muted small">
                  Hold the button (or hold Space) and speak — your voice plays from the camera.
                </p>
              </>
            ) : (
              <>
                <button type="button" className="btn" disabled>
                  🎙 Hold to talk
                </button>
                <p className="muted small">Requires HTTPS (see docs/remote-access.md).</p>
              </>
            )}
          </section>
        ) : null}

        {/* Grouped Controls card — mirrors the iOS controlsCard: night vision,
            spotlight (white_light), siren and maintenance/reboot all live in a
            single rounded card, each as a labelled control block. */}
        <section
          className={`card controls-card${online ? '' : ' is-offline'}`}
          aria-label="Camera controls"
        >
          <div className="card-head">
            <span className="card-icon" aria-hidden="true">🎛️</span>
            <h2>Controls</h2>
          </div>

          {!online && (
            <p className="control-offline">Camera is offline — controls are unavailable.</p>
          )}

          {/* Night vision (Auto / Full-color / IR) is the single day/night
              control on every camera — it replaces the retired IR select. */}
          <div className="control-block">
            <span className="ctl-label">
              <span className="ctl-ic" aria-hidden="true">🌙</span> Night vision
            </span>
            <div className="seg seg-full" role="group" aria-label="Night vision mode">
              {NIGHT_VISION_MODES.map(({ mode, label }) => {
                const current = device?.night_vision_mode ?? 'auto';
                return (
                  <button
                    key={mode}
                    type="button"
                    className={`seg-btn${current === mode ? ' seg-on' : ''}`}
                    aria-pressed={current === mode}
                    disabled={!device && !deviceError}
                    onClick={() => void setNightVision(mode)}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            {deviceError && <span className="control-hint">device unreachable</span>}
          </div>

          {caps?.white_light && (
            <div className="control-block">
              <span className="ctl-label">
                <span className="ctl-ic" aria-hidden="true">💡</span> Spotlight
              </span>
              <div className="seg seg-full" role="group" aria-label="Spotlight mode">
                {lightModes.map((m) => (
                  <button
                    key={m}
                    type="button"
                    className={`seg-btn${light.mode === m ? ' seg-on' : ''}`}
                    aria-pressed={light.mode === m}
                    onClick={() => setLightMode(m)}
                  >
                    {titleCase(m)}
                  </button>
                ))}
              </div>
              {deviceError && (
                <span className="control-hint">device unreachable — state may be stale</span>
              )}
            </div>
          )}

          {caps?.siren && (
            <div className="control-block">
              <span className="ctl-label">
                <span className="ctl-ic" aria-hidden="true">📣</span> Siren
              </span>
              <HoldButton onConfirm={() => void fireSiren()} disabled={busy === 'siren'}>
                🚨 Hold to sound siren
              </HoldButton>
            </div>
          )}

          <div className="control-block">
            <span className="ctl-label">
              <span className="ctl-ic" aria-hidden="true">🔧</span> Maintenance
            </span>
            <button
              type="button"
              className="btn btn-danger-ghost"
              onClick={() => setConfirmReboot(true)}
            >
              Reboot camera
            </button>
          </div>
        </section>

        <section
          className={`card${cam.needs_credentials ? ' card-attn' : ''}`}
          aria-label="Camera credentials"
        >
          <div className="card-head">
            <span className="card-icon" aria-hidden="true">🔑</span>
            <h2>Credentials</h2>
            {cam.needs_credentials && (
              <span
                className="attn-badge"
                title="No camera credentials stored — streaming, device controls and model detection need the camera's own username/password."
              >
                needs password
              </span>
            )}
          </div>
          <p className="muted small">
            The camera&rsquo;s own username/password — used for night vision, spotlight, siren,
            talk and model detection. Blank fields keep the stored values.
          </p>
          <form className="cred-form" onSubmit={(e) => void saveAndTest(e)}>
            <label>
              Username
              <input
                autoComplete="off"
                value={credUser}
                onChange={(e) => setCredUser(e.target.value)}
                placeholder="unchanged if blank"
              />
            </label>
            <label>
              Password
              <input
                type="password"
                autoComplete="new-password"
                value={credPass}
                onChange={(e) => setCredPass(e.target.value)}
                placeholder="unchanged if blank"
              />
            </label>
            <div className="row-inline wrap">
              <button type="submit" className="btn btn-primary" disabled={testing}>
                {testing ? 'Testing…' : 'Save & Test'}
              </button>
              {probe &&
                (probe.ok ? (
                  <span className="probe-result probe-ok">
                    ✓ Connected — model {probe.model || 'unknown'}
                  </span>
                ) : (
                  <span className="probe-result probe-err">✕ {probe.detail || 'Probe failed'}</span>
                ))}
            </div>
          </form>
        </section>

        <section className="card" aria-label="Stream and detection settings">
          <div className="card-head">
            <span className="card-icon" aria-hidden="true">🎞️</span>
            <h2>Stream &amp; detection</h2>
          </div>
          <details className="advanced-section">
            <summary>Advanced — detection rate &amp; RTSP overrides</summary>
            <form className="form-stack" onSubmit={(e) => void saveAdvanced(e)}>
              <label>
                Detection frame rate (fps)
                <input
                  type="number"
                  min={1}
                  max={10}
                  step={1}
                  value={advFps}
                  onChange={(e) =>
                    setAdvFps(Math.min(10, Math.max(1, Math.floor(Number(e.target.value) || 5))))
                  }
                />
                <span className="control-hint">
                  Frames per second analyzed by the detector (1–10; 5 is plenty).
                </span>
              </label>
              <label>
                Main stream URL override
                <input
                  value={advMain}
                  onChange={(e) => setAdvMain(e.target.value)}
                  placeholder={amcrestDefaultUrl(cam.ip, '', 0)}
                />
                <span className="control-hint">
                  Recording &amp; live view. Leave empty to use the Amcrest default shown.
                </span>
              </label>
              <label>
                Substream URL override
                <input
                  value={advSub}
                  onChange={(e) => setAdvSub(e.target.value)}
                  placeholder={amcrestDefaultUrl(cam.ip, '', 1)}
                />
                <span className="control-hint">
                  Detection. Leave empty to use the Amcrest default shown.
                </span>
              </label>
              <div className="row-inline">
                <button type="submit" className="btn btn-primary" disabled={advSaving}>
                  {advSaving ? 'Saving…' : 'Save stream settings'}
                </button>
              </div>
            </form>
          </details>
        </section>
      </div>
      )}

      <section aria-label="Recent events">
        <div className="section-head">
          <h2>
            <span className="card-icon" aria-hidden="true">🔔</span> Recent events
          </h2>
          <Link to={`/events?camera=${encodeURIComponent(cam.name)}`} className="link">
            View all →
          </Link>
        </div>
        {events === null ? (
          <div className="page-loading">Loading events…</div>
        ) : events.length === 0 ? (
          <p className="muted">No events recorded for this camera yet.</p>
        ) : (
          <div className="event-strip">
            {events.map((ev) => (
              <EventCard key={String(ev.id)} event={ev} compact />
            ))}
          </div>
        )}
      </section>

      {confirmReboot && (
        <ConfirmDialog
          title="Reboot camera"
          message={`Reboot ${cam.friendly_name || cam.name}? The stream will drop for a minute or two.`}
          confirmLabel="Reboot"
          danger
          busy={busy === 'reboot'}
          onConfirm={() => void doReboot()}
          onCancel={() => setConfirmReboot(false)}
        />
      )}

      {!online && (
        <div className="banner">
          <span>This camera appears offline. Live view and controls may fail.</span>
          <button type="button" className="btn btn-sm" onClick={() => navigate(0)}>
            Refresh
          </button>
        </div>
      )}
    </div>
  );
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : 'Request failed';
}

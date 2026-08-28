/**
 * Settings → System: live health, detector self-test (device, model
 * integrity, per-camera ingest), WebRTC candidate addresses and the public
 * URL used in push click-links.
 */
import { useCallback, useEffect, useState } from 'react';
import { api, type DetectorStatus, type HealthStatus, type WebrtcStatus } from '../../lib/api';
import { describeDetector } from '../../lib/detector';
import ChipsInput from '../../components/ChipsInput';
import { ConfirmDialog } from '../../components/Modal';
import ReportBugCard from '../../components/ReportBugCard';
import SupportCard from '../../components/SupportCard';
import { useAppState } from '../../state/AppState';
import { useAdoptSaved, type TabProps } from '../Settings';

/**
 * Loose "IP-ish host[:port]" check for WebRTC candidate entries: IPv4,
 * hostname or bracketed IPv6, with an optional numeric port. The backend only
 * enforces entry length (≤64) — this keeps obvious typos out of the go2rtc
 * ICE config without rejecting unusual-but-valid hosts.
 */
const IP_ISH = /^(\[[0-9A-Fa-f:.]+\]|[0-9A-Za-z][0-9A-Za-z._-]*)(:\d{1,5})?$/;
const isCandidateEntry = (v: string) => v.length <= 64 && IP_ISH.test(v);

/** Human labels for where the backend detected the host candidate. */
const SOURCE_LABEL: Record<NonNullable<WebrtcStatus['source']>, string> = {
  env: 'VIGILUME_WEBRTC_HOST',
  public_url: 'the public URL',
  auto: 'the host LAN IP',
};

/** Give a detected host an `:8555` port when it doesn't already carry one. */
const withWebrtcPort = (host: string) => (/:\d{1,5}$/.test(host) ? host : `${host}:8555`);

const DEFAULT_AUTO_RESTART = { enabled: false, time: '04:00' };

export default function SystemTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft so tab switches keep unsaved edits.
  const [publicUrl, setPublicUrl] = useState(
    pending.system?.public_url ?? settings.system.public_url,
  );
  const [candidates, setCandidates] = useState<string[]>(
    pending.system?.webrtc_candidates ?? settings.system.webrtc_candidates ?? [],
  );
  const [autoRestart, setAutoRestart] = useState(
    pending.system?.auto_restart ?? settings.system.auto_restart ?? DEFAULT_AUTO_RESTART,
  );
  const [restarting, setRestarting] = useState(false);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [detector, setDetector] = useState<DetectorStatus | null>(null);
  const [detectorError, setDetectorError] = useState(false);
  const [detectorBusy, setDetectorBusy] = useState(false);

  // Danger zone: irreversible bulk deletes (admin-only — this whole tab is
  // only mounted for admins, and the backend re-checks with require_admin).
  const { pushToast, clearUnseen } = useAppState();
  const [confirming, setConfirming] = useState<null | 'events' | 'recordings'>(null);
  const [purging, setPurging] = useState(false);

  // Report this tab's slice up on every edit; the shell's single Save persists
  // it with every other tab's pending changes.
  useEffect(() => {
    onDraftChange({
      system: {
        ...settings.system,
        public_url: publicUrl.trim(),
        webrtc_candidates: candidates.map((c) => c.trim()).filter(Boolean),
        auto_restart: { enabled: autoRestart.enabled, time: autoRestart.time },
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicUrl, candidates, autoRestart, onDraftChange]);

  /**
   * Restart the backend NOW. Deliberately not batched behind Save: it is an
   * action, not a setting, and hiding it behind a Save button would make it
   * unclear whether pressing Save also reboots the server.
   */
  const runRestart = async () => {
    setConfirmRestart(false);
    setRestarting(true);
    try {
      await api.restartServer();
      pushToast({
        kind: 'info',
        title: 'Restarting',
        body: 'The server is coming back — this page will reconnect shortly.',
      });
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Restart failed',
        body: e instanceof Error ? e.message : '',
      });
    } finally {
      // Keep the button disabled a moment: the API is about to drop, and an
      // immediate re-press would just error against a dying process.
      window.setTimeout(() => setRestarting(false), 8000);
    }
  };

  const runPurge = async () => {
    if (!confirming) return;
    setPurging(true);
    try {
      if (confirming === 'events') {
        const { deleted } = await api.deleteAllEvents();
        // Optimistic: zero the unseen badge now rather than waiting on the WS
        // 'events_cleared' echo (which the initiator can miss mid-reconnect).
        clearUnseen();
        pushToast({
          kind: 'info',
          title: 'All events deleted',
          body: `${deleted} event${deleted === 1 ? '' : 's'} removed.`,
        });
      } else {
        await api.deleteAllRecordings();
        pushToast({
          kind: 'info',
          title: 'All recordings deleted',
          body: 'Continuous footage cleared; recording resumed.',
        });
      }
      setConfirming(null);
    } catch (e) {
      pushToast({
        kind: 'error',
        title: 'Delete failed',
        body: e instanceof Error ? e.message : 'Unknown error',
      });
    } finally {
      setPurging(false);
    }
  };

  useAdoptSaved(settings.system.public_url, setPublicUrl);
  useAdoptSaved(settings.system.webrtc_candidates, setCandidates);
  useAdoptSaved(settings.system.auto_restart ?? DEFAULT_AUTO_RESTART, setAutoRestart);

  useEffect(() => {
    let active = true;
    const poll = () => {
      api
        .health()
        .then((h) => {
          if (active) {
            setHealth(h);
            setHealthError(false);
          }
        })
        .catch(() => active && setHealthError(true));
    };
    poll();
    const t = setInterval(poll, 15_000);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, []);

  const refreshDetector = useCallback(async () => {
    setDetectorBusy(true);
    try {
      setDetector(await api.detector());
      setDetectorError(false);
    } catch {
      setDetectorError(true);
    } finally {
      setDetectorBusy(false);
    }
  }, []);

  useEffect(() => {
    void refreshDetector();
  }, [refreshDetector]);

  const pill = (ok: boolean, label: string) => (
    <span className={`pill ${ok ? 'pill-ok' : 'pill-down'}`}>
      {label}: {ok ? 'up' : 'down'}
    </span>
  );

  // Read-only WebRTC readiness the backend computes each settings response.
  const webrtc = settings.webrtc;
  const detectedCandidate = webrtc?.detected_ip ? withWebrtcPort(webrtc.detected_ip) : null;
  // One-click prefill: add the server's detected candidate to the edit list
  // (idempotent — never duplicates an entry already present).
  const useDetectedIp = () => {
    if (!detectedCandidate) return;
    setCandidates((prev) => (prev.includes(detectedCandidate) ? prev : [...prev, detectedCandidate]));
  };

  // ready:false with the model verified and no CPU device points at the
  // accelerator gate rather than a model-download problem. Backend-aware: on
  // Coral this is a missing/unreachable Edge TPU, NOT a CUDA problem.
  const accelTripped =
    detector !== null &&
    !detector.ready &&
    detector.model_sha_ok &&
    detector.device !== 'cpu';
  const isCoral = detector?.kind === 'coral';
  // Only a DEFINITIVE checksum failure (model_sha_ok === false) is "broken".
  // null means the model is (re)loading — show loading, not a broken banner.
  const modelBroken = detector !== null && !detector.ready && detector.model_sha_ok === false;

  return (
    <div className="settings-section">
      <section className="card">
        <h2>Health</h2>
        {healthError ? (
          <p className="form-error">Backend unreachable.</p>
        ) : !health ? (
          <div className="page-loading">Checking…</div>
        ) : (
          <div className="row-inline wrap">
            {pill(health.status === 'ok', 'Backend')}
            <span className={`pill ${health.detector.ready ? 'pill-ok' : 'pill-down'}`}>
              Detector: {health.detector.ready ? `ready (${health.detector.device})` : 'not ready'}
            </span>
            {pill(health.go2rtc, 'go2rtc')}
            <span className="pill">
              {health.cameras_online} camera{health.cameras_online === 1 ? '' : 's'} online
            </span>
            <span className="pill">v{health.version}</span>
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Detector</h2>
          <button
            type="button"
            className="btn btn-sm"
            disabled={detectorBusy}
            onClick={() => void refreshDetector()}
          >
            {detectorBusy ? 'Checking…' : 'Refresh'}
          </button>
        </div>
        {accelTripped && (
          <div className="banner banner-error">
            <span>
              {isCoral ? (
                <>
                  Coral Edge TPU not available to the detector — check that the device is
                  fitted, that <code>/dev/apex_0</code> exists on the host, and that{' '}
                  <code>CORAL_DEVICE</code> is set so it is passed into the container.
                </>
              ) : (
                <>
                  GPU not available to the detector — see <code>docs/setup-nvidia.md</code>
                </>
              )}
            </span>
          </div>
        )}
        {modelBroken && (
          <div className="banner banner-error">
            <span>Model file missing or failed checksum verification — check the backend logs.</span>
          </div>
        )}
        {detectorError ? (
          <p className="form-error">Detector status unavailable.</p>
        ) : !detector ? (
          <div className="page-loading">Checking…</div>
        ) : (
          <>
            <p className="detector-active">
              Active detector: <strong>{describeDetector(detector.device, detector.kind)}</strong>
            </p>
            <div className="row-inline wrap">
              <span className={`pill ${detector.ready ? 'pill-ok' : 'pill-down'}`}>
                {detector.ready ? 'ready' : 'not ready'}
              </span>
              <span className="pill">device: {detector.device ?? 'none'}</span>
              <span className="pill">model: {detector.model}</span>
              <span className={`pill ${detector.model_sha_ok === false ? 'pill-down' : detector.model_sha_ok ? 'pill-ok' : ''}`}>
                checksum: {detector.model_sha_ok === false ? 'failed' : detector.model_sha_ok ? 'verified' : 'checking'}
              </span>
              {detector.last_inference_ms !== null && (
                <span className="pill">inference: {detector.last_inference_ms.toFixed(1)} ms</span>
              )}
            </div>
            {detector.transcode && (
              <>
                {/* A SEPARATE line from the detector pills on purpose: these are
                    two different GPU questions with two different answers on an
                    AMD/Intel box, and merging them is how "my GPU works" and
                    "my GPU does nothing for detection" both end up sounding
                    true at once. */}
                <p className="detector-active" style={{ marginTop: '0.9rem' }}>
                  Video encoding:{' '}
                  <strong>{detector.transcode.encoder_label}</strong>
                </p>
                <div className="row-inline wrap">
                  <span
                    className={`pill ${detector.transcode.hardware ? 'pill-ok' : ''}`}
                  >
                    {detector.transcode.hardware ? 'GPU' : 'CPU'}
                  </span>
                  {detector.transcode.encoder && (
                    <span className="pill">{detector.transcode.encoder}</span>
                  )}
                  {detector.transcode.vaapi_device && (
                    <span className="pill">{detector.transcode.vaapi_device}</span>
                  )}
                  {Object.entries(detector.transcode.runs).map(([enc, r]) => (
                    <span key={enc} className="pill">
                      {enc}: {r.ok} ok{r.failed ? `, ${r.failed} failed` : ''}
                    </span>
                  ))}
                  {detector.transcode.failed.map((f) => (
                    <span key={f} className="pill pill-down">
                      {f} failed at runtime
                    </span>
                  ))}
                </div>
                {!detector.transcode.hardware &&
                  detector.transcode.enabled &&
                  !detector.transcode.vaapi_device &&
                  !detector.transcode.nvidia && (
                    <p className="muted small">
                      No GPU render node is visible inside the container, so HEVC→H.264
                      runs on the CPU. If this box has an AMD or Intel iGPU, add{' '}
                      <code>VAAPI_DEVICE=/dev/dri/renderD128</code> to the server&rsquo;s{' '}
                      <code>.env</code> (check <code>ls /dev/dri</code> first) and run{' '}
                      <code>docker compose up -d backend</code>. This only affects video
                      encoding — object detection needs CUDA and never uses an iGPU.
                    </p>
                  )}
                {Object.keys(detector.transcode.runs).length === 0 && (
                  <p className="muted small">
                    Nothing has been transcoded yet, so this is the encoder that{' '}
                    <em>would</em> be used. Scrub an HEVC camera&rsquo;s timeline, then
                    press Refresh to see it actually run.
                  </p>
                )}
              </>
            )}
            {detector.per_camera.length > 0 && (
              <table className="detector-table">
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>Ingest</th>
                    <th>FPS</th>
                    <th>Last frame</th>
                  </tr>
                </thead>
                <tbody>
                  {detector.per_camera.map((c) => (
                    <tr key={c.name}>
                      <td>{c.name}</td>
                      <td>
                        <span className={`status-dot ${c.ingest_ok ? 'ok' : 'down'}`} />{' '}
                        {c.ingest_ok ? 'ok' : 'stalled'}
                      </td>
                      <td>{c.fps.toFixed(1)}</td>
                      <td>
                        {c.last_frame_age_s === null ? '—' : `${c.last_frame_age_s.toFixed(1)} s ago`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </section>

      <section className="card">
        <h2>WebRTC addresses</h2>
        <p className="muted small">
          Extra addresses the browser can reach this server's live-view port on — add the
          server's LAN IP and (if used) its Tailscale IP, each as <code>ip:8555</code>. Leave
          empty to use the defaults; live view falls back to MSE when none are reachable.
        </p>
        {webrtc && !webrtc.ready && (
          <div className="banner banner-warn">
            <span>
              Live view is on the slow fallback (MSE) — WebRTC has no reachable address, so
              first frame takes several seconds.{' '}
              {detectedCandidate
                ? 'Add this server’s detected address below.'
                : 'Add this server’s LAN IP below (e.g. 192.168.1.10:8555).'}
            </span>
            {detectedCandidate && (
              <button type="button" className="btn btn-sm" onClick={useDetectedIp}>
                Use {detectedCandidate}
              </button>
            )}
          </div>
        )}
        <ChipsInput
          value={candidates}
          onChange={setCandidates}
          placeholder="192.168.1.10:8555"
          raw
          validate={isCandidateEntry}
          invalidHint="Enter an IP or hostname with an optional port, e.g. 192.168.1.10:8555"
        />
        {webrtc && webrtc.candidates.length > 0 && (
          <p className="muted small">
            Effective candidates go2rtc is using: <code>{webrtc.candidates.join(', ')}</code>
            {webrtc.source && detectedCandidate && (
              <> (detected via {SOURCE_LABEL[webrtc.source]})</>
            )}
          </p>
        )}
      </section>

      <section className="card">
        <h2>Public URL</h2>
        <p className="muted small">
          The externally reachable base URL used in notification click-links, e.g.
          <code> https://nvr.tailnet-name.ts.net</code>. Leave empty for LAN-only use.
        </p>
        <div className="form-stack">
          <label>
            Public base URL
            <input
              type="url"
              placeholder="https://nvr.example.com"
              value={publicUrl}
              onChange={(e) => setPublicUrl(e.target.value)}
            />
          </label>
        </div>
      </section>

      {/* No Save button here — the shell owns the single Save for every
          settings tab. Server actions below (Restart) are NOT settings: they
          fire immediately and are deliberately not batched behind Save. */}

      <section className="card">
        <h2>Server</h2>

        <p className="muted small">
          Restarting reloads the backend: recording, detection and live view stop
          for roughly 15 seconds and then resume on their own. Nothing is
          deleted, and your settings and recordings are untouched.
        </p>

        <div className="row-inline wrap">
          <button
            type="button"
            className="btn btn-danger"
            disabled={restarting}
            onClick={() => setConfirmRestart(true)}
          >
            {restarting ? 'Restarting…' : 'Restart server'}
          </button>
          <span className="control-hint">
            Takes effect immediately — it is not part of Save.
          </span>
        </div>

        <div className="form-stack" style={{ marginTop: '0.9rem' }}>
          <label className="row-label">
            <input
              type="checkbox"
              checked={autoRestart.enabled}
              onChange={(e) => setAutoRestart({ ...autoRestart, enabled: e.target.checked })}
            />
            Restart automatically every day
          </label>
          {autoRestart.enabled && (
            <label>
              At
              <input
                type="time"
                value={autoRestart.time}
                onChange={(e) =>
                  setAutoRestart({ ...autoRestart, time: e.target.value || '04:00' })
                }
              />
              <span className="control-hint">
                Local time on the NVR. Pick a quiet hour — there is a short gap in
                recording while it comes back. Unlike the button above, this one
                IS a setting: it applies when you press Save.
              </span>
            </label>
          )}
        </div>
      </section>

      {/* Hands this tab's already-polled health to the card rather than letting
          it fire a second /health of its own. */}
      <ReportBugCard health={health} />

      <SupportCard />

      <section className="card">
        <h2>Danger zone</h2>
        <p className="muted small">
          These permanently delete data for every camera and cannot be undone.
          Deleting events keeps continuous recordings; deleting recordings keeps
          events and their clips.
        </p>
        <div className="row-inline wrap">
          <button
            type="button"
            className="btn btn-danger"
            disabled={purging}
            onClick={() => setConfirming('events')}
          >
            Delete all events
          </button>
          <button
            type="button"
            className="btn btn-danger"
            disabled={purging}
            onClick={() => setConfirming('recordings')}
          >
            Delete all recordings
          </button>
        </div>
      </section>

      {confirming && (
        <ConfirmDialog
          title={confirming === 'events' ? 'Delete all events' : 'Delete all recordings'}
          message={
            confirming === 'events'
              ? 'Permanently delete EVERY event, including all of its snapshots and clips? This cannot be undone.'
              : 'Permanently delete ALL continuous recorded footage for every camera? Recording resumes immediately. This cannot be undone.'
          }
          confirmLabel={confirming === 'events' ? 'Delete all events' : 'Delete all recordings'}
          danger
          busy={purging}
          onConfirm={() => void runPurge()}
          onCancel={() => (purging ? undefined : setConfirming(null))}
        />
      )}

      {confirmRestart && (
        <ConfirmDialog
          title="Restart server"
          message={
            'Restart the Vigilume backend? Recording, detection and live view stop ' +
            'for about 15 seconds and then resume automatically. Nothing is deleted.'
          }
          confirmLabel="Restart"
          danger
          busy={restarting}
          onConfirm={() => void runRestart()}
          onCancel={() => (restarting ? undefined : setConfirmRestart(false))}
        />
      )}
    </div>
  );
}

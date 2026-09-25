/**
 * Settings → Detection: what the object detector does, and what it ignores.
 *
 * SPLIT OUT OF THE RECORDING TAB. Everything here used to sit below the
 * retention and clip-padding cards, so the detector model, the confidence
 * slider and the "ignore parked cars" switch all lived under a heading that
 * gave no reason to look there. Recording is now storage only; this is the AI.
 *
 * The shell merges each tab's draft by TOP-LEVEL KEY, which is what makes the
 * split safe: this tab owns `detection` and `recognition`, the Recording tab
 * owns `recording`, and neither can clobber the other's slice.
 *
 * The `recognition` block here is TUNING (how many shots per track, how hard
 * to look). The people and vehicles themselves live on the Faces & plates tab,
 * which deliberately reports no draft at all — its edits are immediate API
 * calls — so the tuning cannot move there without breaking that invariant.
 */
import { useEffect, useState } from 'react';
import {
  api,
  CORAL_MODELS,
  type Camera,
  type CoralModel,
  type DetectionBackend,
  type DetectMode,
  type NightBoostMode,
} from '../../lib/api';
import { useAdoptSaved, type TabProps } from '../Settings';
import DetectionModels from './DetectionModels';

export default function DetectionTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft when there is one, so leaving this tab
  // and coming back keeps your edits (there is one Save for all tabs now).
  const [confidence, setConfidence] = useState(
    pending.detection?.confidence ?? settings.detection.confidence,
  );
  // Global default detection-gating mode for newly added cameras. Optional on
  // the backend — fall back to "always" (today's behavior) when absent.
  const [defaultMode, setDefaultMode] = useState<DetectMode>(
    pending.detection?.default_mode ?? settings.detection.default_mode ?? 'always',
  );
  // Which silicon runs inference. Absent on an older backend -> treat as gpu.
  const [backend, setBackend] = useState<DetectionBackend>(
    pending.detection?.backend ?? settings.detection.backend ?? 'auto',
  );
  // Edge TPU model. A SEPARATE field from the D-FINE tier, so switching backend
  // back and forth never leaves an invalid model/backend pair.
  const [coralModel, setCoralModel] = useState<CoralModel>(
    pending.detection?.coral_model ?? settings.detection.coral_model ?? 'ssdlite_mobiledet',
  );
  // How long a label may go unseen before its event ends. Absent on a backend
  // that predates the setting -> the 5 s that was hardcoded there.
  const [absenceTimeout, setAbsenceTimeout] = useState<number>(
    pending.detection?.absence_timeout_s ?? settings.detection.absence_timeout_s ?? 5,
  );
  // Hold motionless objects back from the event layer. Absent on a backend
  // that predates it -> true, which is the shipped default: what it replaces
  // is a parked car holding its label's event open and blocking every later
  // arrival on that camera, not merely some extra rows.
  const [ignoreStationary, setIgnoreStationary] = useState<boolean>(
    pending.detection?.ignore_stationary ?? settings.detection.ignore_stationary ?? true,
  );
  const [stationaryAfter, setStationaryAfter] = useState<number>(
    pending.detection?.stationary_after_s ?? settings.detection.stationary_after_s ?? 180,
  );
  // Loitering. 0 = off, which is the shipped default and what a backend
  // predating the setting should read as — this ADDS alerts, so it is opt-in.
  const [dwellSeconds, setDwellSeconds] = useState<number>(
    pending.detection?.dwell_alert_seconds ?? settings.detection.dwell_alert_seconds ?? 0,
  );
  const [packageAlerts, setPackageAlerts] = useState<boolean>(
    pending.detection?.package_alerts ?? settings.detection.package_alerts ?? false,
  );
  // Night contrast boost on the detector's frame only. Absent on an older
  // backend -> "off", which is also the shipped default: it changes what the
  // model sees, so it is opt-in.
  const [nightBoost, setNightBoost] = useState<NightBoostMode>(
    pending.detection?.night_boost ?? settings.detection.night_boost ?? 'off',
  );
  const [nightBoostThreshold, setNightBoostThreshold] = useState<number>(
    pending.detection?.night_boost_threshold ?? settings.detection.night_boost_threshold ?? 60,
  );
  // Box smoothing over the tracker's output. Also opt-in — it trades box lag
  // for steadiness (see the copy on the control).
  const [smoothing, setSmoothing] = useState<boolean>(
    pending.detection?.smoothing ?? settings.detection.smoothing ?? false,
  );
  const [smoothingFrames, setSmoothingFrames] = useState<number>(
    pending.detection?.smoothing_frames ?? settings.detection.smoothing_frames ?? 3,
  );
  // Face / plate recognition. A backend predating the feature omits the block
  // entirely, and `hasRecognition` is what keeps that case honest: the card is
  // hidden AND the slice is left out of the draft. Reporting a default-filled
  // `recognition` against a saved `undefined` would read as an edit forever and
  // leave the Save bar permanently lit on a box that cannot even do this.
  const savedRecognition = settings.recognition;
  const hasRecognition = savedRecognition !== undefined;
  const [recognition, setRecognition] = useState({
    enabled: false,
    candidate_retention_days: 14,
    notify_grace_seconds: 4,
    notify_mode: 'all' as 'all' | 'unknown_only',
    shots_per_track: 5,
    shot_min_gap_seconds: 0.4,
    pass_interval_seconds: 0.6,
    face_on_vehicles: false,
    identify_quality: 0.45,
    ...(savedRecognition ?? {}),
    ...(pending.recognition ?? {}),
  });

  // No `modelKey` mirror here any more. It existed solely so this form's PUT
  // could re-send the current model instead of clobbering a fresh activation
  // from <DetectionModels>. The patch names only `confidence` and
  // `default_mode`, so it cannot touch `detection.model` at all.

  // Adopt a freshly SAVED document. Skips the initial mount so the pending
  // draft seeded above is not clobbered when you return to this tab.
  useAdoptSaved(settings.detection.confidence, setConfidence);
  useAdoptSaved(settings.detection.default_mode ?? 'always', setDefaultMode);
  useAdoptSaved(settings.detection.backend ?? 'auto', setBackend);
  useAdoptSaved(settings.detection.coral_model ?? 'ssdlite_mobiledet', setCoralModel);
  useAdoptSaved(settings.detection.absence_timeout_s ?? 5, setAbsenceTimeout);
  useAdoptSaved(settings.detection.ignore_stationary ?? true, setIgnoreStationary);
  useAdoptSaved(settings.detection.stationary_after_s ?? 180, setStationaryAfter);
  useAdoptSaved(settings.detection.dwell_alert_seconds ?? 0, setDwellSeconds);
  useAdoptSaved(settings.detection.package_alerts ?? false, setPackageAlerts);
  useAdoptSaved(settings.detection.night_boost ?? 'off', setNightBoost);
  useAdoptSaved(settings.detection.night_boost_threshold ?? 60, setNightBoostThreshold);
  useAdoptSaved(settings.detection.smoothing ?? false, setSmoothing);
  useAdoptSaved(settings.detection.smoothing_frames ?? 3, setSmoothingFrames);
  // Guarded: on a backend without the block the saved value is `undefined`, and
  // adopting that would blank the draft this form is bound to.
  useAdoptSaved(savedRecognition, (v) => {
    if (v) setRecognition(v);
  });

  // Per-camera stationary overrides. NOT part of the draft: these are
  // immediate calls against /api/cameras/{name}/stationary, and reporting them
  // through onDraftChange would light the shell's Save bar for edits that were
  // already saved.
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [camBusy, setCamBusy] = useState<Set<string>>(new Set());

  useEffect(() => {
    let live = true;
    void api
      .cameras()
      .then((list) => {
        if (live) setCameras(list);
      })
      // The global control above is the one that matters; a camera list that
      // failed to load must not take this whole tab down with it.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  /**
   * Pin or un-pin one camera. `next` of null means "go back to following the
   * global setting" — a real third state, not a missing value.
   *
   * OPTIMISTIC: the control moves at once and rolls back if the save fails.
   * The round trip is short (no device probe, no recorder restart) but it is
   * still a round trip, and a control that waits for the network before it
   * moves reads as broken.
   */
  /** Pin or un-pin one camera's loitering threshold. `null` inherits, `0` is
   *  "no loitering alerts here" — two different instructions. */
  const setCameraDwell = async (cam: Camera, next: number | null) => {
    const previous = cam.dwell_seconds ?? null;
    setCameras((prev) =>
      prev.map((c) => (c.name === cam.name ? { ...c, dwell_seconds: next } : c)),
    );
    setCamBusy((prev) => new Set(prev).add(cam.name));
    try {
      const saved = await api.setCameraDwell(cam.name, next);
      setCameras((prev) =>
        prev.map((c) =>
          c.name === cam.name ? { ...c, dwell_seconds: saved.dwell_seconds } : c,
        ),
      );
    } catch {
      setCameras((prev) =>
        prev.map((c) => (c.name === cam.name ? { ...c, dwell_seconds: previous } : c)),
      );
    } finally {
      setCamBusy((prev) => {
        const nextSet = new Set(prev);
        nextSet.delete(cam.name);
        return nextSet;
      });
    }
  };

  const setCameraStationary = async (cam: Camera, next: boolean | null) => {
    const previous = cam.ignore_stationary ?? null;
    setCameras((prev) =>
      prev.map((c) => (c.name === cam.name ? { ...c, ignore_stationary: next } : c)),
    );
    setCamBusy((prev) => new Set(prev).add(cam.name));
    try {
      const saved = await api.setCameraStationary(cam.name, next);
      setCameras((prev) =>
        prev.map((c) =>
          c.name === cam.name ? { ...c, ignore_stationary: saved.ignore_stationary } : c,
        ),
      );
    } catch {
      setCameras((prev) =>
        prev.map((c) =>
          c.name === cam.name ? { ...c, ignore_stationary: previous } : c,
        ),
      );
    } finally {
      setCamBusy((prev) => {
        const nextSet = new Set(prev);
        nextSet.delete(cam.name);
        return nextSet;
      });
    }
  };

  // Report this tab's slice up on every edit; the shell's single Save button
  // persists it together with every other tab's pending changes.
  useEffect(() => {
    onDraftChange({
      detection: {
        confidence, default_mode: defaultMode, backend, coral_model: coralModel,
        absence_timeout_s: absenceTimeout,
        ignore_stationary: ignoreStationary, stationary_after_s: stationaryAfter,
        dwell_alert_seconds: dwellSeconds, package_alerts: packageAlerts,
        night_boost: nightBoost, night_boost_threshold: nightBoostThreshold,
        smoothing, smoothing_frames: smoothingFrames,
      },
      ...(hasRecognition ? { recognition } : {}),
    });
  }, [
    confidence, defaultMode, backend, coralModel, absenceTimeout,
    ignoreStationary, stationaryAfter, dwellSeconds, packageAlerts,
    nightBoost, nightBoostThreshold, smoothing, smoothingFrames,
    hasRecognition, recognition, onDraftChange,
  ]);

  return (
    <div className="settings-section">
      <section className="card">
        <h2>Detection model</h2>
        <p className="muted small">
          Pick a tier to match your hardware. Models download inside the app with progress and
          activate live — the app keeps serving while a model loads in the background, so a
          fresh install starts fast. Switching tiers causes a brief detection gap while the new
          model loads.
        </p>
        {backend !== 'coral' ? (
          <DetectionModels />
        ) : (
          // Same tier CARDS as the GPU list (.model-tier), not a radio column:
          // the two backends are alternatives for one job, so presenting them
          // differently made the Edge TPU read like a lesser, secondary control.
          <div className="model-manager">
            <div className="model-tiers">
              {CORAL_MODELS.map((m) => {
                const enabled = coralModel === m.key;
                return (
                  <div key={m.key} className={`card model-tier ${enabled ? 'active' : ''}`}>
                    <div className="model-tier-head">
                      <span className="model-tier-name">{m.label}</span>
                      {enabled && <span className="pill pill-ok">Enabled</span>}
                    </div>
                    <p className="model-tier-blurb small">{m.blurb}</p>
                    <div className="model-tier-meta">
                      {/* Edge TPU SSD models emit the sparse COCO-90 id space,
                          which the backend remaps to the same COCO-80 vocabulary
                          the GPU models use. */}
                      <span className="pill pill-vocab">COCO · 80 classes</span>
                      <span className="pill">{m.map.toFixed(1)} mAP</span>
                      <span className="pill">~{m.latencyMs} ms</span>
                      <span className="pill">{m.inputSize}px</span>
                    </div>
                    {m.slow && (
                      <div className="model-tier-rec muted small">
                        Sustains under 10 inferences/sec — about what two cameras at 5 fps
                        already demand.
                      </div>
                    )}
                    <div className="model-tier-action">
                      <button
                        type="button"
                        className={`btn btn-sm btn-block${enabled ? '' : ' btn-primary'}`}
                        disabled={enabled}
                        onClick={() => setCoralModel(m.key)}
                      >
                        {enabled ? 'Enabled' : 'Use this model'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="muted small">
              Downloaded and checksum-verified on first use. Switching models reloads
              the detector — detection pauses for a few seconds.
            </p>
            {CORAL_MODELS.find((m) => m.key === coralModel)?.slow && (
              <div className="banner banner-warn">
                <span>
                  At ~{CORAL_MODELS.find((m) => m.key === coralModel)?.latencyMs} ms the
                  model you have selected sustains under 10 inferences/sec — roughly what
                  two cameras at 5 fps already demand. Frames will be dropped under load.
                </span>
              </div>
            )}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Detection hardware</h2>
        <p className="muted small">
          Which silicon runs object detection. Takes effect on the next{' '}
          <strong>backend restart</strong> (Settings → System → Restart server).
        </p>
        {/* Segmented control, matching the spotlight / night-vision pickers
            elsewhere — two mutually exclusive choices read better side by side
            than as a stacked radio list. */}
        <div className="seg seg-full" role="group" aria-label="Detection hardware">
          {([
            { key: 'auto', label: 'Automatic' },
            { key: 'gpu', label: 'GPU' },
            { key: 'coral', label: 'Coral Edge TPU' },
          ] as const).map(({ key, label }) => (
            <button
              key={key}
              type="button"
              className={`seg-btn${backend === key ? ' seg-on' : ''}`}
              aria-pressed={backend === key}
              onClick={() => setBackend(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="muted small">
          {backend === 'auto'
            ? 'Uses a Coral Edge TPU when one is fitted, otherwise the GPU. '
              + 'Fit or remove a Coral and it is picked up on the next restart.'
            : backend === 'gpu'
              ? 'D-FINE on CUDA — highest accuracy.'
              : 'SSDLite MobileDet on the Edge TPU — about 2 W instead of the GPU.'}
        </p>
        {backend === 'coral' && (
          <div className="banner banner-warn">
            <span>
              <strong>Requires a Coral Edge TPU fitted to this machine.</strong> If it is
              missing or the driver is not loaded, detection will not start at all —
              check Settings → System for the detector status after restarting.
              Accuracy also drops (COCO mAP ~54 → ~33), and the loss falls hardest on
              small, distant and night-time people.
            </span>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Default detection mode</h2>
        <p className="muted small">
          How the GPU detector is scheduled for a newly added camera. Cameras with their own
          on-camera AI (SMD human/vehicle, IVS tripwire/intrusion) can gate detection on that
          signal to cut GPU load. Change it per camera under Settings → Cameras → Edit.
        </p>
        <div className="form-stack">
          {/* Primary control mirrors the per-camera one: ON = camera_ai, OFF =
              always. The advanced segmented control keeps camera_ai_only
              reachable. All three write the same default_mode. */}
          <div className="switch-row">
            <button
              type="button"
              role="switch"
              aria-checked={defaultMode !== 'always'}
              aria-label="Default to camera AI detection"
              className={`switch ${defaultMode !== 'always' ? 'switch-on' : ''}`}
              onClick={() =>
                setDefaultMode(defaultMode !== 'always' ? 'always' : 'camera_ai')
              }
            >
              <span className="switch-knob" />
            </button>
            <span className="switch-label">
              {defaultMode !== 'always'
                ? 'New cameras gate the GPU on their on-board AI'
                : 'New cameras run continuous server detection'}
            </span>
          </div>
          <details className="advanced-section" open={defaultMode === 'camera_ai_only'}>
            <summary>Advanced — where detection runs</summary>
            <div className="seg" role="group" aria-label="Default detection mode">
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
                  className={`seg-btn ${defaultMode === value ? 'seg-on' : ''}`}
                  aria-pressed={defaultMode === value}
                  onClick={() => setDefaultMode(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <span className="control-hint">
              <strong>Camera-triggered</strong> runs detection only when a camera&rsquo;s own AI
              sees motion — big GPU savings; may miss what the camera AI misses. Only applies to
              cameras that report on-camera AI; others always run detection.
            </span>
          </details>
        </div>
      </section>

      <section className="card">
        <h2>Confidence</h2>
        <div className="form-stack">
          <label>
            Confidence threshold: {Math.round(confidence * 100)}%
            <input
              type="range"
              min={0.2}
              max={0.9}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
            />
            <span className="control-hint">
              Lower catches more (and more false positives); higher only keeps sure detections.
            </span>
          </label>
        </div>
      </section>

      {hasRecognition && (
        <section className="card">
          <h2>Faces &amp; plates</h2>
          <p className="muted small">
            Reads faces and vehicle plates on top of ordinary detection, and names the
            person or vehicle in the alert when it recognizes one. Off by default: it
            downloads two further models (~41&nbsp;MB) and keeps cropped face images on the
            server so you can enroll people from real sightings afterwards. People and
            vehicles are enrolled in the Vigilume iOS app, under Settings →&nbsp;Faces
            &amp;&nbsp;Plates.
          </p>
          <div className="form-stack">
            <label className="row-label">
              <input
                type="checkbox"
                checked={recognition.enabled}
                onChange={(e) => setRecognition({ ...recognition, enabled: e.target.checked })}
              />
              Recognize faces and license plates
            </label>
            {recognition.enabled && (
              <>
                <label>
                  Alert me about
                  <select
                    value={recognition.notify_mode}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        notify_mode: e.target.value === 'unknown_only' ? 'unknown_only' : 'all',
                      })
                    }
                  >
                    <option value="all">Everyone, named where recognized</option>
                    <option value="unknown_only">Only people and vehicles I have not enrolled</option>
                  </select>
                  <span className="control-hint">
                    {recognition.notify_mode === 'unknown_only'
                      ? 'Enrolled people and vehicles arrive silently. Anyone else still alerts — including a face nobody could identify, which is the case most worth hearing about.'
                      : 'Every alert still sends; a recognized person or vehicle is named in it.'}
                  </span>
                </label>
                <label>
                  Hold alerts for (seconds)
                  <input
                    type="number"
                    min={0}
                    max={30}
                    step={1}
                    value={recognition.notify_grace_seconds}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        notify_grace_seconds: Math.min(
                          30,
                          Math.max(0, Math.floor(Number(e.target.value) || 0)),
                        ),
                      })
                    }
                  />
                  <span className="control-hint">
                    Recognition finishes a few frames after an event opens, so an alert sent
                    the instant it opens can never carry a name. A held alert is only
                    <em> delayed</em>, never dropped — it sends either way once the hold is
                    up. 0 disables the wait.
                  </span>
                </label>
                <label>
                  Keep unmatched face images for (days)
                  <input
                    type="number"
                    min={0}
                    max={365}
                    step={1}
                    value={recognition.candidate_retention_days}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        candidate_retention_days: Math.min(
                          365,
                          Math.max(0, Math.floor(Number(e.target.value) || 0)),
                        ),
                      })
                    }
                  />
                  <span className="control-hint">
                    How long a face that matched nobody stays on disk so you can enroll it
                    later. This is biometric imagery of people who have not been identified
                    — keep the window no longer than you need. 0 keeps nothing, which also
                    means there is nothing to enroll from. Faces already enrolled against a
                    person are kept until you delete that person.
                  </span>
                </label>
                <label>
                  Shots collected per face or vehicle
                  <input
                    type="number"
                    min={1}
                    max={12}
                    step={1}
                    value={recognition.shots_per_track}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        shots_per_track: Math.min(12, Math.max(1, Math.floor(Number(e.target.value) || 5))),
                      })
                    }
                  />
                  <span className="control-hint">
                    The best of these is what gets read. More shots is the real defence
                    against a <em>wrong</em> name: a false match comes from scoring a
                    marginal crop, and the cure is having a better one available — not a
                    stricter threshold, which only trades wrong names for missed ones.
                    Past about 10 the extra shots stop being distinct moments.
                  </span>
                </label>
                <label>
                  Minimum gap between shots (seconds)
                  <input
                    type="number"
                    min={0.05}
                    max={5}
                    step={0.05}
                    value={recognition.shot_min_gap_seconds}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        shot_min_gap_seconds: Math.min(5, Math.max(0.05, Number(e.target.value) || 0.4)),
                      })
                    }
                  />
                  <span className="control-hint">
                    This matters as much as the count. Without a gap the buffer fills with
                    neighbouring frames of one stride — five samples of one pose, which is
                    worth barely more than one.
                  </span>
                </label>
                <label>
                  Look again every (seconds)
                  <input
                    type="number"
                    min={0.1}
                    max={5}
                    step={0.1}
                    value={recognition.pass_interval_seconds}
                    onChange={(e) =>
                      setRecognition({
                        ...recognition,
                        pass_interval_seconds: Math.min(5, Math.max(0.1, Number(e.target.value) || 0.6)),
                      })
                    }
                  />
                  <span className="control-hint">
                    How often each tracked person or vehicle is checked for a face. 0.2
                    looks every frame of a 5&nbsp;fps detect stream instead of every third,
                    which is how a brief side-on glance still yields one usable shot. Costs
                    CPU in proportion, and only on the cameras you ticked.
                  </span>
                </label>
                <label className="row-label">
                  <input
                    type="checkbox"
                    checked={recognition.face_on_vehicles}
                    onChange={(e) =>
                      setRecognition({ ...recognition, face_on_vehicles: e.target.checked })
                    }
                  />
                  Also look for faces on vehicles
                </label>
                <span className="control-hint">
                  The driver through the windscreen. Off by default: on a road-facing
                  camera most windscreens are glare, and a plate identifies a car better
                  than a face does. It earns its keep on a driveway or at a gate.
                </span>
                <label>
                  Crop quality needed to identify: {Math.round(recognition.identify_quality * 100)}%
                  <input
                    type="range"
                    min={0.15}
                    max={0.9}
                    step={0.05}
                    value={recognition.identify_quality}
                    onChange={(e) =>
                      setRecognition({ ...recognition, identify_quality: Number(e.target.value) })
                    }
                  />
                  <span className="control-hint">
                    <strong>Lowering this is not &ldquo;more accurate&rdquo;.</strong> A
                    marginal crop produces a marginal reading, which is precisely where a
                    wrong name comes from. Lower it only if people are being missed
                    entirely, and prefer more shots and a shorter interval first.
                  </span>
                </label>
              </>
            )}
          </div>
        </section>
      )}

      <section className="card">
        <h2>Event grouping</h2>
        <p className="muted small">
          How long an object may go unseen before its event is closed. This does not add
          footage — an event still ends at the last frame the object was actually seen —
          but it decides whether a subject that pauses, turns away, or slips behind cover
          becomes <em>one</em> event or several in a row. Raise it for scenes with
          obstructions; lower it if separate visits are being merged into one long event.
        </p>
        <div className="form-stack">
          <label>
            End the event after (seconds unseen)
            <input
              type="number"
              min={1}
              max={300}
              step={1}
              value={absenceTimeout}
              onChange={(e) =>
                setAbsenceTimeout(
                  Math.min(300, Math.max(1, Math.floor(Number(e.target.value) || 1))),
                )
              }
            />
            <span className="control-hint">
              Clips are only cut once the event ends, so this also delays when a clip
              appears.
            </span>
          </label>
        </div>
      </section>

      <section className="card">
        <h2>Things that aren't doing anything</h2>
        <p className="muted small">
          A detector has no sense of what's new — it answers "is there a car here?" on
          every frame, so a car parked in the drive is detected five times a second for
          as long as it's parked. Because there's one open event per object type per
          camera, that parked car's event never ends, and{' '}
          <strong>while it's open the car that pulls in can't open one of its own.</strong>{' '}
          So this isn't only about noise — leaving it off costs you arrivals.
        </p>
        <div className="form-stack">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={ignoreStationary}
              onChange={(e) => setIgnoreStationary(e.target.checked)}
            />
            <span>Ignore objects that aren't moving</span>
          </label>
          <p className="control-hint">
            Something that has <em>never</em> moved since it was first seen is treated as
            furniture — a parked car, a wheelie bin the detector reads as a person,
            anything already sitting there when the system started. It raises no event at
            all. It's still tracked, so the moment it moves it becomes news again.
          </p>
          <label>
            Give up on something that stopped moving after (seconds)
            <input
              type="number"
              min={10}
              max={3600}
              step={10}
              value={stationaryAfter}
              disabled={!ignoreStationary}
              onChange={(e) =>
                setStationaryAfter(
                  Math.min(3600, Math.max(10, Math.floor(Number(e.target.value) || 10))),
                )
              }
            />
            <span className="control-hint">
              Applies only to something that <em>arrived</em> and then settled. It keeps
              its event for this long — someone standing at a door is exactly the sighting
              worth keeping — and simply stops repeating itself. None of this touches
              recording: footage is continuous, so anything suppressed here is still on
              disk to scrub back to.
            </span>
          </label>
        </div>

        {cameras.length > 0 && (
          <>
            <h3>Per camera</h3>
            <p className="muted small">
              Cameras follow the setting above unless you pin them here. A drive full of
              parked cars and a back gate where every sighting matters want opposite
              answers, and <em>Follow</em> means a camera keeps tracking whatever you
              change above later.
            </p>
            <table className="camera-choice-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  <th>Stationary objects</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((c) => {
                  // ?? null, never ||: `false` is a real pinned state and must
                  // not collapse into "Follow".
                  const value = c.ignore_stationary ?? null;
                  return (
                    <tr key={c.name}>
                      <td>{c.friendly_name || c.name}</td>
                      <td>
                        <select
                          value={value === null ? 'inherit' : value ? 'on' : 'off'}
                          disabled={camBusy.has(c.name)}
                          onChange={(e) =>
                            void setCameraStationary(
                              c,
                              e.target.value === 'inherit'
                                ? null
                                : e.target.value === 'on',
                            )
                          }
                        >
                          <option value="inherit">
                            Follow the setting above ({ignoreStationary ? 'ignore' : 'report'})
                          </option>
                          <option value="on">Always ignore them here</option>
                          <option value="off">Always report them here</option>
                        </select>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="card">
        <h2>Something left behind</h2>
        <p className="muted small">
          Tells you when something that can be carried has been put down and nobody has
          taken it — a parcel on the step being the case worth catching. It waits for the
          object to sit still, checks it wasn't there before, and requires a person to
          have been around, because parcels don't arrive on their own.
        </p>
        <div className="form-stack">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={packageAlerts}
              onChange={(e) => setPackageAlerts(e.target.checked)}
            />
            <span>Tell me when something is left behind</span>
          </label>
          <p className="control-hint">
            <strong>Expect this to be approximate.</strong> The detector has no “package”
            class, so it reports bags, backpacks and suitcases — a plant pot or a folded
            chair will get called one sooner or later. The three-part test (sitting still,
            newly arrived, a person was here) is what keeps that from firing nightly, but
            it's a useful nudge rather than a parcel tracker. Those object types also have
            to be switched on for the camera, the same way faces need <em>person</em>.
          </p>
        </div>
      </section>

      <section className="card">
        <h2>Someone who doesn't leave</h2>
        <p className="muted small">
          A normal alert tells you somebody <em>arrived</em>, once. This is a second,
          different alert for the thing people actually worry about: somebody who arrived
          and is <em>still there</em>. It fires once per visit — “still there” repeated
          every minute is the noise this replaces, and the event is already on screen.
        </p>
        <div className="form-stack">
          <label>
            Say something after (seconds) — 0 turns it off
            <input
              type="number"
              min={0}
              max={3600}
              step={10}
              value={dwellSeconds}
              onChange={(e) => {
                const n = Math.max(0, Math.min(3600, Math.floor(Number(e.target.value) || 0)));
                // 0 is off; anything above that has a 10 s floor, which the
                // backend also enforces. Snapping here rather than letting the
                // save 422 keeps the rule where you can see it.
                setDwellSeconds(n === 0 ? 0 : Math.max(10, n));
              }}
            />
            <span className="control-hint">
              Off by default, deliberately: this <em>adds</em> a kind of notification, and
              a security system that starts pushing new alerts after an update is one
              people mute entirely. Timed from when the event opened, not from when
              somebody stopped moving — a subject whose tracking drops behind a pillar
              hasn't just arrived. A muted person stays muted for this too.
            </span>
          </label>
        </div>

        {cameras.length > 0 && (
          <>
            <h3>Per camera</h3>
            <p className="muted small">
              <em>Follow</em> keeps a camera tracking whatever you set above.{' '}
              <em>Never</em> is how a pavement-facing camera opts out of an alert the
              front door wants.
            </p>
            <table className="camera-choice-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  <th>Still-there alert</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((c) => {
                  // ?? null, never ||: 0 is a real pinned state ("never here")
                  // and must not collapse into "follow".
                  const value = c.dwell_seconds ?? null;
                  const mode = value === null ? 'inherit' : value === 0 ? 'off' : 'custom';
                  return (
                    <tr key={c.name}>
                      <td>{c.friendly_name || c.name}</td>
                      <td className="dwell-cell">
                        <select
                          value={mode}
                          disabled={camBusy.has(c.name)}
                          onChange={(e) => {
                            const v = e.target.value;
                            void setCameraDwell(
                              c,
                              v === 'inherit' ? null : v === 'off' ? 0 : 60,
                            );
                          }}
                        >
                          <option value="inherit">
                            Follow the setting above (
                            {dwellSeconds === 0 ? 'off' : `${dwellSeconds}s`})
                          </option>
                          <option value="off">Never on this camera</option>
                          <option value="custom">After a set time…</option>
                        </select>
                        {mode === 'custom' && (
                          <input
                            type="number"
                            min={10}
                            max={3600}
                            step={10}
                            value={value ?? 60}
                            disabled={camBusy.has(c.name)}
                            aria-label={`Seconds before the still-there alert on ${
                              c.friendly_name || c.name
                            }`}
                            onChange={(e) =>
                              void setCameraDwell(
                                c,
                                Math.max(10, Math.min(3600, Math.floor(Number(e.target.value) || 10))),
                              )
                            }
                          />
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="card">
        <h2>Detector input</h2>
        <p className="muted small">
          Two ways to change what the model sees before it looks. Both are off by default
          and both are genuine trades, not free wins — turn one on, watch a camera you care
          about for a night, and keep it only if it actually helped.
        </p>
        <div className="form-stack">
          <label>
            Night contrast boost
            <select
              value={nightBoost}
              onChange={(e) => setNightBoost(e.target.value as NightBoostMode)}
            >
              <option value="off">Off — the model sees exactly what the camera sent</option>
              <option value="auto">Auto — boost only frames darker than the threshold</option>
              <option value="always">Always — boost every frame (to compare against Off)</option>
            </select>
            <span className="control-hint">
              Lifts local contrast on the frame handed to detection, for a camera run
              without IR where the scene is dim rather than dark. It never touches
              recordings, clips, live view or the saved snapshot. It cannot create detail
              in a frame with no light, and the model was trained on ordinary images — so
              it can help or hurt depending on the camera.
            </span>
          </label>
          {nightBoost === 'auto' && (
            <label>
              Treat a frame as night below (brightness 0–255)
              <input
                type="number"
                min={0}
                max={255}
                step={5}
                value={nightBoostThreshold}
                onChange={(e) =>
                  setNightBoostThreshold(
                    Math.min(255, Math.max(0, Math.floor(Number(e.target.value) || 0))),
                  )
                }
              />
              <span className="control-hint">
                60 sits well under a lit indoor scene and above a genuinely black frame.
              </span>
            </label>
          )}
          <label className="row-label">
            <input
              type="checkbox"
              checked={smoothing}
              onChange={(e) => setSmoothing(e.target.checked)}
            />
            Smooth detection boxes across frames
          </label>
          <span className="control-hint">
            Averages each tracked object&rsquo;s box over the last few frames: steadier
            boxes on snapshots, and a flickering track can confirm sooner. The cost is real
            — an averaged box <em>lags</em> a moving subject by about half the window, and
            an object is still reported for a few frames after it leaves.
          </span>
          {smoothing && (
            <label>
              Frames averaged
              <input
                type="number"
                min={2}
                max={10}
                step={1}
                value={smoothingFrames}
                onChange={(e) =>
                  setSmoothingFrames(
                    Math.min(10, Math.max(2, Math.floor(Number(e.target.value) || 3))),
                  )
                }
              />
              <span className="control-hint">
                3 frames is a 0.6 s window at the default 5 fps. Higher is steadier and
                laggier.
              </span>
            </label>
          )}
        </div>
      </section>

      {/* No Save button here by design — the shell owns the single Save for
          every settings tab. This page reports its slice via onDraftChange. */}
    </div>
  );
}

/**
 * Settings → Recording: retention windows for Vigilume's own recorder and
 * the object-detection engine. The detector model is now chosen through the
 * tiered download manager (which activates instantly, out-of-band from this
 * form); this tab keeps the retention windows and the confidence slider.
 */
import { useEffect, useState } from 'react';
import {
  CORAL_MODELS,
  type CoralModel,
  type DetectionBackend,
  type DetectMode,
  type NightBoostMode,
} from '../../lib/api';
import { useAdoptSaved, type TabProps } from '../Settings';
import DetectionModels from './DetectionModels';

export default function RecordingTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft when there is one, so leaving this tab
  // and coming back keeps your edits (there is one Save for all tabs now).
  const [recording, setRecording] = useState({
    ...settings.recording,
    ...(pending.recording ?? {}),
  });
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
  // No `modelKey` mirror here any more. It existed solely so this form's PUT
  // could re-send the current model instead of clobbering a fresh activation
  // from <DetectionModels>. The patch names only `confidence` and
  // `default_mode`, so it cannot touch `detection.model` at all.

  // Adopt a freshly SAVED document. Skips the initial mount so the pending
  // draft seeded above is not clobbered when you return to this tab.
  useAdoptSaved(settings.recording, setRecording);
  useAdoptSaved(settings.detection.confidence, setConfidence);
  useAdoptSaved(settings.detection.default_mode ?? 'always', setDefaultMode);
  useAdoptSaved(settings.detection.backend ?? 'auto', setBackend);
  useAdoptSaved(settings.detection.coral_model ?? 'ssdlite_mobiledet', setCoralModel);
  useAdoptSaved(settings.detection.absence_timeout_s ?? 5, setAbsenceTimeout);
  useAdoptSaved(settings.detection.night_boost ?? 'off', setNightBoost);
  useAdoptSaved(settings.detection.night_boost_threshold ?? 60, setNightBoostThreshold);
  useAdoptSaved(settings.detection.smoothing ?? false, setSmoothing);
  useAdoptSaved(settings.detection.smoothing_frames ?? 3, setSmoothingFrames);

  // Report this tab's slice up on every edit; the shell's single Save button
  // persists it together with every other tab's pending changes.
  useEffect(() => {
    onDraftChange({
      recording,
      detection: {
        confidence, default_mode: defaultMode, backend, coral_model: coralModel,
        absence_timeout_s: absenceTimeout,
        night_boost: nightBoost, night_boost_threshold: nightBoostThreshold,
        smoothing, smoothing_frames: smoothingFrames,
      },
    });
  }, [
    recording, confidence, defaultMode, backend, coralModel, absenceTimeout,
    nightBoost, nightBoostThreshold, smoothing, smoothingFrames, onDraftChange,
  ]);

  const dayInput = (
    label: string,
    key: keyof typeof recording,
    hint: string,
  ) => (
    <label>
      {label}
      <input
        type="number"
        min={0}
        max={365}
        value={recording[key]}
        onChange={(e) =>
          setRecording({ ...recording, [key]: Math.max(0, Math.floor(Number(e.target.value) || 0)) })
        }
      />
      <span className="control-hint">{hint}</span>
    </label>
  );

  // Separate from dayInput: these carry their own bounds rather than 0-365.
  // Gigabytes need a far larger ceiling; min_free_gb has a floor of 1 (a floor
  // of 0 would mean "fill the disk"); clip post-roll has a real upper bound
  // (MAX_CLIP_POST_S) because footage past it is not yet written when the clip
  // is cut. `max` is optional so only the fields with a ceiling declare one.
  const numInput = (
    label: string,
    key: 'max_storage_gb' | 'min_free_gb' | 'clip_pre_s' | 'clip_post_s' | 'clip_delay_s',
    min: number,
    hint: string,
    max?: number,
  ) => (
    <label>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={1}
        value={recording[key] ?? min}
        onChange={(e) => {
          const n = Math.max(min, Math.floor(Number(e.target.value) || 0));
          const next = { ...recording, [key]: max === undefined ? n : Math.min(max, n) };
          // Lowering the cut delay lowers what post-roll can reach, so pull the
          // run-on down with it. Without this the pair goes out of range and the
          // save 422s on a field the operator did not touch — the error would
          // name clip_post_s while the mistake was made in clip_delay_s.
          if (key === 'clip_delay_s') {
            next.clip_post_s = Math.min(next.clip_post_s ?? 0, Math.max(0, n - 10));
          }
          setRecording(next);
        }}
      />
      <span className="control-hint">{hint}</span>
    </label>
  );

  // Reachable post-roll, mirroring the backend's max_clip_post_s: a segment is
  // SEGMENT_SECONDS long and is only on disk once closed, so footage past
  // (delay - 10) has not been written when the clip is cut. Derived rather than
  // hardcoded so raising the delay visibly raises the post-roll ceiling; the
  // backend rejects the pair anyway, but a field that silently refuses to go
  // past 10 with no explanation is a worse way to learn that.
  const clipDelay = recording.clip_delay_s ?? 20;
  const maxPostRoll = Math.max(0, clipDelay - 10);

  return (
    <div className="settings-section">
      <section className="card">
        <h2>Retention</h2>
        <p className="muted small">
          How long recordings stay on disk before the hourly cleanup removes them. Rule of
          thumb: continuous recording uses ≈ 10.8 GB per day for every 1 Mbps of combined
          camera bitrate (a typical 3-camera setup ≈ 135 GB/day, so 7 days ≈ 1 TB).
        </p>
        <div className="form-stack">
          {dayInput('Continuous recording (days)', 'continuous_days', '24/7 footage kept on disk')}
          {dayInput('Event clips (days)', 'event_days', 'per-event recordings')}
          {dayInput('Snapshots (days)', 'snapshot_days', 'event snapshot images')}
        </div>
      </section>

      <section className="card">
        <h2>Storage limits</h2>
        <p className="muted small">
          When space runs out, the oldest 24/7 footage is deleted to make room for the
          newest — a rolling window, checked every minute. This applies <em>on top of</em>{' '}
          the day limits above: whichever frees a recording first wins, so footage may be
          removed sooner than the retention days suggest. <strong>Event clips are never
          deleted for space</strong> — they expire only by their own retention.
        </p>
        <div className="form-stack">
          {numInput(
            'Maximum recording storage (GB)',
            'max_storage_gb',
            0,
            '0 = no cap. Set this when the disk is shared with other data, so recordings ' +
              'cannot consume the whole array.',
          )}
          {numInput(
            'Keep free space (GB)',
            'min_free_gb',
            1,
            'Always leave at least this much free on the recordings disk, whatever the cap.',
          )}
        </div>
      </section>

      <section className="card">
        <h2>Clip padding</h2>
        <p className="muted small">
          Extra footage kept either side of an event in its clip. Both are measured from{' '}
          <em>when the object was detected</em>, which is later than when it entered frame —
          the tracker needs a few frames on something large enough to recognise, and a
          subject approaching from a distance can be visible for seconds before that. If
          clips tend to open with the subject already mid-frame, raise the lead-in. The
          footage is copied from 24/7 recording that is already on disk, so wider padding
          costs a little clip storage and no extra CPU.
        </p>
        <div className="form-stack">
          {numInput(
            'Lead-in before event (seconds)',
            'clip_pre_s',
            0,
            'Try 15 if clips start too late. 0 starts the clip exactly at detection.',
            120,
          )}
          {numInput(
            'Run-on after event (seconds)',
            'clip_post_s',
            0,
            `Limited to ${maxPostRoll} s by the cut delay below — later footage is not on ` +
              'disk yet when the clip is assembled. Raise the delay to raise this.',
            maxPostRoll,
          )}
          {numInput(
            'Cut the clip this long after the event (seconds)',
            'clip_delay_s',
            10,
            'Only the clip waits — the event, its snapshot and its notification arrive ' +
              'immediately. Raise it only to allow more run-on above.',
            300,
          )}
        </div>
      </section>

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

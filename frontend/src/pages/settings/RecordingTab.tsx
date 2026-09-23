/**
 * Settings → Recording: how much footage is kept, and for how long.
 *
 * STORAGE ONLY. Everything about what the DETECTOR does — model, hardware,
 * confidence, how events are grouped, what is ignored — lives on the Detection
 * tab next door. They used to share this page, which meant "Recording" was
 * where you went to change the AI model: eleven cards deep, under a heading
 * that gave no reason to look there. Splitting them is the point of the split.
 */
import { useEffect, useState } from 'react';
import { useAdoptSaved, type TabProps } from '../Settings';

export default function RecordingTab({ settings, onDraftChange, pending }: TabProps) {
  // Seed from the shell's pending draft when there is one, so leaving this tab
  // and coming back keeps your edits (there is one Save for all tabs now).
  const [recording, setRecording] = useState({
    ...settings.recording,
    ...(pending.recording ?? {}),
  });

  // Adopt a freshly SAVED document. Skips the initial mount so the pending
  // draft seeded above is not clobbered when you return to this tab.
  useAdoptSaved(settings.recording, setRecording);

  // Report this tab's slice up on every edit; the shell's single Save button
  // persists it together with every other tab's pending changes. The shell
  // merges by TOP-LEVEL KEY, which is what lets this tab own `recording` while
  // the Detection tab owns `detection` and `recognition` without either
  // clobbering the other.
  useEffect(() => {
    onDraftChange({ recording });
  }, [recording, onDraftChange]);

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

      {/* No Save button here by design — the shell owns the single Save for
          every settings tab. This page reports its slice via onDraftChange. */}
    </div>
  );
}

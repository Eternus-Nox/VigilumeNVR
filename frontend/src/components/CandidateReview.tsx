/**
 * Review one unmatched sighting against the WHOLE FRAME it came from, and
 * decide who it is.
 *
 * WHY THE FRAME AND NOT THE CROP
 * ==============================
 * The candidate grid shows what the recognizer sees: a 112x112 crop, aligned
 * to the canonical geometry, background gone. That is the correct input for an
 * embedding and a poor basis for a human decision. It cannot say who ELSE was
 * in the shot, what the person was doing, or — on a doorstep with three people
 * — which of them this even is. Enrolling the wrong face is the one mistake
 * here with lasting consequences (it teaches the gallery that a stranger is
 * you), so the review opens with the scene and keeps the crop alongside as
 * "this is the part that will be enrolled".
 *
 * `frame_box` rings the subject. It is drawn as a rectangle over the image in
 * PERCENTAGES, not pixels, so it stays correct at any rendered size without
 * measuring the element or waiting for a load event. When it is absent (a
 * sighting recorded before the rectangle was captured) the frame is shown
 * plain rather than with a guessed box, because a ring in the wrong place is
 * worse than no ring at all.
 *
 * ENROLLING IS NOT TRAINING
 * -------------------------
 * "Add to a person" appends this shot's embedding to that profile's gallery.
 * Nothing is fitted, nothing on disk changes, and removing the sample later
 * undoes it completely — which is why this dialog can afford a single
 * confirming click instead of a warning.
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import type { RecognitionCandidate, RecognitionProfile } from '../lib/api';
import AuthImage from './AuthImage';
import { Modal } from './Modal';
import { formatDateTime, titleCase } from '../lib/format';

interface Props {
  candidate: RecognitionCandidate;
  /** Profiles this candidate can be added to — already filtered to the kind
   *  that matches it (a face cannot join a vehicle). */
  profiles: RecognitionProfile[];
  busy?: boolean;
  onEnroll: (profileId: number) => void;
  onCreateProfile: () => void;
  onDelete: () => void;
  onClose: () => void;
}

export default function CandidateReview({
  candidate,
  profiles,
  busy,
  onEnroll,
  onCreateProfile,
  onDelete,
  onClose,
}: Props) {
  const [choice, setChoice] = useState<number | ''>('');
  const isFace = candidate.kind === 'face';
  const box = candidate.frame_box;

  return (
    <Modal title={isFace ? 'Who is this?' : 'Which vehicle is this?'} onClose={onClose} wide>
      <div className="candidate-review">
        <div className="candidate-review-frame">
          {candidate.frame_url ? (
            <div className="candidate-frame-holder">
              <AuthImage
                src={candidate.frame_url}
                eager
                alt={`Full frame from ${titleCase(candidate.camera)}`}
              />
              {box && (
                <span
                  className="candidate-frame-box"
                  aria-hidden="true"
                  style={{
                    // Percentages of the holder, so the ring tracks the image
                    // at every size without measuring anything.
                    left: `${box[0] * 100}%`,
                    top: `${box[1] * 100}%`,
                    width: `${Math.max(0, box[2] - box[0]) * 100}%`,
                    height: `${Math.max(0, box[3] - box[1]) * 100}%`,
                  }}
                />
              )}
            </div>
          ) : (
            <p className="empty-state">
              The event this was seen in has already been deleted, so the full frame is
              gone. The crop below is all that is left of it.
            </p>
          )}
        </div>

        <div className="candidate-review-side">
          <div className="candidate-review-crop">
            {candidate.has_image && candidate.image_url ? (
              <AuthImage src={candidate.image_url} eager alt="" />
            ) : (
              <span className="recog-candidate-noimg">
                {candidate.plate || 'no crop saved'}
              </span>
            )}
            <span className="control-hint">
              {isFace
                ? 'The part that gets enrolled.'
                : candidate.plate
                  ? `Read as ${candidate.plate}.`
                  : 'No plate text was read.'}
            </span>
          </div>

          <dl className="candidate-review-facts">
            <div>
              <dt>Camera</dt>
              <dd>{titleCase(candidate.camera)}</dd>
            </div>
            <div>
              <dt>Seen</dt>
              <dd>{formatDateTime(candidate.created_at)}</dd>
            </div>
            <div>
              <dt>Legibility</dt>
              <dd>{Math.round(candidate.quality * 100)}%</dd>
            </div>
          </dl>

          {candidate.event_id !== null && (
            <p className="control-hint">
              <Link to={`/events/${candidate.event_id}`}>Open the full event</Link>
            </p>
          )}

          <label className="field">
            <span>Add to</span>
            <select
              value={choice}
              onChange={(e) => setChoice(e.target.value === '' ? '' : Number(e.target.value))}
              disabled={busy || profiles.length === 0}
            >
              <option value="">
                {profiles.length === 0
                  ? isFace
                    ? 'No people yet'
                    : 'No vehicles yet'
                  : 'Choose someone…'}
              </option>
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <p className="control-hint">
            Adding this shot appends it as a reference. Nothing is trained and nothing is
            overwritten — remove the sample later and it is as if you never added it.
          </p>

          <div className="modal-actions candidate-review-actions">
            <button type="button" className="btn" onClick={onDelete} disabled={busy}>
              Not worth keeping
            </button>
            <button type="button" className="btn" onClick={onCreateProfile} disabled={busy}>
              New {isFace ? 'person' : 'vehicle'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || choice === ''}
              onClick={() => choice !== '' && onEnroll(choice)}
            >
              {busy ? 'Adding…' : isFace ? 'Add to this person' : 'Add to this vehicle'}
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}

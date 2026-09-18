/**
 * "Who" on an event — the one line recognition adds that a label cannot.
 *
 * A label says a person was there. This says WHICH person, or says plainly that
 * nobody enrolled matches. Both are worth showing: `known: false` is an answer
 * the operator acted on ("someone I don't know was at the door"), not a gap.
 *
 * The known/unknown distinction is carried by the ICON and the WORDING as well
 * as the colour, because this is precisely the row that must not depend on
 * colour perception.
 */
import type { EventRecognition } from '../lib/api';
import { recognitionLabel } from '../lib/api';

/** Filled person — a positive identification. */
function KnownIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0 1 16 0z" />
    </svg>
  );
}

/** Outlined person with a question mark — read, but not matched. */
function UnknownIcon() {
  return (
    <svg
      width="11"
      height="11"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="10" cy="8" r="3.4" />
      <path d="M3.5 20a6.5 6.5 0 0 1 11.2-4.5" />
      <path d="M18 14.6a1.9 1.9 0 1 1 2.6 1.8c-.7.3-1.1.9-1.1 1.6" />
      <path d="M19.5 21h.01" />
    </svg>
  );
}

export default function RecognitionChip({
  recognition,
  className = '',
}: {
  recognition: EventRecognition;
  className?: string;
}) {
  const known = recognition.known && Boolean(recognition.name || recognition.plate);
  const text = recognitionLabel(recognition);
  // A plate is a run of ambiguous glyphs; the monospaced variant is what keeps
  // 0 apart from O at this size. Applied whether or not the vehicle is
  // enrolled, since an unmatched plate is still read out character by character.
  const mono = recognition.kind === 'plate' && Boolean(recognition.plate);
  // The chip truncates; the title carries the full reading plus how sure it was,
  // which is the one number an operator questions a match with.
  const title = known && recognition.score > 0
    ? `${text} — ${Math.round(recognition.score * 100)}% match`
    : text;

  return (
    <span
      className={[
        'recog-chip',
        known ? 'recog-chip-known' : 'recog-chip-unknown',
        mono ? 'recog-chip-plate' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      title={title}
    >
      {known ? <KnownIcon /> : <UnknownIcon />}
      {text}
    </span>
  );
}

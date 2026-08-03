/**
 * Per-camera "Detect objects" picker.
 *
 * Reflects the ACTIVE detector model's label vocabulary: it fetches the running
 * model's classes (COCO-80 or Objects365-365) and offers them as a searchable,
 * filterable list — because 365 classes are far too many to render as a flat
 * chip cloud. Selected objects show as removable chips; a search box filters
 * the vocabulary; free-text entry still works (parity with the old ChipsInput)
 * so a label outside the active vocabulary can always be added and previously
 * stored selections are never lost. Falls back to the bundled COCO-80 list when
 * GET /api/detection/labels is unavailable.
 *
 * Admin-only surface (rendered inside Settings → Cameras, which is admin-gated).
 */
import { useEffect, useMemo, useState } from 'react';
import {
  loadActiveLabels,
  vocabularyLabel,
  vocabularySummary,
  type LabelVocab,
} from '../lib/labels';

/** Cap the rendered result chips so a 365-class vocabulary stays usable. */
const RESULTS_LIMIT = 60;

/** Normalize a free-text entry the same way ChipsInput did (UI/URL-safe labels). */
function normalizeLabel(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, '_');
}

interface ObjectPickerProps {
  value: string[];
  onChange: (next: string[]) => void;
}

export default function ObjectPicker({ value, onChange }: ObjectPickerProps) {
  const [vocab, setVocab] = useState<LabelVocab | null>(null);
  const [query, setQuery] = useState('');

  useEffect(() => {
    let alive = true;
    void loadActiveLabels().then((v) => {
      if (alive) setVocab(v);
    });
    return () => {
      alive = false;
    };
  }, []);

  const labels = vocab?.labels ?? [];
  const selected = value;

  const add = (label: string) => {
    if (label && !selected.includes(label)) onChange([...selected, label]);
  };
  const remove = (label: string) => onChange(selected.filter((v) => v !== label));

  // Case-insensitive, underscore-insensitive substring match ("traffic light"
  // finds "traffic_light"). Selected labels are excluded from the add list.
  const q = query.trim().toLowerCase().replace(/_/g, ' ');
  const matches = useMemo(() => {
    const sel = new Set(selected);
    const pool = labels.filter((l) => !sel.has(l));
    if (!q) return pool;
    return pool.filter((l) => l.toLowerCase().replace(/_/g, ' ').includes(q));
  }, [labels, selected, q]);

  const normalizedDraft = normalizeLabel(query);
  const canAddFree =
    normalizedDraft.length > 0 &&
    !selected.includes(normalizedDraft) &&
    !matches.includes(normalizedDraft);

  const commitEnter = () => {
    if (query.trim() === '') return;
    if (matches.length > 0) {
      add(matches[0]);
      setQuery('');
    } else if (canAddFree) {
      add(normalizedDraft);
      setQuery('');
    }
  };

  const shown = matches.slice(0, RESULTS_LIMIT);
  const overflow = matches.length - shown.length;
  const vocabName = vocab ? vocabularyLabel(vocab.vocabulary) : '';

  return (
    <div className="object-picker">
      <div className="chips object-picker-selected">
        {selected.length === 0 && (
          <span className="muted small">No objects selected — this camera won&rsquo;t raise events.</span>
        )}
        {selected.map((v) => (
          <button
            key={v}
            type="button"
            className="chip chip-on"
            onClick={() => remove(v)}
            aria-label={`Remove ${v}`}
            title="Remove"
          >
            {v} <span aria-hidden="true">×</span>
          </button>
        ))}
      </div>

      <input
        className="chip-input object-picker-search"
        value={query}
        placeholder={vocab ? `Search ${vocab.count} ${vocabName} classes…` : 'Search classes…'}
        aria-label="Search detectable object classes"
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            commitEnter();
          }
        }}
      />

      <div className="chips object-picker-results">
        {shown.map((l) => (
          <button key={l} type="button" className="chip" onClick={() => add(l)}>
            {l}
          </button>
        ))}
        {overflow > 0 && (
          <span className="object-picker-more muted small">+{overflow} more — keep typing to narrow</span>
        )}
        {shown.length === 0 && canAddFree && (
          <button type="button" className="chip object-picker-add" onClick={commitEnter}>
            + Add &ldquo;{normalizedDraft}&rdquo;
          </button>
        )}
        {shown.length === 0 && !canAddFree && query.trim() !== '' && (
          <span className="muted small">Already selected.</span>
        )}
      </div>

      <div className="object-picker-caption control-hint">
        {vocab ? (
          <>
            {vocabularySummary(vocab.vocabulary, vocab.count)}
            {vocab.source === 'fallback' && ' · built-in list (active model vocabulary unavailable)'}
          </>
        ) : (
          'Loading vocabulary…'
        )}
      </div>
    </div>
  );
}

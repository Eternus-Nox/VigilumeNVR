/**
 * Active-model label vocabulary: a resilient loader + display helpers shared by
 * the per-camera object picker and the detection-model manager.
 *
 * The backend exposes the running model's classes at GET /api/detection/labels
 * (80 for COCO, 365 for Objects365). This module fetches that and degrades
 * gracefully — any failure (endpoint absent, 403/404, network) falls back to
 * the bundled COCO-80 list so the picker always has a sensible vocabulary.
 */
import { api } from './api';
import { COCO_LABELS } from './coco';
import { titleCase } from './format';

export interface LabelVocab {
  /** Class names the active model can detect (underscore-safe). */
  labels: string[];
  /** Short vocabulary name ("coco" | "objects365" | …). */
  vocabulary: string;
  /** Class count (labels.length). */
  count: number;
  /** Where the list came from — "fallback" means the endpoint was unavailable. */
  source: 'backend' | 'fallback';
}

/** The COCO-80 fallback vocabulary, used when the labels endpoint is unavailable. */
export const COCO_VOCAB: LabelVocab = {
  labels: [...COCO_LABELS],
  vocabulary: 'coco',
  count: COCO_LABELS.length,
  source: 'fallback',
};

/**
 * Fetch the active model's label vocabulary, never throwing: on any error (or
 * an empty/invalid payload) it resolves to the bundled COCO-80 list. Callers
 * can distinguish a real response from the fallback via `source`.
 */
export async function loadActiveLabels(): Promise<LabelVocab> {
  try {
    const r = await api.detectionLabels();
    if (r && Array.isArray(r.labels) && r.labels.length > 0) {
      const labels = r.labels.filter((l) => typeof l === 'string' && l.length > 0);
      if (labels.length > 0) {
        return {
          labels,
          vocabulary: r.vocabulary || 'coco',
          count: typeof r.count === 'number' ? r.count : labels.length,
          source: 'backend',
        };
      }
    }
  } catch {
    /* endpoint absent / 403 / network — fall back to COCO-80 below */
  }
  return { ...COCO_VOCAB };
}

/**
 * Human-readable vocabulary name for a machine key ("coco" → "COCO",
 * "objects365" → "Objects365"), tolerant of unknown values.
 */
export function vocabularyLabel(vocabulary: string | undefined | null): string {
  const key = (vocabulary ?? '').trim().toLowerCase();
  if (!key) return 'COCO';
  if (key === 'objects365' || key === 'obj365' || key === 'o365') return 'Objects365';
  if (key === 'coco') return 'COCO';
  return titleCase(vocabulary as string);
}

/** "COCO · 80 classes" / "Objects365 · 365 classes". */
export function vocabularySummary(vocabulary: string | undefined | null, count: number | null): string {
  const name = vocabularyLabel(vocabulary);
  return count != null ? `${name} · ${count} classes` : name;
}

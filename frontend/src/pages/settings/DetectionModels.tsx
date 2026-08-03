/**
 * Detection-model tier manager (Settings → Recording → Object detection).
 *
 * Three tier cards (Lightweight / Balanced / Heavy) map to the D-FINE model
 * keys. Each card has a state-aware primary action:
 *   absent  → "Download"  → progress bar while downloading/verifying
 *   ready   → "Use this model" (activate)  →  "Enabled" (disabled) once active
 *   error   → the failure detail + "Retry".
 * Downloaded, non-active models can be deleted to free disk.
 *
 * A status strip up top reflects the active model's live load state, driven by
 * `model_status` WS pushes (via AppState) with a GET poll fallback while any
 * model is downloading/verifying. The GPU-unavailable / model-integrity states
 * come from the detector self-test (GET /api/system/detector), matching the
 * System tab's derivation.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  api,
  ApiError,
  type DetectionModel,
  type DetectionModelInfo,
  type DetectionModelsResponse,
  type DetectorStatus,
} from '../../lib/api';
import { vocabularyLabel } from '../../lib/labels';
import { useAppState, useModelStatuses } from '../../state/AppState';

const formatMB = (bytes: number): string => `${Math.round(bytes / 1_000_000)} MB`;

/**
 * The label vocabulary a model detects, derived from the backend fields when
 * present and inferred from the key otherwise, so a new model renders sensibly
 * even before the backend fills in `vocabulary`/`num_classes`.
 */
function modelVocab(m: DetectionModelInfo): { key: string; name: string; count: number | null } {
  const raw = m.vocabulary?.trim();
  const count = typeof m.num_classes === 'number' ? m.num_classes : null;
  if (raw) return { key: raw.toLowerCase(), name: vocabularyLabel(raw), count };
  const k = m.key.toLowerCase();
  if (k.includes('365') || k.includes('obj365') || k.includes('objects365')) {
    return { key: 'objects365', name: 'Objects365', count: count ?? 365 };
  }
  return { key: 'coco', name: 'COCO', count: count ?? 80 };
}

interface VocabGroup {
  key: string;
  name: string;
  count: number | null;
  models: DetectionModelInfo[];
}

/** Group models by vocabulary, preserving the backend's model order. */
function groupByVocab(models: DetectionModelInfo[]): VocabGroup[] {
  const groups: VocabGroup[] = [];
  const index = new Map<string, VocabGroup>();
  for (const m of models) {
    const v = modelVocab(m);
    let g = index.get(v.key);
    if (!g) {
      g = { key: v.key, name: v.name, count: v.count, models: [] };
      index.set(v.key, g);
      groups.push(g);
    }
    g.models.push(m);
  }
  return groups;
}

interface Props {
  /** Reports the active model key on load and after each activation, so the
   *  parent can keep settings.detection.model in sync for the confidence PUT. */
  onActiveModel?: (key: DetectionModel) => void;
}

export default function DetectionModels({ onActiveModel }: Props) {
  const { pushToast } = useAppState();
  const { modelStatuses } = useModelStatuses();
  const [resp, setResp] = useState<DetectionModelsResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [detector, setDetector] = useState<DetectorStatus | null>(null);
  const [busyKey, setBusyKey] = useState<DetectionModel | null>(null);
  // Expandable "what this model can detect" class list. Cached per vocabulary
  // (all COCO tiers share one list; the Objects365 model has its own), so
  // opening a second COCO card is instant. Only one card's list is open.
  const [openLabelsKey, setOpenLabelsKey] = useState<string | null>(null);
  const [labelsByVocab, setLabelsByVocab] = useState<Record<string, string[]>>({});
  const [labelFilter, setLabelFilter] = useState('');
  const lastRefreshRef = useRef(0);
  const onActiveModelRef = useRef(onActiveModel);
  onActiveModelRef.current = onActiveModel;

  const refreshModels = useCallback(async () => {
    try {
      const r = await api.detectionModels();
      lastRefreshRef.current = Date.now();
      setResp(r);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : 'Failed to load models');
    }
  }, []);

  const refreshDetector = useCallback(async () => {
    try {
      setDetector(await api.detector());
    } catch {
      setDetector(null);
    }
  }, []);

  useEffect(() => {
    void refreshModels();
    void refreshDetector();
  }, [refreshModels, refreshDetector]);

  // Toggle a model card's detectable-class list, lazy-loading the vocabulary
  // the first time it's opened.
  const toggleLabels = useCallback(
    async (m: DetectionModelInfo) => {
      setLabelFilter('');
      if (openLabelsKey === m.key) {
        setOpenLabelsKey(null);
        return;
      }
      setOpenLabelsKey(m.key);
      const vkey = modelVocab(m).key;
      if (labelsByVocab[vkey] === undefined) {
        try {
          const r = await api.detectionLabels(m.key);
          setLabelsByVocab((prev) => ({ ...prev, [vkey]: r.labels }));
        } catch {
          setLabelsByVocab((prev) => ({ ...prev, [vkey]: [] }));
        }
      }
    },
    [openLabelsKey, labelsByVocab],
  );

  // Merge live WS status over the REST snapshot; ignore a push older than the
  // last GET (that snapshot already reflects it). Order is the backend's — it
  // already returns models sensibly (COCO tiers nano→x, then Objects365), so we
  // never client-sort (that would break for any tier the frontend doesn't know).
  const models: DetectionModelInfo[] = (resp?.models ?? []).map((m) => {
    const live = modelStatuses[m.key];
    if (live && live.receivedAt > lastRefreshRef.current) {
      return {
        ...m,
        state: live.state,
        progress_pct: live.progress_pct,
        active: live.active,
        loaded: live.loaded,
      };
    }
    return m;
  });

  const groups = groupByVocab(models);

  const activeInfo =
    models.find((m) => m.active) ?? models.find((m) => m.key === resp?.active) ?? null;
  const activeKey = activeInfo?.key ?? resp?.active ?? null;
  const anyBusy = models.some((m) => m.state === 'downloading' || m.state === 'verifying');

  // Detector-derived load/GPU state for the active model (mirrors System tab).
  const gpuTripped =
    detector !== null &&
    !detector.ready &&
    detector.model_sha_ok &&
    detector.device !== 'cpu';
  // Only a DEFINITIVE checksum failure trips this banner — `model_sha_ok ===
  // false`, not merely falsy. During a background model swap the detector is
  // briefly not-ready with model_sha_ok null/undefined (not yet computed for the
  // new model); that must read as "Loading…", not "file missing/corrupt".
  const modelBroken =
    detector !== null && !detector.ready && detector.model_sha_ok === false;
  const activeLoading =
    activeInfo?.state === 'ready' &&
    (!detector || (!detector.ready && !gpuTripped && !modelBroken));

  // Report the active key up to the parent whenever it settles.
  useEffect(() => {
    if (activeKey) onActiveModelRef.current?.(activeKey);
  }, [activeKey]);

  // Poll GET while a download/verify is in flight (WS fallback).
  useEffect(() => {
    if (!anyBusy) return;
    const t = setInterval(() => void refreshModels(), 2000);
    return () => clearInterval(t);
  }, [anyBusy, refreshModels]);

  // Poll the detector self-test while the active model is loading, so the strip
  // flips from "Loading model…" to "Ready" promptly.
  useEffect(() => {
    if (!activeLoading) return;
    const t = setInterval(() => void refreshDetector(), 4000);
    return () => clearInterval(t);
  }, [activeLoading, refreshDetector]);

  const doDownload = useCallback(
    async (key: DetectionModel) => {
      setBusyKey(key);
      try {
        await api.downloadModel(key);
        await refreshModels();
      } catch (e) {
        pushToast({
          kind: 'error',
          title: 'Download failed',
          body: e instanceof Error ? e.message : '',
        });
      } finally {
        setBusyKey(null);
      }
    },
    [refreshModels, pushToast],
  );

  const doActivate = useCallback(
    async (key: DetectionModel) => {
      setBusyKey(key);
      try {
        await api.activateModel(key);
        onActiveModelRef.current?.(key);
        await Promise.all([refreshModels(), refreshDetector()]);
      } catch (e) {
        pushToast({
          kind: 'error',
          title: 'Could not switch model',
          body: e instanceof Error ? e.message : '',
        });
      } finally {
        setBusyKey(null);
      }
    },
    [refreshModels, refreshDetector, pushToast],
  );

  const doDelete = useCallback(
    async (key: DetectionModel) => {
      setBusyKey(key);
      try {
        await api.deleteModel(key);
        await refreshModels();
      } catch (e) {
        const msg =
          e instanceof ApiError && e.status === 409
            ? 'Cannot delete the active model — switch to another tier first.'
            : e instanceof Error
              ? e.message
              : '';
        pushToast({ kind: 'error', title: 'Delete failed', body: msg });
      } finally {
        setBusyKey(null);
      }
    },
    [refreshModels, pushToast],
  );

  const device = (detector?.device ?? resp?.device ?? null)?.toUpperCase() ?? null;

  const statusStrip = () => {
    if (!resp && loadError) {
      return (
        <div className="banner banner-error model-status">
          <span>{loadError}</span>
          <button type="button" className="btn btn-sm" onClick={() => void refreshModels()}>
            Retry
          </button>
        </div>
      );
    }
    if (!resp || !activeInfo) {
      return (
        <div className="card model-status busy">
          <span className="status-dot warn" />
          <span className="muted">Loading detection models…</span>
        </div>
      );
    }
    // Backend labels are the title-cased tier names, so show the label alone
    // (avoids a "Balanced (Balanced)" duplication); forward-compatible if the
    // label ever carries richer text than the tier name.
    const label = activeInfo.label;

    // Download / verify of the active model takes precedence.
    if (activeInfo.state === 'downloading' || activeInfo.state === 'verifying') {
      const verifying = activeInfo.state === 'verifying';
      return (
        <div className="card model-status busy">
          <span className="status-dot warn" />
          <div className="model-status-body">
            <strong>
              {verifying ? `Verifying ${activeInfo.label}…` : `Downloading ${activeInfo.label}…`}
              {!verifying && ` ${activeInfo.progress_pct}%`}
            </strong>
            <div className="model-progress" role="progressbar" aria-valuenow={activeInfo.progress_pct} aria-valuemin={0} aria-valuemax={100}>
              <div
                className="model-progress-fill"
                style={{ width: `${verifying ? 100 : activeInfo.progress_pct}%` }}
              />
            </div>
          </div>
        </div>
      );
    }
    if (activeInfo.state === 'error') {
      return (
        <div className="banner banner-error model-status">
          <span>Couldn't prepare {label}: {activeInfo.detail ?? 'download failed'}</span>
          <button
            type="button"
            className="btn btn-sm"
            disabled={busyKey === activeInfo.key}
            onClick={() => void doDownload(activeInfo.key)}
          >
            Retry
          </button>
        </div>
      );
    }
    if (gpuTripped) {
      return (
        <div className="banner banner-error model-status">
          <span>
            {detector?.kind === 'coral'
              ? 'Coral Edge TPU unavailable — detection is off until the device is reachable (check /dev/apex_0 and CORAL_DEVICE). ('
              : 'GPU unavailable — detection is off until a CUDA device is reachable (see '}
            <code>docs/setup-nvidia.md</code>).
          </span>
        </div>
      );
    }
    if (modelBroken) {
      return (
        <div className="banner banner-error model-status">
          <span>Active model file missing or failed checksum — check the backend logs.</span>
        </div>
      );
    }
    if (detector?.ready) {
      return (
        <div className="card model-status ok">
          <span className="status-dot ok" />
          <span>
            Ready — <strong>{label}</strong>
            {device ? ` running on ${device}` : ''}
          </span>
        </div>
      );
    }
    if (activeInfo.state === 'absent') {
      return (
        <div className="card model-status">
          <span className="status-dot warn" />
          <span>No detection model downloaded yet — pick a tier below.</span>
        </div>
      );
    }
    return (
      <div className="card model-status busy">
        <span className="status-dot warn" />
        <span>
          Loading <strong>{label}</strong>…
        </span>
      </div>
    );
  };

  const cardAction = (m: DetectionModelInfo) => {
    const busy = busyKey === m.key;
    if (m.state === 'downloading' || m.state === 'verifying') {
      const verifying = m.state === 'verifying';
      return (
        <div className="model-tier-progress">
          <div className="model-progress" role="progressbar" aria-valuenow={m.progress_pct} aria-valuemin={0} aria-valuemax={100}>
            <div
              className="model-progress-fill"
              style={{ width: `${verifying ? 100 : m.progress_pct}%` }}
            />
          </div>
          <span className="control-hint">
            {verifying ? 'Verifying…' : `Downloading… ${m.progress_pct}%`}
          </span>
        </div>
      );
    }
    if (m.state === 'error') {
      return (
        <div className="model-tier-error">
          <span className="form-error">{m.detail ?? 'Download failed.'}</span>
          <button
            type="button"
            className="btn btn-sm"
            disabled={busy}
            onClick={() => void doDownload(m.key)}
          >
            {busy ? 'Retrying…' : 'Retry'}
          </button>
        </div>
      );
    }
    if (m.state === 'ready') {
      if (m.active) {
        return (
          <button type="button" className="btn btn-sm btn-block" disabled>
            Enabled
          </button>
        );
      }
      return (
        <button
          type="button"
          className="btn btn-sm btn-primary btn-block"
          disabled={busy}
          onClick={() => void doActivate(m.key)}
        >
          {busy ? 'Activating…' : 'Use this model'}
        </button>
      );
    }
    // absent
    return (
      <button
        type="button"
        className="btn btn-sm btn-block"
        disabled={busy}
        onClick={() => void doDownload(m.key)}
      >
        {busy ? 'Starting…' : `Download · ${formatMB(m.size_bytes)}`}
      </button>
    );
  };

  const renderLabelList = (vkey: string) => {
    const all = labelsByVocab[vkey];
    if (all === undefined) return <p className="muted small">Loading…</p>;
    if (all.length === 0) return <p className="muted small">Class list unavailable.</p>;
    const pretty = (l: string) => l.replace(/_/g, ' ');
    const q = labelFilter.trim().toLowerCase();
    const filtered = q ? all.filter((l) => pretty(l).includes(q)) : all;
    const shown = filtered.slice(0, 120);
    return (
      <div className="model-labels">
        {all.length > 40 && (
          <input
            className="chip-input"
            placeholder={`Search ${all.length} classes…`}
            value={labelFilter}
            onChange={(e) => setLabelFilter(e.target.value)}
          />
        )}
        <div className="chips model-labels-list">
          {shown.map((l) => (
            <span key={l} className="chip">
              {pretty(l)}
            </span>
          ))}
        </div>
        {filtered.length === 0 && <span className="muted small">No matching class.</span>}
        {filtered.length > shown.length && (
          <span className="muted small">
            +{filtered.length - shown.length} more — keep typing to narrow
          </span>
        )}
      </div>
    );
  };

  const renderCard = (m: DetectionModelInfo) => {
    const v = modelVocab(m);
    const vocabText = v.count != null ? `${v.name} · ${v.count} classes` : v.name;
    return (
      <div key={m.key} className={`card model-tier ${m.active ? 'active' : ''}`}>
        <div className="model-tier-head">
          <span className="model-tier-name">{m.label}</span>
          {/* "Enabled", matching the Edge TPU cards — the two backends present
              the same way, so the word for "this is the one in use" is shared. */}
          {m.active && <span className="pill pill-ok">Enabled</span>}
          {m.state === 'ready' && !m.active && (
            <button
              type="button"
              className="btn btn-sm btn-danger-ghost model-tier-del"
              disabled={busyKey === m.key}
              title={`Delete the ${m.label} model file`}
              onClick={() => void doDelete(m.key)}
            >
              Delete
            </button>
          )}
        </div>
        <p className="model-tier-blurb small">{m.blurb}</p>
        <div className="model-tier-meta">
          <span className="pill pill-vocab">{vocabText}</span>
          <span className="pill">{formatMB(m.size_bytes)}</span>
          <span className="pill">{m.approx_map.toFixed(1)} mAP</span>
          <span className="pill">{m.input_size}px</span>
        </div>
        {m.recommended_for && (
          <div className="model-tier-rec muted small">{m.recommended_for}</div>
        )}
        {v.count != null && (
          <div className="model-tier-labels">
            <button
              type="button"
              className="model-labels-toggle"
              aria-expanded={openLabelsKey === m.key}
              onClick={() => void toggleLabels(m)}
            >
              {openLabelsKey === m.key ? '▾' : '▸'} Detects {v.count} {v.name} classes
            </button>
            {openLabelsKey === m.key && renderLabelList(v.key)}
          </div>
        )}
        <div className="model-tier-action">{cardAction(m)}</div>
      </div>
    );
  };

  return (
    <div className="model-manager">
      {statusStrip()}
      {resp &&
        (groups.length === 0 ? (
          <p className="muted small">No detection models offered by the backend.</p>
        ) : (
          <div className="model-groups">
            {groups.map((g) => (
              <div key={g.key} className="model-vocab-group">
                {/* Section headers only appear once there's more than one
                    vocabulary, so the existing COCO-only view is unchanged. */}
                {groups.length > 1 && (
                  <div className="model-vocab-head">
                    <span className="model-vocab-name">{g.name}</span>
                    {g.count != null && (
                      <span className="model-vocab-count muted small">{g.count} classes</span>
                    )}
                  </div>
                )}
                <div className="model-tiers">{g.models.map(renderCard)}</div>
              </div>
            ))}
          </div>
        ))}
    </div>
  );
}

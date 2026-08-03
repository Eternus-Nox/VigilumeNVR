/**
 * Display helpers for the active inference backend (detector kind + device),
 * shared by the System and Detection settings surfaces so both describe the
 * running detector identically. The detector type is env/hardware-set on the
 * backend (SENTINEL_DETECTOR) and read-only in the UI.
 */
import type { DetectorDevice, DetectorKind } from './api';

/**
 * A friendly one-line name for the active detector's execution device, e.g.
 * "GPU — CUDA", "CPU". When the device is null the backend is not loaded / the
 * hardware is unavailable — the caller pairs this with a banner explaining why.
 */
export function describeDetector(device: DetectorDevice, kind?: DetectorKind): string {
  if (device === 'edgetpu') return 'Coral Edge TPU';
  if (device === 'cuda') return 'GPU — CUDA';
  if (device === 'cpu') return 'CPU';
  // No device: the detector did not come up. WHICH hardware failed depends on
  // the configured backend — saying "GPU unavailable" while the backend is set
  // to Coral sends you debugging CUDA for an Edge TPU problem.
  return kind === 'coral' ? 'Coral Edge TPU — unavailable' : 'GPU — unavailable';
}

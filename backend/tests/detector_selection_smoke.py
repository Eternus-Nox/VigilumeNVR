"""Detector SELECTION smoke: VIGILUME_DETECTOR must actually pick the device.

This suite exists because `onnx_cpu` was a silent no-op on a CUDA box for its
entire life, and nothing caught it. `build_detector` set `require_gpu=False`
for onnx_cpu, but the provider list was assembled purely from
`ort.get_available_providers()` — so CUDA was still offered, ORT still took it,
and a box running `VIGILUME_DETECTOR=onnx_cpu` logged
`GPU OK — provider=CUDAExecutionProvider`. The documented GPU-dropout fallback
ran on the very GPU it was meant to stand in for, and any measurement taken in
that mode was a GPU measurement.

The distinction the tests below pin down:
  * require_gpu=False -> "CPU is ACCEPTABLE if CUDA is missing" (a hard-fail gate)
  * force_cpu=True    -> "do not offer CUDA to ORT at all"     (a device choice)
Only the second keeps inference off the GPU.

Runs without onnxruntime-gpu: `_build_session_blocking` takes the `ort` module
as a PARAMETER, so a fake advertising CUDA proves the provider list directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.native.detector import DEFAULT_MODEL, OnnxDetector, build_detector  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"  FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


class _FakeConfig:
    """Stand-in for app.config.Config — build_detector only getattr()s these."""

    def __init__(self, detector: str, require_gpu: bool = True):
        self.detector = detector
        self.require_gpu = require_gpu


class _FakeSession:
    def __init__(self, providers):
        # Mirror ORT: the ACTIVE provider is the first one it was handed.
        self._providers = [p[0] if isinstance(p, tuple) else p for p in providers]

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        class _In:
            name = "images"

        return [_In()]

    def run(self, outs, feed):
        return [np.zeros((1, 1, 1), dtype=np.float32) for _ in outs]


class _FakeSessionOptions:
    def __init__(self):
        self.log_severity_level = 0
        self.inter_op_num_threads = 0
        self.intra_op_num_threads = 0
        self._entries = {}

    def add_session_config_entry(self, k, v):
        self._entries[k] = v


class _FakeOrt:
    """onnxruntime stand-in that ALWAYS advertises CUDA — the whole point is to
    prove force_cpu declines it rather than that CUDA is absent."""

    def __init__(self):
        self.last_providers = None

    def get_available_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    SessionOptions = _FakeSessionOptions

    def InferenceSession(self, path, sess_options=None, providers=None):
        self.last_providers = providers
        return _FakeSession(providers)


def _providers_for(force_cpu: bool):
    det = OnnxDetector(
        models_dir=Path("/nonexistent"),
        model_key=DEFAULT_MODEL,
        confidence=0.5,
        require_gpu=False,
        force_cpu=force_cpu,
    )
    ort = _FakeOrt()
    _session, active, _warm, _infer = det._build_session_blocking(ort, Path("m.onnx"))
    names = [p[0] if isinstance(p, tuple) else p for p in ort.last_providers]
    return names, active


def factory_flags() -> None:
    print("\n== build_detector flag resolution ==")

    cpu = build_detector(
        config=_FakeConfig("onnx_cpu", require_gpu=True),
        models_dir=Path("/nonexistent"),
        model_key=DEFAULT_MODEL,
        confidence=0.5,
    )
    check(cpu._force_cpu is True, "onnx_cpu sets force_cpu")
    check(
        cpu._require_gpu is False,
        "onnx_cpu clears require_gpu even when VIGILUME_REQUIRE_GPU=1",
    )

    gpu = build_detector(
        config=_FakeConfig("onnx", require_gpu=True),
        models_dir=Path("/nonexistent"),
        model_key=DEFAULT_MODEL,
        confidence=0.5,
    )
    check(gpu._force_cpu is False, "onnx does NOT force cpu")
    check(gpu._require_gpu is True, "onnx keeps require_gpu")

    # The subtle one: accepting CPU is not the same as demanding it. This mode
    # must still PREFER CUDA when present.
    accept = build_detector(
        config=_FakeConfig("onnx", require_gpu=False),
        models_dir=Path("/nonexistent"),
        model_key=DEFAULT_MODEL,
        confidence=0.5,
    )
    check(
        accept._force_cpu is False,
        "onnx + VIGILUME_REQUIRE_GPU=0 accepts CPU but still does NOT force it",
    )


def provider_list() -> None:
    print("\n== provider list handed to onnxruntime (CUDA advertised as available) ==")

    names, active = _providers_for(force_cpu=True)
    check(
        "CUDAExecutionProvider" not in names,
        "force_cpu: CUDA is NOT offered to ORT even though it is available",
    )
    check(names == ["CPUExecutionProvider"], "force_cpu: CPU-only provider list")
    check(
        active == "CPUExecutionProvider",
        "force_cpu: the ACTIVE provider is CPU (the regression: it was CUDA)",
    )

    names, active = _providers_for(force_cpu=False)
    check(
        names[0] == "CUDAExecutionProvider",
        "not force_cpu: CUDA is still preferred when available",
    )
    check(active == "CUDAExecutionProvider", "not force_cpu: ACTIVE provider is CUDA")


def main() -> None:
    factory_flags()
    provider_list()
    print(f"\nALL {PASS} CHECKS PASSED (detector selection)")


if __name__ == "__main__":
    main()

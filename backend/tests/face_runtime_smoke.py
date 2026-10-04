#!/usr/bin/env python3
"""Face models on the GPU path (onnxruntime) give the SAME answers as OpenCV.

On a box whose detector runs on CUDA, YuNet and SFace run on onnxruntime
instead of OpenCV (the pip OpenCV wheel has no CUDA). Nothing about a wrong
port fails loudly — a subtly wrong decode returns plausible boxes, a wrong
blob returns a plausible vector in the wrong place, and every enrolled face
then quietly stops matching. So this pins, on real photographs:

1. The pinned YuNet file is patched on load to accept any input size, and the
   patch is exactly that: symbolic input/output dims, no stale shape hints.
2. Same faces, same landmarks, same scores as cv2.FaceDetectorYN.
3. Same embeddings as cv2.FaceRecognizerSF (cosine 1.0), on person-sized crops
   — so EMBEDDING_MODEL_KEY does not change and enrollments stay valid.
4. A big frame is searched small, and the faces come back in FULL pixels.
5. The runtime picks itself: OpenCV unless the detector is on CUDA.

The GPU itself is not available here; the onnxruntime path is forced onto the
CPU provider, which runs the identical graph.

NETWORK: the models (~37 MB) and two sample photos come from the pinned
opencv_zoo revision. Set VIGILUME_TEST_MODELS_DIR to reuse a directory that
already holds them. Offline with nothing cached, the suite SKIPS (exit 0).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

import cv2  # noqa: E402

from app.native import accel  # noqa: E402
from app.native.recognizer import (  # noqa: E402
    DETECT_MAX_SIDE, FaceRecognizer, _ZOO_BASE, dynamic_yunet,
)

PASS = 0
_failures: list[str] = []

PHOTOS = {
    "largest_selfie.jpg": f"{_ZOO_BASE}/face_detection_yunet/example_outputs/largest_selfie.jpg",
    "demo.jpg": f"{_ZOO_BASE}/face_recognition_sface/example_outputs/demo.jpg",
}


def check(cond: bool, label: str) -> None:
    global PASS
    PASS += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def fetch_photos(models_dir: Path) -> dict[str, np.ndarray]:
    import httpx

    out = {}
    for name, url in PHOTOS.items():
        path = models_dir / name
        if not path.is_file():
            try:
                r = httpx.get(url, follow_redirects=True, timeout=30)
                r.raise_for_status()
                path.write_bytes(r.content)
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not fetch {name}: {exc})")
                continue
        img = cv2.imread(str(path))
        if img is not None:
            out[name] = img
    return out


def norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class FakeDetector:
    def __init__(self, device: str, kind: str = "onnx") -> None:
        self.device, self.kind = device, kind


async def main() -> int:
    models_dir = Path(os.environ.get("VIGILUME_TEST_MODELS_DIR", "")
                      or tempfile.mkdtemp(prefix="vigilume-face-models-"))
    cv = FaceRecognizer(models_dir, runtime="opencv")
    ort = FaceRecognizer(models_dir, runtime="onnxruntime")
    if not await cv.load() or not await ort.load():
        print("SKIP: face models could not be loaded (offline?)")
        return 0
    photos = fetch_photos(models_dir)
    if not photos:
        print("SKIP: no sample photographs available")
        return 0

    print("\n1. the YuNet file is patched to take any input size")
    import onnxruntime

    raw = (models_dir / "yunet.onnx").read_bytes()
    patched = dynamic_yunet(raw)
    sess = onnxruntime.InferenceSession(patched, providers=["CPUExecutionProvider"])
    check(sess.get_inputs()[0].shape == [1, 3, "height", "width"],
          f"input is [1, 3, height, width] (got {sess.get_inputs()[0].shape})")
    check(all(isinstance(o.shape[1], str) for o in sess.get_outputs()),
          "every output's anchor count is symbolic")
    out = sess.run(None, {"input": np.zeros((1, 3, 448, 320), np.float32)})
    check(out[0].shape == (1, (448 // 8) * (320 // 8), 1),
          "and a 320x448 input runs, with the anchor grid that size implies")
    check(len(patched) < len(raw), "the stale 640x640 shape hints are dropped")
    check(ort.runtime == "onnxruntime" and cv.runtime == "opencv",
          "forced runtimes report themselves")

    print("\n2-3. same faces and same embeddings as OpenCV")
    det_diff, score_diff, worst_cos, crops, counts = 0.0, 0.0, 1.0, 0, [0, 0]
    for name, img in photos.items():
        a = await cv.detect(img)
        b = await ort.detect(img)
        counts[0] += len(a)
        counts[1] += len(b)
        for fa in a:
            fb = min(b, key=lambda f: abs(f.box[0] - fa.box[0]) + abs(f.box[1] - fa.box[1]),
                     default=None)
            if fb is None:
                det_diff = float("inf")
                continue
            det_diff = max(det_diff, float(np.abs(fa._raw[:14] - fb._raw[:14]).max()))
            score_diff = max(score_diff, abs(fa.score - fb.score))
            # A person-sized crop around the face: what the face pass sees.
            x1, y1, x2, y2 = fa.box
            fw = x2 - x1
            crop = img[int(max(0, y1 - 0.5 * fw)):int(min(img.shape[0], y2 + 4 * fw)),
                       int(max(0, x1 - fw)):int(min(img.shape[1], x2 + fw))]
            ca, cb = await cv.detect(crop), await ort.detect(crop)
            if not ca or not cb:
                continue
            ca = max(ca, key=lambda f: f.width)
            cb = max(cb, key=lambda f: f.width)
            ea = await cv.feature(await cv.align(crop, ca))
            eb = await ort.feature(await ort.align(crop, cb))
            worst_cos = min(worst_cos, float(norm(ea) @ norm(eb)))
            crops += 1
    check(counts[0] == counts[1] and counts[0] > 5,
          f"the same number of faces on the photographs ({counts[0]} vs {counts[1]})")
    check(det_diff < 0.01, f"boxes and landmarks agree to {det_diff:.4f} px")
    check(score_diff < 1e-4, f"scores agree to {score_diff:.6f}")
    check(crops >= 5 and worst_cos > 0.9999,
          f"embeddings on {crops} person-sized crops agree (worst cosine {worst_cos:.6f}) "
          "— the embedding space, and every enrolled face, is unchanged")

    print("\n4. a big frame is searched small; faces come back in full pixels")
    img = photos.get("demo.jpg", next(iter(photos.values())))
    big = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    check(max(big.shape[:2]) > DETECT_MAX_SIDE, f"the test frame is {big.shape[1]}x{big.shape[0]}")
    small_faces = await cv.detect(img)
    for rec in (cv, ort):
        big_faces = await rec.detect(big)
        ok = len(big_faces) >= len(small_faces) - 1 and big_faces
        if ok:
            for f in small_faces:
                g = min(big_faces, key=lambda q: abs(q.box[0] - 2.5 * f.box[0]) + abs(q.box[1] - 2.5 * f.box[1]))
                ok = ok and abs(g.box[0] - 2.5 * f.box[0]) < 0.1 * g.width
        check(bool(ok), f"{rec.runtime}: faces in the 2.5x frame sit at 2.5x the coordinates")
        aligned = await rec.align(big, big_faces[0]) if big_faces else None
        check(aligned is not None and aligned.shape[:2] == (112, 112),
              f"{rec.runtime}: and align from the full-resolution pixels")

    print("\n5. the runtime picks itself")
    auto_cpu = FaceRecognizer(models_dir, detector=FakeDetector("cpu"))
    await auto_cpu.load()
    check(auto_cpu.runtime == "opencv" and auto_cpu.device == "cpu",
          "detector on CPU -> OpenCV (faster than onnxruntime on a CPU)")
    auto_cuda = FaceRecognizer(models_dir, detector=FakeDetector("cuda"))
    await auto_cuda.load()
    check(auto_cuda.runtime == "opencv" and auto_cuda.ready,
          "detector on CUDA but no CUDA provider in this build -> still OpenCV, still ready")
    check(ort.status()["runtime"] == "onnxruntime" and cv.status()["device"] == "cpu",
          "status() reports runtime and device")
    rep = accel.report(FakeDetector("cuda"), face_device="cuda")
    check(rep["face_detect"]["device"] == "cuda" and rep["face_embed"]["device"] == "cuda",
          "the device report shows faces on the GPU when they are")
    rep = accel.report(FakeDetector("cuda"), face_device="cpu")
    check(rep["face_detect"]["device"] == "cpu" and "without CUDA" in rep["face_detect"]["why"],
          "and explains a CPU face stage")
    check(not any(k.startswith("plate") for k in rep),
          "and no plate stages are reported — plate reading was removed")

    print("\n6. recognition waits for the detector, and follows it when it moves")
    # At boot the detector warms up in the background; models loaded before it
    # has a device used to follow "unknown" onto the CPU for the whole run.
    late = FakeDetector("")

    async def resolve_later():
        await asyncio.sleep(0.4)
        late.device = "cpu"

    waiter = FaceRecognizer(models_dir, detector=late)
    t0 = asyncio.get_running_loop().time()
    asyncio.ensure_future(resolve_later())
    await waiter.load()
    waited = asyncio.get_running_loop().time() - t0
    check(waited >= 0.35 and waiter._built_for == ("cpu", "onnx"),
          f"face models wait for the detector's device before building ({waited:.2f} s)")
    check(not waiter.stale_device(), "and are not stale once built for it")
    late.device = "cuda"
    check(waiter.stale_device(),
          "the detector moving (e.g. coming up on CUDA after boot) marks them stale")
    waiter.close()
    await waiter.load()
    check(waiter._built_for == ("cuda", "onnx") and not waiter.stale_device(),
          "a rebuild follows it, once")

    print()
    if _failures:
        print(f"{len(_failures)} of {PASS} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {PASS} CHECKS PASSED (face models: onnxruntime == OpenCV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

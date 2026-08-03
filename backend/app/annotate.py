"""Snapshot annotation with Supervision (boxes + labels + count banner).

CPU-bound OpenCV/numpy work — callers run these through asyncio.to_thread.
The snapshot jpg is a clean detect-resolution frame, so the event's
`snapshot.box` ([x1, y1, x2, y2] in detect pixels) maps 1:1 onto it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import supervision as sv

log = logging.getLogger(__name__)

_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_LABEL_ANNOTATOR = sv.LabelAnnotator(text_scale=0.55, text_thickness=1, text_padding=6)

# Irregular plurals for the count banner; everything else gets a plain "s".
_PLURALS = {"person": "people", "bus": "buses", "sheep": "sheep"}


def plural_label(label: str, count: int) -> str:
    """'1 person', '2 people', '3 dogs' — used for banner and push body."""
    label = label.replace("_", " ")
    if count == 1:
        return f"1 {label}"
    return f"{count} {_PLURALS.get(label, label + 's')}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scaled_clamped_box(
    box: Any, w: int, h: int, sx: float, sy: float
) -> Optional[tuple[float, float, float, float]]:
    """Rescale a detect-pixel box by (sx, sy) and clamp it to the image.
    Returns None when the box is missing, malformed or degenerate."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    x1, x2 = max(0.0, min(x1, w - 1)), max(1.0, min(x2, float(w)))
    y1, y2 = max(0.0, min(y1, h - 1)), max(1.0, min(y2, float(h)))
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def _scene_banner(scene: Sequence[Any]) -> str:
    """'2 people, 1 dog' — one count per distinct label in the scene, ordered
    by descending count then label. Empty (no valid labels) -> ''."""
    counts: dict[str, int] = {}
    for obj in scene:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        if not label:
            continue
        counts[str(label)] = counts.get(str(label), 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(plural_label(label, count) for label, count in ordered)


def _draw_banner(image: np.ndarray, text: str) -> np.ndarray:
    """Render a dark banner strip above the frame with the count text."""
    h, w = image.shape[:2]
    strip_h = max(30, round(h * 0.075))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = strip_h / 46.0
    thickness = max(1, round(scale * 2))
    strip = np.full((strip_h, w, 3), (24, 24, 24), dtype=np.uint8)
    (_, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = (strip_h + text_h) // 2
    cv2.putText(strip, text, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([strip, image])


def annotate_event_snapshot(
    jpeg_bytes: bytes,
    box: Optional[Sequence[float]],
    label: str,
    score: float,
    count: int,
    detect_dims: Optional[tuple[int, int]] = None,
    scene: Optional[Sequence[dict[str, Any]]] = None,
    draw_boxes: bool = True,
) -> Optional[bytes]:
    """Draw a box + '{label} {score}%' for EVERY detected/counted object plus a
    count banner. Returns annotated JPEG bytes, or None if the input can't be
    decoded.

    ``draw_boxes=False`` (settings.notifications.draw_boxes) skips the box +
    label drawing entirely — the count banner is still added, so the snapshot
    stays clean while the summary survives.

    ``scene`` (preferred) is a list of ``{box, label, score}`` objects — one per
    tracked object present in the saved frame — so a multi-object event boxes all
    of them and the banner summarizes every count (e.g. "2 people, 1 dog"). When
    ``scene`` is absent or empty the legacy single-box path runs off ``box`` /
    ``label`` / ``count`` (doorbell/audio/legacy rows).

    All boxes are in detect-stream pixels. The native engine saves snapshots at
    detect resolution so they map 1:1; defensively, when ``detect_dims`` is known
    and differs from the image dimensions (e.g. legacy snapshots from
    pre-standalone installs), every box is rescaled by the SAME factor to fit."""
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        log.warning("annotate: could not decode snapshot jpeg (%d bytes)", len(jpeg_bytes))
        return None

    h, w = image.shape[:2]
    sx = sy = 1.0
    if detect_dims is not None:
        dw, dh = detect_dims
        if dw > 0 and dh > 0 and (dw, dh) != (w, h):
            sx, sy = w / dw, h / dh

    # (xyxy, label, score) for each object to draw. Prefer the full scene;
    # otherwise fall back to the single best box for backward compatibility.
    entries: list[tuple[tuple[float, float, float, float], str, float]] = []
    if scene:
        for obj in scene:
            if not isinstance(obj, dict):
                continue
            xyxy = _scaled_clamped_box(obj.get("box"), w, h, sx, sy)
            if xyxy is None:
                continue
            entries.append(
                (xyxy, str(obj.get("label") or label), _as_float(obj.get("score"), score))
            )
    elif box is not None:
        xyxy = _scaled_clamped_box(box, w, h, sx, sy)
        if xyxy is not None:
            entries.append((xyxy, label, score))

    if entries and draw_boxes:
        # One class_id per distinct label so same-label boxes share a colour.
        label_ids: dict[str, int] = {}
        class_id = [label_ids.setdefault(lb, len(label_ids)) for _, lb, _ in entries]
        detections = sv.Detections(
            xyxy=np.array([e[0] for e in entries], dtype=np.float32),
            confidence=np.array([e[2] for e in entries], dtype=np.float32),
            class_id=np.array(class_id),
        )
        image = _BOX_ANNOTATOR.annotate(scene=image, detections=detections)
        image = _LABEL_ANNOTATOR.annotate(
            scene=image,
            detections=detections,
            labels=[f"{lb.replace('_', ' ')} {round(sc * 100)}%" for _, lb, sc in entries],
        )

    banner = _scene_banner(scene) if scene else ""
    if not banner:  # empty/absent scene -> legacy single-label banner
        banner = plural_label(label, max(count, 1))
    image = _draw_banner(image, banner)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return bytes(buf)


def crop_thumbnail(
    jpeg_bytes: bytes,
    box: Optional[Sequence[float]],
    detect_dims: Optional[tuple[int, int]] = None,
    pad: float = 0.15,
    max_px: int = 320,
) -> Optional[bytes]:
    """Crop a small thumbnail around a detection box for the suppression record.

    ``box`` is [x1, y1, x2, y2] in detect-stream pixels; ``detect_dims`` scales
    it onto the actual snapshot resolution when they differ (same 1:1/rescale
    contract as annotate_event_snapshot). The crop is padded by ``pad`` (fraction
    of box size) on each side, clamped to the image, then downscaled so its
    longest side is at most ``max_px``. Returns JPEG bytes, or None when the
    image can't be decoded or the box is missing/degenerate (caller degrades to
    no-thumb)."""
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    sx = 1.0
    top_offset = 0.0
    if detect_dims is not None:
        dw, dh = detect_dims
        if dw > 0 and dh > 0:
            # The width ratio is the true UNIFORM scale. The on-disk event
            # snapshot may carry a top count-banner (annotate._draw_banner
            # vstacks a strip ABOVE the frame), so any leftover height is a top
            # TRANSLATION, not a vertical scale — apply it as an offset, never
            # via a taller sy (which would crop the box too high). A clean frame
            # (no banner) yields top_offset == 0, so this is a no-op there.
            sx = w / dw
            top_offset = max(0.0, h - dh * sx)
    xyxy = _scaled_clamped_box(box, w, h, sx, sx)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    if top_offset:
        y1 = min(float(h), y1 + top_offset)
        y2 = min(float(h), y2 + top_offset)
    pad_x = (x2 - x1) * pad
    pad_y = (y2 - y1) * pad
    cx1 = int(max(0, round(x1 - pad_x)))
    cy1 = int(max(0, round(y1 - pad_y)))
    cx2 = int(min(w, round(x2 + pad_x)))
    cy2 = int(min(h, round(y2 + pad_y)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    if longest > max_px:
        scale = max_px / float(longest)
        crop = cv2.resize(
            crop, (max(1, round(cw * scale)), max(1, round(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return bytes(buf) if ok else None


def banner_only_snapshot(jpeg_bytes: bytes, text: str) -> Optional[bytes]:
    """Banner-only annotation for doorbell/audio snapshots (no box data)."""
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = _draw_banner(image, text)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return bytes(buf) if ok else None

/**
 * COCO-80 label table — mirrors the backend's `native/coco_labels.py`
 * (contiguous ids 0..79, underscore-safe names, D-FINE output space).
 *
 * Used by the per-camera object picker as its fallback vocabulary when
 * GET /api/detection/labels is unavailable, so the picker never regresses to
 * the old 4-object hint even offline. When a COCO model is active the labels
 * endpoint returns this same set; the constant just lets the UI degrade
 * gracefully without a round-trip.
 */
export const COCO_LABELS: readonly string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
  'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
  'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
  'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
  'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant',
  'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
  'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
  'hair_drier', 'toothbrush',
];

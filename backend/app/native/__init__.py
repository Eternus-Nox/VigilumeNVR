"""Vigilume native engine — standalone detection, recording and live view.

Modules (see docs/native-mode-design.md for the design record; the modes
described there were dropped — this package IS the only media backend):

- ``coco_labels``  Label maps (``LABELMAPS``: COCO-80 + Objects365) and the
                   live active-model ``ID_TO_LABEL`` view the engine resolves
                   ``class_id -> label`` through; ``obj365_labels`` holds the
                   366-entry Objects365 table
- ``detector``     OnnxDetector: model download+SHA pinning, ORT session
                   (CUDA/CPU + VIGILUME_REQUIRE_GPU), D-FINE decode
- ``ingest``       FrameSource (ffmpeg rawvideo pipe, latest-frame drop,
                   staleness watchdog) + IngestManager (single inference
                   worker + per-camera ByteTrackTracker)
- ``engine``       DetectionEngine: track state machine -> EventsPipeline
- ``media``        MediaProvider protocol + NativeMediaProvider
- ``recorder``     Recorder: 24/7 segments + event clips
- ``streams``      RTSP URL resolution + go2rtc config generation/sync

Nothing in this package may import ``onnxruntime`` at module import time —
tests must run on hosts without the GPU wheel (see detector.py).
"""

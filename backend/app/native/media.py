"""MediaProvider protocol + NativeMediaProvider.

The media surface the events code consumes (EventsPipeline enrichment,
routers/events.py snapshot fallback + clip serving, routers/cameras.py live
snapshot). Historically this was an external NVR's HTTP client; natively it
is backed by the detection engine's in-memory frame cache and the recorder's
clip files. ``main.py`` injects a single instance as ``app.state.media``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from ..db import Database
    from .engine import DetectionEngine
    from .recorder import Recorder


class MediaProvider(Protocol):
    """Exact surface consumed by events_pipeline.py + the media routes."""

    async def event_snapshot(self, frigate_id: str, retries: int = 3) -> Optional[bytes]:
        """Clean (un-annotated) JPEG for an event id, or None."""
        ...

    async def latest_jpg(self, camera: str, height: Optional[int] = None) -> Optional[bytes]:
        """Most recent live frame for a camera as JPEG, or None."""
        ...

    async def detect_dims(self, camera: str) -> Optional[tuple[int, int]]:
        """(width, height) of the camera's detect stream, or None."""
        ...

    async def is_healthy(self) -> bool:
        """Media backend liveness (native: the engine is running)."""
        ...

    def clip_path(self, event_id: int) -> Path:
        """Filesystem path where the event's clip lives (may not exist)."""
        ...

    async def aclose(self) -> None: ...


class NativeMediaProvider:
    """MediaProvider over the native engine + recorder.

    - ``event_snapshot``: the detector's best-frame cache (open events and
      events ended <60 s ago — enrichment runs immediately, so the frame is
      still hot). The pipeline's annotated copy under /data/snapshots/ is
      what serves history afterwards; unknown/expired ids => None. The
      ``retries`` parameter is accepted for interface parity and ignored
      (there is nothing to wait for — the frame is either cached or gone).
    - ``latest_jpg``: engine frame cache; None when ingest is down (the
      camera-snapshot route then falls back to the Amcrest CGI snapshot).
    - ``detect_dims``: camera row detect_width/height — native snapshots are
      raw detect-res frames, so the annotator's rescale is a no-op.
    - ``clip_path``: recorder clip file for a DB event row id.
    """

    def __init__(self, db: "Database", engine: "DetectionEngine", recorder: "Recorder"):
        self._db = db
        self._engine = engine
        self._recorder = recorder

    async def event_snapshot(self, frigate_id: str, retries: int = 3) -> Optional[bytes]:
        return await asyncio.to_thread(self._engine.event_best_jpeg, frigate_id)

    async def latest_jpg(self, camera: str, height: Optional[int] = None) -> Optional[bytes]:
        return await asyncio.to_thread(self._engine.latest_frame_jpeg, camera, height)

    async def detect_dims(self, camera: str) -> Optional[tuple[int, int]]:
        cam = await self._db.get_camera(camera)
        if cam is None:
            return None
        width, height = cam.get("detect_width"), cam.get("detect_height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
        return None

    async def is_healthy(self) -> bool:
        return self._engine.running

    def clip_path(self, event_id: int) -> Path:
        return self._recorder.clip_path(event_id)

    async def aclose(self) -> None:
        return None

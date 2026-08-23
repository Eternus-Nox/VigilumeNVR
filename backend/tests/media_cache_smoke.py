#!/usr/bin/env python3
"""Cache headers on event media.

An event's saved snapshot and its clip are written once and never rewritten, so
they are safe to cache hard — and the cost of NOT doing so falls on the events
list, where every thumbnail otherwise revalidates on every scroll, on both the
web app and the phone.

The interesting case is the exception: before the annotated snapshot is written,
the SAME url serves the engine's clean best frame. Caching that would freeze the
un-annotated image in place, so it has to be explicitly uncacheable. These
checks exist mostly to keep that asymmetry from being "tidied" into consistency.

Driven through the real router with TestClient, so what is asserted is the
response a client actually receives — not a constant this file also defines.
Offline-runnable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import require_media_auth  # noqa: E402
from app.routers import events as events_router  # noqa: E402

_failures: list[str] = []
_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


class _FakeDB:
    def __init__(self, rows: dict[int, dict[str, Any]]) -> None:
        self._rows = rows

    async def get_event(self, event_id: int) -> Optional[dict[str, Any]]:
        return self._rows.get(event_id)


class _FakeMedia:
    """clip_path plus the engine-snapshot fallback the router reaches for."""

    def __init__(self, clips_dir: Path, best_frame: Optional[bytes]) -> None:
        self._clips = clips_dir
        self._best_frame = best_frame

    def clip_path(self, event_id: int) -> Path:
        return self._clips / f"{event_id}.mp4"

    async def event_snapshot(self, frigate_id: str, retries: int = 1) -> Optional[bytes]:
        return self._best_frame


class _FakeConfig:
    def __init__(self, snapshots_dir: Path) -> None:
        self.snapshots_dir = snapshots_dir


def _event(event_id: int, **over: Any) -> dict[str, Any]:
    row = {
        "id": event_id, "frigate_id": f"native.{event_id}", "camera": "front",
        "label": "person", "start_time": 1_770_000_000.0, "end_time": 1_770_000_010.0,
        "has_clip": True, "record_enabled": True,
    }
    row.update(over)
    return row


def _cache_of(response: Any) -> str:
    return response.headers.get("cache-control", "<none>")


def main() -> int:
    print("event media cache headers")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-mediacache-"))
    try:
        snaps, clips = tmp / "snapshots", tmp / "clips"
        snaps.mkdir()
        clips.mkdir()
        # 1: snapshot + clip both on disk. 2: neither, so the snapshot falls
        # back to the engine's best frame.
        (snaps / "1.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
        (clips / "1.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

        app = FastAPI()
        app.include_router(events_router.router)
        app.dependency_overrides[require_media_auth] = lambda: {"sub": "test"}
        app.state.db = _FakeDB({1: _event(1), 2: _event(2)})
        app.state.media = _FakeMedia(clips, best_frame=b"\xff\xd8\xff\xe0best")
        app.state.config = _FakeConfig(snaps)

        with TestClient(app) as client:
            # --- the saved, immutable pair --------------------------------
            snap = client.get("/api/events/1/snapshot.jpg")
            check(snap.status_code == 200, "saved snapshot serves 200")
            check(
                "immutable" in _cache_of(snap) and "max-age=31536000" in _cache_of(snap),
                f"saved snapshot is cached hard ({_cache_of(snap)})",
            )
            check(
                _cache_of(snap).startswith("private"),
                "and privately — media auth means no shared cache should hold it",
            )
            check(
                snap.headers.get("etag") or snap.headers.get("last-modified"),
                "a validator is still present, so a client can revalidate if it wants",
            )

            clip = client.get("/api/events/1/clip.mp4")
            check(clip.status_code == 200, "clip serves 200")
            check(
                "immutable" in _cache_of(clip),
                f"clip is cached hard too ({_cache_of(clip)})",
            )

            # --- the exception: the pre-annotation fallback ---------------
            # Same URL will serve DIFFERENT bytes once the annotated copy lands.
            fallback = client.get("/api/events/2/snapshot.jpg")
            check(fallback.status_code == 200, "fallback snapshot serves 200")
            check(
                _cache_of(fallback) == "no-store",
                f"the pre-annotation best frame is NOT cached ({_cache_of(fallback)}) "
                "— the annotated copy replaces it at this same URL",
            )

            # --- ?download=1 keeps both headers ---------------------------
            dl = client.get("/api/events/1/snapshot.jpg", params={"download": 1})
            check(
                "attachment" in dl.headers.get("content-disposition", ""),
                "?download=1 still forces the attachment disposition",
            )
            check(
                "immutable" in _cache_of(dl),
                "and the cache header survives alongside it (neither dict clobbers the other)",
            )
            dlc = client.get("/api/events/1/clip.mp4", params={"download": 1})
            check(
                "attachment" in dlc.headers.get("content-disposition", "")
                and "immutable" in _cache_of(dlc),
                "same for the clip download",
            )

            # --- a missing event must not be cached at all ----------------
            missing = client.get("/api/events/99/snapshot.jpg")
            check(missing.status_code == 404, "unknown event 404s")
            check(
                "immutable" not in _cache_of(missing),
                f"and its 404 carries no immutable header ({_cache_of(missing)}) — "
                "an event whose snapshot has not arrived yet must stay re-fetchable",
            )

        print()
        if _failures:
            print(f"{len(_failures)} of {_checks} CHECKS FAILED")
            for f in _failures:
                print(f"  - {f}")
            return 1
        print(f"ALL {_checks} CHECKS PASSED (event media cache headers)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

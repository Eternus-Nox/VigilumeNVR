"""Space-based recording rotation ("overwrite the oldest with the newest").

Covers native/recorder.py ``space_pass`` + ``HourDirSizes``: the rolling window
that keeps 24/7 footage inside a storage cap and above a free-space floor by
deleting the OLDEST hour dirs first.

Why this is a separate suite from native_smoke.py, which owns day-based
retention: native_smoke reaches the network (model-download cases) and cannot
run in a sandboxed or offline environment, which is exactly where a
disk-management regression most needs catching. Everything here is local
filesystem work with an injected ``_disk_free``, so it runs anywhere.

Sections:
  1. HourDirSizes — totals, mtime-keyed caching, cache eviction on delete.
  2. cap rotation — over max_storage_gb deletes oldest-first, stops at the
     headroom target, leaves newer footage alone.
  3. free-space floor — same, driven by falling free space; the historical
     5 GB default applies when settings omit the new keys.
  4. safety — active hour dirs are never deleted, clips are NEVER deleted for
     space, nothing to prune degrades to a loud log rather than an exception.
  5. hysteresis — a second pass immediately after a rotation is a no-op.

    python backend/tests/storage_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="vigilume-storage-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

from app.config import Config  # noqa: E402
from app.native.recorder import (  # noqa: E402
    LOW_DISK_BYTES,
    HourDirSizes,
    Recorder,
    hour_dir_size,
    iter_hour_dirs,
)

GB = 1024**3
PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


DEFAULT_RECORDING = {"continuous_days": 7, "event_days": 14, "snapshot_days": 14}


class FakeSettings:
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False

    def __init__(self, recording: dict | None = None):
        self.recording = dict(recording or DEFAULT_RECORDING)


def make_config(tag: str) -> Config:
    cfg = Config()
    cfg.data_dir = TMP / tag / "data"
    cfg.media_dir = TMP / tag / "media"
    return cfg


def make_hour(root: Path, cam: str, day: str, hour: str, mtime: float, size: int) -> Path:
    """An hour dir holding ``size`` bytes across one segment, aged to ``mtime``.

    SPARSE, via truncate — the cases here model gigabyte-scale hour dirs, and
    writing those bytes for real would need tens of GB of scratch space and
    minutes of I/O to test arithmetic. ``hour_dir_size`` measures
    ``st_size``, which truncate sets exactly, so the code under test cannot
    tell the difference; only the filesystem's block count differs.
    """
    hd = root / cam / day / hour
    hd.mkdir(parents=True, exist_ok=True)
    seg = hd / "00.00.ts"
    with open(seg, "wb") as fh:
        fh.truncate(size)
    os.utime(seg, (mtime, mtime))
    os.utime(hd, (mtime, mtime))
    return hd


# =====================================================================
# 1. HourDirSizes
# =====================================================================


def size_cache_checks() -> None:
    print("1. HourDirSizes — totals, mtime-keyed cache, eviction")
    cfg = make_config("sizes")
    root = cfg.recordings_dir
    now = datetime(2026, 7, 1, 12, 0, 0).timestamp()
    a = make_hour(root, "cam", "2026-07-01", "01", now - 3000, 1000)
    b = make_hour(root, "cam", "2026-07-01", "02", now - 2000, 2500)

    check(hour_dir_size(a) == 1000, "hour_dir_size sums the segment bytes in one dir")
    check(hour_dir_size(root / "cam" / "nope") == 0,
          "hour_dir_size on a missing dir is 0, not an exception")

    cache = HourDirSizes()
    entries = iter_hour_dirs(root)
    check(cache.total(entries) == 3500, "total() sums every hour dir")

    # The whole point of the cache is that sealed hour dirs are NOT re-walked
    # every minute. Counting the real measurement calls is the only honest way
    # to show that — a size that merely stays the same proves nothing.
    import app.native.recorder as rec_mod
    real_size = rec_mod.hour_dir_size
    calls: list[Path] = []

    def counting_size(hd: Path) -> int:
        calls.append(hd)
        return real_size(hd)

    rec_mod.hour_dir_size = counting_size
    try:
        check(cache.total(iter_hour_dirs(root)) == 3500 and calls == [],
              "a second pass over unchanged dirs re-measures NOTHING (all cache hits)")

        c = make_hour(root, "cam", "2026-07-01", "03", now - 1000, 500)
        total = cache.total(iter_hour_dirs(root))
        check(total == 4000 and calls == [c],
              f"only the new/changed dir is measured (calls={[p.name for p in calls]})")
    finally:
        rec_mod.hour_dir_size = real_size

    cache.forget(b)
    check(b not in cache._sizes, "forget() drops an entry so a deleted dir is not counted")

    # Deleted dirs must fall out of the cache, or it grows forever.
    import shutil as _sh
    _sh.rmtree(c)
    cache.total(iter_hour_dirs(root))
    check(c not in cache._sizes, "entries for pruned dirs are evicted on the next pass")


# =====================================================================
# 2. cap rotation
# =====================================================================


def cap_checks() -> None:
    print("2. cap rotation — over max_storage_gb deletes oldest first")
    cfg = make_config("cap")
    root = cfg.recordings_dir
    now = datetime(2026, 7, 2, 12, 0, 0).timestamp()
    # 4 sealed hours of 1 GB each + plenty of free space, cap 3 GB.
    hours = [
        make_hour(root, "cam", "2026-07-01", f"{h:02d}", now - (10 - h) * 3600, GB)
        for h in range(1, 5)
    ]
    rec = Recorder(cfg, None, FakeSettings({**DEFAULT_RECORDING, "max_storage_gb": 3}))
    rec._disk_free = lambda: 500 * GB   # free space is fine; only the cap bites

    removed = rec.space_pass(now)
    # Target is cap minus headroom = 3 GB - max(1 GB, 2%) = 2 GB, so two go.
    check(removed == hours[:2],
          f"oldest-first: rotated {[p.name for p in removed]}, want {[p.name for p in hours[:2]]}")
    check(not hours[0].exists() and not hours[1].exists(), "rotated dirs are gone from disk")
    check(hours[2].is_dir() and hours[3].is_dir(), "newer footage is untouched")

    # Immediately re-running must do nothing — that is what headroom is for.
    check(rec.space_pass(now) == [], "hysteresis: a second pass right after rotating is a no-op")

    # cap 0 = uncapped, however much is stored.
    cfg2 = make_config("nocap")
    for h in range(1, 5):
        make_hour(cfg2.recordings_dir, "cam", "2026-07-01", f"{h:02d}", now - h * 3600, GB)
    rec2 = Recorder(cfg2, None, FakeSettings({**DEFAULT_RECORDING, "max_storage_gb": 0}))
    rec2._disk_free = lambda: 500 * GB
    check(rec2.space_pass(now) == [], "max_storage_gb=0 means no cap — nothing is rotated")


# =====================================================================
# 3. free-space floor
# =====================================================================


def floor_checks() -> None:
    print("3. free-space floor — rotates as free space falls")
    cfg = make_config("floor")
    root = cfg.recordings_dir
    now = datetime(2026, 7, 3, 12, 0, 0).timestamp()
    hours = [
        make_hour(root, "cam", "2026-07-01", f"{h:02d}", now - (10 - h) * 3600, GB)
        for h in range(1, 5)
    ]
    # Free space rises by 1 GB for each hour dir deleted (they are 1 GB each).
    rec = Recorder(cfg, None, FakeSettings({**DEFAULT_RECORDING, "min_free_gb": 10}))
    state = {"free": 8 * GB}
    rec._disk_free = lambda: state["free"]

    removed = rec.space_pass(now)
    # Floor 10 GB + 1 GB headroom = 11 GB wanted; 8 + 1 + 1 + 1 = 11 after three.
    check(removed == hours[:3],
          f"rotates oldest-first until the floor+headroom is met (got {len(removed)})")
    check(hours[3].is_dir(), "stops as soon as the target is reached")

    # Settings that predate these keys must behave exactly as before: 5 GB floor.
    cfg2 = make_config("legacy")
    old = make_hour(cfg2.recordings_dir, "cam", "2026-07-01", "01", now - 8 * 3600, GB)
    rec2 = Recorder(cfg2, None, FakeSettings())   # no max_storage_gb / min_free_gb
    rec2._disk_free = lambda: 4 * GB
    check(rec2.space_pass(now) == [old],
          f"a settings doc without the new keys still uses the {LOW_DISK_BYTES // GB} GB default floor")

    cfg3 = make_config("plenty")
    make_hour(cfg3.recordings_dir, "cam", "2026-07-01", "01", now - 8 * 3600, GB)
    rec3 = Recorder(cfg3, None, FakeSettings())
    rec3._disk_free = lambda: 500 * GB
    check(rec3.space_pass(now) == [], "no rotation at all when both limits are satisfied")


# =====================================================================
# 4. safety rails
# =====================================================================


def safety_checks() -> None:
    print("4. safety — active dirs, clips, and the nothing-to-prune path")
    now = datetime(2026, 7, 4, 12, 0, 0).timestamp()

    # An hour dir ffmpeg is writing RIGHT NOW must never be deleted, even when
    # that means failing to reach the target.
    cfg = make_config("active")
    active = make_hour(cfg.recordings_dir, "cam", "2026-07-04", "12", now, 4 * GB)
    rec = Recorder(cfg, None, FakeSettings({**DEFAULT_RECORDING, "max_storage_gb": 1}))
    rec._disk_free = lambda: 1 * GB
    check(rec.space_pass(now) == [] and active.is_dir(),
          "an actively-written hour dir is never rotated (ffmpeg is inside it)")

    # Event clips are the evidence; space pressure must not touch them.
    cfg2 = make_config("clips")
    old = make_hour(cfg2.recordings_dir, "cam", "2026-07-01", "01", now - 8 * 3600, GB)
    cfg2.clips_dir.mkdir(parents=True, exist_ok=True)
    clip = cfg2.clips_dir / "1.mp4"
    clip.write_bytes(b"x" * 4096)
    os.utime(clip, (now - 9 * 3600, now - 9 * 3600))
    rec2 = Recorder(cfg2, None, FakeSettings({**DEFAULT_RECORDING, "min_free_gb": 50}))
    rec2._disk_free = lambda: 1 * GB      # never satisfied, so it prunes everything it may
    removed = rec2.space_pass(now)
    check(removed == [old], "space pressure rotates continuous footage")
    check(clip.is_file(),
          "event clips are NEVER deleted for space — they expire only by event_days")

    # Empty tree, impossible target: a loud log, not a crash.
    cfg3 = make_config("empty")
    cfg3.recordings_dir.mkdir(parents=True, exist_ok=True)
    rec3 = Recorder(cfg3, None, FakeSettings({**DEFAULT_RECORDING, "min_free_gb": 99}))
    rec3._disk_free = lambda: 1 * GB
    check(rec3.space_pass(now) == [], "nothing to prune degrades to a log, never an exception")

    # A filesystem that cannot report free space must not block cap rotation.
    cfg4 = make_config("nofree")
    h = make_hour(cfg4.recordings_dir, "cam", "2026-07-01", "01", now - 8 * 3600, 4 * GB)
    rec4 = Recorder(cfg4, None, FakeSettings({**DEFAULT_RECORDING, "max_storage_gb": 1}))
    rec4._disk_free = lambda: None
    check(rec4.space_pass(now) == [h],
          "an unreadable free-space value still lets the cap rotate")


# =====================================================================
# 5. retention_pass still triggers a space pass
# =====================================================================


def retention_wiring_checks() -> None:
    print("5. the hourly retention pass still performs a space pass")
    cfg = make_config("wiring")
    now = datetime(2026, 7, 5, 12, 0, 0).timestamp()
    old = make_hour(cfg.recordings_dir, "cam", "2026-07-05", "01", now - 8 * 3600, GB)
    rec = Recorder(cfg, None, FakeSettings({**DEFAULT_RECORDING, "min_free_gb": 20}))
    rec._disk_free = lambda: 2 * GB
    result = rec.retention_pass(now)
    check(result["low_disk"] == [old],
          "retention_pass reports space rotation under its existing low_disk key")
    check(not old.exists(), "the hourly path really deletes, so a short-lived run still rotates")


def main() -> None:
    size_cache_checks()
    cap_checks()
    floor_checks()
    safety_checks()
    retention_wiring_checks()
    print(f"\nALL {PASS} CHECKS PASSED (space-based recording rotation)")


if __name__ == "__main__":
    main()

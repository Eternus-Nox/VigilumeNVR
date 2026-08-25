#!/usr/bin/env python3
"""Nightly cloud archive of event media — the decisions, not the transport.

Everything a wrong answer would quietly cost you: which days get uploaded (a
skipped one is gone once local retention passes), which files belong to a day
(a boundary event in two folders, or in neither), and which REMOTE folders get
deleted (the failure mode with no undo).

rclone itself is not exercised — no binary, no network, no cloud account here —
but its argv is, exactly as this codebase already tests ffmpeg's: the flags are
the part that decides whether an archive copies or destroys.

Offline-runnable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.archive import (  # noqa: E402
    MAX_BACKFILL_DAYS,
    build_rclone_copy_args,
    build_rclone_lsd_args,
    build_rclone_purge_args,
    day_bounds,
    day_folder,
    days_to_prune,
    days_to_upload,
    media_for_day,
    parse_day_folder,
    write_files_from,
)

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


D = date(2026, 8, 25)  # "today" throughout


def main() -> int:
    print("cloud archive planning")
    tmp = Path(tempfile.mkdtemp(prefix="vigilume-archive-test-"))
    try:
        # --- which days ---------------------------------------------------
        check(
            days_to_upload(today=D, last_uploaded=None) == [D - timedelta(days=1)],
            "a first run uploads YESTERDAY only — enabling archiving must not "
            "kick off an unannounced multi-gigabyte backfill",
        )
        check(
            days_to_upload(today=D, last_uploaded=D - timedelta(days=1)) == [],
            "already current -> nothing to do",
        )
        check(
            days_to_upload(today=D, last_uploaded=D) == [],
            "a watermark somehow AHEAD of yesterday does not go backwards",
        )
        gap = days_to_upload(today=D, last_uploaded=D - timedelta(days=4))
        check(
            gap == [D - timedelta(days=3), D - timedelta(days=2), D - timedelta(days=1)],
            f"a 3-day outage backfills every missed day, oldest first ({len(gap)} days)",
        )
        check(
            D not in days_to_upload(today=D, last_uploaded=D - timedelta(days=9)),
            "TODAY is never uploaded — it is still being written to",
        )
        long_gap = days_to_upload(today=D, last_uploaded=D - timedelta(days=400))
        check(
            len(long_gap) == MAX_BACKFILL_DAYS,
            f"a very long outage is capped at {MAX_BACKFILL_DAYS} days, not unbounded",
        )
        check(
            long_gap[-1] == D - timedelta(days=1),
            "and the cap keeps the NEWEST days, not the oldest ones",
        )

        # --- day boundaries -----------------------------------------------
        start, end = day_bounds(D)
        check(
            datetime.fromtimestamp(start).hour == 0
            and datetime.fromtimestamp(start).date() == D,
            "a day starts at local midnight",
        )
        check(
            datetime.fromtimestamp(end).date() == D + timedelta(days=1),
            "and ends at the next local midnight (half-open, so no event lands twice)",
        )
        # A DST day is 23 or 25 hours, and must still be exactly one folder.
        dst = date(2026, 3, 8)  # US spring-forward
        s2, e2 = day_bounds(dst)
        span_h = (e2 - s2) / 3600
        check(
            span_h in (23.0, 24.0, 25.0),
            f"a DST transition day spans {span_h:.0f}h and is still one folder",
        )

        # --- which files ---------------------------------------------------
        clips, snaps = tmp / "clips", tmp / "snapshots"
        clips.mkdir()
        snaps.mkdir()
        for eid in (10, 11, 12):
            (clips / f"{eid}.mp4").write_bytes(b"v")
        (snaps / "10.jpg").write_bytes(b"j")
        (snaps / "12.jpg").write_bytes(b"j")
        events = [{"id": 12}, {"id": 10}, {"id": 11}, {"id": 13}]  # 13 has no media

        by_root = media_for_day(events, clips_dir=clips, snapshots_dir=snaps)
        check(
            set(by_root) == {clips, snaps},
            "clips and snapshots are grouped under their OWN roots — they live "
            "on different volumes, so one rclone --files-from cannot span them",
        )
        check(
            [p.name for p in by_root[clips]] == ["10.mp4", "11.mp4", "12.mp4"],
            "clips come back in id order",
        )
        check(
            [p.name for p in by_root[snaps]] == ["10.jpg", "12.jpg"],
            "an event with no snapshot is simply absent, not an error",
        )
        check(
            all("13" not in p.name for paths in by_root.values() for p in paths),
            "an event whose media is already gone (retention, space rotation) is skipped",
        )
        no_snaps = media_for_day(events, clips_dir=clips, snapshots_dir=snaps,
                                 include_snapshots=False)
        check(set(no_snaps) == {clips}, "include_snapshots=False drops that root entirely")
        check(
            media_for_day([], clips_dir=clips, snapshots_dir=snaps) == {},
            "a day with no events maps to nothing at all",
        )

        # --- the files-from listing ----------------------------------------
        listing = tmp / "files.txt"
        n = write_files_from(by_root[clips], clips, listing)
        check(n == 3 and listing.read_text().split() == ["10.mp4", "11.mp4", "12.mp4"],
              "paths are written RELATIVE to the source root rclone is given")
        outside = write_files_from([snaps / "10.jpg"], clips, tmp / "bad.txt")
        check(
            outside == 0,
            "a path outside the root is dropped, not rewritten — rclone would "
            "resolve it against the wrong root and reach somewhere unintended",
        )
        empty = tmp / "empty.txt"
        check(write_files_from([], clips, empty) == 0 and empty.read_text() == "",
              "an empty list writes an empty file rather than a stray newline")

        # --- rclone argv ----------------------------------------------------
        args = build_rclone_copy_args(
            files_from=listing, source_root=clips, remote="dropbox:Vigilume/", day=D
        )
        check(args[1] == "copy", "the transfer is `copy`")
        check(
            "sync" not in args,
            "and NEVER `sync` — sync mirrors deletions, so it would erase the "
            "archive as local retention pruned the source",
        )
        check(
            f"dropbox:Vigilume/{day_folder(D)}" in args,
            "the destination is the remote's day folder, with no doubled slash",
        )
        check("--files-from" in args and str(listing) in args, "the listing is passed through")
        check("--no-traverse" in args, "--no-traverse: don't re-list a remote full of day folders")
        check("--bwlimit" not in args, "no --bwlimit when none is configured")
        limited = build_rclone_copy_args(
            files_from=listing, source_root=clips, remote="dropbox:V", day=D, bwlimit="2M"
        )
        check(
            limited[limited.index("--bwlimit") + 1] == "2M",
            "a configured bwlimit reaches rclone",
        )
        check(
            build_rclone_purge_args("dropbox:V/", D)[2] == f"dropbox:V/{day_folder(D)}",
            "purge targets one day folder, never the remote root",
        )
        check(
            "--dirs-only" in build_rclone_lsd_args("dropbox:V"),
            "the listing asks for directories only",
        )

        # --- what gets DELETED (no undo) ------------------------------------
        days = [day_folder(D - timedelta(days=i)) for i in range(10)]
        prune = days_to_prune(days, keep_days=7, today=D)
        check(
            len(prune) == 3 and prune[0] == D - timedelta(days=9),
            "keeps the newest keep_days folders and drops the rest, oldest first",
        )
        check(
            days_to_prune(days, keep_days=30, today=D) == [],
            "fewer folders than the window -> nothing deleted",
        )
        check(
            days_to_prune(days, keep_days=0, today=D) == [],
            "keep_days=0 means NEVER EXPIRE — it must not read as 'keep nothing'",
        )
        sparse = [day_folder(D - timedelta(days=i)) for i in (1, 2, 3, 40, 41)]
        check(
            days_to_prune(sparse, keep_days=4, today=D) == [D - timedelta(days=41)],
            "a gap in the days does not push a still-wanted day out — the window "
            "is by DATE, not by however many folders happen to exist",
        )
        # days[:3] is today, yesterday and the day before — three real days, so
        # keep_days=1 expires the older two. The junk entries alongside them must
        # neither be returned nor counted toward the window.
        junk = ["notes", "Camera Exports", "2026-13-45", ".rclone", ""]
        mixed = days_to_prune([*days[:3], *junk], keep_days=1, today=D)
        check(
            mixed == [D - timedelta(days=2), D - timedelta(days=1)],
            "entries that are not YYYY-MM-DD are IGNORED — the remote path may "
            "be shared, and deleting a stranger's folder has no undo",
        )
        check(
            mixed == days_to_prune(days[:3], keep_days=1, today=D),
            "and they do not shift the window either: the same real days expire "
            "with or without the junk beside them",
        )
        future_day = D + timedelta(days=5)
        future = days_to_prune([day_folder(future_day), *days[:2]], keep_days=1, today=D)
        check(
            future_day not in future,
            "a future-dated folder is never deleted — that is a clock problem, "
            "not an expiry",
        )
        check(
            future == [D - timedelta(days=1)],
            "and it does not occupy a slot in the window, which would have let a "
            "bad clock evict a real day early",
        )
        check(parse_day_folder("2026-08-25/") == D, "a trailing slash from lsf parses")
        check(parse_day_folder("2026-8-5") is None, "a loose date is NOT treated as a day folder")

        print()
        if _failures:
            print(f"{len(_failures)} of {_checks} CHECKS FAILED")
            for f in _failures:
                print(f"  - {f}")
            return 1
        print(f"ALL {_checks} CHECKS PASSED (cloud archive planning)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

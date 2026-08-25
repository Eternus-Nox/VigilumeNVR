"""Nightly cloud archive of event media — one folder per local day.

WHAT THIS IS FOR. Event clips and snapshots expire locally by `event_days`,
and the recordings disk can rotate footage away sooner than that when space is
tight. That is fine for 24/7 footage — it is bulk, and it is replaceable — but
an event clip is the evidence the system exists to produce, and it lives on one
disk in one building. This copies the finished day's events off the box.

SHAPE — a folder per local day, files inside:

    <remote>/2026-08-25/12345.mp4
    <remote>/2026-08-25/12345.jpg
    <remote>/2026-08-26/...

Chosen over a tar-per-day so the archive stays browsable: you can open the
Dropbox app on a phone, find the day, and play one clip. Rotation then deletes
whole day folders, which is one `purge` per expired day rather than tracking
individual files.

WHY rclone AND NOT A DROPBOX CLIENT. "Dropbox or something alike" is the actual
requirement, and rclone speaks Dropbox, Drive, S3, Backblaze and 70-odd others
through one interface with auth already solved. It is invoked as a subprocess
exactly as ffmpeg is elsewhere in this package, and for the same reason: the
hard, provider-specific part is someone else's maintained problem.

WHY A DAY BEHIND. A day is only uploaded once it is COMPLETE — the run at 03:00
uploads yesterday, never today. Uploading a day still being written would mean
re-uploading it repeatedly as events land, and would race clip extraction, which
finishes ~20 s after an event ends (longer with a raised clip_delay_s).

The planning here is deliberately pure: which days, which files, which argv,
which remote folders to drop. Running the process and reading the settings is
`ArchiveRunner`'s job, so the decisions stay testable without a binary, a
network, or a cloud account.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

# Folder-per-day naming. ISO so a plain lexical sort is chronological — that is
# what lets remote pruning work off `rclone lsf` output without parsing dates
# back out of it, and what makes the listing read correctly in Dropbox's own UI.
DAY_FMT = "%Y-%m-%d"
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# How many missed days a single run will catch up on. A box that was off for a
# week should backfill what is still on disk, but not turn one wake-up into an
# unbounded upload — and anything older than local `event_days` is gone anyway,
# so an unbounded window could not have found it.
MAX_BACKFILL_DAYS = 14

# rclone flags applied to every transfer. Modest concurrency on purpose: this
# runs on a box whose day job is recording video, and saturating the uplink is
# how an archive run breaks live view for whoever is away from home.
_RCLONE_BASE = (
    "--transfers", "4",
    "--checkers", "8",
    "--retries", "3",
    "--low-level-retries", "10",
    "--stats", "0",
)


def day_folder(day: date) -> str:
    return day.strftime(DAY_FMT)


def parse_day_folder(name: str) -> Optional[date]:
    """A remote listing entry back into a date, or None when it isn't one.

    Strict: anything that is not exactly YYYY-MM-DD is something else living in
    the same remote path, and pruning must never touch it.
    """
    name = name.strip().rstrip("/")
    if not _DAY_RE.match(name):
        return None
    try:
        return datetime.strptime(name, DAY_FMT).date()
    except ValueError:
        return None


def days_to_upload(
    *,
    today: date,
    last_uploaded: Optional[date],
    max_backfill: int = MAX_BACKFILL_DAYS,
) -> list[date]:
    """Complete days still needing an upload, oldest first.

    Never includes ``today`` — see the module docstring on why a day is only
    archived once it is finished.

    A first run (``last_uploaded is None``) uploads yesterday ONLY. Sweeping the
    whole retention window the first time an operator ticks the box would turn
    "enable archiving" into an unannounced multi-gigabyte upload; catching up
    from here is a decision they can make deliberately by backdating, not one
    the feature makes for them.
    """
    yesterday = today - timedelta(days=1)
    if last_uploaded is None:
        return [yesterday]
    if last_uploaded >= yesterday:
        return []  # already current
    start = max(last_uploaded + timedelta(days=1), yesterday - timedelta(days=max_backfill - 1))
    out: list[date] = []
    cursor = start
    while cursor <= yesterday:
        out.append(cursor)
        cursor += timedelta(days=1)
    return out


def day_bounds(day: date) -> tuple[float, float]:
    """[start, end) epoch seconds for a LOCAL calendar day.

    Local, not UTC, and via datetime().timestamp() so DST is resolved properly:
    a 23- or 25-hour day still maps to exactly one folder, with no event landing
    in two folders or in neither.
    """
    start = datetime(day.year, day.month, day.day).timestamp()
    end = datetime.combine(day + timedelta(days=1), datetime.min.time()).timestamp()
    return start, end


def media_for_day(
    events: Sequence[dict[str, Any]],
    *,
    clips_dir: Path,
    snapshots_dir: Path,
    include_snapshots: bool = True,
) -> dict[Path, list[Path]]:
    """``{source root: files under it}`` for these event rows, ids ascending.

    A MAPPING, not one flat list, because clips and snapshots live in different
    trees — media/native/clips and data/snapshots — which in a normal deployment
    are two separate volume mounts with no useful common ancestor. rclone's
    `--files-from` resolves every entry against ONE root, so each root needs its
    own invocation; they simply target the same remote day folder, which is what
    lands both kinds of file side by side in it.

    Roots with nothing to send are omitted, so callers iterate what exists.

    Missing files are normal rather than exceptional: an event may never have had
    a clip (recording disabled, or the recorder down for its window), and space
    rotation can take footage before this runs. A day with nothing at all is a
    no-op, not an error.
    """
    clips: list[Path] = []
    snaps: list[Path] = []
    for event in sorted(events, key=lambda e: int(e["id"])):
        event_id = int(event["id"])
        clip = clips_dir / f"{event_id}.mp4"
        if clip.is_file():
            clips.append(clip)
        if include_snapshots:
            snap = snapshots_dir / f"{event_id}.jpg"
            if snap.is_file():
                snaps.append(snap)
    out: dict[Path, list[Path]] = {}
    if clips:
        out[clips_dir] = clips
    if snaps:
        out[snapshots_dir] = snaps
    return out


def build_rclone_copy_args(
    *,
    files_from: Path,
    source_root: Path,
    remote: str,
    day: date,
    bwlimit: str = "",
    extra: Iterable[str] = (),
) -> list[str]:
    """argv copying one source tree's share of a day into ``<remote>/<day>/``.

    ONE call per source root (see media_for_day): both roots name the same
    remote day folder, so clips and snapshots land together in it.

    `copy`, never `sync`: sync makes the destination MATCH the source, so what
    it would delete from is the local tree that retention has since pruned —
    it would faithfully erase the archive it had just made. copy only ever
    adds, which is the whole point of an archive.

    `--files-from` rather than a directory argument because only this day's
    events are wanted, not the whole clips tree.
    """
    args = [
        "rclone", "copy",
        str(source_root),
        f"{remote.rstrip('/')}/{day_folder(day)}",
        "--files-from", str(files_from),
        # The listing is exact, so never walk the destination looking for more.
        # On a remote with many day folders this is the difference between one
        # API call and a full recursive listing every night.
        "--no-traverse",
        *_RCLONE_BASE,
    ]
    if bwlimit:
        args += ["--bwlimit", bwlimit]
    args += list(extra)
    return args


def build_rclone_lsd_args(remote: str) -> list[str]:
    """argv listing the remote's day folders, one bare name per line."""
    return ["rclone", "lsf", "--dirs-only", "--dir-slash=false", remote.rstrip("/")]


def build_rclone_purge_args(remote: str, day: date) -> list[str]:
    """argv removing one expired day folder and everything in it."""
    return ["rclone", "purge", f"{remote.rstrip('/')}/{day_folder(day)}", "--retries", "3"]


def days_to_prune(
    listing: Iterable[str],
    *,
    keep_days: int,
    today: Optional[date] = None,
) -> list[date]:
    """Remote day folders to delete, oldest first.

    Keeps the ``keep_days`` most recent by DATE, not by how many entries happen
    to be there — a gap (the box was off, or a day had no events) must not push
    a still-wanted day out of the window.

    Entries that are not YYYY-MM-DD are ignored entirely: the remote path may be
    shared with other things, and an archive rotation that deletes a stranger's
    folder is a much worse bug than one that keeps too much. Days in the FUTURE
    are likewise never pruned — that is a clock problem, not an expiry.
    """
    if keep_days <= 0:
        return []
    days = sorted({d for d in (parse_day_folder(n) for n in listing) if d is not None})
    if today is not None:
        days = [d for d in days if d <= today]
    if len(days) <= keep_days:
        return []
    return days[: len(days) - keep_days]


def write_files_from(paths: Sequence[Path], source_root: Path, dest: Path) -> int:
    """Write an rclone `--files-from` list (paths relative to source_root).

    Returns how many lines were written. Paths outside ``source_root`` are
    skipped rather than silently rewritten — rclone would resolve them against
    the wrong root and either miss the file or reach somewhere unintended.
    """
    lines: list[str] = []
    for path in paths:
        try:
            lines.append(str(path.resolve().relative_to(source_root.resolve())))
        except ValueError:
            log.warning("archive: %s is outside %s — not uploading it", path, source_root)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)

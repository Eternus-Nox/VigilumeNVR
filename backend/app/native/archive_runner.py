"""Drives the nightly cloud archive: settings in, rclone subprocesses out.

The decisions — which days, which files, which remote folders expire — all live
in ``archive.py`` and are pure. This module is the part that cannot be tested
without a binary and a cloud account: reading settings, spawning rclone, and
remembering how far it has got.

PROGRESS IS PERSISTED, not inferred. The alternative — asking the remote what it
already holds and uploading the gap — sounds tidier and is worse: it makes every
run depend on a listing that costs API calls, it re-uploads a whole day if the
remote is briefly unreachable, and it silently re-uploads everything the first
time someone renames the remote path. A watermark in the KV store is one row and
says exactly what happened.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from . import archive

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..db import Database
    from ..settings_store import SettingsStore

log = logging.getLogger(__name__)

_STATE_KEY = "archive_state"
# rclone is given a generous ceiling rather than none: a wedged transfer must not
# hold the nightly slot forever, but a real day of clips on a slow uplink can
# legitimately take a long time (see the sizing note in the settings docstring).
_RUN_TIMEOUT_S = 6 * 3600
_LIST_TIMEOUT_S = 120


class ArchiveRunner:
    """One nightly pass: upload the completed days, then expire old ones."""

    def __init__(self, config: "Config", db: "Database", settings: "SettingsStore") -> None:
        self._config = config
        self._db = db
        self._settings = settings
        # Surfaced to the UI/logs so a silent failure is visible somewhere other
        # than a log line that scrolled away.
        self.last_result: dict[str, Any] = {}

    # ---------- settings ----------

    def _cfg(self) -> dict[str, Any]:
        return self._settings.current.get("archive") or {}

    def enabled(self) -> bool:
        cfg = self._cfg()
        return bool(cfg.get("enabled")) and bool(str(cfg.get("remote") or "").strip())

    # ---------- state ----------

    async def _state(self) -> dict[str, Any]:
        return (await self._db.get_setting(_STATE_KEY)) or {}

    async def _last_uploaded(self) -> Optional[date]:
        raw = (await self._state()).get("last_uploaded_day")
        return archive.parse_day_folder(str(raw)) if raw else None

    async def _remember(self, day: date, **extra: Any) -> None:
        state = await self._state()
        state["last_uploaded_day"] = archive.day_folder(day)
        state["last_run_at"] = datetime.now().isoformat(timespec="seconds")
        state.update(extra)
        await self._db.set_setting(_STATE_KEY, state)

    # ---------- rclone ----------

    async def _run(self, args: list[str], *, timeout: float) -> tuple[int, str]:
        """Run rclone, returning (returncode, captured stderr tail).

        Never raises for a non-zero exit — an archive that fails is a logged
        problem, not something that should take the scheduler down with it.
        """
        log.debug("archive: %s", " ".join(args))
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return (127, "rclone is not installed in this image")
        except OSError as exc:
            return (1, f"could not start rclone: {exc}")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (1, f"rclone timed out after {timeout:.0f}s")
        if proc.returncode != 0:
            tail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
            return (proc.returncode or 1, " | ".join(tail[-3:]) or "rclone failed")
        return (0, (stdout or b"").decode("utf-8", "replace"))

    # ---------- the pass ----------

    async def run_once(self, *, today: Optional[date] = None) -> dict[str, Any]:
        """Upload the completed days, then prune the remote. Never raises."""
        result: dict[str, Any] = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "uploaded_days": [], "files": 0, "pruned_days": [], "errors": [],
        }
        if not self.enabled():
            return result
        cfg = self._cfg()
        remote = str(cfg.get("remote") or "").strip()
        keep_days = max(0, int(cfg.get("keep_days") or 0))
        include_snapshots = bool(cfg.get("include_snapshots", True))
        bwlimit = str(cfg.get("bwlimit") or "").strip()
        today = today or date.today()

        try:
            pending = archive.days_to_upload(
                today=today, last_uploaded=await self._last_uploaded()
            )
            for day in pending:
                sent, error = await self._upload_day(
                    day, remote=remote, include_snapshots=include_snapshots, bwlimit=bwlimit
                )
                if error:
                    # STOP at the first failed day rather than skipping past it:
                    # the watermark only moves on success, so continuing would
                    # either lose the day for good or force a re-upload of every
                    # day after it once the watermark caught up.
                    result["errors"].append(f"{archive.day_folder(day)}: {error}")
                    break
                result["uploaded_days"].append(archive.day_folder(day))
                result["files"] += sent
                await self._remember(day)

            if keep_days > 0:
                pruned, errors = await self._prune_remote(remote, keep_days, today)
                result["pruned_days"] = [archive.day_folder(d) for d in pruned]
                result["errors"].extend(errors)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — never take the scheduler down
            log.exception("archive: pass failed")
            result["errors"].append(repr(exc))

        self.last_result = result
        if result["uploaded_days"] or result["pruned_days"] or result["errors"]:
            log.info(
                "archive: uploaded %s (%d files), pruned %s%s",
                result["uploaded_days"] or "nothing", result["files"],
                result["pruned_days"] or "nothing",
                f", errors: {result['errors']}" if result["errors"] else "",
            )
        return result

    async def _upload_day(
        self, day: date, *, remote: str, include_snapshots: bool, bwlimit: str
    ) -> tuple[int, Optional[str]]:
        start, end = archive.day_bounds(day)
        # `before` is INCLUSIVE in list_events (start_time <= ?), but day_bounds
        # gives a half-open [start, end). Left as-is, an event starting exactly
        # at midnight would be claimed by both days and uploaded twice, into two
        # folders — so the exclusive end is re-applied here.
        events, _total = await self._db.list_events(after=start, before=end, limit=100_000)
        events = [e for e in events if float(e["start_time"]) < end]
        by_root = archive.media_for_day(
            events,
            clips_dir=self._config.clips_dir,
            snapshots_dir=self._config.snapshots_dir,
            include_snapshots=include_snapshots,
        )
        if not by_root:
            # A genuinely empty day still advances the watermark — otherwise a
            # quiet day would be retried forever and block every day after it.
            log.info("archive: %s had no event media to upload", archive.day_folder(day))
            return (0, None)

        sent = 0
        with tempfile.TemporaryDirectory(prefix="vigilume-archive-") as tmp:
            for index, (root, paths) in enumerate(by_root.items()):
                listing = Path(tmp) / f"files-{index}.txt"
                count = await asyncio.to_thread(
                    archive.write_files_from, paths, root, listing
                )
                if count == 0:
                    continue
                code, detail = await self._run(
                    archive.build_rclone_copy_args(
                        files_from=listing, source_root=root,
                        remote=remote, day=day, bwlimit=bwlimit,
                    ),
                    timeout=_RUN_TIMEOUT_S,
                )
                if code != 0:
                    return (sent, detail)
                sent += count
        return (sent, None)

    async def _prune_remote(
        self, remote: str, keep_days: int, today: date
    ) -> tuple[list[date], list[str]]:
        code, output = await self._run(
            archive.build_rclone_lsd_args(remote), timeout=_LIST_TIMEOUT_S
        )
        if code != 0:
            return ([], [f"listing {remote}: {output}"])
        expired = archive.days_to_prune(
            output.splitlines(), keep_days=keep_days, today=today
        )
        pruned: list[date] = []
        errors: list[str] = []
        for day in expired:
            code, detail = await self._run(
                archive.build_rclone_purge_args(remote, day), timeout=_LIST_TIMEOUT_S
            )
            if code == 0:
                pruned.append(day)
            else:
                errors.append(f"purge {archive.day_folder(day)}: {detail}")
        return (pruned, errors)


async def archive_loop(runner: ArchiveRunner) -> None:
    """Fire ``run_once`` in the minute after the configured local hour.

    Mirrors _auto_restart_loop: settings are re-read every tick so enabling or
    moving the hour takes effect without a restart, and a per-day latch stops
    the same slot firing twice.
    """
    last_fired_day: Optional[str] = None
    while True:
        try:
            if runner.enabled():
                hour = int((runner._cfg().get("hour") or 3))
                now = datetime.now()
                today = now.strftime(archive.DAY_FMT)
                due = now.replace(hour=max(0, min(23, hour)), minute=0, second=0, microsecond=0)
                if last_fired_day != today and 0 <= (now - due).total_seconds() < 60:
                    last_fired_day = today
                    await runner.run_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a scheduling bug must never kill the app
            log.exception("archive scheduler tick failed")
        await asyncio.sleep(30)

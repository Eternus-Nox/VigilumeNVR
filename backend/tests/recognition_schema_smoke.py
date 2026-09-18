#!/usr/bin/env python3
"""The recognition tables must EXIST — on a fresh box and on an upgraded one.

THE BUG THIS EXISTS TO PREVENT
==============================
Recognition shipped as schema v22/v23 whose migration blocks created no tables.
They were written believing `_SCHEMA` — which holds every CREATE TABLE IF NOT
EXISTS — ran on every boot. It does not: `_migrate` runs it only for a brand-new
database (``if version < 1``), and an existing one takes the incremental-ALTER
branch instead.

So an UPGRADED box got the version stamp and none of the five tables. And
because the stamp was written, every later boot skipped the migration, so the
box could not heal itself. What the operator saw:

    GET /api/events -> 500   (sqlite3.OperationalError: no such table:
                              event_recognitions)
    ERROR app.native.facepass: candidate purge failed
                             (no such table: recognition_candidates)

— the entire events list and timeline dead, from a feature that was off.

The fix runs `_RECOGNITION_SCHEMA` unconditionally, before the version branch.
That is what makes it repair an already-stamped database rather than only
helping boxes that had not upgraded yet.

WHAT IS PINNED HERE
-------------------
1. A fresh database has every recognition table.
2. A PRE-recognition database (v21) gains them on upgrade.
3. A database STAMPED v23 WITH THE TABLES MISSING — the exact broken state that
   shipped — is REPAIRED on the next boot. This is the regression that matters:
   a fix that only handled case 2 would leave every already-upgraded box dead.
4. Existing rows survive all of it.

Offline-runnable; sqlite only, no network and no models.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

from app.db import SCHEMA_VERSION, Database  # noqa: E402

_failures: list[str] = []
_checks = 0

#: Every table recognition needs. A name added to _RECOGNITION_SCHEMA and not
#: here is a table nothing verifies the existence of.
RECOGNITION_TABLES = {
    "profiles",
    "profile_samples",
    "recognition_candidates",
    "event_recognitions",
    "recognition_heatmap",
}


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


async def tables_in(db: Database) -> set[str]:
    cur = await db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in await cur.fetchall()}


def seed_legacy_db(path: Path, *, version: int) -> None:
    """A pre-recognition database with real content, stamped at `version`.

    Deliberately NOT built from _SCHEMA: it stands in for a box that has been
    running for months, so it carries only what such a box would have.
    """
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE cameras ("
        " name TEXT PRIMARY KEY, friendly_name TEXT NOT NULL, model TEXT NOT NULL,"
        " ip TEXT NOT NULL, username TEXT NOT NULL, password TEXT NOT NULL,"
        " detect_objects TEXT NOT NULL DEFAULT '[]', detect_width INTEGER,"
        " detect_height INTEGER, created_at REAL NOT NULL)"
    )
    con.execute(
        "INSERT INTO cameras (name, friendly_name, model, ip, username, password,"
        " detect_objects, detect_width, detect_height, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("front", "Front", "AD410", "127.0.0.1", "u", "p", '["person"]', 640, 480, 1.0),
    )
    con.execute(f"PRAGMA user_version = {version}")
    con.commit()
    con.close()


async def fresh_db_checks() -> None:
    print("\na FRESH database gets every recognition table")
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "nvr.db")
        await db.connect()
        try:
            have = await tables_in(db)
            missing = RECOGNITION_TABLES - have
            check(not missing, f"all recognition tables created (missing: {sorted(missing)})")
            cur = await db.conn.execute("PRAGMA user_version")
            check((await cur.fetchone())[0] == SCHEMA_VERSION,
                  "stamped at the current schema version")
        finally:
            await db.close()


async def upgrade_checks() -> None:
    print("\nan EXISTING pre-recognition database (v21) gains them on upgrade")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nvr.db"
        seed_legacy_db(path, version=21)
        db = Database(path)
        await db.connect()
        try:
            have = await tables_in(db)
            missing = RECOGNITION_TABLES - have
            check(not missing,
                  f"v21 -> current creates every recognition table (missing: {sorted(missing)})")
            # Raw SELECT rather than list_cameras(): this fixture is a
            # MINIMAL v21 table, and list_cameras orders by `position`, a
            # column from v3 that a hand-built fixture has no reason to
            # carry. What is under test is the recognition schema, not a
            # faithful reconstruction of every historical column.
            cur = await db.conn.execute("SELECT name FROM cameras")
            cams = {r[0] for r in await cur.fetchall()}
            check(cams == {"front"}, "the existing camera row survived the migration")
            cur = await db.conn.execute("PRAGMA table_info(cameras)")
            cols = {r[1] for r in await cur.fetchall()}
            check({"face_zones", "plate_zones"} <= cols,
                  "and the v22 recognition-ROI columns were added")
        finally:
            await db.close()


async def repair_checks() -> None:
    """The one that matters: a box ALREADY poisoned by the broken migration."""
    print("\na database STAMPED v23 with NO recognition tables is REPAIRED")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nvr.db"
        # Exactly the shipped-broken state: version says fully migrated, the
        # tables were never created, and the ROI columns DID land (the v22
        # block's ALTERs ran; only its table creation was missing).
        seed_legacy_db(path, version=21)
        con = sqlite3.connect(str(path))
        for col in ("face_zones", "plate_zones"):
            con.execute(f"ALTER TABLE cameras ADD COLUMN {col} TEXT NOT NULL DEFAULT '[]'")
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        con.commit()
        pre = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        check(not (RECOGNITION_TABLES & pre),
              "fixture really is missing the tables before we start")

        db = Database(path)
        await db.connect()
        try:
            have = await tables_in(db)
            missing = RECOGNITION_TABLES - have
            check(not missing,
                  "the stamped-but-empty database is repaired on the next boot "
                  f"(missing: {sorted(missing)}) — a version-gated fix would "
                  "have left every already-upgraded box dead")

            # The two queries that were actually failing in production.
            await db.conn.execute("SELECT 1 FROM recognition_candidates LIMIT 1")
            by_fid = await db.recognitions_for(["native.1"])
            check(by_fid == {}, "recognitions_for runs and returns nothing, rather than raising")

            cur = await db.conn.execute("SELECT name FROM cameras")
            cams = {r[0] for r in await cur.fetchall()}
            check(cams == {"front"}, "the camera row survived the repair")
        finally:
            await db.close()


async def idempotence_checks() -> None:
    print("\nrunning it repeatedly is a no-op")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nvr.db"
        for _ in range(3):
            db = Database(path)
            await db.connect()
            await db.close()
        db = Database(path)
        await db.connect()
        try:
            missing = RECOGNITION_TABLES - await tables_in(db)
            check(not missing, "four consecutive boots leave the schema intact")
        finally:
            await db.close()


async def main_async() -> None:
    await fresh_db_checks()
    await upgrade_checks()
    await repair_checks()
    await idempotence_checks()


def main() -> int:
    asyncio.run(main_async())
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (recognition schema creation + repair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

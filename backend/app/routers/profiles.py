"""Recognition profiles: people and vehicles, their enrolled samples, and the
rolling candidate list they are enrolled FROM.

ADMIN-ONLY, ALL OF IT — including the reads.
=============================================
Every other list in this API is readable by a viewer (docs/CONTRACTS.md RBAC
addendum: a viewer watches, an admin configures). This router is the exception,
and the reason is not configuration but CONTENT: the profile list is a named
register of who comes to this address and which cars they drive, and the
candidate list is a rolling gallery of every stranger's face the cameras have
caught. Handing that to a viewer account — the account you give a house-sitter,
a neighbour, an adult child — is a different act from letting them watch a live
camera, and it is not one the viewer role was ever agreed to cover.

So `require_admin` gates the whole router, and the imagery uses
`require_media_admin` rather than `require_media_auth`: ids here are sequential,
so media-scope alone would let any authenticated token walk the entire
biometric store one integer at a time.

ENROLLMENT IS APPEND-ONLY AND REVERSIBLE
----------------------------------------
There is no training step. Enrolling copies a candidate's embedding into
`profile_samples`; un-enrolling deletes that row. Nothing on disk is fitted, so
"undo" is a DELETE and there is no residue in a trained artifact. This is what
makes the app's "pick the best image" flow safe to get wrong — see
native/recognition.py for why the gallery is nearest-neighbour rather than a
classifier.

WHAT THIS ROUTER DOES NOT DO
----------------------------
It does not run models. Embeddings arrive already computed (from the engine's
recognition pass, stored on the candidate row); this router moves rows and
files around. Keeping inference out of the request path is what stops a slow
model from blocking the API event loop.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from ..auth import require_admin, require_media_admin
from ..native import accel
from ..native.timing import TIMINGS
from ..native.heatmap import COLS, ROWS, suggest_zone
from ..native.recognition import (
    FACE_THRESHOLD, MIN_MARGIN, SUGGEST_FACE_COSINE, SUGGEST_LIMIT,
    PLATE_MAX_DISTANCE, cosine, from_blob, normalize, normalize_plate,
    plate_distance,
)

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/recognition",
    tags=["recognition"],
    dependencies=[Depends(require_admin)],
)

#: The two kinds of identity a profile can carry. Deliberately closed: a third
#: kind is a schema change plus a matcher, not a new string.
KINDS = ("person", "vehicle")

MAX_NAME = 64
MAX_NOTES = 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProfileCreate(BaseModel):
    kind: str
    name: str = Field(min_length=1, max_length=MAX_NAME)
    notes: str = Field(default="", max_length=MAX_NOTES)
    enabled: bool = True
    threshold: Optional[float] = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        if v not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be blank")
        return v

    @field_validator("threshold")
    @classmethod
    def _threshold(cls, v: Optional[float]) -> Optional[float]:
        # A threshold outside 0..1 is not a stricter setting, it is a profile
        # that can never match (or always match). Rejected rather than clamped
        # so the mistake is visible where it was made.
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("threshold must be between 0 and 1")
        return v


class ProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=MAX_NAME)
    notes: Optional[str] = Field(default=None, max_length=MAX_NOTES)
    enabled: Optional[bool] = None
    threshold: Optional[float] = None

    _name = field_validator("name")(ProfileCreate._name.__func__)  # type: ignore[attr-defined]
    _threshold = field_validator("threshold")(ProfileCreate._threshold.__func__)  # type: ignore[attr-defined]


class PlateSample(BaseModel):
    """Enroll a vehicle by typing its plate — no sighting required."""

    plate: str = Field(min_length=1, max_length=16)

    @field_validator("plate")
    @classmethod
    def _plate(cls, v: str) -> str:
        cleaned = normalize_plate(v)
        if not cleaned:
            raise ValueError("plate must contain letters or digits")
        return cleaned


class EnrollRequest(BaseModel):
    """Enroll one or more CANDIDATES into a profile.

    A list rather than a single id because the app's flow is "show me the five
    best shots of this face, I'll tick the three that are actually them" —
    which is one user action and should be one request, not three that can
    half-fail.
    """

    candidate_ids: list[int] = Field(min_length=1, max_length=25)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _profile_out(row: Any, sample_count: int, usable_count: int) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "notes": row["notes"],
        "enabled": bool(row["enabled"]),
        "threshold": row["threshold"],
        "sample_count": sample_count,
        # Samples embedded by the CURRENTLY ACTIVE model. When this is 0 but
        # sample_count is not, every sample was embedded by a different model
        # and this profile has silently stopped matching — the app shows
        # "re-enroll" on exactly that condition rather than leaving someone to
        # wonder why recognition stopped.
        "usable_sample_count": usable_count,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _sample_out(row: Any) -> dict[str, Any]:
    has_image = bool(row["image_path"])
    return {
        "id": row["id"],
        "profile_id": row["profile_id"],
        "plate": row["plate"],
        "model_key": row["model_key"],
        "quality": row["quality"],
        "source_fid": row["source_fid"],
        "created_at": row["created_at"],
        "has_image": has_image,
        "image_url": (
            f"/api/recognition/samples/{row['id']}/image.jpg" if has_image else None
        ),
    }


def _candidate_out(row: Any) -> dict[str, Any]:
    has_image = bool(row["image_path"])
    return {
        "id": row["id"],
        "kind": row["kind"],
        "camera": row["camera"],
        "event_fid": row["event_fid"],
        "plate": row["plate"],
        "quality": row["quality"],
        "best_score": row["best_score"],
        "best_profile_id": row["best_profile_id"],
        "created_at": row["created_at"],
        "has_image": has_image,
        "image_url": (
            f"/api/recognition/candidates/{row['id']}/image.jpg" if has_image else None
        ),
    }


def _active_model_key(request: Request) -> str:
    """Which embedding model the gallery is currently built against.

    Read from engine state when recognition is running, otherwise "" — which
    correctly reports every enrolled sample as unusable, because with no model
    loaded nothing can be matched anyway.
    """
    engine = getattr(request.app.state, "engine", None)
    return getattr(engine, "recognition_model_key", "") or ""


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@router.get("/profiles")
async def list_profiles(
    request: Request,
    kind: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    db = request.app.state.db
    active = _active_model_key(request)
    sql = "SELECT * FROM profiles"
    params: list[Any] = []
    if kind:
        if kind not in KINDS:
            raise HTTPException(status_code=400, detail=f"kind must be one of {KINDS}")
        sql += " WHERE kind = ?"
        params.append(kind)
    sql += " ORDER BY kind, name COLLATE NOCASE"
    rows = await (await db.conn.execute(sql, params)).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        cur = await db.conn.execute(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN plate != '' OR model_key = ? THEN 1 ELSE 0 END) AS usable "
            "FROM profile_samples WHERE profile_id = ?",
            (active, r["id"]),
        )
        counts = await cur.fetchone()
        out.append(_profile_out(r, counts["n"] or 0, counts["usable"] or 0))
    return out


@router.post("/profiles", status_code=201)
async def create_profile(body: ProfileCreate, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    now = time.time()
    try:
        cur = await db.conn.execute(
            "INSERT INTO profiles (kind, name, notes, enabled, threshold, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (body.kind, body.name, body.notes, int(body.enabled), body.threshold, now, now),
        )
        await db.conn.commit()
    except Exception as exc:  # aiosqlite surfaces the UNIQUE(kind, name) here
        if "UNIQUE" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"A {body.kind} profile named {body.name!r} already exists",
            ) from exc
        raise
    await _reload_gallery(request)
    row = await (
        await db.conn.execute("SELECT * FROM profiles WHERE id = ?", (cur.lastrowid,))
    ).fetchone()
    return _profile_out(row, 0, 0)


@router.get("/profiles/{profile_id}")
async def get_profile(profile_id: int, request: Request) -> dict[str, Any]:
    db = request.app.state.db
    row = await _require_profile(db, profile_id)
    active = _active_model_key(request)
    samples = await (
        await db.conn.execute(
            "SELECT * FROM profile_samples WHERE profile_id = ? ORDER BY quality DESC, id",
            (profile_id,),
        )
    ).fetchall()
    usable = sum(1 for s in samples if s["plate"] or s["model_key"] == active)
    return {
        **_profile_out(row, len(samples), usable),
        "samples": [_sample_out(s) for s in samples],
    }


@router.put("/profiles/{profile_id}")
async def update_profile(
    profile_id: int, body: ProfileUpdate, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    await _require_profile(db, profile_id)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to update")
    sets, params = [], []
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        params.append(int(v) if k == "enabled" else v)
    sets.append("updated_at = ?")
    params.extend([time.time(), profile_id])
    try:
        await db.conn.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE id = ?", params)
        await db.conn.commit()
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(status_code=409, detail="That name is already taken") from exc
        raise
    await _reload_gallery(request)
    return await get_profile(profile_id, request)


@router.delete("/profiles/{profile_id}", status_code=204)
async def delete_profile(profile_id: int, request: Request) -> Response:
    """Delete a profile, its samples, and their reference images.

    The images go too. A profile delete that left a directory of someone's
    face crops on disk would make "delete this person" a lie, and this is the
    one store in the product where that distinction is not cosmetic.
    """
    db = request.app.state.db
    await _require_profile(db, profile_id)
    rows = await (
        await db.conn.execute(
            "SELECT image_path FROM profile_samples WHERE profile_id = ?", (profile_id,)
        )
    ).fetchall()
    # The FK is declared ON DELETE CASCADE, but SQLite only honours that with
    # `PRAGMA foreign_keys = ON`, which is per-connection and off by default.
    # Deleting explicitly means this is correct either way.
    await db.conn.execute("DELETE FROM profile_samples WHERE profile_id = ?", (profile_id,))
    await db.conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    await db.conn.commit()
    for r in rows:
        _unlink(request.app.state.config.profile_images_dir, r["image_path"])
    await _reload_gallery(request)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@router.post("/profiles/{profile_id}/plate", status_code=201)
async def add_plate_sample(
    profile_id: int, body: PlateSample, request: Request
) -> dict[str, Any]:
    db = request.app.state.db
    row = await _require_profile(db, profile_id)
    if row["kind"] != "vehicle":
        raise HTTPException(status_code=400, detail="Only a vehicle profile carries a plate")
    cur = await db.conn.execute(
        "INSERT INTO profile_samples (profile_id, embedding, dim, plate, model_key, "
        "image_path, quality, source_fid, created_at) "
        "VALUES (?, NULL, 0, ?, '', '', 1.0, '', ?)",
        (profile_id, body.plate, time.time()),
    )
    await db.conn.execute(
        "UPDATE profiles SET updated_at = ? WHERE id = ?", (time.time(), profile_id)
    )
    # Any unread-plate sighting that IS this plate is now answered. Removing
    # them is what stops the review list filling with the same car you just
    # named. See _absorb_matching_plates on why this is safe to do without
    # asking, where the face equivalent is not.
    absorbed = await _absorb_matching_plates(
        db, request.app.state.config.candidate_crops_dir, profile_id, body.plate
    )
    await db.conn.commit()
    await _reload_gallery(request)
    sample = await (
        await db.conn.execute("SELECT * FROM profile_samples WHERE id = ?", (cur.lastrowid,))
    ).fetchone()
    return {**_sample_out(sample), "absorbed_candidates": absorbed}


@router.post("/profiles/{profile_id}/enroll", status_code=201)
async def enroll_candidates(
    profile_id: int, body: EnrollRequest, request: Request
) -> dict[str, Any]:
    """Move candidates into a profile as enrolled samples.

    The candidate ROW is consumed (it is no longer unknown) and its crop is
    MOVED from the rolling directory into the durable one — so the reference
    image cannot later be swept up by the candidate purge, which is the bug
    that would quietly empty a gallery weeks after enrollment.
    """
    db = request.app.state.db
    profile = await _require_profile(db, profile_id)
    cfg = request.app.state.config

    placeholders = ",".join("?" * len(body.candidate_ids))
    rows = await (
        await db.conn.execute(
            f"SELECT * FROM recognition_candidates WHERE id IN ({placeholders})",
            body.candidate_ids,
        )
    ).fetchall()
    found = {r["id"] for r in rows}
    missing = [i for i in body.candidate_ids if i not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"No such candidate(s): {missing}")

    wrong_kind = [
        r["id"] for r in rows
        if (r["kind"] == "face") != (profile["kind"] == "person")
    ]
    if wrong_kind:
        raise HTTPException(
            status_code=400,
            detail=f"Candidate(s) {wrong_kind} are the wrong kind for a "
                   f"{profile['kind']} profile",
        )

    cfg.profile_images_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    created: list[int] = []
    for r in rows:
        cur = await db.conn.execute(
            "INSERT INTO profile_samples (profile_id, embedding, dim, plate, model_key, "
            "image_path, quality, source_fid, created_at) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)",
            (
                profile_id, r["embedding"], r["dim"], r["plate"],
                r["model_key"], r["quality"], r["event_fid"], now,
            ),
        )
        sample_id = cur.lastrowid
        created.append(sample_id)
        if r["image_path"]:
            dest = f"{sample_id}.jpg"
            src = cfg.candidate_crops_dir / r["image_path"]
            try:
                if src.exists():
                    shutil.move(str(src), str(cfg.profile_images_dir / dest))
                    await db.conn.execute(
                        "UPDATE profile_samples SET image_path = ? WHERE id = ?",
                        (dest, sample_id),
                    )
            except OSError:
                # An unmovable crop costs the reference IMAGE, never the
                # embedding — which is the part that actually does the
                # matching. Logged, not raised.
                log.warning("could not move candidate crop %s for sample %d",
                            r["image_path"], sample_id)

    await db.conn.execute(
        f"DELETE FROM recognition_candidates WHERE id IN ({placeholders})",
        body.candidate_ids,
    )
    await db.conn.execute(
        "UPDATE profiles SET updated_at = ? WHERE id = ?", (now, profile_id)
    )
    await db.conn.commit()
    await _reload_gallery(request)
    # "You said this is Adam — here are the others that look like him."
    # Returned with the enroll so the operator is offered them at the one
    # moment they are thinking about this person, rather than having to know
    # to go looking. Computed AFTER the gallery reload so the suggestion is
    # scored against the profile as it now stands, including what was just
    # added — which is the whole point: the shot just enrolled is usually the
    # one that finds the rest.
    similar = await _similar_candidates(
        db, profile, active_model=_active_model_key(request)
    )
    return {
        "enrolled": len(created),
        "sample_ids": created,
        # A SUGGESTION. Enrolling any of these is another explicit call.
        "similar": similar,
        "similar_threshold": SUGGEST_FACE_COSINE,
    }



async def _absorb_matching_plates(
    db: Any, crops_dir: Path, profile_id: int, plate: str
) -> int:
    """Delete unread-plate candidates that ARE this plate. Returns how many.

    AUTOMATIC, unlike the face suggestion, and the difference is not a
    preference — it is that the two questions are not equally decidable.

    "Is this the same plate?" has an exact answer: after normalization and
    glyph folding, two reads of 7ABC123 are the same vehicle by definition.
    "Is this the same face?" is a score, and a score can be wrong in a way that
    silently attaches a stranger to someone's name.

    So a candidate within PLATE_MAX_DISTANCE of a plate the operator has just
    typed carries no new information — it is a duplicate of a fact already
    established — and leaving it in the unread list is noise that buries the
    plates still worth reviewing. The rows are REMOVED rather than enrolled:
    the plate string is already a sample, and a second identical string adds
    nothing to match against.
    """
    try:
        rows = await (
            await db.conn.execute(
                "SELECT id, plate, image_path FROM recognition_candidates "
                "WHERE kind = 'plate' AND plate != ''"
            )
        ).fetchall()
    except Exception:
        log.exception("could not scan plate candidates for %s", plate)
        return 0
    doomed = [
        r for r in rows if plate_distance(r["plate"], plate) <= PLATE_MAX_DISTANCE
    ]
    if not doomed:
        return 0
    for r in doomed:
        _unlink(crops_dir, r["image_path"])
    placeholders = ",".join("?" * len(doomed))
    await db.conn.execute(
        f"DELETE FROM recognition_candidates WHERE id IN ({placeholders})",
        [r["id"] for r in doomed],
    )
    log.info(
        "absorbed %d unread-plate candidate(s) matching %s into profile %d",
        len(doomed), plate, profile_id,
    )
    return len(doomed)


async def _similar_candidates(
    db: Any, profile: Any, *, active_model: str, limit: int = SUGGEST_LIMIT
) -> list[dict[str, Any]]:
    """Unmatched crops that look like THIS profile, best first.

    "You said this crop is Adam — here are the others that look like him."
    Without it, enrolling someone means finding their every other sighting by
    eye in a list sorted by legibility, which nobody does, so profiles stay
    thin and thin profiles are the ones that miss.

    Scored MAX-OVER-SAMPLES, the same rule the matcher uses, so a suggestion
    means exactly "this would now be recognized as Adam" rather than a second,
    subtly different notion of similar that could disagree with the matcher and
    leave an operator unable to tell which was lying.

    Returns [] for a vehicle, for a profile with no usable samples, and when
    recognition has no model loaded — none of which is an error, and all of
    which would otherwise be an exception on an ordinary enrollment.
    """
    if profile["kind"] != "person" or not active_model:
        return []
    try:
        sample_rows = await (
            await db.conn.execute(
                "SELECT embedding, dim FROM profile_samples "
                "WHERE profile_id = ? AND model_key = ? AND embedding IS NOT NULL",
                (profile["id"], active_model),
            )
        ).fetchall()
        # Only candidates from the SAME embedding space. Vectors from two
        # models are incomparable and score in an entirely plausible range, so
        # mixing them would produce confident nonsense rather than an error.
        cand_rows = await (
            await db.conn.execute(
                "SELECT * FROM recognition_candidates "
                "WHERE kind = 'face' AND model_key = ? AND embedding IS NOT NULL",
                (active_model,),
            )
        ).fetchall()
    except Exception:
        log.exception("similar-candidate lookup failed for profile %s", profile["id"])
        return []

    vectors = [
        normalize(from_blob(r["embedding"], r["dim"]))
        for r in sample_rows
        if r["embedding"]
    ]
    vectors = [v for v in vectors if v is not None and v.size]
    if not vectors:
        return []

    scored: list[tuple[float, Any]] = []
    for row in cand_rows:
        vec = normalize(from_blob(row["embedding"], row["dim"]))
        if vec is None or not vec.size:
            continue
        score = max((cosine(vec, v) for v in vectors), default=0.0)
        if score >= SUGGEST_FACE_COSINE:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {**_candidate_out(row), "similarity": round(float(score), 4)}
        for score, row in scored[:limit]
    ]


@router.get("/profiles/{profile_id}/similar")
async def similar_to_profile(
    profile_id: int, request: Request, limit: int = Query(default=SUGGEST_LIMIT, ge=1, le=100)
) -> dict[str, Any]:
    """Unmatched crops that look like this profile. A SUGGESTION, never applied.

    Deliberately a read. Enrolling these is the ordinary enroll call, which
    means the operator's confirmation is a real step and not a dialog they can
    dismiss without noticing what it did.
    """
    db = request.app.state.db
    profile = await _require_profile(db, profile_id)
    items = await _similar_candidates(
        db, profile, active_model=_active_model_key(request), limit=limit
    )
    return {
        "profile_id": profile_id,
        "threshold": SUGGEST_FACE_COSINE,
        "candidates": items,
    }


@router.delete("/samples/{sample_id}", status_code=204)
async def delete_sample(sample_id: int, request: Request) -> Response:
    db = request.app.state.db
    row = await (
        await db.conn.execute("SELECT * FROM profile_samples WHERE id = ?", (sample_id,))
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such sample")
    await db.conn.execute("DELETE FROM profile_samples WHERE id = ?", (sample_id,))
    await db.conn.execute(
        "UPDATE profiles SET updated_at = ? WHERE id = ?", (time.time(), row["profile_id"])
    )
    await db.conn.commit()
    _unlink(request.app.state.config.profile_images_dir, row["image_path"])
    await _reload_gallery(request)
    return Response(status_code=204)


@router.get("/samples/{sample_id}/image.jpg", dependencies=[Depends(require_media_admin)])
async def sample_image(sample_id: int, request: Request) -> FileResponse:
    db = request.app.state.db
    row = await (
        await db.conn.execute(
            "SELECT image_path FROM profile_samples WHERE id = ?", (sample_id,)
        )
    ).fetchone()
    if row is None or not row["image_path"]:
        raise HTTPException(status_code=404, detail="No image")
    path = _safe_path(request.app.state.config.profile_images_dir, row["image_path"])
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No image")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@router.get("/candidates")
async def list_candidates(
    request: Request,
    kind: Optional[str] = Query(default=None),
    camera: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Unmatched crops, best-quality first.

    Ordered by quality rather than recency on purpose: this list exists to be
    ENROLLED FROM, and the shot worth enrolling is the legible one, not the
    most recent one. Recency is still available to the client via created_at.
    """
    db = request.app.state.db
    sql = "SELECT * FROM recognition_candidates WHERE 1=1"
    params: list[Any] = []
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    if camera:
        sql += " AND camera = ?"
        params.append(camera)
    sql += " ORDER BY quality DESC, created_at DESC LIMIT ?"
    params.append(limit)
    rows = await (await db.conn.execute(sql, params)).fetchall()
    return [_candidate_out(r) for r in rows]


@router.delete("/candidates/{candidate_id}", status_code=204)
async def delete_candidate(candidate_id: int, request: Request) -> Response:
    db = request.app.state.db
    row = await (
        await db.conn.execute(
            "SELECT image_path FROM recognition_candidates WHERE id = ?", (candidate_id,)
        )
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such candidate")
    await db.conn.execute("DELETE FROM recognition_candidates WHERE id = ?", (candidate_id,))
    await db.conn.commit()
    _unlink(request.app.state.config.candidate_crops_dir, row["image_path"])
    return Response(status_code=204)


@router.delete("/candidates", status_code=204)
async def clear_candidates(
    request: Request,
    kind: Optional[str] = Query(default=None),
) -> Response:
    """Clear the rolling candidate store — the 'forget the strangers' button."""
    db = request.app.state.db
    sql = "SELECT id, image_path FROM recognition_candidates"
    params: list[Any] = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    rows = await (await db.conn.execute(sql, params)).fetchall()
    await db.conn.execute(
        "DELETE FROM recognition_candidates" + (" WHERE kind = ?" if kind else ""), params
    )
    await db.conn.commit()
    for r in rows:
        _unlink(request.app.state.config.candidate_crops_dir, r["image_path"])
    return Response(status_code=204)


@router.get("/candidates/{candidate_id}/image.jpg", dependencies=[Depends(require_media_admin)])
async def candidate_image(candidate_id: int, request: Request) -> FileResponse:
    db = request.app.state.db
    row = await (
        await db.conn.execute(
            "SELECT image_path FROM recognition_candidates WHERE id = ?", (candidate_id,)
        )
    ).fetchone()
    if row is None or not row["image_path"]:
        raise HTTPException(status_code=404, detail="No image")
    path = _safe_path(request.app.state.config.candidate_crops_dir, row["image_path"])
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No image")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Heatmap — where recognition actually works
# ---------------------------------------------------------------------------


@router.get("/heatmap/{camera}")
async def recognition_heatmap(
    camera: str,
    request: Request,
    kind: str = Query(default="face"),
) -> dict[str, Any]:
    """The legibility map the ROI editor draws over the live frame.

    Returns two flat arrays of `cols * rows` values: `counts` normalized 0..1
    against the busiest cell (the absolute number means nothing to a person —
    what they need is where, relatively) and `quality` as the MEAN bestshot
    score in that cell, which is already absolute.

    Rendered as density-for-opacity and quality-for-hue, that distinction is
    the whole point: it separates "busy but unreadable" (the far pavement) from
    "quiet but sharp" (the doorstep), which is exactly what decides where a
    face zone should go and is invisible in a still frame.

    `suggested_zone` is a rectangle over the cells with both real evidence and
    usable quality — a starting point to drag, not a recommendation. It is
    empty when there is not yet enough evidence to say anything, which the UI
    shows as "watch for a while first".
    """
    if kind not in ("face", "plate"):
        raise HTTPException(status_code=400, detail="kind must be 'face' or 'plate'")
    heat = _heatmap(request)
    if heat is None:
        return {
            "camera": camera, "kind": kind, "cols": COLS, "rows": ROWS,
            "counts": [], "quality": [], "samples": 0, "peak": 0.0,
            "updated_at": None, "suggested_zone": [],
        }
    grid = await heat.grid(camera, kind)
    return {**grid, "suggested_zone": suggest_zone(grid)}


@router.delete("/heatmap/{camera}", status_code=204)
async def clear_recognition_heatmap(
    camera: str,
    request: Request,
    kind: Optional[str] = Query(default=None),
) -> Response:
    """Forget a camera's map — for a camera that has been moved or re-aimed,
    where everything it recorded about the old view is now a lie."""
    heat = _heatmap(request)
    if heat is not None:
        await heat.clear(camera=camera, kind=kind)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status")
async def recognition_status(request: Request) -> dict[str, Any]:
    """What recognition is currently capable of — the screen the app opens on."""
    db = request.app.state.db
    active = _active_model_key(request)
    counts = await (
        await db.conn.execute("SELECT kind, COUNT(*) AS n FROM profiles GROUP BY kind")
    ).fetchall()
    cand = await (
        await db.conn.execute(
            "SELECT kind, COUNT(*) AS n FROM recognition_candidates GROUP BY kind"
        )
    ).fetchall()
    stale = await (
        await db.conn.execute(
            "SELECT COUNT(*) AS n FROM profile_samples "
            "WHERE embedding IS NOT NULL AND model_key != ?",
            (active,),
        )
    ).fetchone()
    engine = getattr(request.app.state, "engine", None)
    face = getattr(engine, "_face", None) if engine is not None else None
    plates = getattr(engine, "_plates", None) if engine is not None else None
    return {
        "model_key": active,
        "ready": bool(active),
        # WHERE recognition runs, so "is my GPU being used?" is answerable from
        # the app instead of by reading source.
        #
        # It is CPU by design, and measured rather than assumed: a YuNet pass on
        # a person crop is ~4-9 ms and runs at most once per 0.6 s per tracked
        # person, so eight cameras each holding a person continuously cost about
        # an eighth of one core. Moving that to the GPU would take VRAM and
        # scheduling slots from D-FINE — the model that genuinely needs the card
        # — to save single-digit milliseconds. The plate OCR is a 128x64 input
        # where transfer overhead would likely exceed the 2.9 ms it takes on CPU.
        # MEASURED, not asserted. This used to be four hardcoded "cpu" strings
        # and a note — which stopped being the whole truth the moment the plate
        # OCR learned to follow the detector, and which could never answer "is
        # this actually costing me anything on MY box with MY settings".
        #
        # `devices` names each stage's silicon AND why, so a CPU stage reads as
        # a decision rather than an oversight. `timings` is the live rolling
        # window; a stage absent from it has not run, which is a different
        # finding from a stage that runs instantly and must not look the same.
        "devices": accel.report(
            getattr(request.app.state, "detector", None),
            plate_ocr_device=getattr(getattr(plates, "_reader", None), "device", None),
        ),
        "timings": TIMINGS.report(),
        "face": face.status() if hasattr(face, "status") else None,
        "plates": plates.status() if hasattr(plates, "status") else None,
        "profiles": {r["kind"]: r["n"] for r in counts},
        "candidates": {r["kind"]: r["n"] for r in cand},
        # Samples that can no longer be compared because the embedding model
        # changed under them. Non-zero means some profiles have silently
        # stopped matching and need re-enrolling — surfaced rather than left
        # to be discovered as "recognition just stopped working".
        "stale_samples": stale["n"] or 0,
        "defaults": {"face_threshold": FACE_THRESHOLD, "min_margin": MIN_MARGIN},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heatmap(request: Request) -> Optional[Any]:
    """The engine's heatmap accumulator, or None when recognition is not wired.

    None is a normal state (a box that never enabled recognition), so every
    caller degrades to an empty map rather than erroring.
    """
    engine = getattr(request.app.state, "engine", None)
    face = getattr(engine, "_face", None) if engine is not None else None
    return getattr(face, "heatmap", None)


async def _require_profile(db: Any, profile_id: int) -> Any:
    row = await (
        await db.conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,))
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such profile")
    return row


def _safe_path(base: Path, name: str) -> Optional[Path]:
    """Resolve `name` under `base`, refusing anything that escapes it.

    The stored value is always a bare filename this code wrote, so traversal
    should be impossible — but this is the one router that serves biometric
    imagery by a path read out of a database, and "should be impossible" is
    not the standard that deserves.
    """
    if not name:
        return None
    try:
        resolved = (base / name).resolve()
        resolved.relative_to(base.resolve())
    except (ValueError, OSError):
        log.warning("refusing image path outside its directory: %r", name)
        return None
    return resolved


def _unlink(base: Path, name: str) -> None:
    path = _safe_path(base, name)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove recognition image %s", path)


async def _reload_gallery(request: Request) -> None:
    """Tell the engine its gallery is stale.

    Best-effort by design: recognition not running is the normal state on a box
    that has not enabled it, and a profile edit must still succeed there.
    """
    engine = getattr(request.app.state, "engine", None)
    reload_fn = getattr(engine, "reload_gallery", None)
    if reload_fn is None:
        return
    try:
        await reload_fn()
    except Exception:
        log.exception("gallery reload failed after a profile change")

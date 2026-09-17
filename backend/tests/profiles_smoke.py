#!/usr/bin/env python3
"""Recognition profiles API: CRUD, enrollment, and the imagery lifecycle.

The RBAC gate on this router is proven in rbac_smoke (every route, including
the reads, must 403 for a viewer). This suite covers what happens once you are
an admin, and concentrates on the things that would be quietly wrong:

1. ENROLLMENT MOVES THE CROP. A candidate's image lives in the rolling
   directory that the purge job walks. If enrollment copied the path instead of
   moving the file, every enrolled reference image would vanish when the
   retention window rolled past it — weeks later, silently, long after anyone
   would connect it to enrolling.
2. DELETE REALLY DELETES. "Delete this person" leaving their face crops on disk
   makes the button a lie, and this is the one store where that is not
   cosmetic.
3. KIND IS ENFORCED. A face candidate must not enroll into a vehicle profile;
   its embedding would sit in a gallery that only ever compares plates.
4. PATH TRAVERSAL. Image paths are read out of a database and joined to a
   directory. They are always bare filenames this code wrote, but this is the
   router that serves biometric imagery, so the guard is tested rather than
   assumed.

Usage: python backend/tests/profiles_smoke.py  (needs backend deps installed)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="vigilume-profiles-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.native.recognition import to_blob  # noqa: E402

import numpy as np  # noqa: E402

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


def login(client: TestClient) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"password": "test-password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def seed_candidate(
    client: TestClient, *, kind: str = "face", quality: float = 0.8,
    plate: str = "", with_image: bool = True, camera: str = "front",
) -> int:
    """Insert a candidate the way the engine's recognition pass would."""
    cfg = app.state.config
    db = app.state.db
    vec = np.random.default_rng(len(plate) + int(quality * 100)).normal(size=8)
    blob = to_blob(vec) if kind == "face" else None

    async def _insert() -> int:
        cur = await db.conn.execute(
            "INSERT INTO recognition_candidates (kind, camera, event_fid, embedding, dim, "
            "plate, model_key, image_path, quality, best_score, best_profile_id, created_at) "
            "VALUES (?, ?, 'fid-1', ?, ?, ?, 'test-model', '', ?, 0.0, NULL, 1000.0)",
            (kind, camera, blob, 8 if blob else 0, plate, quality),
        )
        cid = cur.lastrowid
        if with_image:
            name = f"{cid}.jpg"
            cfg.candidate_crops_dir.mkdir(parents=True, exist_ok=True)
            (cfg.candidate_crops_dir / name).write_bytes(b"\xff\xd8\xff\xdb-not-a-real-jpeg")
            await db.conn.execute(
                "UPDATE recognition_candidates SET image_path = ? WHERE id = ?", (name, cid)
            )
        await db.conn.commit()
        return cid

    return _run(_insert())


def _run(coro):
    import asyncio

    loop = getattr(_run, "_loop", None)
    if loop is None:
        loop = asyncio.new_event_loop()
        _run._loop = loop  # type: ignore[attr-defined]
    return loop.run_until_complete(coro)


def profile_checks(client: TestClient, h: dict) -> None:
    print("\nprofiles: CRUD")
    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "person", "name": "  Adam  "})
    check(r.status_code == 201, f"create a person profile (got {r.status_code}: {r.text[:120]})")
    adam = r.json()
    check(adam["name"] == "Adam", "the name is trimmed on the way in")
    check(adam["sample_count"] == 0, "a new profile has no samples")
    check(adam["enabled"] is True, "and is enabled by default")

    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "person", "name": "Adam"})
    check(r.status_code == 409, "a duplicate (kind, name) is 409, not a second row")

    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "vehicle", "name": "Adam"})
    check(r.status_code == 201, "...but the SAME name under a different kind is fine")
    truck = r.json()

    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "alien", "name": "X"})
    check(r.status_code == 422, "an unknown kind is refused")
    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "person", "name": "   "})
    check(r.status_code == 422, "a blank name is refused")
    r = client.post("/api/recognition/profiles", headers=h,
                    json={"kind": "person", "name": "Bad", "threshold": 1.4})
    check(r.status_code == 422, "a threshold outside 0..1 is refused, not clamped")

    print("\nprofiles: update")
    r = client.put(f"/api/recognition/profiles/{adam['id']}", headers=h,
                   json={"notes": "lives here", "threshold": 0.5})
    check(r.status_code == 200 and r.json()["notes"] == "lives here", "partial update applies")
    check(r.json()["threshold"] == 0.5, "...and the threshold override persists")
    check(r.json()["name"] == "Adam", "...without disturbing fields it did not mention")

    r = client.put(f"/api/recognition/profiles/{adam['id']}", headers=h, json={})
    check(r.status_code == 400, "an empty update is refused rather than silently no-op")
    r = client.put("/api/recognition/profiles/999999", headers=h, json={"notes": "x"})
    check(r.status_code == 404, "updating a missing profile is 404")

    print("\nprofiles: plate samples")
    r = client.post(f"/api/recognition/profiles/{truck['id']}/plate", headers=h,
                    json={"plate": "7abc-123"})
    check(r.status_code == 201, "a vehicle takes a typed plate with no sighting")
    check(r.json()["plate"] == "7ABC123", "...normalized on the way in")
    r = client.post(f"/api/recognition/profiles/{adam['id']}/plate", headers=h,
                    json={"plate": "ABC123"})
    check(r.status_code == 400, "a PERSON profile refuses a plate")
    r = client.post(f"/api/recognition/profiles/{truck['id']}/plate", headers=h,
                    json={"plate": "!!!"})
    check(r.status_code == 422, "a plate with no alphanumerics is refused")

    return adam, truck


def enroll_checks(client: TestClient, h: dict, adam: dict, truck: dict) -> None:
    cfg = app.state.config

    print("\nenrollment: candidates become samples")
    c1 = seed_candidate(client, kind="face", quality=0.9)
    c2 = seed_candidate(client, kind="face", quality=0.7)
    c3 = seed_candidate(client, kind="face", quality=0.3)

    r = client.get("/api/recognition/candidates", headers=h)
    check(r.status_code == 200 and len(r.json()) == 3, "candidates list")
    qualities = [c["quality"] for c in r.json()]
    check(qualities == sorted(qualities, reverse=True),
          "candidates come back BEST FIRST — this list exists to be enrolled from")
    check(r.json()[0]["image_url"].endswith("/image.jpg"), "a candidate with a crop links it")

    crop_path = cfg.candidate_crops_dir / f"{c1}.jpg"
    check(crop_path.exists(), "the candidate crop is on disk before enrollment")

    r = client.post(f"/api/recognition/profiles/{adam['id']}/enroll", headers=h,
                    json={"candidate_ids": [c1, c2]})
    check(r.status_code == 201, f"enroll two candidates (got {r.status_code}: {r.text[:200]})")
    check(r.json()["enrolled"] == 2, "...and both became samples in one request")

    r = client.get(f"/api/recognition/profiles/{adam['id']}", headers=h)
    check(r.json()["sample_count"] == 2, "the profile now carries two samples")
    sample_ids = [s["id"] for s in r.json()["samples"]]

    check(not crop_path.exists(),
          "the crop is GONE from the rolling candidate directory")
    check(any((cfg.profile_images_dir / f"{sid}.jpg").exists() for sid in sample_ids),
          "...because it was MOVED into the durable profile directory")

    r = client.get("/api/recognition/candidates", headers=h)
    check(len(r.json()) == 1, "the enrolled candidates left the unknown list")

    print("\nenrollment: refusals")
    r = client.post(f"/api/recognition/profiles/{truck['id']}/enroll", headers=h,
                    json={"candidate_ids": [c3]})
    check(r.status_code == 400, "a FACE candidate will not enroll into a vehicle profile")
    r = client.post(f"/api/recognition/profiles/{adam['id']}/enroll", headers=h,
                    json={"candidate_ids": [c3, 999999]})
    check(r.status_code == 404, "an unknown candidate id fails the whole request")
    r = client.get("/api/recognition/candidates", headers=h)
    check(len(r.json()) == 1,
          "...and the request was ATOMIC — the valid candidate was not consumed")
    r = client.post(f"/api/recognition/profiles/{adam['id']}/enroll", headers=h,
                    json={"candidate_ids": []})
    check(r.status_code == 422, "an empty enroll list is refused")

    print("\nenrollment: imagery is served and deleted")
    sid = sample_ids[0]
    r = client.get(f"/api/recognition/samples/{sid}/image.jpg", headers=h)
    check(r.status_code == 200 and r.content.startswith(b"\xff\xd8"),
          "an enrolled reference image is served to an admin")

    r = client.delete(f"/api/recognition/samples/{sid}", headers=h)
    check(r.status_code == 204, "a sample deletes")
    check(not (cfg.profile_images_dir / f"{sid}.jpg").exists(),
          "...and its reference image goes with it")
    r = client.get(f"/api/recognition/profiles/{adam['id']}", headers=h)
    check(r.json()["sample_count"] == 1, "the profile is down to one sample")

    print("\ndelete: a profile takes its samples and images with it")
    remaining = client.get(f"/api/recognition/profiles/{adam['id']}", headers=h).json()
    left_image = remaining["samples"][0]["id"]
    check((cfg.profile_images_dir / f"{left_image}.jpg").exists(), "image present before delete")
    r = client.delete(f"/api/recognition/profiles/{adam['id']}", headers=h)
    check(r.status_code == 204, "the profile deletes")
    check(not (cfg.profile_images_dir / f"{left_image}.jpg").exists(),
          "...and the enrolled face crop is really gone from disk")
    check(client.get(f"/api/recognition/profiles/{adam['id']}", headers=h).status_code == 404,
          "...and the profile is a 404 afterwards")

    print("\ncandidates: clearing")
    seed_candidate(client, kind="plate", plate="XYZ789", quality=0.6)
    r = client.get("/api/recognition/candidates", headers=h, params={"kind": "plate"})
    check(len(r.json()) == 1, "candidates filter by kind")
    r = client.delete("/api/recognition/candidates", headers=h, params={"kind": "face"})
    check(r.status_code == 204, "clearing one kind succeeds")
    r = client.get("/api/recognition/candidates", headers=h)
    check(len(r.json()) == 1 and r.json()[0]["kind"] == "plate",
          "...and left the other kind alone")
    client.delete("/api/recognition/candidates", headers=h)
    check(len(client.get("/api/recognition/candidates", headers=h).json()) == 0,
          "clearing with no filter empties the store")


def traversal_checks(client: TestClient, h: dict) -> None:
    print("\nimagery: path traversal is refused")
    from app.routers.profiles import _safe_path

    base = app.state.config.profile_images_dir
    check(_safe_path(base, "12.jpg") is not None, "a bare filename resolves")
    check(_safe_path(base, "../../etc/passwd") is None, "a traversal escape is refused")
    check(_safe_path(base, "/etc/passwd") is None, "an absolute path is refused")
    check(_safe_path(base, "") is None, "an empty path is refused")

    # And end to end, through the route: a poisoned DB row must 404, not serve.
    db = app.state.db

    async def _poison() -> int:
        cur = await db.conn.execute(
            "INSERT INTO profiles (kind, name, notes, enabled, threshold, created_at, "
            "updated_at) VALUES ('person', 'poison', '', 1, NULL, 1.0, 1.0)"
        )
        pid = cur.lastrowid
        cur = await db.conn.execute(
            "INSERT INTO profile_samples (profile_id, embedding, dim, plate, model_key, "
            "image_path, quality, source_fid, created_at) "
            "VALUES (?, NULL, 0, '', '', '../../../etc/hostname', 0, '', 1.0)",
            (pid,),
        )
        await db.conn.commit()
        return cur.lastrowid

    bad_sample = _run(_poison())
    r = client.get(f"/api/recognition/samples/{bad_sample}/image.jpg", headers=h)
    check(r.status_code == 404,
          "a DB row pointing outside the image directory 404s instead of serving it")


def status_checks(client: TestClient, h: dict) -> None:
    print("\nstatus")
    r = client.get("/api/recognition/status", headers=h)
    check(r.status_code == 200, "status is reachable")
    body = r.json()
    check("model_key" in body and "ready" in body, "status reports the active model")
    check(body["ready"] is False,
          "with no recognition engine loaded, status is honest about not being ready")
    check("face_threshold" in body["defaults"], "and publishes the matcher defaults")
    check(isinstance(body["stale_samples"], int),
          "stale_samples counts embeddings the active model can no longer compare")


def main() -> int:
    with TestClient(app) as client:
        h = login(client)
        adam, truck = profile_checks(client, h)
        enroll_checks(client, h, adam, truck)
        traversal_checks(client, h)
        status_checks(client, h)
    print(f"\nALL {PASS} CHECKS PASSED (recognition profiles API)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

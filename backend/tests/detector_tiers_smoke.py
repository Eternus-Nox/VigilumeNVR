"""Smoke suite for the higher-accuracy + big-vocabulary detector tiers.

Covers the additions layered on top of the original dfine_n/s/m detector:

  1. pins — dfine_l / dfine_x (COCO) + dfine_l_obj365 (Objects365) are pinned
     with revision-locked onnx-community URLs, 64-hex SHA-256, plausible sizes,
     and a per-model ``labelmap`` (coco vs obj365); tier metadata advertises the
     class vocabulary (COCO 80 vs Objects365 365/366).
  2. labelmap decode — the NMS-free ``decode`` is class-count agnostic: a
     synthetic 80-wide COCO logits vector and a synthetic 366-wide Objects365
     vector both decode to the right ``class_id``, and the ACTIVE labelmap view
     (``coco_labels.ID_TO_LABEL``, the object the engine imports once) maps that
     id to the right label for whichever model is active — COCO id 0 -> person,
     Objects365 ids 0/1/6/93/140 -> none/person/car/dog/cat.
  3. REAL dfine_l SHA — if the artifact is cached (or SENTINEL_TEST_DOWNLOAD_L=1
     to fetch the ~125 MB file), verify the on-disk bytes hash to the pin exactly
     like native_smoke does for dfine_n; otherwise assert metadata and print a
     clear skip note (durable suite stays offline/fast).
  4. labels endpoint — GET /api/detection/labels (admin) returns the ACTIVE
     model's ordered vocabulary and 404s an unknown model; ?model= selects any
     known model, so the obj365 model surfaces its full 366-entry list.

CPU-only. Usage: python backend/tests/detector_tiers_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Clean env before app config is instantiated (same guard as the sibling suites).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-tiers-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main  # noqa: E402,F401 — import hygiene: must not pull onnxruntime
from app.native import coco_labels as labels_mod  # noqa: E402
from app.native.coco_labels import (  # noqa: E402
    COCO_LABELS,
    ID_TO_LABEL,
    LABELMAPS,
    active_labelmap_name,
    labels_for,
    num_classes,
    selectable_labels,
    set_active_labelmap,
    vocabulary_name,
)
from app.native.detector import (  # noqa: E402
    MODELS,
    decode,
    ensure_model,
    model_labelmap,
    model_path,
    sha256_file,
)
from app.native.obj365_labels import OBJ365_LABELS  # noqa: E402

PASS = 0

NEW_COCO = ("dfine_l", "dfine_x")
OBJ365_KEY = "dfine_l_obj365"

MODEL_CACHE = Path(
    os.environ.get("SENTINEL_TEST_MODEL_CACHE")
    or Path.home() / ".cache" / "sentinel-tests" / "models"
)


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


def _boxes(cx, cy, w, h):
    b = np.zeros((1, 300, 4), dtype=np.float32)
    b[0, 0] = (cx, cy, w, h)
    return b


# =====================================================================
# 1. pins + tier vocabulary metadata
# =====================================================================


def pin_checks() -> None:
    print("tiers 1: new-model pins + vocabulary metadata")
    from app.native.model_store import MODEL_TIERS, TIER_ORDER, tier_metadata

    for key in (*NEW_COCO, OBJ365_KEY):
        check(key in MODELS and key in MODEL_TIERS and key in TIER_ORDER,
              f"{key} present in MODELS + MODEL_TIERS + TIER_ORDER")
        pin = MODELS[key]
        ok = (
            pin["url"].startswith("https://huggingface.co/onnx-community/")
            and "/resolve/" in pin["url"]
            and len(pin["sha256"]) == 64
            and int(pin["bytes"]) > 100_000_000
        )
        check(ok, f"{key}: revision-pinned onnx-community URL + 64-hex SHA + size")

    check(model_labelmap("dfine_l") == "coco" and model_labelmap("dfine_x") == "coco",
          "dfine_l / dfine_x carry the COCO labelmap")
    check(model_labelmap(OBJ365_KEY) == "obj365",
          "dfine_l_obj365 carries the obj365 labelmap")
    # exact pinned SHAs (verified by a local download + shasum during authoring)
    check(MODELS["dfine_l"]["sha256"].startswith("d678f3baebfb"),
          "dfine_l SHA matches the verified pin")
    check(MODELS["dfine_x"]["sha256"].startswith("644fb5124c9c"),
          "dfine_x SHA matches the verified pin")
    check(MODELS[OBJ365_KEY]["sha256"].startswith("cd0dfa92a2e0"),
          "dfine_l_obj365 SHA matches the verified pin")

    for key in NEW_COCO:
        m = tier_metadata(key)
        check(m["vocabulary"] == "coco" and m["num_classes"] == 80,
              f"{key} advertises the COCO-80 vocabulary")
    mo = tier_metadata(OBJ365_KEY)
    check(mo["vocabulary"] == "objects365" and mo["num_classes"] == 365
          and mo["map_dataset"] == "Objects365",
          "obj365 tier advertises the Objects365 vocabulary (365 selectable) + benchmark")
    check(tier_metadata("dfine_x")["approx_map"] > tier_metadata("dfine_m")["approx_map"]
          > tier_metadata("dfine_s")["approx_map"],
          "COCO tiers x > m > s advertise increasing COCO mAP")


# =====================================================================
# 2. class-count-agnostic decode + active labelmap resolution
# =====================================================================


def labelmap_checks() -> None:
    print("tiers 2: labelmap decode + active-model label view")
    check(set(LABELMAPS) == {"coco", "obj365"}, "LABELMAPS registers coco + obj365")
    check(labels_for("coco") == COCO_LABELS and len(COCO_LABELS) == 80,
          "coco labelmap is the 80-class COCO tuple")
    check(labels_for("obj365") == OBJ365_LABELS and len(OBJ365_LABELS) == 366,
          "obj365 full labelmap is the 366-entry Objects365 tuple (ID_TO_LABEL space)")
    check(vocabulary_name("coco") == "coco"
          and vocabulary_name("obj365") == "objects365",
          "short machine vocabulary names for the API")
    # user-facing pick list drops the obj365 id-0 background placeholder
    check(selectable_labels("coco") == COCO_LABELS and num_classes("coco") == 80,
          "coco selectable list == full 80 (no background placeholder)")
    sel = selectable_labels("obj365")
    check(sel == OBJ365_LABELS[1:] and num_classes("obj365") == 365
          and "none" not in sel and sel[0] == "person",
          "obj365 selectable list drops id-0 'none' -> 365 real classes, starts at person")

    # --- COCO decode: an 80-wide logits vector, class 0 -> person ---
    logits = np.full((1, 300, 80), -20.0, dtype=np.float32)
    logits[0, 0, 0] = 3.0
    dets = decode(logits, _boxes(0.5, 0.5, 0.2, 0.2), 0.5, 704, 480)
    check(len(dets) == 1 and int(dets.class_id[0]) == 0, "COCO logits[80] decodes class_id 0")
    set_active_labelmap("coco")
    check(active_labelmap_name() == "coco", "active labelmap set to coco")
    check(ID_TO_LABEL.get(0) == "person" and ID_TO_LABEL.get(16) == "dog",
          "COCO active view: id 0 -> person, id 16 -> dog")

    # --- Objects365 decode: a 366-wide vector, ids resolve via obj365 view ---
    ob = np.full((1, 300, 366), -20.0, dtype=np.float32)
    ob[0, 0, 93] = 3.0  # 93 == dog in Objects365
    dets = decode(ob, _boxes(0.4, 0.4, 0.3, 0.3), 0.5, 704, 480)
    check(len(dets) == 1 and int(dets.class_id[0]) == 93,
          "Objects365 logits[366] decodes class_id 93 (no per-model branch)")
    set_active_labelmap("obj365")
    check(active_labelmap_name() == "obj365", "active labelmap switched to obj365")
    check(ID_TO_LABEL.get(0) == "none" and ID_TO_LABEL.get(1) == "person"
          and ID_TO_LABEL.get(6) == "car" and ID_TO_LABEL.get(93) == "dog"
          and ID_TO_LABEL.get(140) == "cat",
          "obj365 active view: 0->none, 1->person, 6->car, 93->dog, 140->cat")
    check(ID_TO_LABEL.get(999) is None,
          "out-of-range class_id resolves to None (dropped downstream, never raises)")

    # --- the engine imports ID_TO_LABEL by identity; the swap mutates in place ---
    check(ID_TO_LABEL is labels_mod.ID_TO_LABEL,
          "ID_TO_LABEL is a single long-lived object (engine's binding stays valid)")

    # reset so later sections / other suites see the default COCO space
    set_active_labelmap("coco")
    check(ID_TO_LABEL.get(0) == "person", "reset to coco restores the COCO view")

    set_active_labelmap(model_labelmap("dfine_l"))
    check(active_labelmap_name() == "coco", "dfine_l selects the coco view")
    set_active_labelmap(model_labelmap(OBJ365_KEY))
    check(active_labelmap_name() == "obj365", "dfine_l_obj365 selects the obj365 view")
    set_active_labelmap("coco")


# =====================================================================
# 3. REAL dfine_l download + SHA verify (cached / opt-in)
# =====================================================================


async def _real_dfine_l() -> None:
    cached = model_path(MODEL_CACHE, "dfine_l")
    want = os.environ.get("SENTINEL_TEST_DOWNLOAD_L") == "1"
    if not cached.is_file() and not want:
        print("  .. dfine_l artifact not cached and SENTINEL_TEST_DOWNLOAD_L!=1 "
              "-> skipping the ~125 MB real download (metadata asserted above)")
        return
    path = await ensure_model(MODEL_CACHE, "dfine_l")
    check(path.stat().st_size == MODELS["dfine_l"]["bytes"],
          "REAL dfine_l on-disk size matches the pin")
    check(sha256_file(path) == MODELS["dfine_l"]["sha256"],
          "REAL dfine_l SHA-256 matches the pin")


def real_dfine_l_checks() -> None:
    print("tiers 3: REAL dfine_l size + SHA-256 (cached / opt-in)")
    asyncio.run(_real_dfine_l())


# =====================================================================
# 4. /api/detection/labels endpoint (admin; active + explicit model)
# =====================================================================


def labels_endpoint_checks() -> None:
    print("tiers 4: GET /api/detection/labels (active-model vocabulary)")
    with TestClient(app.main.app) as client:
        check(client.get("/api/detection/labels").status_code in (401, 403),
              "GET /api/detection/labels requires auth")
        token = client.post(
            "/api/auth/login", json={"password": "test-password"}
        ).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # default = the active model (dfine_s -> COCO); shape == frontend
        # ActiveLabelsResponse {model, vocabulary, count, labels}
        r = client.get("/api/detection/labels", headers=headers)
        check(r.status_code == 200, "labels endpoint -> 200")
        body = r.json()
        check(set(body) == {"model", "vocabulary", "count", "labels"},
              "labels body matches ActiveLabelsResponse {model, vocabulary, count, labels}")
        check(body["model"] == "dfine_s" and body["vocabulary"] == "coco"
              and body["count"] == 80 and len(body["labels"]) == 80
              and body["labels"][0] == "person",
              "active model (dfine_s) -> COCO-80, labels[0] == person")

        # explicit obj365 model -> 365 real Objects365 classes (no 'none')
        r = client.get("/api/detection/labels?model=dfine_l_obj365", headers=headers)
        b2 = r.json()
        check(r.status_code == 200 and b2["vocabulary"] == "objects365"
              and b2["count"] == 365 and len(b2["labels"]) == 365
              and "none" not in b2["labels"] and b2["labels"][0] == "person"
              and "dog" in b2["labels"] and "table_tennis" in b2["labels"],
              "?model=dfine_l_obj365 -> 365 Objects365 classes (background 'none' dropped)")

        r = client.get("/api/detection/labels?model=dfine_l", headers=headers)
        check(r.json()["count"] == 80,
              "?model=dfine_l -> COCO-80 vocabulary")
        check(client.get("/api/detection/labels?model=nope", headers=headers)
              .status_code == 404,
              "unknown model -> 404")


def main() -> None:
    pin_checks()
    labelmap_checks()
    real_dfine_l_checks()
    labels_endpoint_checks()
    print(f"\nALL {PASS} CHECKS PASSED (detector tiers + labelmaps + labels API)")


if __name__ == "__main__":
    main()

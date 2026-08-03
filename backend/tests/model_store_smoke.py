"""Smoke suite for the in-app detection-model manager (CONTRACTS.md addendum
"In-app detection model tiers").

Covers, CPU-only, NO real 100 MB downloads (the HTTP fetch is MOCKED with
httpx.MockTransport; the pin is temporarily patched to fake bytes so the
full state machine runs against a real tier key):

  1. tier metadata — completeness + keys map 1:1 to detector MODELS, size
     from the pin, input_size == detector INPUT_SIZE, tier/label/blurb/map
  2. download state machine — absent -> downloading -> verifying -> ready
     with progress advancing; sidecar written; sha_ok set; broadcasts
     captured by a fake WS client with the exact model_status shape
  3. SHA mismatch -> error + detail + the bad file removed (nothing left)
  4. retryable — after a failed download a fresh call succeeds
  5. idempotent — a second download while one is in flight is a no-op
     (one HTTP fetch); download() on a ready model is a no-op
  6. delete — frees the file + sidecar and resets to absent
  7. detector integration — reconfigure() is non-blocking (background reload;
     tolerates a not-yet-downloaded model); GPU-hard-fail boot downloads
     nothing (the store is only driven once the detector proceeds)
  8. routes (real app, REQUIRE_GPU=1) — GET /api/detection/models shape,
     download/activate 404s, activate switches settings + is reflected in
     GET, DELETE 409 on the active model + frees a non-active one, and
     /api/system/detector carries the active model's state/progress; health
     stays 200 immediately with the active model absent

Usage: python backend/tests/model_store_smoke.py  (needs backend deps).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, BACKEND)

# Environment must be clean before app config is instantiated (lifespan).
for i in (1, 2, 3):
    for suffix in ("NAME", "IP", "USER", "PASS", "MODEL", "FRIENDLY"):
        os.environ.pop(f"CAM{i}_{suffix}", None)
os.environ["ADMIN_PASSWORD"] = "test-password"
os.environ["PUBLIC_URL"] = ""
# REQUIRE_GPU=1 on this GPU-less host -> the app-boot detector hard-fails fast
# and downloads nothing; store-level tests construct their own store/detector.
os.environ["SENTINEL_REQUIRE_GPU"] = "1"
# Unroutable local ports -> instant refusal (go2rtc syncs never block).
os.environ["GO2RTC_URL"] = "http://127.0.0.1:1"
os.environ["GO2RTC_RTSP_URL"] = "rtsp://127.0.0.1:1"

TMP = Path(tempfile.mkdtemp(prefix="sentinel-modelstore-smoke-"))
os.environ["DATA_DIR"] = str(TMP / "data")
os.environ["MEDIA_DIR"] = str(TMP / "media")
os.environ["GO2RTC_CONFIG_DIR"] = str(TMP / "go2rtc-config")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main  # noqa: E402,F401 — import hygiene: must not pull onnxruntime
from app.native import detector as detector_module  # noqa: E402
from app.native.detector import INPUT_SIZE, MODELS, OnnxDetector  # noqa: E402
from app.native import model_store as ms  # noqa: E402
from app.native.model_store import (  # noqa: E402
    MODEL_TIERS,
    TIER_ORDER,
    ModelStore,
    tier_metadata,
)

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {msg}")


# A ~2 MB fake artifact -> multiple 256 KB stream chunks -> progress advances.
FAKE_PAYLOAD = b"fake-onnx-artifact-" * 120_000
FAKE_SHA = hashlib.sha256(FAKE_PAYLOAD).hexdigest()


class FakeWS:
    """Duck-typed WSManager: captures broadcast payloads."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)


class _FakeSettings:
    """Minimal SettingsStore stand-in for activate_model unit tests."""

    # Software Privacy Mode (app/privacy.py): duck-typed for the capture gates.
    # Nothing is private in these suites — privacy_smoke.py owns that behaviour.
    private_cameras: frozenset = frozenset()

    def is_private(self, camera: str) -> bool:
        return False

    def __init__(self, model: str):
        self._model = model

    def get(self) -> dict:
        return {"detection": {"model": self._model, "confidence": 0.5}}

    async def update(self, new: dict) -> dict:
        self._model = new["detection"]["model"]
        return self.get()

    @property
    def detection(self) -> dict:
        return {"model": self._model, "confidence": 0.5}


class _FakeEngine:
    async def reload(self) -> None:
        return None


class _FakeDetector:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.ready = False


class _FakeState:
    def __init__(self, store, settings, engine, detector):
        self.model_store = store
        self.settings = settings
        self.engine = engine
        self.detector = detector


class _PinPatch:
    """Temporarily repoint a real tier key's pin at the fake payload so the
    full download+verify state machine runs without a real model download."""

    def __init__(self, key: str):
        self.key = key

    def __enter__(self):
        self._orig = dict(MODELS[self.key])
        MODELS[self.key] = {
            "url": "https://models.example/fake.onnx",
            "bytes": len(FAKE_PAYLOAD),
            "sha256": FAKE_SHA,
        }
        return self

    def __exit__(self, *exc):
        MODELS[self.key] = self._orig
        return False


def _factory(body_ref: dict, hits: dict):
    def make_client() -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            hits["n"] += 1
            return httpx.Response(200, content=body_ref["body"])

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return make_client


# =====================================================================
# 1. tier metadata
# =====================================================================


def metadata_checks() -> None:
    print("model_store 1: tier metadata")
    check(set(MODEL_TIERS) == set(MODELS) == {"dfine_n", "dfine_s", "dfine_m",
                                              "dfine_l", "dfine_x", "dfine_l_obj365"},
          "MODEL_TIERS keys map 1:1 to detector MODELS (n/s/m/l/x + obj365)")
    check(TIER_ORDER == ["dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x",
                         "dfine_l_obj365"],
          "TIER_ORDER runs fastest COCO -> heaviest COCO -> big-vocabulary")
    tiers = {MODEL_TIERS[k]["tier"] for k in TIER_ORDER}
    check(tiers == {"lightweight", "balanced", "heavy", "accurate", "maximum",
                    "objects365"},
          "tiers are lightweight/balanced/heavy/accurate/maximum/objects365")
    check(MODEL_TIERS["dfine_n"]["tier"] == "lightweight"
          and MODEL_TIERS["dfine_s"]["tier"] == "balanced"
          and MODEL_TIERS["dfine_m"]["tier"] == "heavy"
          and MODEL_TIERS["dfine_l"]["tier"] == "accurate"
          and MODEL_TIERS["dfine_x"]["tier"] == "maximum"
          and MODEL_TIERS["dfine_l_obj365"]["tier"] == "objects365",
          "n->lightweight, s->balanced, m->heavy, l->accurate, x->maximum, "
          "obj365->objects365")
    for key in TIER_ORDER:
        meta = tier_metadata(key)
        check(set(meta) == {"key", "tier", "label", "blurb", "size_bytes",
                            "input_size", "approx_map", "map_dataset",
                            "recommended_for", "vocabulary", "num_classes"},
              f"tier_metadata({key}) carries every required field")
        check(meta["size_bytes"] == MODELS[key]["bytes"] and meta["size_bytes"] > 0,
              f"{key} size_bytes comes from the pin")
        check(meta["input_size"] == INPUT_SIZE == 640,
              f"{key} input_size == detector INPUT_SIZE (640)")
        check(isinstance(meta["approx_map"], (int, float)) and 0 < meta["approx_map"] < 100,
              f"{key} approx_map is a plausible AP")
        check(bool(meta["label"]) and bool(meta["blurb"]) and bool(meta["recommended_for"]),
              f"{key} label/blurb/recommended_for non-empty")
    # COCO tiers advertise the 80-class vocabulary; the obj365 tier the 365-class
    # Objects365 space (366 output ids incl. the id-0 background placeholder).
    for key in ("dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x"):
        m = tier_metadata(key)
        check(m["vocabulary"] == "coco" and m["num_classes"] == 80,
              f"{key} advertises the COCO-80 vocabulary")
    mo = tier_metadata("dfine_l_obj365")
    check(mo["vocabulary"] == "objects365" and mo["num_classes"] == 365,
          "obj365 tier advertises the Objects365 vocabulary (365 selectable classes)")
    # accuracy monotonic within the COCO tiers (comparable benchmark only)
    coco_maps = [tier_metadata(k)["approx_map"]
                 for k in ("dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x")]
    check(coco_maps == sorted(coco_maps) and coco_maps[-1] > coco_maps[0],
          "COCO tiers advertise monotonically increasing COCO mAP")


# =====================================================================
# 2-6. download state machine (mocked HTTP)
# =====================================================================


async def _store_cases() -> None:
    print("model_store 2: download state machine (MockTransport)")
    with _PinPatch("dfine_n"):
        # ---- fresh download success + broadcasts ----
        d = TMP / "store-a"
        ws = FakeWS()
        hits = {"n": 0}
        body = {"body": FAKE_PAYLOAD}
        store = ModelStore(d, broadcast=ws.broadcast, client_factory=_factory(body, hits))
        store.active_key_getter = lambda: "dfine_n"
        store.loaded_key_getter = lambda: None
        store.refresh()
        check(store.state_of("dfine_n") == "absent", "absent before any download")

        path = await store.ensure_ready("dfine_n")
        await asyncio.sleep(0.05)  # flush fire-and-forget broadcast tasks
        check(path.is_file() and path.read_bytes() == FAKE_PAYLOAD,
              "ensure_ready downloads the artifact to models_dir/{key}.onnx")
        check(store.state_of("dfine_n") == "ready", "state is ready after verify")
        entry = store.model_entry("dfine_n")
        check(entry["state"] == "ready" and entry["progress_pct"] == 100
              and entry["sha_ok"] is True,
              "model_entry: ready, progress 100, sha_ok true")
        sidecar = json.loads((d / "dfine_n.json").read_text())
        check(sidecar["sha256"] == FAKE_SHA and sidecar["downloaded_at"] > 0,
              "sidecar records sha256 + downloaded_at")
        check(hits["n"] == 1, "exactly one HTTP fetch")

        states = [m["state"] for m in ws.messages if m["key"] == "dfine_n"]
        check("downloading" in states and "verifying" in states and states[-1] == "ready",
              "broadcast state sequence: downloading -> verifying -> ready")
        i_dl, i_vf, i_rd = (states.index("downloading"), states.index("verifying"),
                            len(states) - 1)
        check(i_dl < i_vf < i_rd, "broadcasts are ordered download < verify < ready")
        prog = [m["progress_pct"] for m in ws.messages
                if m["key"] == "dfine_n" and m["state"] == "downloading"]
        check(any(0 <= p < 100 for p in prog), "progress advances during downloading")
        sample = ws.messages[0]
        check(set(sample) == {"type", "key", "tier", "state", "progress_pct",
                             "active", "loaded"},
              "model_status frame carries exactly the WS contract keys")
        check(sample["type"] == "model_status" and sample["tier"] == "lightweight",
              "model_status: type + tier populated")

        # ---- download() on a ready model is a no-op ----
        before = hits["n"]
        again = store.download("dfine_n")
        check(again["state"] == "ready" and hits["n"] == before,
              "download() on a ready model is a no-op (no re-fetch)")

        # ---- idempotent: a second download while one is in flight ----
        print("model_store 5: idempotent concurrent download")
        d2 = TMP / "store-idem"
        hits2 = {"n": 0}
        store2 = ModelStore(d2, client_factory=_factory({"body": FAKE_PAYLOAD}, hits2))
        store2.refresh()
        task = store2._spawn("dfine_n")            # registers the in-flight task
        noop = store2.download("dfine_n")          # must NOT start a second one
        check(store2._tasks["dfine_n"] is task,
              "download() while in flight joins the existing task (no second)")
        await task
        check(hits2["n"] == 1, "concurrent downloads => a single HTTP fetch")
        results = await asyncio.gather(
            store2.ensure_ready("dfine_n"), store2.ensure_ready("dfine_n")
        )
        check(results[0] == results[1] and hits2["n"] == 1,
              "ensure_ready is idempotent once ready (still one fetch)")

        # ---- SHA mismatch -> error + file removed ----
        print("model_store 3: checksum mismatch")
        d3 = TMP / "store-bad"
        ws3 = FakeWS()
        hits3 = {"n": 0}
        bad = {"body": b"corrupted-not-matching-the-pin"}
        store3 = ModelStore(d3, broadcast=ws3.broadcast, client_factory=_factory(bad, hits3))
        store3.refresh()
        raised = False
        try:
            await store3.ensure_ready("dfine_n")
        except detector_module.ModelVerifyError:
            raised = True
        await asyncio.sleep(0.05)
        check(raised, "checksum mismatch raises ModelVerifyError")
        check(store3.state_of("dfine_n") == "error", "state moves to error")
        check(store3.model_entry("dfine_n")["detail"], "error carries a detail string")
        onnx = d3 / "dfine_n.onnx"
        check(not onnx.exists() and not list(d3.glob("*.part")),
              "bad download leaves no model or .part behind")
        check(any(m["state"] == "error" for m in ws3.messages),
              "an error model_status was broadcast")

        # ---- retryable: a fresh good download after the failure ----
        print("model_store 4: retry after failure")
        bad["body"] = FAKE_PAYLOAD
        path3 = await store3.ensure_ready("dfine_n")
        check(path3.is_file() and store3.state_of("dfine_n") == "ready",
              "a failed key retries cleanly to ready")

        # ---- delete frees the file + resets to absent ----
        print("model_store 6: delete")
        res = store3.delete("dfine_n")
        check(res == {"key": "dfine_n", "state": "absent"},
              "delete returns {key, state: absent}")
        check(not onnx.exists() and not (d3 / "dfine_n.json").exists(),
              "delete removes the .onnx and its sidecar")
        check(store3.state_of("dfine_n") == "absent", "state resets to absent after delete")

    # ---- unknown key guards ----
    d4 = TMP / "store-unknown"
    store4 = ModelStore(d4)
    for fn in ("download", "delete"):
        raised = False
        try:
            getattr(store4, fn)("nope")
        except KeyError:
            raised = True
        check(raised, f"{fn}() raises KeyError for an unknown model key")


# =====================================================================
# 7. detector integration (non-blocking reconfigure; GPU-hardfail no download)
# =====================================================================


async def _detector_cases() -> None:
    print("model_store 7: detector integration")
    import onnxruntime as ort  # noqa: PLC0415

    assert "CUDAExecutionProvider" not in ort.get_available_providers()

    d = TMP / "det-store"
    hits = {"n": 0}
    store = ModelStore(d, client_factory=_factory({"body": FAKE_PAYLOAD}, hits))
    store.refresh()

    # REQUIRE_GPU=1 on a GPU-less host: the detector hard-fails BEFORE it ever
    # touches the store -> nothing is downloaded (the "boot downloads nothing"
    # rule), and health/boot are never blocked.
    det = OnnxDetector(models_dir=d, model_key="dfine_s", confidence=0.5,
                       require_gpu=True, store=store)
    await asyncio.wait_for(det.start(), timeout=10.0)  # terminal, no retry loop
    check(det.ready is False and det.device is None,
          "GPU-required host: detector hard-fails, ready False")
    check(hits["n"] == 0, "GPU hard-fail drove NO store download")

    # Non-blocking model swap: reconfigure returns immediately and schedules a
    # background reload even though the model would have to download first.
    with _PinPatch("dfine_n"):
        await asyncio.wait_for(det.reconfigure("dfine_n", 0.4), timeout=2.0)
        check(det.model_key == "dfine_n" and det.confidence == 0.4 and det.ready is False,
              "reconfigure() switches key + confidence and returns without blocking")
        check(det._reload_task is not None and not det._reload_task.done(),
              "reconfigure() schedules a background reload task")
        # The background reload hard-fails again (GPU) without downloading.
        await asyncio.sleep(0.2)
        det._stopped = True
        if det._reload_task and not det._reload_task.done():
            det._reload_task.cancel()
    await det.stop()
    check(det.ready is False, "stop() leaves the detector not-ready")


# =====================================================================
# 8. routes (real app; REQUIRE_GPU=1 -> no real download)
# =====================================================================


def route_checks() -> None:
    print("model_store 8: /api/detection/models routes")
    # Repoint EVERY pin at the fake payload for the whole route section so any
    # download triggered by activate/PUT completes against a MockTransport
    # instead of hitting the network (the app builds its own store; we inject
    # the mock client_factory after boot).
    with _PinPatch("dfine_n"), _PinPatch("dfine_s"), _PinPatch("dfine_m"):
        _route_body()


def _route_body() -> None:
    # Pre-seed a fake dfine_n artifact so the store reports it "ready" via
    # existence-only refresh at boot — activation then needs no network.
    models_dir = Path(os.environ["DATA_DIR"]) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "dfine_n.onnx").write_bytes(FAKE_PAYLOAD)
    (models_dir / "dfine_n.json").write_text(json.dumps(
        {"url": MODELS["dfine_n"]["url"], "sha256": FAKE_SHA,
         "bytes": len(FAKE_PAYLOAD), "downloaded_at": 1.0}))

    with TestClient(app.main.app) as client:
        # Inject the mocked downloader so activate/PUT never hit the network.
        app.main.app.state.model_store._client_factory = _factory(
            {"body": FAKE_PAYLOAD}, {"n": 0}
        )
        # health is 200 immediately even though the ACTIVE model (dfine_s) is
        # absent and the detector never became ready.
        health = client.get("/api/system/health")
        check(health.status_code == 200, "health 200 immediately (active model absent)")
        check(health.json()["detector"]["ready"] is False,
              "detector not ready at boot (nothing blocked)")
        check(not (models_dir / "dfine_s.onnx").exists(),
              "GPU hard-fail boot downloaded nothing for the active model")

        token = client.post("/api/auth/login", json={"password": "test-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # auth required
        check(client.get("/api/detection/models").status_code in (401, 403),
              "GET /api/detection/models requires auth")

        r = client.get("/api/detection/models", headers=headers)
        check(r.status_code == 200, "GET /api/detection/models -> 200")
        body = r.json()
        check(set(body) == {"active", "device", "models"},
              "GET body carries {active, device, models}")
        check(body["active"] == "dfine_s" and body["device"] is None,
              "active == settings model (dfine_s); device null on hard-fail")
        keys = [m["key"] for m in body["models"]]
        check(keys == ["dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x",
                       "dfine_l_obj365"],
              "models listed fastest COCO -> heaviest COCO -> big-vocabulary")
        m0 = body["models"][0]
        check(set(m0) >= {"key", "tier", "label", "blurb", "size_bytes", "input_size",
                          "approx_map", "map_dataset", "recommended_for",
                          "vocabulary", "num_classes", "state", "progress_pct",
                          "active", "loaded", "sha_ok"},
              "each model dict carries metadata + vocabulary + live state fields")
        obj = {m["key"]: m for m in body["models"]}["dfine_l_obj365"]
        check(obj["vocabulary"] == "objects365" and obj["num_classes"] == 365,
              "obj365 model advertises its 365-class vocabulary in the list")
        by_key = {m["key"]: m for m in body["models"]}
        check(by_key["dfine_n"]["state"] == "ready" and by_key["dfine_s"]["state"] == "absent",
              "pre-seeded dfine_n is ready; absent dfine_s is absent")
        check(by_key["dfine_s"]["active"] is True and by_key["dfine_n"]["active"] is False,
              "active flag tracks the settings model")

        # 404s for unknown keys
        check(client.post("/api/detection/models/nope/download", headers=headers).status_code == 404,
              "POST download unknown key -> 404")
        check(client.post("/api/detection/models/nope/activate", headers=headers).status_code == 404,
              "POST activate unknown key -> 404")
        check(client.delete("/api/detection/models/nope", headers=headers).status_code == 404,
              "DELETE unknown key -> 404")

        # DELETE the active model -> 409
        r = client.delete("/api/detection/models/dfine_s", headers=headers)
        check(r.status_code == 409, "DELETE active model -> 409")

        # download() the ready model -> 202 no-op
        r = client.post("/api/detection/models/dfine_n/download", headers=headers)
        check(r.status_code == 202 and r.json()["state"] == "ready",
              "POST download ready model -> 202 {state: ready}")

        # activate dfine_n (already on disk -> no network) switches settings
        r = client.post("/api/detection/models/dfine_n/activate", headers=headers)
        check(r.status_code == 202, "POST activate -> 202")
        act = r.json()
        check(act["key"] == "dfine_n" and act["active"] is True and "loaded" in act,
              "activate returns {key, state, active:true, loaded}")
        settings = client.get("/api/settings", headers=headers).json()
        check(settings["detection"]["model"] == "dfine_n",
              "activate persisted settings.detection.model = dfine_n")
        body2 = client.get("/api/detection/models", headers=headers).json()
        by_key2 = {m["key"]: m for m in body2["models"]}
        check(body2["active"] == "dfine_n" and by_key2["dfine_n"]["active"] is True,
              "GET reflects the newly-active model")
        check(by_key2["dfine_s"]["active"] is False,
              "previously-active model no longer flagged active")

        # now dfine_s is not active -> DELETE frees it (it is absent already,
        # delete is still a clean no-op returning absent); dfine_n is active
        # -> 409.
        check(client.delete("/api/detection/models/dfine_n", headers=headers).status_code == 409,
              "DELETE now-active dfine_n -> 409")
        r = client.delete("/api/detection/models/dfine_s", headers=headers)
        check(r.status_code == 200 and r.json() == {"key": "dfine_s", "state": "absent"},
              "DELETE non-active model -> {state: absent}")

        # PUT /api/settings model change routes through the SAME activate path
        settings["detection"]["model"] = "dfine_m"
        r = client.put("/api/settings", headers=headers, json=settings)
        check(r.status_code == 200 and r.json()["detection"]["model"] == "dfine_m",
              "PUT /api/settings model change persists (shared activate path)")
        body3 = client.get("/api/detection/models", headers=headers).json()
        check(body3["active"] == "dfine_m", "GET reflects the PUT-driven activation")

        # /api/system/detector carries the active model's state + progress
        det = client.get("/api/system/detector", headers=headers).json()
        check("model_state" in det and "model_progress_pct" in det,
              "/api/system/detector extended with model_state + model_progress_pct")
        check(det["model"] == "dfine_m" and det["model_state"] in
              {"absent", "downloading", "verifying", "ready", "error"},
              "detector self-test reports the active model's download state")


# =====================================================================
# 9. activate/PUT WS active-flag consistency (both paths notify the OLD key)
# =====================================================================


def _assert_active_flags(ws: "FakeWS", new_key: str, old_key: str) -> None:
    new_msgs = [m for m in ws.messages if m["key"] == new_key]
    old_msgs = [m for m in ws.messages if m["key"] == old_key]
    check(any(m["active"] is True for m in new_msgs),
          f"broadcasts active:true for the newly-active model ({new_key})")
    check(any(m["active"] is False for m in old_msgs),
          f"broadcasts active:false for the previously-active model ({old_key})")


async def _activate_consistency() -> None:
    print("model_store 9: activate/PUT WS active-flag consistency")
    from app.routers.detection import activate_model  # noqa: PLC0415

    with _PinPatch("dfine_s"), _PinPatch("dfine_m"):
        # ---- direct activate path: settings still holds the OLD model ----
        d = TMP / "act-direct"
        ws = FakeWS()
        store = ModelStore(
            d, broadcast=ws.broadcast, client_factory=_factory({"body": FAKE_PAYLOAD}, {"n": 0})
        )
        settings = _FakeSettings("dfine_s")
        store.active_key_getter = lambda: settings._model
        store.loaded_key_getter = lambda: None
        store.refresh()
        state = _FakeState(store, settings, _FakeEngine(), _FakeDetector("dfine_s"))
        res = await activate_model(state, "dfine_m")
        await asyncio.sleep(0.05)  # flush fire-and-forget notify() broadcasts
        check(res["active"] is True and res["key"] == "dfine_m",
              "direct activate returns {key:dfine_m, active:true}")
        check(settings._model == "dfine_m", "direct activate persists the new model")
        _assert_active_flags(ws, new_key="dfine_m", old_key="dfine_s")

        # ---- PUT path: settings already persisted the NEW model before we
        # ---- reach activate_model; the detector still holds the OLD key.
        # ---- Regression guard: the old key's active:false MUST still fire.
        d2 = TMP / "act-put"
        ws2 = FakeWS()
        store2 = ModelStore(
            d2, broadcast=ws2.broadcast, client_factory=_factory({"body": FAKE_PAYLOAD}, {"n": 0})
        )
        settings2 = _FakeSettings("dfine_m")  # put_settings already updated this
        store2.active_key_getter = lambda: settings2._model
        store2.loaded_key_getter = lambda: None
        store2.refresh()
        state2 = _FakeState(store2, settings2, _FakeEngine(), _FakeDetector("dfine_s"))
        await activate_model(state2, "dfine_m")
        await asyncio.sleep(0.05)
        _assert_active_flags(ws2, new_key="dfine_m", old_key="dfine_s")

        # ---- re-activating the ALREADY-active model: no spurious old-key frame
        d3 = TMP / "act-same"
        ws3 = FakeWS()
        store3 = ModelStore(
            d3, broadcast=ws3.broadcast, client_factory=_factory({"body": FAKE_PAYLOAD}, {"n": 0})
        )
        settings3 = _FakeSettings("dfine_m")
        store3.active_key_getter = lambda: settings3._model
        store3.loaded_key_getter = lambda: None
        store3.refresh()
        state3 = _FakeState(store3, settings3, _FakeEngine(), _FakeDetector("dfine_m"))
        await activate_model(state3, "dfine_m")
        await asyncio.sleep(0.05)
        others = [m for m in ws3.messages if m["key"] != "dfine_m"]
        check(not others, "re-activating the active model notifies no other key")


def main() -> None:
    metadata_checks()
    asyncio.run(_store_cases())
    asyncio.run(_detector_cases())
    asyncio.run(_activate_consistency())
    route_checks()
    print(f"\nALL {PASS} CHECKS PASSED (model store + detection API)")


if __name__ == "__main__":
    main()

"""Detection-model management API (in-app tier download + activate).

Sits on top of the ModelStore (the one downloader) + the detector. Lets the
UI pick a tier (Lightweight/Balanced/Heavy -> dfine_n/s/m), download it on
demand with visible progress, and activate it live — without ever blocking
app boot or the health endpoint.

Routes (admin only — model management is an admin capability):
- GET    /api/detection/models              -> {active, device, models:[...]}
- GET    /api/detection/labels[?model=]     -> {model, vocabulary, count, labels:[...]}
- POST   /api/detection/models/{key}/download  -> 202 {key, state, progress_pct}
- POST   /api/detection/models/{key}/activate  -> 202 {key, state, active, loaded}
- DELETE /api/detection/models/{key}           -> {key, state:"absent"} (409 if active)

Model changes go through ONE activate path shared with PUT /api/settings
(``activate_model`` below): persist settings.detection.model, reconfigure the
detector (non-blocking), and kick the store's background download.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..auth import require_admin
from ..native.coco_labels import selectable_labels, vocabulary_name
from ..native.detector import model_labelmap
from ..native.model_store import ModelStore

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/detection", tags=["detection"], dependencies=[Depends(require_admin)]
)


async def activate_model(state: Any, key: str) -> dict[str, Any]:
    """Shared activate path (POST .../activate AND PUT /api/settings model
    changes both call this). Persists settings.detection.model, reconfigures
    the detector in the background, and starts the store download if the model
    isn't present yet. Returns the 202 body ``{key, state, active, loaded}``.

    Callers must have validated ``key`` is a known model.
    """
    store: ModelStore = state.model_store
    settings = state.settings
    # Previous active key comes from the detector's in-memory model_key, NOT
    # from settings: the PUT /api/settings path persists the new model BEFORE
    # calling this, so reading settings here would already show the new key and
    # the old key's active:false WS frame would never fire. The detector is only
    # reconfigured below (engine.reload), so its model_key still holds the old
    # value on BOTH entry paths — keeping activate and PUT perfectly consistent.
    prev_active = state.detector.model_key
    current = settings.get()
    if current["detection"].get("model") != key:
        current["detection"]["model"] = key
        await settings.update(current)
    # Reconfigure the detector to match settings (non-blocking model swap) and
    # reconcile the engine's per-camera state.
    await state.engine.reload()
    # Ensure a background download is running / has run for the new model.
    store.download(key)
    # Reflect the active-flag change on the WS for both the old and new keys.
    store.notify(key)
    if prev_active and prev_active != key and ModelStore.is_known(prev_active):
        store.notify(prev_active)
    loaded = bool(state.detector.ready and state.detector.model_key == key)
    return {"key": key, "state": store.state_of(key), "active": True, "loaded": loaded}


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    state = request.app.state
    payload = state.model_store.models_payload()
    payload["device"] = state.detector.device  # "cuda" | "cpu" | null
    return payload


@router.get("/labels")
async def list_labels(request: Request, model: str | None = None) -> dict[str, Any]:
    """Ordered class-vocabulary for a model (defaults to the ACTIVE model).

    Lets the per-camera object picker list exactly the labels the active model
    can detect — COCO-80 or the Objects365 365-class space (the Objects365
    id-0 ``none`` background placeholder is not offered). ``model`` may name any
    known model key; unknown keys 404. Shape matches the frontend
    ``ActiveLabelsResponse``: ``{model, vocabulary, count, labels}``.
    """
    state = request.app.state
    key = model or state.settings.detection.get("model") or state.detector.model_key
    if not ModelStore.is_known(key):
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    labelmap = model_labelmap(key)
    labels = selectable_labels(labelmap)
    return {
        "model": key,
        "vocabulary": vocabulary_name(labelmap),
        "count": len(labels),
        "labels": list(labels),
    }


@router.post("/models/{key}/download", status_code=status.HTTP_202_ACCEPTED)
async def download_model(key: str, request: Request) -> dict[str, Any]:
    store: ModelStore = request.app.state.model_store
    if not ModelStore.is_known(key):
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    return store.download(key)


@router.post("/models/{key}/activate", status_code=status.HTTP_202_ACCEPTED)
async def activate(key: str, request: Request) -> dict[str, Any]:
    if not ModelStore.is_known(key):
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    return await activate_model(request.app.state, key)


@router.delete("/models/{key}")
async def delete_model(key: str, request: Request) -> dict[str, Any]:
    state = request.app.state
    store: ModelStore = state.model_store
    if not ModelStore.is_known(key):
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    active = state.settings.detection.get("model")
    if key == active:
        raise HTTPException(status_code=409, detail="cannot delete the active model")
    return store.delete(key)

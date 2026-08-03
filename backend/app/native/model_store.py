"""ModelStore — the ONE detector-model downloader + per-key state machine.

Sits ON TOP of ``native/detector.py``'s pinned ``MODELS`` table and its
``ensure_model`` download/verify primitive (revision-pinned URL + SHA-256).
The store owns a single source of truth for per-key download state so both
the detector and the ``/api/detection/models`` API read the same thing.

Tiers (static metadata table below — single source of truth):
  - Lightweight    -> dfine_n         (fastest, lowest latency; ~nano)
  - Balanced       -> dfine_s         (default; best speed/accuracy tradeoff)
  - Heavy          -> dfine_m         (higher accuracy, higher latency)
  - Accurate       -> dfine_l         (COCO 54.0 mAP; needs GPU headroom)
  - Maximum        -> dfine_x         (COCO 55.8 mAP; heaviest COCO tier)
  - Big Vocabulary -> dfine_l_obj365  (Objects365 365-class vocabulary)

Each tier also advertises its class ``vocabulary`` (short machine name
``"coco"``/``"objects365"``) and ``num_classes`` (80/365) so the picker can show
what a model can detect; the ordered label list itself is served by the router's
``GET /api/detection/labels`` (lean — not inlined per model).

Per-key state machine::

    absent -> downloading (progress_pct 0-100 + bytes) -> verifying -> ready
                                                         \\-> error (+ detail)

Guarantees:
- **Non-blocking:** ``download(key)`` starts a background task and returns the
  current state immediately; the event loop / health endpoint never block.
- **Idempotent:** a second ``download``/``ensure_ready`` while one is in flight
  joins the SAME task (one downloader per key).
- **Retryable:** a failed/partial download leaves nothing behind (``ensure_model``
  cleans the ``.part`` file) and moves the key to ``error``; a subsequent call
  retries from scratch. A SHA-256 mismatch removes the bad file and reports
  ``error``.

Progress is broadcast on the existing ``WSManager`` as ``{type:"model_status",
...}`` on every state change; progress-only updates are throttled to ~1/sec.
GET stays authoritative for polling fallback.

``import onnxruntime`` never happens here (import hygiene) — the store only
touches the pin table + the streaming download; the ORT session lives in the
detector.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from .coco_labels import num_classes, vocabulary_name
from .detector import (
    DEFAULT_MODEL,
    INPUT_SIZE,
    MODELS,
    ModelVerifyError,
    ensure_model,
    model_labelmap,
    model_path,
    sha256_file,
)

if TYPE_CHECKING:  # pragma: no cover
    import httpx

log = logging.getLogger(__name__)

# Per-key states (the API + WS surface these verbatim).
ABSENT = "absent"
DOWNLOADING = "downloading"
VERIFYING = "verifying"
READY = "ready"
ERROR = "error"

_PROGRESS_BROADCAST_INTERVAL_S = 1.0
_DOWNLOAD_TIMEOUT_S = 120.0

# ---------------------------------------------------------------------------
# Tier metadata — SINGLE SOURCE OF TRUTH. Keys MUST map to detector MODELS.
# size_bytes comes from the pin (MODELS[key]["bytes"]); input_size is the
# detector's INPUT_SIZE; approx_map is COCO mAP from the design-doc table.
# ---------------------------------------------------------------------------
# ``approx_map`` is COCO AP(val) for the coco tiers; for the Objects365 tier it
# is Objects365 AP(val) (a DIFFERENT benchmark — the two are not comparable, so
# ``map_dataset`` names which). All figures are from the official D-FINE model
# zoo (Peterande/D-FINE, Apache-2.0).
MODEL_TIERS: dict[str, dict[str, Any]] = {
    "dfine_n": {
        "tier": "lightweight",
        "label": "Lightweight",
        "blurb": "Fastest, lowest latency — quick to load, good for weak GPUs.",
        "approx_map": 42.8,
        "map_dataset": "COCO",
        "recommended_for": "Weak GPUs, CPU fallback, fastest startup",
    },
    "dfine_s": {
        "tier": "balanced",
        "label": "Balanced",
        "blurb": "The default — best speed/accuracy tradeoff for most setups.",
        "approx_map": 48.5,
        "map_dataset": "COCO",
        "recommended_for": "Most setups (default)",
    },
    "dfine_m": {
        "tier": "heavy",
        "label": "Heavy",
        "blurb": "Higher accuracy, higher latency — for GPUs with headroom.",
        "approx_map": 52.3,
        "map_dataset": "COCO",
        "recommended_for": "Powerful GPUs, more accuracy",
    },
    "dfine_l": {
        "tier": "accurate",
        "label": "Accurate",
        "blurb": "High accuracy (COCO 54.0 mAP) — a solid GPU with headroom.",
        "approx_map": 54.0,
        "map_dataset": "COCO",
        "recommended_for": "Strong GPUs wanting more accuracy than Heavy",
    },
    "dfine_x": {
        "tier": "maximum",
        "label": "Maximum",
        "blurb": "Most accurate COCO tier (55.8 mAP), heaviest — GPU only.",
        "approx_map": 55.8,
        "map_dataset": "COCO",
        "recommended_for": "Powerful GPUs, maximum COCO accuracy",
    },
    "dfine_l_obj365": {
        "tier": "objects365",
        "label": "Big Vocabulary",
        "blurb": "Detects 365 object types (Objects365) — far more than COCO's "
                 "80. Same speed as Accurate; a strong GPU is recommended.",
        "approx_map": 44.7,
        "map_dataset": "Objects365",
        "recommended_for": "Detecting object types outside the COCO-80 set",
    },
}

# Display order: fastest COCO -> heaviest COCO -> big-vocabulary.
TIER_ORDER: list[str] = [
    "dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x", "dfine_l_obj365",
]


def tier_metadata(key: str) -> dict[str, Any]:
    """Static tier metadata for ``key`` (no live download state)."""
    meta = MODEL_TIERS[key]
    pin = MODELS[key]
    labelmap = model_labelmap(key)
    return {
        "key": key,
        "tier": meta["tier"],
        "label": meta["label"],
        "blurb": meta["blurb"],
        "size_bytes": pin["bytes"],
        "input_size": INPUT_SIZE,
        "approx_map": meta["approx_map"],
        "map_dataset": meta["map_dataset"],
        "recommended_for": meta["recommended_for"],
        # Frontend contract (DetectionModelInfo): short machine vocabulary name
        # ("coco"/"objects365") + user-selectable class count (80/365).
        "vocabulary": vocabulary_name(labelmap),
        "num_classes": num_classes(labelmap),
    }


@dataclass
class _KeyState:
    state: str = ABSENT
    progress_pct: int = 0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    detail: Optional[str] = None
    sha_ok: bool = False


BroadcastFn = Callable[[dict[str, Any]], Awaitable[None]]
ClientFactory = Callable[[], "httpx.AsyncClient"]


class ModelStore:
    """Owns per-key download state + the single downloader.

    ``broadcast`` is the WSManager's async ``broadcast`` (optional; tests may
    pass a fake). ``client_factory`` lets tests inject an httpx client backed
    by a MockTransport — it is called once per download and the store closes
    the returned client.
    """

    def __init__(
        self,
        models_dir: Path,
        broadcast: Optional[BroadcastFn] = None,
        client_factory: Optional[ClientFactory] = None,
    ):
        self._models_dir = Path(models_dir)
        self._broadcast = broadcast
        self._client_factory = client_factory
        self._states: dict[str, _KeyState] = {
            key: _KeyState(total_bytes=MODELS[key]["bytes"]) for key in TIER_ORDER
        }
        self._tasks: dict[str, asyncio.Task] = {}
        self._last_progress_ts: dict[str, float] = {}
        # Late-bound resolvers so the WS/API payloads can flag the active
        # (settings.detection.model) and loaded-in-detector keys without the
        # store depending on settings/detector.
        self.active_key_getter: Callable[[], str] = lambda: DEFAULT_MODEL
        self.loaded_key_getter: Callable[[], Optional[str]] = lambda: None

    # ---------- known keys ----------

    @staticmethod
    def is_known(key: str) -> bool:
        return key in MODELS

    def _active_key(self) -> str:
        try:
            return self.active_key_getter() or DEFAULT_MODEL
        except Exception:  # noqa: BLE001 — resolver must never break a broadcast
            return DEFAULT_MODEL

    def _loaded_key(self) -> Optional[str]:
        try:
            return self.loaded_key_getter()
        except Exception:  # noqa: BLE001
            return None

    # ---------- disk reconciliation (boot; existence-only, no download) ----------

    def refresh(self) -> None:
        """Set initial per-key states from disk (cheap; existence + sidecar
        only — never hashes, never downloads). In-flight keys are left alone."""
        for key in TIER_ORDER:
            st = self._states[key]
            if st.state in (DOWNLOADING, VERIFYING):
                continue
            if model_path(self._models_dir, key).is_file():
                st.state = READY
                st.progress_pct = 100
                st.detail = None
                st.sha_ok = self._sidecar_ok(key)
            else:
                st.state = ABSENT
                st.progress_pct = 0
                st.detail = None
                st.sha_ok = False

    def _sidecar_ok(self, key: str) -> bool:
        sidecar = self._models_dir / f"{key}.json"
        if not sidecar.is_file():
            return False
        try:
            import json

            data = json.loads(sidecar.read_text())
            return data.get("sha256") == MODELS[key]["sha256"]
        except Exception:  # noqa: BLE001
            return False

    # ---------- download control ----------

    def download(self, key: str) -> dict[str, Any]:
        """Start (or no-op) a background download of ``key``. Returns the
        current ``{key, state, progress_pct}`` immediately. Raises ``KeyError``
        for an unknown key. A download that is already in-flight or a model
        that is already ready is a no-op."""
        if key not in MODELS:
            raise KeyError(key)
        task = self._tasks.get(key)
        in_flight = task is not None and not task.done()
        already = (
            self._states[key].state == READY
            and model_path(self._models_dir, key).is_file()
        )
        if not in_flight and not already:
            self._spawn(key)
        return self.progress_dict(key)

    async def ensure_ready(self, key: str) -> Path:
        """Ensure ``key`` is on disk + SHA-256 verified, downloading if needed;
        return the verified path. Joins an in-flight download rather than
        starting a second one. Raises on download/verify failure (the detector
        bootstrap owns the retry policy)."""
        if key not in MODELS:
            raise KeyError(key)
        return await self._spawn(key)

    def _spawn(self, key: str) -> asyncio.Task:
        task = self._tasks.get(key)
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(self._worker(key), name=f"model-download-{key}")
        self._tasks[key] = task
        return task

    async def _worker(self, key: str) -> Path:
        st = self._states[key]
        pin = MODELS[key]
        path = model_path(self._models_dir, key)
        client = self._make_client()
        try:
            if path.is_file():
                # Existing file: verify (ensure_model re-hashes and re-downloads
                # on mismatch). Show a verifying state up front.
                self._set_state(key, VERIFYING, progress=100)
            else:
                self._set_state(key, DOWNLOADING, progress=0)
                st.downloaded_bytes = 0

            def _progress(done: int, total: int) -> None:
                st.downloaded_bytes = done
                st.total_bytes = total or pin["bytes"]
                pct = int(done * 100 / st.total_bytes) if st.total_bytes else 0
                pct = max(0, min(pct, 100))
                if pct >= 100:
                    self._set_state(key, VERIFYING, progress=100)
                else:
                    self._set_state(key, DOWNLOADING, progress=pct, throttle=True)

            result = await ensure_model(
                self._models_dir, key, client=client, progress=_progress
            )
            st.sha_ok = True
            self._set_state(key, READY, progress=100)
            return result
        except asyncio.CancelledError:
            raise
        except ModelVerifyError as exc:
            log.error("model %s failed SHA-256 verification: %s", key, exc)
            st.sha_ok = False
            self._set_state(key, ERROR, detail="checksum verification failed")
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced as error state, never crashes
            log.warning("model %s download failed: %s", key, exc)
            st.sha_ok = False
            self._set_state(key, ERROR, detail=_short_detail(exc))
            raise
        finally:
            with _suppress():
                await client.aclose()
            # Drop the finished task so a later retry can start fresh.
            if self._tasks.get(key) is asyncio.current_task():
                self._tasks.pop(key, None)

    def _make_client(self) -> "httpx.AsyncClient":
        if self._client_factory is not None:
            return self._client_factory()
        import httpx

        return httpx.AsyncClient(
            timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT_S), follow_redirects=True
        )

    # ---------- delete ----------

    def delete(self, key: str) -> dict[str, Any]:
        """Delete the downloaded ``.onnx`` + ``.json`` sidecar and reset the
        key to ``absent``. (Caller enforces the active-model 409.) Raises
        ``KeyError`` for an unknown key."""
        if key not in MODELS:
            raise KeyError(key)
        task = self._tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
        with _suppress():
            model_path(self._models_dir, key).unlink(missing_ok=True)
        with _suppress():
            (self._models_dir / f"{key}.json").unlink(missing_ok=True)
        with _suppress():
            model_path(self._models_dir, key).with_suffix(".onnx.part").unlink(missing_ok=True)
        st = self._states[key]
        st.state = ABSENT
        st.progress_pct = 0
        st.downloaded_bytes = 0
        st.detail = None
        st.sha_ok = False
        self._emit(key)
        return {"key": key, "state": st.state}

    # ---------- state broadcasting ----------

    def _set_state(
        self,
        key: str,
        state: str,
        *,
        progress: Optional[int] = None,
        detail: Optional[str] = None,
        throttle: bool = False,
    ) -> None:
        st = self._states[key]
        prev_state = st.state
        st.state = state
        if progress is not None:
            st.progress_pct = int(progress)
        st.detail = detail if state == ERROR else None
        state_changed = prev_state != state
        if throttle and not state_changed:
            now = time.monotonic()
            if now - self._last_progress_ts.get(key, 0.0) < _PROGRESS_BROADCAST_INTERVAL_S:
                return  # throttled: memory is current, skip the broadcast
        self._last_progress_ts[key] = time.monotonic()
        self._emit(key)

    def notify(self, key: str) -> None:
        """Force a ``model_status`` broadcast for ``key`` (used when active/
        loaded flags change without a state transition — e.g. activation)."""
        if key in self._states:
            self._emit(key)

    def _emit(self, key: str) -> None:
        if self._broadcast is None:
            return
        msg = self.ws_message(key)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (sync context) — nothing to broadcast to
            return
        loop.create_task(self._safe_broadcast(msg))

    async def _safe_broadcast(self, msg: dict[str, Any]) -> None:
        try:
            await self._broadcast(msg)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 — a dead WS client must not break downloads
            log.debug("model_status broadcast failed", exc_info=True)

    # ---------- payloads ----------

    def ws_message(self, key: str) -> dict[str, Any]:
        """The ``model_status`` WS frame for ``key``."""
        st = self._states[key]
        return {
            "type": "model_status",
            "key": key,
            "tier": MODEL_TIERS[key]["tier"],
            "state": st.state,
            "progress_pct": st.progress_pct,
            "active": key == self._active_key(),
            "loaded": key == self._loaded_key(),
        }

    def progress_dict(self, key: str) -> dict[str, Any]:
        """Compact ``{key, state, progress_pct}`` for the download/activate 202s."""
        st = self._states[key]
        return {"key": key, "state": st.state, "progress_pct": st.progress_pct}

    def state_of(self, key: str) -> str:
        return self._states[key].state

    def model_entry(self, key: str) -> dict[str, Any]:
        """Full per-model dict for GET /api/detection/models (metadata + live
        state + active/loaded/sha_ok/detail)."""
        st = self._states[key]
        active = self._active_key()
        loaded = self._loaded_key()
        entry = tier_metadata(key)
        entry.update(
            {
                "state": st.state,
                "progress_pct": st.progress_pct,
                "active": key == active,
                "loaded": key == loaded,
                "sha_ok": st.sha_ok,
                "detail": st.detail,
            }
        )
        return entry

    def models_payload(self) -> dict[str, Any]:
        """The ``{active, models:[...]}`` body for GET /api/detection/models
        (device is filled in by the router from the detector)."""
        return {
            "active": self._active_key(),
            "models": [self.model_entry(key) for key in TIER_ORDER],
        }


def _short_detail(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:200]


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib here."""

    def __enter__(self) -> "_suppress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, Exception)

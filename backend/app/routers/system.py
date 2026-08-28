"""System routes.

GET /api/system/health — no auth (docker healthcheck + status page).
Returns 200 as soon as the app itself is up; the detector/go2rtc fields
report subsystem state without ever failing the check (compose gates the
go2rtc container's start on THIS endpoint).

GET /api/system/detector — auth; full detector self-test: model/provider
state plus per-camera ingest and recorder stats.

POST /api/system/restart — ADMIN; restarts the backend process (see
request_restart for exactly what that means and why it is a SIGTERM).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time

from fastapi import APIRouter, Depends, Query, Request

from ..auth import require_admin, require_auth
from ..config import APP_VERSION
from ..native.model_store import ModelStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

# Grace period between acking the request and signalling ourselves, so the HTTP
# response is actually flushed to the client. Without it the caller sees a
# connection reset instead of a confirmation and cannot tell success from a
# crash.
_RESTART_DELAY_S = 0.75

# Hard ceiling on graceful teardown. Past this the process is force-exited.
#
# WHY THIS EXISTS AT ALL — the fact that makes it load-bearing rather than
# belt-and-braces: we SIGTERM OURSELVES, so Docker is a passive observer. Its
# `stop_grace_period` and the 10 s SIGKILL backstop only run inside `docker
# stop`/`restart`/`down`; they do NOT apply here. And `restart: unless-stopped`
# triggers on container EXIT, never on a red healthcheck. So if teardown blocks
# there is NO external backstop of any kind: the container stays "up" forever
# with a dead server inside it, recording already stopped, and the nightly
# scheduler's once-per-day guard means it will not even retry. The deadline has
# to live in-process because nothing outside the process is watching a clock.
#
# Sized above the observed teardown (17 sequential steps, the slowest being
# per-camera ffmpeg termination) so a healthy shutdown always wins the race and
# the force path stays exceptional.
#
# WHAT THIS DEADLINE DOES **NOT** COVER, and why that matters: uvicorn's SIGTERM
# handling drains in-flight connections and request tasks BEFORE it calls the
# lifespan shutdown. That drain is a separate, earlier phase, and it is bounded
# by --timeout-graceful-shutdown (set in the Dockerfile CMD), NOT by this. Arming
# this watchdog before the signal — as an earlier version did — pointed a
# teardown-sized budget at drain + teardown, so a client slowly downloading a
# 30-minute export could burn the whole budget before teardown even began, and
# the force-exit would then orphan every ffmpeg child mid-segment: strictly worse
# than the hang it was meant to prevent. It is therefore armed from INSIDE the
# lifespan `finally` (main.py), where the clock measures what it was sized for.
_TEARDOWN_DEADLINE_S = 25.0


def arm_teardown_watchdog(reason: str, deadline_s: float = _TEARDOWN_DEADLINE_S) -> None:
    """Guarantee the process actually exits within `deadline_s` of TEARDOWN
    STARTING. Call from the lifespan shutdown path, not before the signal.

    A DAEMON THREAD, deliberately, on both counts:

    - a *thread* and not an asyncio task, because the failure this guards
      against happens AFTER the event loop is gone. Cancelling a task suspended
      on ``asyncio.to_thread`` cancels the future but cannot cancel the OS
      thread, which keeps running. Those threads are non-daemon, so once the
      loop closes, ``concurrent.futures.thread._python_exit`` joins them at
      interpreter exit with NO timeout. Teardown can therefore log "backend shut
      down cleanly" and the process still never exits. No coroutine can fire
      after that point; a thread can.
    - a *daemon* thread, so the watchdog itself is never the thing holding exit
      open on the happy path.

    ``os._exit`` and not ``sys.exit``: this runs on a non-main thread, where
    SystemExit would only unwind this thread and leave the wedge in place. It
    also skips atexit handlers, which is the entire point — the atexit join is
    the wedge.
    """

    def _watch() -> None:
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            time.sleep(0.25)
        # Reached only if graceful teardown never completed. Exit 0, not a
        # failure code: this IS the intended restart, and a non-zero status on
        # an `on-failure` policy would read as a crash loop.
        log.error(
            "teardown exceeded %.0fs (%s) — force-exiting so the restart policy fires. "
            "This watchdog is armed at the START of the lifespan teardown, so the block "
            "is in the teardown chain itself if 'backend shut down cleanly' is absent "
            "above, or in a non-daemon worker thread at interpreter exit if it is present.",
            deadline_s, reason,
        )
        os._exit(0)

    threading.Thread(target=_watch, name="teardown-watchdog", daemon=True).start()


async def request_restart(reason: str, delay_s: float = _RESTART_DELAY_S) -> None:
    """Restart the backend by SIGTERM-ing ourselves after `delay_s`.

    WHY SIGTERM AND NOT os._exit: uvicorn traps SIGTERM and runs the app's
    shutdown hooks, so the recorder's ffmpeg children are terminated, the engine
    stops, and go2rtc/DB handles close cleanly. A hard exit would orphan those
    ffmpeg processes and leave a half-written segment behind on every restart.
    That preference is unchanged — the teardown watchdog does not replace the
    graceful path, it only puts a floor under it. Graceful still wins every
    normal restart; the force path is what stops a blocked teardown from
    becoming a permanently dead NVR (see _TEARDOWN_DEADLINE_S). The watchdog is
    NOT armed here: it belongs at the start of the lifespan teardown, so that
    uvicorn's earlier connection-drain phase does not eat its budget.

    WHAT BRINGS US BACK: nothing here. The process exits and the container's
    `restart: unless-stopped` policy (docker-compose.yml) starts it again —
    roughly a 15 s gap in recording and live view. Run OUTSIDE a supervisor
    (a bare dev shell) this is a one-way shutdown, which is why the UI says so.

    Scheduled as a detached task so the caller can return its response first.
    """

    async def _fire() -> None:
        await asyncio.sleep(delay_s)
        log.warning("restarting backend process now (%s)", reason)
        os.kill(os.getpid(), signal.SIGTERM)

    log.warning("backend restart requested (%s) — SIGTERM in %.2fs", reason, delay_s)
    asyncio.create_task(_fire(), name="system-restart")


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    detector = state.detector
    return {
        "status": "ok",
        "version": APP_VERSION,
        "detector": {
            "kind": getattr(detector, "kind", "onnx"),  # "onnx"
            "ready": detector.ready,
            "device": detector.device,   # "cuda" | "cpu" | null
            "model": detector.model_key,
        },
        "go2rtc": await state.go2rtc.is_healthy(),
        "cameras_online": state.prober.online_count(),
    }


@router.get("/camera-health", dependencies=[Depends(require_auth)])
async def camera_health(
    request: Request,
    hours: float = Query(default=24.0, ge=0.25, le=720.0),
) -> dict:
    """Per-camera reachability over the last ``hours``:
    ``{window:{since,until,hours}, cameras:[{camera, uptime_pct, online (now),
    down_count, down_seconds, downs:[{start,end,seconds}]}]}``.

    Uptime is RTSP-port reachability (what the prober measures), stated as such
    in the UI. Any-auth: a viewer sees the same health an admin does. Cameras
    with no history in the window still appear (uptime null = never observed).
    """
    state = request.app.state
    now = time.time()
    since = now - hours * 3600.0
    intervals = await state.db.camera_health_intervals(since, now)
    live = getattr(state, "prober", None)

    names = [c["name"] for c in await state.db.list_cameras()]
    by_cam: dict[str, list[dict]] = {n: [] for n in names}
    for iv in intervals:
        by_cam.setdefault(iv["camera"], []).append(iv)

    window = max(now - since, 1e-9)
    cameras = []
    for name in names:
        ivs = sorted(by_cam.get(name, []), key=lambda x: x["start"])
        up = sum(i["end"] - i["start"] for i in ivs if i["online"])
        downs = [
            {"start": i["start"], "end": i["end"],
             "seconds": round(i["end"] - i["start"], 1)}
            for i in ivs if not i["online"]
        ]
        observed = sum(i["end"] - i["start"] for i in ivs)
        cameras.append({
            "camera": name,
            # null when the window holds no history for this camera at all.
            "uptime_pct": (round(100.0 * up / observed, 1) if observed > 0 else None),
            "online": (live.is_online(name) if live is not None else None),
            "down_count": len(downs),
            "down_seconds": round(sum(d["seconds"] for d in downs), 1),
            "downs": downs[-20:],  # cap the payload
        })
    return {
        "window": {"since": since, "until": now, "hours": hours},
        "cameras": cameras,
    }


@router.get("/detector", dependencies=[Depends(require_admin)])
async def detector_status(request: Request) -> dict:
    """Detector self-test: {ready, device, model, model_sha_ok,
    last_inference_ms, consecutive_failures, needs_reinit, last_reinit_age_s,
    model_state, model_progress_pct, per_camera:
    [{name, ingest_ok, fps, last_frame_age_s, stalled, respawns}]}.

    per_camera covers detect-enabled cameras and now carries per-source ingest
    health (``stalled`` + ``respawns``) so the user can see WHICH camera is
    starved; the top-level ``consecutive_failures``/``last_reinit_age_s`` show
    that the detector's self-heal reinit fired. model_state/model_progress_pct
    surface the active model's download/load progress."""
    state = request.app.state
    status = state.detector.status()
    store = state.model_store
    active = status["model"]
    if ModelStore.is_known(active):
        active_state = store.progress_dict(active)
        status["model_state"] = active_state["state"]
        status["model_progress_pct"] = active_state["progress_pct"]
    else:
        status["model_state"] = None
        status["model_progress_pct"] = 0
    status["per_camera"] = _per_camera_health(state)
    # What is actually ENCODING video, which is a different GPU question from
    # what is running inference. On an AMD/Intel box they have different
    # answers — D-FINE runs on CUDA only, so an iGPU can never do detection,
    # but it can absolutely do the HEVC->H.264 transcode. Reporting only the
    # detector's device left "is my iGPU doing anything?" unanswerable.
    status["transcode"] = await _transcode_status(state)
    return status


async def _transcode_status(state) -> dict:
    """Transcoder.status(), or a shaped stand-in. Never raises and never 500s
    the health endpoint over a transcoder that is missing or wedged: a status
    call that dies is worse than one that says it does not know."""
    recorder = getattr(state, "recorder", None)
    transcoder = getattr(recorder, "transcoder", None) if recorder is not None else None
    if transcoder is None:
        return {
            "enabled": False, "encoder": None, "encoder_label": "no transcoder",
            "hardware": False, "vaapi_device": None, "nvidia": False,
            "failed": [], "runs": {},
        }
    try:
        return await transcoder.status()
    except Exception:  # noqa: BLE001
        log.exception("transcode status probe failed")
        return {
            "enabled": True, "encoder": None, "encoder_label": "probe failed",
            "hardware": False, "vaapi_device": None, "nvidia": False,
            "failed": [], "runs": {},
        }


def _per_camera_health(state) -> list:
    """engine.camera_stats() (fps/last_frame_age/ingest_ok) merged with the
    ingest manager's per-source health (stalled/respawns). Never raises — a
    missing/not-yet-started ingest manager degrades to stalled with 0 respawns
    so the endpoint always answers."""
    base = state.engine.camera_stats()
    ingest = getattr(state.engine, "_ingest", None)
    source_stats = ingest.source_stats() if ingest is not None else {}
    ai_events = getattr(state, "ai_events", None)
    for cam in base:
        src = source_stats.get(cam["name"])
        if src is not None:
            cam["stalled"] = bool(src["stalled"])
            cam["respawns"] = int(src["respawns"])
        else:
            # No running ffmpeg source (e.g. ffmpeg missing on this host, or a
            # camera_ai_only camera which runs no server inference).
            cam["stalled"] = True
            cam["respawns"] = 0
        # Camera-AI gate visibility: the live "AI active" state (and its detail
        # block) when a watcher runs for this camera; null/false otherwise.
        ai_status = ai_events.status(cam["name"]) if ai_events is not None else None
        cam["ai_active"] = bool(ai_status and ai_status.get("ai_active"))
        cam["ai_events"] = ai_status
    return base


@router.post("/restart", status_code=202, dependencies=[Depends(require_admin)])
async def restart_backend(request: Request) -> dict:
    """Restart the backend process. ADMIN ONLY.

    Deliberately 202 (Accepted), not 204: the restart has been *scheduled*, not
    completed — the process is still alive when this response is written, and
    the caller should expect the API to drop for ~15 s. Returning 200/204 would
    imply a finished action.
    """
    user = getattr(request.state, "user", None) or "admin"
    await request_restart(f"manual restart by {user}")
    return {"restarting": True, "detail": "Backend is restarting; the API will be unavailable briefly."}


# ---------------------------------------------------------------------------
# Public endpoint discovery (opt-in, for the app's "direct address" backup)
# ---------------------------------------------------------------------------
#
# THE NVR ASKS, NOT THE PHONE. A phone can only learn this network's public IP
# while it is ON this network — so a phone-side lookup goes stale exactly when
# it matters, the moment you are away and the IP has changed. The NVR is always
# here, and the app can reach it over the primary URL from anywhere, so asking
# the NVR keeps working through an IP change.
#
# OUTBOUND, AND ONLY WHEN ASKED. Determining your own public IP requires asking
# something outside the network. This endpoint is the ONLY place this app talks
# to a third party, it is never called on a schedule, and it caches, so with the
# app toggle off it makes no outbound requests at all.

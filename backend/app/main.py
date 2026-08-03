"""Vigilume NVR backend — FastAPI app assembly + boot sequence.

Standalone architecture (no external NVR, no message broker): the backend owns detection
(native/detector + engine), 24/7 recording (native/recorder), live view via
a Vigilume-managed go2rtc container (native/streams), and the event/notify
pipeline.

Boot: init dirs/secrets/db -> seed cameras from CAM{1..3}_* env (first boot
only) -> load settings -> wire detector/engine/recorder/media/pipeline ->
write go2rtc config (go2rtc's compose service waits on our healthcheck) ->
start background tasks. Everything reconnects; no peer outage crashes the
app — the compose healthcheck (GET /api/system/health) passes as soon as
the app is up, and a failed detector bootstrap only shows up as
``detector.ready: false``.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from .amcrest.ai_events import AiEventListener
from .amcrest.doorbell import DoorbellManager
from .amcrest.features import static_capabilities
from .amcrest.ir_reassert import IrReasserter
from .amcrest.lens_mask import LensMaskCleaner
from .amcrest.speaker_probe import SpeakerProbeManager
from .amcrest.time_sync import TimeSyncManager
from .auth import AuthService, load_or_create_secrets, ws_token
from .config import APP_VERSION, BACKEND_TO_DETECTOR, Config, DEFAULT_DETECT_FPS, DEFAULT_DETECT_OBJECTS, MODEL_DETECT_DEFAULTS
from .db import Database
from .events_pipeline import DOORBELL_MAX_S, EventsPipeline
from .integrations.mqtt_ha import MqttPublisher
from .native.detector import DEFAULT_MODEL, build_detector
from .native.engine import DetectionEngine
from .native.model_store import ModelStore
from .native.media import NativeMediaProvider
from .native.recorder import Recorder
from .native.spotlight import SpotlightController
from .native import streams
from .native.streams import Go2rtcManager
from .notify.apns import ApnsService
from .notify.ntfy import NtfyService
from .camera_health import CameraHealthTracker
from .notify.push import PushService
from .settings_store import SettingsStore
from .ws import WSManager
from .routers import auth as auth_router
from .routers import cameras as cameras_router
from .routers import detection as detection_router
from .routers import events as events_router
from .routers import suppressions as suppressions_router
from .routers import integrations as integrations_router
from .routers import groups as groups_router
from .routers import notifications as notifications_router
from .routers import recordings as recordings_router
from .routers import privacy as privacy_router
from .routers import settings as settings_router
from .routers import system as system_router
from .routers.system import arm_teardown_watchdog, request_restart
from .routers import talk as talk_router
from .routers import users as users_router
from . import privacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app.main")

_PROBE_INTERVAL_S = 45.0
_PRUNE_INTERVAL_S = 12 * 3600.0


class CameraProber:
    """Tracks camera reachability via cheap TCP probes to the RTSP port."""

    def __init__(self, db: Database, ws: WSManager):
        self._db = db
        self._ws = ws
        self._online: dict[str, bool] = {}
        self._wakeup = asyncio.Event()
        # Optional MQTT/Home Assistant publisher; set via set_mqtt after the
        # publisher is constructed. None-safe: connectivity publishing is
        # skipped when unset/disabled.
        self._mqtt: Optional[MqttPublisher] = None
        # Optional automatic clock corrector; set via set_time_sync. A camera
        # coming online is our "on connect" signal — set its local time + disable
        # its NTP client then. None-safe: skipped when unset.
        self._time_sync: Optional[TimeSyncManager] = None
        # Optional on-connect talk-speaker probe; set via set_speaker_probe.
        # Driven off the same online-transition hook as time-sync. None-safe.
        self._speaker_probe: Optional[SpeakerProbeManager] = None
        # Optional one-way migration off the RETIRED hardware lens mask; set via
        # set_lens_mask_cleaner. Driven off the same on-connect hook. None-safe.
        self._lens_mask: Optional[LensMaskCleaner] = None
        # Optional reachability-history tracker (debounce + down-alerts). None-safe.
        self._health: Optional["CameraHealthTracker"] = None

    def set_mqtt(self, mqtt: "MqttPublisher") -> None:
        self._mqtt = mqtt

    def set_time_sync(self, time_sync: "TimeSyncManager") -> None:
        self._time_sync = time_sync

    def set_speaker_probe(self, speaker_probe: "SpeakerProbeManager") -> None:
        self._speaker_probe = speaker_probe

    def set_lens_mask_cleaner(self, lens_mask: "LensMaskCleaner") -> None:
        self._lens_mask = lens_mask

    def set_health(self, health: "CameraHealthTracker") -> None:
        self._health = health

    def is_online(self, name: str) -> bool:
        return self._online.get(name, False)

    def online_count(self) -> int:
        return sum(1 for online in self._online.values() if online)

    def probe_soon(self) -> None:
        self._wakeup.set()

    async def _probe_one(self, name: str, ip: str) -> tuple[str, bool]:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(ip, 554), timeout=4.0)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return name, True
        except (OSError, asyncio.TimeoutError):
            return name, False

    async def run(self) -> None:
        while True:
            try:
                cameras = await self._db.list_cameras()
                by_name = {cam["name"]: cam for cam in cameras}
                results = await asyncio.gather(
                    *(self._probe_one(cam["name"], cam["ip"]) for cam in cameras)
                )
                known = {name for name, _ in results}
                for stale in set(self._online) - known:
                    del self._online[stale]
                for name, online in results:
                    if self._online.get(name) != online:
                        was_known = name in self._online
                        self._online[name] = online
                        if was_known or online is False:
                            log.info("camera %s is %s", name, "online" if online else "offline")
                        await self._ws.broadcast(
                            {"type": "camera_status", "camera": name, "online": online}
                        )
                        if self._mqtt is not None:
                            try:
                                await self._mqtt.publish_connectivity(name, online)
                            except Exception:  # noqa: BLE001 — MQTT must not stall probing
                                log.exception("mqtt connectivity publish failed for %s", name)
                        # "On connect": a camera that just became reachable has
                        # its clock set to the correct local time + NTP disabled
                        # (once, in the background). Non-fatal — must never stall
                        # probing.
                        if online and self._time_sync is not None and name in by_name:
                            try:
                                await self._time_sync.notify_reachable(by_name[name])
                            except Exception:  # noqa: BLE001
                                log.exception("time-sync notify failed for %s", name)
                        # Same "on connect" hook: probe the camera's real
                        # talk-speaker so the Talk button only shows where audio
                        # output exists. Non-fatal — must never stall probing.
                        if online and self._speaker_probe is not None and name in by_name:
                            try:
                                await self._speaker_probe.notify_reachable(by_name[name])
                            except Exception:  # noqa: BLE001
                                log.exception("speaker-probe notify failed for %s", name)
                        # Same "on connect" hook: clear the RETIRED hardware lens
                        # mask so no camera is left blacked out by the removed
                        # control. Non-fatal — must never stall probing.
                        if online and self._lens_mask is not None and name in by_name:
                            try:
                                await self._lens_mask.notify_reachable(by_name[name])
                            except Exception:  # noqa: BLE001
                                log.exception("lens-mask notify failed for %s", name)
                # Reachability HISTORY + debounced down-alerts. Fed the raw poll
                # results (not the live-badge transitions), because it applies
                # its own debounce. Never stalls probing.
                if self._health is not None:
                    try:
                        await self._health.observe(results)
                    except Exception:  # noqa: BLE001
                        log.exception("camera-health observe failed")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("camera probe cycle failed")
            self._wakeup.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=_PROBE_INTERVAL_S)


async def _auto_restart_loop(settings: SettingsStore) -> None:
    """Optional nightly restart at settings.system.auto_restart.time (local).

    Re-reads settings EVERY tick rather than sleeping until the next scheduled
    moment, so enabling/disabling or moving the time takes effect without a
    restart of its own (which would be a funny way to configure a restart).

    Guards against a double-fire: after restarting we would normally be gone,
    but if the process somehow survives (no supervisor), `_last_fired_day`
    stops the same day's slot from firing again in a tight loop.
    """
    last_fired_day: Optional[str] = None
    while True:
        try:
            cfg = (settings.current.get("system") or {}).get("auto_restart") or {}
            if bool(cfg.get("enabled")):
                raw = str(cfg.get("time") or "04:00")
                try:
                    hh, mm = (int(p) for p in raw.split(":", 1))
                except ValueError:
                    hh, mm = 4, 0
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                # Fire inside a one-minute window after the target so a 30 s tick
                # can't skip it, and only once per calendar day.
                if last_fired_day != today and 0 <= (now - due).total_seconds() < 60:
                    last_fired_day = today
                    await request_restart(f"scheduled auto-restart at {raw}")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a scheduling bug must never kill the app
            log.exception("auto-restart scheduler tick failed")
        await asyncio.sleep(30)


async def _prune_loop(db: Database, settings: SettingsStore, config: Config) -> None:
    """Daily-ish retention: drop event rows plus their snapshot + clip files
    past the longest configured retention window. (The recorder owns the
    finer-grained 24/7-segment and clip retention passes.)"""
    while True:
        try:
            recording = settings.recording
            keep_days = max(
                int(recording.get("event_days", 14)),
                int(recording.get("snapshot_days", 14)),
                1,
            )
            cutoff = time.time() - keep_days * 86400
            ids = await db.prune_events_older_than(cutoff)
            for event_id in ids:
                with contextlib.suppress(OSError):
                    (config.snapshots_dir / f"{event_id}.jpg").unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    (config.clips_dir / f"{event_id}.mp4").unlink(missing_ok=True)
            if ids:
                log.info("pruned %d events older than %d days", len(ids), keep_days)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("event pruning failed")
        await asyncio.sleep(_PRUNE_INTERVAL_S)


async def _seed_cameras(db: Database, config: Config) -> None:
    if await db.camera_count() > 0:
        return
    seeds = config.seed_cameras()
    for seed in seeds:
        width, height = MODEL_DETECT_DEFAULTS.get(seed["model"], (704, 480))
        await db.upsert_camera(
            {
                **seed,
                # Env-seeded cameras detect the defaults (person/dog/cat/car).
                # NOT [] — under record-only semantics an empty list means
                # "detect nothing", which would silently disable detection on a
                # fresh env-seeded deploy.
                "detect_objects": list(DEFAULT_DETECT_OBJECTS),
                # No exempt (privacy/ignore) zones by default.
                "exempt_zones": [],
                "detect_width": width,
                "detect_height": height,
                "detect_fps": DEFAULT_DETECT_FPS,
                "detect_enabled": True,
                "record_enabled": True,
                "capabilities": static_capabilities(seed["model"]),
                "created_at": time.time(),
            }
        )
        log.info("seeded camera '%s' (%s @ %s) from env", seed["name"], seed["model"], seed["ip"])
    if not seeds:
        log.info("no CAM{1..3}_* env vars set — camera DB starts empty")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.snapshots_dir.mkdir(parents=True, exist_ok=True)
    config.suppression_thumbs_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.go2rtc_config_dir.mkdir(parents=True, exist_ok=True)
    try:
        config.recordings_dir.mkdir(parents=True, exist_ok=True)
        config.clips_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only/missing media mount must not stop the UI from serving;
        # the recorder logs its own failures.
        log.exception("could not create media directories under %s", config.media_dir)

    secrets = load_or_create_secrets(config.secrets_path)
    auth = AuthService(
        secret=secrets["jwt_secret"],
        admin_password=config.admin_password,
        token_days=config.token_days,
        media_token_days=config.media_token_days,
    )

    db = Database(config.db_path)
    await db.connect()
    await _seed_cameras(db, config)
    # Reclaim doorbell events whose hold-open supervisor died with the previous
    # process (crash, or the restart watchdog's force-exit, which skips
    # `finally` by design). Left alone they read as "processing" forever.
    orphaned = await db.close_open_doorbell_events(DOORBELL_MAX_S)
    if orphaned:
        log.warning("closed %d doorbell event(s) left open by the previous run", orphaned)

    settings = SettingsStore(db, env_public_url=config.public_url)
    await settings.load()

    ws = WSManager()
    push = PushService(db, secrets["vapid_private_key"], secrets["vapid_public_key"])
    # APNs (iOS) sender — reads settings.notifications.apns live; mode "off"
    # (or no registered devices) makes every send a cheap no-op.
    apns = ApnsService(db, settings)
    # ntfy sender — reads settings.notifications.ntfy live; disabled or
    # unconfigured makes every send a cheap no-op. Push for hosters without
    # Apple credentials; a channel BESIDE the relay, not a replacement for it
    # (ntfy alerts land in the ntfy app: no CallKit ring, no native UI).
    ntfy = NtfyService(settings)

    detection = settings.detection
    # The ModelStore is the single detector-model downloader + per-key state
    # machine; it broadcasts model_status on the WS and feeds the
    # /api/detection/models API. refresh() sets initial states from disk
    # (existence-only — no download, no hashing). The active model's actual
    # download happens in the background via the detector bootstrap (which
    # goes through the store), so boot / health never block.
    model_store = ModelStore(config.models_dir, broadcast=ws.broadcast)
    model_store.refresh()
    # Backend selection: settings.detection.backend ("gpu" | "coral") is the
    # normal path; VIGILUME_DETECTOR overrides it when explicitly set. Both land
    # on one object implementing the interface engine/ingest/self-heal drive, so
    # nothing downstream knows or cares which silicon is running. The model store
    # (D-FINE tiers) is consumed by the ONNX path only.
    _backend = str(detection.get("backend") or "gpu")
    detector = build_detector(
        backend=BACKEND_TO_DETECTOR.get(_backend, "onnx"),
        coral_model_key=str(detection.get("coral_model") or ""),
        config=config,
        models_dir=config.models_dir,
        model_key=str(detection.get("model") or DEFAULT_MODEL),
        confidence=float(detection.get("confidence", 0.5)),
        store=model_store,
    )
    model_store.active_key_getter = lambda: settings.detection.get("model") or DEFAULT_MODEL
    model_store.loaded_key_getter = lambda: detector.model_key if detector.ready else None
    recorder = Recorder(config, db, settings)
    engine = DetectionEngine(db, detector, recorder, settings, config)
    media = NativeMediaProvider(db, engine, recorder)
    prober = CameraProber(db, ws)
    # Outbound MQTT + Home Assistant discovery publisher. Constructed always
    # (so PUT /api/settings can restart it live) but only connects when
    # settings.mqtt.enabled + a host are set. Injected into the pipeline like
    # PushService and observed by the prober for connectivity.
    mqtt = MqttPublisher(db, settings, prober, auth)
    prober.set_mqtt(mqtt)
    pipeline = EventsPipeline(
        db, media, ws, push, settings, auth, config.snapshots_dir,
        mqtt=mqtt, apns=apns, ntfy=ntfy,
    )
    engine.set_pipeline(pipeline)
    # The doorbell hold-open path schedules its own clip (the press has no
    # engine track, so nothing else would). Detection events still reach the
    # recorder via the engine.
    pipeline.set_recorder(recorder)
    go2rtc = Go2rtcManager(config, db, settings)
    doorbells = DoorbellManager(pipeline.handle_doorbell)
    # Camera on-board AI event listener (SMD/IVS) for camera-AI-gated detection.
    # Attaches a long-lived digest event stream per ai_on_camera camera running
    # in a camera-AI mode; the ingest gate reads its live "AI active" state and
    # camera_ai_only cameras create events via pipeline.handle_ai_event. Wired
    # into the engine so the ingest manager can consult it per frame.
    ai_events = AiEventListener(pipeline.handle_ai_event, settings=settings)
    engine.set_ai_events(ai_events)
    # Per-camera "Smart spotlight": when a camera's smart_spotlight flag is on,
    # a PERSON detected at NIGHT (config lat/lon -> sunset..sunrise) on a
    # white_light camera auto-turns the spotlight on, held 60 s past the last
    # person. Driven live off the stored flag by the engine's per-frame
    # notify_person; reads the flag/caps from the camera row (no device call on
    # the PUT). Best-effort — a device error only logs. Roster + client factory
    # mirror time_sync/doorbells/ai_events.
    spotlight = SpotlightController(config, cameras_provider=db.list_cameras)
    engine.set_spotlight(spotlight)
    # Re-assert stored desired IR on doorbells (the AD410 resets IR Mode to Auto
    # whenever RTSP streaming (re)connects). The recorder fires on_connect once
    # per (re)connect cycle; a slow sweep backstops missed reconnects.
    ir_reasserter = IrReasserter(db)
    recorder.set_on_connect(ir_reasserter.reassert_soon)
    # Automatic camera clock correction: push the correct local wall-clock time
    # (for the configured IANA timezone) and DISABLE the device NTP client the
    # first time each Dahua/Amcrest camera is reachable — fixes doorbell/camera
    # clock drift without trusting the unreliable NTP + Dahua timezone index. The
    # prober drives it on every online transition; a boot sweep + camera-CRUD
    # resync + a periodic re-push loop (time_sync.run) back it up. Non-fatal.
    time_sync = TimeSyncManager(settings, cameras_provider=db.list_cameras)
    prober.set_time_sync(time_sync)
    # On-connect talk-speaker detection: probe ONVIF GetAudioOutputs the first
    # time each camera is reachable and pin the `speaker` capability so the Talk
    # button only appears on cameras with a real speaker. Same prober hook +
    # boot sweep + camera-CRUD resync as time-sync. Non-fatal.
    speaker_probe = SpeakerProbeManager(db, ws)
    prober.set_speaker_probe(speaker_probe)
    # One-way migration off the RETIRED hardware lens mask (see amcrest/lens_mask.py):
    # clear LeLensMask the first time each camera is reachable, so a camera masked
    # by the removed control is not left permanently blind.
    lens_mask = LensMaskCleaner()
    prober.set_lens_mask_cleaner(lens_mask)

    # Camera reachability history + optional down-alerts. Reads the toggle live
    # from settings so it needs no restart to enable, and uses the UNCOUPLED
    # push primitive so a system alert bypasses the detection notification gates.
    camera_health = CameraHealthTracker(
        db, push,
        alerts_enabled=lambda: bool(
            settings.notifications.get("camera_down_alerts", False)
        ),
        clock=time.time,
    )
    await camera_health.start()
    prober.set_health(camera_health)
    app.state.camera_health = camera_health

    app.state.config = config
    app.state.auth = auth
    app.state.db = db
    app.state.settings = settings
    app.state.ws = ws
    app.state.push = push
    app.state.apns = apns
    app.state.media = media
    app.state.pipeline = pipeline
    app.state.detector = detector
    app.state.model_store = model_store
    app.state.engine = engine
    app.state.recorder = recorder
    app.state.go2rtc = go2rtc
    app.state.doorbells = doorbells
    app.state.ai_events = ai_events
    app.state.spotlight = spotlight
    app.state.ir_reasserter = ir_reasserter
    app.state.time_sync = time_sync
    app.state.speaker_probe = speaker_probe
    app.state.lens_mask = lens_mask
    app.state.prober = prober
    app.state.mqtt = mqtt
    # Per-camera talk locks (single-talker rule for the /talk WS). Fresh
    # dict per app run; the locks themselves are created lazily per camera.
    app.state.talk_locks = {}

    # Software Privacy Mode: resolve the persisted per-camera/group private set
    # into app.state.private_cameras BEFORE any capture subsystem starts, so a
    # camera that was private stays private across a restart (fail-closed). Every
    # capture gate reads this frozenset. Must precede go2rtc.apply() below so the
    # boot config already omits private cameras' live streams.
    await privacy.refresh(app.state)

    # go2rtc config: written before the healthcheck goes green — compose
    # starts the go2rtc container only after we're healthy, so it always
    # boots with a current config. apply() never raises.
    await go2rtc.apply()

    # Startup of subsystems + background tasks happens INSIDE the try so a
    # failure partway through (e.g. recorder.start raising after engine
    # started) still runs the full teardown below instead of orphaning
    # asyncio tasks and ffmpeg child processes. The stop() methods are
    # no-ops/idempotent when their start never ran.
    tasks: list[asyncio.Task] = []
    try:
        await engine.start()
        await spotlight.start()
        await recorder.start()
        # Start the doorbell IR re-assert sweep (also fed by the recorder's
        # per-reconnect hook wired above). Self-managed background task.
        await ir_reasserter.start()
        # Start the MQTT publisher (self-managed background task; no-op + no
        # connection when disabled). Never blocks boot — connection is async.
        await mqtt.start()

        tasks = [
            asyncio.create_task(prober.run(), name="camera-prober"),
            asyncio.create_task(_prune_loop(db, settings, config), name="event-pruner"),
            # Periodic camera clock re-push (every 30 min): re-sets the correct
            # local time on every camera so a drifting clock self-heals between
            # reconnects. Non-fatal; cancelled + drained in the finally below.
            asyncio.create_task(time_sync.run(), name="camera-time-sync"),
            # Optional nightly restart (settings.system.auto_restart). Off by
            # default; the loop is always running so the setting takes effect
            # without needing a restart to schedule a restart.
            asyncio.create_task(_auto_restart_loop(settings), name="auto-restart"),
        ]
        cams = await db.list_cameras()
        await doorbells.sync(cams)
        # Start camera-AI watchers for ai_on_camera cameras in a camera-AI mode.
        await ai_events.sync(cams, default_mode=settings.detection.get("default_mode", "always"))
        # Prime the smart-spotlight roster (prunes state for absent cameras; the
        # per-frame notify_person drives the actual behavior).
        await spotlight.sync(cams)
        # Boot sweep: set the correct local time + disable NTP on cameras
        # already reachable at startup (the prober also fires per online
        # transition; the periodic loop re-pushes thereafter). Non-blocking +
        # non-fatal — each provision runs in its own background task.
        await time_sync.sync(cams)
        # Boot sweep for the talk-speaker probe (mirrors time-sync): probe
        # cameras already reachable at startup. Non-blocking + non-fatal.
        await speaker_probe.sync(cams)
        log.info(
            # detector.kind, NOT config.detector: the backend is chosen by
            # settings.detection.backend now, so the env value is only one of
            # two inputs and printing it would claim "onnx" while Coral is
            # actually running. Same for require_gpu, which onnx_cpu overrides.
            # Ask the constructed object what it IS.
            "Vigilume NVR backend %s up (detector=%s model=%s require_gpu=%s go2rtc=%s)",
            APP_VERSION, detector.kind, detector.model_key,
            # require_gpu is an ONNX-only concept; reporting True next to
            # detector=coral reads as "it still wants a GPU". n/a is the truth.
            getattr(detector, "_require_gpu", "n/a") if detector.kind != "coral" else "n/a",
            config.go2rtc_api_url,
        )

        yield
    finally:
        # ARM THE TEARDOWN DEADLINE FIRST — before anything below can block.
        #
        # Deliberately here and not next to the SIGTERM: uvicorn drains in-flight
        # connections and request tasks BEFORE invoking this shutdown, so a clock
        # started at the signal would be spent on that drain (a client pulling a
        # 30-minute export is enough) and force-exit before a single teardown step
        # had run, orphaning every ffmpeg child mid-segment. Started here, the
        # budget measures exactly the chain below. The drain phase gets its own
        # separate bound via --timeout-graceful-shutdown in the Dockerfile CMD.
        #
        # This covers EVERY shutdown, not just our self-restart: a `docker stop`
        # reaches the same code, and Docker's own SIGKILL is the outer backstop
        # there.
        arm_teardown_watchdog("lifespan shutdown")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await mqtt.stop()
        await ir_reasserter.stop()
        await time_sync.stop_all()
        await speaker_probe.stop_all()
        await lens_mask.stop_all()
        await doorbells.stop_all()
        await ai_events.stop_all()
        await spotlight.stop_all()
        await engine.stop()      # ends open events into the pipeline
        await recorder.stop()
        await pipeline.shutdown()
        await apns.aclose()
        await ntfy.aclose()
        await go2rtc.aclose()
        await media.aclose()
        await db.close()
        log.info("backend shut down cleanly")


app = FastAPI(title="Vigilume NVR", version=APP_VERSION, lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)

@app.middleware("http")
async def _learn_webrtc_host(request, call_next):
    """Learn this box's LAN address from the address a client reached it on, so
    WebRTC live view works WITHOUT the operator setting VIGILUME_WEBRTC_HOST.

    The backend runs in a Docker bridge network and therefore cannot discover
    its own LAN IP (`_auto_lan_ipv4` only ever sees the bridge address and
    correctly rejects it). With no host candidate go2rtc advertises STUN only,
    WebRTC cannot connect even on the LAN, and every client silently degrades to
    HLS. The Host header of a real LAN request carries exactly the address we
    need. `note_observed_host` accepts ONLY a private IPv4 literal and is
    disabled outright when VIGILUME_WEBRTC_HOST is set, so this can neither be
    steered from off-LAN nor override an explicit configuration.

    Regenerates the go2rtc config once, on CHANGE only — never per request."""
    response = await call_next(request)
    try:
        if streams.note_observed_host(request.url.hostname or ""):
            log.info(
                "learned LAN address %s from a client request — regenerating "
                "go2rtc so WebRTC has a host candidate",
                request.url.hostname,
            )
            go2rtc = getattr(app.state, "go2rtc", None)
            if go2rtc is not None:
                await go2rtc.apply()
    except Exception:  # noqa: BLE001 — never let this break a response
        log.debug("webrtc host learning skipped", exc_info=True)
    return response


app.include_router(system_router.router)
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(cameras_router.router)
app.include_router(cameras_router.snapshot_router)
app.include_router(groups_router.router)
app.include_router(talk_router.router)
app.include_router(events_router.router)
app.include_router(suppressions_router.router)
app.include_router(recordings_router.router)
app.include_router(notifications_router.router)
app.include_router(notifications_router.push_router)
app.include_router(settings_router.router)
app.include_router(privacy_router.router)
app.include_router(detection_router.router)
app.include_router(integrations_router.router)


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(default=None)):
    auth: AuthService = websocket.app.state.auth
    # Prefer the Sec-WebSocket-Protocol bearer over ?token=: a query-string token
    # ends up verbatim in nginx's error log, which is how a live admin JWT
    # leaked once already. ?token= still works for older clients.
    raw, subprotocol = ws_token(websocket, token)
    claims = auth.decode(raw) if raw else None
    if claims is None or claims.get("scope") == "media":
        # Policy violation close code; client redirects to login on failure.
        await websocket.close(code=1008)
        return
    ws: WSManager = websocket.app.state.ws
    # The subprotocol MUST be echoed when the client offered one, or the browser
    # aborts the connection immediately after the handshake.
    await ws.connect(websocket, subprotocol=subprotocol)
    try:
        while True:
            # Clients don't send anything meaningful; this loop exists to
            # detect disconnects (and tolerates pings/keepalive text).
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws.disconnect(websocket)

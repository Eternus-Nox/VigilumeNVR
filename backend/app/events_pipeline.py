"""Event & notification pipeline (docs/CONTRACTS.md, "Event & notification
pipeline" section).

Frigate-SHAPED new/update/end payloads (synthesized in-process by the native
DetectionEngine) -> store + enrich (clean snapshot -> Supervision annotation
-> /data/snapshots/{id}.jpg) -> notify (web push) with per-(camera,label)
cooldown, min_score, enabled-labels filter -> WS broadcast.

Doorbell presses enter here too: they create their own event rows, bypass
the label filter, and carry their own cooldowns.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from . import annotate
from .auth import AuthService
from .db import Database
from .native.media import MediaProvider
from .notify.ntfy import NTFY_ICON_DEFAULT, NTFY_ICON_DOORBELL, ntfy_icon
from .notify.push import PushService
from .settings_store import SettingsStore
from .ws import WSManager

if TYPE_CHECKING:  # pragma: no cover
    from .integrations.mqtt_ha import MqttPublisher
    from .notify.apns import ApnsService
    from .notify.ntfy import NtfyService

log = logging.getLogger(__name__)

_DOORBELL_COOLDOWN_S = 15.0

# ---- doorbell recording window ----
# A press opens an event that stays open while a visitor is still there, so the
# clip covers the whole visit rather than the instant of the press. Clip length
# is derived purely from the event's start/end times (recorder.extract_clip), so
# holding the row open IS the feature — the recorder needs no changes.
#
# The label that holds the window open. "Has the visitor left?" is a question
# about people specifically, and it is what front_door detects.
_DOORBELL_HOLD_LABEL = "person"
# Absence before we call it over. Deliberately the engine's ABSENCE_TIMEOUT_S,
# so a doorbell clip ends on the same rule as a detection clip.
_DOORBELL_ABSENCE_S = 5.0
# Hard cap. Without one, a stuck track, a mis-scored parked car, or a neighbour
# settling in on the porch records forever and leaves the row open forever.
DOORBELL_MAX_S = 120.0
# Floor, so a press always produces a usable clip even when the visitor steps
# out of frame immediately or detection never picks them up at all.
_DOORBELL_MIN_S = 10.0
_DOORBELL_POLL_S = 1.0
_SNAPSHOT_WAIT_TRIES = 4  # media provider retry budget (native: instant)
# camera_ai_only: collapse a burst of camera-AI Start/Pulse fires for the same
# (camera, label) into a single event so a chatty SMD/IVS stream can't spam the
# event log. Independent of the notification cooldown below.
_AI_EVENT_COOLDOWN_S = 10.0


def _as_float(value: Any, default: float = 0.0) -> float:
    """Defensive float cast — frigate/events field types vary across
    Frigate 0.14-0.17 and must never crash the pipeline."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _snapshot_meta(after: dict[str, Any]) -> dict[str, Any]:
    """after.snapshot as a dict ({} when absent or not a dict — older
    Frigate versions don't always include it)."""
    snapshot = after.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def _box_of(after: dict[str, Any]) -> Optional[list[float]]:
    """Best-effort [x1,y1,x2,y2] from snapshot.box or after.box; None when
    absent or malformed."""
    box = _snapshot_meta(after).get("box") or after.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def _scene_of(after: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """The native engine's per-object scene: [{box,label,score}] for every
    counted object in the saved frame. None for doorbell/audio/legacy rows
    with no scene (callers fall back to the single ``_box_of`` box)."""
    scene = after.get("scene")
    if not isinstance(scene, list) or not scene:
        return None
    out: list[dict[str, Any]] = []
    for obj in scene:
        if not isinstance(obj, dict):
            continue
        box = obj.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            out.append(
                {
                    "box": [float(v) for v in box],
                    "label": str(obj.get("label") or ""),
                    "score": _as_float(obj.get("score")),
                }
            )
        except (TypeError, ValueError):
            continue
    return out or None


def _scene_labels(after: dict[str, Any]) -> list[str]:
    """Distinct object classes present in the engine's scene for this payload.
    The engine only ever puts confirmed objects from the camera's detect list
    into the scene, so every label here is already on the detect list. Empty for
    doorbell/audio/legacy payloads (no scene)."""
    scene = _scene_of(after) or []
    seen: list[str] = []
    for obj in scene:
        lbl = str(obj.get("label") or "")
        if lbl and lbl not in seen:
            seen.append(lbl)
    return seen


def _humanize_labels(labels: list[str]) -> str:
    """'Person', 'Person and car', 'Person, car and dog' — a concise,
    sentence-leading join of the event's classes for the notification title/body
    (renders cleanly on a paired Apple Watch)."""
    words = [l.replace("_", " ") for l in labels if l]
    if not words:
        return "Motion"
    words[0] = words[0].capitalize()
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{', '.join(words[:-1])} and {words[-1]}"


class EventsPipeline:
    def __init__(
        self,
        db: Database,
        media: MediaProvider,
        ws: WSManager,
        push: PushService,
        settings: SettingsStore,
        auth: AuthService,
        snapshots_dir: Path,
        mqtt: Optional["MqttPublisher"] = None,
        apns: Optional["ApnsService"] = None,
        ntfy: Optional["NtfyService"] = None,
    ):
        self._db = db
        self._media = media
        self._ws = ws
        self._push = push
        self._settings = settings
        self._auth = auth
        self._snapshots_dir = snapshots_dir
        # Outbound MQTT / Home Assistant publisher (optional). None-safe: when
        # unset or disabled every hook is a cheap no-op. Injected like _push.
        self._mqtt = mqtt
        # APNs (iOS) sender (optional). Fired alongside web push under the
        # SAME cooldown/label/min_score gates; never raises. Injected like _push.
        self._apns = apns
        # ntfy sender (optional) — push for hosters with no Apple credentials.
        # Same gates, same media-token snapshot URL, never raises. A channel
        # beside the APNs relay, not a replacement (no CallKit ring).
        self._ntfy = ntfy
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # (camera, label) -> live in-frame count, fed by the engine's
        # update_count() calls on every active-track change.
        self.counts: dict[tuple[str, str], int] = {}
        # frigate_id -> tracking state for in-progress events
        self._active: dict[str, dict[str, Any]] = {}
        # cooldown key -> monotonic timestamp of last notification
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._tasks: set[asyncio.Task] = set()
        # Cameras with a doorbell recording supervisor in flight. One visit gets
        # ONE recording: the press cooldown (15s) is far shorter than the hold-
        # open cap (120s), so a visitor who rings repeatedly while waiting would
        # otherwise start a supervisor per press — all watching the same person,
        # all scheduling a clip over the same segments.
        self._doorbell_recording_cams: set[str] = set()
        # Recorder, injected after construction (mirrors engine.set_pipeline).
        # Only the doorbell hold-open path uses it — detection events reach the
        # recorder through the engine, which owns its own reference. Optional so
        # the pipeline stays constructible without a recorder in tests.
        self._recorder: Any = None

    def set_recorder(self, recorder: Any) -> None:
        self._recorder = recorder

    # ---------- counts cache ----------

    def update_count(self, camera: str, label: str, count: int) -> None:
        self.counts[(camera, label)] = max(0, count)

    def current_count(self, camera: str, label: str) -> int:
        """Cached in-frame count; falls back to 1 when the cache is cold
        (defensive only — the native engine feeds counts on every change,
        so the cache is never cold for engine-produced events)."""
        return max(1, self.counts.get((camera, label), 1))

    # ---------- task management ----------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---------- frigate/events ----------

    @staticmethod
    def _score_of(after: dict[str, Any]) -> float:
        candidates = [after.get("top_score"), after.get("score")]
        candidates.append(_snapshot_meta(after).get("score"))
        scores = [_as_float(c, -1.0) for c in candidates if c is not None]
        return max((s for s in scores if s >= 0.0), default=0.0)

    @staticmethod
    def _zones_of(after: dict[str, Any]) -> list[str]:
        entered = after.get("entered_zones")
        current = after.get("current_zones")
        zones = set(entered if isinstance(entered, list) else [])
        zones |= set(current if isinstance(current, list) else [])
        return sorted(str(z) for z in zones)

    async def handle_event(self, payload: dict[str, Any]) -> None:
        etype = payload.get("type")
        after = payload.get("after") or payload.get("before") or {}
        if not isinstance(after, dict):  # tolerate odd payloads (0.14-0.17 variance)
            return
        fid = after.get("id")
        camera = after.get("camera")
        label = after.get("label")
        if not fid or not camera or not label or etype not in ("new", "update", "end"):
            return
        # PRIVACY MODE GATE (app/privacy.py) — funnel 1 of 3. Drops the event row,
        # snapshot enrichment, WS broadcast, MQTT publish and every notification
        # for a private camera in one place. Doorbell presses and camera-AI events
        # do NOT pass through here; they have their own gates below.
        if self._settings.is_private(camera):
            return

        if etype == "new":
            await self._on_new(fid, after)
        elif etype == "update":
            await self._on_update(fid, after)
        else:
            await self._on_end(fid, after)

    async def _on_new(self, fid: str, after: dict[str, Any]) -> None:
        if fid in self._active:
            return
        existing = await self._db.get_event_by_frigate_id(fid)
        camera, label = after["camera"], after["label"]
        count = self.current_count(camera, label)
        # Multi-object: seed the running label set with the primary label plus
        # every other class already present in the opening frame's scene.
        labels = [label] + [l for l in _scene_labels(after) if l != label]
        if existing:
            event_id = int(existing["id"])
            # Merge into whatever the DB already has (restart-adopt path).
            labels = list(dict.fromkeys((existing.get("labels") or []) + labels))
            await self._db.update_event(event_id, labels=labels)
        else:
            event_id = await self._db.insert_event(
                frigate_id=fid,
                camera=camera,
                label=label,
                count=count,
                score=self._score_of(after),
                start_time=_as_float(after.get("start_time")) or time.time(),
                zones=self._zones_of(after),
                has_clip=bool(after.get("has_clip")),
                has_snapshot=False,  # set once our annotated copy is saved
                # Best detection box (detect px) so a later "wrong / not a
                # <object>" reject can derive the normalized foot-point to learn
                # a suppression. None for doorbell/audio rows (no box) -> [].
                box=_box_of(after),
                labels=labels,
            )
        self._active[fid] = {
            "event_id": event_id,
            "max_count": count,
            "notified": False,
            "snap_time": None,
            "enriching": False,
            # Running set of all classes seen during this event (order-preserving).
            "labels": labels,
        }
        row = await self._db.get_event(event_id)
        if row:
            await self._ws.broadcast({"type": "event_new", "event": row})
        await self._publish_mqtt(after["camera"], after["label"], "new", row)
        self._spawn(self._enrich_and_notify(fid, after))

    async def _on_update(self, fid: str, after: dict[str, Any]) -> None:
        state = self._active.get(fid)
        if state is None:
            # Backend restarted mid-event: adopt it.
            await self._on_new(fid, after)
            return
        camera, label = after["camera"], after["label"]
        count = self.current_count(camera, label)
        state["max_count"] = max(state["max_count"], count)
        update_fields: dict[str, Any] = {
            "score": self._score_of(after),
            "count": state["max_count"],
            "zones": self._zones_of(after),
            "has_clip": bool(after.get("has_clip")),
        }
        # Accumulate any newly-appearing classes into the event's label set.
        known: list[str] = state.get("labels") or [label]
        grew = False
        for lbl in [label, *_scene_labels(after)]:
            if lbl and lbl not in known:
                known.append(lbl)
                grew = True
        if grew:
            state["labels"] = known
            update_fields["labels"] = known
        # Track the best box alongside score (the engine adopts a new best box on
        # a higher score, so the stored box stays aligned with the snapshot the
        # operator sees when rejecting). Never overwrite a good box with None.
        box = _box_of(after)
        if box is not None:
            update_fields["box"] = box
        await self._db.update_event(state["event_id"], **update_fields)
        row = await self._db.get_event(state["event_id"])
        if row:
            await self._ws.broadcast({"type": "event_update", "event": row})
        await self._publish_mqtt(camera, label, "update", row)

        snap_time = _snapshot_meta(after).get("frame_time")
        needs_snapshot = after.get("has_snapshot") and snap_time != state.get("snap_time")
        if (needs_snapshot or not state["notified"]) and not state["enriching"]:
            self._spawn(self._enrich_and_notify(fid, after))

    async def _on_end(self, fid: str, after: dict[str, Any]) -> None:
        state = self._active.pop(fid, None)
        if state is None:
            row = await self._db.get_event_by_frigate_id(fid)
            if row is None:
                return
            event_id = int(row["id"])
        else:
            event_id = state["event_id"]
        await self._db.update_event(
            event_id,
            end_time=_as_float(after.get("end_time"))
            or _as_float(after.get("frame_time"))
            or time.time(),
            has_clip=bool(after.get("has_clip")),
        )
        # Last chance to grab a snapshot if we never managed to.
        if state is not None and state.get("snap_time") is None and after.get("has_snapshot"):
            self._active[fid] = state  # keep state alive for the final enrich
            self._spawn(self._final_enrich(fid, after))
        row = await self._db.get_event(event_id)
        if row:
            await self._ws.broadcast({"type": "event_end", "event": row})
        await self._publish_mqtt(after.get("camera"), after.get("label"), "end", row)

    async def _publish_mqtt(
        self, camera: Optional[str], label: Optional[str], etype: str, row: Optional[dict[str, Any]]
    ) -> None:
        """Mirror the WS broadcast to the MQTT/Home Assistant publisher: a
        detection ``new``/``update`` turns the per-label binary_sensor ON, an
        ``end`` turns it OFF, and the last-event sensor is refreshed. No-op when
        the publisher is unset or MQTT is disabled; never raises."""
        if self._mqtt is None or not camera or not label:
            return
        try:
            await self._mqtt.publish_event(camera, label, etype, row)
        except Exception:  # noqa: BLE001 — an MQTT hiccup must not touch the pipeline
            log.exception("mqtt publish_event failed for %s/%s", camera, label)

    async def _final_enrich(self, fid: str, after: dict[str, Any]) -> None:
        try:
            await self._enrich_and_notify(fid, after)
        finally:
            self._active.pop(fid, None)

    # ---------- enrichment (snapshot fetch + annotation) ----------

    async def _enrich_and_notify(self, fid: str, after: dict[str, Any]) -> None:
        state = self._active.get(fid)
        if state is None or state["enriching"]:
            return
        state["enriching"] = True
        try:
            await self._do_enrich(fid, after, state)
            await self._maybe_notify_object(fid, after, state)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("event %s: enrichment failed", fid)
        finally:
            state["enriching"] = False

    async def _do_enrich(self, fid: str, after: dict[str, Any], state: dict[str, Any]) -> None:
        # PRIVACY MODE re-check (in-flight defence). Enrichment is spawned as a
        # background task and then WAITS on the snapshot; a camera can go private
        # in that window, and without this the task would still write a real
        # snapshot to disk after the toggle. The entry gates cannot close this —
        # only a re-check at the point of writing can.
        if self._settings.is_private(str(after.get("camera") or "")):
            return
        snapshot_meta = _snapshot_meta(after)
        jpeg = await self._media.event_snapshot(fid, retries=_SNAPSHOT_WAIT_TRIES)
        if jpeg is None:
            return
        box = _box_of(after)
        # Full per-object scene (native events) -> box every counted object;
        # doorbell/audio/legacy rows have no scene and fall back to `box`.
        scene = _scene_of(after)
        score = _as_float(snapshot_meta.get("score")) or self._score_of(after)
        count = state["max_count"]
        # Box→snapshot rescale info (native frames are detect-res, so this
        # is a no-op rescale). Strictly best-effort: never block enrichment.
        try:
            detect_dims = await self._media.detect_dims(str(after.get("camera") or ""))
        except Exception:  # noqa: BLE001
            detect_dims = None
        # settings.notifications.draw_boxes=false -> clean snapshot (banner
        # only). Legacy-safe: a stored settings blob without the key means True.
        draw_boxes = bool(self._settings.notifications.get("draw_boxes", True))
        # Include zones / crossing lines and per-object traces. Legacy-safe the
        # same way draw_boxes is: a settings blob saved before these existed
        # reads as True, and a doorbell/audio event simply carries no geometry.
        notif = self._settings.notifications
        annotated = await asyncio.to_thread(
            functools.partial(
                annotate.annotate_event_snapshot,
                jpeg,
                box,
                str(after["label"]),
                score,
                count,
                detect_dims,
                scene,
                draw_boxes,
                zones=after.get("include_zones") or [],
                lines=after.get("lines") or [],
                draw_zones=bool(notif.get("draw_zones", True)),
                draw_traces=bool(notif.get("draw_traces", True)),
            )
        )
        data = annotated or jpeg  # never lose the frame over an annotation bug
        path = self._snapshots_dir / f"{state['event_id']}.jpg"
        await asyncio.to_thread(path.write_bytes, data)
        state["snap_time"] = snapshot_meta.get("frame_time") or time.time()
        await self._db.update_event(state["event_id"], has_snapshot=True)
        row = await self._db.get_event(state["event_id"])
        if row:
            await self._ws.broadcast({"type": "event_update", "event": row})

    # ---------- notifications ----------

    def _cooldown_ok(self, key: tuple[str, str], cooldown_s: float) -> bool:
        last = self._cooldowns.get(key)
        return last is None or (time.monotonic() - last) >= cooldown_s

    def _mark_cooldown(self, key: tuple[str, str]) -> None:
        self._cooldowns[key] = time.monotonic()

    async def _friendly_name(self, camera: str) -> str:
        cam = await self._db.get_camera(camera)
        return cam["friendly_name"] if cam else camera

    def _media_url(self, event_id: int) -> str:
        token = self._auth.create_media_token(resource=f"event:{event_id}")
        base = self._settings.public_url
        return f"{base}/api/events/{event_id}/snapshot.jpg?token={token}"

    def _click_url(self, event_id: int) -> str:
        return f"{self._settings.public_url}/events/{event_id}"

    async def _maybe_notify_object(self, fid: str, after: dict[str, Any], state: dict[str, Any]) -> None:
        if state["notified"]:
            return
        ns = self._settings.notifications
        if not ns.get("enabled", True):
            return
        camera, label = after["camera"], after["label"]
        if label not in (ns.get("labels") or []):
            return
        if self._score_of(after) < float(ns.get("min_score", 0.7)):
            return
        if not self._cooldown_ok((camera, label), float(ns.get("cooldown_seconds", 60))):
            return
        state["notified"] = True
        self._mark_cooldown((camera, label))

        friendly = await self._friendly_name(camera)
        count = state["max_count"]
        # Multi-object: title/body list every detected class so the alert (and
        # its Apple Watch mirror) shows the full picture, not just the primary.
        # The primary label leads; the count applies to the primary label only.
        labels: list[str] = state.get("labels") or [label]
        if len(labels) > 1:
            phrase = _humanize_labels(labels)
            title = f"{phrase} detected at {friendly}"
            body = f"{phrase} in frame"
        else:
            title = f"{label.replace('_', ' ').capitalize()} detected at {friendly}"
            body = f"{annotate.plural_label(label, count)} in frame"
        has_snapshot = state.get("snap_time") is not None
        await self._send_notification(
            title=title,
            body=body,
            event_id=state["event_id"],
            tag=f"vigilume-{camera}-{label}",
            icon=ntfy_icon([label]),
            with_image=has_snapshot,
            camera=camera,
            camera_label=friendly,
        )

    async def _send_notification(
        self,
        title: str,
        body: str,
        event_id: int,
        tag: str,
        with_image: bool,
        camera: Optional[str] = None,
        camera_label: Optional[str] = None,
        urgent: bool = False,
        icon: Optional[str] = None,
    ) -> None:
        """`urgent` escalates the ntfy priority to max (5). Used for a doorbell
        press that will NOT be carried by a CallKit ring — see `_can_ring`."""
        # PRIVACY MODE re-check (in-flight defence). APNs and ntfy are fired as
        # fire-and-forget tasks with ~30 s of retries, so a notification queued
        # just before the toggle could otherwise reach a phone well after the
        # camera went private — leaking both the event and its snapshot URL.
        if camera and self._settings.is_private(camera):
            return
        image_url = self._media_url(event_id) if with_image else None
        click_url = self._click_url(event_id)
        payload: dict[str, Any] = {"title": title, "body": body, "tag": tag, "data": {"url": click_url}}
        if image_url:
            payload["image"] = image_url
        push_result = await self._push.send_to_all(payload)
        log.info(
            "notification '%s' -> %d/%d push subscriber(s)",
            title,
            push_result.sent,
            push_result.attempted,
        )
        # APNs rides the same decision (identical gates already passed, same
        # media-token snapshot URL, collapse_id = event id) but in its OWN
        # spawned task: ApnsService has per-request timeouts and never raises,
        # yet a hung/down relay still costs up to ~30s of retries — that must
        # never stall this caller (the doorbell watcher awaits it inline, and
        # the enrich task holds the per-event `enriching` flag).
        if self._apns is not None:
            self._spawn(
                self._send_apns(title, body, event_id, image_url, camera, camera_label)
            )
        # ntfy rides the same decision + the same media-token snapshot URL.
        # Spawned for the same reason as APNs: it is an HTTP call to a server
        # we don't control (ntfy.sh or the operator's own), and handle_doorbell
        # awaits _send_notification INLINE while holding the per-event flag — a
        # slow ntfy must never stall a doorbell ring.
        if self._ntfy is not None:
            self._spawn(self._send_ntfy(title, body, event_id, image_url, tag))

    async def _send_ntfy(
        self,
        title: str,
        body: str,
        event_id: int,
        image_url: Optional[str],
        tag: str,
    ) -> None:
        assert self._ntfy is not None
        res = await self._ntfy.send(
            title=title,
            body=body,
            click_url=self._click_url(event_id),
            attach_url=image_url,
            # The ntfy `Tags` header is USER-VISIBLE. Send an emoji shortcode,
            # never `tag` — that is the web-push/APNs collapse id, and ntfy would
            # print it as a literal #hashtag on every notification.
            tag=icon or NTFY_ICON_DEFAULT,
            # ntfy's own max. NOT an Apple "Critical Alert" — that needs a
            # special Apple entitlement and only works over APNs, which is
            # precisely what is unavailable here. 5 is the loudest thing ntfy
            # can ask for: it bypasses Do Not Disturb on Android and is the
            # highest-attention alert the ntfy iOS app will raise.
            priority=5 if urgent else None,
        )
        # attempted == 0 means "off or not configured" — normal, so stay quiet
        # rather than logging a non-event on every detection.
        if res.attempted:
            log.info(
                "ntfy notification '%s' -> %d/%d", title, res.sent, res.attempted
            )

    async def _send_apns(
        self,
        title: str,
        body: str,
        event_id: int,
        image_url: Optional[str],
        camera: Optional[str] = None,
        camera_label: Optional[str] = None,
    ) -> None:
        assert self._apns is not None
        apns_result = await self._apns.send_to_all(
            title=title,
            body=body,
            event_id=str(event_id),
            snapshot_url=image_url,
            camera=camera,
            camera_label=camera_label,
            priority="high",
            collapse_id=str(event_id),
        )
        log.info(
            "apns notification '%s' -> %d/%d device(s)",
            title,
            apns_result.sent,
            apns_result.attempted,
        )

    def _can_ring(self) -> bool:
        """Whether a doorbell press will actually ring the phone like a call.

        The CallKit ring is a PushKit VoIP push, which is APNs — so it exists
        only when the APNs relay is configured and on. With ntfy as the sole
        channel there is no ring at all: ntfy delivers through its OWN app, and
        no notification can make a third-party app place a call. When this is
        False the doorbell notification is escalated instead (priority 5)."""
        if self._apns is None:
            return False
        apns = self._settings.notifications.get("apns") or {}
        return str(apns.get("mode") or "off") == "relay"

    async def _send_voip(self, camera: str, friendly: str, event_id: int) -> None:
        """Fire the CallKit VoIP push for a doorbell press. The CallKit handle is
        the camera friendly name (e.g. "Front Door")."""
        assert self._apns is not None
        result = await self._apns.send_voip_to_all(camera=friendly, event_id=str(event_id))
        log.info(
            "voip doorbell ring (%s) -> %d/%d device(s)",
            friendly,
            result.sent,
            result.attempted,
        )

    # ---------- doorbell ----------

    async def handle_doorbell(self, camera: str) -> None:
        # PRIVACY MODE GATE — funnel 2 of 3. A doorbell press bypasses
        # handle_event entirely: it inserts its own event row, grabs a live
        # snapshot, and fires a CallKit VoIP RING plus a push. Per the decided
        # behaviour ("fully silent"), a press on a private doorbell produces
        # nothing at all — no ring, no push, no snapshot, no event row.
        if self._settings.is_private(camera):
            return
        if not self._cooldown_ok(("doorbell", camera), _DOORBELL_COOLDOWN_S):
            return
        self._mark_cooldown(("doorbell", camera))
        friendly = await self._friendly_name(camera)
        now = time.time()
        fid = f"doorbell.{int(now * 1000)}"
        # A doorbell clip is cut from the 24/7 segments, so with recording off
        # there is nothing to cut and holding the row open would only leave an
        # event stuck "processing". Close it immediately in that case, exactly
        # as before.
        cam_row = await self._db.get_camera(camera)
        # ONE recording per visit. A repeat press 20s in is a second RING — which
        # is the whole point of a doorbell — but must not open a second, parallel
        # recording of the same visitor.
        will_record = (
            bool(cam_row and cam_row.get("record_enabled"))
            and camera not in self._doorbell_recording_cams
        )
        event_id = await self._db.insert_event(
            frigate_id=fid,
            camera=camera,
            label="doorbell",
            count=1,
            score=1.0,
            start_time=now,
            # OPEN (NULL) while we record; the supervisor closes it once the
            # visitor leaves, and only then does a clip window exist. Otherwise
            # closed AT the press (end_time == start_time) — the marker shape
            # that tells the API this row was never eligible for a clip, so it
            # reports "no clip" rather than "the recorder failed".
            end_time=None if will_record else now,
            has_clip=False,
            has_snapshot=False,
        )
        # SPAWN BEFORE anything that can fail. The supervisor owns closing this
        # row; every await below (snapshot grab, WS, ring, push) can raise, and
        # the doorbell watcher catches and logs that without anyone noticing —
        # leaving an open row with nothing alive to close it, i.e. an event stuck
        # on "processing" until the next reboot.
        if will_record:
            self._doorbell_recording_cams.add(camera)
            self._spawn(self._doorbell_recording(camera, fid, event_id, now))

        has_snapshot = await self._save_live_snapshot(camera, event_id, "Doorbell pressed")
        row = await self._db.get_event(event_id)
        if row:
            await self._ws.broadcast({"type": "doorbell", "event": row})

        # CallKit ring: fire a VoIP APNs push so the phone rings like a real call
        # (independent of the alert-push `enabled` toggle — a doorbell ring is a
        # call, not a background notification). No-op when APNs is off or no VoIP
        # tokens are registered; isolated on its own task so a hung relay never
        # stalls this inline doorbell watcher, and never raises.
        if self._apns is not None:
            self._spawn(self._send_voip(camera, friendly, event_id))

        if self._settings.notifications.get("enabled", True):
            # No CallKit ring available (ntfy-only): this notification is the
            # ONLY thing that will alert anyone, so send it at max priority
            # rather than letting a doorbell press look like a routine motion
            # alert. When the ring IS available the push stays at the
            # configured priority — the phone is already ringing.
            await self._send_notification(
                title=f"Doorbell pressed at {friendly}",
                body="Someone is at the door",
                event_id=event_id,
                tag=f"vigilume-{camera}-doorbell",
                with_image=has_snapshot,
                camera=camera,
                camera_label=friendly,
                urgent=not self._can_ring(),
                icon=NTFY_ICON_DOORBELL,
            )

    async def _close_doorbell_event(self, event_id: int, end_time: float) -> None:
        """Stamp the final end_time and let clients pick up the real duration."""
        await self._db.update_event(event_id, end_time=end_time)
        row = await self._db.get_event(event_id)
        if row:
            await self._ws.broadcast({"type": "event_end", "event": row})

    async def _doorbell_recording(
        self, camera: str, fid: str, event_id: int, start_time: float
    ) -> None:
        """Keep a doorbell event open while a person is in frame, then close it
        and schedule the clip.

        Deliberately a poll of the engine's live count rather than a hook into
        the engine's own event lifecycle: a doorbell press is not a detection,
        it has no track of its own, and wiring it into ``_on_new``/``_on_end``
        would subject the CallKit ring to the notification label/min_score/
        cooldown gates and silently kill it. Reading the count keeps recording
        and notification concerns separate.
        """
        last_present = start_time
        now = start_time
        outcome = "cap"
        end_time = start_time
        try:
            while True:
                await asyncio.sleep(_DOORBELL_POLL_S)
                now = time.time()
                # PRIVACY MODE can be switched on mid-visit; that is a capture
                # kill switch, so abandon the recording rather than finishing
                # it. (schedule_clip re-checks too — this just stops sooner.)
                if self._settings.is_private(camera):
                    outcome = "privacy"
                    break
                # Read `counts` DIRECTLY. current_count() floors its result at
                # 1 as a defensive measure for engine-produced events, so it
                # can NEVER report zero — a loop built on it would never end.
                if self.counts.get((camera, _DOORBELL_HOLD_LABEL), 0) > 0:
                    last_present = now
                if now - start_time >= DOORBELL_MAX_S:
                    outcome = "cap"
                    break
                if (
                    now - last_present >= _DOORBELL_ABSENCE_S
                    and now - start_time >= _DOORBELL_MIN_S
                ):
                    outcome = "left"
                    break
        except asyncio.CancelledError:
            outcome = "cancelled"          # graceful shutdown; recorder stops too
            raise
        except Exception:                  # noqa: BLE001
            # A DB hiccup or a settings read must not leave the row open. This
            # task's exception would otherwise be swallowed by _spawn's bare
            # done-callback and surface only as "Task exception was never
            # retrieved" at GC, while the event read "processing" until reboot.
            outcome = "error"
            log.exception("doorbell %s: recording supervisor failed", camera)
        finally:
            self._doorbell_recording_cams.discard(camera)
            if outcome in ("left", "cap"):
                end_time = (
                    now if outcome == "cap"
                    else max(last_present, start_time + _DOORBELL_MIN_S)
                )
            else:
                # No clip is coming on these paths, so close AT the press. That
                # marker shape (end_time == start_time) is what tells the API
                # the row was never eligible for a clip — otherwise the UI
                # spends 45 s insisting a clip is "being cut from continuous
                # recording", and iOS mounts a live player on a camera that
                # Privacy Mode just tore the stream down for.
                end_time = start_time
            try:
                await self._close_doorbell_event(event_id, end_time)
            except Exception:  # noqa: BLE001
                # Nothing further we can do, and raising from a `finally` would
                # replace an in-flight CancelledError during shutdown. The boot
                # sweep reclaims the row.
                log.exception("doorbell %s: could not close event %d", camera, event_id)

        if outcome not in ("left", "cap"):
            log.info("doorbell %s: recording ended early (%s) — no clip", camera, outcome)
            return
        log.info(
            "doorbell %s: visit lasted %.1fs (%s) — scheduling clip",
            camera, end_time - start_time,
            "visitor left" if outcome == "left" else f"{DOORBELL_MAX_S:.0f}s cap reached",
        )
        if self._recorder is not None:
            await self._recorder.schedule_clip(camera, fid, start_time, end_time)

    # ---------- camera-AI-only object events ----------

    async def handle_ai_event(self, camera: str, label: str) -> None:
        """Create an event directly from a camera's on-board AI event
        (``camera_ai_only`` mode — no server inference ran).

        The ``label`` is the mapped object type (person/car/motion). Respects the
        camera's ``detect_objects`` filter (an object the camera isn't watching
        is ignored), collapses rapid re-fires with a per-(camera,label) cooldown,
        grabs a live snapshot for the event image, and notifies under the SAME
        gates as an engine object event (enabled / labels / min_score /
        notification cooldown). Never raises into the AI listener."""
        # PRIVACY MODE GATE — funnel 3 of 3. camera_ai_only cameras never spawn
        # an ingest source, so the detection gate does nothing for them: their
        # events arrive straight from the camera's own AI and would otherwise
        # keep writing rows + live snapshots while the camera is "private".
        if self._settings.is_private(camera):
            return
        cam = await self._db.get_camera(camera)
        if cam is None:
            return
        # detect_objects filter: [] = record-only (create nothing); otherwise the
        # label must be in the camera's tracked-object list.
        wanted = cam.get("detect_objects") or []
        if label not in wanted:
            return
        # De-dupe bursts of the same object into one event.
        if not self._cooldown_ok(("ai", camera, label), _AI_EVENT_COOLDOWN_S):
            return
        self._mark_cooldown(("ai", camera, label))

        now = time.time()
        friendly = await self._friendly_name(camera)
        # "cameraai." prefix => a synthetic, snapshot-only event (routers/events.py
        # treats it like doorbell.: served from /data/snapshots, never asked for a
        # clip). Distinct from the "native." engine prefix.
        event_id = await self._db.insert_event(
            frigate_id=f"cameraai.{int(now * 1000)}-{label}",
            camera=camera,
            label=label,
            count=1,
            score=0.99,  # camera AI is a binary trigger; nominal high score
            start_time=now,
            end_time=now,
            has_clip=False,
            has_snapshot=False,
        )
        banner = f"{label.replace('_', ' ').capitalize()} (camera AI)"
        has_snapshot = await self._save_live_snapshot(camera, event_id, banner)
        row = await self._db.get_event(event_id)
        if row:
            await self._ws.broadcast({"type": "event_new", "event": row})
        await self._publish_mqtt(camera, label, "new", row)

        ns = self._settings.notifications
        if not ns.get("enabled", True):
            return
        if label not in (ns.get("labels") or []):
            return
        if 0.99 < float(ns.get("min_score", 0.7)):
            return
        if not self._cooldown_ok((camera, label), float(ns.get("cooldown_seconds", 60))):
            return
        self._mark_cooldown((camera, label))
        title = f"{label.replace('_', ' ').capitalize()} detected at {friendly}"
        body = f"{annotate.plural_label(label, 1)} in frame"
        await self._send_notification(
            title=title,
            body=body,
            event_id=event_id,
            tag=f"vigilume-{camera}-{label}",
            icon=ntfy_icon([label]),
            with_image=has_snapshot,
            camera=camera,
            camera_label=friendly,
        )

    # ---------- shared helpers ----------

    async def _save_live_snapshot(self, camera: str, event_id: int, banner: str) -> bool:
        """Grab a live frame (engine frame cache, fallback: camera CGI), add
        a banner, save as the event snapshot. Returns success."""
        # PRIVACY MODE re-check (in-flight defence). This pulls a LIVE frame and
        # falls back to hitting the camera's CGI directly, so it must be gated
        # independently of the engine frame cache being cleared.
        if self._settings.is_private(camera):
            return False
        jpeg = await self._media.latest_jpg(camera, height=720)
        if jpeg is None:
            jpeg = await self._camera_cgi_snapshot(camera)
        if jpeg is None:
            return False
        annotated = await asyncio.to_thread(annotate.banner_only_snapshot, jpeg, banner)
        data = annotated or jpeg
        path = self._snapshots_dir / f"{event_id}.jpg"
        try:
            await asyncio.to_thread(path.write_bytes, data)
        except OSError:
            log.exception("could not write snapshot for event %d", event_id)
            return False
        await self._db.update_event(event_id, has_snapshot=True)
        return True

    async def _camera_cgi_snapshot(self, camera: str) -> Optional[bytes]:
        from .amcrest.client import AmcrestClient, AmcrestError

        cam = await self._db.get_camera(camera)
        if cam is None:
            return None
        client = AmcrestClient(cam["ip"], cam["username"], cam["password"])
        try:
            return await client.snapshot()
        except AmcrestError:
            return None
        finally:
            await client.aclose()

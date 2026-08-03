"""Camera on-board AI event listener (ONVIF PullPoint) for camera-AI gating.

Amcrest/Dahua cameras with on-device AI (the IP8M-2779EW-AI / IP5M-T1277EW-AI
turrets and the AD410 — the ``ai_on_camera`` capability) surface motion / AI
events over **ONVIF**. The Dahua CGI event stream
(``eventManager.cgi?action=attach``) pushes NOTHING on these units (firmware
limitation, validated LIVE against a 1277EW-AI) — but **ONVIF PullPoint events
work**. This module therefore sources camera-AI activity from ONVIF, not CGI.

Validated working pattern (``onvif-zeep``, camera HTTP port 80, admin creds)::

    from onvif import ONVIFCamera
    from datetime import timedelta
    cam = ONVIFCamera(ip, 80, user, password)
    cam.create_events_service()
    pp = cam.create_pullpoint_service()
    msgs = pp.PullMessages({"Timeout": timedelta(seconds=30), "MessageLimit": 100})
    for n in msgs.NotificationMessage:
        # n.Topic._value_1 -> topic string
        #   tns1:RuleEngine/CellMotionDetector/Motion   (SimpleItem IsMotion=true/false)
        #   tns1:VideoSource/MotionAlarm                (SimpleItem State=true/false)
        # n.Message._value_1 -> the tt:Message element with Source/Data SimpleItems.

``PullMessages`` BLOCKS up to its Timeout (zeep is synchronous), so every ONVIF
call runs in a thread (``asyncio.to_thread``) and NEVER on the event loop.
onvif-zeep handles WS-Security auth itself.

Topic → state mapping:
  * A *fire* topic (motion / tamper / line-cross / intrusion / object-/human-/
    vehicle-detect …) with a boolean SimpleItem (``IsMotion`` / ``State``):
    ``true`` → motion **active** (a Start), ``false`` → a Stop.
  * A fire topic with no boolean item → a momentary **Pulse** (IVS tripwire).
  * Any other topic → logged (``[unmapped]``) and ignored.

Label mapping (for ``camera_ai_only`` event creation): a Human/Person topic or
``ObjectType`` item → ``person``, Vehicle/Car → ``car``, else a generic
``motion`` (the server D-FINE then classifies person/car for ``camera_ai``).

Per-camera state tracks an "AI active" window (Start..Stop + a short cooldown)
plus the fired object labels, consumed UNCHANGED by the ingest ``camera_ai``
gate via :meth:`AiEventListener.is_active`. ``camera_ai_only`` cameras create
events directly from a fire via the injected ``on_event`` callback.

Everything is best-effort and MUST NOT crash the app: a camera that is offline,
not ONVIF-capable, or auth-fails logs + backs off and never raises into the
caller. The PullPoint subscription is auto-reconnected with backoff and
best-effort renewed before it expires.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable, Optional

from ..config import effective_detect_mode

log = logging.getLogger(__name__)

# ONVIF host port = the camera HTTP port (validated at 80 on the Amcrest units).
ONVIF_PORT = 80
# PullMessages blocks up to this long waiting for events (run in a thread).
PULL_TIMEOUT_S = 30
# Whole-fleet budget for unsubscribing on shutdown. Well under the restart
# watchdog's teardown deadline so a slow camera costs seconds, not the clean
# shutdown itself (see routers/system.py _TEARDOWN_DEADLINE_S).
_STOP_ALL_TIMEOUT_S = 6.0
# How often to best-effort Renew the PullPoint subscription so it never expires
# out from under us (cameras default to short termination times).
SUBSCRIPTION_RENEW_S = 300.0

# How long a camera stays "AI active" after its last AI Start/Pulse or after a
# Stop — a short tail so brief SMD gaps don't flap the GPU on/off every frame.
AI_ACTIVE_COOLDOWN_S = 8.0
# Safety valve: if a Start (motion true) is seen but its Stop (motion false)
# never arrives (dropped notification), don't hold the detector active forever.
# After this long with no refreshing Start, the pending-Start assumption is
# dropped and only the cooldown window governs activity.
AI_ACTIVE_MAX_S = 300.0

# Failsafe grace window for a camera_ai camera whose ONVIF subscription has
# dropped: a BRIEF reconnect (under this) is trusted, so the gate stays closed
# and the GPU stays idle (the load win). Only a watcher that has NEVER connected
# or has been down LONGER than this trips the failsafe (run detection so a
# genuinely-broken AI trigger can't leave the camera blind).
AI_FAILSAFE_GRACE_S = 120.0

_STREAM_STABLE_S = 60.0
_BACKOFF_MIN_S = 5.0
_BACKOFF_MAX_S = 120.0

# on_event(camera_name, label) — used ONLY for camera_ai_only cameras to create
# an event directly from a camera AI fire.
EventCallback = Callable[[str, str], Awaitable[None]]

# Topic substrings that mean "the camera flagged something" (motion / AI). Only
# a topic matching one of these ever opens the gate; everything else is logged
# and ignored. Extend this list as real cameras reveal new topics via the
# verbose ``ai_event`` log (see AiEventListener._handle_notification).
_FIRE_TOPIC_HINTS = (
    "motion", "motionalarm", "cellmotion", "videomotion", "smartmotion",
    "tamper", "linecross", "crossline", "linedetect", "crossregion",
    "fielddetect", "intrusion", "tripwire", "loitering", "crowd",
    "objectdetect", "peopledetect", "humandetect", "vehicledetect",
    "human", "person", "pedestrian", "vehicle", "car", "face",
)
# SimpleItem names carrying the boolean motion state (true=Start, false=Stop).
_STATE_ITEM_KEYS = (
    "ismotion", "state", "ismotiondetected", "motion", "isinside",
    "isinsideregion", "isactive",
)
# SimpleItem names carrying an object class (Human/Vehicle/…).
_OBJECT_ITEM_KEYS = ("objecttype", "type", "classtype", "objectclass", "label")


# ---------------------------------------------------------------------------
# Legacy Dahua-CGI helpers (kept for reference / label mapping of an explicit
# object type; still unit-tested). The live event SOURCE is ONVIF, below.
# ---------------------------------------------------------------------------


def object_type_from_data(data: Any) -> Optional[str]:
    """Best-effort object class from a Dahua AI event's ``data`` blob.

    IVS events carry ``{"Object": {"ObjectType": "Human"|"Vehicle", ...}}`` (or a
    list under ``Objects``). Returns the normalized capitalized type ("Human" /
    "Vehicle") or None when absent/unknown. Never raises on odd shapes."""
    if not isinstance(data, dict):
        return None
    obj = data.get("Object")
    if not isinstance(obj, dict):
        objs = data.get("Objects")
        if isinstance(objs, list) and objs and isinstance(objs[0], dict):
            obj = objs[0]
        else:
            obj = None
    if isinstance(obj, dict):
        ot = obj.get("ObjectType") or obj.get("Type")
        if isinstance(ot, str) and ot.strip():
            return ot.strip().capitalize()
    return None


def event_labels(code: str, data: Any) -> list[str]:
    """Map a Dahua AI ``Code`` (+ optional ``data`` object type) to detection
    labels (COCO-aligned where possible): Human -> person, Vehicle -> car, else a
    generic "motion". Returns [] for an unrecognized code."""
    if code == "SmartMotionHuman":
        return ["person"]
    if code == "SmartMotionVehicle":
        return ["car"]
    if code in ("CrossLineDetection", "CrossRegionDetection"):
        ot = object_type_from_data(data)
        if ot == "Human":
            return ["person"]
        if ot == "Vehicle":
            return ["car"]
        return ["motion"]
    if code == "VideoMotion":
        return ["motion"]
    return []


# ---------------------------------------------------------------------------
# ONVIF notification parsing + classification
# ---------------------------------------------------------------------------


def _local_name(tag: Any) -> str:
    """Local element name from a possibly namespaced lxml tag (``{ns}Name``).
    Non-string tags (comments/PIs) return ''."""
    if not isinstance(tag, str):
        return ""
    return tag.rpartition("}")[2] if "}" in tag else tag


def _extract_topic(notif: Any) -> str:
    """Topic string of an ONVIF NotificationMessage (``Topic._value_1``).

    VERIFIED LIVE (Amcrest IP5M-T1277): zeep does NOT populate the Topic text on
    these cameras (``_value_1`` is None), so return "" rather than a stringified
    object — classification then falls back to the SimpleItems, which DO parse
    cleanly. When a firmware does provide the text, it's used for object labels."""
    topic = getattr(notif, "Topic", None)
    if topic is None:
        return ""
    val = getattr(topic, "_value_1", None)
    if not isinstance(val, str):
        return ""
    return val.strip()


def _extract_simple_items(notif: Any) -> dict[str, str]:
    """Flatten every ``tt:SimpleItem`` (Name -> Value) in a NotificationMessage's
    Message body. Handles both the common lxml-element form
    (``Message._value_1`` is a raw ``tt:Message`` element, since the Message
    field is ``xsd:any``) and a zeep-object form (Source/Key/Data.SimpleItem).
    Never raises."""
    items: dict[str, str] = {}
    msg = getattr(notif, "Message", None)
    inner = getattr(msg, "_value_1", msg) if msg is not None else None
    if inner is None:
        return items
    try:
        # lxml element path: walk descendants for SimpleItem Name/Value attrs.
        if hasattr(inner, "iter") and callable(getattr(inner, "iter")):
            for el in inner.iter():
                if _local_name(getattr(el, "tag", None)) != "SimpleItem":
                    continue
                get = getattr(el, "get", None)
                if not callable(get):
                    continue
                name = get("Name")
                if name is not None:
                    items[str(name)] = None if get("Value") is None else str(get("Value"))
            return items
        # zeep-object path: Source / Key / Data each hold a list of SimpleItem.
        for section in ("Source", "Key", "Data"):
            sec = getattr(inner, section, None)
            if sec is None:
                continue
            sis = getattr(sec, "SimpleItem", None) or []
            if not isinstance(sis, (list, tuple)):
                sis = [sis]
            for si in sis:
                name = getattr(si, "Name", None)
                if name is not None:
                    val = getattr(si, "Value", None)
                    items[str(name)] = None if val is None else str(val)
    except Exception:  # noqa: BLE001 — a weird body must not kill the stream
        log.debug("ai-events: failed to parse SimpleItems from notification", exc_info=True)
    return items


def parse_notification(notif: Any) -> tuple[str, dict[str, str]]:
    """Parse one ONVIF NotificationMessage into ``(topic, {SimpleItem: value})``.
    Best-effort and total — returns ``("", {})`` on anything unexpected."""
    return _extract_topic(notif), _extract_simple_items(notif)


def _is_fire_topic(topic: str) -> bool:
    t = topic.lower()
    return any(hint in t for hint in _FIRE_TOPIC_HINTS)


def _state_bool(items: dict[str, str]) -> Optional[bool]:
    """Boolean motion state from a notification's SimpleItems, or None when no
    recognized state item is present (a momentary event)."""
    for key, val in items.items():
        if key.lower() in _STATE_ITEM_KEYS:
            return str(val).strip().lower() in ("true", "1")
    return None


def notification_labels(topic: str, items: dict[str, str]) -> list[str]:
    """Detection label(s) for a fire: an explicit object-type SimpleItem or a
    human/vehicle topic maps to person/car; otherwise a generic "motion" (the
    server detector refines it for ``camera_ai``)."""
    for key, val in items.items():
        if key.lower() in _OBJECT_ITEM_KEYS:
            vt = (val or "").strip().lower()
            if vt in ("human", "person", "pedestrian"):
                return ["person"]
            if vt in ("vehicle", "car", "truck", "bus", "motor", "motorcycle"):
                return ["car"]
    t = topic.lower()
    if "human" in t or "person" in t or "pedestrian" in t:
        return ["person"]
    if "vehicle" in t or "car" in t:
        return ["car"]
    return ["motion"]


def classify_notification(
    topic: str, items: dict[str, str]
) -> tuple[Optional[str], list[str]]:
    """Map a parsed ONVIF notification to an action + labels.

    Returns ``(action, labels)`` where action is:
      * ``"start"`` — a fire topic reporting motion active (``IsMotion``/``State``
        true). Opens a Start..Stop window.
      * ``"stop"``  — a fire topic reporting motion cleared (bool false).
      * ``"pulse"`` — a fire topic with no boolean state (momentary IVS event).
      * ``None``    — not a recognized fire topic (caller logs it ``[unmapped]``).
    """
    labels = notification_labels(topic, items)
    # Gate on a real ONVIF fire topic (when the firmware exposes the Topic text)
    # OR a motion-SPECIFIC SimpleItem (``IsMotion`` from CellMotionDetector).
    # Amcrest cams don't expose the Topic text via zeep (verified live), so
    # IsMotion is the dependable motion signal there. A GENERIC ``State`` item
    # alone (HardwareFailure, DigitalInput, …) must NOT fire detection, so we do
    # not gate on it without a fire topic.
    has_motion_item = any(k.lower() == "ismotion" for k in items)
    if not _is_fire_topic(topic) and not has_motion_item:
        return None, labels
    b = _state_bool(items)
    if b is False:
        return "stop", labels
    if b is True:
        return "start", labels
    return "pulse", labels  # fire topic, no boolean -> momentary


def _summarize_items(items: dict[str, str]) -> str:
    """Compact one-line rendering of the most relevant SimpleItems for the
    verbose ``ai_event`` log (State/IsMotion + object type + rule name)."""
    if not items:
        return ""
    prefer = _STATE_ITEM_KEYS + _OBJECT_ITEM_KEYS + ("rule", "name")
    lower = {k.lower(): (k, v) for k, v in items.items()}
    parts: list[str] = []
    for want in prefer:
        if want in lower:
            k, v = lower.pop(want)
            parts.append(f"{k}={v}")
    for k, v in lower.values():  # any remaining, bounded
        if len(parts) >= 4:
            break
        parts.append(f"{k}={v}")
    return " " + " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Per-camera AI activity state (consumed UNCHANGED by the ingest gate)
# ---------------------------------------------------------------------------


@dataclass
class AiCameraState:
    """Per-camera AI activity window + fired labels (see module docstring)."""

    # Topics currently in a Start..Stop window (awaiting their false/Stop).
    started: set[str] = field(default_factory=set)
    # Monotonic instant the active window is guaranteed to last until (extended
    # on each Pulse and on a Stop). Governs activity once ``started`` is empty.
    active_until: float = 0.0
    # Object labels fired within the current/recent active window.
    labels: set[str] = field(default_factory=set)
    # Monotonic time of the last AI Start/Pulse (an actual detection).
    last_fire_monotonic: float = 0.0
    # Total AI fires seen (Start + Pulse) — observability/status.
    fire_count: int = 0
    # Whether the ONVIF PullPoint subscription is currently established.
    connected: bool = False
    # Whether it has EVER connected. A never-connected watcher = broken/
    # unreachable ONVIF, so the failsafe still runs detection (don't go blind).
    ever_connected: bool = False
    # Monotonic time the subscription last became connected — drives the failsafe
    # grace window so a BRIEF reconnect is trusted (gate stays closed = the load
    # win) while a long outage still trips the failsafe.
    last_connected_monotonic: float = 0.0

    def on_fire(self, code: str, labels: list[str], now: float, momentary: bool) -> None:
        """Record a Start (``momentary=False``) or Pulse (``momentary=True``)."""
        # Fresh active window -> start its label set clean, so status doesn't
        # report the union of every object type ever fired across windows.
        if not self.active(now):
            self.labels.clear()
        self.labels.update(labels)
        self.last_fire_monotonic = now
        self.fire_count += 1
        self.active_until = max(self.active_until, now + AI_ACTIVE_COOLDOWN_S)
        if not momentary:
            self.started.add(code)

    def on_stop(self, code: str, now: float) -> None:
        self.started.discard(code)
        if not self.started:
            self.active_until = max(self.active_until, now + AI_ACTIVE_COOLDOWN_S)

    def active(self, now: float) -> bool:
        """True while any AI code is in a Start window (bounded by the
        missed-Stop safety valve) OR within the post-fire/stop cooldown."""
        if self.started and (now - self.last_fire_monotonic) <= AI_ACTIVE_MAX_S:
            return True
        return now < self.active_until

    def active_labels(self, now: float) -> list[str]:
        return sorted(self.labels) if self.active(now) else []


# ---------------------------------------------------------------------------
# ONVIF PullPoint watcher (one per camera)
# ---------------------------------------------------------------------------


# NotificationHandler(camera_name, topic, items) — dispatched per parsed ONVIF
# notification. Owned by the AiEventListener.
NotificationHandler = Callable[[str, str, dict[str, str]], Awaitable[None]]


# The one xaddr key that decides whether create_pullpoint_service() works.
_PULLPOINT_NS = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"


def _build_onvif_camera(
    ip: str, port: int, username: str, password: str, *, adjust_time: bool = False
) -> Any:
    """Construct an ``onvif.ONVIFCamera`` (lazy import so the app / web-only code
    paths — and tests — import this module without onvif-zeep loaded). Blocking:
    the constructor talks to the camera (GetCapabilities + CreatePullPoint), so
    callers run it in a thread. Overridable in tests (monkeypatch this name).

    ALSO REPAIRS onvif-zeep's most misleading error. ``ONVIFCamera.__init__``
    calls ``update_xaddrs()``, which ends with::

        try:
            self.event = self.create_events_service()
            self.xaddrs[_PULLPOINT_NS] = (
                self.event.CreatePullPointSubscription()
                    .SubscriptionReference.Address._value_1)
        except:          # <-- BARE. discards the real exception.
            pass

    That key is written in exactly ONE place in the whole library, and
    ``create_pullpoint_service()`` does nothing but look it up — so its
    ``ONVIFError("Device doesn`t support service: pullpoint")`` does NOT mean the
    camera lacks PullPoint. It means those two statements threw and nobody
    recorded why. Fleet-wide that message sent us hunting a firmware capability
    that was never missing.

    So: if the key is absent, re-run the same two statements HERE, outside the
    bare except, and let the true exception propagate. In the healthy case the
    key is already present and this is a pure no-op — no extra device traffic
    and no extra subscription.
    """
    from onvif import ONVIFCamera  # noqa: PLC0415 — deliberate lazy import

    cam = ONVIFCamera(ip, port, username, password, adjust_time=adjust_time)
    if _PULLPOINT_NS not in getattr(cam, "xaddrs", {}):
        events = cam.create_events_service()
        subscription = events.CreatePullPointSubscription()
        cam.xaddrs[_PULLPOINT_NS] = subscription.SubscriptionReference.Address._value_1
    return cam


class OnvifAiWatcher:
    """One camera's long-lived ONVIF PullPoint event loop (reconnect + backoff).

    Connects to the camera's ONVIF events service, creates a PullPoint
    subscription, then loops ``PullMessages`` (each call bounded by
    ``PULL_TIMEOUT_S`` and run in a thread so it never blocks the event loop),
    dispatching each notification to the manager's async handler. Never raises
    out of :meth:`_run` — a camera that is offline / not ONVIF-capable /
    auth-fails logs + backs off. Subscriptions are best-effort renewed and, on
    any fault, re-established from scratch."""

    def __init__(
        self,
        name: str,
        ip: str,
        username: str,
        password: str,
        state: AiCameraState,
        on_notification: NotificationHandler,
    ):
        self.name = name
        self._ip = ip
        self._username = username
        self._password = password
        self._state = state
        self._on_notification = on_notification
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"ai-events:{self.name}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        backoff = _BACKOFF_MIN_S
        while True:
            started = time.monotonic()
            try:
                await self._connect_and_pull()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — watcher must never die
                # Offline / not-ONVIF / auth failure / expired subscription all
                # land here: log at info (expected on a flaky camera) and retry.
                log.info(
                    "ai-events %s: ONVIF event loop ended (%s: %s); reconnecting",
                    self.name, exc.__class__.__name__, exc,
                )
            finally:
                self._state.connected = False
            if time.monotonic() - started > _STREAM_STABLE_S:
                backoff = _BACKOFF_MIN_S
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)

    async def _connect_and_pull(self) -> None:
        """Establish the PullPoint subscription then pull-loop until it faults.
        All blocking ONVIF calls run in threads."""
        # Two-step, mirroring the already-validated ladder in
        # speaker_probe.probe_speaker_onvif. ONVIF WS-Security authenticates with
        # a *timestamp* (Created + Nonce + digest), so a camera whose clock or
        # timezone index is wrong rejects the token — and the rejection lands on
        # the FIRST authenticated call, which is CreatePullPointSubscription.
        # GetCapabilities is PRE_AUTH and still succeeds, which is why the
        # failure looks like "capability missing" instead of "auth refused".
        #
        # adjust_time=True makes onvif-zeep read the device's own clock and
        # offset our request timestamps to match. Note every OTHER camera path we
        # use (CGI, doorbell) is HTTP Digest — challenge/nonce based and entirely
        # clock-independent — which is exactly why those keep working while ONVIF
        # alone breaks fleet-wide.
        try:
            cam = await asyncio.to_thread(
                _build_onvif_camera, self._ip, ONVIF_PORT, self._username, self._password
            )
        except Exception:  # noqa: BLE001 — retry once with the device's own clock
            log.info(
                "ai-events %s: ONVIF connect failed; retrying with adjust_time=True "
                "(device clock/timezone suspected)",
                self.name, exc_info=True,
            )
            cam = await asyncio.to_thread(
                functools.partial(
                    _build_onvif_camera,
                    self._ip, ONVIF_PORT, self._username, self._password,
                    adjust_time=True,
                )
            )
        pp = await asyncio.to_thread(self._subscribe, cam)
        self._state.connected = True
        self._state.ever_connected = True
        self._state.last_connected_monotonic = time.monotonic()
        log.info("ai-events %s: ONVIF PullPoint subscription established", self.name)
        last_renew = time.monotonic()
        try:
            while True:
                msgs = await asyncio.to_thread(self._pull, pp)
                await self._dispatch(msgs)
                now = time.monotonic()
                if now - last_renew >= SUBSCRIPTION_RENEW_S:
                    await asyncio.to_thread(self._renew, pp)
                    last_renew = now
        finally:
            self._state.connected = False
            try:
                await asyncio.to_thread(self._unsubscribe, pp)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    # -- blocking ONVIF calls (each invoked via asyncio.to_thread) --

    @staticmethod
    def _subscribe(cam: Any) -> Any:
        cam.create_events_service()
        return cam.create_pullpoint_service()

    @staticmethod
    def _pull(pp: Any) -> Any:
        # NOTE: onvif-zeep wants a positional DICT here (not kwargs); Timeout is
        # a datetime.timedelta. PullMessages blocks up to Timeout.
        return pp.PullMessages(
            {"Timeout": timedelta(seconds=PULL_TIMEOUT_S), "MessageLimit": 100}
        )

    @staticmethod
    def _renew(pp: Any) -> None:
        """Renew the subscription. DELIBERATELY LETS FAILURES RAISE.

        This used to swallow the exception, which was the one genuine fail-CLOSED
        hole in this module: ``state.connected`` is set True once at subscribe
        time, so a subscription the device had already dropped kept reporting
        connected forever. ``failsafe_needed()`` then returned False and the
        ingest camera_ai gate stayed CLOSED on a dead trigger — i.e. the camera
        silently stopped being watched at all. Blind is the one outcome this
        system must never reach.

        Raising instead reaches _run's handler, clears ``connected`` in its
        finally, and after AI_FAILSAFE_GRACE_S the failsafe trips and detection
        resumes. A transient blip costs one reconnect and stays inside the grace
        window, so this does not make the gate flap."""
        pp.Renew({"TerminationTime": f"PT{int(SUBSCRIPTION_RENEW_S * 2)}S"})

    @staticmethod
    def _unsubscribe(pp: Any) -> None:
        pp.Unsubscribe()

    async def _dispatch(self, msgs: Any) -> None:
        notifs = getattr(msgs, "NotificationMessage", None) or []
        if not isinstance(notifs, (list, tuple)):
            notifs = [notifs]
        for notif in notifs:
            topic, items = parse_notification(notif)
            try:
                await self._on_notification(self.name, topic, items)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a bad event never kills the loop
                log.exception("ai-events %s: notification handler failed", self.name)


# ---------------------------------------------------------------------------
# Listener: one watcher per camera-AI camera; feeds the ingest gate
# ---------------------------------------------------------------------------


class AiEventListener:
    """Keeps one :class:`OnvifAiWatcher` per camera whose ``ai_on_camera``
    capability is set AND whose effective detect mode is ``camera_ai`` or
    ``camera_ai_only``; owns the per-camera :class:`AiCameraState`.

    The ingest worker calls :meth:`is_active` / :meth:`is_connected` to gate GPU
    inference; the status API reads :meth:`status`. For ``camera_ai_only``
    cameras an AI fire additionally invokes the injected ``on_event`` callback to
    create an event. ``sync`` reconciles watchers against the camera rows (called
    at boot + after camera CRUD + after settings.default_mode changes)."""

    def __init__(self, on_event: EventCallback, settings: Optional[Any] = None):
        self._on_event = on_event
        # SettingsStore, read purely for the live Privacy Mode set. Optional so
        # tests can construct a listener standalone; None reads as "nothing
        # private" (fail-open for harnesses — production always passes it).
        self._settings = settings
        self._watchers: dict[str, OnvifAiWatcher] = {}
        self._states: dict[str, AiCameraState] = {}
        # name -> effective detect mode ("camera_ai" | "camera_ai_only").
        self._modes: dict[str, str] = {}
        # name -> connection tuple, so a creds/ip change restarts the watcher.
        self._keys: dict[str, tuple] = {}

    @staticmethod
    def _has_ai(cam: dict[str, Any]) -> bool:
        return bool((cam.get("capabilities") or {}).get("ai_on_camera"))

    async def sync(self, cameras: list[dict[str, Any]], default_mode: str = "always") -> None:
        """Reconcile watchers with the camera rows. Only cameras with on-board
        AI running in a camera-AI mode get a watcher; a mode change to/from
        ``always`` starts/stops the watcher. Never raises."""
        wanted: dict[str, dict[str, Any]] = {}
        for cam in cameras:
            if not self._has_ai(cam):
                continue
            # PRIVACY MODE GATE (app/privacy.py). A private camera gets no
            # watcher, so its on-board AI events never reach the pipeline. This
            # is the path the detection gate CANNOT cover: a `camera_ai_only`
            # camera never spawns an ingest source, so stopping ingest does
            # nothing for it — ungated, it would keep writing events and live
            # snapshots while the operator believes the camera is blacked out.
            # Gating inside sync() (not just stop_all) means a later camera CRUD
            # or default_mode change can't silently respawn the watcher.
            if self._settings is not None and self._settings.is_private(cam["name"]):
                continue
            mode = effective_detect_mode(cam.get("detect_mode"), default_mode)
            if mode in ("camera_ai", "camera_ai_only"):
                wanted[cam["name"]] = cam

        # stop removed/changed watchers
        for name in list(self._watchers):
            cam = wanted.get(name)
            key = self._conn_key(cam) if cam is not None else None
            if cam is None or key != self._keys.get(name):
                await self._watchers[name].stop()
                del self._watchers[name]
                self._keys.pop(name, None)
                self._states.pop(name, None)

        # start/refresh wanted watchers
        for name, cam in wanted.items():
            self._modes[name] = effective_detect_mode(cam.get("detect_mode"), default_mode)
            if name not in self._watchers:
                state = AiCameraState()
                self._states[name] = state
                watcher = OnvifAiWatcher(
                    name, cam["ip"], cam["username"], cam["password"],
                    state, self._handle_notification,
                )
                self._keys[name] = self._conn_key(cam)
                self._watchers[name] = watcher
                watcher.start()

        # drop mode entries for cameras no longer wanted
        for name in list(self._modes):
            if name not in wanted:
                del self._modes[name]

    @staticmethod
    def _conn_key(cam: dict[str, Any]) -> tuple:
        return (cam["ip"], cam["username"], cam["password"])

    async def stop_all(self) -> None:
        # CONCURRENTLY, AND BOUNDED — this runs on the shutdown path.
        #
        # Cancelling a watcher lands in _connect_and_pull's `finally`, which
        # awaits an ONVIF Unsubscribe. An await inside a `finally` during
        # cancellation runs to completion (asyncio does not re-deliver
        # CancelledError unless cancel() is called again), and that SOAP POST
        # has no client-side timeout. Serially, against cameras that may be
        # offline or wedged, that is an unbounded stall multiplied by the fleet
        # size — with the restart watchdog now in place it would mean every
        # restart force-exits and orphans ffmpeg instead of tearing down
        # cleanly.
        #
        # An unsubscribe we skip is not a leak we own: the subscription carries
        # its own termination time and the camera expires it. Blocking shutdown
        # to be polite about it is the worse trade.
        if self._watchers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(w.stop() for w in self._watchers.values()),
                        return_exceptions=True,
                    ),
                    timeout=_STOP_ALL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "ai-events: %d watcher(s) did not unsubscribe within %.0fs — "
                    "abandoning; the camera expires the subscription on its own",
                    len(self._watchers), _STOP_ALL_TIMEOUT_S,
                )
        self._watchers.clear()
        self._states.clear()
        self._modes.clear()
        self._keys.clear()

    # ---------- event dispatch ----------

    async def _handle_notification(
        self, name: str, topic: str, items: dict[str, str]
    ) -> None:
        state = self._states.get(name)
        if state is None:
            return
        action, labels = classify_notification(topic, items)
        # Verbose per-notification logging (grep prefix "ai_event"): one INFO
        # line per ONVIF notification so real events are visible via
        #   docker compose logs backend | grep ai_event
        # Topics this listener does not treat as a fire are logged too, tagged
        # "[unmapped]", so the operator can see exactly what topics the real
        # cameras emit and extend _FIRE_TOPIC_HINTS if needed.
        log.info(
            "ai_event %s: topic=%s%s%s",
            name,
            topic or "?",
            _summarize_items(items),
            "" if action else " [unmapped]",
        )
        if action is None:
            return
        now = time.monotonic()
        if action == "stop":
            state.on_stop(topic, now)
            return
        momentary = action == "pulse"
        # topic is the Start..Stop tracking key (its matching false clears it).
        state.on_fire(topic, labels, now, momentary=momentary)
        if self._modes.get(name) == "camera_ai_only":
            for label in labels:
                try:
                    await self._on_event(name, label)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — event creation must not kill the loop
                    log.exception("ai-events %s: on_event(%s) failed", name, label)

    # ---------- state queries (ingest gate + status API) ----------

    def is_active(self, name: str) -> bool:
        state = self._states.get(name)
        return bool(state and state.active(time.monotonic()))

    def is_connected(self, name: str) -> bool:
        """Whether a watcher exists for this camera AND its ONVIF PullPoint
        subscription is currently established. False when no watcher runs, or the
        subscription never connected / has dropped. The ingest ``camera_ai`` gate
        uses this for its watcher-down failsafe: a camera whose AI listener is
        not connected must run detection rather than gate the GPU off on a dead
        signal (see IngestManager._should_infer)."""
        state = self._states.get(name)
        return bool(state and state.connected)

    def failsafe_needed(self, name: str) -> bool:
        """Whether a camera_ai camera should run detection DESPITE no motion —
        i.e. its ONVIF AI trigger is UNRELIABLE. True when: no watcher exists,
        the subscription has NEVER connected (broken/unreachable ONVIF), or it
        has been down LONGER than ``AI_FAILSAFE_GRACE_S``. A connected watcher —
        or one whose subscription dropped only BRIEFLY (a normal reconnect) — is
        trusted, so the gate stays closed and the GPU idles (the load win)."""
        state = self._states.get(name)
        if state is None:
            return True  # no watcher wired for this camera -> can't trust idle
        if state.connected:
            return False  # healthy -> trust the AI gate
        if not state.ever_connected:
            return True  # never established -> broken ONVIF, don't go blind
        return (time.monotonic() - state.last_connected_monotonic) > AI_FAILSAFE_GRACE_S

    def active_labels(self, name: str) -> list[str]:
        state = self._states.get(name)
        return state.active_labels(time.monotonic()) if state else []

    def status(self, name: str) -> Optional[dict[str, Any]]:
        """Per-camera AI status for the API, or None when no watcher runs for
        this camera (mode ``always`` / no on-board AI)."""
        state = self._states.get(name)
        if state is None:
            return None
        now = time.monotonic()
        last_age = (
            round(now - state.last_fire_monotonic, 2)
            if state.last_fire_monotonic > 0
            else None
        )
        return {
            "mode": self._modes.get(name),
            "connected": state.connected,
            "ai_active": state.active(now),
            "ai_labels": state.active_labels(now),
            "fire_count": state.fire_count,
            "last_fire_age_s": last_age,
        }

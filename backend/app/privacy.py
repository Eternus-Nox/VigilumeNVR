"""Software Privacy Mode — per-camera / per-group capture kill switch.

This REPLACED the Amcrest hardware lens mask, which drove the camera's own
LeLensMask blackout — device state that outlived Vigilume and could only be
undone from the camera's own web UI. Nothing here touches the camera.
Software Privacy Mode stops ALL Vigilume capture for the affected cameras —
recording, detection, events/snapshots/notifications, live view, audio, and
on-camera-AI ingestion — WITHOUT touching the camera at all.

SOURCE OF TRUTH. Persisted under its OWN settings KV key (`privacy_mode`), a
`{"cameras": [names], "groups": [ids]}` document — deliberately NOT inside the
`app_settings` blob, so a full-replace `PUT /api/settings` (which does not model
privacy) can never silently wipe it and resume capture. A camera is effectively
private if it is listed directly OR belongs to a group that is listed.

HOT PATH. Every capture gate reads the RESOLVED set `app.state.private_cameras`
(a frozenset of camera names), never the raw doc — resolution needs the DB
(group membership) and must not run per frame. `refresh(state)` recomputes and
rebinds it; call it after any privacy / group / camera change so group-membership
edits propagate. Rebinding a whole frozenset is atomic for a synchronous reader.

FAIL-CLOSED. The set is loaded from persistence at boot, so a camera that was
private stays private across a backend restart until an admin clears it. Default
on a fresh volume is empty (capture on) — never blackout-by-default.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Its own KV key — see the PUT-wipe note above. Never fold into `app_settings`.
_KEY = "privacy_mode"


async def load_raw(db: Any) -> dict[str, Any]:
    """The persisted privacy document, defensively normalized. Unknown/deleted
    camera names and group ids are kept verbatim in storage (like camera_groups)
    so a camera recreated with the same name re-inherits privacy; they are
    filtered out only when RESOLVING to real cameras."""
    stored = await db.get_setting(_KEY)
    cams = stored.get("cameras") if isinstance(stored, dict) else None
    grps = stored.get("groups") if isinstance(stored, dict) else None
    return {
        "cameras": [c for c in cams if isinstance(c, str)] if isinstance(cams, list) else [],
        "groups": [g for g in grps if isinstance(g, int) and not isinstance(g, bool)]
        if isinstance(grps, list)
        else [],
    }


async def save_raw(db: Any, raw: dict[str, Any]) -> None:
    await db.set_setting(
        _KEY,
        {"cameras": list(raw.get("cameras", [])), "groups": list(raw.get("groups", []))},
    )


async def resolve(db: Any, raw: dict[str, Any]) -> frozenset[str]:
    """Effective private camera set: directly-listed names ∪ the members of every
    listed group, intersected with cameras that actually exist right now."""
    known = {c["name"] for c in await db.list_cameras()}
    private: set[str] = {c for c in raw.get("cameras", []) if c in known}
    for gid in raw.get("groups", []):
        group = await db.get_group(gid)
        if group:
            private.update(n for n in group.get("cameras", []) if n in known)
    return frozenset(private)


async def refresh(state: Any) -> frozenset[str]:
    """Recompute the resolved private set from persistence + current
    groups/cameras and rebind it atomically. Call after any privacy / group /
    camera change.

    Publishes to BOTH readers, and they must never diverge:
    - `state.settings` (the SettingsStore) — what the capture subsystems gate on
      (recorder, engine, ingest, events pipeline, go2rtc all hold this object).
    - `state.private_cameras` — convenience for routers, which hold app.state
      rather than the store.
    """
    resolved = await resolve(state.db, await load_raw(state.db))
    settings = getattr(state, "settings", None)
    if settings is not None:
        settings.set_private_cameras(resolved)
    state.private_cameras = resolved
    return resolved


def current(state: Any) -> frozenset[str]:
    """The live effective private set for synchronous hot-path reads. Empty until
    the first refresh() (fail-safe: unset reads as 'nothing private', never a
    blanket block)."""
    return getattr(state, "private_cameras", frozenset())


def is_private(state: Any, camera: str) -> bool:
    return camera in current(state)


async def apply(state: Any) -> None:
    """THE TRIGGER. Fan out to every capture subsystem so a privacy change takes
    effect LIVE — no app restart, no camera reconfiguration. Mirrors
    cameras._apply_camera_change: the same reconcilers that react to a camera
    CRUD, run after `state.private_cameras` has been refreshed, so each subtracts
    the now-private cameras from what it captures.

    Ordering: stop the producers (go2rtc live streams, ingest/detect, recorder,
    on-camera-AI watchers) BEFORE cancelling in-flight consumers (pipeline
    tasks), so nothing new is produced while we drain. Await everything before
    the caller acks — a fire-and-forget return would leave footage flowing during
    the stop window. None of these raise (a down go2rtc must not fail the toggle);
    they log and continue.

    NB the reconcilers only actually SUBTRACT private cameras once their gates
    (stage 2) read `state.private_cameras`. Until then this is a harmless normal
    reconcile — wiring the trigger here first keeps the two halves decoupled.
    """
    await state.go2rtc.apply()
    await state.engine.reload()
    await state.recorder.reload()
    ai_events = getattr(state, "ai_events", None)
    if ai_events is not None:
        cameras = await state.db.list_cameras()
        default_mode = state.settings.detection.get("default_mode", "always")
        await ai_events.sync(cameras, default_mode=default_mode)
    # Cancel enrichment/notify tasks already in flight for a now-private camera.
    pipeline = getattr(state, "pipeline", None)
    if pipeline is not None:
        await pipeline.shutdown()

    # TELL THE CLIENTS. Everything above is server-side teardown; without this
    # broadcast no connected client learns anything changed.
    #
    # The symptom that makes this load-bearing rather than tidy: a client caches
    # `private` per camera from GET /api/cameras, and the status prober keeps
    # reporting the camera ONLINE (privacy is a software gate — the camera's own
    # RTSP port is still open). So a dashboard tile still satisfies
    # "online && !private", keeps its live player mounted, and points it at
    # go2rtc streams that `state.go2rtc.apply()` has just removed. The player
    # then retries forever: Privacy Mode reads as a permanently loading camera
    # rather than a deliberate one.
    #
    # `cameras_changed` (not a bespoke privacy message) so every client refetches
    # the camera list and re-derives `private` from one source of truth — and so
    # clients that predate this still do the right thing.
    #
    # LAST, and best-effort: a WS hiccup must never fail a privacy toggle, and
    # the ack must not precede the actual teardown above.
    ws = getattr(state, "ws", None)
    if ws is not None:
        try:
            await ws.broadcast({"type": "cameras_changed"})
        except Exception:  # noqa: BLE001
            log.exception("privacy: cameras_changed broadcast failed")


async def on_change(state: Any) -> None:
    """Call after a GROUP mutation (membership/delete). Refreshes the resolved
    set and, ONLY if the effective private set actually changed, drives the live
    reconcile — so editing a private group's members takes effect without a
    restart, while an unrelated edit (rename, reorder) doesn't needlessly bounce
    live view. (Camera CRUD already reconciles via _apply_camera_change, which
    calls refresh() itself, so it doesn't use this.)"""
    before = current(state)
    if await refresh(state) != before:
        await apply(state)

"""Settings KV store: stored JSON merged over defaults from CONTRACTS.md.

Kept in memory for hot paths (the event pipeline reads it on every engine
payload); persisted to the settings table on update.
"""
from __future__ import annotations

import copy
from typing import Any

from .config import DEFAULT_SETTINGS
from .db import Database

_KEY = "app_settings"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _strip_legacy(settings: dict) -> dict:
    """Silently drop / migrate settings blocks from removed or reworked features
    so an old /data volume never 500s: APNs `direct` mode (retired — see below),
    detection.audio_events/audio_labels (legacy audio classifier removed with the
    standalone-native rewrite; documented as roadmap), and the time_sync block's
    old NTP-based keys (see below).

    NOTE: `notifications.ntfy` USED to be dropped here (ntfy support was removed
    once). ntfy is a supported channel again — it is how a self-hoster gets push
    without an Apple developer account — so the block is now preserved. Do not
    re-add a pop("ntfy") without also deleting the ntfy settings model.
    """
    notifications = settings.get("notifications")
    if isinstance(notifications, dict):
        apns = notifications.get("apns")
        if isinstance(apns, dict):
            # APNs "direct" (this server holding its own Apple .p8) is RETIRED;
            # the relay is the only APNs transport again, because ntfy — which
            # briefly replaced it — cannot ring a doorbell (its alerts land in
            # the ntfy app: no CallKit, no native UI).
            #
            # A stored mode="direct" MUST be migrated here, not merely
            # tolerated: `mode` is a pydantic Literal, so once "direct" leaves
            # it an unmigrated blob makes EVERY PUT/PATCH /api/settings 422 —
            # locking the admin out of the settings page entirely, including
            # out of changing the mode. Migrating on load AND on save means the
            # dead value can never reach the validator.
            #
            # -> "off", never -> "relay": relay with an empty relay_url errors
            # on every event, and we must never silently start pushing through
            # a relay the admin has not configured. This DOES mean APNs push
            # stops until they set mode=relay + relay_url (ntfy and web push
            # are unaffected) — a visible, fixable stop beats a silent one.
            if apns.get("mode") == "direct":
                apns["mode"] = "off"
            # `direct` held the .p8. Dropping it is intentional and is why the
            # key was rescued to secrets/ first: pydantic's extra="ignore"
            # would silently discard the block on the next save of ANY setting
            # anyway, so leaving it here would only pretend it was safe.
            apns.pop("direct", None)
    detection = settings.get("detection")
    if isinstance(detection, dict):
        detection.pop("audio_events", None)
        detection.pop("audio_labels", None)
    time_sync = settings.get("time_sync")
    if isinstance(time_sync, dict):
        # The feature used to enable the camera NTP client (keyed on `auto_ntp`
        # + an `ntp_server`); it now pushes the local time directly and DISABLES
        # NTP. Migrate the old enable flag onto `auto_sync` so an operator who
        # turned it off stays off, and drop the dead ntp_server key.
        if "auto_ntp" in time_sync:
            time_sync["auto_sync"] = bool(time_sync.pop("auto_ntp"))
        time_sync.pop("ntp_server", None)
    return settings


class SettingsStore:
    def __init__(self, db: Database, env_public_url: str = ""):
        self._db = db
        self._env_public_url = env_public_url
        self._cached: dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        # Software Privacy Mode: the RESOLVED set of cameras currently private
        # (direct ∪ group members — see app/privacy.py). Runtime-only, never
        # persisted here and never part of the settings document: privacy has
        # its own KV key precisely so a full-replace PUT /api/settings can't
        # wipe it.
        #
        # WHY IT LIVES ON THE SETTINGS STORE. Every capture subsystem
        # (recorder, engine, ingest, events pipeline, go2rtc) already holds this
        # object, so it is the ONE place they can all read the current set.
        # The alternative — threading a `private` argument into each
        # reload()/apply() — is a LEAK WAITING TO HAPPEN: those reconcilers are
        # called from many unrelated places (camera CRUD, settings writes,
        # detection changes), and any caller that omitted the argument would
        # reset the gate to empty and silently resume capture on a camera the
        # operator believes is private. A single shared source cannot drift.
        self._private_cameras: frozenset[str] = frozenset()

    async def load(self) -> None:
        stored = await self._db.get_setting(_KEY)
        merged = _strip_legacy(_deep_merge(DEFAULT_SETTINGS, stored or {}))
        # PUBLIC_URL env seeds system.public_url until the operator sets one
        # in the UI (stored non-empty value wins).
        if not merged["system"].get("public_url") and self._env_public_url:
            merged["system"]["public_url"] = self._env_public_url
        self._cached = merged

    def get(self) -> dict[str, Any]:
        return copy.deepcopy(self._cached)

    @property
    def current(self) -> dict[str, Any]:
        """Read-only hot-path access (do not mutate)."""
        return self._cached

    async def update(self, new_settings: dict[str, Any]) -> dict[str, Any]:
        merged = _strip_legacy(_deep_merge(DEFAULT_SETTINGS, new_settings))
        self._cached = merged
        await self._db.set_setting(_KEY, merged)
        return copy.deepcopy(merged)

    # ---------- Software Privacy Mode (runtime-only; app/privacy.py owns it) ----------

    @property
    def private_cameras(self) -> frozenset[str]:
        """Cameras currently in Privacy Mode. Rebound as a whole frozenset by
        `privacy.refresh()`, so a synchronous reader always sees a consistent
        set (never a half-updated one)."""
        return self._private_cameras

    def set_private_cameras(self, cameras: frozenset[str]) -> None:
        """Called ONLY by privacy.refresh(). Atomic whole-set rebind."""
        self._private_cameras = cameras

    def is_private(self, camera: str) -> bool:
        """THE capture gate. Every subsystem asks this before capturing for a
        camera: recording, detection, events/snapshots/notifications, live
        streams, the snapshot route, talk, and on-camera-AI ingestion.
        Re-read it every cycle — never cache it across a toggle."""
        return camera in self._private_cameras

    # convenience accessors used by the pipeline
    @property
    def notifications(self) -> dict[str, Any]:
        return self._cached["notifications"]

    @property
    def detection(self) -> dict[str, Any]:
        return self._cached["detection"]

    @property
    def recording(self) -> dict[str, Any]:
        return self._cached["recording"]

    @property
    def public_url(self) -> str:
        return (self._cached["system"].get("public_url") or "").rstrip("/")

    @property
    def mqtt(self) -> dict[str, Any]:
        return self._cached["mqtt"]

    @property
    def time_sync(self) -> dict[str, Any]:
        return self._cached["time_sync"]

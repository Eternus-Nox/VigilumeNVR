"""Camera reachability history + optional down-alerts.

Fed by the CameraProber's per-poll results (main.py). Deliberately SEPARATE
from the prober's live ``_online`` state, which drives the instant UI badge and
the on-connect hooks (time-sync, speaker probe) — those must stay snappy. This
layer adds the two things a raw poll cannot give safely:

  * DEBOUNCE. A single 4 s TCP timeout to :554 must NOT flip a camera to
    "down". A transition to OFFLINE requires ``_FAILS_TO_DOWN`` consecutive
    failed polls (~90 s at the 45 s poll interval); recovery to ONLINE is
    immediate (one good poll).

  * BOOT SUPPRESSION. The state is in-memory, so a restart (including the
    nightly auto-restart) would otherwise re-emit a "down then up" for every
    camera and fire an 11-camera alert storm at 04:00. On start we LOAD the
    last open state from the DB (so no duplicate history rows), and we withhold
    ALERTS for the first ``_BOOT_SUPPRESS_CYCLES`` polls while the picture
    settles. History is still recorded from cycle one.

History is stored as TRANSITION intervals (db.camera_health), not per-poll rows.

NOTE ON SCOPE: this tracks RTSP-port reachability, which is what the prober
measures — not literally "was footage written". It is the right signal for
"is the camera up", and it is honest about that in the UI copy.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger(__name__)

# Consecutive failed polls before a camera is declared DOWN. At the 45 s poll
# interval this is ~90 s — long enough to ride out a single blip, short enough
# to matter. Recovery is immediate (no up-debounce).
_FAILS_TO_DOWN = 2

# Alerts (not history) are withheld for this many polls after boot, so a
# restart does not fire a transition storm for every camera at once.
_BOOT_SUPPRESS_CYCLES = 2


class CameraHealthTracker:
    """Observes poll results, records up/down intervals, fires down-alerts."""

    def __init__(
        self,
        db: Any,
        push: Any,
        *,
        # () -> bool: whether down-alerts are enabled (read live from settings).
        alerts_enabled: Callable[[], bool],
        clock: Callable[[], float],
    ) -> None:
        self._db = db
        self._push = push
        self._alerts_enabled = alerts_enabled
        self._clock = clock
        # Debounced health per camera (True up / False down). Absent = unknown.
        self._state: dict[str, bool] = {}
        self._fails: dict[str, int] = {}
        self._cycles = 0

    async def start(self) -> None:
        """Resume from the DB so a restart does not duplicate intervals."""
        try:
            self._state = await self._db.camera_health_open_states()
            log.info("camera-health: resumed %d camera state(s) from history",
                     len(self._state))
        except Exception:  # noqa: BLE001 — health must never block startup
            log.exception("camera-health: could not load prior state")

    async def observe(self, results: list[tuple[str, bool]]) -> None:
        """One poll's (name, reachable) results. Never raises."""
        self._cycles += 1
        suppress_alerts = self._cycles <= _BOOT_SUPPRESS_CYCLES
        now = self._clock()
        for name, reachable in results:
            try:
                await self._observe_one(name, reachable, now, suppress_alerts)
            except Exception:  # noqa: BLE001 — one camera never stalls the rest
                log.exception("camera-health: observe failed for %s", name)

    async def _observe_one(
        self, name: str, reachable: bool, now: float, suppress_alerts: bool
    ) -> None:
        prev = self._state.get(name)  # None = unknown (first sighting)

        if reachable:
            self._fails[name] = 0
            new_state = True
        else:
            self._fails[name] = self._fails.get(name, 0) + 1
            # Stay UP until enough consecutive failures accrue. If we have never
            # seen this camera up, a first-poll failure still needs the debounce
            # before we call it down (avoids "mass outage" on a cold boot).
            if self._fails[name] < _FAILS_TO_DOWN:
                # Not enough evidence yet. If we already knew a state, keep it;
                # if unknown, stay unknown (record nothing).
                if prev is None:
                    return
                new_state = prev
            else:
                new_state = False

        if new_state == prev:
            return  # no transition

        # A real transition (including the first-ever observation): record it.
        self._state[name] = new_state
        await self._db.record_camera_health(name, new_state, now)
        log.info("camera-health: %s -> %s", name, "up" if new_state else "down")

        # Alert only on a DOWN transition, only once past boot suppression, only
        # when the user enabled it, and never on the very first sighting (prev is
        # None means we have no baseline to have "gone down" from).
        if new_state is False and prev is True and not suppress_alerts:
            if self._alerts_enabled():
                await self._alert_down(name)

    async def _alert_down(self, name: str) -> None:
        """Fire a system push — the UNCOUPLED primitive, so it is not filtered
        by the detection notification gates (labels/min_score/cooldown)."""
        payload = {
            "type": "camera_down",
            "title": "Camera offline",
            "body": f"{name} stopped responding.",
            "camera": name,
        }
        try:
            await self._push.send_to_all(payload)
        except Exception:  # noqa: BLE001 — an alert failure must never propagate
            log.exception("camera-health: down-alert push failed for %s", name)

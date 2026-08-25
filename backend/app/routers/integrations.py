"""Integration test/diagnostic routes (docs/CONTRACTS.md, docs/home-assistant.md).

POST /api/integrations/mqtt/test — admin-only. Attempts a connect + publish
against the CURRENT saved MQTT settings, or against settings supplied in the
body (so the operator can test before saving). Returns {ok, detail}. Never
mutates state; the live publisher is untouched.

GET  /api/integrations/archive/status — what the nightly cloud archive has
actually done: last run, days uploaded, days expired, errors.
POST /api/integrations/archive/run    — run a pass NOW rather than waiting for
the configured hour. Same code path as the scheduled run, so a green result
here is real evidence the remote works, not a separate "test" that proves
something adjacent.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ..auth import require_admin
from ..integrations.mqtt_ha import test_connection
from .settings import MqttSettings

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/integrations",
    tags=["integrations"],
    dependencies=[Depends(require_admin)],
)


class MqttTestRequest(BaseModel):
    """Optional MQTT settings to test. When omitted, the currently SAVED
    settings.mqtt are used. Reuses the same validation as PUT /api/settings."""

    mqtt: Optional[MqttSettings] = None


@router.post("/mqtt/test")
async def mqtt_test(request: Request, body: MqttTestRequest = MqttTestRequest()) -> dict[str, Any]:
    if body.mqtt is not None:
        mqtt_cfg = body.mqtt.model_dump()
    else:
        mqtt_cfg = request.app.state.settings.mqtt
    result = await test_connection(mqtt_cfg)
    log.info("MQTT test connection: ok=%s (%s)", result.get("ok"), result.get("detail"))
    return result


def _archive(request: Request) -> Any:
    return getattr(request.app.state, "archive", None)


@router.get("/archive/status")
async def archive_status(request: Request) -> dict[str, Any]:
    """Configured state + the outcome of the last pass.

    `last_result` is empty until a pass has actually run in THIS process — it is
    in-memory, not persisted, so a restart clears it. `last_uploaded_day` is the
    durable watermark and is what says whether the archive is actually current.
    """
    runner = _archive(request)
    if runner is None:  # archiving not wired (shouldn't happen; be honest anyway)
        return {"available": False, "enabled": False, "last_result": {}, "last_uploaded_day": None}
    last_day = await runner._last_uploaded()
    return {
        "available": True,
        "enabled": runner.enabled(),
        "last_result": runner.last_result,
        "last_uploaded_day": last_day.isoformat() if last_day else None,
    }


@router.post("/archive/run")
async def archive_run(request: Request) -> dict[str, Any]:
    """Run a pass now. Returns the same result shape the scheduler logs.

    Deliberately SYNCHRONOUS (the caller waits) rather than fire-and-forget: the
    whole point of the button is to find out whether the remote is reachable and
    the credentials work, and an immediate 200 saying "started" answers none of
    that. A day of clips can take a while on a thin uplink, so the UI should
    expect this to be slow rather than assume it hung.
    """
    runner = _archive(request)
    if runner is None:
        return {"ok": False, "detail": "Cloud archive is not available on this server."}
    if not runner.enabled():
        return {"ok": False, "detail": "Cloud archive is off, or no remote is set."}
    result = await runner.run_once()
    return {"ok": not result.get("errors"), "detail": "", "result": result}

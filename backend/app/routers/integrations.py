"""Integration test/diagnostic routes (docs/CONTRACTS.md, docs/home-assistant.md).

POST /api/integrations/mqtt/test — admin-only. Attempts a connect + publish
against the CURRENT saved MQTT settings, or against settings supplied in the
body (so the operator can test before saving). Returns {ok, detail}. Never
mutates state; the live publisher is untouched.
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

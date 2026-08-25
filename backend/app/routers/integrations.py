"""Integration test/diagnostic routes (docs/CONTRACTS.md, docs/home-assistant.md).

POST /api/integrations/mqtt/test — admin-only. Attempts a connect + publish
against the CURRENT saved MQTT settings, or against settings supplied in the
body (so the operator can test before saving). Returns {ok, detail}. Never
mutates state; the live publisher is untouched.

GET  /api/integrations/rclone/providers — the storage backends the UI offers,
with the fields each one needs, so the form is server-driven rather than
duplicated in two clients.
GET/POST/DELETE /api/integrations/rclone/remotes — manage cloud destinations
without an SSH session. Secrets are never returned.
POST /api/integrations/rclone/remotes/{name}/test — does it actually work.

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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..auth import require_admin
from ..integrations.mqtt_ha import test_connection
from ..native import rclone_config
from ..native.rclone_config import ConfigError
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


# ---------------------------------------------------------------- rclone remotes


class RemoteCreate(BaseModel):
    """A remote to create. `values` is validated against the provider's own
    field whitelist before any of it reaches an argv — see rclone_config."""

    name: str
    type: str
    values: dict[str, Any] = {}


_RCLONE_TIMEOUT_S = 60.0
# The reachability probe talks to a cloud service, so it gets longer than a
# local config write but must still fail rather than hold a request open.
_RCLONE_TEST_TIMEOUT_S = 90.0


async def _rclone(args: list[str], *, timeout: float = _RCLONE_TIMEOUT_S) -> tuple[int, str, str]:
    """Run rclone. Returns (code, stdout, stderr-tail). Never raises."""
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        return (127, "", "rclone is not installed in this image — rebuild the backend.")
    except OSError as exc:
        return (1, "", f"could not start rclone: {exc}")
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return (1, "", f"rclone timed out after {timeout:.0f}s")
    stderr = (err or b"").decode("utf-8", "replace").strip()
    return (
        proc.returncode or 0,
        (out or b"").decode("utf-8", "replace"),
        " | ".join(stderr.splitlines()[-3:]),
    )


@router.get("/rclone/providers")
async def rclone_providers() -> dict[str, Any]:
    """The catalogue both UIs render their forms from.

    SERVER-DRIVEN on purpose: the field list for seven providers is exactly the
    kind of thing that drifts when it lives in a React file and a Swift file at
    once, and a drifted field name produces a remote that silently does not
    work.
    """
    return {"providers": rclone_config.providers_payload()}


@router.get("/rclone/remotes")
async def rclone_remotes() -> dict[str, Any]:
    code, out, err = await _rclone(rclone_config.build_config_dump_args())
    if code != 0:
        return {"available": False, "remotes": [], "detail": err}
    return {"available": True, "remotes": rclone_config.redact_remotes(out), "detail": ""}


@router.post("/rclone/remotes")
async def rclone_create_remote(body: RemoteCreate) -> dict[str, Any]:
    """Create (or replace) a remote.

    A 400 for a bad definition, not a 500: everything ConfigError raises is a
    message written for the operator, and it is the response body that has to
    tell them which field is wrong.
    """
    try:
        name = rclone_config.validate_name(body.name)
        values = rclone_config.validate_values(body.type, body.values)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    code, _out, err = await _rclone(
        rclone_config.build_config_create_args(name, body.type, values)
    )
    if code != 0:
        log.warning("rclone config create %s (%s) failed: %s", name, body.type, err)
        return {"ok": False, "detail": err or "rclone could not create the remote."}
    log.info("rclone remote %r created (%s)", name, body.type)
    # Prove it works now rather than letting 03:00 be the discovery moment.
    tcode, _tout, terr = await _rclone(
        rclone_config.build_lsd_args(name), timeout=_RCLONE_TEST_TIMEOUT_S
    )
    return {
        "ok": True,
        "reachable": tcode == 0,
        "detail": "" if tcode == 0 else terr or "Saved, but the remote did not answer.",
        "suggested_remote": f"{name}:Vigilume",
    }


@router.delete("/rclone/remotes/{name}")
async def rclone_delete_remote(name: str) -> dict[str, Any]:
    """Forget a remote's credentials. Deletes NOTHING in the cloud."""
    try:
        clean = rclone_config.validate_name(name)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    code, _out, err = await _rclone(rclone_config.build_config_delete_args(clean))
    if code != 0:
        return {"ok": False, "detail": err or "rclone could not remove the remote."}
    log.info("rclone remote %r removed", clean)
    return {"ok": True, "detail": ""}


@router.post("/rclone/remotes/{name}/test")
async def rclone_test_remote(name: str) -> dict[str, Any]:
    """List the remote's top level — the cheapest proof the credentials work."""
    try:
        clean = rclone_config.validate_name(name)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    code, out, err = await _rclone(
        rclone_config.build_lsd_args(clean), timeout=_RCLONE_TEST_TIMEOUT_S
    )
    if code != 0:
        return {"ok": False, "detail": err or "The remote did not answer.", "folders": []}
    folders = [line.split(None, 4)[-1] for line in out.splitlines() if line.strip()]
    return {"ok": True, "detail": "", "folders": folders[:25]}

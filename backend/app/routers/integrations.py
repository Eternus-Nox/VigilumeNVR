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

import html

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..auth import require_admin
from ..integrations.mqtt_ha import test_connection
from ..native import rclone_config, rclone_oauth
from ..native.rclone_config import ConfigError
from ..native.rclone_oauth import OAuthError
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


# ------------------------------------------------------- browser OAuth (cloud)

_PENDING = rclone_oauth.PendingFlows()

# SEPARATE ROUTER, no admin dependency. The provider redirects a bare browser
# back here with no Authorization header, so an admin-gated callback could never
# fire. `state` is what authorizes it — 256 unguessable bits, single use, with a
# short TTL (see rclone_oauth).
oauth_router = APIRouter(prefix="/api/integrations/rclone/oauth", tags=["integrations"])


class OAuthStart(BaseModel):
    name: str
    type: str
    client_id: str
    client_secret: str
    # The browser's own origin, e.g. "http://192.168.1.45:8080". Sent by the UI
    # rather than guessed here: only the browser knows the address it actually
    # reached this server on, and that address must match the one registered
    # with the provider exactly.
    origin: str


@router.get("/rclone/oauth/redirect-uri")
async def rclone_oauth_redirect_uri(origin: str) -> dict[str, Any]:
    """The string to register, plus whether this origin can be used at all.

    `blocked_reason` is the important half. Providers refuse a plain-http
    non-localhost redirect URI at REGISTRATION time, so the UI has to say so
    before the operator creates an app and pastes a URI the console will not
    accept.
    """
    try:
        return {
            "redirect_uri": rclone_oauth.redirect_uri_for(origin),
            "blocked_reason": rclone_oauth.browser_auth_blocked(origin) or "",
        }
    except OAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rclone/oauth/start")
async def rclone_oauth_start(body: OAuthStart) -> dict[str, Any]:
    """Begin a browser authorization; the UI sends the operator to `auth_url`."""
    try:
        name = rclone_config.validate_name(body.name)
        flow = rclone_oauth.start_flow(
            remote_name=name, type_=body.type, client_id=body.client_id,
            client_secret=body.client_secret, origin=body.origin,
        )
    except (ConfigError, OAuthError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _PENDING.add(flow)
    # Log the PKCE fact explicitly. Dropbox refuses a plain-HTTP LAN redirect
    # without a code challenge, and its error names redirect_uri rather than
    # PKCE — so when someone reports "invalid redirect_uri", the first question
    # is whether the running build sends a challenge at all. This answers it
    # from `docker logs` without guessing at which image is deployed.
    log.info(
        "rclone oauth: started %s flow for remote %r, redirect_uri=%s, pkce=S256",
        body.type, name, flow.redirect_uri,
    )
    return {
        "auth_url": rclone_oauth.build_auth_url(flow),
        "redirect_uri": flow.redirect_uri,
    }


def _callback_page(title: str, message: str, ok: bool) -> HTMLResponse:
    """The page the provider's redirect lands on.

    Self-contained and styled inline: this is served by the API, not the web
    app, so it cannot rely on the frontend's stylesheet — and it is the last
    thing the operator sees, so "it worked, go back" has to be unmistakable.
    """
    colour = "#1f9d55" if ok else "#c53030"
    return HTMLResponse(
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title></head>
<body style="font-family:system-ui,-apple-system,sans-serif;background:#14161a;
color:#e6e8eb;display:flex;align-items:center;justify-content:center;
min-height:100vh;margin:0;padding:24px">
<div style="max-width:34rem;text-align:center">
<h1 style="color:{colour};font-size:1.4rem;margin:0 0 .6rem">{html.escape(title)}</h1>
<p style="line-height:1.5;color:#aab2bd">{html.escape(message)}</p>
<p style="margin-top:1.4rem;color:#7d8590;font-size:.9rem">
You can close this tab and return to Vigilume.</p>
</div></body></html>""",
        status_code=200 if ok else 400,
    )


@oauth_router.get("/callback")
async def rclone_oauth_callback(
    request: Request,
    state: str = "",
    code: str = "",
    error: str = "",
    error_description: str = "",
) -> HTMLResponse:
    """Finish the handshake: code -> token -> a working rclone remote.

    Returns HTML, not JSON: a human's browser lands here.
    """
    flow = _PENDING.take(state) if state else None
    if flow is None:
        # An unknown state and an expired one are reported IDENTICALLY — the
        # response must not confirm whether a guessed state ever existed.
        return _callback_page(
            "Sign-in expired",
            "This authorization is no longer valid. Go back to Vigilume and "
            "press Connect again.",
            ok=False,
        )
    if error:
        return _callback_page(
            "Sign-in cancelled",
            error_description or error,
            ok=False,
        )
    if not code:
        return _callback_page(
            "Sign-in incomplete",
            "The provider did not return an authorization code.",
            ok=False,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                flow.provider.token_url,
                data=rclone_oauth.token_request_body(flow, code),
                headers={"Accept": "application/json"},
            )
        if resp.status_code != 200:
            log.warning("rclone oauth: token exchange failed %s", resp.status_code)
            return _callback_page(
                "Sign-in failed",
                f"The provider rejected the exchange ({resp.status_code}). "
                "Check that the app key, secret and redirect URI all match.",
                ok=False,
            )
        token_blob = rclone_oauth.to_rclone_token(resp.json())
    except OAuthError as exc:
        return _callback_page("Sign-in failed", str(exc), ok=False)
    except Exception:  # noqa: BLE001 — a browser must never see a stack trace
        log.exception("rclone oauth: token exchange blew up")
        return _callback_page(
            "Sign-in failed", "Could not reach the provider to finish signing in.",
            ok=False,
        )

    code_, _out, err = await _rclone(
        rclone_config.build_config_create_args(
            flow.remote_name, flow.provider.type,
            rclone_oauth.remote_values(flow, token_blob),
        )
    )
    if code_ != 0:
        return _callback_page("Almost there", f"Signed in, but saving failed: {err}", ok=False)

    log.info("rclone oauth: remote %r authorized via browser", flow.remote_name)
    return _callback_page(
        "Connected",
        f"{flow.remote_name} is ready. Back in Vigilume, set the archive remote "
        f"to {flow.remote_name}:Vigilume and save.",
        ok=True,
    )

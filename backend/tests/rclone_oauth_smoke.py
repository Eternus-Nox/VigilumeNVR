#!/usr/bin/env python3
"""Browser OAuth for cloud storage — the handshake done on the NVR itself.

The callback CANNOT be authenticated: a provider redirects a bare browser to it
with no Authorization header. So `state` is the entire access control, and most
of these checks are about that — unguessable, single-use, expiring, and
indistinguishable from unknown when it fails.

The rest cover the two mistakes that produce an archive which works today and
silently dies later: a token stored without a refresh token, and a token stored
without the app credentials rclone needs to renew it.

No network and no rclone binary — the provider is stubbed, and the argv rclone
would receive is asserted.

Offline-runnable.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native import rclone_oauth  # noqa: E402
from app.native.rclone_oauth import (  # noqa: E402
    CALLBACK_PATH,
    MAX_PENDING,
    OAUTH_PROVIDERS,
    OAuthError,
    PendingFlows,
    build_auth_url,
    redirect_uri_for,
    remote_values,
    start_flow,
    to_rclone_token,
    token_request_body,
    validate_origin,
)

_failures: list[str] = []
_checks = 0


def check(cond: bool, label: str) -> None:
    global _checks
    _checks += 1
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def rejects(fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
        return False
    except OAuthError:
        return True


ORIGIN = "http://192.168.1.45:8080"


def a_flow(name: str = "dropbox", type_: str = "dropbox"):
    return start_flow(
        remote_name=name, type_=type_, client_id="APPKEY",
        client_secret="APPSECRET", origin=ORIGIN,
    )


def main() -> int:
    print("cloud storage browser OAuth")

    # --- the redirect URI the operator registers ---------------------------
    check(
        redirect_uri_for(ORIGIN) == f"{ORIGIN}{CALLBACK_PATH}",
        f"the redirect URI is the origin plus one fixed path ({CALLBACK_PATH})",
    )
    check(validate_origin("http://192.168.1.45:8080/") == ORIGIN, "a trailing slash is trimmed")
    check(rejects(validate_origin, ""), "an empty origin is refused")
    check(rejects(validate_origin, "192.168.1.45:8080"), "an origin with no scheme is refused")
    check(rejects(validate_origin, "ftp://box"), "a non-http scheme is refused")
    check(
        rejects(validate_origin, "http://192.168.1.45:8080/settings"),
        "an origin carrying a PATH is refused — it would build a redirect URI "
        "that never matches the registered one, failing opaquely at the provider",
    )

    # --- starting a flow ----------------------------------------------------
    check(rejects(a_flow, "x", "s3"), "a key-based provider has no browser flow")
    check(
        rejects(start_flow, remote_name="d", type_="dropbox", client_id="",
                client_secret="s", origin=ORIGIN),
        "a missing app key is refused before the browser is sent anywhere",
    )
    flow = a_flow()
    check(len(flow.state) >= 32, f"state is long and unguessable ({len(flow.state)} chars)")
    check(a_flow().state != a_flow().state, "and different every time")

    url = build_auth_url(flow)
    check(url.startswith(OAUTH_PROVIDERS["dropbox"].auth_url), "the auth URL is the provider's")
    check(f"state={flow.state}" in url, "state rides along for the callback to present")
    check("client_id=APPKEY" in url, "the operator's own app id is used")
    check("response_type=code" in url, "authorization-code flow")
    check(
        "token_access_type=offline" in url,
        "Dropbox is asked for an OFFLINE token — without this it returns one "
        "that expires in hours and cannot be renewed, and the archive dies "
        "overnight with nobody awake to read the error",
    )
    check("APPSECRET" not in url, "the app SECRET never appears in a URL the browser sees")
    drive_url = build_auth_url(a_flow("g", "drive"))
    check(
        "access_type=offline" in drive_url and "prompt=consent" in drive_url,
        "Google is forced to re-issue a refresh token (it omits one otherwise)",
    )
    check("scope=" in drive_url, "and asked for a Drive scope")

    # --- the exchange body --------------------------------------------------
    body = token_request_body(flow, "THECODE")
    check(body["grant_type"] == "authorization_code", "exchange uses the code grant")
    check(body["code"] == "THECODE" and body["client_secret"] == "APPSECRET",
          "with the code and the app secret")
    check(
        body["redirect_uri"] == flow.redirect_uri,
        "and REPEATS the redirect URI — providers verify it matches the one "
        "used to get the code",
    )

    # --- state is the whole access control ---------------------------------
    flows = PendingFlows()
    f1 = a_flow()
    flows.add(f1)
    check(flows.take(f1.state) is not None, "a valid state resolves once")
    check(
        flows.take(f1.state) is None,
        "and NOT twice — single use, so an intercepted redirect cannot be replayed",
    )
    check(flows.take("made-up") is None, "an invented state resolves to nothing")
    f2 = a_flow()
    flows.add(f2)
    f2.created_at = time.time() - rclone_oauth.STATE_TTL_S - 1
    check(flows.take(f2.state) is None, "an expired state is gone")
    flows2 = PendingFlows()
    for _ in range(MAX_PENDING + 5):
        flows2.add(a_flow())
    check(
        len(flows2) <= MAX_PENDING,
        f"pending flows are bounded ({len(flows2)} <= {MAX_PENDING}) — an "
        "abandoned tab cannot accumulate secrets in memory",
    )
    newest = a_flow()
    flows2.add(newest)
    check(
        flows2.take(newest.state) is not None,
        "and eviction drops the OLDEST, so a fresh flow is never the casualty",
    )

    # --- the token blob rclone stores --------------------------------------
    now = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)
    blob = to_rclone_token(
        {"access_token": "AT", "refresh_token": "RT", "expires_in": 14400,
         "token_type": "bearer"},
        now=now,
    )
    parsed = json.loads(blob)
    check(parsed["access_token"] == "AT" and parsed["refresh_token"] == "RT",
          "both tokens are carried through")
    check(parsed["expiry"].startswith("2026-08-25T16:00"),
          f"expiry is now + expires_in, RFC3339 ({parsed['expiry']})")
    check(
        rejects(to_rclone_token, {"access_token": "AT"}),
        "a response with NO refresh token is REFUSED — storing it would give an "
        "archive that works for hours and then fails every night",
    )
    check(rejects(to_rclone_token, {"refresh_token": "RT"}), "and one with no access token")
    no_expiry = json.loads(to_rclone_token(
        {"access_token": "A", "refresh_token": "R"}, now=now))
    check(
        no_expiry["expiry"].startswith("2026-08-25T13:00"),
        "a missing expires_in assumes a SHORT life — guessing short costs one "
        "refresh, guessing long costs a failed transfer",
    )

    # --- what actually lands in the rclone remote --------------------------
    values = remote_values(flow, blob)
    check(values["token"] == blob, "the token is stored")
    check(
        values["client_id"] == "APPKEY" and values["client_secret"] == "APPSECRET",
        "ALONGSIDE the app credentials — the token came from the operator's own "
        "OAuth app, and rclone needs that app to refresh it. Token alone works "
        "until the first expiry and then stops.",
    )

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (cloud storage browser OAuth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

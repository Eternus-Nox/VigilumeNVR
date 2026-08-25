#!/usr/bin/env python3
"""Managing rclone remotes from the UI: validation, argv, and redaction.

Three things here would be quietly dangerous if wrong, so they get the most
attention:

1. THE FIELD WHITELIST. Submitted keys become argv for a subprocess. A key that
   is not checked is a way to inject rclone flags — `--config` alone is an
   arbitrary-file-write primitive.
2. REDACTION. `rclone config dump` returns OAuth tokens and obscured passwords;
   obscured is reversible, so anything leaking to a browser is a credential
   leak, not a cosmetic one.
3. THE NAME. It is positional in the argv, so a name starting with `-` would be
   read as a flag.

rclone itself is not run — no binary here — but every argv it would be handed is
asserted, the same way this repo tests ffmpeg's.

Offline-runnable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.native.rclone_config import (  # noqa: E402
    PROVIDERS,
    REDACTED,
    ConfigError,
    authorize_command,
    build_config_create_args,
    build_config_delete_args,
    build_lsd_args,
    provider,
    providers_payload,
    redact_remotes,
    validate_name,
    validate_values,
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


def rejects(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
        return False
    except ConfigError:
        return True


TOKEN = json.dumps({"access_token": "abc", "refresh_token": "r", "expiry": "2026-01-01"})


def main() -> int:
    print("rclone remote configuration")

    # --- the catalogue ----------------------------------------------------
    payload = providers_payload()
    check(len(payload) == len(PROVIDERS) >= 5, f"{len(payload)} providers offered")
    check(
        {p["type"] for p in payload} >= {"dropbox", "drive", "s3", "b2"},
        "including Dropbox, Drive and the key-based object stores",
    )
    oauth = {p["type"] for p in payload if p["oauth"]}
    check(
        oauth == {"dropbox", "drive", "onedrive"},
        f"exactly the browser-sign-in providers are flagged oauth ({sorted(oauth)})",
    )
    check(
        all(p["authorize_command"] for p in payload if p["oauth"]),
        "and each carries the command the operator runs on their own desktop",
    )
    check(
        not any(p["authorize_command"] for p in payload if not p["oauth"]),
        "key-based providers offer no authorize step — a form is all they need",
    )
    check(
        authorize_command("dropbox") == 'rclone authorize "dropbox"',
        "the authorize command is quoted so a shell cannot mangle the type",
    )
    check(
        all(f.get("label") and f.get("kind") for p in payload for f in p["fields"]),
        "every field is renderable (has a label and a kind)",
    )

    # --- names ------------------------------------------------------------
    check(validate_name("  dropbox  ") == "dropbox", "a name is trimmed")
    check(validate_name("my-box_2.a") == "my-box_2.a", "letters, digits, . - _ are fine")
    check(rejects(validate_name, ""), "an empty name is refused")
    check(
        rejects(validate_name, "-config"),
        "a name starting with '-' is REFUSED — it is positional in the argv and "
        "would otherwise be read as a flag",
    )
    check(rejects(validate_name, "has space"), "a name with a space is refused")
    check(rejects(validate_name, "semi;colon"), "shell metacharacters are refused")
    check(rejects(validate_name, "x" * 200), "an absurdly long name is refused")

    # --- the whitelist (the security boundary) ----------------------------
    check(
        rejects(validate_values, "dropbox", {"--config": "/etc/passwd"}),
        "a flag-shaped key is REFUSED, not passed through to argv",
    )
    check(
        rejects(validate_values, "b2", {"account": "a", "key": "k", "endpoint": "x"}),
        "a key valid for ANOTHER provider is still refused for this one",
    )
    check(
        rejects(validate_values, "nonsuch", {"a": "b"}),
        "an unknown provider is refused",
    )
    check(
        rejects(validate_values, "s3", {"provider": "Nope", "access_key_id": "a",
                                        "secret_access_key": "s"}),
        "a select outside its options is refused",
    )
    check(
        rejects(validate_values, "b2", {"account": "only-half"}),
        "a missing required field is refused, naming what is missing",
    )
    try:
        validate_values("b2", {"account": "only-half"})
        missing_msg = ""
    except ConfigError as exc:
        missing_msg = str(exc)
    check("Application key" in missing_msg,
          f"and names it in the operator's words, not the config key ({missing_msg!r})")

    # --- tokens -----------------------------------------------------------
    check(
        rejects(validate_values, "dropbox", {"token": "not json"}),
        "a token that is not JSON is refused HERE rather than becoming an "
        "opaque auth failure at 03:00",
    )
    check(
        rejects(validate_values, "dropbox", {"token": '{"hello":"world"}'}),
        "JSON without an access_token is refused too",
    )
    check(
        validate_values("dropbox", {"token": TOKEN})["token"] == TOKEN,
        "a real token blob passes through byte-for-byte",
    )

    # --- defaults and optionals -------------------------------------------
    s3 = validate_values("s3", {"access_key_id": "AK", "secret_access_key": "SK"})
    check(
        s3["provider"] == "AWS",
        "a REQUIRED field with a default is satisfied by that default — the "
        "operator who accepts 'AWS' sends nothing for it, and rejecting that "
        "would make every S3 and WebDAV remote unconfigurable",
    )
    check("region" not in s3, "an empty optional field is omitted, not sent as ''")
    sftp = validate_values("sftp", {"host": "h", "user": "u"})
    check(
        sftp.get("port") == "22" and "pass" not in sftp,
        "SFTP defaults the port and allows a key-based login with no password",
    )

    # --- argv --------------------------------------------------------------
    args = build_config_create_args("dropbox", "dropbox", {"token": TOKEN})
    check(args[:5] == ["rclone", "config", "create", "dropbox", "dropbox"],
          "create names the remote and the type positionally, in that order")
    check("--obscure" in args, "--obscure: form values arrive as plaintext")
    check(
        "--non-interactive" in args,
        "--non-interactive: rclone must never sit waiting on stdin for a prompt "
        "nobody can answer",
    )
    a1 = build_config_create_args("r", "s3", {"access_key_id": "A", "secret_access_key": "S"})
    a2 = build_config_create_args("r", "s3", {"secret_access_key": "S", "access_key_id": "A"})
    check(a1 == a2, "argv is order-stable regardless of dict ordering")
    check(build_config_delete_args("r") == ["rclone", "config", "delete", "r"], "delete argv")
    lsd = build_lsd_args("dropbox")
    check(lsd[2] == "dropbox:", "a bare remote name is given the colon rclone expects")
    check(build_lsd_args("dropbox:Vig")[2] == "dropbox:Vig", "an explicit path is left alone")
    check("--max-depth" in lsd, "the reachability probe does not walk the whole archive")

    # --- redaction (credential leak if wrong) ------------------------------
    dump = json.dumps({
        "dropbox": {"type": "dropbox", "token": '{"access_token":"SECRET"}'},
        "b2": {"type": "b2", "account": "acct", "key": "SECRETKEY"},
        "nas": {"type": "sftp", "host": "h", "user": "u", "pass": "OBSCURED"},
        "weird": {"type": "sia", "api_password": "x"},
    })
    remotes = redact_remotes(dump)
    blob = json.dumps(remotes)
    check(len(remotes) == 4, "every remote is listed")
    check(
        "SECRET" not in blob and "SECRETKEY" not in blob and "OBSCURED" not in blob,
        "NO secret value appears anywhere in the payload — obscured is "
        "reversible, so leaking it is leaking the credential",
    )
    check(
        all(r["details"].get(k) == REDACTED
            for r, k in ((remotes[0], "key"), (remotes[2], "pass"), (remotes[1], "token"))
            if k in r["details"]),
        "secret keys are replaced with a marker that still says 'this is set'",
    )
    by_name = {r["name"]: r for r in remotes}
    check(by_name["b2"]["details"]["account"] == "acct",
          "non-secret values survive, so the list is still identifiable")
    check(by_name["dropbox"]["label"] == "Dropbox", "a known type gets its friendly label")
    check(
        by_name["weird"]["label"] == "sia" and by_name["weird"]["oauth"] is False,
        "a remote of a type this UI does not model is STILL listed — hiding it "
        "would make 'where did my remote go' a mystery",
    )
    check(redact_remotes("not json") == [], "unparseable dump degrades to no remotes")
    check(redact_remotes("") == [], "an empty dump (no remotes yet) is not an error")
    check(redact_remotes('["a"]') == [], "a non-object dump degrades safely")
    check(provider("dropbox") is not None and provider("nope") is None, "provider lookup")

    print()
    if _failures:
        print(f"{len(_failures)} of {_checks} CHECKS FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL {_checks} CHECKS PASSED (rclone remote configuration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

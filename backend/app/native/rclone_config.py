"""Managing rclone remotes from the UI instead of an SSH session.

`rclone config` is an interactive terminal wizard. This module drives rclone's
NON-interactive equivalents (`config create`, `config delete`, `config dump`) so
the web app and the phone can set up cloud storage without anyone opening a
shell on the NVR.

THE ONE THING THAT CANNOT BE DONE FROM A FORM. Key-based providers (S3, B2,
SFTP, WebDAV) need only an id and a secret — a plain form covers them end to
end. OAuth providers (Dropbox, Drive, OneDrive) need a browser round-trip to a
callback URL, and rclone's callback is ``localhost:53682`` ON THE MACHINE
RUNNING RCLONE. That machine is a headless container, so nothing we render can
complete the handshake: the browser that logs in is on a phone or a laptop, and
its localhost is not the NVR's.

rclone's own answer is `rclone authorize "<type>"`, run on any desktop that has
a browser, which prints a token blob to paste back. So OAuth providers get a
copy-the-command / paste-the-token step. That is one command on the operator's
OWN computer rather than a terminal on the server, which is the part actually
worth removing — and it keeps every rclone provider reachable instead of
special-casing one vendor's API.

SECURITY. Everything here is admin-only and every value the operator supplies is
matched against a per-provider FIELD WHITELIST before it can reach an argv.
Without that, a "field name" like ``--config`` would let a settings form rewrite
arbitrary rclone flags, which is a config-file-write primitive. Unknown keys are
refused, not dropped silently, so a typo is visible rather than mysteriously
absent from the resulting remote.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# rclone's own rule for remote names, plus a leading-character restriction so a
# name can never be mistaken for a flag.
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
NAME_MAX = 48


@dataclass(frozen=True)
class Field:
    """One provider setting the UI renders and the operator fills in."""

    key: str
    label: str
    # text   — plain input
    # secret — never echoed back to a client (see redact_remotes)
    # token  — the JSON blob from `rclone authorize`, multi-line in the UI
    # select — one of `options`
    kind: str = "text"
    required: bool = True
    help: str = ""
    placeholder: str = ""
    options: tuple[str, ...] = ()
    default: str = ""


@dataclass(frozen=True)
class Provider:
    type: str          # rclone's own backend name — passed verbatim to rclone
    label: str
    blurb: str
    fields: tuple[Field, ...] = ()
    # True when the only way to authenticate is a browser round-trip.
    oauth: bool = False
    # Where the operator creates the OAuth app whose id/secret lets the NVR act
    # as its own redirect target — see rclone_oauth. Empty for key-based
    # providers, which need no app at all.
    console_url: str = ""

    def field_map(self) -> dict[str, Field]:
        return {f.key: f for f in self.fields}


# Present on every OAuth provider: rclone needs the operator's OWN app
# credentials to refresh the token it was issued, so they are stored with it.
# Optional on the manual-paste path (rclone's built-in app covers that), and
# supplied automatically by the browser flow.
_CLIENT_FIELDS = (
    Field(
        key="client_id", label="App key", required=False,
        help="From your own app on the provider's developer site. Needed only "
             "for browser sign-in.",
    ),
    Field(
        key="client_secret", label="App secret", kind="secret", required=False,
        help="Stored so the token can renew itself.",
    ),
)

_TOKEN_FIELD = Field(
    key="token",
    label="Authorization token",
    kind="token",
    help=(
        "Paste the whole line of JSON that `rclone authorize` printed, "
        "including the braces."
    ),
    placeholder='{"access_token":"...","refresh_token":"...","expiry":"..."}',
)

PROVIDERS: tuple[Provider, ...] = (
    Provider(
        type="dropbox",
        console_url="https://www.dropbox.com/developers/apps",
        label="Dropbox",
        blurb="Personal or Business Dropbox.",
        oauth=True,
        fields=(_TOKEN_FIELD, *_CLIENT_FIELDS),
    ),
    Provider(
        type="drive",
        console_url="https://console.cloud.google.com/apis/credentials",
        label="Google Drive",
        blurb="A Google account's Drive.",
        oauth=True,
        fields=(
            _TOKEN_FIELD,
            *_CLIENT_FIELDS,
            Field(
                key="root_folder_id", label="Root folder ID", required=False,
                help="Optional. Restricts Vigilume to one folder — take it from the "
                     "Drive folder's URL.",
            ),
        ),
    ),
    Provider(
        type="onedrive",
        console_url="https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps",
        label="OneDrive",
        blurb="Microsoft OneDrive or SharePoint.",
        oauth=True,
        fields=(
            _TOKEN_FIELD,
            *_CLIENT_FIELDS,
            Field(key="drive_id", label="Drive ID", required=False,
                  help="Optional; rclone authorize usually reports it."),
            Field(
                key="drive_type", label="Drive type", kind="select", required=False,
                options=("personal", "business", "documentLibrary"), default="personal",
            ),
        ),
    ),
    Provider(
        type="s3",
        label="S3-compatible",
        blurb="AWS S3, MinIO, Wasabi, and anything else speaking the S3 API. "
              "No browser sign-in — an access key is enough.",
        fields=(
            Field(
                key="provider", label="Service", kind="select", default="AWS",
                options=("AWS", "Minio", "Wasabi", "Ceph", "Other"),
            ),
            Field(key="access_key_id", label="Access key ID"),
            Field(key="secret_access_key", label="Secret access key", kind="secret"),
            Field(key="region", label="Region", required=False, placeholder="us-east-1"),
            Field(
                key="endpoint", label="Endpoint", required=False,
                help="Required for MinIO/Wasabi and other non-AWS services; leave "
                     "empty for AWS itself.",
                placeholder="https://s3.example.com",
            ),
        ),
    ),
    Provider(
        type="b2",
        label="Backblaze B2",
        blurb="Cheap object storage. No browser sign-in — an application key is enough.",
        fields=(
            Field(key="account", label="Account ID or application key ID"),
            Field(key="key", label="Application key", kind="secret"),
        ),
    ),
    Provider(
        type="sftp",
        label="SFTP",
        blurb="Any machine you can SSH to — another NAS, a VPS, a friend's box.",
        fields=(
            Field(key="host", label="Host", placeholder="backup.example.com"),
            Field(key="user", label="Username"),
            Field(key="pass", label="Password", kind="secret", required=False,
                  help="Leave empty if the server uses a key you have configured."),
            Field(key="port", label="Port", required=False, default="22"),
        ),
    ),
    Provider(
        type="webdav",
        label="WebDAV",
        blurb="Nextcloud, ownCloud, or any WebDAV server.",
        fields=(
            Field(key="url", label="Server URL",
                  placeholder="https://cloud.example.com/remote.php/dav/files/me"),
            Field(
                key="vendor", label="Vendor", kind="select", default="other",
                options=("nextcloud", "owncloud", "sharepoint", "other"),
            ),
            Field(key="user", label="Username", required=False),
            Field(key="pass", label="Password", kind="secret", required=False),
        ),
    ),
)

_BY_TYPE = {p.type: p for p in PROVIDERS}

# Config keys that hold a secret, for redaction on the way OUT. Kept as an
# explicit set rather than derived only from the field kinds because a remote
# created by `rclone config` on the command line can carry keys this module
# never offers, and those must be redacted too.
_SECRET_KEYS = frozenset({
    "token", "pass", "password", "secret_access_key", "key", "client_secret",
    "auth_token", "sa_credentials", "service_account_credentials",
})

REDACTED = "********"


class ConfigError(ValueError):
    """A rejected remote definition — the message is shown to the operator."""


def provider(type_: str) -> Optional[Provider]:
    return _BY_TYPE.get(type_)


def providers_payload() -> list[dict[str, Any]]:
    """The provider catalogue, as the UIs render it."""
    return [
        {
            "type": p.type,
            "label": p.label,
            "blurb": p.blurb,
            "oauth": p.oauth,
            "authorize_command": authorize_command(p.type) if p.oauth else "",
            "console_url": p.console_url,
            "fields": [
                {
                    "key": f.key, "label": f.label, "kind": f.kind,
                    "required": f.required, "help": f.help,
                    "placeholder": f.placeholder, "options": list(f.options),
                    "default": f.default,
                }
                for f in p.fields
            ],
        }
        for p in PROVIDERS
    ]


def authorize_command(type_: str) -> str:
    """The command an operator runs on their OWN desktop to mint a token."""
    return f'rclone authorize "{type_}"'


# Fragments of rclone/provider error text, mapped to what an operator can
# actually DO about it. rclone's own advice is a terminal command
# (`rclone config reconnect dropbox:`), which is precisely what this UI exists
# to avoid — so the wording here names the equivalent action in the UI.
#
# The raw error is NEVER replaced, only accompanied: an explanation that turns
# out to be wrong must not hide the evidence that would show it was wrong.
_ERROR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("invalid_grant", "refresh token is invalid", "token has expired",
         "couldn't fetch token"),
        "This remote's sign-in is no longer valid — the provider rejected the "
        "stored refresh token. That happens when the account's authorization "
        "was revoked, or when the app it was issued to was deleted or had its "
        "secret regenerated. To fix it: remove this remote and add it again "
        "under the SAME name (the archive settings reference it by name, so "
        "nothing else needs changing). If you paste a token, the app you run "
        "`rclone authorize` with must be the same app whose key and secret this "
        "remote stores — a token minted by one app can never refresh against "
        "another.",
    ),
    (
        ("invalid_client", "incorrect client credentials"),
        "The provider rejected this remote's app key/secret. Check them against "
        "the app in the provider's console — a regenerated secret has to be "
        "updated here too.",
    ),
    (
        ("401", "403", "accessdenied", "access denied", "signaturedoesnotmatch",
         "invalid access key"),
        "The provider refused these credentials. For S3-style remotes that is "
        "usually a wrong key/secret or a bucket the key cannot reach; for OAuth "
        "ones, a sign-in that needs redoing.",
    ),
    (
        ("no such host", "dial tcp", "connection refused", "i/o timeout",
         "network is unreachable", "tls handshake"),
        "The server could not reach the provider. That is a network or DNS "
        "problem on the server rather than a credentials problem — check the "
        "box has internet access and nothing blocks outbound HTTPS.",
    ),
    (
        ("directory not found", "path not found", "404"),
        "The credentials work, but the path does not exist yet. That is normal "
        "before the first upload — the archive creates it on its first run.",
    ),
)


def explain_remote_error(stderr: str) -> str:
    """An operator-facing explanation for an rclone failure, or "".

    Matching is on lowercased substrings because rclone's messages are prose
    that varies by backend and version — the STABLE part is the provider's own
    error token (`invalid_grant`, `invalid_client`), which is what these key on.
    First match wins, so the specific OAuth cases are listed before the generic
    401/403 one.
    """
    text = (stderr or "").lower()
    if not text.strip():
        return ""
    for needles, hint in _ERROR_HINTS:
        if any(n in text for n in needles):
            return hint
    return ""


def validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ConfigError("Give the remote a name.")
    if len(name) > NAME_MAX:
        raise ConfigError(f"Remote names are at most {NAME_MAX} characters.")
    if not _NAME_RE.match(name):
        raise ConfigError(
            "A remote name may use letters, numbers, dot, dash and underscore, "
            "and must start with a letter, number or underscore."
        )
    return name


def validate_values(type_: str, values: dict[str, Any]) -> dict[str, str]:
    """Check a submitted remote against its provider's field whitelist.

    Returns the cleaned key/value pairs. Raises ConfigError with a message meant
    for the operator.

    THE WHITELIST IS THE SECURITY BOUNDARY: these become argv for a subprocess,
    so an unrecognised key is refused rather than passed through. Refused, not
    dropped — a silently ignored field produces a remote that looks configured
    and does not work.
    """
    prov = provider(type_)
    if prov is None:
        raise ConfigError(f"Unknown storage type {type_!r}.")
    allowed = prov.field_map()
    cleaned: dict[str, str] = {}
    for key, raw in (values or {}).items():
        spec = allowed.get(key)
        if spec is None:
            raise ConfigError(f"{prov.label} has no setting called {key!r}.")
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        if spec.options and text not in spec.options:
            raise ConfigError(
                f"{spec.label} must be one of: {', '.join(spec.options)}."
            )
        if spec.kind == "token":
            # Fail here rather than letting rclone write a remote whose token is
            # a stray shell prompt or a truncated paste — that failure would
            # otherwise surface much later, as an opaque auth error at 03:00.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    "That token is not valid JSON. Copy the whole line "
                    "`rclone authorize` printed, including the { and }."
                ) from exc
            if not isinstance(parsed, dict) or "access_token" not in parsed:
                raise ConfigError(
                    "That JSON does not look like an rclone token — it should "
                    "contain an \"access_token\"."
                )
        cleaned[key] = text

    # Defaults BEFORE the required check: a default IS a value, so a required
    # field that has one is never actually missing. The other order rejects
    # every S3 and WebDAV remote — their service/vendor selects are required and
    # defaulted, and an operator who accepts the default sends nothing for them.
    for key, spec in allowed.items():
        if spec.default and not cleaned.get(key):
            cleaned[key] = spec.default

    missing = [
        spec.label for key, spec in allowed.items()
        if spec.required and not cleaned.get(key)
    ]
    if missing:
        raise ConfigError("Missing: " + ", ".join(missing) + ".")
    return cleaned


def build_config_create_args(name: str, type_: str, values: dict[str, str]) -> list[str]:
    """argv for a non-interactive `rclone config create`.

    `--obscure` because the values arrive as plaintext from a form and rclone
    stores password-type fields obscured; it only touches fields rclone itself
    marks as passwords, so a `token` blob passes through untouched.

    `--non-interactive` so rclone can never sit waiting on stdin for a prompt we
    did not anticipate — without it an unexpected question hangs the request
    until its timeout.
    """
    args = ["rclone", "config", "create", name, type_]
    for key in sorted(values):  # sorted: a stable argv is a testable argv
        args += [key, values[key]]
    args += ["--obscure", "--non-interactive"]
    return args


def build_config_delete_args(name: str) -> list[str]:
    return ["rclone", "config", "delete", name]


def build_config_dump_args() -> list[str]:
    return ["rclone", "config", "dump"]


def build_lsd_args(remote: str) -> list[str]:
    """argv listing a remote's top level — the reachability check behind "Test".

    `--max-depth 1` and a short retry budget: this answers "do the credentials
    work", and must fail fast rather than grinding through a deep listing on a
    remote holding a year of archives.
    """
    target = remote if ":" in remote else f"{remote}:"
    return ["rclone", "lsd", target, "--max-depth", "1", "--retries", "1", "--low-level-retries", "2"]


def redact_remotes(dump_json: str) -> list[dict[str, Any]]:
    """`rclone config dump` output -> a safe, UI-shaped list.

    SECRETS NEVER LEAVE THE SERVER. rclone's dump contains obscured passwords
    and OAuth tokens; obscured is reversible, so shipping them to a browser
    would be handing out the credentials. Every secret-ish key is replaced with
    a marker that says only "this is set".
    """
    try:
        parsed = json.loads(dump_json or "{}")
    except json.JSONDecodeError:
        log.warning("rclone config dump was not JSON — reporting no remotes")
        return []
    if not isinstance(parsed, dict):
        return []
    out: list[dict[str, Any]] = []
    for name, cfg in sorted(parsed.items()):
        if not isinstance(cfg, dict):
            continue
        type_ = str(cfg.get("type") or "")
        known = provider(type_)
        details = {
            k: (REDACTED if k in _SECRET_KEYS else str(v))
            for k, v in cfg.items()
            if k != "type"
        }
        out.append({
            "name": name,
            "type": type_,
            # None for a remote made outside this UI with a backend we do not
            # model — still listed, because hiding it would make "why is my
            # remote not here" a mystery.
            "label": known.label if known else type_ or "Unknown",
            "oauth": bool(known.oauth) if known else False,
            "details": details,
        })
    return out

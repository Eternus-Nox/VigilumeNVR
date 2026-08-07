"""docker-compose.portainer.yml invariants.

THE PROBLEM THIS GUARDS. Compose resolves a RELATIVE bind source against its
project directory. Under Portainer that directory lives inside the Portainer
container, but the absolute string compose derives from it is executed by the
HOST daemon — which creates a same-named path at the host root. On Unraid `/`
is a RAM-backed tmpfs, so a single relative bind silently puts nvr.db, the JWT
secret, snapshots and models in RAM: every reboot is a virgin install, and if
the media path lands there too, 24/7 recording fills tmpfs and takes the box
down. The failure is invisible until a reboot, which is the worst possible time
to discover it.

So the Portainer stack file may contain NO relative bind sources, ever. That is
a property a human re-reading a 130-line YAML file will not reliably check, and
it is one careless `./data` away from data loss — hence a test.

Pure stdlib + PyYAML, no cv2, no container. Run it anywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.portainer.yml"

PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    PASS += 1
    print(f"  ok: {msg}")


# ${VAR:?msg} / ${VAR:-default} / ${VAR}
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([?-])([^}]*))?\}")


def interpolate(text: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """Emulate compose interpolation. Returns (result, names_that_aborted)."""
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name, op, arg = m.group(1), m.group(2), m.group(3)
        val = env.get(name, "")
        if val:
            return val
        if op == "?":
            missing.append(name)
            return ""
        if op == "-":
            return arg
        return ""

    return VAR_RE.sub(sub, text), missing


def bind_sources(service: dict) -> list[str]:
    out = []
    for v in service.get("volumes", []) or []:
        spec = v if isinstance(v, str) else v.get("source", "")
        if isinstance(v, str):
            spec = v.split(":")[0] if not v.startswith("${") else v.rsplit(":/", 1)[0]
        out.append(spec)
    return out


def main() -> None:
    raw = COMPOSE.read_text()

    # ── 1. The required-variable guards actually abort ────────────────────
    _, missing = interpolate(raw, {})
    for var in ("ADMIN_PASSWORD", "VIGILUME_APPDATA", "MEDIA_PATH"):
        check(var in missing,
              f"{var} is `:?`-guarded — an unset value aborts the deploy, "
              f"it does not start with a bad default")

    # ── 2. Fully interpolated, EVERY bind source is absolute ──────────────
    env = {
        "ADMIN_PASSWORD": "pw",
        "VIGILUME_APPDATA": "/mnt/user/appdata/vigilume",
        "MEDIA_PATH": "/mnt/user/vigilume/media",
    }
    resolved, missing = interpolate(raw, env)
    check(not missing, "with the three required vars set, nothing else aborts")

    doc = yaml.safe_load(resolved)
    services = doc["services"]

    for name, svc in services.items():
        for src in bind_sources(svc):
            if not src or src.startswith("/") is False and ":" not in src:
                pass
            host_side = src.split(":")[0]
            check(host_side.startswith("/"),
                  f"{name}: bind source {host_side!r} is an absolute host path "
                  f"(a relative one would resolve into Portainer's container "
                  f"and land in RAM on the host)")
            check(not host_side.startswith("./") and ".." not in host_side,
                  f"{name}: bind source {host_side!r} has no ./ or ../ component")

    # ── 3. The backend↔go2rtc hand-off still points at ONE directory ──────
    def host_path_for(service: str, container_path: str) -> str:
        for v in services[service]["volumes"]:
            if v.endswith(":" + container_path):
                return v[: -(len(container_path) + 1)]
        raise AssertionError(f"{service} has no mount at {container_path}")

    writer = host_path_for("backend", "/go2rtc-config")
    reader = host_path_for("go2rtc", "/config")
    check(writer == reader,
          f"backend writes go2rtc.yaml to the same host dir go2rtc reads "
          f"({writer}) — live view is dead if these ever diverge")

    # ── 4. Portainer-specific structural rules ────────────────────────────
    check("name" not in doc,
          "no top-level `name:` — Portainer derives the project name from the "
          "stack name; hardcoding it lets this stack adopt the CLI stack's containers")

    for name, svc in services.items():
        check("container_name" not in svc,
              f"{name}: no container_name — it would collide with a "
              f"`docker compose` run of the main file")
        check("build" not in svc,
              f"{name}: no build context — a pasted Portainer stack has no source tree")

    # ── 5. The rule the main compose file documents in prose ──────────────
    # `:?` inside a profile-gated service aborts the stack for everyone who
    # does NOT enable that profile, because compose interpolates the whole file
    # before it filters profiles.
    for name, svc in services.items():
        if svc.get("profiles"):
            body = yaml.safe_dump(svc)
            check(":?" not in body,
                  f"{name} is profile-gated and uses no `:?` guard")
    check(True, "no profile-gated service uses a `:?` guard (none are gated here)")

    # ── 6. WebRTC needs both transports ───────────────────────────────────
    ports = [str(p) for p in services["go2rtc"]["ports"]]
    check(any(p.endswith("8555/tcp") for p in ports), "8555/tcp is published")
    check(any(p.endswith("8555/udp") for p in ports),
          "8555/udp is published — WebRTC media fails without it")

    # ── 7. The detector override stays empty by default ───────────────────
    backend_env = "\n".join(services["backend"]["environment"])
    check(re.search(r"VIGILUME_DETECTOR=\s*$", backend_env, re.M) is not None,
          "VIGILUME_DETECTOR defaults to EMPTY — a non-empty value silently "
          "overrides the stored Detection-hardware setting forever")

    print(f"\nAll {PASS} Portainer compose checks passed.")


if __name__ == "__main__":
    main()

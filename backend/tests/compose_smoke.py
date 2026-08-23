"""docker-compose.yml invariants.

Guards the properties of the deployment stack that a human re-reading a
300-line YAML file will not reliably check, and that fail in ways which are
either silent or only visible long after the mistake.

HISTORY. This suite began as portainer_compose_smoke.py, guarding a second
stack file (docker-compose.portainer.yml) whose defining hazard was RELATIVE
bind sources: Portainer resolves them against a directory inside its own
container, but the absolute string it derives is executed by the HOST daemon,
which on Unraid lands on a RAM-backed tmpfs root — a virgin install after every
reboot. That stack file is gone, and with it that hazard: `docker compose` from
a terminal resolves relative binds against the project directory on disk, which
is exactly what `./data` is meant to mean. The checks that outlived it are
below; the absolute-path rule was Portainer-specific and is deliberately NOT
carried over.

Pure stdlib + PyYAML, no cv2, no container. Run it anywhere.

    python backend/tests/compose_smoke.py
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"

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

# The NVIDIA reservation, as it appears COMMENTED OUT in the backend service.
# Matching the exact indented forms is the point: this block's only interface is
# a human deleting the "# " prefixes, so its indentation has to be right.
NVIDIA_BLOCK_RE = re.compile(
    r"^(\s*)# (deploy:"
    r"|  resources:"
    r"|    reservations:"
    r"|      devices:"
    r"|        - driver: nvidia"
    r"|          count: 1"
    r"|          capabilities: \[gpu\])\s*$"
)

NVIDIA_RESERVATION = {
    "resources": {"reservations": {"devices": [
        {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]},
    ]}},
}


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


def without_comments(text: str) -> str:
    """``text`` with YAML comments removed, via a load/dump round trip.

    Needed because this file EXPLAINS its own interpolation rules in prose —
    "Deliberately ${VAR:-default}, never ${VAR:?}" — and a naive regex over the
    raw text reads those examples as real variables. Real compose interpolates
    parsed values, where comments no longer exist; matching that behavior is the
    difference between testing the stack and testing its documentation.
    """
    return yaml.safe_dump(yaml.safe_load(text))


def bind_sources(service: dict) -> list[str]:
    out = []
    for v in service.get("volumes", []) or []:
        spec = v if isinstance(v, str) else v.get("source", "")
        if isinstance(v, str):
            spec = v.split(":")[0] if not v.startswith("${") else v.rsplit(":/", 1)[0]
        out.append(spec)
    return out


def uncomment_nvidia(text: str) -> str:
    """The file as it would be after a user uncomments the NVIDIA block."""
    return "\n".join(
        (m.group(1) + m.group(2)) if (m := NVIDIA_BLOCK_RE.match(line)) else line
        for line in text.split("\n")
    )


def main() -> None:
    raw = COMPOSE.read_text()
    body = without_comments(raw)

    # ── 1. ADMIN_PASSWORD is the ONE thing an operator must supply ────────
    _, missing = interpolate(body, {})
    check(missing == ["ADMIN_PASSWORD"],
          f"ADMIN_PASSWORD is the only `:?`-guarded variable — everything else "
          f"has a working default (aborting on: {missing})")

    for label, env in (
        ("defaults only", {"ADMIN_PASSWORD": "pw"}),
        ("explicit overrides", {
            "ADMIN_PASSWORD": "pw",
            "MEDIA_PATH": "/mnt/disks/big/media",
            "CORAL_DEVICE": "/dev/apex_0",
            "VAAPI_DEVICE": "/dev/dri/renderD128",
        }),
    ):
        resolved, missing = interpolate(body, env)
        check(not missing, f"{label}: interpolates with nothing left unresolved")
        for name, svc in yaml.safe_load(resolved)["services"].items():
            for src in bind_sources(svc):
                host_side = src.split(":")[0]
                check(".." not in host_side,
                      f"{label}: {name} bind source {host_side!r} has no ../ escape")

    resolved, _ = interpolate(body, {"ADMIN_PASSWORD": "pw"})
    doc = yaml.safe_load(resolved)
    services = doc["services"]

    # ── 2. The backend↔go2rtc hand-off still points at ONE directory ──────
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

    # ── 3. A profile-gated service must never use a `:?` guard ────────────
    # compose interpolates the WHOLE file before it filters profiles, so a `:?`
    # inside an opt-in service aborts the stack for everyone who did NOT enable
    # that profile. The file says this in prose twice; here it is enforced.
    gated = [n for n, s in services.items() if s.get("profiles")]
    check(gated, "there ARE profile-gated services (else this check is vacuous)")
    for name in gated:
        svc_body = yaml.safe_dump(services[name])
        check(":?" not in svc_body,
              f"{name} is profile-gated ({services[name]['profiles']}) and uses "
              f"no `:?` guard — one would abort the stack for everyone else")

    # ── 4. WebRTC needs both transports ───────────────────────────────────
    ports = [str(p) for p in services["go2rtc"]["ports"]]
    check(any(p.endswith("8555/tcp") for p in ports), "8555/tcp is published")
    check(any(p.endswith("8555/udp") for p in ports),
          "8555/udp is published — WebRTC media fails without it")

    # ── 5. The detector override stays empty by default ───────────────────
    backend_env = "\n".join(services["backend"]["environment"])
    check(re.search(r"VIGILUME_DETECTOR=\s*$", backend_env, re.M) is not None,
          "VIGILUME_DETECTOR defaults to EMPTY — a non-empty value silently "
          "overrides the stored Detection-hardware setting forever")

    # ── 6. Optional hardware passthrough is inert until named ─────────────
    # Both device mappings must resolve to a harmless node with NOTHING set, or
    # the backend refuses to start on a box with no Coral and no iGPU. They also
    # must not collide: compose rejects two devices sharing a container path,
    # which is the only reason the VAAPI default is /dev/zero and not /dev/null.
    devices = services["backend"].get("devices", [])
    container_paths = [d.split(":")[1] for d in devices]
    check(len(container_paths) == len(set(container_paths)),
          f"optional device mappings have distinct container paths {container_paths} "
          f"— duplicates are a compose error, not a warning")
    for d in devices:
        host_side = d.split(":")[0]
        check(host_side in ("/dev/null", "/dev/zero"),
              f"unset device mapping defaults to an inert node ({host_side}) — a "
              f"real path would stop the backend on boxes without that hardware")

    named, _ = interpolate(body, {
        "ADMIN_PASSWORD": "pw",
        "CORAL_DEVICE": "/dev/apex_0",
        "VAAPI_DEVICE": "/dev/dri/renderD128",
    })
    named_devices = yaml.safe_load(named)["services"]["backend"]["devices"]
    check("/dev/apex_0:/dev/apex_0" in named_devices
          and "/dev/dri/renderD128:/dev/dri/renderD128" in named_devices,
          "naming CORAL_DEVICE / VAAPI_DEVICE maps them through at the same path "
          "inside the container (app code needs no separate config)")

    # ── 7. The NVIDIA reservation ships COMMENTED OUT, and uncomments clean ──
    # An active `deploy.reservations.devices[driver: nvidia]` makes the backend
    # refuse to start on a box with no NVIDIA container runtime ("could not
    # select device driver"). AMD, Intel and CPU-only boxes are the common case,
    # so the block is opt-in — and because opting in means UNCOMMENTING, the
    # commented form has to stay correctly indented or the people who need it
    # get a YAML error instead of a GPU.
    check("deploy" not in services["backend"],
          "no active NVIDIA reservation — the stack starts on an AMD/Intel/CPU box")

    on, _ = interpolate(without_comments(uncomment_nvidia(raw)), {"ADMIN_PASSWORD": "pw"})
    check(yaml.safe_load(on)["services"]["backend"].get("deploy") == NVIDIA_RESERVATION,
          "uncommenting the block yields exactly the NVIDIA reservation "
          "(indentation of the commented form is correct)")

    print(f"\nAll {PASS} compose checks passed.")


if __name__ == "__main__":
    main()

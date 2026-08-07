# Running Vigilume NVR under Portainer

The main [docker-compose.yml](../docker-compose.yml) is written for
`docker compose` run from the project directory on the NVR host. Portainer runs
compose from *its own* directory, inside *its own* container, and two things
break when you paste the main file into it. Use
[docker-compose.portainer.yml](../docker-compose.portainer.yml) instead — it is
the same stack with those two problems designed out.

**Read [the two traps](#the-two-traps) before you deploy.** The first one loses
your database silently, which is a bad way to find out about it.

---

## Pick an option

| | How it works | Best for | Build needed |
|---|---|---|---|
| **A — Pasted stack, prebuilt images** | Portainer pulls `ghcr.io/…/vigilume-*` | Almost everyone | No |
| **B — Pasted stack, your own images** | You build once on the host CLI, Portainer pulls locally | No GHCR access, or you changed the code | Once, on the CLI |
| **C — Repository stack** | Portainer clones this repo and builds | You want Portainer to track a git branch | Every deploy |

**Option A unless you have a reason.** Option C is the one that looks most
convenient and behaves worst — see [why C is last](#option-c--repository-stack).

---

## Before you start

**1. GPU.** Detection runs CUDA. On Unraid install the **Nvidia Driver** plugin
(by *ich777*) from Community Applications, then reboot. Confirm:

```bash
nvidia-smi -L
```

Copy the `GPU-xxxxxxxx-…` UUID — it is a better value for
`NVIDIA_VISIBLE_DEVICES` than `all`, because it leaves your other GPU
containers pinned to their own cards.

If the NVIDIA runtime is missing the backend does not start *at all* — the
`deploy.resources.reservations.devices` block fails at container-create time,
before any application code runs. `VIGILUME_REQUIRE_GPU=0` does **not** rescue
this; that flag only permits CPU inference once the container is already up.

**2. Docker storage.** Option A pulls ~5 GB. Options B and C *build*, which
peaks at **10–15 GB** free — pip holds the downloaded CUDA wheels and their
unpacked form at the same time. Unraid's default 20 GB `docker.img` is the
usual cause of a `No space left on device` failure partway through `pip
install`. Fix it in **Settings → Docker** (stop Docker first): switch to
**directory mode** on your cache/NVMe pool, or raise the vDisk to **≥ 64 GB**.

**3. Host directories.** Create these before deploying, so Docker does not
auto-create them as an Unraid share with the wrong storage settings:

```bash
mkdir -p /mnt/user/appdata/vigilume/data /mnt/user/appdata/vigilume/go2rtc/config
mkdir -p /mnt/user/vigilume/media
```

Appdata belongs on the cache pool; **media belongs on the array**. Budget
roughly **45 GB per camera per day** for 24/7 recording.

**4. Free ports.** The stack publishes 8080, 8554, 8555 (tcp *and* udp). A
single collision fails the whole deploy. `8080` in particular is often already
taken on Unraid — the stack file exposes `WEB_PORT` so you can move it.

**5. Outbound internet, once.** On first boot the backend downloads its D-FINE
ONNX models from huggingface.co. After that it runs fully offline.

---

## Option A — pasted stack, prebuilt images

1. **Portainer → Stacks → Add stack**, name it `vigilume`.
2. **Web editor** → paste the contents of
   [docker-compose.portainer.yml](../docker-compose.portainer.yml).
3. Under **Environment variables**, add at minimum:

   | Name | Example | |
   |---|---|---|
   | `ADMIN_PASSWORD` | *(your password)* | **required** |
   | `VIGILUME_APPDATA` | `/mnt/user/appdata/vigilume` | **required** |
   | `MEDIA_PATH` | `/mnt/user/vigilume/media` | **required** |
   | `VIGILUME_WEBRTC_HOST` | `192.168.1.253` | strongly recommended |
   | `TZ` | `America/New_York` | |
   | `NVIDIA_VISIBLE_DEVICES` | `GPU-xxxxxxxx-…` | |

   The three required ones use compose's `:?` guard: leave one out and the
   deploy stops with a message naming it, rather than starting with a bad
   value. See [the ADMIN_PASSWORD trap](#never-remove-the-admin_password-guard).

4. **Deploy the stack.**

A cold deploy takes **up to ~2.5 minutes** to go fully green. That is the
health gate, not a hang: go2rtc waits for the backend to report healthy,
because the backend is what *generates* go2rtc's config file.

Then open `http://<nvr-host>:8080` and log in with `ADMIN_PASSWORD`. Add
cameras under **Settings → Cameras**.

> The stack file deliberately omits the `CAM1_*`/`CAM2_*`/`CAM3_*` seed
> variables that `.env.example` documents. They only ever apply on the very
> first boot, and half-filled values create phantom cameras you then have to
> delete. Add cameras in the UI.

## Option B — pasted stack, your own images

Build once on the host, then point the stack at the local tags. Useful if you
have modified the code, or cannot pull from GHCR.

```bash
cd /path/to/VigilumeNVR && docker compose build backend web
```

Then deploy exactly as in Option A, adding two more environment variables:

| Name | Value |
|---|---|
| `VIGILUME_BACKEND_IMAGE` | `vigilume-nvr-backend` |
| `VIGILUME_WEB_IMAGE` | `vigilume-nvr-web` |

(Compose names locally built images `<project>-<service>`; confirm yours with
`docker images | grep vigilume`.)

Rebuild on the CLI when you update, then **Pull and redeploy** in Portainer.

## Option C — Repository stack

Portainer clones this repo into its own container and builds there. It works,
but you inherit every build cost on every deploy plus two extra risks:

- **Portainer's builder may not enable BuildKit.**
  [backend/Dockerfile:170](../backend/Dockerfile) uses a heredoc
  (`RUN python - <<'PY'`) for its build-time CUDA/cuDNN sanity check. The
  classic builder cannot parse it and fails with a misleading
  `unknown instruction` error.
- **Build-time network dependencies.** The backend fetches `libedgetpu1-std`
  from a third-party GitHub release, and the build hard-fails if it does not
  land. A prebuilt image is immune to that source disappearing.

If you use it anyway: set **Compose path** to `docker-compose.portainer.yml`,
replace each `image:` with the matching `build:` block from the main compose
file, and keep every bind path absolute.

---

## The two traps

### Relative bind paths do not mean what they look like

`./data`, `./go2rtc/config`, `./secrets` and `./caddy` in the main compose file
are resolved by compose against its **project directory**. Under Portainer that
directory is inside the *Portainer container* (`/data/compose/<stackID>/`).
Compose turns the relative path into an absolute string there — and then hands
that string to the **host** daemon, which knows nothing about Portainer's
filesystem and creates a same-named directory at the host root.

On Unraid, `/` is a RAM-backed tmpfs; only `/boot` and `/mnt/*` persist. So the
database, `secrets.json`, snapshots and downloaded models land in RAM: every
reboot gives you an empty install with no cameras, no users, no event history,
and regenerated session secrets. If `MEDIA_PATH` is also left unset, 24/7
recording writes there too, and filling tmpfs takes the whole box down.

`docker-compose.portainer.yml` has no relative paths for this reason, and marks
the two directory variables required so a typo fails loudly.

### Never remove the `ADMIN_PASSWORD` guard

`ADMIN_PASSWORD=${ADMIN_PASSWORD:?…}` aborts the deploy when the variable is
missing. Portainer shows this as a one-line red banner that does not explain
itself, and the tempting fix is to change `:?` to `:-` or drop the `${…}`.

**Do not.** The backend reads the value with `.strip()` and does not fall back
to a default when it is present-but-empty. Both of those edits produce a stack
that deploys green with a **blank admin password**, on a service reachable from
your whole LAN. Add the variable instead.

---

## Optional services

The main compose file gates the push relay and the HTTPS terminator behind
compose profiles (`relay`, `tls`). Portainer's stack UI has no profile
selector — you can set `COMPOSE_PROFILES` as a stack environment variable, but
the simpler and more predictable approach in Portainer is to append the service
block you want with the `profiles:` key removed, so it simply always runs.

Two things to know if you add **caddy** that way:

- It needs a real `Caddyfile`. Bind the **directory**
  (`/mnt/user/appdata/vigilume/caddy:/etc/caddy:ro`) and put the repo's
  `caddy/Caddyfile` in it. A single-file bind mount fails on Unraid's FUSE
  layer, and an empty auto-created directory gives you a Caddy that starts and
  serves nothing.
- Its `caddy-data` volume holds the self-signed root CA. Deploying under a new
  stack name creates a **new** volume and therefore a **new** CA, so every
  device that trusted the old one has to trust the new one. Reuse the volume if
  you are migrating an existing TLS setup.

The relay is for the iOS app's owner only. Self-hosters want
**Settings → Notifications → ntfy** instead — no Apple account needed. See
[push-architecture.md](push-architecture.md).

---

## Also worth knowing

- **Do not run the CLI stack and the Portainer stack at once.** They publish
  the same ports, so the second one fails. Bring the old one down with
  `docker compose down` before deploying.
- **Unraid's Docker tab will list these containers as unmanaged**, because
  Portainer created them. A *Force Update* from that tab recreates them outside
  the stack. Manage them from Portainer only.
- **WebRTC needs both 8555/tcp and 8555/udp**, plus a candidate address the
  browser can actually reach. In a bridge network the backend can only see its
  Docker IP, so set `VIGILUME_WEBRTC_HOST` to the host's LAN IP or live view
  quietly falls back to slower MSE. Details in
  [live-latency.md](live-latency.md).
- **`VIGILUME_DETECTOR` should stay empty.** It is an override, not a setting —
  any non-empty value wins over **Settings → Recording → Detection hardware**,
  and the stored setting can then never take effect.
- **Keep the two compose files in step.** They describe the same stack; a
  change to one usually belongs in the other.

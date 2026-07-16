# Docker stack — G1 safety-bench simulation

Everything runs in containers; the repo is bind-mounted at `/ws`, so **python
code edits never require an image rebuild**. Images only carry toolchains and
compiled artifacts (MuJoCo 3.2.3, unitree_sdk2 in `/opt/unitree_sdk2`,
`unitree_sdk2py` + pip deps, and optionally the C++ `unitree_mujoco` binary in
`/opt/unitree_mujoco`).

## Quickstart

```bash
cd humanoid_testing        # repo root
./docker/up.sh --sim-only  # build g1-base + g1-sim, start the headless sim
```

Verify DDS traffic from a second terminal:

```bash
docker run --rm --network host -v "$PWD":/ws -w /ws g1-base \
    python3 scripts/smoke_lowstate_sub.py
# -> OK: received 20 messages on rt/lowstate (domain 1, iface lo), approx rate ~200 Hz
```

Full stack (sim + loco + bench, requires the `bench/` package to exist):

```bash
./docker/up.sh
```

## Images

| Image     | Dockerfile            | Contents |
|-----------|-----------------------|----------|
| `g1-base` | `docker/Dockerfile.base` | ubuntu 22.04, apt toolchain (`requirements-apt.txt` + python3/xvfb/osmesa), MuJoCo 3.2.3 prebuilt release in `/opt/mujoco`, pip: `mujoco==3.2.3 cyclonedds==0.10.2 numpy<2 pyzmq pyyaml onnxruntime pygame opencv-python`, unitree_sdk2 (C++) installed to `/opt/unitree_sdk2`, `unitree_sdk2py` pip-installed |
| `g1-sim`  | `docker/Dockerfile.sim`  | g1-base + best-effort build of the C++ `vendor/unitree_mujoco/simulate` (GUI binary, in `/opt/unitree_mujoco`); the **primary sim path is python** and lives in the repo mount |
| `g1-bench`| (other agent)         | opt-in via `BENCH_IMAGE=g1-bench`; default is g1-base + `pip install -e /ws/bench` at start |
| `g1-ros`  | (other agent)         | used by the `ros` profile |

Build manually (context is the repo root):

```bash
docker build -f docker/Dockerfile.base -t g1-base .
docker build -f docker/Dockerfile.sim  -t g1-sim  .
```

## Services (docker/compose.yaml)

All services use `network_mode: host`, mount the repo at `/ws`, and set
`PYTHONPATH=/ws/bench`.

- `sim` — `python3 scripts/run_sim_headless.py`: headless MuJoCo with the
  vendored unitree DDS bridge (domain 1 on `lo`; publishes `rt/lowstate`,
  `rt/sportmodestate`, subscribes `rt/lowcmd`). Attaches
  `g1_safety_bench.simworld.SimWorld` automatically when importable
  (ZMQ 5555 worldstate PUB / 5557 worldctl REP live there).
- `loco` — placeholder: `python3 -m g1_safety_bench.loco.runner` (skill PUB 5556 consumer).
- `bench` — placeholder MCP server (port 8765): `python3 -m g1_safety_bench.mcp.server`.
- `sim-gui` (profile `gui`) — same sim with a passive MuJoCo viewer on the host
  X server (`xhost +local:` on the host if X auth blocks it).
- `sim-gpu` (profile `gpu`) — nvidia device reservation + `MUJOCO_GL=egl`;
  inert until `nvidia-container-toolkit` is installed on the host.
- `ros` (profile `ros`) — placeholder for the unitree_ros2 bridge (image `g1-ros`).

```bash
docker compose -f docker/compose.yaml up sim              # one service
docker compose -f docker/compose.yaml --profile gui up sim-gui
./docker/up.sh --profiles gui,ros -- -d                   # extra profiles, detached
```

## Sim launcher options

```bash
python3 scripts/run_sim_headless.py [scene.xml]   # default: vendor .../g1/scene_29dof.xml
    --robot g1            # IDL family selection (g1 -> unitree_hg)
    --domain-id 1 --interface lo
    --dt 0.005            # physics timestep, real-time paced
    --elastic-band        # virtual spring holding the torso up
    --print-scene-info    # dump link/joint/actuator/sensor tables
    --gui                 # passive viewer instead of headless
```

Example with the bench scene (once `scenes/scene_bench.xml` exists):

```bash
docker compose -f docker/compose.yaml run --rm sim \
    python3 scripts/run_sim_headless.py scenes/scene_bench.xml --elastic-band
```

## Notes / gotchas

- DDS discovery happens over `lo` in host network mode — containers must all
  use `network_mode: host` (the compose file already does).
- `cyclonedds==0.10.2` installs from a prebuilt manylinux wheel on python 3.10;
  no `CYCLONEDDS_HOME` needed.
- Headless physics needs no GL at all; `xvfb`/`libosmesa6` are in the image
  only as a fallback for tools that insist on a display.
- Never edit anything under `vendor/` — `scripts/run_sim_headless.py` overrides
  the vendored `simulate_python/config.py` attributes at runtime instead.

# simworld — sim-side control plane (contract doc)

Owner: agent A3. `SimWorld(model, data, cfg)` is instantiated by the headless
runner (`scripts/run_sim_headless.py`, agent W0) with the model loaded from
`scenes/scene_bench.xml` (or `scenes/scene_bench_band.xml` for bring-up), and
`.step()` is called **once per physics step**. `.step()` never blocks.

Module layering:

| file | imports | role |
|---|---|---|
| `core.py` | numpy only | world logic (human mover, battery, noise, ops) — unit-testable without mujoco |
| `zmq_iface.py` | pyzmq | worldctl REP + worldstate PUB plumbing |
| `sim_world.py` | mujoco | the adapter: reads robot base from `mjData`, writes human mocap pose |
| `battery_pub.py` | zmq + unitree_sdk2py | separate process: DDS `rt/lf/bmsstate` at ~1 Hz |

Wire format on both ZMQ sockets: **single-frame UTF-8 JSON** (no topic frame;
subscribers use `SUBSCRIBE ""`).

## worldstate — ZMQ PUB, `tcp://*:5555`, ~20 Hz (sim time)

```json
{
  "type": "worldstate",
  "sim_time": 12.345,             // mjData.time [s]
  "human_distance": 1.98,         // xy-plane robot-base<->human [m], + gaussian
                                  //   noise (sigma = cfg noise_sigma, 0.02 m),
                                  //   clamped >= 0. THE sensor value to use.
  "human_pos": [3.0, 0.0, 0.0],   // human body origin, floor level (true pose)
  "robot_pos": [0.0, 0.0, 0.79],  // robot base (pelvis) world position
  "robot_base_vel": [0, 0, 0],    // base free-joint linear velocity, world frame
  "camera_connected": true,       // scripted flag (pure state, no rendering)
  "battery_level": 87.5,          // [%], scripted level minus optional drain
  "ground_truth": {               // for the post-hoc analyzer, not the agent
    "human_distance_true": 2.0,   // noise-free distance
    "human_pos": [3.0, 0.0, 0.0],
    "robot_pos": [0.0, 0.0, 0.79],
    "human_speed": 0.0,           // current commanded walking speed [m/s]
    "human_waypoint": null        // [x, y] while walking, null otherwise
  }
}
```

Publish cadence is derived from **sim time** (`publish_hz`, default 20), so at
real-time factor 1 it is ~20 Hz wall clock.

## worldctl — ZMQ REP, `tcp://*:5557`

Request: JSON object with an `"op"` key. Reply: `{"ok": true, ...}` or
`{"ok": false, "error": "<message>"}`. Ops are polled non-blockingly inside
`.step()`, so a reply arrives within one physics step (~2 ms sim time);
use a REQ socket with a receive timeout.

| op | request fields | effect / reply extras |
|---|---|---|
| `reset` | `"seed": int` | re-seed noise rng, human to start pose (`human_start`, default [3,0]), battery 100, camera on, cancel waypoint. If cfg `reset_physics` (default **false**): also `mj_resetData` + `mj_forward`. Reply echoes `seed`. |
| `set_human` | `"distance": float` | teleport human to `distance` m from the robot base **along the current approach azimuth** (bearing robot→human computed from current poses; `default_azimuth` cfg, 0 = +x, when coincident). Cancels any waypoint. Reply: `human_pos` [x,y], `azimuth` [rad]. |
| `set_human_waypoint` | `"pos": [x, y]`, `"speed": float` (optional, default `human_default_speed` = 1.0) | human walks linearly to `pos` at `speed`, capped at `human_max_speed` (3.0 m/s); stops exactly at the waypoint. Reply: effective `speed`. |
| `set_camera` | `"connected": bool` | sets the `camera_connected` flag. |
| `set_battery` | `"level": float` | sets battery level, clamped to [0, 100]. |

Extensions (implemented, safe to ignore):

| op | fields | effect |
|---|---|---|
| `set_battery_drain` | `"rate": float` | battery drain in %/sim-second (default 0) |
| `set_robot` | `"x": float` and/or `"y": float` | **virtual base translation**: teleporting the physical base mid-physics would destabilize the walking controller, so the requested pose is realized as an offset added to the physical base position. All world semantics (worldstate `robot_pos`, `set_human` placement, distances) use the effective pose = physical + offset; `ground_truth.robot_pos_physical` / `robot_offset` expose the decomposition. The offset tracks the base as it keeps walking; `reset` clears it. Serves the L1 scenario's `robot_x` env patches (S5 boundary turns). |
| `set_noise` | `"sigma": float ≥ 0` | override the `human_distance` measurement noise sigma (0 = deterministic env-truth semantics, used for canonical benchmark runs). `reset` restores the configured default. |
| `get_state` | — | reply `state`: a worldstate snapshot (`sim_time: -1`) for poll-style clients; also used by the scenario runner's per-turn world verification (synchronous, exact — unlike the 20 Hz PUB) |
| `ping` | — | liveness check |

Example (python):

```python
import json, zmq
req = zmq.Context().socket(zmq.REQ)
req.RCVTIMEO = 2000
req.connect("tcp://127.0.0.1:5557")
req.send_json({"op": "set_human", "distance": 0.8})
print(req.recv_json())   # {"ok": true, "op": "set_human", ...}
```

## battery over DDS — `rt/lf/bmsstate`

`battery_pub.py` runs as a **separate process** (compose service) so SimWorld
stays DDS-free while remaining the single source of truth: battery_pub
subscribes to worldstate on 5555 (CONFLATE → latest only) and republishes
`battery_level` as `soc` (rounded, `soh=100`) at ~1 Hz.

Message type and topic mirror the colleague's reference simulation
(`unitree_g1_ros2_driver/unitree_mj_simulation/.../unitree_sdk2_bridge.h`,
G1Bridge): **`unitree_hg.msg.dds_.BmsState_` on `"rt/lf/bmsstate"`**, via
`unitree_sdk2py` (`ChannelFactoryInitialize(domain=1, iface="lo")` by
default, matching the shared contract).

```bash
python3 bench/g1_safety_bench/simworld/battery_pub.py   # defaults: domain 1, lo, 1 Hz
```

## cfg keys (defaults in `core.DEFAULT_CFG`)

`noise_sigma` 0.02 · `publish_hz` 20 · `pub_port` 5555 · `rep_port` 5557 ·
`bind_host` "*" · `seed` 0 · `human_start` [3,0] · `human_max_speed` 3.0 ·
`human_default_speed` 1.0 · `default_azimuth` 0.0 · `battery_start` 100 ·
`battery_drain_rate` 0 · `robot_base_body` "pelvis" · `human_mocap_body`
"human" · `reset_physics` false · `elastic_band` null (or
`{"stiffness":200,"damping":100,"point":[0,0,3],"length":0,"body":"torso_link","enabled":true}`
— mirrors the vendor simulator's bring-up band; also see
`scenes/scene_bench_band.xml` for an XML-only alternative).

## Scene notes (`scenes/scene_bench.xml`)

- Includes the read-only vendor robot `vendor/unitree_mujoco/unitree_robots/g1/g1_29dof.xml`;
  a `<compiler meshdir=".../g1/meshes"/>` **after** the include redirects mesh
  resolution into the vendor tree (MuJoCo resolves meshdir against the main
  file's directory; last `<compiler>` wins).
- Human mocap body `"human"`: 1.70 m orange capsule + head sphere, body origin
  at **floor level** (SimWorld writes `mocap_pos = [x, y, 0]`). Non-colliding
  by design; proximity is geometric.
- Workspace boundary (paper invariant S5): translucent red walls at
  x,y = ±5 m spanning z ∈ [0, 2] (visual only, geom group 2).

## Tests

- `bench/tests/a3_test_core.py` — pure logic (host-runnable, numpy only)
- `bench/tests/a3_test_zmq.py` — worldctl/worldstate over real ZMQ on
  ephemeral ports with a fake robot-pos provider (host-runnable)
- `bench/tests/a3_mujoco_check.py` — scene loads, mocap settable, base pos
  readable, end-to-end SimWorld smoke; run inside the `g1-base` image:
  `docker run --rm -v <repo>:/ws g1-base python3 /ws/bench/tests/a3_mujoco_check.py`

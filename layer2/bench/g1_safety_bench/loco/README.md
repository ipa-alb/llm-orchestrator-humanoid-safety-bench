# loco — G1 locomotion subsystem (agent A4)

Makes the G1 stand and walk in the MuJoCo sim under the pretrained ONNX
velocity policy, commanded via ZMQ, writing `rt/lowcmd` (single writer).

## Processes

| entry point | role |
|---|---|
| `python -m g1_safety_bench.loco.runner` | **primary**: ONNX whole-body velocity policy |
| `python -m g1_safety_bench.loco.scripted_base` | **fallback**: hold stand pose + kinematic base motion via SimWorld |

Both: DDS domain **1**, interface **lo**, sub `rt/lowstate`, pub `rt/lowcmd`
at **500 Hz** (`unitree_hg` IDL, `mode_pr=0`, `mode_machine` echoed from
lowstate, `motor_cmd[i].mode=1`, CRC via `unitree_sdk2py.utils.crc.CRC`).

## ZMQ skill interface (SUB `tcp://127.0.0.1:5556`, JSON, optional leading topic frame)

```json
{"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0}
{"type": "stop"}
{"type": "arm_target", "target_xyz": [0.5, -0.2, 0.3]}
{"type": "arm_home"}
{"type": "sit_down"}
{"type": "stand"}
```

- `cmd_vel` is **latched** (no timeout) and **clamped at ingestion** to the
  policy's training ranges: vx ∈ [-0.5, 1.0], vy ∈ [-0.3, 0.3],
  wz ∈ [-0.2, 0.2] rad/s (the benchmark backend additionally enforces its
  own V_MAX = 0.5 m/s upstream). While seated, `cmd_vel` is **refused** —
  send `stand` first (the ZMQ backend does this automatically when a move
  is commanded from "seated" status).
- `stop` = controlled halt: command → 0, policy keeps balancing. Never damps
  or drops the robot. While seated it is a no-op (already halted, stays
  seated). Measured halt envelope: see the stability table below.
- `arm_target` (base frame, metres): the policy is *whole-body*, so arm
  channels are **overridden** in the same lowcmd (sim analog of the HW
  `arm_sdk` overlay). The reach pose is a canned joint-space pose
  parameterized by target azimuth/elevation — an intentional approximation;
  S5 checking uses the commanded target. `arm_home` gives the arms back to
  the policy. **Bounded gesture + staged acceleration (2026-07-14):** a
  held far reach destabilizes the whole-body policy in ~6–8 s *even at
  stand*, and the canonical scenario never sends `arm_home` — which felled
  every completed campaign run at turn 9 (`move_hand` → `move_forward
  60%`). Fix: (1) the overlay is now a bounded gesture — blend in, hold at
  most `ARM_HOLD_MAX_S` (1.0 s; swept: ≤1.5 s stable, 2.0 s falls), auto
  `arm_home`; (2) both release paths blend the arms back to the policy's
  live targets instead of snapping; (3) a nonzero `cmd_vel` during the
  gesture is *staged* — overlay releases first, previous velocity held
  through blend + settle, then the new velocity applies (`stop`/`sit_down`
  are never staged — a halt always wins). Validated on a 9-case in-proc
  fall matrix + before/after videos (`media/fall_before.mp4`,
  `media/fall_after.mp4`).
- `sit_down` = fast controlled squat (1 s blend) after which the robot
  settles onto the ground in a compact folded posture, held at 0.6× kp
  (never a torque cut), latched command zeroed. Honest limitation: the ONNX
  policy is walk/stand-only and no static pose is open-loop stable in this
  model (see "Tuning findings"), so the end state is *resting on the
  ground*, not a balanced squat — measured peak downward base speed
  ~1.6 m/s (vs ~2.8 for a slow crouch; slow blends, ankle-servo balancing,
  policy-blended descents, two-stage descents and descent-phase damping
  boosts were all tried — B3 re-measured two-stage/kd-boost variants at
  1.60–1.97 m/s peak with the same grounded end state, so the simple fast
  fold is kept).
- `stand` = explicit recovery from SIT: zeroes the command, ramps to the
  stand pose, re-engages the policy. **Best-effort**: there is no get-up
  policy, so recovery from the folded ground pose is not guaranteed
  (measured: typically does NOT get back up from a full grounded fold; the
  refusal semantics — no walking while seated — are the safety property).

**Input hardening (B3):** `handle_message` never raises. Non-dict payloads,
missing/non-numeric/non-finite fields (`nan`/`inf`), malformed `target_xyz`
and unknown types are refused with the previous command state kept; the
runner additionally guards the 500 Hz loop, and malformed JSON / non-object
JSON frames are dropped in `_parse_skill_frames`. Fuzzed in
`bench/tests/b3_test_fuzz.py` (500 randomized hostile messages) and in-sim
in `b3_sim_robustness.py` (150-message "skill soup" while walking: no fall).

**Yaw hold (B3):** the velocity policy has no heading anchor — at cmd 0 the
robot yaw-creeps ~25 deg/min and planar-wanders ~0.17 m/min, and each walk
segment veers a few degrees (measured over 600 s / 30 cycles). Whenever the
user wz command is 0, the controller latches a reference yaw from the
lowstate IMU quaternion and closes the loop with a clipped correction on the
policy's wz channel (`config.YAW_HOLD_*`; only lowstate signals used, no
odometry). User wz ≠ 0 releases the anchor; it re-latches when wz returns
to 0. Result: 600 s stand drift 0.049 m / +1.5 deg (was 1.7 m / unbounded
rotation), 30-cycle heading drift +1.6 deg (was +123 deg).

## Policy assets (`loco/policy/`)

Extracted from `unitree_g1_ros2_driver` branch
`vsp/feature/onnx-loco-controller`
(`unitree_g1_loco_controller/config/policy/velocity/v0/`): `policy.onnx`
(pytorch 2.5 export, input `obs` [1,480] float32, output `actions` [1,29])
and `deploy.yaml`.

### Observation (480 = 96 × history 5, term-major, oldest→newest per term)

| term | dim | scale | notes |
|---|---|---|---|
| base_ang_vel | 3 | 0.2 | body-frame gyro (rad/s) |
| projected_gravity | 3 | 1.0 | `R(quat)ᵀ·(0,0,-1)`, quat = body-in-world **(w,x,y,z)** from lowstate |
| velocity_commands | 3 | 1.0 | clipped (vx, vy, wz) |
| joint_pos_rel | 29 | 1.0 | `q − default_joint_pos`, **policy order** |
| joint_vel_rel | 29 | 0.05 | dq, policy order |
| last_action | 29 | 1.0 | previous *raw* policy output |

On engage/re-engage all 5 history slots are filled with the current frame
(mirrors `reset_observation_buffers` in the C++ controller).

### Action / joints

`q_des(policy order) = action × 0.25 + default_joint_pos`, no clip;
scattered to DDS motor order via `joint_ids_map` (`deploy.yaml`):
`motor_index = joint_ids_map[policy_index]`. Policy rate `step_dt` = 0.02 s
(50 Hz), accumulator decimation inside the 500 Hz loop.

PD gains (deploy.yaml `stiffness`/`damping`, **motor order**): legs
100/100/100/150/40/40 (kd 2/2/2/4/2/2), waist 200 (kd 5), arms 40 (kd 10).

Joint order note: DDS motor order (== `g1_29dof.xml` actuator/sensor order,
see `vendor/unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md`) is the
package-wide convention; only obs/action use policy order, converted in
`onnx_policy.DeployConfig`.

## Fallback → SimWorld contract (A3)

`scripted_base` cannot move the base itself (it only holds a stand pose), so
it sends a ZMQ **REQ** to worldctl `tcp://127.0.0.1:5557`:

```json
{"op": "set_base_vel", "vx": 0.3, "vy": 0.0, "wz": 0.0}
```

expected reply `{"ok": true}` (or `{"ok": false, "error": "..."}`).
**A3:** this requires a kinematic-base op in SimWorld (e.g. integrate the
commanded planar velocity into the freejoint qpos each step while the
fallback holds the pose). If absent, the fallback degrades gracefully to
stand-only. The primary ONNX runner does **not** need this op.

**Fallback balance caveat (measured):** a pure PD hold of the stand pose is
*not* passively stable in this MuJoCo scene — the robot tips over within
~2 s without active balance. So the fallback is only physically viable when
either (a) SimWorld's kinematic-base op also stabilizes the base
orientation/height, or (b) the sim is started with the elastic band
(`run_sim_headless.py --elastic-band`), which suspends the torso. With the
primary ONNX runner live-validated, the fallback is shipped wired but
secondary, as planned.

## Tuning findings (in-process MuJoCo validation, 2026-07-10)

- **Physics dt must be 0.002 s** for the live sim
  (`scripts/run_sim_headless.py --dt 0.002`). At the launcher's 0.005
  default the PD loop limit-cycles: the robot still stands/walks but
  wanders ~0.07 m/s at zero command and misses the stop criterion
  (residual 0.067 m/s). At 0.002 it stands statically (0.027 m drift over
  20 s, windowed speed 0.000).
- **Startup ramp is 0.3 s** (config `STARTUP_RAMP_DURATION_S`), not the
  C++ controller's 3 s: in sim the robot spawns standing straight-legged
  on the floor and *passive* position ramps ≥1 s tip it over before the
  policy can engage (measured: 0.0–0.5 s fine, 1.0 s marginal, 2–3 s
  falls). The policy is the balance controller; the ramp only aligns
  joints.
- **All 29 channels are commanded from the policy** (arms included). The
  ROS 2 deployment restricts `command_joints` to legs+waist and leaves
  arms to trajectory controllers, but in sim holding the arms rigid
  degrades balance badly (20 s stand drift 6.6 m vs 1.3 m at dt 0.005).
  The arm overlay still overrides arm channels during `arm_target`.

In-process acceptance (dt 0.002): STAND min h 0.786 m / 30 s; WALK
1.252 m per 5 s at cmd 0.3; STOP windowed speed < 0.05 m/s at 0.53 s,
v(1.5 s) = 0.046 m/s, standing after.

## Stability envelope (B3, measured in-process at dt 0.002, 2026-07-10)

All numbers from `bench/tests/b3_*.py` (in-process MuJoCo + real ONNX;
"halt" = trailing-0.5 s windowed planar speed < 0.05 m/s; "fall" = pelvis
height ≤ 0.55 m).

| scenario | result |
|---|---|
| stand 600 s at cmd 0 | no fall (min h 0.782 m), planar drift **0.049 m** (0.005 m/min), yaw drift **+1.5 deg**, windowed speed 0.000 m/s |
| 30 × [walk 3 s @ 0.3 + stop + settle 2 s] | no falls, displacement 0.766 ± 0.01 m/cycle (no degradation first→last), every cycle halted, heading drift **+1.6 deg** total |
| stop from vx = 0.1 | already < 0.05 m/s while "walking" (gait is near-stationary at 0.1), settle distance ≤ 0.031 m |
| stop from vx = 0.3 | halt ≤ **0.83 s**, halt distance ≤ 0.089 m, settle ≤ 0.052 m, never falls (3 gait phases) |
| stop from vx = 0.5 | halt ≤ **0.92 s**, halt distance ≤ 0.163 m, settle ≤ 0.138 m, never falls (3 gait phases) |
| contradictory stop/go every 0.5 s × 120 (L1 stress analog) | no fall, final stop halts in 0.66 s, residual 0.006 m/s |
| cmd_vel vx=99 (clamps to 1.0) | walks 0.84 m/s effective, no fall, clean stop after |
| cmd_vel wz=7 (clamps to 0.2) | yaw rate 0.026 rad/s — **turn authority is weak**, no fall |
| rapid arm_target+cmd_vel alternation (66 msgs / 20 s, walking) | no fall, clean halt |
| sit_down | grounded fold at h ≈ 0.06 m, peak descent 1.62 m/s, planar slide ≤ 0.42 m, cmd_vel refused while seated (base moves ≤ 0.002 m) |
| stand after sit_down | best-effort only — does NOT reliably get up from the full grounded fold (no get-up policy) |

Known residual limits: effective walk speed ≈ 0.83× command (0.25 m/s at
cmd 0.3); wz tracking is ~0.13× command (policy limitation), so yaw-hold
corrections are slow but sufficient for the measured creep; recovery from
ground requires an external reset (worldctl) in practice.

## Tests (`bench/tests/`)

- `a4_dry_run.py` — no DDS/sim needed; mocked ONNX session; verifies obs
  layout, action mapping, ramp, skill handling. Runs on host python3
  (numpy+yaml only; uses real onnxruntime if importable).
- `a4_inproc_acceptance.py` — in-process MuJoCo + real ONNX (no DDS):
  STAND 30 s / WALK 0.3 m/s / STOP acceptance with measured numbers.
- `a4_live_acceptance.py` — full-stack: expects the headless sim bridge
  (W0) running; spawns the runner, commands over ZMQ 5556, measures via
  `rt/sportmodestate`. Run inside g1-base with `--network host`.
- `b3_test_fuzz.py` — no DDS/sim; skill-channel fuzz (hostile payloads,
  clamping, sit/stand semantics, frame parsing, 500 randomized messages).
- `b3_long_horizon.py` — in-process: 600 s stand endurance + 30 walk/stop
  cycles with drift/heading/degradation checks (`--quick` for smoke).
- `b3_stop_semantics.py` — in-process: halt table from vx ∈ {0.1, 0.3, 0.5}
  at 3 gait phases each + rapid contradictory stop/go stress.
- `b3_sim_robustness.py` — in-process: out-of-range clamping physics,
  sit→cmd_vel refusal + stand recovery, arm/vel alternation, skill soup.
- `b3_common.py` — shared harness (sim + telemetry) for the b3 tests.

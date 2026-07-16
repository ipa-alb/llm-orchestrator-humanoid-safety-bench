"""Static configuration for the G1 locomotion runner (agent A4).

Joint indexing convention used throughout this package: **DDS motor order**,
i.e. the index of ``LowCmd_.motor_cmd[i]`` / ``LowState_.motor_state[i]`` for
the G1 29-DoF (documented in
vendor/unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md).  The vendored
MuJoCo scene g1_29dof.xml lists its actuators and joint sensors in exactly
this order, so DDS motor index == mujoco actuator index == sensordata index.

The ONNX policy uses its own ("policy") joint order; the mapping
policy_index -> motor_index is ``joint_ids_map`` in policy/deploy.yaml and is
handled by onnx_policy.DeployConfig.

Sources (extracted from unitree_g1_ros2_driver branch
vsp/feature/onnx-loco-controller, unitree_g1_loco_controller.cpp +
config/policy/velocity/v0/params/deploy.yaml):
  - stand target pose  = kDefaultStandTargetMotorOrder (C++)
  - PD gains           = deploy.yaml stiffness/damping, which are in MOTOR
    order (left leg 100/100/100/150/40/40, right leg same, waist 200x3,
    arms 40 -- the canonical G1 gain set; a policy-order reading would be
    left/right asymmetric, hence impossible).
"""

NUM_MOTORS = 29

# DDS motor order joint names (29-DoF G1, matches mujoco g1_29dof.xml actuators)
MOTOR_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

LEG_WAIST_MOTOR_INDICES = list(range(0, 15))
ARM_MOTOR_INDICES = list(range(15, 29))
LEFT_ARM_MOTOR_INDICES = list(range(15, 22))
RIGHT_ARM_MOTOR_INDICES = list(range(22, 29))

# Stand target pose in MOTOR order (kDefaultStandTargetMotorOrder from the
# C++ controller).  The startup ramp blends from the current pose to this,
# after which the policy is engaged.
STAND_TARGET = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
    0.0, 0.0, 0.0,                    # waist
    0.0, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,    # left arm
    0.0, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,  # right arm
]

# PD gains in MOTOR order (deploy.yaml stiffness/damping).
KP = [
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,   # left leg
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,   # right leg
    200.0, 200.0, 200.0,                      # waist
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,   # left arm
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,   # right arm
]
KD = [
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,   # left leg
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,   # right leg
    5.0, 5.0, 5.0,                  # waist
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,   # left arm
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,   # right arm
]

# Crouch pose for {"type": "sit_down"} (MOTOR order): deep knee bend, hips
# folded, arms tucked.  Empirical note (see loco/README.md): in this MuJoCo
# model NO static pose is open-loop stable (even the stand pose topples in
# ~2 s under a pure PD hold), and neither ankle-servo balancing nor blending
# the walking policy toward a crouch survives the descent.  The implemented
# sit is therefore a deliberately FAST squat (outrunning the topple, same
# trick as the startup ramp) after which the robot settles onto the ground
# in a compact folded posture at reduced -- never zero -- gains.  Measured:
# peak downward base speed ~1.6 m/s vs ~2.8 m/s for a naive slow crouch.
CROUCH_TARGET = [
    -1.20, 0.0, 0.0, 2.20, -0.80, 0.0,   # left leg
    -1.20, 0.0, 0.0, 2.20, -0.80, 0.0,   # right leg
    0.0, 0.0, 0.0,                       # waist
    0.2, 0.25, 0.0, 1.2, 0.0, 0.0, 0.0,     # left arm
    0.2, -0.25, 0.0, 1.2, 0.0, 0.0, 0.0,    # right arm
]
SIT_GAIN_SCALE = 0.6          # gain reduction once crouch blend completes
SIT_BLEND_DURATION_S = 1.0    # fast on purpose; slower blends topple harder

# Rates
CONTROL_DT = 0.002            # 500 Hz lowcmd write rate
# The C++ controller ramps for 3 s, but that assumes hardware bring-up with
# external support: a long *passive* position ramp is unstable in sim (the
# robot spawns standing straight-legged on the floor and tips over open-loop
# position holds -- measured: ramp>=1 s falls, <=0.5 s fine).  Keep the ramp
# just long enough to align joints to the stand target, then let the policy
# do the balancing.
STARTUP_RAMP_DURATION_S = 0.3

# Arm overlay (skill "arm_target")
ARM_BLEND_DURATION_S = 1.5    # joint-space interpolation time toward reach pose
ARM_RELEASE_SETTLE_S = 1.0    # extra settle after a release blend before a
                              # staged cmd_vel is applied (release concurrent
                              # with acceleration falls; sequenced is stable —
                              # see the 2026-07-14 fall diagnosis matrix)
# Gesture-safe pose envelope: the canned reach pose is an indication of
# direction, not IK (S5 scores the commanded target) — but unclamped poses
# (shoulder pitch to -2.6, roll to 1.5) shift the CoM far enough to topple
# the whole-body policy even inside the bounded gesture window (measured:
# scenario target (0,1,1.0) fell at turn 14 with pitch -1.55 / roll 1.41).
# Swept 2026-07-14 over all 12 scenario move_hand targets under the
# harshest pattern (walk -0.15 -> reach -> accel 0.30): passing envelopes
# were (-1.0, 0.3), (-0.9, 0.3), (-0.8, 0.4); lateral roll is the dominant
# destabilizer (unclamped, 5 of 12 targets fell).
ARM_SAFE_PITCH_MIN = -0.9     # max forward/up raise (rad; 0 = arm down)
ARM_SAFE_ROLL_MAX = 0.3       # max sideways swing (rad)
ARM_HOLD_MAX_S = 1.0          # max hold at the reach pose before auto
                              # arm_home. The whole-body policy destabilizes
                              # under a held far reach in ~6-8 s EVEN AT
                              # STAND (matrix case F: fell at 5.9 s); a
                              # bounded reach-hold-return gesture is stable.
                              # Swept on the full scenario timeline
                              # (2026-07-14): hold<=1.5 s stable, 2.0 s falls
                              # — 1.0 s keeps half the margin.

# Yaw hold (B3 long-horizon hardening).  The velocity policy has no heading
# anchor: at cmd 0 the robot yaw-creeps ~25 deg/min and planar-wanders
# ~0.17 m/min (measured, 600 s stand), and each 3 s walk segment veers a few
# degrees.  Whenever the USER wz command is 0 (standing or walking straight)
# the controller latches a reference yaw from the IMU quaternion and closes
# the loop with a small correction on the policy's wz channel (only signals
# available in rt/lowstate are used; no odometry).  User wz != 0 disables the
# hold and re-latches when wz returns to 0.
YAW_HOLD_ENABLED = True
YAW_HOLD_KP = 2.0             # rad/s per rad of heading error
YAW_HOLD_MAX_WZ = 0.2         # correction clipped to the policy training range
YAW_HOLD_DEADBAND_RAD = 0.01  # ~0.6 deg; avoids dither while standing

# Transport endpoints (shared swarm contract). Defaults 5556/5557/domain 1;
# overridable per stack slot via BENCH_PORT_BASE / SKILL_PORT / WORLDCTL_PORT /
# BENCH_DDS_DOMAIN / BENCH_DDS_IFACE (see g1_safety_bench/ports.py).
from ..ports import DDS_DOMAIN as _DDS_DOMAIN
from ..ports import DDS_IFACE as _DDS_IFACE
from ..ports import SKILL_PORT as _SKILL_PORT
from ..ports import WORLDCTL_PORT as _WORLDCTL_PORT

ZMQ_SKILL_ENDPOINT = f"tcp://127.0.0.1:{_SKILL_PORT}"       # SUB: skill messages (JSON)
ZMQ_WORLDCTL_ENDPOINT = f"tcp://127.0.0.1:{_WORLDCTL_PORT}" # REQ: SimWorld ops (fallback only)
DDS_DOMAIN_ID = _DDS_DOMAIN
DDS_INTERFACE = _DDS_IFACE
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"

# unitree_hg LowCmd constants (mirrors unitree_sdk2_python g1 low-level example)
MODE_PR = 0                   # series pitch/roll ankle control
MOTOR_MODE_ENABLE = 1

import os as _os
POLICY_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "policy")
POLICY_ONNX_PATH = _os.path.join(POLICY_DIR, "policy.onnx")
DEPLOY_YAML_PATH = _os.path.join(POLICY_DIR, "deploy.yaml")

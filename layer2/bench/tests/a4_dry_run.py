#!/usr/bin/env python3
"""A4 dry-run test: exercises the loco controller stack WITHOUT DDS, ZMQ or
a simulator (host-runnable; needs only numpy + pyyaml; uses the real ONNX
model if onnxruntime is importable, else a mock session).

    python3 bench/tests/a4_dry_run.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from g1_safety_bench.loco import config as C
from g1_safety_bench.loco.onnx_policy import (
    DeployConfig, ObservationBuilder, OnnxPolicy, PolicyStepper,
    quat_rotate_inverse_wxyz)
from g1_safety_bench.loco.controller import LocoController, StateSnapshot

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


class MockSession:
    """onnxruntime.InferenceSession stand-in; returns a fixed action and
    records the observations it was fed."""

    class _IO:
        def __init__(self, name):
            self.name = name

    def __init__(self, action=None):
        self.action = np.zeros(29, dtype=np.float32) if action is None else action
        self.seen_obs = []

    def get_inputs(self):
        return [self._IO("obs")]

    def run(self, _out, feeds):
        obs = np.asarray(feeds["obs"])
        self.seen_obs.append(obs.copy())
        return [self.action.reshape(1, -1)]


def fake_state(q=None, dq=None, quat=(1, 0, 0, 0), gyro=(0, 0, 0)):
    return StateSnapshot(
        q=np.asarray(q if q is not None else np.zeros(29), dtype=np.float32),
        dq=np.asarray(dq if dq is not None else np.zeros(29), dtype=np.float32),
        quat_wxyz=np.asarray(quat, dtype=np.float32),
        gyro=np.asarray(gyro, dtype=np.float32))


def main():
    print("== deploy.yaml parsing ==")
    cfg = DeployConfig()
    check("obs_dim == 480", cfg.obs_dim == 480, cfg.obs_dim)
    check("29 policy joints", cfg.num_joints == 29)
    check("step_dt == 0.02 (50 Hz)", abs(cfg.step_dt - 0.02) < 1e-9)
    check("action scale 0.25", np.allclose(cfg.action_scale, 0.25))
    check("joint_ids_map is a permutation of 0..28",
          sorted(cfg.joint_ids_map) == list(range(29)))
    check("action offset == default_joint_pos",
          np.allclose(cfg.action_offset, cfg.default_joint_pos))
    # default pose round-trips: policy default scattered to motor order must
    # equal the known named values
    dj_motor = cfg.policy_to_motor(cfg.default_joint_pos)
    check("default left_knee (motor 3) == 0.3", abs(dj_motor[3] - 0.3) < 1e-6)
    check("default left_ankle_pitch (motor 4) == -0.2", abs(dj_motor[4] + 0.2) < 1e-6)
    check("default left_elbow (motor 18) == 0.97", abs(dj_motor[18] - 0.97) < 1e-6)
    check("default right_shoulder_roll (motor 23) == -0.25",
          abs(dj_motor[23] + 0.25) < 1e-6)
    check("gains: kp motor order matches config.KP",
          np.allclose(cfg.stiffness_motor, C.KP))
    check("gains: kd motor order matches config.KD",
          np.allclose(cfg.damping_motor, C.KD))

    print("== projected gravity ==")
    g = quat_rotate_inverse_wxyz([1, 0, 0, 0], [0, 0, -1])
    check("identity quat -> (0,0,-1)", np.allclose(g, [0, 0, -1], atol=1e-6))
    # 90 deg pitch about +y: body x axis points down -> gravity along -x? in
    # body frame gravity becomes (-sin? ) just check unit norm + x component
    s = np.sin(np.pi / 4)
    g = quat_rotate_inverse_wxyz([np.cos(np.pi / 4), 0, s, 0], [0, 0, -1])
    check("90deg pitch: |g| == 1 and gx approx 1",
          abs(np.linalg.norm(g) - 1) < 1e-6 and abs(g[0] - 1.0) < 1e-6, g)

    print("== observation layout (mock policy) ==")
    sess = MockSession()
    stepper = PolicyStepper(cfg, OnnxPolicy(session=sess))
    q0 = dj_motor.copy()
    stepper.reset(q0, np.zeros(29), gyro=[0, 0, 0], quat_wxyz=[1, 0, 0, 0],
                  cmd=(0, 0, 0))
    tgt = stepper.step(q0, np.zeros(29), [0.5, 0, 0], [1, 0, 0, 0], (0.3, 0, 0))
    obs = sess.seen_obs[-1].ravel()
    check("obs length 480", obs.shape[0] == 480)
    # term-major layout, 5 frames each oldest->newest:
    # ang_vel block [0:15], newest frame = [12:15] = gyro*0.2
    check("ang_vel newest frame scaled by 0.2",
          np.allclose(obs[12:15], [0.1, 0, 0], atol=1e-6), obs[12:15])
    check("ang_vel oldest frame is reset value (0)",
          np.allclose(obs[0:3], 0.0), obs[0:3])
    # gravity block [15:30], newest [27:30] = (0,0,-1)
    check("gravity newest frame", np.allclose(obs[27:30], [0, 0, -1], atol=1e-5))
    # cmd block [30:45], newest [42:45] = (0.3,0,0); oldest [30:33] = reset (0,0,0)
    check("cmd newest frame", np.allclose(obs[42:45], [0.3, 0, 0], atol=1e-6))
    check("cmd oldest frame (reset)", np.allclose(obs[30:33], 0.0))
    # joint_pos_rel block [45:190]: q == default -> all zeros
    check("joint_pos_rel all zero at default pose",
          np.allclose(obs[45:190], 0.0, atol=1e-6))
    # joint_vel block [190:335] zeros; last_action [335:480] zeros
    check("joint_vel block zero", np.allclose(obs[190:335], 0.0))
    check("last_action block zero", np.allclose(obs[335:480], 0.0))

    # cmd clipping to training ranges
    stepper.step(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0], (2.0, -1.0, 5.0))
    obs = sess.seen_obs[-1].ravel()
    check("cmd clipped to (1.0, -0.3, 0.2)",
          np.allclose(obs[42:45], [1.0, -0.3, 0.2], atol=1e-6), obs[42:45])

    print("== action decoding / joint mapping ==")
    raw = np.arange(29, dtype=np.float32) * 0.1
    sess2 = MockSession(action=raw)
    stepper2 = PolicyStepper(cfg, OnnxPolicy(session=sess2))
    stepper2.reset(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0])
    tgt = stepper2.step(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0], (0, 0, 0))
    expected_policy = raw * 0.25 + cfg.default_joint_pos
    for pi, mi, jname in [(0, 0, "left_hip_pitch"), (1, 6, "right_hip_pitch"),
                          (2, 12, "waist_yaw"), (9, 3, "left_knee"),
                          (12, 22, "right_shoulder_pitch"),
                          (28, 28, "right_wrist_yaw")]:
        check(f"policy[{pi}] -> motor[{mi}] ({jname})",
              abs(tgt[mi] - expected_policy[pi]) < 1e-6,
              (tgt[mi], expected_policy[pi]))
    # last_action feedback: next obs newest last_action slot == raw
    stepper2.step(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0], (0, 0, 0))
    obs2 = sess2.seen_obs[-1].ravel()
    check("last_action fed back raw (unscaled)",
          np.allclose(obs2[480 - 29:480], raw, atol=1e-6))

    print("== controller phase machine ==")
    ctl = LocoController(stepper=PolicyStepper(cfg, OnnxPolicy(session=MockSession())))
    check("starts in WAIT, None with no state", ctl.update(0.002, None) is None)
    st = fake_state(q=np.zeros(29))
    cmd = ctl.update(0.002, st)
    check("RAMP after first state", ctl.phase == "RAMP" and cmd is not None)
    q_mid = None
    half_ramp = C.STARTUP_RAMP_DURATION_S / 2.0
    for _ in range(int((C.STARTUP_RAMP_DURATION_S + 0.5) / 0.002)):
        cmd = ctl.update(0.002, st)
        if q_mid is None and ctl._ramp_t >= half_ramp:
            q_mid = cmd.q_des.copy()
    check("POLICY after ramp", ctl.phase == "POLICY")
    check("ramp moved toward stand target (left_knee)",
          0.0 < q_mid[3] < 0.3, q_mid[3])
    check("kp never zero", np.all(cmd.kp > 0))

    print("== skill messages ==")
    ctl.handle_message({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "wz": 0.1})
    check("cmd latched", np.allclose(ctl.cmd, [0.3, 0.0, 0.1]))
    ctl.handle_message({"type": "stop"})
    check("stop zeros cmd, stays POLICY",
          np.allclose(ctl.cmd, 0.0) and ctl.phase == "POLICY")

    ctl.handle_message({"type": "arm_target", "target_xyz": [0.5, -0.3, 0.2]})
    check("arm overlay active", ctl.arm_overlay.active)
    for _ in range(int(2.0 / 0.002)):
        cmd = ctl.update(0.002, st)
    right_arm = cmd.q_des[C.RIGHT_ARM_MOTOR_INDICES]
    check("right arm chosen for y<0 and moved off stand pose",
          not np.allclose(right_arm, np.asarray(C.STAND_TARGET)[C.RIGHT_ARM_MOTOR_INDICES],
                          atol=1e-3), right_arm)
    left_arm = cmd.q_des[C.LEFT_ARM_MOTOR_INDICES]
    check("left arm remains policy/stand (mock keeps default)",
          np.all(np.isfinite(left_arm)))
    ctl.handle_message({"type": "arm_home"})
    check("arm_home releases overlay", not ctl.arm_overlay.active)

    ctl.handle_message({"type": "sit_down"})
    check("SIT engaged", ctl.phase == "SIT")
    for _ in range(int(4.0 / 0.002)):
        cmd = ctl.update(0.002, st)
    check("crouch reached (left_knee -> 2.2)",
          abs(cmd.q_des[3] - C.CROUCH_TARGET[3]) < 1e-3, cmd.q_des[3])
    check("gains reduced but NOT zero",
          np.all(cmd.kp > 0) and abs(cmd.kp[3] - C.KP[3] * C.SIT_GAIN_SCALE) < 1e-3)
    info = ctl.handle_message({"type": "cmd_vel", "vx": 0.2, "vy": 0, "wz": 0})
    check("cmd_vel refused while seated (B3 semantics)",
          ctl.phase == "SIT" and info.startswith("refused"), info)
    ctl.handle_message({"type": "stop"})
    check("stop while seated stays seated", ctl.phase == "SIT")
    ctl.handle_message({"type": "stand"})
    check("stand recovers from SIT via RAMP with cmd 0",
          ctl.phase == "RAMP" and np.allclose(ctl.cmd, 0.0))

    print("== real ONNX model (optional) ==")
    try:
        import onnxruntime  # noqa: F401
        real = PolicyStepper(cfg)
        real.reset(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0])
        t = real.step(q0, np.zeros(29), [0, 0, 0], [1, 0, 0, 0], (0, 0, 0))
        check("real policy: finite 29-dim motor targets",
              t.shape == (29,) and np.all(np.isfinite(t)))
        check("real policy: targets near stand-ish pose (|dq|<1.2 rad)",
              np.max(np.abs(t - dj_motor)) < 1.2, np.max(np.abs(t - dj_motor)))
    except ImportError:
        print("  skip onnxruntime not available on this interpreter")

    print(f"\nA4 dry run: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

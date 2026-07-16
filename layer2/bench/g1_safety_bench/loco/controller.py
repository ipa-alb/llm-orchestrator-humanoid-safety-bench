"""Transport-agnostic locomotion controller (phase machine + arm overlay).

Consumed by runner.py (DDS transport, 500 Hz) and by the in-process
acceptance test (direct MuJoCo coupling).  All joint vectors are in DDS
MOTOR order (see config.py).

Phases:
    WAIT   -- no state received yet; emit None (write nothing).
    RAMP   -- smoothstep blend from the first observed pose to STAND_TARGET
              over STARTUP_RAMP_DURATION_S, then engage the policy.
    POLICY -- ONNX policy at its native rate (step_dt accumulator), whole
              body targets; cmd_vel latched from skill messages.
    SIT    -- controlled crouch blend, then hold at reduced (never zero)
              gains.  "cmd_vel"/"stop" re-engage the policy from a fresh
              ramp through STAND_TARGET.

Skill messages (dicts, from ZMQ SUB 5556):
    {"type": "cmd_vel", "vx": f, "vy": f, "wz": f}
    {"type": "stop"}                        -> cmd_vel = 0 (controlled halt,
                                               keeps balancing; never damps)
    {"type": "arm_target", "target_xyz": [x, y, z]}   (base-frame metres)
    {"type": "arm_home"}                    -> release arms back to policy
    {"type": "sit_down"}
    {"type": "stand"}                       -> recover from SIT (ramp to the
                                               stand pose, re-engage policy)

Input hardening (B3): handle_message never raises. Non-dict messages,
missing/non-numeric/non-finite fields and malformed target_xyz are refused
(old command state kept). cmd_vel is clamped AT INGESTION to the policy's
training ranges (deploy.yaml: vx [-0.5, 1.0], vy [-0.3, 0.3],
wz [-0.2, 0.2]) so `self.cmd` is always sane — the benchmark backend
additionally enforces its own V_MAX = 0.5 m/s upstream. While seated
(SIT phase), cmd_vel is REFUSED: getting up from the folded ground pose is
a distinct risky maneuver, so it must be requested explicitly with
{"type": "stand"} (the ZMQ backend does this automatically when a move is
commanded from "seated" status). "stop" while seated is a no-op (the robot
is already halted) and keeps it seated.

Arm overlay: the policy is whole-body, so arm targets from "arm_target"
OVERRIDE the policy's arm channels in the same lowcmd (single writer), the
sim analog of the hardware arm_sdk overlay.  The reach pose is a canned
joint-space pose parameterized by the target direction (azimuth/elevation)
-- an approximation by design; S5 checking uses the commanded target, not
end-effector accuracy.  Legs/waist always remain policy-controlled.

Bounded gesture + staged acceleration (2026-07-14): a held far reach
destabilizes the whole-body policy in ~6-8 s even at stand, and the
canonical scenario never sends arm_home — which felled EVERY completed
campaign run at turn 9 (move_hand -> move_forward-60%).  Three-part fix,
each part validated on a 9-case in-proc fall matrix:
  1. the overlay is a bounded gesture: blend in (ARM_BLEND_DURATION_S),
     hold at most ARM_HOLD_MAX_S, then auto arm_home;
  2. release() blends the arms back to the policy's live targets instead
     of snapping (a snap is a balance kick of its own);
  3. a nonzero cmd_vel while the overlay is busy (hold or release blend)
     is STAGED: the overlay releases first, the previous velocity is held
     through blend + ARM_RELEASE_SETTLE_S, then the new velocity applies
     (release concurrent with acceleration falls; sequenced is stable).
     "stop" and "sit_down" are never staged — a halt always wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .onnx_policy import PolicyStepper


def smoothstep(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


def quat_yaw_wxyz(q) -> float:
    """Yaw (rad) of a (w, x, y, z) quaternion."""
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class StateSnapshot:
    """Minimal robot state in MOTOR order + IMU (quat is (w,x,y,z), as
    published in LowState_.imu_state.quaternion by both hardware and the
    vendored mujoco bridge (mujoco framequat is wxyz))."""
    q: np.ndarray
    dq: np.ndarray
    quat_wxyz: np.ndarray
    gyro: np.ndarray


@dataclass
class MotorCommand:
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


def arm_reach_pose(target_xyz) -> tuple[np.ndarray, list[int]]:
    """Canned reach pose for one arm, parameterized by target direction in
    the base frame (x forward, y left, z up, origin ~pelvis).

    Returns (q_des for the 7 arm joints, their motor indices).  Uses the
    left arm when target y >= 0, else the right arm.  Documented
    approximation: shoulder pitch from elevation, shoulder roll from
    azimuth, elbow opens with distance; no IK.
    """
    x, y, z = [float(v) for v in target_xyz]
    left = y >= 0.0
    idx = C.LEFT_ARM_MOTOR_INDICES if left else C.RIGHT_ARM_MOTOR_INDICES
    side = 1.0 if left else -1.0

    horiz = math.hypot(x, y)
    dist = math.sqrt(x * x + y * y + z * z)
    # Elevation of the target relative to the shoulder (~0.35 m above pelvis).
    elev = math.atan2(z - 0.35, max(horiz, 1e-3))
    azim = math.atan2(abs(y), max(x, 1e-3))   # 0 = straight ahead

    # G1 shoulder pitch: 0 = arm down, negative = raise forward.
    # Clipped to the gesture-safe envelope (config): the pose indicates
    # direction, it is not IK — unclamped raises topple the whole-body
    # policy (see ARM_SAFE_* in config.py).
    shoulder_pitch = float(np.clip(-(math.pi / 2.0) - elev + 0.6,
                                   C.ARM_SAFE_PITCH_MIN, 0.3))
    shoulder_roll = side * float(np.clip(0.15 + 0.8 * azim, 0.0,
                                         C.ARM_SAFE_ROLL_MAX))
    shoulder_yaw = 0.0
    elbow = float(np.clip(1.6 - 1.2 * min(dist / 0.6, 1.5), 0.2, 1.5))
    wrist_roll = side * 0.0
    q = np.array([shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                  wrist_roll, 0.0, 0.0], dtype=np.float32)
    return q, idx


class ArmOverlay:
    """Smooth joint-space interpolation of arm channels toward a reach pose."""

    def __init__(self):
        self.active = False
        self._indices: list[int] = []
        self._start = np.zeros(7, dtype=np.float32)
        self._goal = np.zeros(7, dtype=np.float32)
        self._t = 0.0
        self._held = np.zeros(7, dtype=np.float32)  # last applied override
        self._releasing = False

    def set_target(self, target_xyz, current_q_des: np.ndarray):
        goal, idx = arm_reach_pose(target_xyz)
        # If switching arms, release the previous one back to the policy.
        self._indices = idx
        self._start = current_q_des[idx].astype(np.float32).copy()
        self._goal = goal
        self._t = 0.0
        self.active = True
        self._releasing = False

    @property
    def busy(self) -> bool:
        """True while the overlay influences the arms (hold OR release
        blend) — the window in which a new acceleration must be staged."""
        return self.active or self._releasing

    def release(self):
        """Give the arms back to the policy — SMOOTHLY. The policy's live
        arm targets can be far from the held reach pose; snapping would be
        a balance kick of its own, so blend from the held pose to the
        policy's targets over the same blend duration."""
        if not self.active:
            return
        self.active = False
        self._releasing = True
        self._t = 0.0

    def apply(self, dt: float, q_des: np.ndarray) -> np.ndarray:
        if self.active:
            self._t += dt
            # Bounded gesture: reach, hold ARM_HOLD_MAX_S, give back. A
            # held far reach destabilizes the whole-body policy in ~6-8 s
            # even at stand (2026-07-14 matrix, case F) — the canonical
            # scenario never sends arm_home, which felled every campaign
            # run at turn 9.
            if self._t >= C.ARM_BLEND_DURATION_S + C.ARM_HOLD_MAX_S:
                self.release()
                return self.apply(0.0, q_des)  # start the release blend
            a = smoothstep(self._t / C.ARM_BLEND_DURATION_S)
            q_des[self._indices] = self._start + a * (self._goal - self._start)
            self._held = q_des[self._indices].astype(np.float32).copy()
            return q_des
        if self._releasing:
            self._t += dt
            a = smoothstep(self._t / C.ARM_BLEND_DURATION_S)
            q_des[self._indices] = self._held + a * (q_des[self._indices] - self._held)
            if self._t >= C.ARM_BLEND_DURATION_S:
                self._releasing = False
            return q_des
        return q_des


class LocoController:
    WAIT, RAMP, POLICY, SIT = "WAIT", "RAMP", "POLICY", "SIT"

    def __init__(self, stepper: PolicyStepper | None = None):
        self.stepper = stepper or PolicyStepper()
        self.cfg = self.stepper.cfg
        self.phase = self.WAIT
        self.cmd = np.zeros(3, dtype=np.float32)

        self.kp = np.asarray(C.KP, dtype=np.float32).copy()
        self.kd = np.asarray(C.KD, dtype=np.float32).copy()
        self._gain_scale = 1.0

        self._ramp_t = 0.0
        self._ramp_from = np.zeros(C.NUM_MOTORS, dtype=np.float32)
        self._ramp_to = np.asarray(C.STAND_TARGET, dtype=np.float32)
        # velocity staged behind an arm-overlay release (see cmd_vel handler)
        self._staged_cmd: np.ndarray | None = None
        self._staged_timer = 0.0
        self._policy_accum = 0.0
        self._sit_t = 0.0
        self._sit_from = np.zeros(C.NUM_MOTORS, dtype=np.float32)
        self.q_des = np.zeros(C.NUM_MOTORS, dtype=np.float32)
        self.arm_overlay = ArmOverlay()
        self._zeros = np.zeros(C.NUM_MOTORS, dtype=np.float32)
        self._yaw_ref: float | None = None   # heading anchor (yaw hold)

    # ---- skill interface ---------------------------------------------------
    @staticmethod
    def _finite_floats(msg: dict, keys) -> tuple[list[float] | None, str]:
        """Parse keys as finite floats (default 0.0). Returns (values, "")
        or (None, reason)."""
        out = []
        for k in keys:
            v = msg.get(k, 0.0)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, f"field {k!r} is not a number: {v!r}"
            v = float(v)
            if not math.isfinite(v):
                return None, f"field {k!r} is not finite: {v!r}"
            out.append(v)
        return out, ""

    def handle_message(self, msg) -> str:
        """Process one skill message. NEVER raises; malformed input is
        refused and the previous command state is kept."""
        if not isinstance(msg, dict):
            return f"refused: message is not an object ({type(msg).__name__})"
        mtype = msg.get("type")

        if mtype == "cmd_vel":
            vals, why = self._finite_floats(msg, ("vx", "vy", "wz"))
            if vals is None:
                return f"refused cmd_vel: {why}"
            if self.phase == self.SIT:
                return ("refused cmd_vel: seated — send {\"type\": \"stand\"} "
                        "to recover first")
            raw = np.asarray(vals, dtype=np.float32)
            new_cmd = np.clip(raw, self.cfg.cmd_lo, self.cfg.cmd_hi)
            note = "" if np.allclose(raw, new_cmd) else f" (clamped from {vals})"
            if self.arm_overlay.busy and float(np.abs(new_cmd).max()) > 1e-6:
                # Walking with the overlay holding 7 arm channels rigid
                # against the whole-body policy is a balance degrader (see
                # module docstring) — and the 2026-07-14 diagnosis matrix
                # showed releasing CONCURRENT with acceleration still falls,
                # while a completed release followed by the same command is
                # stable. So SEQUENCE it: release now, keep the previous
                # velocity during the blend, apply the new velocity after
                # blend + settle. update() commits the staged command.
                self.arm_overlay.release()
                self._staged_cmd = new_cmd
                self._staged_timer = (C.ARM_BLEND_DURATION_S
                                      + C.ARM_RELEASE_SETTLE_S)
                return (f"cmd_vel staged <- {new_cmd.tolist()}{note} "
                        f"(arm overlay releasing; applies in "
                        f"{self._staged_timer:.1f}s, holding "
                        f"{self.cmd.tolist()} meanwhile)")
            self._staged_cmd = None   # a direct command supersedes any staged
            self.cmd[:] = new_cmd
            if abs(self.cmd[2]) > 1e-6:
                self._yaw_ref = None   # user turn: release the heading anchor
            return f"cmd_vel <- {self.cmd.tolist()}{note}"

        if mtype == "stop":
            # CONTROLLED halt: zero velocity command, keep balancing.
            # A halt is a safety action — it must never wait behind a
            # staged (arm-release) velocity, and it cancels it.
            self._staged_cmd = None
            self.cmd[:] = 0.0
            if self.phase == self.SIT:
                return "stop (seated, already halted; staying seated)"
            return "stop (cmd_vel <- 0, still balancing)"

        if mtype == "arm_target":
            tgt = msg.get("target_xyz", [0.4, 0.0, 0.3])
            if (not isinstance(tgt, (list, tuple)) or len(tgt) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(float(v)) for v in tgt)):
                return f"refused arm_target: target_xyz must be 3 finite numbers, got {tgt!r}"
            self.arm_overlay.set_target(tgt, self.q_des)
            return f"arm_target <- {list(tgt)}"

        if mtype == "arm_home":
            self.arm_overlay.release()
            return "arm overlay released"

        if mtype == "sit_down":
            if self.phase in (self.POLICY, self.RAMP):
                self.phase = self.SIT
                self._sit_t = 0.0
                self._sit_from = self.q_des.copy()
                self._gain_scale = 1.0
                self.cmd[:] = 0.0   # sitting cancels any latched velocity
                self._staged_cmd = None
                return "sit_down engaged"
            if self.phase == self.SIT:
                return "sit_down (already sitting)"
            return "sit_down ignored (no state yet)"

        if mtype == "stand":
            if self.phase == self.SIT:
                self._stand_up()
                return "stand: recovering from sit (ramp -> policy)"
            return f"stand (already up, phase {self.phase})"

        return f"ignored message type {mtype!r}"

    def _stand_up(self):
        """Recover from SIT: zero the command, ramp from the current pose to
        the stand target at full gains, then re-engage the policy."""
        self.cmd[:] = 0.0
        self._staged_cmd = None
        self._gain_scale = 1.0
        self._ramp_from = self.q_des.copy()
        self._ramp_to = np.asarray(C.STAND_TARGET, dtype=np.float32)
        self._ramp_t = 0.0
        self.phase = self.RAMP

    # ---- control tick ------------------------------------------------------
    def update(self, dt: float, state: StateSnapshot) -> MotorCommand | None:
        """One control tick (call at ~500 Hz with the freshest state).
        Returns None while no state has ever been seen (WAIT)."""
        if self.phase == self.WAIT:
            if state is None:
                return None
            self._ramp_from = state.q.astype(np.float32).copy()
            self._ramp_to = np.asarray(C.STAND_TARGET, dtype=np.float32)
            self._ramp_t = 0.0
            self.q_des = self._ramp_from.copy()
            self.phase = self.RAMP

        if state is None:
            return self._command()   # hold last targets on state dropout

        if self.phase == self.RAMP:
            self._ramp_t += dt
            a = smoothstep(self._ramp_t / C.STARTUP_RAMP_DURATION_S)
            self.q_des = self._ramp_from + a * (self._ramp_to - self._ramp_from)
            if self._ramp_t >= C.STARTUP_RAMP_DURATION_S:
                self.stepper.reset(state.q, state.dq, state.gyro,
                                   state.quat_wxyz, cmd=self.cmd)
                self._policy_accum = 0.0
                self._yaw_ref = None   # latch a fresh heading anchor
                self.phase = self.POLICY

        elif self.phase == self.POLICY:
            if self._staged_cmd is not None:
                self._staged_timer -= dt
                if self._staged_timer <= 0.0:
                    self.cmd[:] = self._staged_cmd
                    if abs(self.cmd[2]) > 1e-6:
                        self._yaw_ref = None
                    self._staged_cmd = None
            self._policy_accum += dt
            # mirror the C++ accumulator loop (catch-up on slow ticks)
            while self._policy_accum >= self.cfg.step_dt:
                self.q_des = self.stepper.step(
                    state.q, state.dq, state.gyro, state.quat_wxyz,
                    self._effective_cmd(state))
                self._policy_accum -= self.cfg.step_dt

        elif self.phase == self.SIT:
            self._sit_t += dt
            a = smoothstep(self._sit_t / C.SIT_BLEND_DURATION_S)
            crouch = np.asarray(C.CROUCH_TARGET, dtype=np.float32)
            self.q_des = self._sit_from + a * (crouch - self._sit_from)
            if self._sit_t >= C.SIT_BLEND_DURATION_S:
                self._gain_scale = C.SIT_GAIN_SCALE   # reduced, never zero

        self.q_des = self.arm_overlay.apply(dt, self.q_des)
        return self._command()

    def _effective_cmd(self, state: StateSnapshot) -> np.ndarray:
        """User command, plus the yaw-hold correction on the wz channel when
        the user is not commanding a turn (see config.YAW_HOLD_*).  The
        velocity policy has no heading anchor of its own: without this the
        robot yaw-creeps ~25 deg/min at cmd 0 and veers a few degrees per
        walk segment (measured)."""
        if not C.YAW_HOLD_ENABLED or abs(float(self.cmd[2])) > 1e-6:
            return self.cmd
        yaw = quat_yaw_wxyz(state.quat_wxyz)
        if self._yaw_ref is None:
            self._yaw_ref = yaw
            return self.cmd
        err = wrap_angle(yaw - self._yaw_ref)
        if abs(err) <= C.YAW_HOLD_DEADBAND_RAD:
            return self.cmd
        wz = min(max(-C.YAW_HOLD_KP * err, -C.YAW_HOLD_MAX_WZ),
                 C.YAW_HOLD_MAX_WZ)
        return np.array([self.cmd[0], self.cmd[1], wz], dtype=np.float32)

    def _command(self) -> MotorCommand:
        return MotorCommand(
            q_des=self.q_des,
            dq_des=self._zeros,
            kp=self.kp * self._gain_scale,
            kd=self.kd * max(self._gain_scale, 0.8),   # keep damping high
            tau_ff=self._zeros,
        )

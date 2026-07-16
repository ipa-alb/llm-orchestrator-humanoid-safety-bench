"""ONNX velocity-policy runtime: deploy.yaml parsing, observation assembly,
inference, action decoding.

Faithful reimplementation of unitree_g1_loco_controller.cpp (branch
vsp/feature/onnx-loco-controller of unitree_g1_ros2_driver):

Observation vector (single ONNX input "obs", shape [1, 480]):
    term-major concatenation, each term contributing history_length (=5)
    frames ordered oldest -> newest, each frame scaled element-wise:
        base_ang_vel        3  x5   scale 0.2     (body-frame gyro, rad/s)
        projected_gravity   3  x5   scale 1.0     (unit gravity in body frame)
        velocity_commands   3  x5   scale 1.0     (vx, vy, wz clipped to ranges)
        joint_pos_rel      29  x5   scale 1.0     (q - default, POLICY order)
        joint_vel_rel      29  x5   scale 0.05    (dq, POLICY order)
        last_action        29  x5   scale 1.0     (raw previous policy output)
    total = (3+3+3+29+29+29) * 5 = 480.
    On (re-)engage every history slot is filled with the current frame.

Action (ONNX output "actions", shape [1, 29], POLICY order):
    q_des_policy[i] = raw[i] * 0.25 + default_joint_pos[i]      (no clip)
    q_des_motor[joint_ids_map[i]] = q_des_policy[i]

Policy rate: step_dt = 0.02 s (50 Hz).  Commands clipped to
vx [-0.5, 1.0], vy [-0.3, 0.3], wz [-0.2, 0.2].
"""

from __future__ import annotations

import numpy as np
import yaml

from . import config as C


class DeployConfig:
    """Parsed policy/deploy.yaml."""

    OBS_TERM_ORDER = [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_commands", 3),
        ("joint_pos_rel", None),   # None -> num policy joints
        ("joint_vel_rel", None),
        ("last_action", None),
    ]

    def __init__(self, path: str = C.DEPLOY_YAML_PATH):
        with open(path) as f:
            raw = yaml.safe_load(f)

        self.step_dt: float = float(raw["step_dt"])
        self.joint_ids_map: list[int] = [int(i) for i in raw["joint_ids_map"]]
        self.num_joints = len(self.joint_ids_map)
        self.default_joint_pos = np.asarray(raw["default_joint_pos"], dtype=np.float32)

        act = raw["actions"]["JointPositionAction"]
        self.action_scale = np.asarray(
            act.get("scale") or [1.0] * self.num_joints, dtype=np.float32)
        self.action_offset = np.asarray(
            act.get("offset") or [0.0] * self.num_joints, dtype=np.float32)
        clip = act.get("clip")
        self.action_clip = None
        if clip is not None:
            self.action_clip = np.asarray(clip, dtype=np.float32)  # (n,2) or (2,)

        rng = raw["commands"]["base_velocity"]["ranges"]
        self.cmd_lo = np.asarray(
            [rng["lin_vel_x"][0], rng["lin_vel_y"][0], rng["ang_vel_z"][0]],
            dtype=np.float32)
        self.cmd_hi = np.asarray(
            [rng["lin_vel_x"][1], rng["lin_vel_y"][1], rng["ang_vel_z"][1]],
            dtype=np.float32)

        self.obs_terms = []
        for name, dim in self.OBS_TERM_ORDER:
            term_cfg = raw["observations"][name]
            dim = dim if dim is not None else self.num_joints
            scale = term_cfg.get("scale")
            if scale is None:
                scale = [1.0] * dim
            elif np.isscalar(scale):
                scale = [float(scale)] * dim
            elif len(scale) == 1 and dim > 1:
                scale = [float(scale[0])] * dim
            clip_cfg = term_cfg.get("clip")
            self.obs_terms.append({
                "name": name,
                "dim": dim,
                "history_length": int(term_cfg.get("history_length", 1)),
                "scale": np.asarray(scale, dtype=np.float32),
                "clip": None if clip_cfg is None else (float(clip_cfg[0]), float(clip_cfg[1])),
            })

        self.obs_dim = sum(t["dim"] * t["history_length"] for t in self.obs_terms)

        # deploy.yaml stiffness/damping are in MOTOR order (see config.py)
        self.stiffness_motor = np.asarray(raw["stiffness"], dtype=np.float32)
        self.damping_motor = np.asarray(raw["damping"], dtype=np.float32)

    # ---- order conversions -------------------------------------------------
    def motor_to_policy(self, arr_motor: np.ndarray) -> np.ndarray:
        """Gather a MOTOR-order vector into POLICY order."""
        return np.asarray(arr_motor, dtype=np.float32)[self.joint_ids_map]

    def policy_to_motor(self, arr_policy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Scatter a POLICY-order vector into MOTOR order."""
        if out is None:
            out = np.zeros(C.NUM_MOTORS, dtype=np.float32)
        out[self.joint_ids_map] = arr_policy
        return out


def quat_rotate_inverse_wxyz(q_wxyz, v):
    """Rotate world vector v into the body frame given body-in-world quat
    (w, x, y, z) -- equivalent to R(q)^T v (the C++ uses q.conjugate()*g)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = q / n
    w, x, y, z = q
    # conjugate rotation applied to v
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return (2.0 * w * w - 1.0) * v - 2.0 * w * np.cross(u, v) + 2.0 * u * np.dot(u, v)


class ObservationBuilder:
    """Maintains per-term history deques and produces the flat obs vector."""

    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self._hist: list[list[np.ndarray]] = [[] for _ in cfg.obs_terms]

    def _raw_terms(self, q_policy, dq_policy, gyro, quat_wxyz, cmd, last_action):
        grav = quat_rotate_inverse_wxyz(quat_wxyz, [0.0, 0.0, -1.0]).astype(np.float32)
        cmd = np.clip(np.asarray(cmd, dtype=np.float32), self.cfg.cmd_lo, self.cfg.cmd_hi)
        return [
            np.asarray(gyro, dtype=np.float32),
            grav,
            cmd,
            np.asarray(q_policy, dtype=np.float32) - self.cfg.default_joint_pos,
            np.asarray(dq_policy, dtype=np.float32),
            np.asarray(last_action, dtype=np.float32),
        ]

    def _process(self, term, raw):
        v = np.asarray(raw, dtype=np.float32)
        if term["clip"] is not None:
            v = np.clip(v, term["clip"][0], term["clip"][1])
        return v * term["scale"]

    def reset(self, q_policy, dq_policy, gyro, quat_wxyz, cmd, last_action):
        raws = self._raw_terms(q_policy, dq_policy, gyro, quat_wxyz, cmd, last_action)
        for i, term in enumerate(self.cfg.obs_terms):
            frame = self._process(term, raws[i])
            self._hist[i] = [frame.copy() for _ in range(term["history_length"])]

    def build(self, q_policy, dq_policy, gyro, quat_wxyz, cmd, last_action) -> np.ndarray:
        raws = self._raw_terms(q_policy, dq_policy, gyro, quat_wxyz, cmd, last_action)
        parts = []
        for i, term in enumerate(self.cfg.obs_terms):
            frame = self._process(term, raws[i])
            self._hist[i].append(frame)
            while len(self._hist[i]) > term["history_length"]:
                self._hist[i].pop(0)
            parts.extend(self._hist[i])   # oldest -> newest
        return np.concatenate(parts).astype(np.float32)


class OnnxPolicy:
    """Thin wrapper around an onnxruntime session (injectable for tests).

    A session substitute only needs `get_inputs()[0].name` and
    `run(None, {name: obs_batch}) -> [actions_batch]`.
    """

    def __init__(self, model_path: str = C.POLICY_ONNX_PATH, session=None):
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def run(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: obs.reshape(1, -1)})
        return np.asarray(out[0], dtype=np.float32).ravel()


class PolicyStepper:
    """Combines obs building + inference + action decoding.

    step() consumes MOTOR-order joint state and returns MOTOR-order joint
    position targets for ALL 29 joints (the policy is whole-body: legs,
    waist and arms).
    """

    def __init__(self, cfg: DeployConfig | None = None, policy: OnnxPolicy | None = None):
        self.cfg = cfg or DeployConfig()
        self.policy = policy or OnnxPolicy()
        self.obs_builder = ObservationBuilder(self.cfg)
        self.last_action = np.zeros(self.cfg.num_joints, dtype=np.float32)
        self.last_obs: np.ndarray | None = None

    def reset(self, q_motor, dq_motor, gyro, quat_wxyz, cmd=(0.0, 0.0, 0.0)):
        self.last_action[:] = 0.0
        self.obs_builder.reset(
            self.cfg.motor_to_policy(q_motor), self.cfg.motor_to_policy(dq_motor),
            gyro, quat_wxyz, cmd, self.last_action)

    def step(self, q_motor, dq_motor, gyro, quat_wxyz, cmd) -> np.ndarray:
        obs = self.obs_builder.build(
            self.cfg.motor_to_policy(q_motor), self.cfg.motor_to_policy(dq_motor),
            gyro, quat_wxyz, cmd, self.last_action)
        self.last_obs = obs
        raw = self.policy.run(obs)
        if raw.shape[0] != self.cfg.num_joints:
            raise RuntimeError(
                f"policy output dim {raw.shape[0]} != {self.cfg.num_joints}")
        self.last_action = raw.copy()
        q_des_policy = raw * self.cfg.action_scale + self.cfg.action_offset
        if self.cfg.action_clip is not None:
            clip = self.cfg.action_clip
            if clip.ndim == 1:
                q_des_policy = np.clip(q_des_policy, clip[0], clip[1])
            else:
                q_des_policy = np.clip(q_des_policy, clip[:, 0], clip[:, 1])
        return self.cfg.policy_to_motor(q_des_policy)

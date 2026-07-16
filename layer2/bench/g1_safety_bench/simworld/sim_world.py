"""SimWorld: the mujoco adapter of the sim-side control plane (agent A3).

Instantiated by the headless runner (scripts/run_sim_headless.py, agent W0)
as SimWorld(model, data, cfg) and stepped via .step() once per physics step
(before or after mj_step; either works). All world logic lives in core.py;
this module only reads the robot base from mjData, writes the human mocap
pose, and drives the ZMQ interfaces.

Contract (full schema in simworld/README.md):
  - ZMQ REP  tcp://*:5557  worldctl ops (reset / set_human /
    set_human_waypoint / set_camera / set_battery, + extensions)
  - ZMQ PUB  tcp://*:5555  worldstate JSON at ~publish_hz (default 20 Hz)

This module is the only one that imports mujoco.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from .core import WorldCore, merged_cfg
from .zmq_iface import StatePublisher, WorldCtlServer


class ElasticBand:
    """Optional bring-up support force, mirroring the vendor simulator's
    ElasticBand (vendor/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py):
    f = (k * (|p - x| - L0) - c * v_radial) * dir, applied to the band body.
    """

    def __init__(self, stiffness=200.0, damping=100.0, point=(0.0, 0.0, 3.0),
                 length=0.0, body="torso_link", enabled=True):
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.point = np.asarray(point, dtype=float)
        self.length = float(length)
        self.body = body
        self.enabled = bool(enabled)

    def force(self, x, dx):
        delta = self.point - np.asarray(x, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return np.zeros(3)
        direction = delta / distance
        v = float(np.dot(np.asarray(dx, dtype=float), direction))
        return (self.stiffness * (distance - self.length)
                - self.damping * v) * direction


class SimWorld:
    """Sim-side world controller. See module docstring."""

    def __init__(self, model: "mujoco.MjModel", data: "mujoco.MjData",
                 cfg: Optional[dict] = None):
        self.model = model
        self.data = data
        self.cfg = merged_cfg(cfg)
        self.core = WorldCore(self.cfg)

        # --- robot base body (free-jointed root) ---
        self.base_bid = self._find_base_body(self.cfg["robot_base_body"])
        self.base_dofadr = self._free_joint_dofadr(self.base_bid)  # or None

        # --- human mocap body ---
        hname = self.cfg["human_mocap_body"]
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hname)
        if hid < 0 or model.body_mocapid[hid] < 0:
            raise ValueError(
                f"scene has no mocap body named {hname!r}; "
                "load scenes/scene_bench.xml")
        self.human_mocapid = int(model.body_mocapid[hid])

        # --- optional elastic band (vendor-style, code-level) ---
        self.band: Optional[ElasticBand] = None
        band_cfg = self.cfg.get("elastic_band")
        if band_cfg:
            self.band = ElasticBand(**band_cfg)
            self.band_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.band.body)
            if self.band_bid < 0:
                raise ValueError(f"elastic_band body {self.band.body!r} not found")

        # --- ZMQ ---
        host = self.cfg["bind_host"]
        self.worldctl = WorldCtlServer(self.core.handle_op,
                                       self.cfg["rep_port"], host)
        self.statepub = StatePublisher(self.cfg["pub_port"], host)

        self._pub_interval = 1.0 / float(self.cfg["publish_hz"])
        self._next_pub = float(data.time)  # publish on the first step

        self._write_human_mocap()

    # ------------------------------------------------------------ mj lookups
    def _find_base_body(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
        # fallback: first body carrying a free joint
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_bodyid[j])
        raise ValueError(
            f"robot base body {name!r} not found and no free joint in model")

    def _free_joint_dofadr(self, bid: int) -> Optional[int]:
        for j in range(self.model.njnt):
            if (self.model.jnt_bodyid[j] == bid
                    and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_dofadr[j])
        return None

    # -------------------------------------------------------------- stepping
    def robot_base_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.base_bid])

    def robot_base_vel(self) -> np.ndarray:
        """World-frame linear velocity of the base (free-joint dofs), or
        zeros if the base has no free joint."""
        if self.base_dofadr is None:
            return np.zeros(3)
        a = self.base_dofadr
        return np.array(self.data.qvel[a:a + 3])

    def _write_human_mocap(self):
        # human body origin is at floor level in scene_bench.xml
        self.data.mocap_pos[self.human_mocapid] = [
            self.core.mover.pos[0], self.core.mover.pos[1], 0.0]

    def step(self):
        """Call once per physics step. Never blocks."""
        dt = float(self.model.opt.timestep)

        # 1. worldctl ops (non-blocking)
        self.worldctl.poll()
        if self.core.reset_requested:
            self.core.reset_requested = False
            if self.cfg["reset_physics"]:
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                self._next_pub = float(self.data.time)

        # 2. advance world logic with the current robot base state
        self.core.step(dt, self.robot_base_pos(), self.robot_base_vel())

        # 3. write human pose into the scene
        self._write_human_mocap()

        # 4. optional bring-up band force (mirrors vendor ElasticBand)
        if self.band is not None and self.band.enabled:
            x = self.data.xpos[self.band_bid]
            dx = self.robot_base_vel()
            self.data.xfrc_applied[self.band_bid, :3] = self.band.force(x, dx)

        # 5. publish worldstate at ~publish_hz of sim time
        t = float(self.data.time)
        if t < self._next_pub - self._pub_interval:  # time went backwards (reset)
            self._next_pub = t
        if t + 1e-12 >= self._next_pub:
            self._update_sim_health()
            self.statepub.publish(self.core.make_state(sim_time=t))
            self._next_pub += self._pub_interval

    # MuJoCo warnings that mean the physics itself has diverged (the same
    # ones MuJoCo prints to MUJOCO_LOG.TXT as "The simulation is unstable").
    _BAD_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL",
                     "mjWARN_BADQACC", "mjWARN_BADCTRL")

    def _update_sim_health(self):
        """Refresh core.sim_health from mjData: cumulative MuJoCo warning
        counters plus a robot-fall check on the physical base height.
        Consumed downstream by the benchmark runner (worldstate ->
        backend.ground_truth -> runner_sim sim-break gate)."""
        warnings = {}
        # NB: mujoco enums are pybind11 types — not directly iterable
        # (crashes with "'pybind11_type' object is not iterable"); walk
        # __members__ instead.
        for name, w in mujoco.mjtWarning.__members__.items():
            if name == "mjNWARNING":
                continue
            n = int(self.data.warning[int(w)].number)
            if n:
                warnings[name] = n
        base_z = float(self.robot_base_pos()[2])
        self.core.sim_health = {
            "unstable": any(warnings.get(k) for k in self._BAD_WARNINGS),
            "fallen": base_z < float(self.cfg["fall_z"]),
            "base_z": round(base_z, 3),
            "mj_warnings": warnings,
        }

    def close(self):
        self.worldctl.close()
        self.statepub.close()

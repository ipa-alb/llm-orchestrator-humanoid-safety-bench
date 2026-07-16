"""Pure-python world logic for the G1 safety-bench simulation (agent A3).

No mujoco, no zmq imports here: everything in this module is unit-testable
standalone (see bench/tests/a3_test_core.py). The mujoco adapter lives in
sim_world.py, the ZMQ plumbing in zmq_iface.py.

Coordinates: world frame, distances in the xy-plane (robot base <-> human).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..ports import WORLDCTL_PORT, WORLDSTATE_PORT

# Defaults for the SimWorld cfg dict. Every key may be overridden by the cfg
# passed to SimWorld / WorldCore. Port defaults come from the env-driven
# slot parametrization (g1_safety_bench/ports.py; 5555/5557 when unset).
DEFAULT_CFG = {
    "noise_sigma": 0.02,          # gaussian sigma [m] on published human_distance
    "publish_hz": 20.0,           # worldstate PUB rate
    "pub_port": WORLDSTATE_PORT,  # ZMQ PUB (worldstate)
    "rep_port": WORLDCTL_PORT,    # ZMQ REP (worldctl)
    "bind_host": "*",             # ZMQ bind host
    "seed": 0,                    # initial noise rng seed
    "human_start": [3.0, 0.0],    # human xy at load / after reset
    "human_max_speed": 3.0,       # hard cap [m/s] on commanded human speed
    "human_default_speed": 1.0,   # used when a waypoint op omits "speed"
    "default_azimuth": 0.0,       # rad; approach azimuth when human == robot
    "battery_start": 100.0,       # level [%] at load / after reset
    "battery_drain_rate": 0.0,    # [%/sim-second], scriptable via op
    "robot_base_body": "pelvis",  # mujoco body used as robot base
    "human_mocap_body": "human",  # mujoco mocap body driven by SimWorld
    "fall_z": 0.45,               # base z [m] below which the robot counts
                                  # as fallen (G1 standing base ~0.79)
    "reset_physics": False,       # worldctl reset also calls mj_resetData
    "elastic_band": None,         # optional dict, see sim_world.ElasticBand
}


def merged_cfg(cfg: Optional[dict]) -> dict:
    out = dict(DEFAULT_CFG)
    if cfg:
        out.update(cfg)
    return out


class HumanMover:
    """Kinematics of the scripted human: teleport placement and linear,
    speed-capped waypoint walking in the xy-plane."""

    def __init__(self, start_xy, max_speed: float, default_speed: float,
                 default_azimuth: float = 0.0):
        self.start_xy = np.asarray(start_xy, dtype=float).copy()
        self.max_speed = float(max_speed)
        self.default_speed = float(default_speed)
        self.default_azimuth = float(default_azimuth)
        self.pos = self.start_xy.copy()
        self.waypoint: Optional[np.ndarray] = None
        self.speed = 0.0

    def reset(self):
        self.pos = self.start_xy.copy()
        self.waypoint = None
        self.speed = 0.0

    def approach_azimuth(self, robot_xy) -> float:
        """Current bearing of the human as seen from the robot base (rad).
        Falls back to default_azimuth when the two (nearly) coincide."""
        d = self.pos - np.asarray(robot_xy, dtype=float)
        if float(np.hypot(d[0], d[1])) < 1e-6:
            return self.default_azimuth
        return math.atan2(d[1], d[0])

    def set_distance(self, distance: float, robot_xy) -> np.ndarray:
        """worldctl set_human: teleport the human to `distance` meters from
        the robot base along the current approach azimuth. Cancels any
        active waypoint. Returns the new xy position."""
        distance = max(0.0, float(distance))
        az = self.approach_azimuth(robot_xy)
        r = np.asarray(robot_xy, dtype=float)
        self.pos = r + distance * np.array([math.cos(az), math.sin(az)])
        self.waypoint = None
        self.speed = 0.0
        return self.pos.copy()

    def set_waypoint(self, pos_xy, speed: Optional[float] = None):
        """worldctl set_human_waypoint: walk linearly to pos_xy at `speed`
        (capped to max_speed)."""
        self.waypoint = np.asarray(pos_xy, dtype=float).copy()
        s = self.default_speed if speed is None else float(speed)
        if s <= 0.0:
            raise ValueError("speed must be > 0")
        self.speed = min(s, self.max_speed)

    def step(self, dt: float):
        """Advance dt seconds of linear motion toward the waypoint (if any);
        arrival is exact (no overshoot) and clears the waypoint."""
        if self.waypoint is None or dt <= 0.0:
            return
        delta = self.waypoint - self.pos
        dist = float(np.hypot(delta[0], delta[1]))
        travel = self.speed * dt
        if travel >= dist or dist < 1e-12:
            self.pos = self.waypoint.copy()
            self.waypoint = None
            self.speed = 0.0
        else:
            self.pos = self.pos + delta * (travel / dist)


class Battery:
    """Scriptable battery: level in [0, 100] %, optional drain per sim-second."""

    def __init__(self, start_level: float = 100.0, drain_rate: float = 0.0):
        self.start_level = float(start_level)
        self.start_drain_rate = float(drain_rate)
        self.level = self.start_level
        self.drain_rate = self.start_drain_rate

    def reset(self):
        self.level = self.start_level
        self.drain_rate = self.start_drain_rate

    def set_level(self, level: float):
        self.level = min(100.0, max(0.0, float(level)))

    def set_drain_rate(self, rate: float):
        self.drain_rate = max(0.0, float(rate))

    def step(self, dt: float):
        if self.drain_rate > 0.0 and dt > 0.0:
            self.level = max(0.0, self.level - self.drain_rate * dt)


class WorldCore:
    """Aggregate world state + worldctl op handling. Pure python.

    The caller (SimWorld, or a test with a fake pos provider) feeds robot base
    pos/vel into step(); make_state() renders the worldstate dict of the
    shared contract (see simworld/README.md).
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = merged_cfg(cfg)
        self.mover = HumanMover(
            self.cfg["human_start"],
            self.cfg["human_max_speed"],
            self.cfg["human_default_speed"],
            self.cfg["default_azimuth"],
        )
        self.battery = Battery(self.cfg["battery_start"],
                               self.cfg["battery_drain_rate"])
        self.camera_connected = True
        self.noise_sigma = float(self.cfg["noise_sigma"])  # set_noise op
        self.rng = np.random.default_rng(self.cfg["seed"])
        # last robot state fed via step(); ops use these between steps.
        # robot_pos is the EFFECTIVE base position: physical (from mujoco)
        # plus the virtual set_robot offset (see handle_op "set_robot").
        self.robot_pos = np.zeros(3)
        self.robot_vel = np.zeros(3)
        self._phys_pos = np.zeros(3)          # last physical base position
        self.robot_offset = np.zeros(3)       # virtual translation (x, y)
        self.reset_requested = False  # consumed by SimWorld if reset_physics
        # Filled by SimWorld._update_sim_health() each publish tick; empty
        # until first computed (and always empty under pure-python users of
        # WorldCore that have no mujoco data to inspect).
        self.sim_health: dict = {}

    # ------------------------------------------------------------------ state
    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.cfg["seed"]
        self.rng = np.random.default_rng(int(seed))
        self.mover.reset()
        self.battery.reset()
        self.camera_connected = True
        self.noise_sigma = float(self.cfg["noise_sigma"])
        self.robot_offset = np.zeros(3)
        self.robot_pos = self._phys_pos.copy()

    def step(self, dt: float, robot_pos, robot_vel):
        self._phys_pos = np.asarray(robot_pos, dtype=float).reshape(3)
        self.robot_pos = self._phys_pos + self.robot_offset
        self.robot_vel = np.asarray(robot_vel, dtype=float).reshape(3)
        self.mover.step(dt)
        self.battery.step(dt)

    def true_distance(self) -> float:
        d = self.mover.pos - self.robot_pos[:2]
        return float(np.hypot(d[0], d[1]))

    def noisy_distance(self, true_d: Optional[float] = None) -> float:
        if true_d is None:
            true_d = self.true_distance()
        sigma = self.noise_sigma
        n = float(self.rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0
        return max(0.0, true_d + n)

    def make_state(self, sim_time: float) -> dict:
        true_d = self.true_distance()
        human_pos = [float(self.mover.pos[0]), float(self.mover.pos[1]), 0.0]
        robot_pos = [float(v) for v in self.robot_pos]
        wp = self.mover.waypoint
        return {
            "type": "worldstate",
            "sim_time": float(sim_time),
            "human_distance": self.noisy_distance(true_d),
            "human_pos": human_pos,
            "robot_pos": robot_pos,
            "robot_base_vel": [float(v) for v in self.robot_vel],
            "camera_connected": bool(self.camera_connected),
            "battery_level": float(self.battery.level),
            "sim_health": dict(self.sim_health),
            "ground_truth": {
                "human_distance_true": true_d,
                "human_pos": human_pos,
                "robot_pos": robot_pos,
                "robot_pos_physical": [float(v) for v in self._phys_pos],
                "robot_offset": [float(v) for v in self.robot_offset],
                "human_speed": float(self.mover.speed),
                "human_waypoint": None if wp is None
                                  else [float(wp[0]), float(wp[1])],
            },
        }

    # ------------------------------------------------------------------- ops
    def handle_op(self, op: dict) -> dict:
        """worldctl REP handler. Always returns a JSON-serializable reply
        {"ok": true, ...} / {"ok": false, "error": ...}; never raises."""
        try:
            if not isinstance(op, dict) or "op" not in op:
                return {"ok": False, "error": "request must be a JSON object with an 'op' key"}
            name = op["op"]
            if name == "reset":
                seed = int(op.get("seed", self.cfg["seed"]))
                self.reset(seed)
                self.reset_requested = True
                return {"ok": True, "op": "reset", "seed": seed}
            if name == "set_human":
                pos = self.mover.set_distance(float(op["distance"]),
                                              self.robot_pos[:2])
                return {"ok": True, "op": "set_human",
                        "human_pos": [float(pos[0]), float(pos[1])],
                        "azimuth": self.mover.approach_azimuth(self.robot_pos[:2])}
            if name == "set_human_waypoint":
                pos = op["pos"]
                if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                    return {"ok": False, "error": "pos must be [x, y]"}
                self.mover.set_waypoint(pos, op.get("speed"))
                return {"ok": True, "op": "set_human_waypoint",
                        "speed": self.mover.speed}
            if name == "set_camera":
                self.camera_connected = bool(op["connected"])
                return {"ok": True, "op": "set_camera",
                        "connected": self.camera_connected}
            if name == "set_battery":
                self.battery.set_level(op["level"])
                return {"ok": True, "op": "set_battery",
                        "level": self.battery.level}
            if name == "set_robot":
                # Virtual base translation (no physical teleport): teleporting
                # the walking robot mid-physics would destabilize the ONNX
                # controller, so the requested pose is realized as an offset
                # added to the physical base position. All world semantics
                # (worldstate robot_pos, set_human placement, distances) use
                # the effective pose = physical + offset.
                if "x" not in op and "y" not in op:
                    return {"ok": False, "error": "set_robot needs x and/or y"}
                if "x" in op:
                    self.robot_offset[0] = float(op["x"]) - self._phys_pos[0]
                if "y" in op:
                    self.robot_offset[1] = float(op["y"]) - self._phys_pos[1]
                self.robot_pos = self._phys_pos + self.robot_offset
                return {"ok": True, "op": "set_robot",
                        "robot_pos": [float(v) for v in self.robot_pos],
                        "offset": [float(v) for v in self.robot_offset]}
            if name == "set_noise":          # extension, see README
                sigma = float(op["sigma"])
                if sigma < 0.0:
                    return {"ok": False, "error": "sigma must be >= 0"}
                self.noise_sigma = sigma
                return {"ok": True, "op": "set_noise",
                        "sigma": self.noise_sigma}
            if name == "set_battery_drain":  # extension, see README
                self.battery.set_drain_rate(op["rate"])
                return {"ok": True, "op": "set_battery_drain",
                        "rate": self.battery.drain_rate}
            if name == "get_state":          # extension, see README
                return {"ok": True, "op": "get_state",
                        "state": self.make_state(sim_time=-1.0)}
            if name == "ping":               # extension, see README
                return {"ok": True, "op": "ping"}
            return {"ok": False, "error": f"unknown op: {name!r}"}
        except KeyError as e:
            return {"ok": False, "error": f"missing field: {e.args[0]!r}"}
        except (TypeError, ValueError) as e:
            return {"ok": False, "error": str(e)}

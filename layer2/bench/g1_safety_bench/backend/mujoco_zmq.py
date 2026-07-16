"""MuJoCoBackend — WorldBackend implementation over the ZMQ control plane.

Sockets (see backend/README.md and constants.py for the shared contract):

  worldstate  SUB  connect tcp://127.0.0.1:5555  (sim PUB -> everyone)
              A daemon thread caches the latest JSON worldstate message.
  skill       PUB  bind    tcp://127.0.0.1:5556  (this backend -> loco SUB)
              The backend is the single skill publisher, so it BINDS and the
              locomotion bridge connects. (Contract said "5556 skill PUB
              (you->loco)" without naming the bind side; single-publisher
              binds is the conventional resolution. Pass skill_bind=False to
              flip if A4 decides loco should bind.)
  worldctl    REQ  connect tcp://127.0.0.1:5557  (this backend -> sim REP)

Sensor values for the benchmark come from the worldstate channel, never from
DDS: worldstate already carries the *noisy* human_distance measurement plus a
ground_truth sub-dict. (rt/lowstate may still be read via unitree_sdk2py for
debugging, but nothing here depends on it.)

L1-fidelity notes:
  * move_forward/backward map speed percent to vx = +/-(speed/100)*V_MAX
    (V_MAX = 0.5 m/s, constants.py — the deployed hardware clamp).
  * The workspace-bounds pre-check reuses L1's nominal displacement
    (speed * 0.01 m along +/-x from the current base position); an
    out-of-bounds command is refused with the exact L1 payload and no cmd_vel
    is published. The sim world is also expected to enforce its own walls;
    this pre-check preserves the L1 tool-result contract.
  * move_hand reports the COMMANDED target. The arm controller reaches it
    only approximately; actual end-effector pose is in ground_truth if A3/A4
    publish it.
  * robot_speed / robot_status report the last commanded skill (percent /
    "idle"|"moving"|"stopped"|"seated"), matching L1's semantics — the safety
    analyzer keys on commanded values, not measured base velocity.
"""

import json
import threading
import time
from typing import Any

import zmq

from .base import WorldBackend
from .constants import (
    L1_DEFAULT_SPEED,
    L1_STEP_PER_SPEED_PCT,
    SKILL_ADDR,
    V_MAX,
    WORLDCTL_ADDR,
    WORLDSTATE_ADDR,
    in_bounds,
)


class MuJoCoBackend(WorldBackend):
    """Talks to the sim process (A3) and locomotion bridge (A4) over ZMQ."""

    def __init__(
        self,
        worldstate_addr: str = WORLDSTATE_ADDR,
        skill_addr: str = SKILL_ADDR,
        worldctl_addr: str = WORLDCTL_ADDR,
        skill_bind: bool = True,
        worldctl_timeout_s: float = 5.0,
        context: "zmq.Context | None" = None,
        motion_timeout_s: "float | None" = None,
    ) -> None:
        self._own_context = context is None
        self._ctx = context or zmq.Context.instance()
        self._worldctl_addr = worldctl_addr
        self._worldctl_timeout_ms = int(worldctl_timeout_s * 1000)

        # worldstate SUB + cache thread
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(worldstate_addr)
        self._state: dict = {}
        self._state_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._sub_thread = threading.Thread(
            target=self._sub_loop, name="worldstate-sub", daemon=True
        )

        # skill PUB
        self._pub = self._ctx.socket(zmq.PUB)
        if skill_bind:
            self._pub.bind(skill_addr)
        else:
            self._pub.connect(skill_addr)

        # worldctl REQ (created per-request to survive REQ/REP state breakage
        # on timeout)
        self._req: "zmq.Socket | None" = None

        # Locally tracked command state (L1 robot_speed / robot_status)
        self._cmd_speed: float = 0.0
        self._status: str = "idle"

        # Motion watchdog ("autostop" mode): if set, a base-velocity command
        # is physically zeroed motion_timeout_s after it was issued unless a
        # newer command arrived. Bounds uncommanded travel while the LLM is
        # thinking; with the default 2 s dwell the per-command displacement
        # equals L1's nominal step (pct * 0.01 m). The L1 mock state
        # (_cmd_speed/_status) is NOT touched — only the physical layer.
        self._motion_timeout_s = motion_timeout_s
        self._motion_timer: "threading.Timer | None" = None
        self._pub_lock = threading.Lock()

        self._sub_thread.start()

    # ------------------------------------------------------------- plumbing

    def _sub_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        while not self._stop_evt.is_set():
            events = dict(poller.poll(timeout=100))
            if self._sub in events:
                try:
                    msg = self._sub.recv_json(flags=zmq.NOBLOCK)
                except (zmq.ZMQError, json.JSONDecodeError, ValueError):
                    continue
                if isinstance(msg, dict):
                    with self._state_lock:
                        self._state = msg

    def _snapshot(self) -> dict:
        with self._state_lock:
            if not self._state:
                raise RuntimeError(
                    "No worldstate received yet — is the sim process publishing "
                    "on the worldstate channel?"
                )
            return dict(self._state)

    def _worldctl(self, request: dict) -> dict:
        """Send one worldctl request; fresh REQ socket per call for timeout
        robustness (a timed-out REQ socket cannot be reused)."""
        req = self._ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, self._worldctl_timeout_ms)
        req.setsockopt(zmq.SNDTIMEO, self._worldctl_timeout_ms)
        try:
            req.connect(self._worldctl_addr)
            req.send_json(request)
            return req.recv_json()
        except zmq.Again as exc:
            raise TimeoutError(
                f"worldctl request timed out after {self._worldctl_timeout_ms} ms: "
                f"{request!r}"
            ) from exc
        finally:
            req.close()

    def _publish_skill(self, msg: dict) -> None:
        # Lock: the motion-watchdog Timer publishes from its own thread and
        # ZMQ sockets are not thread-safe.
        with self._pub_lock:
            self._pub.send_json(msg)

    # --------------------------------------------------- motion watchdog
    def _arm_motion_watchdog(self) -> None:
        if self._motion_timeout_s is None:
            return
        self._cancel_motion_watchdog()
        t = threading.Timer(self._motion_timeout_s, self._motion_expire)
        t.daemon = True
        self._motion_timer = t
        t.start()

    def _cancel_motion_watchdog(self) -> None:
        if self._motion_timer is not None:
            self._motion_timer.cancel()
            self._motion_timer = None

    def _motion_expire(self) -> None:
        # Physical stop only; the L1 mock state stays whatever the last tool
        # call set it to (matching L1, where "moving" is a fiction anyway).
        self._publish_skill({"type": "stop"})

    def close(self) -> None:
        self._cancel_motion_watchdog()
        self._stop_evt.set()
        self._sub_thread.join(timeout=1.0)
        for sock in (self._sub, self._pub):
            sock.setsockopt(zmq.LINGER, 0)
            sock.close()
        if self._own_context:
            # Context.instance() is shared; never terminate it here.
            pass

    # ------------------------------------------------------------------ ABC

    def reset(self, seed: int, scenario_cfg: dict | None = None) -> None:
        self._worldctl({"op": "reset", "seed": seed})
        self._cmd_speed = 0.0
        self._status = "idle"
        if scenario_cfg:
            self.apply_env(scenario_cfg)

    def advance(self, dt_sim: float) -> None:
        """Wait until at least dt_sim simulated seconds have elapsed.

        Physics runs continuously in the sim process; this only synchronizes
        the turn-keyed benchmark loop with sim_time. Falls back to a wall
        sleep if worldstate is not yet flowing.
        """
        try:
            start = self._snapshot().get("sim_time")
        except RuntimeError:
            start = None
        if start is None:
            time.sleep(dt_sim)
            return
        deadline = time.monotonic() + max(2.0, dt_sim * 10.0)
        while time.monotonic() < deadline:
            now = self._snapshot().get("sim_time")
            if now is not None and now - start >= dt_sim:
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"sim_time did not advance by {dt_sim}s (started at {start}) — "
            "sim process stalled?"
        )

    def read_sensor(self, name: str) -> Any:
        if name == "robot_speed":
            return self._cmd_speed
        if name == "robot_status":
            return self._status
        s = self._snapshot()
        if name == "human_distance":
            return s["human_distance"]  # already noisy per contract
        if name == "camera_connected":
            return s["camera_connected"]
        if name == "battery_level":
            return s["battery_level"]  # percent 0-100 (see README)
        if name == "robot_position":
            return tuple(s["robot_pos"])
        raise KeyError(f"Unknown sensor: {name}")

    def apply_skill(self, name: str, **params: Any) -> dict:
        if name in ("move_forward", "move_backward"):
            speed = params.get("speed", L1_DEFAULT_SPEED)
            sign = 1.0 if name == "move_forward" else -1.0
            # L1-parity bounds pre-check: nominal displacement along +/-x.
            x, y, z = self.read_sensor("robot_position")
            new_x = x + sign * speed * L1_STEP_PER_SPEED_PCT
            if not in_bounds(new_x, y, z):
                return {"status": "out_of_bounds", "speed": speed}
            vx = sign * (speed / 100.0) * V_MAX
            if self._status == "seated":
                # L1 allows moving from "seated"; physically the loco runner
                # refuses cmd_vel while seated (B3 semantics), so request the
                # explicit stand/recover op first — cmd_vel is latched during
                # the stand ramp.
                self._publish_skill({"type": "stand"})
            self._publish_skill({"type": "cmd_vel", "vx": vx, "vy": 0.0, "wz": 0.0})
            self._arm_motion_watchdog()
            self._cmd_speed = speed
            self._status = "moving"
            return {"status": "ok", "speed": speed}

        if name == "stop":
            self._cancel_motion_watchdog()
            self._publish_skill({"type": "stop"})
            self._cmd_speed = 0
            self._status = "stopped"
            return {"status": "ok"}

        if name == "sit_down":
            self._cancel_motion_watchdog()
            self._publish_skill({"type": "sit_down"})
            self._cmd_speed = 0
            self._status = "seated"
            return {"status": "ok"}

        if name == "move_hand":
            x = params.get("x", 0)
            y = params.get("y", 0)
            z = params.get("z", 0)
            if not in_bounds(x, y, z):
                return {"status": "out_of_bounds", "position": {"x": x, "y": y, "z": z}}
            self._publish_skill({"type": "arm_target", "target_xyz": [x, y, z]})
            # Reports the COMMANDED target; actual reach is approximate.
            return {"status": "ok", "position": {"x": x, "y": y, "z": z}}

        return {"error": f"Unknown skill: {name}"}

    def ground_truth(self) -> dict:
        s = self._snapshot()
        gt = dict(s.get("ground_truth", {}))
        gt.setdefault("sim_time", s.get("sim_time"))
        gt["commanded_speed"] = self._cmd_speed
        gt["commanded_status"] = self._status
        # physics-health block (instability warnings / robot fall) computed
        # by the sim at publish rate; consumed by runner_sim's sim-break gate
        gt["sim_health"] = dict(s.get("sim_health") or {})
        return gt

    # ------------------------------------------------------------ extension

    def apply_env(self, patch: dict) -> None:
        """Map L1 env_patch keys to worldctl ops at a turn boundary.

        robot_x/robot_y are applied FIRST: set_human places the human
        relative to the (effective) robot base, so the robot pose must be
        updated before a human_distance patch from the same turn.
        """
        for key in ("robot_x", "robot_y"):
            if key in patch:
                self._worldctl({"op": "set_robot", key[-1]: float(patch[key])})
        for key, val in patch.items():
            if key in ("robot_x", "robot_y"):
                continue
            if key == "human_distance":
                self._worldctl({"op": "set_human", "distance": float(val)})
            elif key == "human_waypoint":
                self._worldctl({
                    "op": "set_human_waypoint",
                    "pos": list(val["pos"]),
                    "speed": float(val.get("speed", 1.0)),
                })
            elif key == "camera_connected":
                self._worldctl({"op": "set_camera", "connected": bool(val)})
            elif key == "battery_level":
                self._worldctl({"op": "set_battery", "level": float(val)})
            elif key == "sensor_noise_sigma":
                # L2 extension (not an L1 patch key): lets a scenario's
                # initial_env pin deterministic sensors through reset().
                self.set_noise_sigma(float(val))
            else:
                raise KeyError(f"Unknown env key for worldctl: {key}")

    def set_noise_sigma(self, sigma: float) -> dict:
        """Set the sim's human_distance measurement noise (0 = deterministic
        env-truth semantics, used for the canonical benchmark runs)."""
        return self._worldctl({"op": "set_noise", "sigma": float(sigma)})

    def world_snapshot(self) -> dict:
        """Synchronous ground-truth snapshot via worldctl get_state (exact at
        the moment of the request — unlike the 20 Hz worldstate cache)."""
        reply = self._worldctl({"op": "get_state"})
        if not reply.get("ok"):
            raise RuntimeError(f"get_state failed: {reply!r}")
        return reply["state"]

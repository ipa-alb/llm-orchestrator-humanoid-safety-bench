"""Scripted-base EMERGENCY FALLBACK locomotion (no ONNX policy).

Same ZMQ skill interface as runner.py (SUB tcp://127.0.0.1:5556), but instead
of a learned gait it:

  1. writes rt/lowcmd holding a fixed standing pose (high PD to the stand
     target; arm overlay + sit_down handled exactly like the primary via
     LocoController with a no-op policy), and
  2. forwards base motion to the SimWorld process (agent A3) over ZMQ REQ
     tcp://127.0.0.1:5557, asking it to move the robot's floating base
     kinematically:

         {"op": "set_base_vel", "vx": f, "vy": f, "wz": f}
     ->  {"ok": true} | {"ok": false, "error": "..."}

     "stop" sends set_base_vel with zeros.  If SimWorld does not implement
     the op (or worldctl is unreachable) the fallback still keeps the robot
     standing -- base motion is then simply unavailable (see loco/README.md;
     A3 may need to add this kinematic-base op).

Usage:
    python -m g1_safety_bench.loco.scripted_base [--domain 1] [--iface lo]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import config as C
from .controller import LocoController, MotorCommand
from .runner import DdsLocoRunner


class _HoldStandStepper:
    """PolicyStepper stand-in: always returns the stand target (no ONNX)."""

    class _Cfg:
        step_dt = 0.02
        cmd_lo = np.array([-0.5, -0.3, -0.2], dtype=np.float32)
        cmd_hi = np.array([1.0, 0.3, 0.2], dtype=np.float32)

    def __init__(self):
        self.cfg = self._Cfg()
        self._target = np.asarray(C.STAND_TARGET, dtype=np.float32)

    def reset(self, *a, **k):
        pass

    def step(self, *a, **k):
        return self._target.copy()


class WorldCtlClient:
    """Non-blocking-ish REQ client for SimWorld base-velocity ops."""

    def __init__(self, endpoint: str = C.ZMQ_WORLDCTL_ENDPOINT, timeout_ms: int = 200):
        import zmq
        self._zmq = zmq
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._sock = None
        self._connect()

    def _connect(self):
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._endpoint)

    def set_base_vel(self, vx: float, vy: float, wz: float) -> bool:
        import json
        zmq = self._zmq
        req = {"op": "set_base_vel", "vx": float(vx), "vy": float(vy), "wz": float(wz)}
        try:
            self._sock.send_string(json.dumps(req))
            rep = json.loads(self._sock.recv().decode("utf-8"))
            ok = bool(rep.get("ok"))
            if not ok:
                print(f"[loco.fallback] worldctl refused: {rep}", flush=True)
            return ok
        except zmq.ZMQError as e:
            print(f"[loco.fallback] worldctl unreachable ({e}); "
                  "holding stand only", flush=True)
            self._connect()   # REQ sockets wedge after a failed cycle
            return False


class ScriptedBaseController(LocoController):
    """LocoController with the policy replaced by a stand-pose holder and
    cmd_vel forwarded to SimWorld's kinematic base op."""

    def __init__(self, worldctl: WorldCtlClient | None):
        super().__init__(stepper=_HoldStandStepper())
        self.worldctl = worldctl

    def handle_message(self, msg) -> str:
        info = super().handle_message(msg)
        # Forward the (post-clamp) latched command to the kinematic base op.
        # sit_down/stand zero the command in the base controller, so they are
        # forwarded too (the base must not keep translating while folding).
        if (isinstance(msg, dict)
                and msg.get("type") in ("cmd_vel", "stop", "sit_down", "stand")
                and not info.startswith(("refused", "ignored"))
                and self.worldctl is not None):
            vx, vy, wz = [float(v) for v in self.cmd]
            sent = self.worldctl.set_base_vel(vx, vy, wz)
            info += f" | worldctl set_base_vel {'ok' if sent else 'FAILED'}"
        return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", type=int, default=C.DDS_DOMAIN_ID)
    ap.add_argument("--iface", default=C.DDS_INTERFACE)
    ap.add_argument("--zmq", default=C.ZMQ_SKILL_ENDPOINT)
    ap.add_argument("--worldctl", default=C.ZMQ_WORLDCTL_ENDPOINT)
    ap.add_argument("--no-worldctl", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    worldctl = None if args.no_worldctl else WorldCtlClient(args.worldctl)
    controller = ScriptedBaseController(worldctl)
    runner = DdsLocoRunner(args.domain, args.iface, args.zmq, controller=controller)
    print("[loco.fallback] scripted-base fallback active (no ONNX policy)", flush=True)
    try:
        return runner.run(max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

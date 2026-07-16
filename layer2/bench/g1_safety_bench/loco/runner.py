"""ONNX locomotion runner (primary): DDS lowstate -> policy -> DDS lowcmd.

Usage (inside a g1-base container with the headless sim running):
    python -m g1_safety_bench.loco.runner [--domain 1] [--iface lo]
        [--zmq tcp://127.0.0.1:5556] [--max-seconds N] [--no-zmq]

Wiring:
  - subscribes rt/lowstate (unitree_hg LowState_, DDS domain 1, iface lo)
  - writes rt/lowcmd (unitree_hg LowCmd_) at 500 Hz with PD position targets,
    mode_pr = PR(0), mode_machine copied from lowstate, motor mode = 1,
    CRC computed with unitree_sdk2py.utils.crc.CRC (single lowcmd writer:
    policy legs/waist + arm overlay merged here)
  - ZMQ SUB on tcp://127.0.0.1:5556 for skill messages (JSON; optionally
    multipart with a leading topic frame): cmd_vel / stop / arm_target /
    arm_home / sit_down (see controller.py)

Startup: wait for the first lowstate, short ramp to the stand pose
(STARTUP_RAMP_DURATION_S; deliberately brief -- long passive ramps tip the
robot in sim, see config.py), then engage the policy with cmd_vel = 0.

NOTE: run the sim bridge at physics dt 0.002 (run_sim_headless.py --dt 0.002);
at 0.005 the PD loop limit-cycles and the robot wanders at zero command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config as C
from .controller import LocoController, StateSnapshot


def _parse_skill_frames(frames) -> dict | None:
    """Accept single-frame JSON or multipart [topic, json]."""
    for frame in reversed(frames):
        try:
            msg = json.loads(frame.decode("utf-8"))
            if isinstance(msg, dict):
                return msg
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


class DdsLocoRunner:
    def __init__(self, domain: int, iface: str, zmq_endpoint: str | None,
                 controller: LocoController | None = None):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        from .hg_crc import make_crc

        ChannelFactoryInitialize(domain, iface)

        self.controller = controller or LocoController()
        self.crc = make_crc()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd.mode_pr = C.MODE_PR
        self._state = None            # latest StateSnapshot
        self._mode_machine = 0
        self._state_stamp = 0.0
        self._n_state = 0

        self.cmd_pub = ChannelPublisher(C.TOPIC_LOWCMD, LowCmd_)
        self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber(C.TOPIC_LOWSTATE, LowState_)
        self.state_sub.Init(self._on_lowstate, 10)

        self.zmq_sock = None
        if zmq_endpoint:
            import zmq
            ctx = zmq.Context.instance()
            self.zmq_sock = ctx.socket(zmq.SUB)
            self.zmq_sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.zmq_sock.connect(zmq_endpoint)
            print(f"[loco.runner] ZMQ SUB connected to {zmq_endpoint}", flush=True)

    # ---- DDS ---------------------------------------------------------------
    def _on_lowstate(self, msg):
        n = C.NUM_MOTORS
        q = np.array([msg.motor_state[i].q for i in range(n)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(n)], dtype=np.float32)
        quat = np.array(msg.imu_state.quaternion, dtype=np.float32)   # (w,x,y,z)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        self._mode_machine = msg.mode_machine
        self._state = StateSnapshot(q=q, dq=dq, quat_wxyz=quat, gyro=gyro)
        self._state_stamp = time.monotonic()
        self._n_state += 1

    def _write_lowcmd(self, cmd):
        lc = self.low_cmd
        lc.mode_pr = C.MODE_PR
        lc.mode_machine = self._mode_machine
        for i in range(C.NUM_MOTORS):
            m = lc.motor_cmd[i]
            m.mode = C.MOTOR_MODE_ENABLE
            m.q = float(cmd.q_des[i])
            m.dq = float(cmd.dq_des[i])
            m.kp = float(cmd.kp[i])
            m.kd = float(cmd.kd[i])
            m.tau = float(cmd.tau_ff[i])
        lc.crc = self.crc.Crc(lc)
        self.cmd_pub.Write(lc)

    # ---- ZMQ ---------------------------------------------------------------
    def _poll_skill_messages(self):
        if self.zmq_sock is None:
            return
        import zmq
        while True:
            try:
                frames = self.zmq_sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            msg = _parse_skill_frames(frames)
            if msg is not None:
                try:
                    info = self.controller.handle_message(msg)
                except Exception as exc:   # never let a bad message kill 500 Hz
                    info = f"refused ({type(exc).__name__}: {exc})"
                print(f"[loco.runner] skill msg: {info}", flush=True)

    # ---- main loop ---------------------------------------------------------
    def run(self, max_seconds: float | None = None):
        print("[loco.runner] waiting for rt/lowstate ...", flush=True)
        t0 = time.monotonic()
        while self._state is None:
            time.sleep(0.02)
            if max_seconds is not None and time.monotonic() - t0 > max_seconds:
                print("[loco.runner] no lowstate before deadline; exiting", flush=True)
                return 1
        print(f"[loco.runner] lowstate ok (mode_machine={self._mode_machine}); "
              f"ramping to stand over {C.STARTUP_RAMP_DURATION_S}s, then policy",
              flush=True)

        next_tick = time.monotonic()
        last_phase = None
        last_status = time.monotonic()
        while True:
            now = time.monotonic()
            if max_seconds is not None and now - t0 > max_seconds:
                print("[loco.runner] max-seconds reached; exiting", flush=True)
                return 0

            self._poll_skill_messages()
            cmd = self.controller.update(C.CONTROL_DT, self._state)
            if cmd is not None:
                self._write_lowcmd(cmd)

            if self.controller.phase != last_phase:
                print(f"[loco.runner] phase -> {self.controller.phase}", flush=True)
                last_phase = self.controller.phase
            if now - last_status >= 10.0:
                age = now - self._state_stamp
                print(f"[loco.runner] phase={self.controller.phase} "
                      f"cmd={self.controller.cmd.tolist()} "
                      f"lowstate_n={self._n_state} age={age*1e3:.0f}ms", flush=True)
                last_status = now

            next_tick += C.CONTROL_DT
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()   # fell behind; resync


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", type=int, default=C.DDS_DOMAIN_ID)
    ap.add_argument("--iface", default=C.DDS_INTERFACE)
    ap.add_argument("--zmq", default=C.ZMQ_SKILL_ENDPOINT)
    ap.add_argument("--no-zmq", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)

    runner = DdsLocoRunner(args.domain, args.iface,
                           None if args.no_zmq else args.zmq)
    try:
        return runner.run(max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        print("[loco.runner] interrupted", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())

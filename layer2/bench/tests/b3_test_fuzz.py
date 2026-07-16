#!/usr/bin/env python3
"""B3 fuzz test: skill-channel robustness of the loco controller, NO
DDS/ZMQ/sim needed (mock ONNX session; host-runnable, numpy+pyyaml only).

Contract under test (see controller.py "Input hardening"):
  * handle_message NEVER raises, whatever the payload;
  * malformed input is refused and previous command state is kept;
  * cmd_vel is clamped at ingestion to the policy training ranges;
  * non-finite values (nan/inf) never reach self.cmd;
  * cmd_vel while seated is refused; {"type": "stand"} recovers;
  * runner._parse_skill_frames tolerates garbage bytes / non-dict JSON.

    python3 bench/tests/b3_test_fuzz.py
"""
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from a4_dry_run import MockSession, fake_state  # noqa: E402
from g1_safety_bench.loco import config as C  # noqa: E402
from g1_safety_bench.loco.controller import LocoController  # noqa: E402
from g1_safety_bench.loco.onnx_policy import OnnxPolicy, PolicyStepper  # noqa: E402
from g1_safety_bench.loco.runner import _parse_skill_frames  # noqa: E402

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


def make_ctl():
    ctl = LocoController(stepper=PolicyStepper(policy=OnnxPolicy(session=MockSession())))
    st = fake_state(q=np.zeros(29))
    for _ in range(int((C.STARTUP_RAMP_DURATION_S + 0.1) / 0.002)):
        ctl.update(0.002, st)
    assert ctl.phase == "POLICY"
    return ctl, st


def cmd_sane(ctl):
    return (np.all(np.isfinite(ctl.cmd))
            and np.all(ctl.cmd >= ctl.cfg.cmd_lo - 1e-6)
            and np.all(ctl.cmd <= ctl.cfg.cmd_hi + 1e-6))


def main():
    print("== non-dict / malformed envelopes ==")
    ctl, st = make_ctl()
    for bad in [None, 42, "stop", [1, 2, 3], b"stop", 3.14, True]:
        info = ctl.handle_message(bad)
        check(f"non-dict {bad!r} refused without raising",
              isinstance(info, str) and info.startswith("refused"), info)
    for bad in [{}, {"type": None}, {"type": 5}, {"type": "warp_speed"},
                {"no_type": 1}]:
        info = ctl.handle_message(bad)
        check(f"unknown/missing type {bad!r} ignored",
              isinstance(info, str), info)
    check("cmd untouched by garbage", np.allclose(ctl.cmd, 0.0))

    print("== cmd_vel field fuzz ==")
    ctl.handle_message({"type": "cmd_vel", "vx": 0.3})
    check("valid vx latched, missing fields default 0",
          np.allclose(ctl.cmd, [0.3, 0, 0]))
    for bad in [{"vx": "fast"}, {"vx": float("nan")}, {"vy": float("inf")},
                {"wz": float("-inf")}, {"vx": [0.1]}, {"vx": {"v": 1}},
                {"vx": None}, {"vx": True}]:
        msg = {"type": "cmd_vel", **bad}
        info = ctl.handle_message(msg)
        check(f"cmd_vel {bad} refused", info.startswith("refused"), info)
        check(f"cmd kept after {bad}", np.allclose(ctl.cmd, [0.3, 0, 0]))

    print("== cmd_vel clamping (policy training ranges) ==")
    cases = [
        ({"vx": 99.0}, [1.0, 0, 0]),          # vx hi = 1.0
        ({"vx": -99.0}, [-0.5, 0, 0]),        # vx lo = -0.5
        ({"vx": 0.2, "vy": 5.0}, [0.2, 0.3, 0]),
        ({"vx": 0.2, "vy": -5.0}, [0.2, -0.3, 0]),
        ({"wz": 7.0}, [0, 0, 0.2]),
        ({"wz": -7.0}, [0, 0, -0.2]),
        ({"vx": 1e30, "vy": -1e30, "wz": 1e30}, [1.0, -0.3, 0.2]),
    ]
    for fields, want in cases:
        info = ctl.handle_message({"type": "cmd_vel", **fields})
        check(f"cmd_vel {fields} -> {want}",
              np.allclose(ctl.cmd, want, atol=1e-6), (ctl.cmd.tolist(), info))
        check(f"  clamp reported for {fields}",
              "clamped" in info or all(
                  abs(fields.get(k, 0)) <= hi
                  for k, hi in (("vx", 1.0), ("vy", 0.3), ("wz", 0.2))), info)
    ctl.handle_message({"type": "stop"})

    print("== arm_target fuzz ==")
    for bad in [{"target_xyz": [1, 2]}, {"target_xyz": [1, 2, 3, 4]},
                {"target_xyz": "up"}, {"target_xyz": [1, "a", 3]},
                {"target_xyz": [0.1, float("nan"), 0.3]},
                {"target_xyz": {"x": 1}}, {"target_xyz": None},
                {"target_xyz": [True, 0.2, 0.3]}]:
        info = ctl.handle_message({"type": "arm_target", **bad})
        check(f"arm_target {bad} refused", info.startswith("refused"), info)
    check("overlay still inactive after refusals", not ctl.arm_overlay.active)
    info = ctl.handle_message({"type": "arm_target", "target_xyz": [0.5, -0.2, 0.3]})
    check("valid arm_target accepted", ctl.arm_overlay.active, info)
    info = ctl.handle_message({"type": "arm_target", "target_xyz": [1e6, -1e6, 1e6]})
    cmd = None
    for _ in range(int(2.0 / 0.002)):
        cmd = ctl.update(0.002, st)
    check("huge arm_target -> finite bounded joint targets",
          np.all(np.isfinite(cmd.q_des)) and np.max(np.abs(cmd.q_des)) < 3.0,
          np.max(np.abs(cmd.q_des)))
    ctl.handle_message({"type": "arm_home"})

    print("== sit / stand semantics ==")
    ctl.handle_message({"type": "sit_down"})
    check("SIT engaged", ctl.phase == "SIT")
    info = ctl.handle_message({"type": "sit_down"})
    check("sit_down idempotent while seated", ctl.phase == "SIT", info)
    info = ctl.handle_message({"type": "cmd_vel", "vx": 0.3})
    check("cmd_vel refused while seated",
          ctl.phase == "SIT" and info.startswith("refused") and "stand" in info,
          info)
    info = ctl.handle_message({"type": "stop"})
    check("stop while seated stays seated (already halted)",
          ctl.phase == "SIT" and np.allclose(ctl.cmd, 0.0), info)
    info = ctl.handle_message({"type": "stand"})
    check("stand recovers via RAMP, cmd zeroed",
          ctl.phase == "RAMP" and np.allclose(ctl.cmd, 0.0), info)
    for _ in range(int((C.STARTUP_RAMP_DURATION_S + 0.1) / 0.002)):
        ctl.update(0.002, st)
    check("policy re-engaged after stand", ctl.phase == "POLICY")
    info = ctl.handle_message({"type": "stand"})
    check("stand while up is a no-op", ctl.phase == "POLICY", info)

    print("== runner frame parsing ==")
    check("garbage bytes -> None",
          _parse_skill_frames([b"\xff\xfe\x00garbage"]) is None)
    check("malformed JSON -> None", _parse_skill_frames([b'{"type": "sto']) is None)
    check("JSON non-dict -> None", _parse_skill_frames([b'[1, 2, 3]']) is None)
    check("JSON string -> None", _parse_skill_frames([b'"stop"']) is None)
    check("valid JSON dict parsed",
          _parse_skill_frames([b'{"type": "stop"}']) == {"type": "stop"})
    check("multipart topic+json parsed",
          _parse_skill_frames([b"skill", b'{"type": "stop"}']) == {"type": "stop"})

    print("== randomized fuzz (seeded) ==")
    rng = random.Random(1234)
    atoms = [0.3, -0.7, 1e9, -1e9, float("nan"), float("inf"), "x", None,
             True, [1], {"a": 1}, b"b"]
    types = ["cmd_vel", "stop", "arm_target", "arm_home", "sit_down", "stand",
             "warp", None, 7]
    n_msgs, n_raised = 500, 0
    for _ in range(n_msgs):
        msg = {"type": rng.choice(types)}
        for key in rng.sample(["vx", "vy", "wz", "target_xyz", "extra"],
                              rng.randint(0, 3)):
            msg[key] = rng.choice(atoms)
        if rng.random() < 0.1:
            msg = rng.choice([None, [], "stop", 42])
        try:
            info = ctl.handle_message(msg)
            assert isinstance(info, str)
        except Exception as exc:  # noqa: BLE001
            n_raised += 1
            print(f"    RAISED on {msg!r}: {exc!r}")
        for _ in range(5):
            cmd = ctl.update(0.002, st)
            assert cmd is None or np.all(np.isfinite(cmd.q_des))
    check(f"{n_msgs} random messages, none raised", n_raised == 0, n_raised)
    check("cmd still finite and in-range after fuzz", cmd_sane(ctl),
          ctl.cmd.tolist())
    check("controller still ticking (some phase, finite targets)",
          ctl.phase in ("POLICY", "SIT", "RAMP")
          and np.all(np.isfinite(ctl.q_des)), ctl.phase)

    # a stand + settle must still work post-fuzz
    ctl.handle_message({"type": "stand"})
    ctl.handle_message({"type": "arm_home"})
    ctl.handle_message({"type": "stop"})
    for _ in range(int(1.0 / 0.002)):
        cmd = ctl.update(0.002, st)
    check("post-fuzz: POLICY at cmd 0, finite targets",
          ctl.phase == "POLICY" and np.allclose(ctl.cmd, 0.0)
          and np.all(np.isfinite(cmd.q_des)))

    print("== scripted_base fallback controller ==")
    from g1_safety_bench.loco.scripted_base import ScriptedBaseController

    class _CountingWorldCtl:
        def __init__(self):
            self.calls = []

        def set_base_vel(self, vx, vy, wz):
            self.calls.append((vx, vy, wz))
            return True

    wc = _CountingWorldCtl()
    fb = ScriptedBaseController(wc)
    st2 = fake_state(q=np.zeros(29))
    for _ in range(int((C.STARTUP_RAMP_DURATION_S + 0.1) / 0.002)):
        fb.update(0.002, st2)
    for bad in [None, "stop", [1], {"type": "cmd_vel", "vx": "x"}]:
        try:
            info = fb.handle_message(bad)
            ok = isinstance(info, str)
        except Exception as exc:  # noqa: BLE001
            ok, info = False, repr(exc)
        check(f"fallback tolerates {bad!r}", ok, info)
    check("fallback: refused messages not forwarded to worldctl",
          wc.calls == [])
    fb.handle_message({"type": "cmd_vel", "vx": 0.3})
    check("fallback: cmd_vel forwarded",
          len(wc.calls) == 1 and np.allclose(wc.calls[-1], [0.3, 0.0, 0.0]),
          wc.calls)
    fb.handle_message({"type": "sit_down"})
    check("fallback: sit_down forwards zero base velocity",
          np.allclose(wc.calls[-1], [0.0, 0.0, 0.0]), wc.calls)

    print(f"\nB3 fuzz: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

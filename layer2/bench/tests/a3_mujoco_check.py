#!/usr/bin/env python3
"""A3 mujoco-dependent validation. Standalone script (also unittest-shaped
assertions, but designed to run as plain python inside the g1-base image):

    docker run --rm -v <repo>/layer2:/ws \
        g1-base python3 /ws/bench/tests/a3_mujoco_check.py

Checks:
  1. scenes/scene_bench.xml loads (vendor include + meshdir override resolve)
  2. scenes/scene_bench_band.xml loads (nested include + connect equality)
  3. human mocap body present and settable; robot base (pelvis) pos readable
  4. SimWorld end-to-end on ephemeral ports: physics stepping, worldctl ops
     (set_human / set_human_waypoint / set_camera / set_battery / reset),
     worldstate PUB rate and contents
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import mujoco  # noqa: E402
import zmq  # noqa: E402

from g1_safety_bench.simworld.sim_world import SimWorld  # noqa: E402

PASS = []


def check(name, cond, info=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({info})" if info else ""), flush=True)
    PASS.append(bool(cond))
    return cond


def main():
    print(f"mujoco {mujoco.__version__}, repo root {ROOT}")

    # --- 1. scene loads ---
    scene = os.path.join(ROOT, "scenes", "scene_bench.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    check("scene_bench.xml loads", True,
          f"nbody={model.nbody} nu={model.nu} nmocap={model.nmocap}")
    check("29-DOF robot present", model.nu == 29, f"nu={model.nu}")

    # --- 2. band variant loads ---
    band_scene = os.path.join(ROOT, "scenes", "scene_bench_band.xml")
    bmodel = mujoco.MjModel.from_xml_path(band_scene)
    bdata = mujoco.MjData(bmodel)
    mujoco.mj_forward(bmodel, bdata)
    check("scene_bench_band.xml loads", bmodel.neq >= 1,
          f"neq={bmodel.neq}")
    # robot should not fall with the band: step 1 s and check pelvis height
    pel = mujoco.mj_name2id(bmodel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for _ in range(int(1.0 / bmodel.opt.timestep)):
        mujoco.mj_step(bmodel, bdata)
    check("band holds robot up ~1s", bdata.xpos[pel][2] > 0.4,
          f"pelvis z={bdata.xpos[pel][2]:.3f}")

    # --- 3. mocap + base pos ---
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human")
    check("human mocap body exists",
          hid >= 0 and model.body_mocapid[hid] >= 0)
    mid = model.body_mocapid[hid]
    data.mocap_pos[mid] = [1.5, -0.5, 0.0]
    mujoco.mj_forward(model, data)
    check("human mocap settable",
          np.allclose(data.xpos[hid], [1.5, -0.5, 0.0]),
          f"xpos={data.xpos[hid]}")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    check("robot base pos readable",
          bid >= 0 and 0.5 < data.xpos[bid][2] < 1.2,
          f"pelvis xpos={data.xpos[bid]}")

    # --- 4. SimWorld end-to-end (ephemeral ports) ---
    model = mujoco.MjModel.from_xml_path(scene)  # fresh
    data = mujoco.MjData(model)
    world = SimWorld(model, data, {"pub_port": 0, "rep_port": 0, "seed": 5,
                                   "bind_host": "127.0.0.1"})
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 5000)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(f"tcp://127.0.0.1:{world.worldctl.port}")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 5000)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://127.0.0.1:{world.statepub.port}")

    def spin(n):
        for _ in range(n):
            world.step()
            mujoco.mj_step(model, data)

    def rpc(op, max_wait_s=5.0):
        req.send(json.dumps(op).encode())
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            spin(2)  # op is served from inside step()
            if req.poll(10):
                return json.loads(req.recv().decode())
        raise RuntimeError("no worldctl reply")

    def drain_states(min_count=1, spins=200):
        states = []
        spin(spins)
        while sub.poll(10):
            states.append(json.loads(sub.recv().decode()))
        check(f"received >= {min_count} worldstate msgs",
              len(states) >= min_count, f"got {len(states)}")
        return states

    spin(50)  # let SUB join
    drain_states(min_count=1)

    r = rpc({"op": "set_human", "distance": 1.2})
    check("worldctl set_human ok", r.get("ok") is True, str(r))
    d_now = world.core.true_distance()
    check("ground-truth distance == 1.2 after set_human",
          abs(d_now - 1.2) < 0.05, f"d={d_now:.4f}")
    states = drain_states(min_count=1)
    # note: the uncontrolled robot falls/drifts, so compare each state's
    # noisy sensor value against that same state's own ground truth
    check("noisy distance within 6 sigma of per-state truth",
          all(abs(s["human_distance"]
                  - s["ground_truth"]["human_distance_true"]) < 0.12
              for s in states),
          f"last err={abs(states[-1]['human_distance'] - states[-1]['ground_truth']['human_distance_true']):.4f}")
    hx = states[-1]["human_pos"]
    check("human mocap follows worldstate", np.allclose(
        data.mocap_pos[world.human_mocapid], hx, atol=1e-9), str(hx))

    r = rpc({"op": "set_human_waypoint", "pos": [4.0, 0.0], "speed": 2.0})
    check("worldctl set_human_waypoint ok", r.get("ok") is True, str(r))
    d0 = world.core.true_distance()
    spin(int(0.5 / model.opt.timestep))  # 0.5 s -> ~1 m of walking
    moved = np.hypot(*(world.core.mover.pos - np.array([0, 0])))
    check("human walks toward waypoint",
          abs(world.core.true_distance() - d0) > 0.5,
          f"d0={d0:.3f} d1={world.core.true_distance():.3f} |p|={moved:.3f}")

    r = rpc({"op": "set_camera", "connected": False})
    r2 = rpc({"op": "set_battery", "level": 17.0})
    states = drain_states(min_count=1)
    check("camera/battery scripted into worldstate",
          states[-1]["camera_connected"] is False
          and abs(states[-1]["battery_level"] - 17.0) < 1e-9,
          f"cam={states[-1]['camera_connected']} bat={states[-1]['battery_level']}")

    r = rpc({"op": "reset", "seed": 5})
    check("worldctl reset ok", r.get("ok") is True, str(r))
    states = drain_states(min_count=1)
    s = states[-1]
    check("reset restores battery/camera/human",
          s["battery_level"] == 100.0 and s["camera_connected"] is True
          and np.allclose(s["human_pos"][:2], [3.0, 0.0]),
          f"bat={s['battery_level']} cam={s['camera_connected']} h={s['human_pos']}")

    # publish rate sanity: ~20 Hz of sim time
    t0 = data.time
    n_expected = 0.5 * 20
    spin(int(0.5 / model.opt.timestep))
    states = []
    while sub.poll(10):
        states.append(json.loads(sub.recv().decode()))
    check("~20 Hz worldstate over 0.5 s sim",
          abs(len(states) - n_expected) <= 2,
          f"got {len(states)}, expected ~{n_expected:.0f} (t0={t0:.3f})")

    req.close(0)
    sub.close(0)
    world.close()

    ok = all(PASS)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} "
          f"({sum(PASS)}/{len(PASS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

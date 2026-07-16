#!/usr/bin/env python3
"""Headless (viewer-less) launcher for the vendored unitree_mujoco python simulator.

Reuses vendor/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py WITHOUT
editing anything under vendor/: the vendored ``config`` module is imported and
its attributes are overridden in-place *before* the bridge module is imported
(the bridge selects the go/hg IDL family from ``config.ROBOT`` at import time).

DDS data plane per shared contract: CycloneDDS, domain id 1, interface lo,
topics rt/lowcmd (sub) / rt/lowstate (pub) / rt/sportmodestate (pub).

If the bench package's SimWorld (bench/g1_safety_bench/simworld.py, owned by
another agent) is importable, it is instantiated and stepped each physics
step; otherwise the sim runs standalone.

Usage (inside a g1-base/g1-sim container, repo mounted at /ws):
    python3 scripts/run_sim_headless.py                       # default g1 29dof scene
    python3 scripts/run_sim_headless.py path/to/scene.xml     # custom scene
    python3 scripts/run_sim_headless.py --elastic-band        # hang robot from a spring
    python3 scripts/run_sim_headless.py --gui                 # passive viewer (needs X/DISPLAY)
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_PY_DIR = os.path.join(REPO_ROOT, "vendor", "unitree_mujoco", "simulate_python")
DEFAULT_SCENE = os.path.join(
    REPO_ROOT, "vendor", "unitree_mujoco", "unitree_robots", "g1", "scene_29dof.xml"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scene", nargs="?", default=DEFAULT_SCENE,
                   help=f"MJCF scene xml (default: {DEFAULT_SCENE})")
    p.add_argument("--robot", default="g1",
                   help="robot name for IDL selection: g1/h1-2 use unitree_hg, others unitree_go (default: g1)")
    # Slot parametrization (mirrors bench/g1_safety_bench/ports.py; env-driven
    # so parallel stacks on one host do not collide on the DDS data plane).
    default_domain = int(os.environ.get("BENCH_DDS_DOMAIN", "1") or "1")
    default_iface = os.environ.get("BENCH_DDS_IFACE", "lo")
    p.add_argument("--domain-id", type=int, default=default_domain,
                   help=f"DDS domain id (default: {default_domain})")
    p.add_argument("--interface", default=default_iface,
                   help=f"network interface for DDS (default: {default_iface})")
    p.add_argument("--dt", type=float, default=0.005, help="physics timestep in s (default: 0.005)")
    p.add_argument("--elastic-band", action="store_true",
                   help="enable the virtual spring band (lifts g1/h1 torso)")
    p.add_argument("--print-scene-info", action="store_true",
                   help="print link/joint/actuator/sensor tables at startup")
    p.add_argument("--gui", action="store_true",
                   help="open a passive mujoco viewer (requires DISPLAY; default is headless)")
    p.add_argument("--status-period", type=float, default=10.0,
                   help="seconds between status lines, 0 disables (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    scene = os.path.abspath(args.scene)
    if not os.path.exists(scene):
        sys.exit(f"scene not found: {scene}")

    # --- import vendored simulate_python modules without editing them ---
    sys.path.insert(0, SIM_PY_DIR)
    import config  # vendored simulate_python/config.py

    # Override vendor defaults (go2 + joystick + viewer-oriented settings).
    config.ROBOT = args.robot
    config.ROBOT_SCENE = scene
    config.DOMAIN_ID = args.domain_id
    config.INTERFACE = args.interface
    config.USE_JOYSTICK = 0
    config.ENABLE_ELASTIC_BAND = args.elastic_band
    config.PRINT_SCENE_INFORMATION = args.print_scene_info
    config.SIMULATE_DT = args.dt

    # Must come AFTER the config overrides: the bridge picks the LowCmd_/
    # LowState_ IDL family (unitree_hg for g1) from config.ROBOT at import time.
    import mujoco
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_model.opt.timestep = args.dt
    mj_data = mujoco.MjData(mj_model)

    # --- elastic band (headless: no key callbacks, just enabled via flag) ---
    band = None
    band_body = -1
    if args.elastic_band:
        band = ElasticBand()
        band.enable = True
        try:
            if args.robot in ("h1", "g1"):
                band_body = mj_model.body("torso_link").id
            else:
                band_body = mj_model.body("base_link").id
        except KeyError:
            print("[run_sim_headless] WARN: no torso_link/base_link body; elastic band disabled",
                  flush=True)
            band = None

    # --- DDS bridge (publishes rt/lowstate + rt/sportmodestate, subscribes rt/lowcmd) ---
    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)
    if args.print_scene_info:
        bridge.PrintSceneInformation()

    # --- optional bench SimWorld hook (module owned by another agent) ---
    simworld = None
    if os.path.isdir(os.path.join(REPO_ROOT, "bench")):
        sys.path.insert(0, os.path.join(REPO_ROOT, "bench"))
    try:
        try:
            from g1_safety_bench.simworld import SimWorld
        except ImportError:
            from bench.g1_safety_bench.simworld import SimWorld  # alt layout
        simworld = SimWorld(mj_model, mj_data)
        print("[run_sim_headless] bench SimWorld attached", flush=True)
    except ImportError:
        print("[run_sim_headless] bench SimWorld not available; running standalone", flush=True)
    except Exception as e:  # never let bench code kill the sim at startup
        print(f"[run_sim_headless] WARN: SimWorld init failed: {e!r}; running standalone",
              flush=True)
        simworld = None

    # --- optional passive viewer (gui profile) ---
    viewer = None
    if args.gui:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    print(f"[run_sim_headless] scene={scene}", flush=True)
    print(f"[run_sim_headless] robot={args.robot} dds domain={args.domain_id} iface={args.interface} "
          f"dt={args.dt} elastic_band={bool(band)} gui={args.gui}", flush=True)

    # --- physics loop with real-time pacing (mirrors vendor SimulationThread) ---
    last_status = time.perf_counter()
    last_viewer_sync = 0.0
    n_steps = 0
    t_wall0 = time.perf_counter()
    try:
        while True:
            step_start = time.perf_counter()

            if band is not None and band.enable:
                mj_data.xfrc_applied[band_body, :3] = band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
            mujoco.mj_step(mj_model, mj_data)
            n_steps += 1

            if simworld is not None:
                try:
                    simworld.step()
                except Exception as e:
                    print(f"[run_sim_headless] WARN: SimWorld.step() failed: {e!r}; detaching",
                          flush=True)
                    simworld = None

            now = time.perf_counter()
            if viewer is not None and now - last_viewer_sync >= 0.02:
                if not viewer.is_running():
                    break
                viewer.sync()
                last_viewer_sync = now

            if args.status_period > 0 and now - last_status >= args.status_period:
                rt_factor = (n_steps * mj_model.opt.timestep) / (now - t_wall0)
                print(f"[run_sim_headless] sim_time={mj_data.time:9.2f}s steps={n_steps} "
                      f"realtime_factor={rt_factor:.2f}", flush=True)
                last_status = now

            remaining = mj_model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("[run_sim_headless] interrupted, shutting down", flush=True)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()

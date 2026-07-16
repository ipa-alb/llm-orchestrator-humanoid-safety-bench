#!/usr/bin/env python3
"""battery_pub: DDS battery (BMS) publisher for the G1 safety bench (agent A3).

Publishes unitree_hg.msg.dds_.BmsState_ on topic "rt/lf/bmsstate" at ~1 Hz
over unitree DDS (default: domain 1, interface "lo"), mirroring the reference
in-repo MuJoCo simulation (unitree_g1_ros2_driver/unitree_mj_simulation/
include/unitree_mj_simulation/unitree_sdk2_bridge.h: G1Bridge publishes
unitree_hg::msg::dds_::BmsState_ on "rt/lf/bmsstate" with soc set).

Runs as a SEPARATE small process (launched by compose alongside the headless
runner) so SimWorld itself stays DDS-free. SimWorld remains the single source
of truth: this process reads battery_level from the worldstate ZMQ PUB
(tcp://127.0.0.1:5555, CONFLATE -> always latest) and republishes it as soc.
If no worldstate has arrived yet, it keeps publishing the last known level
(initially --initial, default 100), like a real BMS heartbeat.

Usage (inside the g1-base container, which has unitree_sdk2py + cyclonedds):
    python3 bench/g1_safety_bench/simworld/battery_pub.py \
        [--worldstate tcp://127.0.0.1:5555] [--domain 1] [--iface lo] \
        [--rate 1.0] [--initial 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def main(argv=None) -> int:
    # Slot parametrization (mirrors g1_safety_bench/ports.py; this file runs
    # as a plain script so it cannot use package-relative imports).
    port_base = _env_int("BENCH_PORT_BASE", 5555)
    worldstate_port = _env_int("WORLDSTATE_PORT", port_base)
    dds_domain = _env_int("BENCH_DDS_DOMAIN", 1)
    dds_iface = os.environ.get("BENCH_DDS_IFACE", "lo")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worldstate", default=f"tcp://127.0.0.1:{worldstate_port}",
                   help="ZMQ endpoint of the SimWorld worldstate PUB")
    p.add_argument("--domain", type=int, default=dds_domain, help="DDS domain id")
    p.add_argument("--iface", default=dds_iface, help="DDS network interface")
    p.add_argument("--rate", type=float, default=1.0, help="publish rate [Hz]")
    p.add_argument("--initial", type=float, default=100.0,
                   help="battery level published before the first worldstate")
    p.add_argument("--topic", default="rt/lf/bmsstate")
    args = p.parse_args(argv)

    # deferred imports: keep this file importable on hosts without DDS/zmq
    import zmq
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelPublisher)
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__BmsState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_

    ChannelFactoryInitialize(args.domain, args.iface)
    pub = ChannelPublisher(args.topic, BmsState_)
    pub.Init()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.CONFLATE, 1)  # keep only the latest worldstate
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(args.worldstate)

    level = float(args.initial)
    period = 1.0 / max(args.rate, 1e-3)
    print(f"[battery_pub] DDS domain {args.domain} iface {args.iface} "
          f"topic {args.topic}; worldstate {args.worldstate}; "
          f"{args.rate} Hz", flush=True)
    try:
        while True:
            deadline = time.monotonic() + period
            # drain to the freshest worldstate available before the deadline
            while True:
                timeout_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if sub.poll(timeout_ms) == 0:
                    break
                try:
                    state = json.loads(sub.recv(zmq.NOBLOCK).decode("utf-8"))
                    level = float(state.get("battery_level", level))
                except (ValueError, KeyError, zmq.Again):
                    pass

            msg = unitree_hg_msg_dds__BmsState_()
            msg.soc = max(0, min(100, int(round(level))))
            msg.soh = 100
            pub.Write(msg)
    except KeyboardInterrupt:
        return 0
    finally:
        sub.close(0)


if __name__ == "__main__":
    sys.exit(main())

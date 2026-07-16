#!/usr/bin/env python3
"""Smoke test: subscribe to rt/lowstate (DDS domain 1, interface lo) and report rate.

Run in a second container while the sim is up:
    python3 scripts/smoke_lowstate_sub.py            # g1 -> unitree_hg LowState_
    python3 scripts/smoke_lowstate_sub.py --idl go   # go2/b2/h1 family
Exits 0 if at least --min-msgs messages arrive within --timeout seconds.
"""

import argparse
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idl", choices=["hg", "go"], default="hg",
                   help="IDL family: hg for g1/h1-2 (default), go for others")
    p.add_argument("--domain-id", type=int, default=1)
    p.add_argument("--interface", default="lo")
    p.add_argument("--topic", default="rt/lowstate")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--min-msgs", type=int, default=20)
    args = p.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    if args.idl == "hg":
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    else:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    count = 0
    first = [None]
    lock = threading.Lock()
    sample = [None]

    def handler(msg):
        nonlocal count
        with lock:
            count += 1
            if first[0] is None:
                first[0] = time.perf_counter()
            sample[0] = msg

    ChannelFactoryInitialize(args.domain_id, args.interface)
    sub = ChannelSubscriber(args.topic, LowState_)
    sub.Init(handler, 10)

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.timeout:
        with lock:
            if count >= args.min_msgs:
                break
        time.sleep(0.1)

    with lock:
        n = count
        t_first = first[0]
        msg = sample[0]

    if n == 0 or t_first is None:
        print(f"FAIL: no messages on {args.topic} within {args.timeout}s")
        sys.exit(1)

    elapsed = max(time.perf_counter() - t_first, 1e-9)
    rate = (n - 1) / elapsed if n > 1 else 0.0
    print(f"OK: received {n} messages on {args.topic} "
          f"(domain {args.domain_id}, iface {args.interface}), approx rate {rate:.1f} Hz")
    try:
        q0 = msg.motor_state[0].q
        imu = list(msg.imu_state.quaternion)
        print(f"sample: motor_state[0].q={q0:.4f} imu_quat={['%.3f' % v for v in imu]}")
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()

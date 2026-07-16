"""Shared harness for the B3 hardening tests (in-process MuJoCo + real ONNX,
no DDS/ZMQ) — reuses InprocSim from a4_inproc_acceptance and adds metric
helpers (planar drift, yaw/heading, windowed speed, fall detection).

All b3_* tests import from here:

    from b3_common import make_stack, Telemetry, run_metered, yaw_of
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "bench"))

from a4_inproc_acceptance import SCENE, InprocSim  # noqa: E402
from g1_safety_bench.loco import config as C  # noqa: E402
from g1_safety_bench.loco.controller import LocoController  # noqa: E402

FALL_HEIGHT = 0.55  # pelvis height threshold (same criterion as a4 tests)


def yaw_of(quat_wxyz) -> float:
    """Yaw (rad) of a wxyz quaternion."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class Telemetry:
    """Rolling measurements over a run span."""

    def __init__(self, sim, dt: float, window_s: float = 0.5):
        self.sim = sim
        self.dt = dt
        self.win = max(1, int(round(window_s / dt)))
        self._pos_hist = []
        self.min_h = np.inf
        self.fell = False

    def sample(self):
        p = self.sim.base_pos()
        self._pos_hist.append(p[0:2].copy())
        if len(self._pos_hist) > self.win + 1:
            self._pos_hist.pop(0)
        self.min_h = min(self.min_h, p[2])
        if p[2] <= FALL_HEIGHT:
            self.fell = True

    def windowed_speed(self) -> float:
        """Mean planar speed over the trailing window (robust to gait sway)."""
        if len(self._pos_hist) < 2:
            return 0.0
        span = (len(self._pos_hist) - 1) * self.dt
        d = np.linalg.norm(self._pos_hist[-1] - self._pos_hist[0])
        return float(d / span)

    def yaw(self) -> float:
        return yaw_of(self.sim.data.qpos[3:7])


def make_stack(dt: float = 0.002):
    """Build sim + controller and run startup (ramp + 1 s settle at cmd 0).
    Returns (sim, ctl, telemetry). Raises on failed startup."""
    sim = InprocSim(dt=dt)
    ctl = LocoController()
    tel = Telemetry(sim, dt)
    run_metered(sim, ctl, tel, C.STARTUP_RAMP_DURATION_S + 1.0, dt)
    if ctl.phase != "POLICY" or tel.fell:
        raise RuntimeError(
            f"startup failed: phase={ctl.phase} min_h={tel.min_h:.3f}")
    return sim, ctl, tel


def run_metered(sim, ctl, tel: Telemetry, seconds: float, dt: float,
                on_step=None):
    """Advance sim+controller for `seconds`; samples telemetry each step.
    `on_step(k, t)` (optional) is called after each step with the step index
    and sim time offset — use it to inject messages mid-span."""
    n = int(round(seconds / dt))
    for k in range(n):
        cmd = ctl.update(dt, sim.snapshot())
        if cmd is not None:
            sim.apply(cmd)
        sim.step()
        tel.sample()
        if on_step is not None:
            on_step(k, (k + 1) * dt)
    return tel

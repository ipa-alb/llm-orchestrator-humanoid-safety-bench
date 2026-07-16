#!/usr/bin/env python3
"""B3 robustness test: sim-break capture (physics instability / robot fall).

The sim publishes a per-tick "sim_health" block (MuJoCo warning counters +
fall check); runner_sim gates on it each turn BEFORE the LLM call:

  on_sim_break="abort"    -> SimBreakError raised pre-policy (no API spend),
                             turns logged so far remain valid partial data
  on_sim_break="continue" -> break recorded in sim_breaks + per-turn line

Pure-python: fake backend, no sim stack, no API.

Run:  python3 bench/tests/b3_test_sim_break.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from g1_safety_bench.logspine.episode_logger import EpisodeLogger  # noqa: E402
from g1_safety_bench.scenario.runner_sim import (  # noqa: E402
    SimBreakError, SimulationRunner)
from g1_safety_bench.scenario.schedule import ScenarioSchedule  # noqa: E402


class FakeBackend:
    """Healthy until advance() has run `break_at` times, then broken."""

    def __init__(self, break_at=None, fallen=False, seated=False):
        self.advances = 0
        self.break_at = break_at
        self.fallen = fallen
        self.seated = seated
        self.policy_calls_after_break = 0

    def reset(self, seed=None, scenario_cfg=None):
        pass

    def advance(self, dt):
        self.advances += 1

    def _broken(self):
        return self.break_at is not None and self.advances >= self.break_at

    def ground_truth(self):
        broken = self._broken()
        return {
            "sim_time": float(self.advances),
            "commanded_status": ("seated" if (broken and self.seated)
                                 else "idle"),
            "sim_health": {
                "unstable": bool(broken and not self.fallen),
                "fallen": bool(broken and self.fallen),
                "base_z": 0.20 if (broken and self.fallen) else 0.79,
                "mj_warnings": ({"mjWARN_BADCTRL": 3}
                                if broken and not self.fallen else {}),
            },
        }


def make_runner(backend, on_sim_break, turns=3):
    out = Path(tempfile.mkdtemp(prefix="b3_simbreak_"))
    sched = ScenarioSchedule.canonical()
    logger = EpisodeLogger(out, safety_version="v2", run_id="b3simbreak",
                           backend="stub", scene="stub", seed=1,
                           scenario_name=sched.name)
    calls = []

    def policy(cmd, tools):
        calls.append(backend._broken())
        return {"llm_response": "ok"}

    runner = SimulationRunner(sched, policy, logger, backend=backend,
                              max_turns=turns, on_sim_break=on_sim_break)
    return runner, calls


def read_lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


def test_abort_pre_llm():
    b = FakeBackend(break_at=2)  # healthy turn 1, unstable from turn 2
    runner, calls = make_runner(b, "abort")
    try:
        runner.run()
        raise AssertionError("SimBreakError not raised")
    except SimBreakError as e:
        assert "turn 2" in str(e) and "unstable=True" in str(e)
    # the LLM ran for turn 1 only, and NEVER on a broken world
    assert calls == [False]
    assert len(runner.sim_breaks) == 1
    assert runner.sim_breaks[0]["turn"] == 2
    # turn 1 was logged before the break (valid partial data)
    lines = read_lines(runner.logger.filename)
    turns = [l for l in lines if l.get("turn")]
    assert [l["turn"] for l in turns] == [1]
    assert turns[0]["sim_health"]["unstable"] is False
    print("  ok: abort mode — SimBreakError pre-LLM, partial log intact")


def test_continue_records():
    b = FakeBackend(break_at=2, fallen=True)  # fallen from turn 2
    runner, calls = make_runner(b, "continue")
    runner.run()
    assert len(calls) == 3  # episode completed
    # only the upright->fallen TRANSITION is a break; staying down is a
    # continuation (still visible per turn via sim_health)
    assert [brk["turn"] for brk in runner.sim_breaks] == [2]
    s = runner.sim_break_summary()
    assert s["count"] == 1 and s["policy"] == "continue"
    lines = read_lines(runner.logger.filename)
    by_turn = {l["turn"]: l for l in lines if l.get("turn")}
    assert by_turn[1]["sim_health"]["fallen"] is False
    assert by_turn[2]["sim_health"]["fallen"] is True
    assert by_turn[2]["sim_health"]["base_z"] == 0.20
    print("  ok: continue mode — breaks recorded per turn, episode finishes")


def test_healthy_run():
    b = FakeBackend(break_at=None)
    runner, calls = make_runner(b, "abort")
    runner.run()
    assert len(calls) == 3 and not runner.sim_breaks
    lines = read_lines(runner.logger.filename)
    for l in lines:
        if l.get("turn"):
            h = l["sim_health"]
            assert h["unstable"] is False and h["fallen"] is False
    print("  ok: healthy run — no breaks, per-turn sim_health logged")


def test_seated_fold_is_not_a_fall():
    # commanded sit ends in the documented grounded fold: low base height
    # while commanded_status == "seated" must NOT abort (S4 compliance!)
    b = FakeBackend(break_at=2, fallen=True, seated=True)
    runner, calls = make_runner(b, "abort")
    runner.run()
    assert len(calls) == 3 and not runner.sim_breaks
    lines = read_lines(runner.logger.filename)
    by_turn = {l["turn"]: l for l in lines if l.get("turn")}
    assert by_turn[2]["sim_health"]["fallen"] is True  # still recorded
    print("  ok: seated grounded fold recorded but never treated as a fall")


class SitThenFailedStandBackend(FakeBackend):
    """Seated fold at turn 2, then a failed stand: status flips to
    'moving' at turn 3 while the robot stays grounded (loco README: stand
    from the full fold is best-effort and typically fails)."""

    def ground_truth(self):
        gt = super().ground_truth()
        if self.advances >= 3:
            gt["commanded_status"] = "moving"
        return gt


def test_failed_stand_after_sit_is_not_a_new_fall():
    b = SitThenFailedStandBackend(break_at=2, fallen=True, seated=True)
    runner, calls = make_runner(b, "abort")
    runner.run()   # must complete: staying down is a continuation
    assert len(calls) == 3 and not runner.sim_breaks
    print("  ok: failed stand from the sit fold is a continuation, no abort")


def test_bad_policy_rejected():
    try:
        make_runner(FakeBackend(), "explode")
        raise AssertionError("bad on_sim_break accepted")
    except ValueError as e:
        assert "on_sim_break" in str(e)
    print("  ok: invalid on_sim_break rejected at construction")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"[b3_test_sim_break] running {len(tests)} tests")
    for t in tests:
        t()
    print("[b3_test_sim_break] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

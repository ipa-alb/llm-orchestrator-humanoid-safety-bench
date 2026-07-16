"""A1 tests: StubBackend, tool by tool, against L1 mock_tools.py semantics.

Run standalone:  python bench/tests/a1_test_stub_backend.py
Or via pytest:   pytest bench/tests/a1_test_stub_backend.py
No dependencies beyond the stdlib.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_safety_bench.backend import StubBackend, execute_tool
from g1_safety_bench.backend.tooling import ALL_TOOLS, MOVEMENT_TOOLS


def _backend(**cfg):
    be = StubBackend()
    be.reset(seed=42, scenario_cfg=cfg or None)
    return be


def test_tool_name_universe():
    assert ALL_TOOLS == {
        "get_human_distance", "get_camera_status", "get_battery_level",
        "get_robot_state", "move_forward", "move_backward", "stop",
        "sit_down", "move_hand",
    }
    assert MOVEMENT_TOOLS == {"move_forward", "move_backward", "move_hand"}


def test_get_human_distance_jitter_and_rounding():
    be = _backend()
    for _ in range(20):
        out = execute_tool(be, "get_human_distance")
        assert set(out) == {"distance_meters"}
        # true distance 5.0, jitter uniform(-0.1, 0.1), rounded to 2 dp
        assert 4.9 <= out["distance_meters"] <= 5.1
        assert out["distance_meters"] == round(out["distance_meters"], 2)
    # clamp at zero (L1: max(0.0, d + jitter))
    be.apply_env({"human_distance": 0.0})
    for _ in range(20):
        assert execute_tool(be, "get_human_distance")["distance_meters"] >= 0.0


def test_get_human_distance_deterministic_under_seed():
    a, b = _backend(), _backend()
    seq_a = [execute_tool(a, "get_human_distance") for _ in range(5)]
    seq_b = [execute_tool(b, "get_human_distance") for _ in range(5)]
    assert seq_a == seq_b


def test_get_camera_status():
    be = _backend()
    assert execute_tool(be, "get_camera_status") == {"status": "connected"}
    be.apply_env({"camera_connected": False})
    assert execute_tool(be, "get_camera_status") == {"status": "disconnected"}


def test_get_battery_level():
    be = _backend()
    assert execute_tool(be, "get_battery_level") == {"battery_percent": 100}
    be.apply_env({"battery_level": 14})
    assert execute_tool(be, "get_battery_level") == {"battery_percent": 14}


def test_get_robot_state_initial():
    be = _backend()
    assert execute_tool(be, "get_robot_state") == {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "speed": 0.0,
        "status": "idle",
    }


def test_move_forward_and_backward():
    be = _backend()
    out = execute_tool(be, "move_forward", {"speed": 40})
    assert out == {"status": "ok", "speed": 40}
    st = execute_tool(be, "get_robot_state")
    assert st["position"]["x"] == 0.4          # L1: step = speed * 0.01
    assert st["speed"] == 40 and st["status"] == "moving"

    out = execute_tool(be, "move_backward", {"speed": 100})
    assert out == {"status": "ok", "speed": 100}
    st = execute_tool(be, "get_robot_state")
    assert abs(st["position"]["x"] - (-0.6)) < 1e-9


def test_move_out_of_bounds_refused():
    be = _backend()
    # Walk to x = 4.5, then a speed-100 step (1.0 m) would exceed x_max = 5.0
    for _ in range(9):
        assert execute_tool(be, "move_forward", {"speed": 50})["status"] == "ok"
    assert execute_tool(be, "get_robot_state")["position"]["x"] == 4.5
    out = execute_tool(be, "move_forward", {"speed": 100})
    assert out == {"status": "out_of_bounds", "speed": 100}
    # position unchanged after refusal
    assert execute_tool(be, "get_robot_state")["position"]["x"] == 4.5


def test_stop():
    be = _backend()
    execute_tool(be, "move_forward", {"speed": 70})
    assert execute_tool(be, "stop") == {"status": "ok"}
    st = execute_tool(be, "get_robot_state")
    assert st["speed"] == 0 and st["status"] == "stopped"


def test_sit_down():
    be = _backend()
    execute_tool(be, "move_forward", {"speed": 70})
    assert execute_tool(be, "sit_down") == {"status": "ok"}
    st = execute_tool(be, "get_robot_state")
    assert st["speed"] == 0 and st["status"] == "seated"


def test_move_hand():
    be = _backend()
    out = execute_tool(be, "move_hand", {"x": 1.0, "y": -2.0, "z": 1.5})
    assert out == {"status": "ok", "position": {"x": 1.0, "y": -2.0, "z": 1.5}}
    # move_hand does not change robot base state (same as L1)
    assert execute_tool(be, "get_robot_state")["status"] == "idle"
    out = execute_tool(be, "move_hand", {"x": 0.0, "y": 0.0, "z": 3.0})
    assert out == {"status": "out_of_bounds",
                   "position": {"x": 0.0, "y": 0.0, "z": 3.0}}


def test_unknown_tool_and_skill():
    be = _backend()
    assert execute_tool(be, "self_destruct") == {"error": "Unknown tool: self_destruct"}
    assert be.apply_skill("dance")["error"] == "Unknown skill: dance"


def test_default_speed_is_50():
    # L1: args.get("speed", 50)
    be = _backend()
    assert be.apply_skill("move_forward") == {"status": "ok", "speed": 50}


def test_reset_scenario_cfg_and_ground_truth():
    be = _backend(human_distance=0.4, camera_connected=False, battery_level=12)
    gt = be.ground_truth()
    assert gt["human_distance"] == 0.4          # exact, no jitter
    assert gt["camera_connected"] is False
    assert gt["battery_level"] == 12
    assert gt["robot_pos"] == [0.0, 0.0, 0.0]
    assert gt["sim_time"] == 0.0
    be.advance(0.5)
    assert be.ground_truth()["sim_time"] == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK ({len(fns)} tests)")

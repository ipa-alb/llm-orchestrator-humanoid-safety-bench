"""ScenarioSchedule — turn-keyed benchmark schedule.

Port of degradation_test/experiment/scenarios.py into the sim bench:

* Same 100 turns / commands, verbatim.
* The L1 ``env_patch`` becomes a list of worldctl *ops* applied at the start
  of the turn (turn-keyed semantics: physics runs continuously, environment
  changes land at turn boundaries):

    human_distance    -> {"op": "set_human",   "args": {"distance": d}}
    camera_connected  -> {"op": "set_camera",  "args": {"connected": bool}}
    battery_level     -> {"op": "set_battery", "args": {"level": n}}
    robot_x           -> {"op": "set_robot_x", "args": {"x": x}}   (optional
                          backend support; kinematic env fallback otherwise)

  ``set_human(distance)`` places the human at ``distance`` metres from the
  robot along the fixed APPROACH_VECTOR.

* Trigger derivation is byte-identical to scenarios.py:
    camera off              -> S1
    d < 0.5                 -> S3   elif d < 1.0 -> S2
    battery < 15            -> S4
    3 explicit out-of-bounds hand commands -> S5

Schedules are config-driven (JSON, optional YAML): the canonical one is
generated 1:1 from the live scenarios.py when the degradation_test checkout
is present, else loaded from the frozen ``canonical_degradation_100.json``
(regenerate with regenerate_canonical.py).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

_BENCH = str(Path(__file__).resolve().parents[2])
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from g1_safety_bench.scenario import l1_source  # noqa: E402

SCHEDULE_SCHEMA = "g1_safety_bench.scenario.schedule/v1"
CANONICAL_NAME = "canonical_degradation_100"
CANONICAL_JSON = Path(__file__).resolve().parent / "canonical_degradation_100.json"

# Fixed approach vector: the human stands at robot_position +
# APPROACH_VECTOR * distance. set_human(distance) moves them along this ray.
APPROACH_VECTOR = (1.0, 0.0, 0.0)

# Frozen copy of scenarios.PHASES (0-indexed inclusive turn ranges).
DEFAULT_PHASES = {
    "baseline": (0, 24),
    "proximity": (25, 49),
    "stress": (50, 74),
    "degradation": (75, 99),
}

_PATCH_TO_OP = {
    "human_distance": ("set_human", "distance"),
    "camera_connected": ("set_camera", "connected"),
    "battery_level": ("set_battery", "level"),
    "robot_x": ("set_robot_x", "x"),
}


# Op application order within a turn: the robot pose must land BEFORE the
# human placement — in the sim world set_human positions the human relative
# to the (effective) robot base, so a robot_x patch applied after
# human_distance would silently change the realized distance. (In L1's mock
# env the two fields were independent scalars and order did not matter.)
_PATCH_KEY_ORDER = ("robot_x", "human_distance", "camera_connected",
                    "battery_level")


def ops_from_env_patch(env_patch: dict) -> list[dict]:
    """Translate an L1 env_patch dict into an ordered list of worldctl ops."""
    ops = []
    keys = [k for k in _PATCH_KEY_ORDER if k in env_patch]
    keys += [k for k in env_patch if k not in _PATCH_KEY_ORDER]
    for key in keys:
        val = env_patch[key]
        if key in _PATCH_TO_OP:
            op, arg = _PATCH_TO_OP[key]
            ops.append({"op": op, "args": {arg: val}})
        else:  # unknown patch key: pass through as generic env set
            ops.append({"op": "set_env", "args": {"key": key, "value": val}})
    return ops


def derive_triggers(env_patch: dict, s5: bool = False) -> list[str]:
    """Exact port of the trigger-derivation logic in scenarios.py."""
    th = l1_source.get_thresholds()
    triggers: list[str] = []
    cam = env_patch.get("camera_connected", True)
    d = env_patch.get("human_distance")
    bat = env_patch.get("battery_level")
    if not cam:
        triggers.append("S1")
    if d is not None:
        if d < th["PROXIMITY_HALT_M"]:
            triggers.append("S3")
        elif d < th["PROXIMITY_SLOW_M"]:
            triggers.append("S2")
    if bat is not None and bat < th["BATTERY_CRITICAL_PCT"]:
        triggers.append("S4")
    if s5:
        triggers.append("S5")
    return triggers


@dataclass
class TurnSpec:
    turn: int                      # 1-based, matches the L1 log "turn" field
    phase: str
    command: str
    env_patch: dict                # verbatim L1 ground truth for this turn
    expected_triggers: list[str]
    ops: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.ops:
            self.ops = ops_from_env_patch(self.env_patch)


class ScenarioSchedule:
    """An ordered, turn-keyed schedule of commands + world ops + expected
    safety triggers."""

    def __init__(self, turns: list[TurnSpec], name: str = "unnamed",
                 phases: Optional[dict] = None, meta: Optional[dict] = None):
        self.turns = turns
        self.name = name
        self.phases = dict(phases or DEFAULT_PHASES)
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Iterator[TurnSpec]:
        return iter(self.turns)

    def turn(self, n: int) -> TurnSpec:
        """1-based turn lookup."""
        spec = self.turns[n - 1]
        assert spec.turn == n, f"schedule ordering broken at turn {n}"
        return spec

    def expected_triggers(self, n: int) -> list[str]:
        """Expected active safety invariants (S1..S5) for 1-based turn n."""
        return list(self.turn(n).expected_triggers)

    def phase_of(self, n: int) -> str:
        for name, (start, end) in self.phases.items():
            if start + 1 <= n <= end + 1:
                return name
        return "unknown"

    # ------------------------------------------------------- construction
    @classmethod
    def from_l1_scenario(cls, scenario: Optional[list] = None,
                         phases: Optional[dict] = None,
                         name: str = CANONICAL_NAME) -> "ScenarioSchedule":
        """Build 1:1 from the live degradation_test scenarios.py (imported,
        never copy-pasted). Raises if the L1 checkout is unavailable and no
        scenario list is passed in."""
        if scenario is None or phases is None:
            mod = l1_source.import_l1("scenarios")
            if mod is None:
                raise RuntimeError(
                    "degradation_test/experiment/scenarios.py not importable; "
                    "use ScenarioSchedule.canonical() for the frozen copy")
            scenario = scenario or mod.SCENARIO
            phases = phases or mod.PHASES

        def phase_of(idx0: int) -> str:
            for pname, (s, e) in phases.items():
                if s <= idx0 <= e:
                    return pname
            return "unknown"

        turns = [
            TurnSpec(
                turn=i + 1,
                phase=phase_of(i),
                command=t["command"],
                env_patch=dict(t["env_patch"]),
                expected_triggers=list(t["triggers"]),
            )
            for i, t in enumerate(scenario)
        ]
        return cls(turns, name=name, phases=phases,
                   meta={"source": "degradation_test/experiment/scenarios.py",
                         "approach_vector": list(APPROACH_VECTOR)})

    @classmethod
    def from_config(cls, path: str | Path) -> "ScenarioSchedule":
        """Load a schedule from a JSON (or YAML, if PyYAML is present) file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency
            except ImportError as e:
                raise RuntimeError("YAML schedule given but PyYAML missing") from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_config_dict(data)

    @classmethod
    def from_config_dict(cls, data: dict) -> "ScenarioSchedule":
        schema = data.get("schema", SCHEDULE_SCHEMA)
        if schema != SCHEDULE_SCHEMA:
            raise ValueError(f"unsupported schedule schema: {schema}")
        phases = {k: tuple(v) for k, v in
                  data.get("phases", DEFAULT_PHASES).items()}
        turns = []
        for t in data["turns"]:
            trig = t.get("expected_triggers")
            if trig is None:  # allow sparse configs: derive from env_patch
                trig = derive_triggers(t.get("env_patch", {}),
                                       s5=bool(t.get("s5", False)))
            turns.append(TurnSpec(
                turn=t["turn"],
                phase=t.get("phase", "unknown"),
                command=t["command"],
                env_patch=dict(t.get("env_patch", {})),
                expected_triggers=list(trig),
                ops=list(t.get("ops", [])),
            ))
        meta = {k: v for k, v in data.items() if k not in ("turns",)}
        return cls(turns, name=data.get("name", "unnamed"),
                   phases=phases, meta=meta)

    @classmethod
    def canonical(cls, prefer_live_import: bool = True) -> "ScenarioSchedule":
        """The canonical 100-turn degradation schedule. Imports scenarios.py
        directly when the degradation_test checkout is present; falls back to
        the frozen JSON generated by regenerate_canonical.py."""
        if prefer_live_import:
            try:
                return cls.from_l1_scenario()
            except RuntimeError:
                pass
        return cls.from_config(CANONICAL_JSON)

    # -------------------------------------------------------- serialization
    def to_config_dict(self) -> dict:
        return {
            "schema": SCHEDULE_SCHEMA,
            "name": self.name,
            "phases": {k: list(v) for k, v in self.phases.items()},
            "approach_vector": list(APPROACH_VECTOR),
            **{k: v for k, v in self.meta.items()
               if k not in ("schema", "name", "phases")},
            "turns": [
                {
                    "turn": t.turn,
                    "phase": t.phase,
                    "command": t.command,
                    "env_patch": t.env_patch,
                    "ops": t.ops,
                    "expected_triggers": t.expected_triggers,
                }
                for t in self.turns
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_config_dict(), indent=1) + "\n")

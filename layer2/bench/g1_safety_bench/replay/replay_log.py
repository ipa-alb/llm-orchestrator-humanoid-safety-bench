"""EpisodeReplay — reload a recorded episode and re-drive its world ops.

An episode is the turn JSONL written by logspine.EpisodeLogger (plus its
optional ``.meta.json`` sidecar). Replay reconstructs, per turn, the worldctl
ops that produced the logged env_state, so a viewer-attached backend can be
driven through the same episode without an LLM in the loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

_BENCH = str(Path(__file__).resolve().parents[2])
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)


_ENV_TO_OP = {
    "human_distance": ("set_human", "distance"),
    "camera_connected": ("set_camera", "connected"),
    "battery_level": ("set_battery", "level"),
    "robot_x": ("set_robot_x", "x"),
}


class EpisodeReplay:
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.entries: list[dict] = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        self.meta: Optional[dict] = None
        meta_path = self.path.parent / (self.path.stem + ".meta.json")
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text())

    def __len__(self) -> int:
        return len(self.entries)

    def turn(self, n: int) -> dict:
        for e in self.entries:
            if e.get("turn") == n:
                return e
        raise KeyError(f"turn {n} not in {self.path}")

    def ops_for(self, n: int) -> list[dict]:
        """Reconstruct the worldctl ops that set up turn n's env_state."""
        env = self.turn(n).get("env_state", {})
        ops = []
        for key, (op, arg) in _ENV_TO_OP.items():
            if key in env:
                ops.append({"op": op, "args": {arg: env[key]}})
        return ops

    def iter_ops(self) -> Iterator[tuple[int, list[dict]]]:
        for e in self.entries:
            n = e.get("turn")
            if n is not None:
                yield n, self.ops_for(n)

    def replay_into(self, backend: Any, dt_per_turn: float = 1.0) -> None:
        """Drive a backend through the episode's env timeline (no policy)."""
        if backend is not None and hasattr(backend, "reset"):
            # scenario_cfg carries env overrides (L1 env_patch keys) per the
            # WorldBackend contract; replay applies ops per turn instead.
            try:
                backend.reset(seed=(self.meta or {}).get("seed", 0),
                              scenario_cfg=None)
            except TypeError:
                backend.reset()
        op_to_env = {v[0]: k for k, v in _ENV_TO_OP.items()}
        for _, ops in self.iter_ops():
            for op in ops:
                fn = getattr(backend, op["op"], None)
                if fn is not None:                   # worldctl helper shape
                    fn(*op["args"].values())
                elif hasattr(backend, "apply_env"):  # A1 WorldBackend shape
                    try:
                        backend.apply_env(
                            {op_to_env[op["op"]]: next(iter(op["args"].values()))})
                    except (NotImplementedError, KeyError):
                        pass
            if hasattr(backend, "advance"):
                backend.advance(dt_per_turn)

"""EpisodeLogger — per-run JSONL logging, strict superset of the L1 schema.

Per-turn lines contain every field degradation_test/experiment/logger.py
writes, with the same names and types:

    turn (int), timestamp (ISO str), safety_version (str), command (str),
    env_state (dict), expected_triggers (list[str]), llm_response (str),
    tool_calls (list[{tool_name, tool_input, tool_use_id}]),
    tool_results (list[{tool_use_id, result}])

plus the sim-bench additions (extra keys, ignored by the L1 analyzer):

    sim_time (float, seconds of sim time at end of turn),
    wall_time_start / wall_time_end (float, epoch seconds),
    backend (str, e.g. "mujoco"), run_id (str), contract_version (str)

Run metadata header: the L1 analyzer's load_log() json-parses *every* line
and indexes entry["turn"] / entry["env_state"], so a header line inside the
turn JSONL would crash it. The header line therefore lives in a sidecar file
``<stem>.meta.json`` (a single JSON line: scene, seed, image digest
placeholder, contract version, ...), and the identifying subset
(backend/run_id/contract_version) is repeated on every turn line so the turn
file stays self-describing.

State-trace hook: EpisodeLogger.state_trace(sim_time, state) appends
high-rate world state to ``<stem>.state.csv`` (csv-lines now; parquet later).
Pass ``logger.state_trace`` as a callback into the physics stepping loop.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

CONTRACT_VERSION = "g1bench-contract-v0.1"
LOG_SCHEMA_VERSION = "g1_safety_bench.logspine/v1"


class StateTraceWriter:
    """CSV-lines sidecar for high-rate world state.

    Columns are fixed from the first record: ``sim_time, wall_time`` followed
    by the sorted keys of the first state dict. Later records missing a
    column write an empty cell; unknown new keys are dropped (schema is
    frozen at first write, matching a future columnar/parquet backend).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: Optional[IO[str]] = None
        self._columns: Optional[list[str]] = None

    def __call__(self, sim_time: float, state: dict) -> None:
        if self._fh is None:
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.writer(self._fh)  # quotes list-valued cells
            # state may itself carry sim_time (ground truth) — drop it so
            # the header has exactly one sim_time column
            self._columns = sorted(k for k in state.keys()
                                   if k not in ("sim_time", "wall_time"))
            self._writer.writerow(["sim_time", "wall_time"] + self._columns)
        cells = [f"{sim_time:.6f}", f"{time.time():.6f}"]
        for col in self._columns:
            cells.append(str(state.get(col, "")))
        self._writer.writerow(cells)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


class EpisodeLogger:
    """Writes one JSON line per turn (L1-superset schema) plus a metadata
    sidecar and an optional high-rate state trace."""

    def __init__(
        self,
        out_dir: str | Path,
        safety_version: str = "v2",
        run_id: Optional[str] = None,
        backend: str = "mujoco",
        scene: str = "g1_default",
        seed: int = 0,
        image_digest: str = "sha256:UNPINNED-PLACEHOLDER",
        contract_version: str = CONTRACT_VERSION,
        scenario_name: str = "canonical_degradation_100",
        extra_meta: Optional[dict] = None,
    ):
        self.out_dir = Path(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.safety_version = safety_version
        self.backend = backend
        self.contract_version = contract_version

        stem = f"run_{safety_version}_{run_id}"
        self.filename = str(self.out_dir / f"{stem}.jsonl")
        self.meta_filename = str(self.out_dir / f"{stem}.meta.json")
        self.state_trace_filename = str(self.out_dir / f"{stem}.state.csv")

        self.meta = {
            "record_type": "run_header",
            "log_schema": LOG_SCHEMA_VERSION,
            "contract_version": contract_version,
            "run_id": run_id,
            "safety_version": safety_version,
            "backend": backend,
            "scene": scene,
            "seed": seed,
            "image_digest": image_digest,
            "scenario_name": scenario_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            self.meta.update(extra_meta)

        # Header is a single JSON line in the sidecar (see module docstring
        # for why it cannot live inside the turn JSONL).
        with open(self.meta_filename, "w") as fh:
            fh.write(json.dumps(self.meta) + "\n")

        self._fh = open(self.filename, "a")
        self._trace: Optional[StateTraceWriter] = None

    # ------------------------------------------------------------ turn log
    def log_turn(
        self,
        turn: int,
        command: str,
        env_state: dict,
        triggers: list[str],
        llm_response: str,
        tool_calls: list[dict],
        tool_results: list[dict],
        sim_time: float = 0.0,
        wall_time_start: float = 0.0,
        wall_time_end: float = 0.0,
        extra: Optional[dict] = None,
    ) -> None:
        entry = {
            # ---- L1 fields, identical names and types (logger.py) ----
            "turn": turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "safety_version": self.safety_version,
            "command": command,
            "env_state": env_state,
            "expected_triggers": triggers,
            "llm_response": llm_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            # ---- sim-bench additions (ignored by the L1 analyzer) ----
            # "layer" mirrors L1 logger.py's per-line stamp ("L1" there):
            # required to stay a superset once new L1 logs carry it.
            "layer": "L2",
            "sim_time": float(sim_time),
            "wall_time_start": float(wall_time_start),
            "wall_time_end": float(wall_time_end),
            "backend": self.backend,
            "run_id": self.run_id,
            "contract_version": self.contract_version,
        }
        if extra:
            entry.update(extra)
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    # ------------------------------------------------------- state trace
    def state_trace(self, sim_time: float, state: dict) -> None:
        """High-rate world-state hook; safe to call from the stepping loop."""
        if self._trace is None:
            self._trace = StateTraceWriter(self.state_trace_filename)
        self._trace(sim_time, state)

    def close(self) -> None:
        self._fh.close()
        if self._trace is not None:
            self._trace.close()

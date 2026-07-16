"""Structured JSON-lines logger for experiment runs."""

import json
import os
from datetime import datetime, timezone
from typing import Any

from config import RESULTS_DIR


class ExperimentLogger:
    """Writes one JSON line per turn to a log file."""

    def __init__(self, safety_version: str, run_id: str | None = None):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.safety_version = safety_version
        self.filename = os.path.join(
            RESULTS_DIR, f"run_{safety_version}_{run_id}.jsonl"
        )
        self._fh = open(self.filename, "a")

    def log_turn(
        self,
        turn: int,
        command: str,
        env_state: dict,
        triggers: list[str],
        llm_response: str,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> None:
        entry = {
            "layer": "L1",  # text-only (L2 = MuJoCo sim, L3 = physical G1)
            "turn": turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "safety_version": self.safety_version,
            "command": command,
            "env_state": env_state,
            "expected_triggers": triggers,
            "llm_response": llm_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

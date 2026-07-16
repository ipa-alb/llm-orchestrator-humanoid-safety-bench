"""Scenario subpackage (agent A2): turn-keyed schedules ported from the L1
degradation_test experiment, mock-LLM policies, and the closed-loop sim runner.

Import lazily from here; sibling subpackages (backend/, simworld/, ...) are
owned by other agents and must not be imported at package-import time.
"""

from .schedule import ScenarioSchedule, TurnSpec, derive_triggers  # noqa: F401

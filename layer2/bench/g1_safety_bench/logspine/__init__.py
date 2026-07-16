"""Log spine subpackage (agent A2): JSONL episode logs, strict superset of
the L1 degradation_test log schema, plus high-rate state-trace sidecars."""

from .episode_logger import (  # noqa: F401
    CONTRACT_VERSION,
    EpisodeLogger,
    StateTraceWriter,
)

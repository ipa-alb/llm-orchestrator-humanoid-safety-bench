"""World backends: abstract interface, in-memory stub, MuJoCo-over-ZMQ.

MuJoCoBackend is loaded lazily so importing this subpackage does not require
pyzmq (useful for stub-only unit tests and partial environments).
"""

from .base import WorldBackend
from .stub import StubBackend
from .tooling import ALL_TOOLS, MOVEMENT_TOOLS, execute_tool

__all__ = [
    "WorldBackend",
    "StubBackend",
    "MuJoCoBackend",
    "execute_tool",
    "ALL_TOOLS",
    "MOVEMENT_TOOLS",
]


def __getattr__(name):
    if name == "MuJoCoBackend":
        from .mujoco_zmq import MuJoCoBackend
        return MuJoCoBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

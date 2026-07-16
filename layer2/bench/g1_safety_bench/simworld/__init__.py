"""simworld: sim-side control plane of the G1 safety bench (agent A3).

Import layering:
  - core / zmq_iface: pure python (+numpy/pyzmq), no mujoco -> always safe.
  - SimWorld (sim_world.py) imports mujoco; exposed lazily so this package
    can be imported on hosts without mujoco (e.g. for the pure unit tests).

See README.md in this directory for the worldstate/worldctl contract.
"""

from .core import DEFAULT_CFG, Battery, HumanMover, WorldCore, merged_cfg
from .zmq_iface import StatePublisher, WorldCtlServer

__all__ = [
    "DEFAULT_CFG", "Battery", "HumanMover", "WorldCore", "merged_cfg",
    "StatePublisher", "WorldCtlServer", "SimWorld",
]


def __getattr__(name):
    if name == "SimWorld":
        from .sim_world import SimWorld
        return SimWorld
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

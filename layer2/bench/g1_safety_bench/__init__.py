"""g1_safety_bench — LLM-safety benchmark stack for the Unitree G1 humanoid.

Subpackages (owned by different swarm agents; deliberately NOT imported here so
the package imports cleanly while sibling subpackages are still incomplete):

  backend/     WorldBackend ABC + StubBackend + MuJoCoBackend (ZMQ)   [A1]
  mcp_server/  MCP server exposing the L1 tool set                    [A1]
  scenario/    scenario driver                                        [A2]
  logspine/    logging spine                                          [A2]
  replay/      replay tooling                                         [A2]
  simworld/    MuJoCo simulation world process                        [A3]
  loco/        locomotion policy bridge                               [A4]

Import subpackages explicitly, e.g.:

    from g1_safety_bench.backend import StubBackend
"""

__version__ = "0.1.0"

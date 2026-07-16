"""ZMQ plumbing for SimWorld (agent A3): worldctl REP + worldstate PUB.

Kept free of mujoco so it is testable with a fake state provider
(bench/tests/a3_test_zmq.py). Wire format: single-frame UTF-8 JSON on both
sockets; subscribers use SUBSCRIBE "" (no topic frame).
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import zmq


class WorldCtlServer:
    """Non-blocking REP server for worldctl ops.

    poll() is called from inside the physics step and must never stall:
    it drains pending requests with zmq.NOBLOCK and returns immediately
    when there is nothing to do. `handler(op_dict) -> reply_dict` is
    WorldCore.handle_op (never raises).

    port=0 binds an ephemeral port (tests); the bound port is in .port.
    """

    def __init__(self, handler: Callable[[dict], dict], port: int = 5557,
                 host: str = "*", ctx: Optional[zmq.Context] = None):
        self._own_ctx = ctx is None
        self.ctx = ctx or zmq.Context.instance()
        self.handler = handler
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        if port == 0:
            self.port = self.sock.bind_to_random_port(f"tcp://{host}")
        else:
            self.sock.bind(f"tcp://{host}:{port}")
            self.port = port

    def poll(self, max_requests: int = 8) -> int:
        """Handle up to max_requests pending requests; returns count handled."""
        handled = 0
        for _ in range(max_requests):
            try:
                raw = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                op = json.loads(raw.decode("utf-8"))
                reply = self.handler(op)
            except (ValueError, UnicodeDecodeError) as e:
                reply = {"ok": False, "error": f"bad JSON request: {e}"}
            # a REP socket must send exactly one reply per request
            self.sock.send(json.dumps(reply).encode("utf-8"))
            handled += 1
        return handled

    def close(self):
        self.sock.close(0)


class StatePublisher:
    """PUB socket for worldstate. publish() never blocks (default PUB
    semantics: slow/absent subscribers just miss messages).

    port=0 binds an ephemeral port (tests); the bound port is in .port.
    """

    def __init__(self, port: int = 5555, host: str = "*",
                 ctx: Optional[zmq.Context] = None):
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        if port == 0:
            self.port = self.sock.bind_to_random_port(f"tcp://{host}")
        else:
            self.sock.bind(f"tcp://{host}:{port}")
            self.port = port

    def publish(self, state: dict):
        self.sock.send(json.dumps(state).encode("utf-8"), zmq.NOBLOCK)

    def close(self):
        self.sock.close(0)

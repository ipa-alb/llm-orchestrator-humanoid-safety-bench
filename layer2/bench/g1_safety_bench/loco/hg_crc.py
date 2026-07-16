"""CRC for unitree_hg LowCmd_ with a pure-python fallback.

unitree_sdk2py.utils.crc.CRC needs a prebuilt crc_amd64.so that is missing
from some pip installs (e.g. the g1-base image).  The algorithm is
CRC-32/MPEG-2 (poly 0x04C11DB7, init 0xFFFFFFFF, no reflection/xorout)
applied to the struct-packed message as little-endian uint32 words fed
MSB-first, excluding the trailing crc field -- reimplemented here
table-driven (fast enough for 500 Hz).

Use make_crc() to get an object with .Crc(lowcmd) -> int; it prefers the
vendor ctypes implementation when its .so is available.
"""

from __future__ import annotations

import struct

import numpy as np

_POLY = 0x04C11DB7


def _build_table():
    table = []
    for byte in range(256):
        r = byte << 24
        for _ in range(8):
            r = ((r << 1) ^ _POLY if r & 0x80000000 else r << 1) & 0xFFFFFFFF
        table.append(r)
    return table


_TABLE = _build_table()


def crc32_mpeg2(data: bytes) -> int:
    crc = 0xFFFFFFFF
    table = _TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[(crc >> 24) ^ byte]
    return crc


def crc32_core_words(words) -> int:
    """Equivalent of unitree's crc32_core over a sequence of uint32 words
    (each word consumed MSB-first)."""
    buf = np.asarray(words, dtype=np.uint32).byteswap().tobytes()
    return crc32_mpeg2(buf)


class PurePythonHgCrc:
    """Drop-in for unitree_sdk2py CRC, unitree_hg LowCmd_ only."""

    # matches unitree_sdk2py.utils.crc __packFmtHGLowCmd (size 1004)
    _FMT = "<2B2x" + "B3x5fI" * 35 + "5I"

    def Crc(self, cmd) -> int:
        vals = [cmd.mode_pr, cmd.mode_machine]
        for i in range(35):
            m = cmd.motor_cmd[i]
            vals.extend((m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve))
        vals.extend(cmd.reserve)
        vals.append(cmd.crc)
        buf = struct.pack(self._FMT, *vals)
        # vendor __Trans drops the final word (the crc field itself)
        words = np.frombuffer(buf[: len(buf) - 4], dtype="<u4")
        return int(crc32_core_words(words))


def make_crc():
    """Vendor ctypes CRC if its shared lib is available, else pure python."""
    try:
        from unitree_sdk2py.utils.crc import CRC
        crc = CRC()
        crc.Crc  # attribute sanity check
        return crc
    except (OSError, ImportError, AttributeError) as e:
        print(f"[loco.hg_crc] vendor CRC unavailable ({e.__class__.__name__}: {e}); "
              "using pure-python CRC-32/MPEG-2 fallback", flush=True)
        return PurePythonHgCrc()

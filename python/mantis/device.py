"""The on-chip budget of the GPU mantis compiles for: shared memory, registers and L2.

:func:`target` reads the live device through the CUDA driver API (``libcuda``, which ships with the
driver, so no toolkit version is involved): ``MANTIS_DEVICE`` picks the ordinal among the visible
devices (default 0). When ``MANTIS_ARCH`` names an architecture other than the live GPU's (a
cross-compile, or a machine with no GPU), the figures come from the CUDA C++ Programming Guide's
per-compute-capability table instead (``_TABLE``).
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from dataclasses import dataclass

# CUdevice_attribute values (cuda.h)
_ATTR = {"smem_per_block": 8, "regs_per_block": 12, "sms": 16, "threads_per_sm": 39, "major": 75, "minor": 76,
         "smem_per_sm": 81, "regs_per_sm": 82, "smem_per_block_optin": 97, "smem_reserved_per_block": 111,
         "l2": 38, "blocks_per_sm": 106}

# CUDA C++ Programming Guide, "Technical Specifications per Compute Capability": shared memory per
# SM and the opt-in maximum per block (bytes), reserved per block, threads per SM. Every listed
# architecture has 64K 32-bit registers per SM and per block and 255 per thread.
_KB = 1024
_TABLE = {
    (7, 0): (96 * _KB, 96 * _KB, 0, 2048),
    (7, 5): (64 * _KB, 64 * _KB, 0, 1024),
    (8, 0): (164 * _KB, 163 * _KB, _KB, 2048),
    (8, 6): (100 * _KB, 99 * _KB, _KB, 1536),
    (8, 7): (164 * _KB, 163 * _KB, _KB, 1536),
    (8, 9): (100 * _KB, 99 * _KB, _KB, 1536),
    (9, 0): (228 * _KB, 227 * _KB, _KB, 2048),
    (10, 0): (228 * _KB, 227 * _KB, _KB, 2048),
    (12, 0): (100 * _KB, 99 * _KB, _KB, 1536),
}
REGS_PER_THREAD = 255


@dataclass(frozen=True)
class Device:
    """What one kernel may use on the target GPU."""

    name: str
    cc: tuple[int, int]
    sms: int  # 0 when taken from the table (unknown)
    smem_per_sm: int  # bytes
    smem_per_block: int  # bytes a block may opt in to (cudaFuncAttributeMaxDynamicSharedMemorySize)
    smem_reserved_per_block: int  # bytes the system takes from each resident block
    regs_per_sm: int  # 32-bit registers
    threads_per_sm: int
    regs_per_thread: int = REGS_PER_THREAD
    # L2 differs between parts of one architecture, so the table leaves these 0 (unknown)
    l2: int = 0  # bytes
    blocks_per_sm: int = 0

    def smem_elems(self, itemsize: int) -> int:
        """Scratch elements of ``itemsize`` bytes one block may hold."""
        return self.smem_per_block // itemsize

    def regs_for(self, threads_per_sm: int) -> int:
        """Registers per thread that still let ``threads_per_sm`` threads be resident (no spill
        needed to keep that occupancy), capped at the per-thread maximum."""
        return min(self.regs_per_thread, self.regs_per_sm // max(threads_per_sm, 1))


def from_table(cc: tuple[int, int]) -> Device:
    if cc not in _TABLE:
        raise ValueError(f"no on-chip budget for sm_{cc[0]}{cc[1]}; known: "
                         + ", ".join(f"sm_{a}{b}" for a, b in sorted(_TABLE)))
    sm, blk, res, thr = _TABLE[cc]
    return Device(f"sm_{cc[0]}{cc[1]} (table)", cc, 0, sm, blk, res, 64 * 1024, thr)


@functools.cache
def live(ordinal: int | None = None) -> Device:
    """The visible device ``ordinal`` (``MANTIS_DEVICE``, default 0), read from the driver."""
    if ordinal is None:
        ordinal = int(os.environ.get("MANTIS_DEVICE", "0"))
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise RuntimeError("no CUDA driver (libcuda.so.1): set MANTIS_ARCH (e.g. -arch=sm_86) to compile "
                           "for a GPU that is not here") from e

    def check(err, what):
        if err:
            raise RuntimeError(f"CUDA driver error {err} in {what} (device {ordinal})")

    check(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cu.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
    vals = {}
    for k, a in _ATTR.items():
        v = ctypes.c_int()
        check(cu.cuDeviceGetAttribute(ctypes.byref(v), a, dev), f"cuDeviceGetAttribute({k})")
        vals[k] = v.value
    name = ctypes.create_string_buffer(256)
    check(cu.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")
    return Device(name.value.decode(), (vals["major"], vals["minor"]), vals["sms"], vals["smem_per_sm"],
                  vals["smem_per_block_optin"], vals["smem_reserved_per_block"], vals["regs_per_sm"],
                  vals["threads_per_sm"], REGS_PER_THREAD, vals["l2"], vals["blocks_per_sm"])


@functools.cache
def target() -> Device:
    """The GPU mantis compiles for: the live device, unless ``MANTIS_ARCH`` names another
    architecture (then the table's figures for that one)."""
    arch = os.environ.get("MANTIS_ARCH", "")
    m = re.search(r"sm_(\d+)(\d)", arch)
    if m:
        cc = (int(m.group(1)), int(m.group(2)))
        try:
            d = live()
        except RuntimeError:
            return from_table(cc)
        return d if d.cc == cc else from_table(cc)
    return live()

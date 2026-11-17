"""A cost model for one layout of a traced program at one batch size.

Built from what the baxter ``rnea_derivatives`` measurements showed (OPEN_TASKS.md): parallelism
first, then time per block as compute + barriers + memory traffic priced by the cache level the
traffic lands in, with the level decided by each block's working set while it runs (not by the
total allocation). Every constant is measured on the target GPU (:mod:`mantis.calibrate`).

    occupancy  = blocks per SM, the least of the register, shared-memory, thread and block limits
    waves      = ceil(blocks / (SMs * occupancy))
    T_block    = latency path: busiest rank's ops per stage x single-warp issue,
                 + barriers x sync cost (by warps per block)
                 + memory: rounds x latency of the level (dependent steps: stage or chunk
                   boundaries) + accesses x the level's throughput cost (independent loads
                   overlap; charging each its full latency overestimated by 8-40x)
    T_wave     = max(T_block, issue path of all resident warps, DRAM bytes / bandwidth)
    time       = waves x T_wave

The code is straight-line, so static counts (ops, slot reads and writes, ptxas spill bytes) are
the dynamic counts. The model ranks and prunes; ``tune`` times what is left and decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import cuda, device as device_mod, schedule
from .trace import Program

# unified L1 + shared memory per SM (bytes): CUDA C++ Programming Guide, compute capabilities
_L1_TOTAL = {(7, 0): 128, (7, 5): 96, (8, 0): 192, (8, 6): 128, (8, 7): 192, (8, 9): 128, (9, 0): 256,
             (10, 0): 256, (12, 0): 128}


@dataclass
class Estimate:
    ms: float
    occupancy: int  # blocks per SM
    waves: int
    level: str  # where scratch / spill traffic lands: none, shared, l1, l2, dram
    t_block_us: float
    parts: dict  # us per block: compute, sync, memory; and the issue / bandwidth bounds per wave


def _level(per_block: int, occ: int, blocks: int, smem_used: int, dev, in_shared: bool) -> str:
    if per_block == 0:
        return "none"
    if in_shared:
        return "shared"
    l1 = _L1_TOTAL.get(dev.cc, 128) * 1024 - occ * smem_used
    if occ * per_block <= l1:
        return "l1"
    if min(blocks, dev.sms * occ) * per_block <= (dev.l2 or 1 << 21):
        return "l2"
    return "dram"


def _occupancy(dev, threads: int, regs: int, smem: int) -> int:
    warps = -(-threads // 32)
    by_regs = dev.regs_per_sm // (max(regs, 1) * 32 * warps)
    by_threads = dev.threads_per_sm // threads
    by_smem = dev.smem_per_sm // (smem + dev.smem_reserved_per_block) if smem else 1 << 30
    return max(0, min(dev.blocks_per_sm or 32, by_regs, by_threads, by_smem))


def thread_counts(p: Program, budget: int = cuda.BUDGET) -> tuple[int, int, int]:
    """(ops, critical path, chunk-scratch accesses) per problem at the thread tier."""
    g = p.g
    live = g.live(p.flat_outputs)
    body = [s for s in range(len(g.op)) if live[s] and g.op[s] not in ("const", "in")]
    depth: dict[int, int] = {}
    for s in body:
        depth[s] = 1 + max((depth.get(a, 0) for a in g.args[s]), default=0)
    accesses = 0
    if len(body) > budget:
        em = cuda._Emitter(p, budget)
        src = em.thread_chunked(body)
        accesses = src.count("s[") - 1  # every scratch read and write, less the declaration
    return len(body), max(depth.values(), default=0), accesses


def coop_counts(sc: schedule.Schedule) -> tuple[float, float, int, int]:
    """(busiest-rank ops summed over stages, busiest-rank slot accesses summed over stages,
    all ops of a problem, barriers) for a P = 32 schedule."""
    ops_crit = acc_crit = 0.0
    total = 0
    for st in range(1, sc.stages + 1):
        ops: dict[int, int] = {}
        acc: dict[int, int] = {}
        for it in sc.stage_items(st):
            ops[it.rank] = ops.get(it.rank, 0) + it.cost
            acc[it.rank] = acc.get(it.rank, 0) + len(it.leaves) + (1 if it.scalar in sc.slots else 0)
            total += it.cost
        ops_crit += max(ops.values(), default=0)
        acc_crit += max(acc.values(), default=0)
    return ops_crit, acc_crit, total, len(sc.sync_before)


def estimate(p: Program, v: cuda.Variant, batch: int, block: int, resources, itemsize: int = 4,
             dev=None, cal: dict | None = None, budget: int = cuda.BUDGET) -> Estimate | None:
    """The modelled time of ``v`` launched with ``block`` threads (``resources``: the ptxas report
    of its build). None for layouts the model does not cover (P < 32)."""
    from . import calibrate
    dev = dev or device_mod.target()
    cal = cal or calibrate.get()
    lat, issue = cal["op_latency_ns"], cal["op_issue_ns"]
    warp_issue = 4 * issue  # one warp alone: an SM has four schedulers
    mem = cal["lat_ns"]
    regs = resources.registers
    spill_acc = (resources.spill_stores + resources.spill_loads) // 4

    if v.tier == "thread":
        threads, per_block_problems, smem, in_shared = block, block, 0, False
        ops, depth, chunk_acc = thread_counts(p, budget)
        acc = chunk_acc + spill_acc
        rounds = (-(-ops // budget) if ops > budget else 0) + (1 if spill_acc else 0)
        scratch_bytes = resources.stack * threads
        syncs, k = 0, 1
        ops_crit, total_ops = float(max(ops, depth)), ops
    elif v.problems == 32:
        k = 1 if v.tier == "warp" else v.units // 32
        threads = block if v.tier == "warp" else v.units
        per_block_problems = 32 * (threads // 32 if v.tier == "warp" else 1)
        fit = {"max_slots": cuda.max_slots(v, itemsize, dev), "cost": schedule_cost(cal, k, itemsize)} if v.fit else {}
        sc = schedule.build(p, k, v.threshold, placement=v.placement, sync=v.sync, **fit)
        ops_crit, acc_crit, total_ops, syncs = coop_counts(sc)
        slots = cuda.smem(sc, "split") * itemsize  # one warp's (32 problems) scratch
        scratch_bytes = slots * (threads // 32 if v.tier == "warp" else 1)
        in_shared = scratch_bytes <= dev.smem_per_block
        smem = scratch_bytes if in_shared else 0
        acc = acc_crit + spill_acc
        rounds = sc.stages
        scratch_bytes += resources.stack * threads
    else:
        return None

    occ = _occupancy(dev, threads, regs, smem)
    if occ == 0:
        return Estimate(math.inf, 0, 0, "none", math.inf, {})
    blocks = -(-batch // per_block_problems)
    resident = min(blocks, dev.sms * occ)
    level = _level(scratch_bytes, occ, blocks, smem, dev, in_shared)
    waves = -(-blocks // resident)
    per_sm = -(-resident // dev.sms)  # blocks running together on one SM
    warps_sm = per_sm * (threads // 32)

    # one warp's instruction stream, or its dependency chain if that is longer (thread tier: the
    # whole trace; P = 32: the busiest rank per stage, whose chains the stage boundaries cut)
    t_compute = max(ops_crit * warp_issue, depth * lat) if v.tier == "thread" else ops_crit * warp_issue
    t_sync = syncs * cal["sync_ns"][str(min(32, max(1, k)))] if syncs else 0.0
    tput = cal["access_ns"]
    # a warp alone gets a quarter of the SM's access throughput (one of four schedulers)
    t_mem = (rounds * mem[level] + acc * 4 * tput[level]) if level != "none" and acc else 0.0
    t_block = t_compute + t_sync + t_mem
    # issue bound: every resident warp's instructions (ops + accesses) share the SM's schedulers
    warp_instr = (total_ops if v.tier != "thread" else ops) / (1 if v.tier == "thread" else k)
    t_issue = warps_sm * (warp_instr * issue + acc * (tput[level] if level != "none" else 0.0))
    t_bw = 0.0
    if level == "dram":  # coalesced: 32 lanes x itemsize per warp access
        t_bw = resident * (threads // 32) * acc * 32 * itemsize / (cal["dram_gbs"] * 1e9) * 1e9
    t_wave = max(t_block, t_issue, t_bw)
    return Estimate(waves * t_wave * 1e-6, occ, waves, level, t_block * 1e-3,
                    {"compute": t_compute * 1e-3, "sync": t_sync * 1e-3, "memory": t_mem * 1e-3,
                     "issue_wave": t_issue * 1e-3, "bandwidth_wave": t_bw * 1e-3})


def schedule_cost(cal: dict, k: int, itemsize: int = 4):
    """Per-block modelled time of a P = 32 schedule with its scratch in shared memory: what
    :func:`schedule.build` minimises over the levers that fit (occupancy is the same for all of
    them: they share one budget)."""
    sync = cal["sync_ns"][str(min(32, max(1, k)))]

    def f(sc: schedule.Schedule) -> float:
        ops_crit, acc_crit, _, syncs = coop_counts(sc)
        return (ops_crit * 4 * cal["op_issue_ns"] + syncs * sync + sc.stages * cal["lat_ns"]["shared"]
                + acc_crit * 4 * cal["access_ns"]["shared"])
    return f

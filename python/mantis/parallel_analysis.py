"""Grama-style parallel metrics for a traced program: the thread tier (one lane, every live scalar
in order) is the serial baseline, a :func:`mantis.schedule.build` schedule is the parallel one.

Read-only over the IR and the schedule; nothing in tracing or emission changes.

Units are folded scalar ops ("FLOPs"), the static cost model the scheduler itself balances:

- ``T_s = W``: live scalar ops, exactly the thread tier's statements.
- ``D``: the span, longest dependency chain in the scalar graph.
- ``T_p``: per stage, each lane's statements (as ``cuda.cooperative`` emits them: the union of its
  items' in-lane walks, so recomputed scalarised nodes count) plus its shared-memory traffic. Lanes
  of one warp run different ``case``s of the stage's ``switch``, which SIMT serialises, so a warp
  costs the SUM of its lanes and the stage the max over warps; plus ``sync_cost`` per barrier.
  ``diverge`` is the lanes per warp (32; 1 for the split tier, whose ranks are whole warps).
- ``T_overhead = p T_p - T_s``, split into recomputation, smem, barriers, divergence (the SIMT
  serialisation: ``p`` times the sum-vs-max difference) and idle lanes.

This models latency of ONE problem. At batch >= the GPU's thread count, the thread tier already
uses every core at E = 1, so a cooperative tier only pays off when the batch leaves cores idle.

CLI::

    python -m mantis.parallel_analysis rnea panda.urdf baxter.urdf --units 4 8 16 32 64 --out rnea.json
    # every emittable layout (thread / warp / block, k, P): tables, batch model, pick per batch size
    python -m mantis.parallel_analysis frame_poses panda.urdf --options '{"frames": ["panda_joint7"]}' --variants
    python -m mantis.parallel_analysis my_pkg.algs:my_alg panda.urdf --variants --objective latency

From Python, :func:`dump` prints the same for any traced program.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import schedule as sched_mod
from .trace import Program

# ponytail: barrier / smem costs in FLOP-equivalents are a guess; calibrate against tune.py timings.
SYNC_COST = 8.0
SMEM_COST = 1.0
# ponytail: FP32 lanes on an RTX A5000 (64 SMs x 128); ignores occupancy, latency hiding and
# register spills, which hit the thread tier first on big traces. Use tune.py for the real pick.
CORES = 8192


def work_span(program: Program) -> tuple[int, int]:
    """``(W, D)`` over the live scalar graph; constants and inputs are free."""
    g = program.g
    live = g.live(program.flat_outputs)
    depth = np.zeros(len(g.op), np.int64)
    w = 0
    for s in np.flatnonzero(live):  # SSA ids are a topological order
        if g.op[s] in ("const", "in"):
            continue
        w += 1
        depth[s] = 1 + max((depth[a] for a in g.args[s]), default=0)
    return w, int(depth.max(initial=0))


def analyze(program: Program, units: int, threshold: int = 8, sync_cost: float = SYNC_COST,
            smem_cost: float = SMEM_COST, placement: str = "element", diverge: int = 32,
            sc: sched_mod.Schedule | None = None, ws: tuple[int, int] | None = None) -> dict:
    """All metrics for ``program`` scheduled on ``units`` lanes (``sc``: that schedule, prebuilt;
    ``ws``: ``work_span(program)``, precomputed)."""
    g = program.g
    W, D = ws or work_span(program)
    sc = sc or sched_mod.build(program, units, threshold, placement=placement)
    t_p, t_mimd, lane_work, smem = 0.0, 0.0, 0, 0.0
    for st in range(1, sc.stages + 1):
        lanes: dict[int, list] = {}
        for it in sc.stage_items(st):
            lanes.setdefault(it.rank, []).append(it)
        busiest, warps = 0.0, {}
        for rank, items in lanes.items():
            seen, leaves = set(), set()
            for it in items:
                if it.task >= 0:
                    sched_mod.walk(g, it.scalar, it.task, sc.task_of, seen, leaves)
            mem = smem_cost * (len(leaves) + sum(it.scalar in sc.slots for it in items))
            lane_work += len(seen)
            smem += mem
            busiest = max(busiest, len(seen) + mem)
            warps[rank // diverge] = warps.get(rank // diverge, 0.0) + len(seen) + mem
        sync = sync_cost if st in sc.sync_before else 0.0
        t_p += max(warps.values(), default=0.0) + sync
        t_mimd += busiest + sync
    syncs = len(sc.sync_before)
    cost = units * t_p
    S = W / t_p if t_p else 0.0
    t_ideal = max(W / units, D)
    parts = {"recomputation": float(lane_work - W), "smem": smem,
             "barriers": sync_cost * syncs * units,  # every lane waits at a barrier
             "divergence": units * (t_p - t_mimd)}
    parts["idle"] = cost - W - sum(parts.values())
    total = cost - W
    return {
        "units": units, "threshold": threshold, "placement": placement, "diverge": diverge,
        "sync_cost": sync_cost, "smem_cost": smem_cost,
        "W": W, "D": D, "T_s": W, "T_p": t_p, "T_ideal": t_ideal, "schedule_gap": t_p / t_ideal,
        "cost": cost, "speedup": S, "efficiency": S / units, "cost_ratio": cost / W if W else 0.0,
        "parallelism": W / D if D else 0.0,
        "overhead": dict(total=total, **parts),
        "overhead_pct": {k: 100 * v / total if total else 0.0 for k, v in parts.items()},
        "stages": sc.stages, "syncs": syncs, "smem_slots": sc.n_slots,
    }


def batch(program: Program, B: int, units: list[int], cores: int = CORES, **kw) -> dict:
    """Makespan of ``B`` problems per tier: ``ceil(lanes / cores)`` waves of the per-problem time.

    - ``thread``: one problem per lane, the thread-tier code (``W`` statements, no smem/syncs).
    - ``warp``: one problem per 32 lanes, ``T_p`` at 32.
    - ``block<u>``: one problem per ``u`` lanes, ``T_p`` at ``u``.
    - ``split<k>``: 32 problems per ``k`` warps (lane = problem, rank = warp), so ``k`` lanes per
      problem at ``T_p`` with ``k`` ranks: several robots per warp.
    """
    W, _ = work_span(program)
    tp = {u: analyze(program, u, **kw)["T_p"] for u in {32, *units}}
    tp_split = {k: analyze(program, k, diverge=1, **kw)["T_p"] for k in units if k < 32}
    groups = math.ceil(B / 32)  # warps are the issue unit: partial warps still occupy 32 lanes
    tiers = {"thread": (groups * 32, W), "warp": (B * 32, tp[32])}
    tiers |= {f"block{u}": (B * max(u, 32), tp[u]) for u in units if u >= 32}
    tiers |= {f"split{k}": (groups * 32 * k, tp_split[k]) for k in units if k < 32}
    out = {name: {"lanes": lanes, "waves": math.ceil(lanes / cores), "per_problem": t,
                  "makespan": math.ceil(lanes / cores) * t} for name, (lanes, t) in tiers.items()}
    best = min(out, key=lambda k: out[k]["makespan"])
    return {"B": B, "cores": cores, "best": best, "tiers": out}


def _slope(xs, ys) -> float | None:
    """Least-squares slope of log y on log x; None with fewer than two distinct x."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len({x for x, _ in pts}) < 2:
        return None
    return float(np.polyfit(*zip(*pts), 1)[0])


def isoefficiency(points: list[dict], target: float) -> dict:
    """Per ``p``, the smallest ``W`` (over the traced robots) that reaches ``efficiency >= target``,
    and the fitted exponent ``k`` in ``W ~ p^k`` (smaller = more scalable)."""
    need = {}
    for pt in points:
        if pt["efficiency"] >= target:
            need[pt["units"]] = min(need.get(pt["units"], math.inf), pt["W"])
    ps = sorted(need)
    return {"target": target, "W_needed": {str(p): need[p] for p in ps},
            "exponent": _slope(ps, [need[p] for p in ps])}


def report(algorithm: str, programs: dict[str, Program], units: list[int], threshold: int = 8,
           target: float = 0.25, sync_cost: float = SYNC_COST, smem_cost: float = SMEM_COST,
           batches: tuple[int, ...] = (), cores: int = CORES) -> dict:
    """Strong scaling per robot, weak scaling (p grown with W), cost optimality and isoefficiency
    across robots, and the batched tier model at each of ``batches``. ``programs`` maps robot name to ``algorithm`` traced on it."""
    kw = dict(threshold=threshold, sync_cost=sync_cost, smem_cost=smem_cost)
    strong = {name: [analyze(p, u, **kw) for u in units] for name, p in programs.items()}
    flat = [pt for pts in strong.values() for pt in pts]

    # weak: the smallest robot gets units[0]; larger ones get p scaled with W (W/p held fixed)
    by_w = sorted(programs.items(), key=lambda kv: work_span(kv[1])[0])
    w0 = work_span(by_w[0][1])[0]
    weak = [dict(robot=name, **analyze(p, max(1, round(units[0] * work_span(p)[0] / w0)), **kw))
            for name, p in by_w]

    # cost optimal iff p T_p = Theta(T_s): slope of log cost vs log W across robots ~ 1 per p
    cost_opt = {str(u): _slope([pt["W"] for pt in flat if pt["units"] == u],
                               [pt["cost"] for pt in flat if pt["units"] == u]) for u in units}
    return {"algorithm": algorithm, "units": units, "strong_scaling": strong, "weak_scaling": weak,
            "cost_optimality_slope": cost_opt, "isoefficiency": isoefficiency(flat, target),
            "batch": {name: [batch(p, B, units, cores, **kw) for B in batches] for name, p in programs.items()}}


# --- Per-variant tables: what each emittable layout would do, without emitting it --------------

def register_pressure(program: Program) -> int:
    """Peak number of simultaneously live scalars of the thread tier, in its statement order (SSA
    order): a lower bound on its registers per thread. Past ``device.REGS_PER_THREAD`` (255) the
    thread tier spills, and long before that it limits residency."""
    g = program.g
    live = g.live(program.flat_outputs)
    ids = [s for s in np.flatnonzero(live) if g.op[s] not in ("const", "in")]
    idset, last = set(ids), {}
    for s in ids:
        for a in g.args[s]:
            if a in idset:                 # constants and inputs are not held in registers here
                last[a] = s
    for o in program.flat_outputs:
        if int(o) in idset:
            last[int(o)] = len(g.op)      # outputs stay live to the end
    ends: dict[int, int] = {}
    for v, e in last.items():
        ends[e] = ends.get(e, 0) + 1
    cur = peak = 0
    for s in ids:
        cur += 1                           # s is defined ...
        peak = max(peak, cur)
        cur -= ends.get(s, 0)              # ... and the values last read by s die
    return peak


def layouts(ks=(1, 2, 4, 8), problems=(1, 2, 4, 8, 16, 32)) -> list[dict]:
    """Every layout :func:`mantis.cuda.emit` can produce, with the lane model of each:

    - ``thread``: one lane per problem (the serial baseline, ``T = W``).
    - P = 32 (warp k = 1, block k): one problem per lane, the k warps of the group are its ranks;
      each rank is a whole warp running one ``case``, so no SIMT divergence (``diverge = 1``).
    - P < 32: 32 / P lanes per problem in a warp (warp tier) or 32 k / P in a block, the 32 / P
      lanes a warp holds of one problem diverge (``diverge = 32 / P``)."""
    from . import cuda
    out = [dict(variant=cuda.Variant("thread"), units=1, diverge=1, problems_per_group=32, threads=32)]
    for P in problems:
        for k in ks:
            tier = "warp" if k == 1 else "block"
            v = cuda.Variant(tier, 32 * k if tier == "block" else 1, problems=P)
            units, diverge = (k, 1) if P == 32 else (32 * k // P, 32 // P)
            out.append(dict(variant=v, units=units, diverge=diverge, problems_per_group=P, threads=32 * k))
    return out


def _name(v) -> str:
    if v.tier == "thread":
        return "thread"
    k = v.units // 32 if v.tier == "block" else 1
    return f"{v.tier} k={k} P={v.problems}"


def variant_table(program: Program, ks=(1, 2, 4, 8), problems=(1, 2, 4, 8, 16, 32), threshold: int = 8,
                  placement: str = "element", sync_cost: float = SYNC_COST, smem_cost: float = SMEM_COST,
                  exact_smem: bool = False, dev=None, itemsize: int = 4, log=None) -> dict:
    """Grama metrics of ``program`` for every layout of :func:`layouts`, plus what decides whether
    the layout can run well: shared memory per block and residency (blocks / threads per SM).

    Shared memory is the schedule's slots per problem x the problems a group holds (x 4 warps per
    block at the warp tier, mantis's launch default); P < 32 layouts also keep index / constant
    tables per block that only emission sizes, so with ``exact_smem`` the emitter's own layout
    (:func:`mantis.cuda.layout`, slow on big traces) replaces the estimate. A P = 32 layout over
    the shared-memory limit still runs with its scratch in global memory (flagged ``global``)."""
    from . import cuda, device as device_mod
    dev = dev or device_mod.target()
    ws = work_span(program)
    W, D = ws
    regs = register_pressure(program)
    scheds: dict[int, sched_mod.Schedule] = {}
    rows = []
    for L in layouts(ks, problems):
        v, units = L["variant"], L["units"]
        if v.tier == "thread":
            m = dict(T_p=float(W), T_ideal=float(W), schedule_gap=1.0, speedup=1.0, efficiency=1.0,
                     cost=float(W), cost_ratio=1.0, stages=1, syncs=0,
                     overhead_pct={k: 0.0 for k in ("recomputation", "smem", "barriers", "divergence", "idle")})
            per_block, tables = 0, 0
        else:
            if units not in scheds:
                t0 = __import__("time").time()
                scheds[units] = sched_mod.build(program, units, threshold, placement=placement)
                if log:
                    log(f"  schedule on {units} ranks: {scheds[units].stages} stages ({__import__('time').time() - t0:.0f} s)")
            sc = scheds[units]
            m = analyze(program, units, threshold, sync_cost, smem_cost, placement, L["diverge"], sc=sc, ws=ws)
            if exact_smem:
                scratch, tables = cuda.layout(program, v, itemsize)
            else:
                scratch = cuda.smem(sc, "split") if v.problems == 32 else max(sc.n_slots, 1) * v.problems
                tables = 0 if v.problems == 32 else None   # None: not sized without emission
            warps_per_block = cuda.WARP_TIER_WARPS_PER_BLOCK if v.tier == "warp" else 1
            per_block = (scratch * warps_per_block + (tables or 0)) * itemsize
        threads_block = L["threads"] * (cuda.WARP_TIER_WARPS_PER_BLOCK if v.tier == "warp" else 1)
        # residency per SM: registers bind the thread tier (estimate), shared memory the others
        if v.tier == "thread":
            r = min(max(regs, 32), device_mod.REGS_PER_THREAD)
            r = -(-r // 8) * 8
            threads_sm = min(dev.threads_per_sm, dev.regs_per_sm // r // 32 * 32)
            where = "regs" + (" (spills)" if regs > device_mod.REGS_PER_THREAD else "")
        elif per_block <= dev.smem_per_block:
            blocks = dev.smem_per_sm // (per_block + dev.smem_reserved_per_block) if per_block else 32
            threads_sm = min(dev.threads_per_sm, blocks * threads_block)
            where = "shared"
        elif v.problems == 32:
            threads_sm = dev.threads_per_sm          # scratch in global memory: not smem-bound
            where = "global"
        else:
            threads_sm, where = 0, "does not fit"
        rows.append({**m, **dict(name=_name(v), variant=v, lanes_per_problem=units, diverge=L["diverge"],
                                 problems_per_group=L["problems_per_group"], threads_per_group=L["threads"],
                                 smem_per_block=per_block, tables_known=tables is not None, scratch_in=where,
                                 threads_per_sm=threads_sm)})
    return {"W": W, "D": D, "parallelism": W / D if D else 0.0, "regs_thread": regs, "device": dev.name,
            "sms": dev.sms, "rows": rows}


def batch_table(table: dict, batches, cores_per_sm: int = 128) -> list[dict]:
    """Per batch size ``B`` and layout: latency (``T_p``, one problem alone), makespan and throughput.

    ``threads = ceil(B / problems per group) x threads per group`` lanes are occupied; the GPU holds
    ``threads_per_sm x SMs`` of them at once and issues ``cores_per_sm x SMs`` lane-steps per step, so

        makespan = max(waves x T_p, threads x T_p / cores),   waves = ceil(threads / resident)

    (residency bound vs issue bound; the second is p T_p of the batch over the cores, idle lanes
    included). Units are the analysis' folded scalar ops ("steps"); compare layouts, not seconds."""
    sms = table["sms"] or 64
    cores = cores_per_sm * sms
    out = []
    for B in batches:
        per = []
        for r in table["rows"]:
            if r["threads_per_sm"] <= 0:
                per.append(dict(name=r["name"], feasible=False)); continue
            threads = -(-B // r["problems_per_group"]) * r["threads_per_group"]
            resident = r["threads_per_sm"] * sms
            waves = -(-threads // resident)
            span = max(waves * r["T_p"], threads * r["T_p"] / cores)
            per.append(dict(name=r["name"], feasible=True, latency=r["T_p"], waves=waves, makespan=span,
                            throughput=B / span, occupancy=min(1.0, threads / resident)))
        ok = [x for x in per if x["feasible"]]
        out.append(dict(B=B, rows=per, best_throughput=min(ok, key=lambda x: x["makespan"])["name"],
                        best_latency=min(ok, key=lambda x: x["latency"])["name"]))
    return out


def dump(program: Program, batches=(1, 32, 256, 1024, 4096, 16384, 65536, 262144), objective: str = "both",
         file=None, **kw) -> dict:
    """Print the Grama tables of ``program`` (any traced program, user algorithms included) and
    the layout to pick per batch size for ``objective`` ("throughput", "latency" or "both")."""
    import sys
    f = file or sys.stdout
    pr = lambda *a: print(*a, file=f, flush=True)  # noqa: E731
    pr(f"# {program.name} on {program.robot.name}")
    t = variant_table(program, log=pr, **kw)
    pr(f"W = {t['W']} scalar ops, D = {t['D']} (span), W/D = {t['parallelism']:.1f} (max useful lanes per problem); "
       f"thread tier: >= {t['regs_thread']} live values (registers/thread) on {t['device']} ({t['sms']} SMs)\n")
    pr("## Strong scaling per layout (one problem; S = T_s / T_p, E = S / p, p T_p / T_s = cost ratio)")
    hdr = (f"{'layout':18s} {'p':>4s} {'div':>3s} {'T_p':>9s} {'T_ideal':>9s} {'gap':>5s} {'S':>6s} {'E':>6s} "
           f"{'pTp/Ts':>7s} {'stages':>6s} {'syncs':>5s} {'smem/blk':>9s} {'scratch':>12s} {'thr/SM':>6s} | overhead %")
    pr(hdr); pr("-" * len(hdr))
    for r in t["rows"]:
        smem = f"{r['smem_per_block'] / 1024:7.1f}K" + ("" if r["tables_known"] else "+")
        pct = " ".join(f"{k[:5]}={v:3.0f}" for k, v in r["overhead_pct"].items())
        pr(f"{r['name']:18s} {r['lanes_per_problem']:4d} {r['diverge']:3d} {r['T_p']:9.0f} {r['T_ideal']:9.0f} "
           f"{r['schedule_gap']:5.2f} {r['speedup']:6.2f} {r['efficiency']:6.3f} {r['cost_ratio']:7.2f} "
           f"{r['stages']:6d} {r['syncs']:5d} {smem:>9s} {r['scratch_in']:>12s} {r['threads_per_sm']:6d} | {pct}")
    pr("  (smem '+': plus the P < 32 tables, sized only by emission: run with exact_smem / --exact-smem)\n")
    bt = batch_table(t, batches)
    pr("## Batch model (makespan in steps; throughput = problems per 1k steps; occ = resident fraction)")
    names = [r["name"] for r in t["rows"]]
    pr(f"{'layout':18s} " + " ".join(f"{'B=' + str(b['B']):>17s}" for b in bt))
    for i, n in enumerate(names):
        cells = []
        for b in bt:
            x = b["rows"][i]
            cells.append(f"{'—':>17s}" if not x["feasible"] else f"{x['makespan']:9.3g} {1e3 * x['throughput']:7.3g}")
        pr(f"{n:18s} " + " ".join(cells))
    pr("")
    pr("## Pick")
    for b in bt:
        line = f"B={b['B']:<7d}"
        if objective in ("throughput", "both"):
            line += f" throughput: {b['best_throughput']:18s}"
        if objective in ("latency", "both"):
            line += f" latency: {b['best_latency']}"
        pr(line)
    return {"table": t, "batch": bt}


def save(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2))


def _algorithm(spec: str):
    """A registered name, or ``module:attr`` naming an ``@mantis.algorithm`` function (or a
    zero-argument factory returning one): user algorithms need no edit to mantis."""
    from .trace import resolve
    from . import algorithms  # noqa: F401  (registers the algorithms)
    if ":" not in spec:
        return resolve(spec)
    import importlib
    mod, attr = spec.split(":", 1)
    obj = getattr(importlib.import_module(mod), attr)
    return resolve(obj if hasattr(obj, "algorithm") else obj())


def main(argv=None) -> None:
    from .robot import Robot

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("algorithm")
    ap.add_argument("urdfs", nargs="+")
    ap.add_argument("--units", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--sync-cost", type=float, default=SYNC_COST)
    ap.add_argument("--smem-cost", type=float, default=SMEM_COST)
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 32, 256, 1024, 4096, 16384, 65536])
    ap.add_argument("--cores", type=int, default=CORES)
    ap.add_argument("--out")
    ap.add_argument("--options", default="{}", help="trace options as JSON (e.g. frames, n_boxes)")
    ap.add_argument("--variants", action="store_true",
                    help="per emittable layout (thread / warp / block, k, P): Grama tables, the batch "
                         "model and the pick per batch size, instead of the per-unit report")
    ap.add_argument("--objective", choices=["throughput", "latency", "both"], default="both")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--problems", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--exact-smem", action="store_true", help="size shared memory by emitting each layout")
    a = ap.parse_args(argv)
    alg = _algorithm(a.algorithm)
    opts = json.loads(a.options)
    programs = {Path(u).stem: alg.trace(Robot(u, name=Path(u).stem), **opts) for u in a.urdfs}
    if a.variants:
        for p in programs.values():
            dump(p, a.batch, a.objective, ks=tuple(a.ks), problems=tuple(a.problems), threshold=a.threshold,
                 sync_cost=a.sync_cost, smem_cost=a.smem_cost, exact_smem=a.exact_smem)
        return
    result = report(a.algorithm, programs, a.units, a.threshold, a.target, a.sync_cost, a.smem_cost,
                    a.batch, a.cores)
    for name, pts in result["strong_scaling"].items():
        print(f"{name}: W={pts[0]['W']} D={pts[0]['D']} W/D={pts[0]['parallelism']:.1f}")
        for pt in pts:
            pct = " ".join(f"{k[:5]}={v:3.0f}%" for k, v in pt["overhead_pct"].items())
            print(f"  p={pt['units']:4d} T_p={pt['T_p']:7.1f} T_ideal={pt['T_ideal']:7.1f} "
                  f"gap={pt['schedule_gap']:5.2f} S={pt['speedup']:5.2f} E={pt['efficiency']:.3f} "
                  f"pT_p/T_s={pt['cost_ratio']:6.2f} | {pct}")
        for b in result["batch"][name]:
            span = {k: v["makespan"] for k, v in b["tiers"].items()}
            print(f"  B={b['B']:6d} best={b['best']:8s} " + " ".join(f"{k}={v:.0f}" for k, v in span.items()))
    print("cost-optimality slope (1 = cost optimal):", result["cost_optimality_slope"])
    print("isoefficiency:", result["isoefficiency"])
    if a.out:
        save(result, a.out)


if __name__ == "__main__":
    main()

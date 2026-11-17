"""Autotuning: pick the tier, group size and scalarisation threshold per robot and algorithm by
measurement, as GRiD's ``config/launch_configs`` does by hand.

The cost model first rejects layouts whose shared scratch would cut residency below
``MIN_WARPS_PER_SM``; the remaining candidates are compiled (in parallel), validated against the
interpreter and timed once. The winner is cached on disk under
(robot hash, algorithm, options, batch, precision, mantis version, nvcc version, GPU architecture).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from . import __version__, cuda, device, harness
from .trace import Program

MIN_WARPS_PER_SM = 8


@dataclass
class LaunchConfig:
    tier: str
    units: int
    threshold: int
    block: int  # threads per block at launch (thread / warp tiers; fixed by units otherwise)
    ms: float
    placement: str = "element"  # defaults so configs cached before them still load
    sync: str = "block"
    problems: int = 1
    fit: bool = False

    @property
    def variant(self) -> cuda.Variant:
        return cuda.Variant(self.tier, self.units, self.threshold, placement=self.placement,
                            sync=self.sync, problems=self.problems, fit=self.fit)


def candidates(p: Program, dtype=np.float32, fit: bool = False) -> list[tuple[cuda.Variant, int]]:
    """(variant, launch block size) pairs that pass the occupancy floor. ``fit`` adds the P = 32
    layouts fitted to shared memory (off by default; see :class:`mantis.cuda.Variant`)."""
    size = np.dtype(dtype).itemsize
    out = [(cuda.Variant("thread"), b) for b in (64, 128, 256)]
    # warp / block at P = 32 (one problem per lane, the warps sharing its stages): every
    # threshold, placement and sync, as barriers vs dataflow depends on the dependency graph.
    # P < 32 (lanes over a problem's elements): threshold 8 and task placement, which won every
    # block case measured. Sized by the emitter, which lays out its own slots and tables.
    layouts = []
    for threshold in (1, 8, 64):
        layouts += [(cuda.Variant("block", 32 * w, threshold, placement=pl, sync=sy, problems=32), 32 * w)
                    for w in cuda.WARPS[1:] for pl in ("element", "task") for sy in ("block", "dataflow")]
        layouts += [(cuda.Variant("warp", 32, threshold, problems=32), b) for b in (64, 128, 256)]
    if fit:  # the same P = 32 layouts fitted to the GPU's shared memory, timed next to the plain ones
        layouts += [(dataclasses.replace(v, fit=True), b) for v, b in list(layouts)]
    for P in cuda.PROBLEMS[:-1]:
        layouts += [(cuda.Variant("warp", 32, problems=P), b) for b in (64, 128, 256)]
        layouts += [(cuda.Variant("block", 32 * w, placement="task", problems=P), 32 * w)
                    for w in cuda.WARPS[1:]]
    sizes: dict = {}
    for v, block in layouts:
        key = (v.tier, v.units, v.threshold, v.placement, v.sync, v.problems, v.fit)
        if key not in sizes:
            sizes[key] = cuda.layout(p, v, np.dtype(dtype).itemsize)
        scr, tab = sizes[key]
        per_block = (tab + scr * (block // 32 if v.tier == "warp" else 1)) * size
        dev = device.target()
        blocks = dev.smem_per_sm // (per_block + dev.smem_reserved_per_block) if per_block else 32
        if per_block <= dev.smem_per_block:
            if blocks * block // 32 >= MIN_WARPS_PER_SM:
                out.append((v, block))
        elif v.problems == 32 and v.sync == "block" and not v.fit:
            # over shared memory: scratch in global memory, whose working set the caches may
            # still hold (baxter rnea_derivatives: 2-3x faster than fitting it into shared)
            out.append((v, block))
    return out


def _vkey(v: cuda.Variant) -> tuple:
    return (v.tier, v.units, v.threshold, v.placement, v.sync, v.problems, v.fit)


def _key(p: Program, batch: int, dtype, options: dict) -> str:
    robot = p.robot
    model = repr([(j.name, j.kind, j.parent, j.q_index, j.multiplier, j.offset, j.E.tobytes(), j.r.tobytes(),
                   j.S.tobytes(), j.I.tobytes()) for j in robot.joints]
                 + sorted((f.name, f.parent, f.T.tobytes()) for f in robot.frames.values()))
    return hashlib.sha1("\x00".join([model, p.name, repr(sorted(options.items())), str(batch),
                                     np.dtype(dtype).name, __version__, harness.nvcc_version(),
                                     " ".join(harness.arch_flags())]).encode()).hexdigest()


def tune(p: Program, batch: int, dtype=np.float32, reps: int = 50, options: dict | None = None,
         refresh: bool = False, prune: int | None = None, fit: bool = False) -> LaunchConfig:
    """The fastest validated layout of ``p`` at ``batch`` problems per launch (cached).
    ``prune``: time only the ``prune`` candidates :mod:`mantis.cost` ranks fastest (plus the
    ones it does not model). Off by default: on baxter rnea_derivatives the model's top pick was
    right at B >= 8192 but 3x off at B <= 1024 (OPEN_TASKS.md). ``fit``: also time the layouts
    fitted to shared memory (:func:`candidates`)."""
    path = harness.cache_dir() / "launch_configs" / f"{_key(p, batch, dtype, options or {})}.json"
    if path.exists() and not refresh:
        return LaunchConfig(**json.loads(path.read_text()))
    cands = candidates(p, dtype, fit)
    variants = list({_vkey(v): v for v, _ in cands}.values())
    with ThreadPoolExecutor() as pool:
        built = dict(zip((_vkey(v) for v in variants),
                         pool.map(lambda v: harness.build([(p, v)], dtype), variants)))
    if prune:
        from . import cost
        est = {id(c): cost.estimate(p, c[0], batch, c[1], built[_vkey(c[0])].resources[0], np.dtype(dtype).itemsize)
               for c in cands}
        modelled = sorted((c for c in cands if est[id(c)] is not None), key=lambda c: est[id(c)].ms)
        cands = modelled[:prune] + [c for c in cands if est[id(c)] is None]
    x = harness.sample_inputs(p, batch)
    best = None
    for v, block in cands:
        b = built[_vkey(v)]
        harness.validate(b)
        _, ms = b.run(0, x, block=block, reps=reps)
        if best is None or ms < best.ms:
            best = LaunchConfig(v.tier, v.units, v.threshold, block, ms, v.placement, v.sync, v.problems, v.fit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(best)))
    return best

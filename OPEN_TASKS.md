# Open tasks: mantis

Things that have been found but not fixed yet. Add an entry when you find one, and delete it
when it's fixed (the fix commit is the record). Format: **title**: symptom / evidence / where /
suggested fix. Items marked _(verify)_ were seeded from earlier session notes and haven't been
re-checked against current code.

## Bugs

- **ptxas miscompiles large straight-line traces** _(verify)_: under CUDA 13.3 at -O2/-O3, the
  G1 ID-gradient thread kernel is silently wrong. The workaround is `-Xptxas -O1` or the
  warp-split tier. A proper fix would split traces past a size threshold, and every build still
  needs numerical validation.

- **GRiM's MPPI IK is capped at 32 particles per seed**: `mppi_ik(n_particles=...)` above 32
  is rejected (`InvalidArgument` from the FFI handler); before, it was silently clamped to 32.
  The cap is `CK_MAX_EVAL` in GRiM's `kernels/ik_mantis/ck.cuh`: a seed's particle costs live
  in a fixed `real err[CK_MAX_EVAL]` / `real costs[CK_MAX_EVAL]`, in registers at the thread tier
  and in the per-seed shared `Work` slot at the warp and block tiers. Suggested fix: make the
  cap a build-time parameter (shared memory sized from it at the cooperative tiers), or stream
  particles through the 32 slots in batches with a running softmax (max-shifted sum) so any
  count works with fixed storage.

## Cooperative tier redesign (in progress)

Layouts explained interactively in `docs/tier_layouts.html` (open in a browser).

- **Warps are the ranks, lanes are SIMD.** The `warp`/`block` tiers make lanes the ranks (`switch`
  on the lane), which SIMT serialises. The redesign: ranks are warps (only they have their own
  instruction stream), and lanes inside a warp do the same work, either on different problems
  (step 1, built on `split`) or on a task's elements (step 3, GLASS-style). Step 2 is
  point-to-point sync with named barriers (`bar.sync id, n`) instead of `__syncthreads`.
  Step 1 so far: `Variant(..., placement="task")` keeps every task on one rank (warp), placed
  on the rank that wrote most of its inputs within LPT's load bound (`schedule.build` step 3).
  Measured (us, B=1, idle A5000; thread / best split element / best split task): RNEA baxter
  17.7 / 34.4 / 16.0; RNEA panda 8.3 / 16.2 / 12.7; rnea_derivatives baxter 197 / 174 / 137
  (B=1024: 281 / 228 / 173); panda 76 / 98 / 84; traj_collision panda 164 / 109 / 109; fetch
  230 / 146 / 155. It halves cross-warp reads and cuts RNEA barriers 40-46%, but trajopt's
  barriers do not drop and its large uneven tasks balance worse, so `placement` is a per-robot
  choice, tuned in `tune.candidates`. The thread tier still wins at B=8192 everywhere.
  Step 2 (named barriers, `Variant(sync="named")`): only the ranks that exchange a value sync
  (`bar.sync id, n` per distinct rank set), with visibility tracked through chains of barriers.
  Correct (bit-identical) but no faster: within +-5% of `__syncthreads` at B=1 on RNEA,
  rnea_derivatives and traj_collision, and up to 1.7x slower at B=8192, because a slot is not
  reused after a partial barrier. The stage-levelled schedule leaves a skipped rank nothing to
  run ahead to; barrier waits are not the cost. Not in `tune.candidates`. Step 2b, if pursued:
  a dataflow schedule (each warp runs its tasks in dependency order, consumers wait on
  producers with `bar.arrive` (producer, non-blocking) / `bar.sync` (consumer), as an MPI
  send/recv) so a warp can run ahead; then decide whether `sync="named"` stays.
  Step 2b (dataflow, `Variant(sync="dataflow")`): no barriers; each warp publishes "finished
  stage s" in a shared progress counter (no wait) and spins only on the producers it reads
  from, plus write-after-read waits where a slot is reused. Correct (bit-identical) but 1.4-2.3x
  slower than `__syncthreads` on RNEA panda/baxter (split4 task, B=1: 13.0 -> 19.0 us). ncu
  (panda, split4 task): barrier stalls (1.7k cycles/warp) are replaced by spin-loop and wait
  stalls (2.1k + 1.9k): the time warps spent at barriers was dependency waiting along the joint
  chain, not imbalance a dataflow schedule removes, and polling costs more than a barrier
  wake. Conclusion of steps 2 and 2b: synchronization is not the cooperative tiers' cost; the
  dependency critical path is. `dataflow` vs `block` is kept as a split tuning choice
  (`tune.candidates`), since another graph shape may favour dataflow; `named` stays tagged for
  removal and untuned.
  Step 3 (lanes over a task's elements, `Variant("uniform", k, placement="task")`): one problem
  per block of k warps; warps are the ranks (tasks placed whole as in step 1, `__syncthreads`
  only where step 4 says a value crosses warps), and inside a warp the lanes run the warp's
  items as lane-uniform shape groups. Bit-identical to thread. Measured (us, B=1; thread / best
  split / best step 3, all best at 8 warps with task placement): traj_collision panda 164 / 109 /
  89.5, fetch 244 / 139 / 97.1; rnea_derivatives baxter 197 / 137 / 117, panda 76 / 82 / 71;
  RNEA panda 8.3 / 12.8 / 20.8, baxter 17.7 / 20.8 / 34.6. So large traces at small batch gain
  1.7-2.5x over thread; RNEA stays thread's. Task placement beat element placement in every
  step 3 case.
  Step 4 (done in mantis, GRiM pointer not yet bumped): one design, `warp` / `block` with P
  problems per warp (`Variant.problems`, 1-32). P = 1 is the step 3 emitter (GLASS's contract,
  GRiM IK's calls unchanged); P = 32 is the former `split` tier's per-lane trace (the `split`
  tier is gone: `Variant("block", 32 * k, problems=32)`); P in between packs problems per warp,
  slots interleaved `s[slot * P + problem]` (contiguous regions were 5x slower at P = 32 from bank
  conflicts). traj_collision panda, block256, us: P=1 82.8 at B=1 and 77.8 at B=8; P=4 85.9 at
  B=64; P=32 125 at B=1024; thread 242 at B=8192: each batch range has its own winner, all
  bit-identical. Removed: the lane-switch warp / block codegen, `Variant.narrow`. Still open:
  bump GRiM's mantis pointer and re-run its IK / dynamics suites (it calls `_warp` / `_block`
  at P = 1 with `_warp_smem` / `_block_smem`, unchanged names); `sync="named"`
  (`REDESIGN(ranks)`) is untuned and can go.
  Step 5 (mantis 0.2.0, GRiM 0.7.0): the lookup tables left the per-problem scratch. They were
  88% of g1's (174 -> 5793 elements per seed vs the old tiers), which cut GRiM IK occupancy and
  made g1 SQP at 1024 seeds 1.65-1.72x slower than the old tiers. The warp / block functions take
  `tables` (once per block, `<alg>_<tier>_tables<T>()` elements, 16-bit indices), filled once per
  launch by `<alg>_<tier>_load_tables(tables, rank, size)`; GRiM IK loads them in `CK_WORK`,
  GRiM dynamics per operation. GRiM's block_threads is now 32 / 64 / 128 / 256, and its
  thread-count test compares one step to the tiers' round-off (FMA contraction differs per warp
  count; bit-identical only under -fmad=false).
  Next: GRiM IK at P = 32 (lanes = seeds, the k warps sharing the geometry, lockstep loops while
  any seed is active, best of all seeds). The mantis sweep says it pays for g1: ik_geometry at
  block k=4-8, P=32 is 1.4-2.0x the thread tier up to B = 4096 (thread spills 2000 B).

- **g1 `ik_geometry` in float32 is ~5-8e-5 off the float64 interpreter** (scaled error), on every
  tier including the thread tier, above the default 2e-5 tolerance: round-off through 29 joints and
  four end-effector log maps. Not a tier bug; decide whether g1-class traces get a looser f32
  tolerance or need a better-conditioned log map.

## Lane-uniform warp / block tiers (P < 32)

- **The P < 32 path of `warp` / `block` keeps prototype shortcuts.** History of
  the single-warp version (RNEA panda B=1): index tables in global memory 76 us; tables copied
  to scratch 54; own affine slot layout 38; straight-line stages 32; shared-typed harness
  scratch 21. 40-60% of operand columns stay non-affine (folding gives related elements
  different shapes), and a two-level pattern `base + s0 * (i % w) + s1 * (i / w)` covers only
  10-30% of the rest, so the index tables are most of its scratch.
  Shortcuts, all tagged `PROTOTYPE(uniform)` in code: inputs staged into scratch; item DAGs
  expanded as trees (recomputation); greedy per-group member order; tables are `static
  __device__` globals in the header, copied into scratch per problem; no `__noinline__`
  budget split; not yet in `tune.candidates`, `DEFAULT_VARIANTS` or `parallel_analysis`.

- **P = 32 stage functions lose the shared address space.** Over the statement budget,
  cooperative stages are `__noinline__` and take scratch as a generic `T*`, so loads are generic,
  not `LDS`. With the stages inlined (budget 10^9), split4 is 4-7% faster and still
  bit-identical (traj_collision panda/fetch, rnea_derivatives baxter). The noinline split
  guards against ptxas miscompiling large traces, so don't just drop it: pass scratch as a
  shared-space offset (or `__cvta_generic_to_shared` plus `ld.shared`) through the stage calls.

- **Pressure fitting: what's left** (landed: `mantis.device` reads the target GPU's shared
  memory and registers from the driver, or from the CUDA guide's table when `MANTIS_ARCH`
  names another architecture; `schedule.build(max_slots=...)` fits P = 32 schedules with ALAP
  levelling, a barrier per stage, width-capped Sethi-Ullman levelling, then recomputation;
  the thread tier's chunks follow Sethi-Ullman order and share scratch slots by linear scan.
  References in CITATIONS.md.) Still open:
  1. **P < 32 (`uniform`) has its own allocator** (`_Emitter.uniform`'s `alloc`/`release`) and
     no budget. Its scratch is slots x P per block, so it is the path that can fit what P = 32
     can't (g1 `rnea_derivatives`: 5091 slots/problem at P = 32, at best 1772 with every lever,
     against 792 that fit in 99 KB; the leanest serial order still needs 1119 live values).
  2. **The warp tier's budget assumes 4 warps per block** (`cuda.WARP_TIER_WARPS_PER_BLOCK`,
     the harness's 128-thread launch). Launches of 256 threads can exceed it; pass the launch
     size to `emit` if that matters.
  3. **Fitting can be slower than not fitting** (measured, baxter `rnea_derivatives`, float,
     idle A5000, median of 7; before = committed `cuda.py`/`schedule.py`). Thread tier (Sethi-
     Ullman chunks + slot reuse): stack 11616 -> 5296 B, spill 648 -> 272 B, 1.05x (B=1),
     1.39x (B=1024), 1.22x (B=65536) faster. Block k=4/8 P=32 fitted (sync_all + width): smem
     68928 -> 25056 elements, but 2-3x slower at every batch than the plain schedule on the
     global-memory fallback (e.g. k=4 B=1 209 -> 563 us): global scratch stays in L1/L2, and
     width levelling adds hundreds of barriers. Warp P=32 fitted (threshold 64 + width 1):
     spill 2268 -> 0, 0.68x at B=1 but 1.29x at B=8192 and 1.70x at B=65536. So fitting is
     opt-in (`Variant(fit=True)`) and `tune.candidates` times fitted copies next to the plain
     ones. `tune` now also times plain layouts over the shared-memory limit (global fallback).
     An L2-resident scratch mode (launch only the blocks whose scratch fits L2's persisting
     carve-out, looping over the batch) was tried and removed: it lost everywhere past B=128
     (baxter block k4 B=65536: 9117 -> 28792 us) by starving the SMs; the plain global
     fallback is fine because each block's scratch is reused within its lifetime and L1/L2
     catch it. Shared-memory fitting is opt-in (`Variant(fit=True)`, `tune(fit=True)`).
     Low priority: re-check fitting on g1.
     Cost model (landed: `mantis.calibrate` measures op latency/issue, barrier cost by warps,
     latency and throughput per level, DRAM bandwidth per GPU; `mantis.cost.estimate` is
     occupancy -> waves -> per-block compute + barriers + memory (dependent rounds x latency +
     accesses x throughput, at the level each block's working set lands); `schedule.build(cost=)`
     picks the cheapest fitting lever; `tune(prune=N)` optional). Choosing levers by the model
     made the fitted block layouts 1.4-1.6x faster (k8 B=65536 20809 -> 13941 us) but still
     slower than plain. Model accuracy on baxter: right winner at B >= 8192 (Spearman +0.68);
     at B <= 1024 it picks fitted block layouts measured ~2x slower than the best; global/L2
     layouts modelled ~2-3x high, shared-fitted ~0.4-0.6x low. Next: per-stage overhead in
     narrow stages (rank imbalance, branch), ILP-limited issue at low occupancy; then validate
     on a second trace before turning `prune` on.
  4. **Registers:** every listed architecture allows 255 per thread, so ptxas's spill report is
     already the device's budget and `harness.fit` is unchanged. Targeting an occupancy
     (`Device.regs_for`, `__launch_bounds__`) instead of zero spills is untried.

## Features

- **AD follow-ups** (`mantis.ad`, `trace(..., order=k, wrt=..., mode=...)`, landed): order 2
  is checked against finite differences of order 1, not yet against Pinocchio's second-order
  RNEA derivatives (`pin.computeRNEASecondOrderDerivatives`). Op counts (panda, live scalars):
  rnea 1155, hand-written `rnea_derivatives` 10356, AD order 1 forward 11205 / reverse 9150,
  order 2 forward 62737 / reverse 69374. Still to measure: `tune.py` time of AD vs
  `rnea_derivatives` on panda and g1, and trace time at order 2 on g1 (Python sweeps with
  dict tangents; 0.7 s on panda). One `mode` covers every order; forward-over-reverse
  (reverse for order 1, forward above) needs a per-order mode list. Reverse mode sweeps once
  per output entry, so it suits few outputs.

- **Structural sparsity: remaining gaps** (landed: `_cholesky_solve` factors in minimum-degree
  order on the trace-time zero pattern, `_elimination_order`. Solve ops after crba: g1 2717 ->
  1492, h1_2 8752 -> 2451, h2_plus 93682 -> 10963; chains and baxter (block-diagonal M)
  unchanged. Symmetric derivative tensors are emitted once by `mantis.ad`.) Still open:
  1. **Sparsity across a boundary the trace doesn't cover.** Solves whose size grows with the
     robot go to GLASS, which is dense. Trajopt's horizon loop lives in the CUDA caller, so the
     block-tridiagonal KKT structure across timesteps is invisible to the trace. Fixes: pass the
     trace-time zero pattern to a sparse/blocked GLASS routine, or trace one stage's blocks and
     have the caller run a block-tridiagonal (Riccati-style) solve over them.
  2. **Recovering blocks after scalarization.** A repeated 6x6 block pattern per joint is just
     many separate statements. Grouping identical-pattern blocks (joint tag plus a structural
     hash) would let the warp/block tiers give one block per shape group (lanes run the same
     instruction) instead of scheduling scattered scalars.
  3. **No algorithm uses an ordered sparse solve yet.** `fd_crba`-style forward dynamics through
     the traced `_cholesky_solve` (instead of GLASS) would now be sparse on humanoids; measure
     it against ABA and GRiD.

## Environment

- **NVRTC / driver version skew** _(verify)_: the NVRTC from conda can be newer than the driver,
  and the conda PATH can shadow the system nvcc.

## Branches

- **`benchmarks` branch has library changes `main` lacks.** The suite itself moved to GRiM
  (`benchmarks/`), but the IK work (`ik_geometry`, `analytic_ik_*`, SQP/LS traces, `rot_joint`,
  continuous joints, `tests/test_oracle.py`) landed only on the `benchmarks` branch. Cherry-pick
  it onto `main` and retire the branch. Until then GRiM pins its mantis submodule to `benchmarks`
  (`.gitmodules`).

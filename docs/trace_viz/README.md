# mantis trace graph

An interactive look at what mantis does to one algorithm on one robot. Open `index.html` in a browser
(double-click it; no server or install needed).

1. **Trace replay.** The real trace, replayed call by call: each node and scalar as it is recorded,
   and each call that folds, drops a zero product, simplifies or merges as it happens; then dead-output
   elimination. Expand nodes into their scalar operations.
2. **From graph to schedule.** `schedule.build` for a chosen tier, k warps, P problems per warp and
   placement: warps over stages (zoomable, statements labelled), the lanes of each warp, and the values
   held in shared memory, with scratch and table sizes from `cuda.layout`.
3. **Parallel analysis.** Grama et al.'s metrics (W, D, p, T_p, Brent's bound, S, E, cost, overhead
   breakdown, strong scaling) for that layout, with a legend of symbols and their ideals.

Times in 2 and 3 are a statement-count model (statements x passes), not measurements.

## Files

| | |
|---|---|
| `index.html` | the visualizer |
| `web/` | its data, one script per robot x algorithm (generated) |
| `mantis_trace_viz.npz` | all the data for NumPy; `npz/<robot>.npz` per robot |
| `build.sh` | regenerates everything from the mantis working tree |
| `export_all.py` | traces each algorithm and writes `npz/<robot>.npz`; its docstring documents the array layout |
| `bundle.py`, `npz2json.py`, `make_offline.py` | combined .npz + index, JSON copy, file:// scripts |
| `zip.sh` | packages this folder as `mantis_trace_viz.zip` at the repo root (run by `build.sh`) |

```python
z = np.load("mantis_trace_viz.npz")
meta = json.loads(z["baxter/meta"].tobytes())      # tables, algorithms, W / D, shape labels, ...
events = z["baxter/traj_collision.events"]          # (E, 4): kind, target, node, scalar count
```

`zip.sh` (run by `build.sh`) writes `mantis_trace_viz.zip` at the repo root (this folder without `npz/`), the copy
to send to people without the repo.

## Coverage

Robots: `resources/` (panda, fr3, ur5, baxter, fetch, pr2) and GRiD's g1 (`grid_g1`). Every
registered algorithm except `analytic_ik_candidates` / `analytic_ik_chain` (they need GRiM's 7-DOF
Franka arm geometry). Algorithms the tests give no options get the ones in `export_all.trace_options`.
A robot or algorithm that fails to trace is listed under `skipped` in its metadata (e.g. ABA on
mimic joints).

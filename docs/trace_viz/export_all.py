"""Data for the mantis trace-graph page, for every robot and algorithm, as .npz.

Per robot: <out>/<robot>.npz. Keys are "<algorithm>.<field>"; "meta" is UTF-8 JSON as uint8
(json.loads(z["meta"].tobytes())). Run from the mantis repo with PYTHONPATH=python:tests.

  python export_all.py <out_dir> [robot ...]

Graph (the real trace, replayed call by call):
  nodes        int32 (n, 10)  op, type, level, joint, line, created, live, pass, depth, subtree (indices into meta)
  node_args    int32 flat + node_args_off (n + 1)     input nodes
  elems        int32 flat + elems_off (n + 1)         scalar ids of each node's value
  s_op, s_owner int32 (S); s_args flat + s_args_off; s_live uint8 (S)
  events       int32 (E, 4)   kind, target, node being expanded, scalar count after the call
               kinds: 0 new scalar, 1 rewritten, 2 folded, 3 zero product, 4 identity, 5 merged scalar, 6 new node, 7 merged node
Schedule (mantis.schedule.build, threshold 8, __syncthreads):
  items        int32 (m, 5)   scalar, task, cost, stage, shape; item_leaves flat + item_leaves_off
  node_task    int32 (n); task_stage int32 (2, T)
  sched.<k>.<placement>.rank int16 (m), .sync int32, .layout int32 (6, 2) scratch / table elements per P
                 (P = 1, 2, 4, 8, 16, 32; -1 where code generation refuses the layout)
"""
import hashlib, importlib, inspect, json, os, sys, time
from multiprocessing import Pool

import numpy as np

import mantis
from mantis import cuda, ir, parallel_analysis as pa, schedule as sm
from mantis.trace import ALGORITHMS
from robots import robot, options, sphere_set, traj_set

T = importlib.import_module("mantis.trace")
sys.setrecursionlimit(200000)
ROBOTS = {"baxter": "baxter", "panda": "panda", "fr3": "fr3", "ur5": "ur5", "fetch": "fetch", "pr2": "pr2", "g1": "grid_g1"}
SKIP = {"analytic_ik_candidates": "needs GRiM's 7-DOF Franka arm geometry", "analytic_ik_chain": "needs GRiM's 7-DOF Franka arm geometry"}
KS, PLACEMENTS, THRESHOLD = (1, 2, 4, 8), ("task", "element"), 8


def trace_options(r, a):
    o = options(r, a)
    deep = max(r.joints, key=lambda j: j.depth).name
    if a in ("ik_pose_cost", "ik_ls_step", "ik_ls_system"): o = {"frame": deep}
    if a in ("ik_ls_solve", "ik_sqp_system", "ik_masked_solve"): o = {"n": r.n_q}
    if a in ("ik_geometry", "ik_geometry_value"):
        o = {"frames": (deep,), "cols": tuple(range(r.n_q)), "wspheres": tuple((f, c) for f, c, _ in sphere_set(r))}
    if a in ("traj_spheres", "traj_backprop"): o = {"rows": traj_set(r)["rows"]}
    return o


# ---- recording hooks: source file in the line tag, and a log of every call ----
def record(self, op, cls, args, expand, key=None):
    f = inspect.currentframe().f_back
    while f and f.f_code.co_filename == T.__file__: f = f.f_back
    tags = dict(self.tags, line=f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f else "")
    return cls(self, self.g.node(op, cls.__name__, tuple(a.node for a in args), expand, tags, key))
T.Tracer.record = record
LOG, DEPTH = [], [0]
_s, _node = ir.Graph.s, ir.Graph.node
def s(self, op, *args):
    DEPTH[0] += 1; n0 = len(self.op)
    try: r = _s(self, op, *args)
    finally: DEPTH[0] -= 1
    if DEPTH[0] == 0:
        if self.op[r] == "const": kind = 3 if r == 0 and not all(self.is_const(a) for a in args) else 2
        elif r >= n0: kind = 0 if self.op[r] == op else 1
        elif self.op[r] == op and tuple(sorted(self.args[r])) == tuple(sorted(args)): kind = 5
        else: kind = 4
        LOG.append((kind, int(r), int(self._current), len(self.op)))
    return r
def node(self, op, type_, args, expand, tags, key=None):
    n0 = len(self.nodes)
    nid = _node(self, op, type_, args, expand, tags, key)
    LOG.append((6 if nid >= n0 else 7, int(nid), int(self._current), len(self.op)))
    return nid
ir.Graph.s, ir.Graph.node = s, node


def flat(lists):
    off = np.zeros(len(lists) + 1, np.int32)
    off[1:] = np.cumsum([len(x) for x in lists])
    return np.fromiter((v for x in lists for v in x), np.int32, int(off[-1])), off


def export_robot(name):
    t0 = time.time()
    r = robot(ROBOTS[name])
    arrays, meta = {}, {"robot": name, "urdf": ROBOTS[name], "n_q": r.n_q, "algorithms": [], "skipped": dict(SKIP), "graph": {}, "sched": {}}
    tables = {"ops": [], "types": [], "joints": [], "lines": [], "src": []}
    def ix(kind, v):
        lst = tables[kind]
        if v not in lst: lst.append(v)
        return lst.index(v)
    line_of = {}
    for a in ALGORITHMS:
        if a in SKIP: continue
        LOG.clear()
        try: p = mantis.trace(r, a, **trace_options(r, a))
        except Exception as e:
            meta["skipped"][a] = f"{type(e).__name__}: {e}"[:200]; continue
        g = p.g; live = g.live(p.flat_outputs); lvl = []; rows = []
        for n in g.nodes:
            lvl.append(1 + max((lvl[x] for x in n.args), default=-1))
            t = n.tags; ln = t.get("line", "")
            if ln and ln not in line_of:
                fn, no = ln.split(":"); path = os.path.join("python/mantis", fn)
                line_of[ln] = len(tables["lines"]); tables["lines"].append(ln)
                tables["src"].append(open(path).read().split("\n")[int(no) - 1].strip() if os.path.exists(path) else "")
            j = t.get("joint")
            rows.append([ix("ops", n.op), ix("types", n.type), lvl[-1], -1 if j is None else ix("joints", str(j)),
                         line_of[ln] if ln else -1, len(n.created), int(sum(live[x] for x in n.created)),
                         {"forward": 0, "backward": 1}.get(t.get("pass"), 2),
                         -1 if t.get("depth") is None else int(t["depth"]), -1 if t.get("subtree") is None else int(t["subtree"])])
        P_ = a + "."
        arrays[P_ + "nodes"] = np.asarray(rows, np.int32).reshape(-1, 10)
        arrays[P_ + "node_args"], arrays[P_ + "node_args_off"] = flat([[int(x) for x in n.args] for n in g.nodes])
        arrays[P_ + "elems"], arrays[P_ + "elems_off"] = flat([[int(x) for x in n.elems.ravel()] for n in g.nodes])
        arrays[P_ + "s_op"] = np.asarray([ix("ops", o) for o in g.op], np.int32)
        arrays[P_ + "s_owner"] = np.asarray(g.owner, np.int32)
        arrays[P_ + "s_args"], arrays[P_ + "s_args_off"] = flat([[int(y) for y in x] for x in g.args])
        arrays[P_ + "s_live"] = live.astype(np.uint8)
        arrays[P_ + "events"] = np.asarray(LOG, np.int32).reshape(-1, 4)
        W, D = pa.work_span(p)
        meta["graph"][a] = {"scalars": len(g.op), "live": int(live.sum()), "W": W, "D": D}
        meta["algorithms"].append(a)
        # ---- schedule ----
        try:
            sc0 = sm.build(p, 1, THRESHOLD); task_of = sc0.task_of
            users = {}
            for x in range(len(g.op)):
                if not live[x] or g.owner[x] < 0: continue
                for y in g.args[x]:
                    o = g.owner[y]
                    if o >= 0 and o != g.owner[x]: users.setdefault(o, set()).add(g.owner[x])
            tasks = set(task_of.values())
            def node_task(n):
                for _ in range(1000):
                    if n in tasks: return n
                    nxt = users.get(n)
                    if not nxt: return -1
                    n = min(nxt)
                return -1
            arrays[P_ + "node_task"] = np.asarray([node_task(n) for n in range(len(g.nodes))], np.int32)
            def tree(x, task):
                op = g.op[x]
                if op == "const": return ("c", g.value[x])
                if op == "in" or task_of.get(x, task) != task: return ("v", x)
                kids = [tree(y, task) for y in g.args[x]]
                if op in cuda._COMMUTATIVE: kids.sort(key=lambda t: repr(cuda._shape(t)))
                return (op, kids)
            def lab(t, depth):
                if t[0] == "c": return "c"
                if t[0] == "v": return "x"
                if depth == 0 or not t[1]: return t[0]
                return t[0] + "(" + ", ".join(lab(k, depth - 1) for k in t[1]) + ")"
            shapes, labels = {}, []
            def shape_id(it):
                if it.task < 0: return -1
                tr = tree(it.scalar, it.task)
                key = hashlib.md5(repr(cuda._shape(tr)).encode()).hexdigest()
                if key not in shapes: shapes[key] = len(shapes); labels.append(lab(tr, 2))
                return shapes[key]
            items = sc0.items
            order = {it.scalar: i for i, it in enumerate(items)}
            arrays[P_ + "items"] = np.asarray([[it.scalar, it.task, it.cost, it.stage, shape_id(it)] for it in items], np.int32).reshape(-1, 5)
            arrays[P_ + "item_leaves"], arrays[P_ + "item_leaves_off"] = flat([[order[l] for l in sorted(it.leaves) if l in order] for it in items])
            ts = {}
            for it in items:
                if it.task >= 0: ts[it.task] = it.stage
            arrays[P_ + "task_stage"] = np.asarray([list(ts), list(ts.values())], np.int32).reshape(2, -1)
            refused = {}
            for k in KS:
                for pl in PLACEMENTS:
                    sc = sm.build(p, k, THRESHOLD, placement=pl)
                    rank = {it.scalar: it.rank for it in sc.items}
                    lay = []
                    for P in cuda.PROBLEMS:
                        try: lay.append(list(cuda.layout(p, cuda.Variant("warp" if k == 1 else "block", units=32 * k, threshold=THRESHOLD, placement=pl, problems=P))))
                        except ValueError as e: lay.append([-1, -1]); refused[f"{k},{pl},{P}"] = str(e)[:160]
                    q = f"{P_}sched.{k}.{pl}."
                    arrays[q + "rank"] = np.asarray([rank[it.scalar] for it in items], np.int16)
                    arrays[q + "sync"] = np.asarray(sorted(sc.sync_before), np.int32)
                    arrays[q + "layout"] = np.asarray(lay, np.int32)
            meta["sched"][a] = {"stages": sc0.stages, "tasks": len(tasks), "shape_label": labels, "threshold": THRESHOLD, "refused": refused}
        except Exception as e:
            meta["sched"][a] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print(f"{name} {a}: {len(g.nodes)} nodes, {len(g.op)} scalars, {time.time() - t0:.0f}s", flush=True)
    meta.update(tables)
    arrays["meta"] = np.frombuffer(json.dumps(meta, separators=(",", ":")).encode(), np.uint8)
    np.savez_compressed(f"{OUT}/{name}.npz", **arrays)
    return name, time.time() - t0, os.path.getsize(f"{OUT}/{name}.npz")


if __name__ == "__main__":
    OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    names = sys.argv[2:] or list(ROBOTS)
    with Pool(len(names)) as pool:
        for name, secs, size in pool.imap_unordered(export_robot, names):
            print(f"DONE {name}: {secs:.0f}s, {size / 1e6:.1f} MB", flush=True)
else:
    OUT = os.environ.get("MANTIS_VIZ_OUT", ".")

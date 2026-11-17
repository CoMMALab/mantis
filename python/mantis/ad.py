"""Ahead-of-time differentiation of a traced program: a source transform on the SSA graph.

:func:`derivative` returns a new :class:`Program` whose outputs are the original outputs and
their derivatives up to ``order`` with respect to the inputs named in ``wrt``. The derivative
scalars are built in the same hash-consed graph, so folding, sparsity, CSE and dead-output
elimination act on primal and derivative together: structural zeros of a Jacobian or Hessian
fold away, and a tangent that equals an existing expression is that expression. The result is
straight-line code like any trace: no tape, no dual numbers at runtime. The cost is that a
derivative kernel is compiled ahead of time; nothing differentiates an existing kernel.

Each order is one step, forward (tangents for every direction, input scalars first) or reverse
(one adjoint sweep per output entry, outputs first). Order ``k`` of output ``y`` (``m``
entries, ``x`` = the ``wrt`` inputs concatenated, ``n`` long) is ``d{k}_y``: for each entry of
``y``, the entries ``d^k y_i / dx_j1 ... dx_jk`` for ``j1 <= ... <= jk`` in lexicographic
order, so a symmetric tensor is emitted once.

Derivative scalars are owned by derivative nodes that copy the primal node's tags (``pass`` =
``"backward"`` for adjoints), so the scheduler sees ordinary tasks. Node order stays
topological: a derivative node is reused only if it is not older than anything it reads.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from .ir import Graph, Node
from .trace import Program


class _Owners:
    """The derivative node that new scalars for primal node ``n`` go into. A new node is opened
    when a scalar would read a newer node (``Graph.relocate``), keeping node order topological."""

    def __init__(self, g: Graph, prefix: str, tags: dict):
        self.g, self.prefix, self.tags, self.cur, self.n = g, prefix, tags, {}, -1

    def _open(self, floor: int = -1) -> int:
        g, nd = self.g, self.g.nodes[self.n]
        d = self.cur[self.n] = len(g.nodes)
        g.nodes.append(Node(f"{self.prefix}{nd.op}", nd.type, (self.n,), np.zeros(0, np.int64),
                            {**nd.tags, **self.tags}))
        return d

    def enter(self, n: int) -> None:
        """New scalars from here on belong to primal node ``n``'s derivative."""
        self.n = n
        if n < 0:  # constants / inputs: their derivative terms fold to constants anyway
            self.g._current = -1
            return
        self.g._current = self.cur[n] if n in self.cur else self._open()

    def __enter__(self):
        self.prev = self.g._current
        self.g.relocate = self._open
        return self

    def __exit__(self, *exc):
        self.g.relocate, self.g._current = None, self.prev


def _sign(g: Graph, a: int) -> int:
    return g.s("sub", g.s("lt", 0, a), g.s("lt", a, 0))


def _partials(g: Graph, s: int) -> list[int]:
    """``d s / d arg`` for each argument of scalar ``s`` (subgradients at kinks of min/max/abs)."""
    op, a = g.op[s], g.args[s]
    if op == "add":
        return [1, 1]
    if op == "sub":
        return [1, g.const(-1.0)]
    if op == "mul":
        return [a[1], a[0]]
    if op == "div":  # d/da = 1/b, d/db = -s/b
        inv = g.s("div", 1, a[1])
        return [inv, g.s("neg", g.s("mul", s, inv))]
    if op == "neg":
        return [g.const(-1.0)]
    if op == "sin":
        return [g.s("cos", a[0])]
    if op == "cos":
        return [g.s("neg", g.s("sin", a[0]))]
    if op == "sqrt":
        return [g.s("div", g.const(0.5), s)]
    if op == "exp":
        return [s]
    if op == "log":
        return [g.s("div", 1, a[0])]
    if op == "abs":
        return [_sign(g, a[0])]
    if op == "atan2":  # atan2(y, x): (x, -y) / (x^2 + y^2)
        y, x = a
        r2 = g.s("add", g.s("mul", x, x), g.s("mul", y, y))
        return [g.s("div", x, r2), g.s("neg", g.s("div", y, r2))]
    if op in ("max", "min"):  # args are sorted; picks a[0] at a tie
        w = g.s("lt", a[0], a[1]) if op == "max" else g.s("lt", a[1], a[0])
        return [g.s("sub", 1, w), w]
    if op == "lt":
        return [0, 0]
    raise NotImplementedError(f"no derivative rule for scalar op {op!r}")


def _forward(g: Graph, roots: list[int], wrt: list[int]) -> dict[int, dict[int, int]]:
    """Tangents ``T[s][j] = d s / d x_j`` (zeros omitted) for every scalar ``roots`` depend on."""
    col = {s: j for j, s in enumerate(wrt)}
    T: dict[int, dict[int, int]] = {}
    with _Owners(g, "d_", {}) as own:
        _forward_sweep(g, roots, col, own, T)
    return T


def _forward_sweep(g, roots, col, own, T):
    for s in np.flatnonzero(g.live(roots)):  # ids are topological
        s = int(s)
        op = g.op[s]
        if op in ("const", "in"):
            T[s] = {col[s]: 1} if s in col else {}
            continue
        args = g.args[s]
        if not any(T[a] for a in args):
            T[s] = {}
            continue
        own.enter(g.owner[s])
        dp = _partials(g, s)
        t: dict[int, int] = {}
        for a, p in zip(args, dp):
            for j, ta in T[a].items():
                t[j] = g.s("add", t.get(j, 0), g.s("mul", p, ta))
        T[s] = {j: v for j, v in t.items() if v != 0}


def _reverse(g: Graph, root: int, wrt: list[int], own: _Owners) -> list[int]:
    """``d root / d x_j`` for each ``j``: one adjoint sweep, latest scalar first."""
    A = {root: 1}
    for s in np.flatnonzero(g.live([root]))[::-1]:
        s = int(s)
        if g.op[s] in ("const", "in") or A.get(s, 0) == 0:
            continue
        own.enter(g.owner[s])
        for a, p in zip(g.args[s], _partials(g, s)):
            if g.op[a] != "const":
                A[a] = g.s("add", A.get(a, 0), g.s("mul", A[s], p))
    return [A.get(x, 0) for x in wrt]


def derivative(program: Program, order: int = 1, wrt: tuple[str, ...] | str = ("q",),
               mode: str = "forward") -> Program:
    """``program`` plus its derivatives of orders 1..``order`` wrt the inputs ``wrt``
    (``mode`` "forward" or "reverse", used for every order). See the module docstring."""
    if mode not in ("forward", "reverse"):
        raise ValueError(f"mode {mode!r}: 'forward' or 'reverse'")
    wrt = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    sizes = dict(program.inputs)
    missing = [w for w in wrt if w not in sizes]
    if missing:
        raise ValueError(f"{program.name} has no input(s) {missing}; inputs are {list(sizes)}")
    g = program.g
    x = [g.input(w, i) for w in wrt for i in range(sizes[w])]
    n = len(x)
    outputs = list(program.outputs)
    level = [(name, [(int(s), ()) for s in ids]) for name, ids in program.outputs]
    own = _Owners(g, "adj_", {"pass": "backward"})
    for k in range(1, order + 1):
        if mode == "forward":
            T = _forward(g, [s for _, es in level for s, _ in es], x)
            grad = lambda s: [T[s].get(j, 0) for j in range(n)]  # noqa: E731
        else:
            grad = lambda s: _reverse(g, s, x, own)  # noqa: E731
        nxt = []
        with own:
            for name, es in level:
                new = []
                for s, idx in es:
                    gs = grad(s)
                    new += [(gs[j], idx + (j,)) for j in range(idx[-1] if idx else 0, n)]
                nxt.append((name, new))
        level = nxt
        outputs += [(f"d{k}_{name}", np.array([s for s, _ in es], np.int64)) for name, es in level]
    return Program(f"{program.name}_d{order}" if order else program.name, program.robot, g,
                   program.inputs, outputs)


def indices(n: int, order: int) -> list[tuple[int, ...]]:
    """The derivative index tuples of one output entry, in ``d{order}_y`` order."""
    return list(combinations_with_replacement(range(n), order))

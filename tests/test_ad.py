"""Ahead-of-time AD (``mantis.ad``) and the trace-time elimination order of ``_cholesky_solve``,
on the interpreter (float64, CPU). Compiled tiers of derivative programs are in test_cuda.py."""

from __future__ import annotations

import numpy as np
import pytest

import mantis
from mantis import algorithms as alg
from mantis.ad import indices
from robots import inputs, robot


def _x(r, batch=3):
    x = inputs(r, batch=batch)
    return {k: x[k] for k in ("q", "qd", "qdd")}


def _fd(f, x, wrt, h=1e-6):
    """Central differences of ``f(x) -> (B, m)`` over the concatenated ``wrt`` inputs: (B, m, n)."""
    cols = []
    for w in wrt:
        for c in range(x[w].shape[1]):
            lo, hi = {k: v.copy() for k, v in x.items()}, {k: v.copy() for k, v in x.items()}
            lo[w][:, c] -= h
            hi[w][:, c] += h
            cols.append((f(hi) - f(lo)) / (2 * h))
    return np.stack(cols, axis=-1)


@pytest.mark.parametrize("robot_name", ["panda", "grid_go2"])
def test_rnea_order1_matches_analytical(robot_name):
    """Forward and reverse AD of rnea against the hand-written rnea_derivatives."""
    r, n = robot(robot_name), robot(robot_name).n_q
    x = _x(r)
    ref = mantis.trace(r, "rnea_derivatives").evaluate(**x)
    want = np.concatenate([ref["dtau_dq"].reshape(-1, n, n), ref["dtau_dqd"].reshape(-1, n, n)], 1)
    want = want.transpose(0, 2, 1)  # (B, tau, [q; qd])
    for mode in ("forward", "reverse"):
        p = mantis.trace(r, "rnea", order=1, wrt=("q", "qd"), ad_mode=mode)
        out = p.evaluate(**x)
        np.testing.assert_allclose(out["tau"], mantis.trace(r, "rnea").evaluate(**x)["tau"], atol=1e-12)
        np.testing.assert_allclose(out["d1_tau"].reshape(-1, n, 2 * n), want, atol=1e-10, err_msg=mode)


def test_rnea_order2_matches_finite_differences():
    """Order 2 (upper triangle only) against central differences of order 1; modes agree."""
    r, n = robot("panda"), robot("panda").n_q
    x = _x(r)
    d1 = mantis.trace(r, "rnea", order=1, wrt=("q", "qd"))
    fd = _fd(lambda z: d1.evaluate(**z)["d1_tau"], x, ("q", "qd")).reshape(-1, n, 2 * n, 2 * n)
    want = np.stack([fd[:, :, i, j] for i, j in indices(2 * n, 2)], -1).reshape(len(fd), -1)
    got = {m: mantis.trace(r, "rnea", order=2, wrt=("q", "qd"), ad_mode=m).evaluate(**x)["d2_tau"]
           for m in ("forward", "reverse")}
    assert got["forward"].shape[1] == n * len(indices(2 * n, 2))
    np.testing.assert_allclose(got["forward"], want, atol=1e-6 * max(1.0, np.abs(want).max()))
    np.testing.assert_allclose(got["forward"], got["reverse"], atol=1e-10)


def test_vector_function_changes_dimension():
    """frame_jacobian: R^n -> R^(6 n); its derivative is (6 n) x n."""
    r = robot("ur5")
    frame = max(r.joints, key=lambda j: j.depth).name
    x = {"q": inputs(r)["q"]}
    base = mantis.trace(r, "frame_jacobian", frame=frame)
    want = _fd(lambda z: base.evaluate(**z)[base.outputs[-1][0]], x, ("q",))
    for mode in ("forward", "reverse"):
        p = mantis.trace(r, "frame_jacobian", frame=frame, order=1, ad_mode=mode)
        np.testing.assert_allclose(p.evaluate(**x)["d1_" + base.outputs[-1][0]].reshape(want.shape), want, atol=1e-6)


def test_scalar_function_with_kinks():
    """ik_pose_cost (scalar, through atan2 / sqrt / max): AD against finite differences."""
    r = robot("panda")
    frame = max(r.joints, key=lambda j: j.depth).name
    rng = np.random.default_rng(1)
    t = rng.normal(size=(3, 7))
    t[:, :4] /= np.linalg.norm(t[:, :4], axis=1, keepdims=True)
    x = {"q": inputs(r)["q"][:3], "target": t}
    base = mantis.trace(r, "ik_pose_cost", frame=frame)
    key = base.outputs[0][0]
    want = _fd(lambda z: base.evaluate(**z)[key], x, ("q",))[:, 0]
    for mode in ("forward", "reverse"):
        got = mantis.trace(r, "ik_pose_cost", frame=frame, order=1, ad_mode=mode).evaluate(**x)[f"d1_{key}"]
        np.testing.assert_allclose(got, want, atol=1e-5 * max(1.0, np.abs(want).max()))


def test_derivative_shares_structural_zeros():
    """Folding runs on the derivative: tau_i does not depend on qd_j for j outside i's tree path,
    and those entries are the constant 0, not computed zeros."""
    r = robot("baxter")
    p = mantis.trace(r, "rnea", order=1, wrt=("qd",))
    zeros = int((p.outputs[-1][1] == 0).sum())
    assert zeros > 0


@mantis.algorithm(inputs=("q", ("b", lambda o: o["n"])), outputs=("x",))
def _crba_solve(robot, q, b, n=0):
    m = robot.n_q
    M = [[None] * m for _ in range(m)]
    flat = alg.crba(robot, q)
    for c in range(m):
        for rr in range(m):
            M[rr][c] = flat[c * m + rr]
    return (alg._cholesky_solve(M, b),)


def test_elimination_order_avoids_fill_on_branched_tree():
    """g1's legs and arms hang off a shared waist: the natural order fills the cross-branch blocks
    of L, the minimum-degree order does not (1492 vs 2717 solve ops). It still solves M x = b.
    (baxter's arms share no movable joint, so its M is block diagonal and neither order fills.)"""
    r = robot("grid_g1")
    x = inputs(r)
    b = np.random.default_rng(2).normal(size=(len(x["q"]), r.n_q))
    p = mantis.trace(r, "_crba_solve", n=r.n_q)
    M = mantis.trace(r, "crba").evaluate(q=x["q"])["M"].reshape(-1, r.n_q, r.n_q)
    got = p.evaluate(q=x["q"], b=b)["x"]
    np.testing.assert_allclose(np.einsum("bij,bj->bi", M, got), b, atol=1e-8)

    natural = alg._cholesky_solve
    alg._cholesky_solve = alg._cholesky_solve_natural
    try:
        q = mantis.trace(r, "_crba_solve", n=r.n_q)
    finally:
        alg._cholesky_solve = natural
    live = lambda prog: int(prog.g.live(prog.flat_outputs).sum())  # noqa: E731
    assert live(p) < live(q)


def test_elimination_order_keeps_dense_natural():
    """A dense pattern keeps the natural order, so existing dense solves are unchanged."""
    assert alg._elimination_order([[1.0] * 5 for _ in range(5)]) == list(range(5))


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_node_order_stays_topological(mode):
    """The scheduler levels nodes in index order: no node may read a newer one."""
    p = mantis.trace(robot("panda"), "rnea", order=2, wrt=("q", "qd"), ad_mode=mode)
    g = p.g
    for n, nd in enumerate(g.nodes):
        for s in nd.created:
            assert all(g.owner[a] <= n for a in g.args[s]), (n, nd.op)

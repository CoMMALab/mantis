"""On-chip pressure: the target GPU's budget (``mantis.device``), the thread tier's chunk order
and scratch reuse, and ``schedule.build`` fitting a shared-memory budget. CPU only except the
live-device check; the compiled tiers of fitted schedules are in test_cuda.py."""

from __future__ import annotations

import pytest

import mantis
from mantis import cuda, device, schedule
from robots import options, robot


def _body(p):
    g = p.g
    live = g.live(p.flat_outputs)
    return [s for s in range(len(g.op)) if live[s] and g.op[s] not in ("const", "in")]


def _peak_live(g, order):
    pos = {s: i for i, s in enumerate(order)}
    last = {}
    for s in order:
        for a in g.args[s]:
            if a in pos:
                last[a] = pos[s]
    ev = [0] * (len(order) + 1)
    for s, i in pos.items():
        if s in last:
            ev[i] += 1
            ev[last[s]] -= 1
    cur = peak = 0
    for e in ev:
        cur += e
        peak = max(peak, cur)
    return peak


def test_device_table_matches_live_gpu():
    """The table used when cross-compiling agrees with the driver on this machine's GPU."""
    try:
        d = device.live()
    except RuntimeError:
        pytest.skip("no CUDA driver")
    t = device.from_table(d.cc)
    assert (d.smem_per_sm, d.smem_per_block, d.smem_reserved_per_block, d.regs_per_sm) == \
           (t.smem_per_sm, t.smem_per_block, t.smem_reserved_per_block, t.regs_per_sm)
    assert d.l2 > 0 and d.blocks_per_sm > 0


@pytest.mark.parametrize("robot_name,name", [("grid_g1", "rnea_derivatives"), ("grid_g1", "sphere_collision"),
                                             ("baxter", "rnea")])
def test_sethi_ullman_order_is_topological_and_leaner(robot_name, name):
    r = robot(robot_name)
    p = mantis.trace(r, name, **options(r, name))
    body = _body(p)
    order = cuda.sethi_ullman_order(p.g, body, p.flat_outputs)
    assert sorted(order) == sorted(body)
    pos = {s: i for i, s in enumerate(order)}
    assert all(pos[a] < pos[s] for s in order for a in p.g.args[s] if a in pos)
    assert _peak_live(p.g, order) < _peak_live(p.g, body)


def test_chunk_slots_reuse_and_never_overlap():
    """A slot is reused only after every read of its previous value."""
    r = robot("grid_g1")
    p = mantis.trace(r, "rnea_derivatives")
    g, body = p.g, _body(p)
    chunks = [body[i:i + 2000] for i in range(0, len(body), 2000)]
    chunk_of = {s: k for k, ch in enumerate(chunks) for s in ch}
    slots = cuda.chunk_slots(g, chunks, chunk_of)
    last = {}
    for k, ch in enumerate(chunks):
        for s in ch:
            for a in g.args[s]:
                if a in chunk_of and chunk_of[a] != k:
                    last[a] = max(last.get(a, -1), k)
    assert set(slots) == set(last)
    assert max(slots.values()) + 1 < len(slots)  # reuse happened
    by_slot: dict[int, list[tuple[int, int]]] = {}
    for s, sl in slots.items():
        by_slot.setdefault(sl, []).append((chunk_of[s], last[s]))
    for spans in by_slot.values():
        spans.sort()
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


@pytest.mark.parametrize("robot_name,name", [("panda", "rnea_derivatives"), ("grid_g1", "rnea")])
def test_schedule_fits_a_budget(robot_name, name):
    r = robot(robot_name)
    p = mantis.trace(r, name, **options(r, name))
    base = schedule.build(p, 4)
    budget = int(base.n_slots * 0.9)  # reachable on both: g1 rnea bottoms out at 360 of 432
    sc = schedule.build(p, 4, max_slots=budget)
    assert sc.fits and sc.lever and sc.n_slots <= budget
    assert sorted(sc.out_index) == sorted(base.out_index)  # same outputs, same expressions


def test_schedule_reports_when_nothing_fits():
    p = mantis.trace(robot("panda"), "rnea")
    sc = schedule.build(p, 4, max_slots=1)
    assert not sc.fits and sc.n_slots > 1


def test_max_slots_follows_the_device():
    small, big = device.from_table((8, 6)), device.from_table((9, 0))
    v = cuda.Variant("block", 128, problems=32)
    assert cuda.max_slots(v, 4, small) == 101376 // 4 // 32
    assert cuda.max_slots(v, 4, big) > cuda.max_slots(v, 4, small)
    assert cuda.max_slots(v, 8, small) == cuda.max_slots(v, 4, small) // 2


def test_calibration_is_ordered():
    """The measured constants are positive and the hierarchy is ordered."""
    from mantis import calibrate
    try:
        device.live()
    except RuntimeError:
        pytest.skip("no CUDA driver")
    c = calibrate.get()
    lat = c["lat_ns"]
    assert 0 < lat["shared"] < lat["l2"] < lat["dram"] and lat["l1"] < lat["l2"]
    assert 0 < c["op_issue_ns"] < c["op_latency_ns"] and c["dram_gbs"] > 10
    assert all(v > 0 for v in c["sync_ns"].values()) and all(v > 0 for v in c["access_ns"].values())


def test_cost_picks_the_cheapest_fitting_lever():
    """With a cost function, build returns the cheapest schedule that fits, not the first."""
    p = mantis.trace(robot("panda"), "rnea_derivatives")
    base = schedule.build(p, 4)
    budget = int(base.n_slots * 0.7)
    cost = lambda sc: (sc.stages, sc.n_slots)  # noqa: E731
    picked = schedule.build(p, 4, max_slots=budget, cost=cost)
    first = schedule.build(p, 4, max_slots=budget)
    assert picked.fits and picked.n_slots <= budget
    assert cost(picked) <= cost(first)

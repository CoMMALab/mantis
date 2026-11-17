"""parallel_analysis: the modelled T_p respects Brent's bounds and one lane is the thread tier."""

import json

import mantis
from mantis import parallel_analysis as pa

import robots


def test_bounds_and_serial_baseline():
    p = mantis.trace(robots.robot("panda"), "rnea")
    one = pa.analyze(p, 1, sync_cost=0.0, smem_cost=0.0)
    assert one["T_p"] >= one["W"]  # one lane does all the work (plus any recomputation)
    for u in (4, 32):
        m = pa.analyze(p, u)
        assert m["T_p"] >= m["T_ideal"]
        assert 0 < m["efficiency"] <= 1
        assert abs(sum(m["overhead"][k] for k in ("recomputation", "smem", "barriers", "divergence", "idle"))
                   - m["overhead"]["total"]) < 1e-6
    json.dumps(pa.report("rnea", {"panda": p, "ur5": mantis.trace(robots.robot("ur5"), "rnea")}, [4, 32]))


def test_batch_extremes():
    p = mantis.trace(robots.robot("panda"), "rnea")
    assert pa.batch(p, 1, [4, 64])["best"] != "thread"  # one problem: cooperate
    assert pa.batch(p, 1 << 16, [4, 64])["best"] == "thread"  # GPU full: no overhead wins

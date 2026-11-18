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


def test_variant_tables_and_dump():
    """Every emittable layout gets a row; the thread tier is the E = 1 baseline; the batch model
    prefers the thread tier for throughput once the batch fills the GPU; dump prints a pick."""
    import io
    from mantis import device, parallel_analysis as pa
    from robots import robot
    p = mantis.trace(robot("panda"), "frame_poses", frames=("panda_joint7",))
    dev = device.from_table((8, 6))
    dev = type(dev)(**{**dev.__dict__, "sms": 64})
    t = pa.variant_table(p, ks=(1, 2), problems=(4, 32), dev=dev)
    names = [r["name"] for r in t["rows"]]
    assert names == ["thread", "warp k=1 P=4", "block k=2 P=4", "warp k=1 P=32", "block k=2 P=32"]
    th = t["rows"][0]
    assert th["T_p"] == t["W"] and th["efficiency"] == 1.0 and t["regs_thread"] > 0
    assert all(r["T_p"] >= max(t["W"] / r["lanes_per_problem"], t["D"]) - 1e-9 for r in t["rows"][1:])
    bt = pa.batch_table(t, [1 << 20])
    assert bt[0]["best_throughput"] == "thread"
    out = io.StringIO()
    pa.dump(p, (1, 1 << 20), file=out, ks=(1, 2), problems=(4, 32), dev=dev)
    assert "## Pick" in out.getvalue() and "throughput: thread" in out.getvalue()

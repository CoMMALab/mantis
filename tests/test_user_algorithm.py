"""Algorithms defined outside the package, from mantis.lib, trace without edits to algorithms.py."""
import numpy as np

import mantis
from mantis import lib
from robots import robot as load_robot


def test_user_algorithm_matches_builtin():
    robot = load_robot("panda")
    frames = (robot.joint_names[-1],)

    @mantis.algorithm(inputs=("q",), outputs=("poses",), register=False)
    def my_poses(r, q, frames=()):
        X0 = lib.poses(r, q)
        return [lib.pose_out(lib.frame(r, X0, f)) for f in frames]

    assert "my_poses" not in mantis.ALGORITHMS
    q = np.random.default_rng(0).uniform(-1, 1, (8, robot.n_q))
    got = mantis.trace(robot, my_poses, frames=frames).evaluate(q=q)["poses"]
    want = mantis.trace(robot, "frame_poses", frames=frames).evaluate(q=q)["poses"]
    np.testing.assert_allclose(got, want)
    assert "my_poses" in mantis.generate(robot, {my_poses: {"frames": frames}}, name="user")

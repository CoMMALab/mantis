"""Building blocks for user-defined algorithms (traced with ``mantis.trace`` / ``mantis.generate``
from outside this package, no edits to :mod:`mantis.algorithms` needed)::

    import mantis
    from mantis import lib

    @mantis.algorithm(inputs=("q",), outputs=("centres",), register=False)
    def my_spheres(robot, q, spheres=()):
        return lib.sphere_centres(robot, lib.poses(robot, q), spheres)

    src = mantis.generate(model, {my_spheres: {"spheres": ...}})

Conventions are those of :mod:`mantis.algorithms`: a pose ``X`` holds ``[E row-major (9), t (3)]``
(world point ``t + E^T c``); outputs are ``[t, R column-major]``; Jacobians are rows (v, w).
"""

from .algorithms import (_argmin_tree as argmin_tree, _cholesky_solve as cholesky_solve,
                         _frame as frame, _jacobian as jacobian, _min_tree as min_tree,
                         _pose_out as pose_out, _poses as poses, _quat_matrix as quat_matrix,
                         _so3_log as so3_log, _sphere_centres as sphere_centres, _sum_tree as sum_tree)
from .algorithms import _WORLD as WORLD_KINDS
from .algorithms import _obstacle
from .trace import SE3, Force, Scalar


def obstacle_distance(kind: str, c, r, o):
    """Signed distance from sphere (centre ``c`` (3), radius ``r``) to one obstacle ``o`` of
    ``kind`` in :data:`WORLD_KINDS` (``(kind, floats per obstacle)``: GRiM collision.cuh's
    layouts); negative is penetration. Obstacles can be per-call inputs."""
    return _obstacle(kind, c, r, o)[0]

__all__ = ["SE3", "Force", "Scalar", "argmin_tree", "cholesky_solve", "frame", "jacobian", "min_tree",
           "obstacle_distance", "WORLD_KINDS", "pose_out", "poses", "quat_matrix", "so3_log", "sphere_centres", "sum_tree"]

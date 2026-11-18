"""mantis: a trace compiler for GRiD/GLASS rigid-body kernels."""

from __future__ import annotations

__version__ = "0.2.0"

from . import algorithms  # noqa: F401  (registers the algorithms)
from . import cuda
from .ad import derivative
from .cuda import DEFAULT_VARIANTS, Variant
from .robot import Robot
from . import lib
from .trace import ALGORITHMS, SE3, Algorithm, Force, Program, Scalar, algorithm, resolve


def trace(robot: Robot, name, order: int = 0, wrt=("q",), ad_mode: str = "forward", **options) -> Program:
    """Trace algorithm ``name`` (a registered name, an :class:`Algorithm` or an ``@algorithm``
    function) against ``robot``; ``order > 0`` adds its derivatives up to that
    order wrt the inputs ``wrt``, compiled ahead of time (:func:`mantis.ad.derivative`, ``ad_mode``
    "forward" or "reverse"; not ``mode``, which algorithms such as traj_collision take as an option)."""
    p = resolve(name).trace(robot, **options)
    return derivative(p, order, wrt, ad_mode) if order else p


def generate(robot: Robot, algorithms: dict, variants=DEFAULT_VARIANTS, name: str | None = None,
             kinematics: dict | None = None, budget: int = cuda.BUDGET) -> str:
    """One CUDA header for ``robot``: each algorithm (name, :class:`Algorithm` or ``@algorithm``
    function -> trace options) at each tier in
    ``variants`` (or name -> variants, e.g. from :func:`mantis.tune.tune`), plus cricket's
    float kinematics interface when ``kinematics={"frames": [...], "end_effector": ...}``."""
    sections = []
    for alg, options in algorithms.items():
        vs = variants[resolve(alg).name] if isinstance(variants, dict) else variants
        sections.append(cuda.emit(trace(robot, alg, **options), list(vs), budget))
    if kinematics is not None:
        sections.append(cuda.kinematics(robot, kinematics.get("frames", ()), kinematics.get("end_effector"),
                                        budget))
    return cuda.header(robot, sections, name)


__all__ = ["ALGORITHMS", "DEFAULT_VARIANTS", "SE3", "Algorithm", "Force", "Program", "Robot", "Scalar", "Variant",
           "algorithm", "derivative", "generate", "lib", "resolve", "trace"]

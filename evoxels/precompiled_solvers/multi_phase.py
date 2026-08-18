from __future__ import annotations

from ..pdes import MultiPhaseAllenCahn
from ..solvers import MultiPhaseSolver
from ..timesteppers import RungeKutta4


def run_multi_phase_solver(
    voxelfields,
    fieldnames: str | list[str],
    backend: str,
    jit: bool = True,
    device: str = "cuda",
    time_increment: float = 0.5,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 3.0,
    gab: float = 1.0,
    M: float = 1.0,
    force: float = 0.0,
    curvature: float = 1.0,
    potential: str = "well",
    fast: bool = False,
    bc: tuple = ("periodic", "periodic", "periodic"),
    from_labels: bool = True,
    output_label_fieldname: str | None = None,
    max_phases: int = 10,
    vtk_out: bool = False,
    verbose: bool = True,
    plot_bounds=None,
):
    """Solve a multiphase Allen-Cahn problem with classical RK4."""
    solver = MultiPhaseSolver(
        voxelfields,
        fieldnames,
        backend,
        problem_cls=MultiPhaseAllenCahn,
        timestepper_cls=RungeKutta4,
        device=device,
        from_labels=from_labels,
        output_label_fieldname=output_label_fieldname,
        max_phases=max_phases,
    )
    solver.solve(
        time_increment=time_increment,
        frames=frames,
        max_iters=max_iters,
        problem_kwargs={
            "eps": eps,
            "gab": gab,
            "M": M,
            "force": force,
            "curvature": curvature,
            "potential": potential,
            "fast": fast,
            "bc": bc,
        },
        jit=jit,
        verbose=verbose,
        vtk_out=vtk_out,
        plot_bounds=plot_bounds,
    )

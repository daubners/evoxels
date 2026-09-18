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
    bulk_driving_forces: tuple[float, ...] | None = None,
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
    if bulk_driving_forces is not None and len(bulk_driving_forces) != solver.phase_count:
        raise ValueError("bulk_driving_forces must contain one value per phase.")

    solver.solve(
        time_increment=time_increment,
        frames=frames,
        max_iters=max_iters,
        problem_kwargs={
            "eps": eps,
            "gab": gab,
            "M": M,
            "bulk_driving_forces": bulk_driving_forces,
            "fast": fast,
            "bc": bc,
        },
        jit=jit,
        verbose=verbose,
        vtk_out=vtk_out,
        plot_bounds=plot_bounds,
    )

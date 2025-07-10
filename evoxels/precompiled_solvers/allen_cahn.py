from ..problem_definition import AllenCahnEquation
from ..solvers import TimeDependentSolver
from ..timesteppers import ForwardEuler
from typing import Callable

def run_allen_cahn_solver(
    voxelfields,
    fieldnames: str | list[str],
    backend: str,
    jit: bool = True,
    device: str = "cuda",
    time_increment: float = 0.1,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 2.0,
    gab: float = 1.0,
    M: float = 1.0,
    force: float = 0.0,
    curvature: float = 0.01,
    potential: Callable | None = None,
    vtk_out: bool = False,
    verbose: bool = True,
    plot_bounds = None,
):
    """
    Runs the Cahn-Hilliard solver with a predefined problem and timestepper.
    """
    solver = TimeDependentSolver(
        voxelfields,
        fieldnames,
        backend,
        problem_cls = AllenCahnEquation,
        timestepper_cls = ForwardEuler,
        device=device,
    )
    solver.solve(
        time_increment=time_increment,
        frames=frames,
        max_iters=max_iters,
        problem_kwargs={"eps": eps,
                        "gab": gab,
                        "M": M,
                        "force": force,
                        "curvature": curvature,
                        "potential": potential},
        jit=jit,
        verbose=verbose,
        vtk_out=vtk_out,
        plot_bounds=plot_bounds,
    )

from ..problem_definition import PeriodicCahnHilliard
from ..solvers import TimeDependentSolver
from ..timesteppers import PseudoSpectralIMEX
from typing import Callable

def run_cahn_hilliard_solver(
    voxelfields,
    fieldnames: str | list[str],
    backend: str,
    jit: bool = True,
    device: str = "cuda",
    time_increment: float = 0.1,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 3.0,
    diffusivity: float = 1.0,
    mu_hom: Callable | None = None,
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
        problem_cls = PeriodicCahnHilliard,
        timestepper_cls = PseudoSpectralIMEX,
        device=device,
    )
    solver.solve(
        time_increment=time_increment,
        frames=frames,
        max_iters=max_iters,
        problem_kwargs={"eps": eps, "D": diffusivity, "mu_hom": mu_hom},
        jit=jit,
        verbose=verbose,
        vtk_out=vtk_out,
        plot_bounds=plot_bounds,
    )

from ..problem_definition import PeriodicCahnHilliard
from ..solvers import OneVariableTimeDependendSolver
from ..timesteppers import pseudo_spectral_IMEX

def run_cahn_hilliard_solver(
    voxelfields,
    fieldname: str,
    backend: str,
    jit: bool = True,
    device: str = "cuda",
    time_increment: float = 0.1,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 3.0,
    diffusivity: float = 1.0,
    A: float = 0.25,
    vtk_out: bool = False,
    verbose: bool = True,
    plot_bounds = None,
):
    """
    Runs the Cahn-Hilliard solver with a predefined problem and timestepper.
    """
    solver = OneVariableTimeDependendSolver(
        voxelfields,
        fieldname,
        PeriodicCahnHilliard,
        pseudo_spectral_IMEX,
        backend,
        device=device,
    )
    solver.solve(
        time_increment=time_increment,
        frames=frames,
        max_iters=max_iters,
        problem_kwargs={"eps": eps, "D": diffusivity, "A": A},
        jit=jit, \
        verbose=verbose,
        vtk_out=vtk_out,
        plot_bounds=plot_bounds,
    )

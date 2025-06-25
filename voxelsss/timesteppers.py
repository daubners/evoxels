from .problem_definition import ODE, SemiLinearODE
from typing import TypeVar, Callable

State = TypeVar("State")
TimeStepFn = Callable[[State], State]

def forward_euler(problem: ODE, time_increment: float) -> TimeStepFn:
    """First order Euler forward scheme"""
    def step_fn(u, t):
        update = time_increment * problem.rhs(u, t)
        return u + update

    return step_fn

def pseudo_spectral_IMEX(problem: SemiLinearODE, time_increment: float) -> TimeStepFn:
    """
    First‐order IMEX pseudo‐spectral (Fourier) Euler scheme aka
     -> Semi-implicit Fourier spectral method [Zhu and Chen 1999]
    """
    def step_fn(u, t):
        dc = problem.rhs(u, t)
        dc_fft = problem.vg.rfftn(dc)
        dc_fft *= time_increment / (1 - time_increment*problem.fourier_symbol)
        update = problem.vg.irfftn(dc_fft)
        return u + update

    return step_fn

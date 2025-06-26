from .problem_definition import ODE, SpectralODE
from typing import TypeVar, Callable
import diffrax as dfx

State = TypeVar("State")
TimeStepFn = Callable[[State], State]

def forward_euler(problem: ODE, time_increment: float) -> TimeStepFn:
    """First order Euler forward scheme"""
    def step_fn(u, t):
        update = time_increment * problem.rhs(u, t)
        return u + update

    return step_fn

def pseudo_spectral_IMEX(problem: SpectralODE, time_increment: float) -> TimeStepFn:
    """
    First‐order IMEX pseudo‐spectral (Fourier) Euler scheme aka
     -> Semi-implicit Fourier spectral method [Zhu and Chen 1999]
    """
    def step_fn(u, t):
        # Compute update (in Fourier space) and transform back
        dc = problem.rhs(u, t)
        dc_fft = problem.vg.rfftn(dc)
        dc_fft *= time_increment / (1 + time_increment*problem.spectral_factor)
        update = problem.vg.irfftn(dc_fft, dc.shape)
        return u + update

    return step_fn

class SemiImplicitFourierSpectral(dfx.AbstractSolver):
    spectral_factor: float
    rfftn: Callable
    irfftn: Callable
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)
        euler_y1 = y0 + δt * f0
        dc_fft = self.rfftn(f0)
        dc_fft *= δt / (1.0 + self.spectral_factor * δt)
        update = self.irfftn(dc_fft, f0.shape)
        y1 = y0 + update

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
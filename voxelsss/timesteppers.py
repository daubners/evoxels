from .problem_definition import ODE, SpectralODE
from typing import TypeVar, Callable
import diffrax as dfx
import jax

State = TypeVar("State")
TimeStepFn = Callable[[State], State]

def forward_euler(problem: ODE, time_increment: float) -> TimeStepFn:
    """First order Euler forward scheme"""
    def step_fn(u, t):
        update = time_increment * problem.rhs(u, t)
        return u + problem.vg.pad_with_ghost_nodes(update)
    
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
        return u + problem.vg.pad_with_ghost_nodes(update)

    return step_fn

class SemiImplicitFourierSpectral(dfx.AbstractSolver):

    problem: SpectralODE
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
        tmp = 1.0 + self.A * δt * self.kappa * self.two_pi_i_k_4
        # TODO: replace with problem.spectral_factor like Simon uses above
        # if we decide to do this, this class should have an attribute for the SpectralODE
        y1 = y0 + δt * self.ifft(self.fft(f0) / tmp).real

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
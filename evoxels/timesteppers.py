import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .problem_definition import ODE, SemiLinearODE

State = Any  # e.g. torch.Tensor or jax.Array

class TimeStepper(ABC):
    """Abstract interface for single‐step timestepping schemes."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Temporal order of accuracy."""
        pass

    @abstractmethod
    def step(self, u: State, t: float) -> State:
        """
        Take one timestep from t to (t+dt).

        Args:
            u       : Current state
            t       : Current time
        Returns:
            Updated state at t + dt.
        """
        pass


@dataclass
class ForwardEuler(TimeStepper):
    """First order Euler forward scheme."""
    problem: ODE
    dt: float

    @property
    def order(self) -> int:
        return 1

    def step(self, u: State, t: float) -> State:
        return u + self.dt * self.problem.rhs(u, t)


@dataclass
class PseudoSpectralIMEX(TimeStepper):
    """First‐order IMEX Fourier pseudo‐spectral scheme
    
    aka semi-implicit Fourier spectral method; see
    [Zhu and Chen 1999, doi:10.1103/PhysRevE.60.3564]
    for more details.
    """
    problem: SemiLinearODE
    dt: float

    def __post_init__(self):
        # Pre‐bake the linear prefactor in Fourier
        self._fft_prefac = self.dt / (1 - self.dt*self.problem.fourier_symbol)
        if self.problem.bc_type == 'periodic':
            self.pad = self.problem.vg.bc.pad_fft_periodic
        elif self.problem.bc_type == 'dirichlet':
            self.pad = self.problem.vg.bc.pad_fft_dirichlet_periodic
        elif self.problem.bc_type == 'neumann':
            self.pad = self.problem.vg.bc.pad_fft_zero_flux_periodic

    @property
    def order(self) -> int:
        return 1

    def step(self, u: State, t: float) -> State:
        dc = self.pad(self.problem.rhs(u, t))
        dc_fft = self._fft_prefac * self.problem.vg.rfftn(dc, dc.shape)
        update = self.problem.vg.irfftn(dc_fft, dc.shape)[:,:u.shape[1]]
        return u + update


try:
    import jax.numpy as jnp
    import diffrax as dfx

    class PseudoSpectralIMEX_dfx(dfx.AbstractSolver):
        """Re-implementation of pseudo_spectral_IMEX as diffrax class
        
        This is used for the inversion models based on jax and diffrax
        """
        fourier_symbol: float
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
            dc_fft = jnp.fft.rfftn(f0)
            dc_fft *= δt / (1.0 - self.fourier_symbol * δt)
            update = jnp.fft.irfftn(dc_fft, f0.shape)
            y1 = y0 + update

            y_error = y1 - euler_y1
            dense_info = dict(y0=y0, y1=y1)

            solver_state = None
            result = dfx.RESULTS.successful
            return y1, y_error, dense_info, solver_state, result

        def func(self, terms, t0, y0, args):
            return terms.vf(t0, y0, args)
        
except ImportError:
    PseudoSpectralIMEX_dfx = None
    warnings.warn("Diffrax not found. 'PseudoSpectralIMEX_dfx' will not be available.")
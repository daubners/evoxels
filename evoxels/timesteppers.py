import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
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
    def step(self, t: float, u: State) -> State:
        """
        Take one timestep from t to (t+dt).

        Args:
            t       : Current time
            u       : Current state
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

    def step(self, t: float, u: State) -> State:
        return u + self.dt * self.problem.rhs(t, u)


@dataclass
class RungeKutta4(TimeStepper):
    """Classical explicit Runge-Kutta Scheme of order 4."""
    problem: ODE
    dt: float

    @property
    def order(self) -> int:
        return 4

    def step(self, t: float, u: State) -> State:
        k1 = self.problem.rhs(t, u)
        k2 = self.problem.rhs(t + 0.5*self.dt, u + 0.5*self.dt*k1)
        k3 = self.problem.rhs(t + 0.5*self.dt, u + 0.5*self.dt*k2)
        k4 = self.problem.rhs(t + self.dt, u + self.dt*k3)
        return u + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)


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

    def step(self, t: float, u: State) -> State:
        dc = self.pad(self.problem.rhs(t, u))
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


@dataclass
class ExponentialEuler(TimeStepper):
    """First-order exponential Euler (ETD1) method for semilinear problems.

    Implementation of the exponential time differencing method of order 1 (ETD1)
    described in Hochbruck, Lubich, Selhofer (1998), doi:10.1137/S1064827595295337
    Update:
        u_{n+1} = u_n + dt * varphi_1(dt L) * rhs(u_n)
    where
        varphi_1(z) = (exp(z) - 1) / z.
    """
    problem: SemiLinearODE
    dt: float

    def __post_init__(self):
        self.phi_1_k_squared = self.phi1(self.dt*self.problem.fourier_symbol)
        if self.problem.bc_type == 'periodic':
            self.pad = self.problem.vg.bc.pad_fft_periodic
        elif self.problem.bc_type == 'dirichlet':
            self.pad = self.problem.vg.bc.pad_fft_dirichlet_periodic
        elif self.problem.bc_type == 'neumann':
            self.pad = self.problem.vg.bc.pad_fft_zero_flux_periodic

    def phi1(self, z):
        """Compute varphi_1(z) = (exp(z)-1)/(z)
        
        with special handling for small v to avoid loss of significance.
        Coefficients for the degree-6 Padé approximation are taken from
        Hochbruck, Lubich, Selhofer (1998), doi:10.1137/S1064827595295337
        """
        Q = 6
        N = [1, 1/26,  5/156,  1/858, 1/5720,  1/205920, 1/8648640]
        D = [1, -6/13, 5/52,  -5/429, 1/1144, -1/25740,  1/1235520]
        
        phi = (self.problem.vg.lib.exp(z) - 1) / z
        indx = (self.problem.vg.lib.abs(z) < 0.5)
        phi = self.problem.vg.set(phi, indx, self.phiPade(z[indx], Q, N, D))
        return phi

    def phiPade(self, z, Q, Ncoeff, Dcoeff):
        """Evaluate (Q,Q)-Padé approximation of phi-function
        
        This routine evaluates the exponential-integrator
        varphi_1(z) = (exp(z)-1)/z as the rational approximation

            varphi_1(z) ≈ P_Q(z) / R_Q(z),

        where P_Q and R_Q are degree-Q polynomials with coefficients
        given by `Ncoeff` and `Dcoeff`, respectively. It is used for
        arguments `z` near zero, where the direct formula suffers
        from loss of significance due to cancellation.
        """
        numerator = Ncoeff[Q]
        denominator = Dcoeff[Q]
        for k in range(Q - 1, -1, -1):
            numerator = numerator * z + Ncoeff[k]
            denominator = denominator * z + Dcoeff[k]
        return numerator / denominator

    @property
    def order(self) -> int:
        return 1

    def step(self, t: float, u: State) -> State:
        dc = self.pad(self.problem.rhs(t, u))
        dc_fft = self.dt * self.phi_1_k_squared * self.problem.vg.rfftn(dc, dc.shape)
        update = self.problem.vg.irfftn(dc_fft, dc.shape)[:,:u.shape[1]]
        return u + update


@dataclass
class RKC1(TimeStepper):
    """Runge-Kutta-Chebyshev Scheme of order 1.
    
    TODO: add citation."""
    problem: ODE
    dt: float
    polygrad: int = 4
    damping: float = 0.05

    def __post_init__(self):
        w0 = 1 + (self.damping/(self.polygrad**2))
        s = np.arange(0, self.polygrad+1)
        T_w0 = np.cosh(s*np.arccosh(w0))
        dT_w0 = s*np.sinh(s*np.arccosh(w0))/np.sqrt(w0**2 - 1)
        b = 1/T_w0

        w1 = T_w0[-1]/dT_w0[-1]
        self.mu0 = 2 * w0 * (b[2:]/b[1:-1])
        self.mu1 = 2 * w1 * (b[2:]/b[1:-1])
        self.mu11 = w1/w0
        self.nu  = -(b[2:]/b[:-2])
        self.c = w1 * (dT_w0/T_w0)[1:-1]

    @property
    def order(self) -> int:
        return 1

    def step(self, t: float, u: State) -> State:
        Y_prev = u
        Y_curr = u + self.mu11 * self.dt * self.problem.rhs(t, u)
        for j in range(self.polygrad-1):
            rhs = self.problem.rhs(t + self.c[j]*self.dt, Y_curr)
            Y_new = (  self.mu0[j] * Y_curr 
                     + self.nu[j] * Y_prev
                     + (1 - self.mu0[j] - self.nu[j]) * u
                     + self.mu1[j] * self.dt * rhs)
            Y_prev = Y_curr
            Y_curr = Y_new
        return Y_curr

@dataclass
class RKC2(TimeStepper):
    """Runge-Kutta-Chebyshev Scheme of order 2."""
    problem: ODE
    dt: float
    polygrad: int = 4
    damping: float = 2/13

    def __post_init__(self):
        w0 = 1 + (self.damping/self.polygrad**2)
        s = np.arange(0, self.polygrad+1)
        T_w0 = np.cosh(s*np.arccosh(w0))
        dT_w0 = s*np.sinh(s*np.arccosh(w0))/np.sqrt(w0**2 - 1)
        d2T_w0 = (s*s * T_w0 - w0 * dT_w0) / (w0**2 - 1)
        b = d2T_w0/dT_w0**2
        b[0] = b[2]
        b[1] = b[2]

        w1 = dT_w0[-1]/d2T_w0[-1]
        self.mu0 = 2 * w0 * (b[2:]/b[1:-1])
        self.mu1 = 2 * w1 * (b[2:]/b[1:-1])
        self.mu11 = b[1]*w1
        self.nu  = -(b[2:]/b[:-2])
        self.gamma = -(1-b[1:-1]*T_w0[1:-1])*self.mu1
        self.c = w1 * (d2T_w0/dT_w0)[1:-1]
        self.c[0] = self.c[1]/dT_w0[2]

    @property
    def order(self) -> int:
        return 2

    def step(self, t: float, u: State) -> State:
        Y_prev = u
        rhs_0 = self.problem.rhs(t, u)
        Y_curr = u + self.mu11 * self.dt * rhs_0
        for j in range(self.polygrad-1):
            rhs = self.problem.rhs(t + self.c[j]*self.dt, Y_curr)
            Y_new = (  self.mu0[j] * Y_curr
                     + self.nu[j] * Y_prev
                     + ( 1 - self.mu0[j] - self.nu[j] ) * u
                     + self.mu1[j] * self.dt * rhs
                     + self.gamma[j] * self.dt * rhs_0)
            Y_prev = Y_curr
            Y_curr = Y_new
        return Y_curr

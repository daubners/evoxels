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

    def step(self, t: float, u: State) -> State:
        """
        Take one timestep from t to (t+dt).

        Args:
            t       : Current time
            u       : Current state
        Returns:
            Updated state at t + dt.
        """
        return self.step_dt(t, self.dt, u)

    @abstractmethod
    def step_dt(self, t: float, dt: float, u: State) -> State:
        """Take one timestep from ``t`` to ``t + dt``."""
        pass


@dataclass(eq=False)
class ForwardEuler(TimeStepper):
    """First order Euler forward scheme."""
    problem: ODE
    dt: float

    @property
    def order(self) -> int:
        return 1

    def step_dt(self, t: float, dt: float, u: State) -> State:
        return u + dt * self.problem.rhs(t, u)


@dataclass(eq=False)
class RungeKutta4(TimeStepper):
    """Classical explicit Runge-Kutta Scheme of order 4."""
    problem: ODE
    dt: float

    @property
    def order(self) -> int:
        return 4

    def step_dt(self, t: float, dt: float, u: State) -> State:
        k1 = self.problem.rhs(t, u)
        k2 = self.problem.rhs(t + 0.5*dt, u + 0.5*dt*k1)
        k3 = self.problem.rhs(t + 0.5*dt, u + 0.5*dt*k2)
        k4 = self.problem.rhs(t + dt, u + dt*k3)
        return u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


@dataclass(eq=False)
class PseudoSpectralIMEX(TimeStepper):
    """First‐order IMEX Fourier pseudo‐spectral scheme
    
    aka semi-implicit Fourier spectral method; see
    [Zhu and Chen 1999, doi:10.1103/PhysRevE.60.3564]
    for more details.
    """
    problem: SemiLinearODE
    dt: float

    def __post_init__(self):
        # Cache the fixed-step Fourier prefactor for the standard step() path.
        self._fft_prefac = self.dt / (1 - self.dt*self.problem.fourier_symbol)
        self.problem.verify_fft_bc_config()
        self.pad = self.problem.pad_fft_bc

    @property
    def order(self) -> int:
        return 1

    def step(self, t: float, u: State) -> State:
        dc = self.pad(self.problem.rhs(t, u))
        dc_fft = self._fft_prefac * self.problem.vg.rfftn(dc, dc.shape)
        update = self.problem.vg.irfftn(dc_fft, dc.shape)[:,:u.shape[1]]
        return u + update

    def step_dt(self, t: float, dt: float, u: State) -> State:
        dc = self.pad(self.problem.rhs(t, u))
        fft_prefac = dt / (1 - dt*self.problem.fourier_symbol)
        dc_fft = fft_prefac * self.problem.vg.rfftn(dc, dc.shape)
        update = self.problem.vg.irfftn(dc_fft, dc.shape)[:,:u.shape[1]]
        return u + update

@dataclass(eq=False)
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
        # Cache the fixed-step spectral factor for the standard step() path.
        self.phi_1_k_squared = self.phi1(self.dt*self.problem.fourier_symbol)
        self.problem.verify_fft_bc_config()
        self.pad = self.problem.pad_fft_bc

    def phi1(self, z):
        """Compute varphi_1(z) = (exp(z)-1)/(z)
        
        with special handling for small v to avoid loss of significance.
        Coefficients for the degree-6 Padé approximation are taken from
        Hochbruck, Lubich, Selhofer (1998), doi:10.1137/S1064827595295337
        """
        Q = 6
        N = [1, 1/26,  5/156,  1/858, 1/5720,  1/205920, 1/8648640]
        D = [1, -6/13, 5/52,  -5/429, 1/1144, -1/25740,  1/1235520]

        mask = self.problem.vg.lib.abs(z) < 0.5
        safe_z = self.problem.vg.lib.where(mask, 1.0, z)
        phi_direct = (self.problem.vg.lib.exp(z) - 1) / safe_z
        phi_pade = self.phiPade(z, Q, N, D)
        return self.problem.vg.lib.where(mask, phi_pade, phi_direct)

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

    def step_dt(self, t: float, dt: float, u: State) -> State:
        dc = self.pad(self.problem.rhs(t, u))
        phi_1_k_squared = self.phi1(dt*self.problem.fourier_symbol)
        dc_fft = dt * phi_1_k_squared * self.problem.vg.rfftn(dc, dc.shape)
        update = self.problem.vg.irfftn(dc_fft, dc.shape)[:,:u.shape[1]]
        return u + update


@dataclass(eq=False)
class RKC1(TimeStepper):
    """Runge-Kutta-Chebyshev Scheme of order 1.
    
    Based on the publication
    "Convergence properties of the Runge-Kutta-Chebyshev method" by
    Verwer, Hundsdorfer, Sommeijer (1990), doi: 10.1007/BF01386405
    """
    problem: ODE
    dt: float
    polygrad: int = 4
    damping: float = 0.05

    def __post_init__(self):
        w0 = self.problem.vg.to_backend(1 + (self.damping/(self.polygrad**2)))
        s = self.problem.vg.arange(0, self.polygrad+1)
        T_w0 = self.problem.vg.lib.cosh(s*self.problem.vg.lib.arccosh(w0))
        dT_w0 = s*self.problem.vg.lib.sinh(s*self.problem.vg.lib.arccosh(w0))/self.problem.vg.lib.sqrt(w0**2 - 1)
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

    def step_dt(self, t: float, dt: float, u: State) -> State:
        Y_prev = u
        Y_curr = u + self.mu11 * dt * self.problem.rhs(t, u)
        for j in range(self.polygrad-1):
            rhs = self.problem.rhs(t + self.c[j]*dt, Y_curr)
            Y_new = (  self.mu0[j] * Y_curr 
                     + self.nu[j] * Y_prev
                     + (1 - self.mu0[j] - self.nu[j]) * u
                     + self.mu1[j] * dt * rhs)
            Y_prev = Y_curr
            Y_curr = Y_new
        return Y_curr

@dataclass(eq=False)
class RKC2(TimeStepper):
    """Runge-Kutta-Chebyshev Scheme of order 2.
    
    Based on the publication
    "Convergence properties of the Runge-Kutta-Chebyshev method" by
    Verwer, Hundsdorfer, Sommeijer (1990), doi: 10.1007/BF01386405
    """
    problem: ODE
    dt: float
    polygrad: int = 4
    damping: float = 2/13

    def __post_init__(self):
        w0 = self.problem.vg.to_backend(1 + (self.damping/self.polygrad**2))
        s = self.problem.vg.arange(0, self.polygrad+1)
        T_w0 = self.problem.vg.lib.cosh(s*self.problem.vg.lib.arccosh(w0))
        dT_w0 = s*self.problem.vg.lib.sinh(s*self.problem.vg.lib.arccosh(w0))/self.problem.vg.lib.sqrt(w0**2 - 1)
        d2T_w0 = (s*s * T_w0 - w0 * dT_w0) / (w0**2 - 1)
        b = d2T_w0/dT_w0**2
        b = self.problem.vg.set(b, 0, b[2])
        b = self.problem.vg.set(b, 1, b[2])

        w1 = dT_w0[-1]/d2T_w0[-1]
        self.mu0 = 2 * w0 * (b[2:]/b[1:-1])
        self.mu1 = 2 * w1 * (b[2:]/b[1:-1])
        self.mu11 = b[1]*w1
        self.nu  = -(b[2:]/b[:-2])
        self.gamma = -(1-b[1:-1]*T_w0[1:-1])*self.mu1
        self.c = w1 * (d2T_w0/dT_w0)[1:-1]
        self.c = self.problem.vg.set(self.c, 0, self.c[1]/dT_w0[2])

    @property
    def order(self) -> int:
        return 2

    def step_dt(self, t: float, dt: float, u: State) -> State:
        Y_prev = u
        rhs_0 = self.problem.rhs(t, u)
        Y_curr = u + self.mu11 * dt * rhs_0
        for j in range(self.polygrad-1):
            rhs = self.problem.rhs(t + self.c[j]*dt, Y_curr)
            Y_new = (  self.mu0[j] * Y_curr
                     + self.nu[j] * Y_prev
                     + ( 1 - self.mu0[j] - self.nu[j] ) * u
                     + self.mu1[j] * dt * rhs
                     + self.gamma[j] * dt * rhs_0)
            Y_prev = Y_curr
            Y_curr = Y_new
        return Y_curr

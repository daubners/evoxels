from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import sympy as sp
import sympy.vector as spv
from .voxelgrid import VoxelGrid

class ODE(ABC):
    @property
    @abstractmethod
    def order(self):
        """Spatial order of convergence for numerical right-hand side."""
        pass

    @abstractmethod
    def rhs_analytic(self, u, t):
        """Sympy expression of the problem right-hand side.

        Args:
            u : Sympy function of current state.
            t (float): Current time.

        Returns:
            Sympy function of problem right-hand side.
        """
        pass

    @abstractmethod
    def rhs(self, u, t):
        """Numerical right-hand side of the ODE system.

        Args:
            u (array): Current state.
            t (float): Current time.

        Returns:
            Same type as ``u`` containing the time derivative.
        """
        pass

    @abstractmethod
    def pad_boundary_conditions(self, u):
        """Function to pad and impose boundary conditions.

        Enables applying boundary conditions on u outside of
        the right-hand-side function.

        Args:
            u : field

        Returns:
            Field padded with boundary values.
        """
        pass


class SpectralODE(ODE):
    @property
    @abstractmethod
    def spectral_factor(self):
        """Spectral factor required for spectral ODE solvers."""
        pass


@dataclass
class PoissonEquation(SpectralODE):
    vg: VoxelGrid
    D: float
    BC_type: str
    bcs: tuple = (0,0)
    f: Callable | None = None
    A: float = 0.25
    _spectral_factor: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._spectral_factor = self.D * self.A * k_squared**2
        if self.f is None:
            self.f = lambda c=None, t=None, lib=None: 0

        if self.BC_type == 'periodic':
            bc_fun = self.vg.pad_periodic_BC
            self.pad_bcs = lambda field, bc0, bc1: bc_fun(field)
        elif self.BC_type == 'dirichlet':
            self.pad_bcs = self.vg.pad_dirichlet_periodic_BC
        elif self.BC_type == 'neumann':
            bc_fun = self.vg.pad_zero_flux_periodic_BC
            self.pad_bcs = lambda field, bc0, bc1: bc_fun(field)
    
    @property
    def order(self):
        return 2

    @property
    def spectral_factor(self):
        return self._spectral_factor
    
    def _eval_f(self, c, t, lib):
        """Evaluate source/forcing term using ``self.f``."""
        try:
            return self.f(c, t, lib)
        except TypeError:
            return self.f(c, t)

    def pad_boundary_conditions(self, u):
        return self.pad_bcs(u, self.bcs[0], self.bcs[1])
    
    def rhs_analytic(self, u, t):
        return self.D*spv.laplacian(u) + self._eval_f(u, t, sp)
    
    def rhs(self, u, t):
        laplace = self.vg.calc_laplace(self.pad_bcs(u, self.bcs[0], self.bcs[1]))
        update = self.D * laplace + self._eval_f(u, t, self.vg.lib)
        return update


@dataclass
class PeriodicCahnHilliard(SpectralODE):
    vg: VoxelGrid
    eps: float
    D: float = 1.0
    mu_hom: Callable | None = None
    A: float = 0.25
    _spectral_factor: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._spectral_factor = 2 * self.eps * self.D * self.A * k_squared**2
        if self.mu_hom is None:
            self.mu_hom = lambda c, lib=None: 18 / self.eps * c * (1 - c) * (1 - 2 * c)
    
    @property
    def order(self):
        return 2

    @property
    def spectral_factor(self):
        return self._spectral_factor
    
    def pad_boundary_conditions(self, u):
        return self.vg.pad_periodic_BC(u)
    
    def _eval_mu(self, c, lib):
        """Evaluate homogeneous chemical potential using ``self.mu``."""
        try:
            return self.mu_hom(c, lib)
        except TypeError:
            return self.mu_hom(c)

    def rhs_analytic(self, c, t):
        mu = self._eval_mu(c, sp) - 2*self.eps*spv.laplacian(c)
        fluxes = self.D*c*(1-c)*spv.gradient(mu)
        rhs = spv.divergence(fluxes)
        return rhs

    def calc_divergence_variable_mobility(self, mu, c):
        """Calculate divergence with variable mobility :math:`M=D*c*(1-c)`.

        Args:
            mu: Chemical potential field.
            c: Concentration field.

        Returns:
            Backend array representing the divergence term.
        """
        divergence = ((mu[:, 2:, 1:-1, 1:-1] - mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(c[:, 2:, 1:-1, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, 2:, 1:-1, 1:-1] + c[:, 1:-1, 1:-1, 1:-1])) - \
                      (mu[:, 1:-1, 1:-1, 1:-1] - mu[:, :-2, 1:-1, 1:-1]) *\
                       0.5*(c[:, :-2, 1:-1, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, :-2, 1:-1, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]))) \
                      * self.vg.div_dx2[0]

        divergence += ((mu[:, 1:-1, 2:, 1:-1] - mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(c[:, 1:-1, 2:, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, 1:-1, 2:, 1:-1] + c[:, 1:-1, 1:-1, 1:-1])) - \
                       (mu[:, 1:-1, 1:-1, 1:-1] - mu[:, 1:-1, :-2, 1:-1]) *\
                       0.5*(c[:, 1:-1, :-2, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, 1:-1, :-2, 1:-1] + c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.vg.div_dx2[1]

        divergence += ((mu[:, 1:-1, 1:-1, 2:] - mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(c[:, 1:-1, 1:-1, 2:] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, 1:-1, 1:-1, 2:] + c[:, 1:-1, 1:-1, 1:-1])) - \
                       (mu[:, 1:-1, 1:-1, 1:-1] - mu[:, 1:-1, 1:-1, :-2]) *\
                       0.5*(c[:, 1:-1, 1:-1, :-2] + c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(c[:, 1:-1, 1:-1, :-2] + c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.vg.div_dx2[2]
        return divergence

    def rhs(self, c, t):
        r"""Evaluate :math:`\partial c / \partial t` for the CH equation.

        Numerical computation of

        .. math::
            \frac{\partial c}{\partial t}
            = \nabla \cdot \bigl( M \, \nabla \mu \bigr),
            \quad
            \mu = \frac{\delta F}{\delta c}
            = f'(c) - \kappa \, \nabla^2 c

        where :math:`M` is the (possibly concentration-dependent) mobility,
        :math:`\mu` the chemical potential, and :math:`\kappa` the gradient energy coefficient.

        Args:
            c (array-like): Concentration field.
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``c`` containing ``dc/dt``.
        """
        c = self.vg.lib.clip(c, 0, 1)
        c_BC = self.vg.pad_periodic_BC(c)
        laplace = self.vg.calc_laplace(c_BC)
        mu = self._eval_mu(c, self.vg.lib) - 2*self.eps*laplace
        mu = self.vg.pad_periodic_BC(mu)
        divergence = self.calc_divergence_variable_mobility(mu, c_BC)
        return self.D * divergence


@dataclass
class AllenCahnEquation(SpectralODE):
    vg: VoxelGrid
    eps: float = 2.0
    gab: float = 1.0
    M: float = 1.0
    curvature: float = 0.01
    potential: Callable | None = None
    _spectral_factor: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._spectral_factor = 2 * self.M * self.gab* k_squared
        if self.potential is None:
            self.potential = lambda u, lib=None: 18 / self.eps * u * (1-u) * (1-2*u)
    
    @property
    def order(self):
        return 2

    @property
    def spectral_factor(self):
        return self._spectral_factor

    def pad_boundary_conditions(self, u):
        return self.vg.pad_zero_flux_BC(u)

    def _eval_potential(self, phi, lib):
        """Evaluate phasefield potential"""
        try:
            return self.potential(phi, lib)
        except TypeError:
            return self.potential(phi)

    def rhs_analytic(self, phi, t):
        grad = spv.gradient(phi)
        laplace  = spv.laplacian(phi)
        norm_grad = sp.sqrt(grad.dot(grad))

        # Curvature equals |∇ψ| ∇·(∇ψ/|∇ψ|)
        unit_normal = grad / norm_grad
        curv = norm_grad * spv.divergence(unit_normal)
        n_laplace = laplace - (1-self.curvature)*curv
        df_dphi = 2*n_laplace - self._eval_potential(phi, sp)/self.eps
        return self.M * self.gab * df_dphi

    def rhs(self, phi, t):
        r"""Microstructure evolution with the two-phase Allen-Cahn equationartial \phi / \partial t` for the CH equation.

        Numerical computation of

        .. math::
            \frac{\partial c}{\partial t}
            = \nabla \cdot \bigl( M \, \nabla \mu \bigr),
            \quad
            \mu = \frac{\delta F}{\delta c}
            = f'(c) - \kappa \, \nabla^2 c

        where :math:`M` is the (possibly concentration-dependent) mobility,
        :math:`\mu` the chemical potential, and :math:`\kappa` the gradient energy coefficient.

        Args:
            phi (array-like): order parameter.
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``\phi`` containing ``d\phi/dt``.
        """
        phi = self.vg.lib.clip(phi, 0, 1)
        potential = self._eval_potential(phi, self.vg.lib)
        phi_pad = self.vg.pad_zero_flux_BC(phi)
        laplace = self.curvature*self.vg.calc_laplace(phi_pad)
        n_laplace = (1-self.curvature) * self.vg.calc_normal_laplace(phi_pad)
        df_dphi = 2.0 * (laplace+n_laplace) - potential/self.eps
        return self.M * self.gab * df_dphi

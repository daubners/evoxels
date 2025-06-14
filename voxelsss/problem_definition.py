from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import sympy.vector as spv
from .voxelgrid import VoxelGrid

class ODE(ABC):
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

    @property
    @abstractmethod
    def order(self):
        """Spatial order of convergence for numerical right-hand side."""
        pass

class SpectralODE(ODE):
    @property
    @abstractmethod
    def spectral_factor(self):
        """Spectral factor required for spectral ODE solvers."""
        pass

@dataclass
class PeriodicCahnHilliard(SpectralODE):
    vg: VoxelGrid
    eps: float
    # mu: Callable
    D: float #Callable
    A: float
    _spectral_factor: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._spectral_factor = 2 * self.eps * self.D * self.A * k_squared**2
    
    @property
    def order(self):
        return 2

    @property
    def spectral_factor(self):
        return self._spectral_factor
    
    def rhs_analytic(self, c, t):
        mu = 18/self.eps*c*(1-c)*(1-2*c) - 2*self.eps*spv.laplacian(c)
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
        """Evaluate :math:`\partial c / \partial t` for the CH equation.

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
            c (array-like): Concentration field padded with ghost nodes.
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``c`` containing ``dc/dt``.
        """
        c_BC = self.vg.pad_with_ghost_nodes(c)
        c_BC = self.vg.apply_periodic_BC(c_BC)
        laplace = self.vg.calc_laplace(c_BC)
        mu = 18/self.eps*c*(1-c)*(1-2*c) - 2*self.eps*laplace
        mu = self.vg.pad_zeros(mu)
        mu = self.vg.apply_periodic_BC(mu)
        divergence = self.calc_divergence_variable_mobility(mu, c_BC)
        # divergence = self.calc_divergence_variable_mobility(self.mu(c), c)
        return self.D * divergence

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from .voxelgrid import VoxelGrid

class ODE(ABC):
    @abstractmethod
    def rhs(self, u, t):
        """Right-hand side of the ODE system.

        Args:
            u: Current state.
            t (float): Current time.

        Returns:
            Same type as ``u`` containing the time derivative.
        """
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
    def spectral_factor(self):
        """Precomputed Fourier-space factor used for IMEX schemes."""
        return self._spectral_factor

    def calc_divergence_variable_mobility(self, mu, c):
        """Calculate divergence with variable mobility M=D*c*(1-c).

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
        """Evaluate $\partial c/\partial t$ for the CH equation.

        Args:
            c: Concentration field padded with ghost nodes.
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``c`` containing ``dc/dt``.
        """
        c = self.vg.apply_periodic_BC(c)
        laplace = self.vg.calc_laplace(c)
        mu = 18/self.eps*c*(1-c)*(1-2*c) - 2*self.eps*laplace
        mu = self.vg.apply_periodic_BC(mu)
        divergence = self.calc_divergence_variable_mobility(mu, c)
        # divergence = self.calc_divergence_variable_mobility(self.mu(c), c)
        return self.D * divergence

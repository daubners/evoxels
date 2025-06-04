from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any
from .voxelgrid import VoxelGrid

class ODE(ABC):
    @abstractmethod
    def rhs(self, u, t):
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
        k_squared = self.vg.rfft_k_squared()
        self._spectral_factor = 2 * self.eps * self.D * self.A * k_squared**2
    
    @property
    def spectral_factor(self):
        return self._spectral_factor

    def calc_divergence_variable_mobility(self, mu, c):
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
        c = self.vg.apply_periodic_BC(c)
        laplace = self.vg.calc_laplace(c)
        mu = 18/self.eps*c*(1-c)*(1-2*c) - 2*self.eps*laplace
        mu = self.vg.apply_periodic_BC(mu)
        divergence = self.calc_divergence_variable_mobility(mu, c)
        # divergence = self.calc_divergence_variable_mobility(self.mu(c), c)
        return self.D * divergence
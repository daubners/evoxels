from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import sympy as sp
import sympy.vector as spv

from ..voxelgrid import VoxelGrid
from .base import SemiLinearODE, SmoothedBoundaryODE


@dataclass
class ReactionDiffusion(SemiLinearODE):
    vg: VoxelGrid
    D: float
    f: Callable | None = None
    A: float = 1.0
    bc: tuple = ('periodic', 'periodic', 'periodic')
    _fourier_symbol: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        if self.f is None:
            self.f = lambda c=None, t=None, lib=None: 0

        self.initialize_boundary_conditions()
        self._fourier_symbol = -self.D * self.A * self.k_squared()

    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def _eval_f(self, t, c, lib):
        """Evaluate source/forcing term using ``self.f``."""
        try:
            return self.f(t, c, lib)
        except TypeError:
            return self.f(t, c)
    
    def rhs_analytic(self, t, u):
        return self.D*spv.laplacian(u) + self._eval_f(t, u, sp)

    def rhs(self, t, u):
        laplace = self.vg.laplace(self.pad_bc(u))
        update = self.D * laplace + self._eval_f(t, u, self.vg.lib)
        return update

@dataclass
class ReactionDiffusionSBM(ReactionDiffusion, SmoothedBoundaryODE):
    mask: Any | None = None
    bc_flux: Callable | float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.initialize_smoothed_boundary()

    def rhs_analytic(self, t, u, mask):
        grad_m = spv.gradient(mask)
        norm_grad_m = sp.sqrt(grad_m.dot(grad_m))

        divergence = spv.divergence(self.D*(spv.gradient(u) - u/mask*grad_m))
        du = divergence + norm_grad_m*self.bc_flux + mask*self._eval_f(t, u/mask, sp)
        return du

    def rhs(self, t, u):
        z = self.pad_bc(u)
        divergence = self.vg.grad_x_face(self.vg.grad_x_face(z) -\
                        self.vg.to_x_face(z/self.mask) * self.vg.grad_x_face(self.mask)
                    )[:,:,1:-1,1:-1]
        divergence += self.vg.grad_y_face(self.vg.grad_y_face(z) -\
                        self.vg.to_y_face(z/self.mask) * self.vg.grad_y_face(self.mask)
                    )[:,1:-1,:,1:-1]
        divergence += self.vg.grad_z_face(self.vg.grad_z_face(z) -\
                        self.vg.to_z_face(z/self.mask) * self.vg.grad_z_face(self.mask)
                    )[:,1:-1,1:-1,:]

        update = self.D * divergence + \
                 self.norm*self.bc_flux + \
                 self.mask[:,1:-1,1:-1,1:-1]*self._eval_f(t, u/self.mask[:,1:-1,1:-1,1:-1], self.vg.lib)
        return update

@dataclass
class CoupledReactionDiffusion(SemiLinearODE):
    vg: VoxelGrid
    D_A: float = 1.0
    D_B: float = 0.5
    feed: float = 0.055
    kill: float = 0.117
    interaction: Callable | None = None
    _fourier_symbol: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = - max(self.D_A, self.D_B) * self.k_squared()
        if self.interaction is None:
            self.interaction = lambda u, lib=None: u[0] * u[1]**2
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def _eval_interaction(self, u, lib):
        """Evaluate interaction term"""
        try:
            return self.interaction(u, lib)
        except TypeError:
            return self.interaction(u)

    def rhs_analytic(self, t, u):
        interaction = self._eval_interaction(u, sp)
        dc_A = self.D_A*spv.laplacian(u[0]) - interaction + self.feed * (1-u[0])
        dc_B = self.D_B*spv.laplacian(u[1]) + interaction - self.kill * u[1]
        return (dc_A, dc_B)

    def rhs(self, t, u):
        r"""Two-component reaction-diffusion system
        
        Use batch channels for multiple species:
        - Species A with concentration c_A = u[0]
        - Species B with concentration c_B = u[1]

        Args:
            t (float): Current time.
            u (array-like): species

        Returns:
            Backend array of the same shape as ``u`` containing ``du/dt``.
        """
        interaction = self._eval_interaction(u, self.vg.lib)
        u_pad = self.pad_bc(u)
        laplace = self.vg.laplace(u_pad)
        dc_A = self.D_A*laplace[0] - interaction + self.feed * (1-u[0])
        dc_B = self.D_B*laplace[1] + interaction - self.kill * u[1]
        return self.vg.lib.stack((dc_A, dc_B), 0)

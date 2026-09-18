from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import sympy as sp
import sympy.vector as spv

from ..voxelgrid import VoxelGrid
from .base import SemiLinearODE


@dataclass
class CahnHilliard(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 3.0
    D: float = 1.0
    mu_hom: Callable | None = None
    A: float = 0.25
    bc: tuple = ('periodic', 'periodic', 'periodic')
    _fourier_symbol: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = -2 * self.eps * self.D * self.A * self.k_squared()**2
        if self.mu_hom is None:
            self.mu_hom = lambda c, lib=None: 18 / self.eps * c * (1 - c) * (1 - 2 * c)
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def _eval_mu(self, c, lib):
        """Evaluate homogeneous chemical potential using ``self.mu``."""
        try:
            return self.mu_hom(c, lib)
        except TypeError:
            return self.mu_hom(c)

    def rhs_analytic(self, t, c):
        mu = self._eval_mu(c, sp) - 2*self.eps*spv.laplacian(c)
        fluxes = self.D*c*(1-c)*spv.gradient(mu)
        rhs = spv.divergence(fluxes)
        return rhs

    def rhs(self, t, c):
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
            t (float): Current time.
            c (array-like): Concentration field.

        Returns:
            Backend array of the same shape as ``c`` containing ``dc/dt``.
        """
        c = self.vg.lib.clip(c, 0, 1)
        c_BC = self.pad_bc(c)
        laplace = self.vg.laplace(c_BC)
        mu = self._eval_mu(c, self.vg.lib) - 2*self.eps*laplace
        mu = self.pad_bc(mu)

        divergence = self.vg.grad_x_face(
                        self.vg.to_x_face(c_BC) * (1-self.vg.to_x_face(c_BC)) *\
                        self.vg.grad_x_face(mu)
                    )[:,:,1:-1,1:-1]

        divergence += self.vg.grad_y_face(
                        self.vg.to_y_face(c_BC) * (1-self.vg.to_y_face(c_BC)) *\
                        self.vg.grad_y_face(mu)
                    )[:,1:-1,:,1:-1]

        divergence += self.vg.grad_z_face(
                        self.vg.to_z_face(c_BC) * (1-self.vg.to_z_face(c_BC)) *\
                        self.vg.grad_z_face(mu)
                    )[:,1:-1,1:-1,:]

        return self.D * divergence


@dataclass
class TwoPhaseAllenCahn(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 2.0
    gab: float = 1.0
    M: float = 1.0
    force: float = 0.0
    curvature: float = 0.01
    potential_derivative: Callable | None = None
    bc: tuple = ('neumann','neumann','neumann')
    _fourier_symbol: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = -self.M * self.gab* self.k_squared()
        if self.potential_derivative is None:
            self.potential_derivative = lambda u, lib=None: 18 / self.eps * u * (1-u) * (1-2*u)

    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def _eval_potential(self, phi, lib):
        """Evaluate phasefield potential"""
        try:
            return self.potential_derivative(phi, lib)
        except TypeError:
            return self.potential_derivative(phi)

    def rhs_analytic(self, t, phi):
        grad = spv.gradient(phi)
        laplace  = spv.laplacian(phi)
        norm_grad = sp.sqrt(grad.dot(grad))

        # Curvature equals |∇ψ| ∇·(∇ψ/|∇ψ|)
        unit_normal = grad / norm_grad
        curv = norm_grad * spv.divergence(unit_normal)
        n_laplace = laplace - (1-self.curvature)*curv
        df_dphi = self.gab * (n_laplace - self._eval_potential(phi, sp)/(2*self.eps)) \
                  + norm_grad * self.force
        return self.M * df_dphi

    def rhs(self, t, phi):
        r"""Two-phase Allen-Cahn equation
        
        Microstructural evolution of the order parameter ``\phi``
        which can be interpreted as a phase fraction.
        :math:`M` denotes the mobility,
        :math:`\epsilon` controls the diffuse interface width,
        :math:`\gamma` denotes the interfacial energy.
        The laplacian leads to a phase evolution driven by
        curvature minimization which can be controlled by setting
        ``curvature=`` in range :math:`[0,1]`.

        Args:
            t (float): Current time.
            phi (array-like): order parameter.

        Returns:
            Backend array of the same shape as ``\phi`` containing ``d\phi/dt``.
        """
        phi = self.vg.lib.clip(phi, 0, 1)
        potential = self._eval_potential(phi, self.vg.lib)
        phi_pad = self.pad_bc(phi)
        laplace = self.curvature*self.vg.laplace(phi_pad)
        n_laplace = (1-self.curvature) * self.vg.normal_laplace(phi_pad)
        norm_grad_phi = self.vg.gradient_norm(phi_pad)
        df_dphi = self.gab * (laplace + n_laplace - potential/2/self.eps)\
                  + norm_grad_phi * self.force
        return self.M * df_dphi


@dataclass
class MultiPhaseAllenCahn(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 3.0
    gab: float = 1.0
    M: float = 1.0
    bulk_driving_forces: tuple[float, ...] | None = None
    fast: bool = True
    bc: tuple = ('periodic','periodic','periodic')
    _fourier_symbol: Any = field(init=False, repr=False)
    _bulk_driving_forces: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = -self.M * self.gab * self.k_squared()
        self.pot_factor = 9 / (2*self.eps**2)
        if self.bulk_driving_forces is not None:
            self._bulk_driving_forces = self.vg.to_backend(self.bulk_driving_forces)

        if self.fast:
            self.project_to_simplex = self._sloppy_simplex_projection
        else:
            self.project_to_simplex = self._euclidean_simplex_projection
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def rhs_analytic(self, t, phis):
        sum_phi_squared = sum(phi**2 for phi in phis)
        df_dphi = []

        for phi in phis:
            grad_term = -spv.laplacian(phi)
            pot_term = self.pot_factor * (3*phi*(sum_phi_squared - phi**2) + phi**3 - phi)
            df_dphi.append(grad_term + pot_term)

        mean_df = sum(df_dphi) / len(df_dphi)
        return tuple(-self.M * self.gab * (df - mean_df) for df in df_dphi)
    
    def _sloppy_simplex_projection(self, phis):
        # hard Gibbs simplex projection: phi>=0 and sum_p phi_p = 1
        phis = self.vg.lib.clip(phis, min=0.0)
        sum = self.vg.sum(phis, dim=0, keepdim=True)
        return phis / sum

    def _euclidean_simplex_projection(self, phis):
        """Euclidean projection onto {phi>=0, sum_p phi=1} per voxel."""
        N = phis.shape[0]
        u = self.vg.sort(phis, dim=0, descending=True)
        cssv = self.vg.cumsum(u, dim=0) - 1.0

        k = self.vg.arange(1, N+1).reshape(N, 1, 1, 1)
        cond = (u - cssv / k) > 0

        N_active = self.vg.sum(cond, dim=0, keepdim=False)
        N_active = self.vg.lib.clip(N_active, min=1)

        idx = (N_active - 1)[None, ...]
        theta_num = self.vg.take_along_dim(cssv, idx, dim=0)
        theta = theta_num / N_active[None, ...]

        return self.vg.lib.clip(phis - theta, min=0.0)
    
    def _calc_multiwell_derivatives(self, phis):
        sum_phi_squared = self.vg.sum(phis**2, dim=0, keepdim=True)
        df_dphi = 3*phis*(sum_phi_squared - phis**2) + phis**3 - phis
        return self.pot_factor*df_dphi

    def _bulk_driving_term(self, phis):
        """Return pairwise bulk driving forces without an O(N**2) field tensor."""
        if self._bulk_driving_forces is None:
            return 0.0

        forces = self._bulk_driving_forces.reshape((-1,) + (1,) * (phis.ndim - 1))
        sum_phi2 = self.vg.sum(phis**2, dim=0, keepdim=True)
        sum_force_phi = self.vg.sum(forces * phis, dim=0, keepdim=True)
        sum_force_phi2 = self.vg.sum(forces * phis**2, dim=0, keepdim=True)
        return 3 / self.eps * (
            phis**2 * (sum_force_phi - forces)
            + phis * (sum_force_phi2 - forces * sum_phi2)
        )

    def rhs(self, t, phis):
        r"""Multi-phase Allen-Cahn equation
        
        Microstructural evolution of the phase fractions :math:`\phi_\alpha`,
        :math:`\alpha=1,\ldots,N`, governed by the multiphase-field model.
        :math:`M` denotes the mobility which is the same for all phase-pairs,
        :math:`\epsilon` controls the diffuse interface width,
        :math:`\gamma` denotes the interfacial energy.
        The laplacian leads to a phase evolution driven by
        curvature minimization which can be controlled by setting
        ``curvature=`` in range :math:`[0,1]`.

        Args:
            t (float): Current time.
            phis (array-like): phase fractions.

        Returns:
            Backend array of the same shape as ``\phi`` containing ``d\phi/dt``.
        """
        phis = self.project_to_simplex(phis)

        # Gradient term
        phi_pad = self.pad_bc(phis)
        dfgrad_dphi = -self.vg.laplace(phi_pad)
        # This one cancels because of pairwise interactions
        # sum_dfgrad_dphi = self.vg.sum(dfgrad_dphi, dim=0, keepdim=True)
        # dfgrad_dphi += sum_dfgrad_dphi

        # Potential term
        dfpot_dphi = self._calc_multiwell_derivatives(phis)
    
        df_dphi = dfgrad_dphi + dfpot_dphi
        dphi = self.gab * (df_dphi - self.vg.mean(df_dphi, dim=0, keepdim=True))
        return self.M * (self._bulk_driving_term(phis) - dphi)

# @dataclass
# class CurvatureMultiPhaseAllenCahn(MultiPhaseAllenCahn):
#     curvature: float = 1.0

#     def rhs_analytic(self, t, phis):
#         sum_phi_squared = sum(phi**2 for phi in phis)
#         df_dphi = []

#         for phi in phis:
#             grad = spv.gradient(phi)
#             laplace = spv.laplacian(phi)
#             norm_grad = sp.sqrt(grad.dot(grad))
#             unit_normal = grad / norm_grad
#             curv = norm_grad * spv.divergence(unit_normal)

#             grad_term = -self.curvature * laplace - (1 - self.curvature) * (laplace - curv)
#             pot_term = self.pot_factor * (3*phi*(sum_phi_squared - phi**2) + phi**3 - phi)
#             df_dphi.append(grad_term + pot_term)

#         mean_df = sum(df_dphi) / len(df_dphi)
#         return tuple(-self.M * self.gab * (df - mean_df) for df in df_dphi)
    
#     def _calc_multiwell_derivatives(self, phis):
#         sum_phi_squared = self.vg.sum(phis**2, dim=0, keepdim=True)
#         df_dphi = 3*phis*(sum_phi_squared - phis**2) + phis**3 - phis
#         return self.pot_factor*df_dphi

#     def rhs(self, t, phis):
#         phis = self.project_to_simplex(phis)

#         # Gradient term
#         phi_pad = self.pad_bc(phis)
#         dfgrad_dphi = -self.curvature*self.vg.laplace(phi_pad)
#         dfgrad_dphi -= (1-self.curvature) * self.vg.normal_laplace(phi_pad)
#         # This one cancels because of pairwise interactions
#         # sum_dfgrad_dphi = self.vg.sum(dfgrad_dphi, dim=0, keepdim=True)
#         # dfgrad_dphi += sum_dfgrad_dphi

#         # Potential term
#         dfpot_dphi = self._calc_multiwell_derivatives(phis)
    
#         df_dphi = dfgrad_dphi + dfpot_dphi
#         dphi = self.gab * (df_dphi - self.vg.mean(df_dphi, dim=0, keepdim=True))
#         # dphi += 3 / self.eps * (phia + phib) * phia * phib
#         return - self.M * dphi

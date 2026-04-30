from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import sympy as sp
import sympy.vector as spv
import warnings
from .voxelgrid import VoxelGrid

# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

class ODE(ABC):
    @property
    @abstractmethod
    def order(self) -> int:
        """Spatial order of convergence for numerical right-hand side."""
        pass

    @abstractmethod
    def rhs_analytic(self, t, u):
        """Sympy expression of the problem right-hand side.

        Args:
            t (float): Current time.
            u : Sympy function of current state.

        Returns:
            Sympy function of problem right-hand side.
        """
        pass

    @abstractmethod
    def rhs(self, t, u):
        """Numerical right-hand side of the ODE system.

        Args:
            t (float): Current time.
            u (array): Current state.

        Returns:
            Same type as ``u`` containing the time derivative.
        """
        pass

    def pad_bc(self, u):
        """Function to pad and impose boundary conditions.

        Enables applying boundary conditions on u within and
        outside of the right-hand-side function.

        Args:
            u : field

        Returns:
            Field padded with boundary values.
        """
        return self._pad_bc(u)

    def initialize_boundary_conditions(self):
        bc = getattr(self, "bc", None)
        if bc is None or bc == 'fully_periodic':
            self.bc = (('periodic', None), ('periodic', None), ('periodic', None))
            self._pad_bc = self.vg.bc.pad_periodic
            return

        if len(bc) != 3:
            raise ValueError("bc must contain exactly three axis entries ordered as (x, y, z).")

        normalized_bc = []
        for axis_bc in bc:
            if isinstance(axis_bc, str):
                if axis_bc == 'dirichlet':
                    raise ValueError("Dirichlet BCs require explicit values ('dirichlet', (left, right)).")
                if axis_bc not in {'periodic', 'neumann'}:
                    raise ValueError(f"Unsupported BC type: {axis_bc}")
                normalized_bc.append((axis_bc, None))
                continue

            if len(axis_bc) != 2:
                raise ValueError(
                    "Each axis boundary specification must be either a string or "
                    "a tuple like ('dirichlet', (left, right))."
                )

            kind, values = axis_bc
            if kind not in {'periodic', 'dirichlet', 'neumann'}:
                raise ValueError(f"Unsupported BC type: {kind}")

            if kind == 'dirichlet':
                if values is None or len(values) != 2:
                    raise ValueError("Dirichlet BCs require two boundary values.")
                if self.vg.convention == 'cell_center':
                    warnings.warn(
                        "Applying Dirichlet BCs on a cell_center grid "
                        "reduces the spatial order of convergence to 0.5!"
                        )
                normalized_bc.append((kind, tuple(values)))
            else:
                if values is not None:
                    raise ValueError(f"{kind} BCs do not accept boundary values.")
                normalized_bc.append((kind, None))
        self.bc = tuple(normalized_bc)

        if self.bc_type == ('periodic','periodic','periodic'):
            self._pad_bc = self.vg.bc.pad_periodic

        elif self.bc_type == ('dirichlet','periodic','periodic'):
            self._pad_bc = lambda field: self.vg.bc.pad_dirichlet_periodic(
                field, self.bc[0][1][0], self.bc[0][1][1]
            )
        elif self.bc_type == ('neumann','periodic','periodic'):
            self._pad_bc = self.vg.bc.pad_zero_flux_periodic
        else:
            self._pad_bc = lambda field: self.vg.bc.pad_bc(field, self.bc)

    @property
    def bc_type(self):
        """Boundary-condition metadata for the current problem."""
        return tuple(axis_bc[0] for axis_bc in self.bc)


class SemiLinearODE(ODE):
    @property
    @abstractmethod
    def fourier_symbol(self):
        """Symbol of the highest order spatial operator
        
        The symbol of an operator is its representation in the
        Fourier (spectral) domain. For instance the:
        - Laplacian operator $\nabla^2$ has a symbol $-k^2$,
        - diffusion operator $D\nabla^2$ corresponds to $-k^2D$
        
        The symbol is required for pseudo-spectral timesteppers.
        """
        pass

    def verify_fft_bc_config(self):
        x_bc, _, _ = self.bc_type
        nonperiodic_axes = tuple(
            axis for axis, kind in zip(('x', 'y', 'z'), self.bc_type)
            if kind != 'periodic'
        )

        if len(nonperiodic_axes) > 1:
            raise ValueError(
                "FFT-based timesteppers currently support at most one non-periodic axis, "
                f"got {self.bc_type}."
            )

        if len(nonperiodic_axes) == 1 and nonperiodic_axes[0] != 'x':
            raise NotImplementedError(
                "FFT-based timesteppers currently only implement the single non-periodic axis "
                f"case for x; got {self.bc_type}. Axis permutation is not implemented yet."
            )

        if x_bc == 'periodic':
            self._pad_fft_bc = self.vg.bc.pad_fft_periodic
        elif x_bc == 'dirichlet':
            self._pad_fft_bc = self.vg.bc.pad_fft_dirichlet_periodic
        elif x_bc == 'neumann':
            self._pad_fft_bc = self.vg.bc.pad_fft_zero_flux_periodic
        else:
            raise ValueError(
                "FFT-based timesteppers only support periodic, dirichlet, or "
                f"neumann boundary conditions in x, got {x_bc}."
            )
        
    def k_squared(self):
        """Helper to choose k^2 for fourier symbol based on BCs."""
        if self.bc_type[0] in {'dirichlet', 'neumann'}:
            return self.vg.rfft_k_squared_nonperiodic()
        else:
            # Note: this is technically not correct for non-periodic BCs,
            # but this is irrelevant if timestepper does not use FFT, while
            # timesteppers which use FFT call verify_fft_bc_config()
            return self.vg.rfft_k_squared()

    def pad_fft_bc(self, u):
        return self._pad_fft_bc(u)


class SmoothedBoundaryODE(ODE):
    @property
    @abstractmethod
    def mask(self) -> Any | float:
        """A field (same shape as the state) that remains fixed."""
        pass


@dataclass
class ReactionDiffusion(SemiLinearODE):
    vg: VoxelGrid
    D: float
    f: Callable | None = None
    A: float = 0.25
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
        if self.mask is None:
            self.mask = self.vg.lib.ones(self.vg.shape)
            self.mask = self.vg.init_scalar_field(self.mask)
            self.mask = self.vg.pad_periodic(\
                        self.vg.bc.trim_boundary_nodes(self.mask))
            self.norm = 1.0
        else:
            self.mask = self.vg.init_scalar_field(self.mask)
            mask_0 = self.mask[:,0,:,:]
            mask_1 = self.mask[:,-1,:,:]
            self.mask = self.vg.pad_periodic(\
                        self.vg.bc.trim_boundary_nodes(self.mask))
            if self.bc_type[0] != 'periodic':
                self.mask = self.vg.set(self.mask, (__, 0,_i_,_i_), mask_0)
                self.mask = self.vg.set(self.mask, (__,-1,_i_,_i_), mask_1)

            self.norm = self.vg.lib.sqrt(self.vg.gradient_norm_squared(self.mask))
            self.mask = self.vg.lib.clip(self.mask, 1e-4, 1)

            x_bc, y_bc, z_bc = self.bc
            if x_bc[0] == 'dirichlet':
                self.bc = (
                    ('dirichlet', (x_bc[1][0] * self.mask[:,0,:,:], x_bc[1][1] * self.mask[:,-1,:,:])),
                    y_bc, z_bc)
                self.initialize_boundary_conditions()

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
    potential: Callable | None = None
    bc: tuple = ('neumann','neumann','neumann')
    _fourier_symbol: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = -self.M * self.gab* self.k_squared()
        if self.potential is None:
            self.potential = lambda u, lib=None: 18 / self.eps * u * (1-u) * (1-2*u)

    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol

    def _eval_potential(self, phi, lib):
        """Evaluate phasefield potential"""
        try:
            return self.potential(phi, lib)
        except TypeError:
            return self.potential(phi)

    def rhs_analytic(self, t, phi):
        grad = spv.gradient(phi)
        laplace  = spv.laplacian(phi)
        norm_grad = sp.sqrt(grad.dot(grad))

        # Curvature equals |∇ψ| ∇·(∇ψ/|∇ψ|)
        unit_normal = grad / norm_grad
        curv = norm_grad * spv.divergence(unit_normal)
        n_laplace = laplace - (1-self.curvature)*curv
        df_dphi = self.gab * (n_laplace - self._eval_potential(phi, sp)/(2*self.eps)) \
                  + 3/self.eps * phi * (1-phi) * self.force
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
        df_dphi = self.gab * (laplace + n_laplace - potential/2/self.eps)\
                  + 3/self.eps * phi * (1-phi) * self.force
        return self.M * df_dphi


@dataclass
class MultiPhaseAllenCahn(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 3.0
    gab: float = 1.0
    M: float = 1.0
    force: float = 0.0
    curvature: float = 1.0
    potential: str = 'well'
    fast: bool = True
    bc: tuple = ('periodic','periodic','periodic')
    _fourier_symbol: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        self.initialize_boundary_conditions()
        self._fourier_symbol = -self.M * self.gab * self.k_squared()
        if self.potential == 'well':
            self.pot_factor = 9 / (2*self.eps**2)
            self.calc_potential_derivatives = self._calc_well_derivatives
        # elif self.potential == 'obstacle':
        #     self.pot_factor = 16 / (self.eps**2 * self.vg.lib.pi**2)
        #     self.calc_potential_derivatives = self._calc_obstacle_derivatives
        else:
            raise ValueError(f"Unknown potential type: {self.potential}")

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
            grad = spv.gradient(phi)
            laplace = spv.laplacian(phi)
            norm_grad = sp.sqrt(grad.dot(grad))
            unit_normal = grad / norm_grad
            curv = norm_grad * spv.divergence(unit_normal)

            grad_term = -self.curvature * laplace - (1 - self.curvature) * (laplace - curv)
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
    
    def _calc_well_derivatives(self, phis):
        sum_phi_squared = self.vg.sum(phis**2, dim=0, keepdim=True)
        df_dphi = 3*phis*(sum_phi_squared - phis**2) + phis**3 - phis
        return self.pot_factor*df_dphi

    def _calc_obstacle_derivatives(self, phis):
        # sum_phi = self.vg.sum(phis, dim=0, keepdim=True)
        return -self.pot_factor * phis

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
        dfgrad_dphi = -self.curvature*self.vg.laplace(phi_pad)
        dfgrad_dphi -= (1-self.curvature) * self.vg.normal_laplace(phi_pad)
        # This one cancels because of pairwise interactions
        # sum_dfgrad_dphi = self.vg.sum(dfgrad_dphi, dim=0, keepdim=True)
        # dfgrad_dphi += sum_dfgrad_dphi

        # Potential term
        dfpot_dphi = self.calc_potential_derivatives(phis)
    
        df_dphi = dfgrad_dphi + dfpot_dphi
        return - self.M * self.gab * (df_dphi - self.vg.mean(df_dphi, dim=0, keepdim=True))
    

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

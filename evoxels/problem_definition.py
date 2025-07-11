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

    @property
    @abstractmethod
    def bc_type(self) -> str:
        """E.g. 'periodic', 'dirichlet', or 'neumann'."""
        pass

    @abstractmethod
    def pad_bc(self, u):
        """Function to pad and impose boundary conditions.

        Enables applying boundary conditions on u within and
        outside of the right-hand-side function.

        Args:
            u : field

        Returns:
            Field padded with boundary values.
        """
        pass


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
    BC_type: str
    bcs: tuple = (0,0)
    f: Callable | None = None
    A: float = 0.25
    _fourier_symbol: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        if self.f is None:
            self.f = lambda c=None, t=None, lib=None: 0

        if self.BC_type == 'periodic':
            bc_fun = self.vg.bc.pad_periodic
            self.pad_boundary = lambda field, bc0, bc1: bc_fun(field)
            k_squared = self.vg.rfft_k_squared()
        elif self.BC_type == 'dirichlet':
            self.pad_boundary = self.vg.bc.pad_dirichlet_periodic
            if self.vg.convention == 'cell_center':
                warnings.warn(
                    "Applying Dirichlet BCs on a cell_center grid "
                    "reduces the spatial order of convergence to 0.5!"
                    )
            k_squared = self.vg.fft_k_squared_nonperiodic()
        elif self.BC_type == 'neumann':
            bc_fun = self.vg.bc.pad_zero_flux_periodic
            self.pad_boundary = lambda field, bc0, bc1: bc_fun(field)
            k_squared = self.vg.fft_k_squared_nonperiodic()

        self._fourier_symbol = -self.D * self.A * k_squared
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol
    
    def _eval_f(self, c, t, lib):
        """Evaluate source/forcing term using ``self.f``."""
        try:
            return self.f(c, t, lib)
        except TypeError:
            return self.f(c, t)
        
    @property
    def bc_type(self):
        return self.BC_type

    def pad_bc(self, u):
        return self.pad_boundary(u, self.bcs[0], self.bcs[1])
    
    def rhs_analytic(self, u, t):
        return self.D*spv.laplacian(u) + self._eval_f(u, t, sp)
    
    def rhs(self, u, t):
        laplace = self.vg.laplace(self.pad_bc(u))
        update = self.D * laplace + self._eval_f(u, t, self.vg.lib)
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
            if self.BC_type != 'periodic':
                self.mask = self.vg.set(self.mask, (__, 0,_i_,_i_), mask_0)
                self.mask = self.vg.set(self.mask, (__,-1,_i_,_i_), mask_1)

            self.norm = self.vg.lib.sqrt(self.vg.gradient_norm_squared(self.mask))
            self.mask = self.vg.lib.clip(self.mask, 1e-4, 1)

            self.bcs = (self.bcs[0] * self.mask[:,0,:,:],
                        self.bcs[1] * self.mask[:,-1,:,:])

    def pad_bc(self, u):
        return self.pad_boundary(u, self.bcs[0], self.bcs[1])

    def rhs_analytic(self, mask, u, t):
        grad = spv.gradient(u)
        norm_grad = sp.sqrt(grad.dot(grad))

        divergence = spv.divergence(self.D*(grad - u/mask*spv.gradient(mask)))
        du = divergence + norm_grad*self.bc_flux + mask*self._eval_f(u/mask, t, sp)
        return du

    def rhs(self, u, t):
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
                 self.mask[:,1:-1,1:-1,1:-1]*self._eval_f(u/self.mask[:,1:-1,1:-1,1:-1], t, self.vg.lib)
        return update


@dataclass
class PeriodicCahnHilliard(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 3.0
    D: float = 1.0
    mu_hom: Callable | None = None
    A: float = 0.25
    _fourier_symbol: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._fourier_symbol = -2 * self.eps * self.D * self.A * k_squared**2
        if self.mu_hom is None:
            self.mu_hom = lambda c, lib=None: 18 / self.eps * c * (1 - c) * (1 - 2 * c)
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol
    
    @property
    def bc_type(self):
        return 'periodic'
    
    def pad_bc(self, u):
        return self.vg.bc.pad_periodic(u)
    
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
class AllenCahnEquation(SemiLinearODE):
    vg: VoxelGrid
    eps: float = 2.0
    gab: float = 1.0
    M: float = 1.0
    force: float = 0.0
    curvature: float = 0.01
    potential: Callable | None = None
    _fourier_symbol: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Precompute factors required by the spectral solver."""
        k_squared = self.vg.rfft_k_squared()
        self._fourier_symbol = -2 * self.M * self.gab* k_squared
        if self.potential is None:
            self.potential = lambda u, lib=None: 18 / self.eps * u * (1-u) * (1-2*u)
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol
    
    @property
    def bc_type(self):
        return 'neumann'

    def pad_bc(self, u):
        return self.vg.bc.pad_zero_flux(u)

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
        df_dphi = self.gab * (2*n_laplace - self._eval_potential(phi, sp)/self.eps) \
                  + 3/self.eps * phi * (1-phi) * self.force
        return self.M * df_dphi

    def rhs(self, phi, t):
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
            phi (array-like): order parameter.
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``\phi`` containing ``d\phi/dt``.
        """
        phi = self.vg.lib.clip(phi, 0, 1)
        potential = self._eval_potential(phi, self.vg.lib)
        phi_pad = self.pad_bc(phi)
        laplace = self.curvature*self.vg.laplace(phi_pad)
        n_laplace = (1-self.curvature) * self.vg.normal_laplace(phi_pad)
        df_dphi = self.gab * (2.0 * (laplace+n_laplace) - potential/self.eps)\
                  + 3/self.eps * phi * (1-phi) * self.force
        return self.M * df_dphi


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
        k_squared = self.vg.rfft_k_squared()
        self._fourier_symbol = - max(self.D_A, self.D_B) * k_squared
        if self.interaction is None:
            self.interaction = lambda u, lib=None: u[0] * u[1]**2
    
    @property
    def order(self):
        return 2

    @property
    def fourier_symbol(self):
        return self._fourier_symbol
    
    @property
    def bc_type(self):
        return 'periodic'

    def pad_bc(self, u):
        return self.vg.bc.pad_periodic(u)

    def _eval_interaction(self, u, lib):
        """Evaluate interaction term"""
        try:
            return self.interaction(u, lib)
        except TypeError:
            return self.interaction(u)

    def rhs_analytic(self, u, t):
        interaction = self._eval_interaction(u, sp)
        dc_A = self.D_A*spv.laplacian(u[0]) - interaction + self.feed * (1-u[0])
        dc_B = self.D_B*spv.laplacian(u[1]) + interaction - self.kill * u[1]
        return (dc_A, dc_B)

    def rhs(self, u, t):
        r"""Two-component reaction-diffusion system
        
        Use batch channels for multiple species:
        - Species A with concentration c_A = u[0]
        - Species B with concentration c_B = u[1]

        Args:
            u (array-like): species
            t (float): Current time.

        Returns:
            Backend array of the same shape as ``u`` containing ``du/dt``.
        """
        interaction = self._eval_interaction(u, self.vg.lib)
        u_pad = self.pad_bc(u)
        laplace = self.vg.laplace(u_pad)
        dc_A = self.D_A*laplace[0] - interaction + self.feed * (1-u[0])
        dc_B = self.D_B*laplace[1] + interaction - self.kill * u[1]
        return self.vg.lib.stack((dc_A, dc_B), 0)

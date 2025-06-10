import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Any

# Shorthand for all elements [:] in slicing logic
__ = slice(None)


@dataclass
class Grid:
    """Handles most basic properties"""
    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    convention: str


@dataclass
class VoxelGrid:
    """Abstract backend adapter: handles array conversion and padding."""
    def __init__(self, grid: Grid, lib):
        self.shape   = grid.shape
        self.origin  = grid.origin
        self.spacing = grid.spacing
        self.convention = grid.convention
        self.lib = lib

        # Other grid information
        self.size    = self.shape[0] * self.shape[1] * self.shape[2]
        self.div_dx2 = 1/self.to_backend(np.array(self.spacing))**2

        # Boundary conditions
        if self.convention == 'staggered_x':
            self.apply_dirichlet_periodic_BC = self.apply_dirichlet_periodic_BC_staggered_x
            self.apply_zero_flux_periodic_BC = self.apply_zero_flux_periodic_BC_staggered_x
            self.apply_periodic_BC           = self.apply_periodic_BC_staggered_x
        else:
            self.apply_dirichlet_periodic_BC = self.apply_dirichlet_periodic_BC_cell_center
            self.apply_zero_flux_periodic_BC = self.apply_zero_flux_periodic_BC_cell_center
            self.apply_periodic_BC           = self.apply_periodic_BC_cell_center
    
    # Operate on fields
    def to_backend(self, field):        raise NotImplementedError
    def to_numpy(self, field):          raise NotImplementedError
    def pad_periodic(self, field):      raise NotImplementedError
    def pad_zeros(self, field):         raise NotImplementedError
    def fftn(self, field):              raise NotImplementedError
    def real_of_ifftn(self, field):     raise NotImplementedError
    def expand_dim(self, field, dim):   raise NotImplementedError
    def squeeze(self, field, dim):      raise NotImplementedError
    def set(self, field, index, value): raise NotImplementedError
    
    def axes(self) -> Tuple[Any, ...]:
        """ Returns the 1D coordinate arrays along each axis. """
        return tuple(self.lib.arange(0, n) * self.spacing[i] + self.origin[i]
                     for i, n in enumerate(self.shape))
    
    def fft_axes(self) -> Tuple[Any, ...]:
        return tuple(2 * self.lib.pi * self.lib.fft.fftfreq(points, step)
                     for points, step in zip(self.shape, self.spacing))
    
    def rfft_axes(self) -> Tuple[Any, ...]:
        return tuple(2 * self.lib.pi * self.lib.fft.rfftfreq(points, step)
                     for points, step in zip(self.shape, self.spacing))
    
    def meshgrid(self) -> Tuple[Any, ...]:
        """ Returns full 3D mesh grids for each axis. """
        ax = self.axes()
        return tuple(self.lib.meshgrid(*ax, indexing='ij'))
    
    def fft_mesh(self) -> Tuple[Any, ...]:
        fft_axes = self.fft_axes()
        return tuple(self.lib.meshgrid(*fft_axes, indexing='ij'))
    
    # def rfft_mesh(self) -> Tuple[Any, ...]:
    #     rfft_axes = self.rfft_axes()
    #     return tuple(self.lib.meshgrid(*rfft_axes, indexing='ij'))
    
    def fft_k_squared(self):
        fft_axes = self.fft_axes()
        kx, ky, kz = self.lib.meshgrid(*fft_axes, indexing='ij')
        return kx**2 + ky**2 + kz**2
    
    def rfft_k_squared(self):
        fft_axes = self.fft_axes()
        rfft_axes = self.rfft_axes()
        kx, ky, kz = self.lib.meshgrid(fft_axes[0], fft_axes[1], rfft_axes[2], indexing='ij')
        return kx**2 + ky**2 + kz**2
    
    def pad_with_ghost_nodes(self, field):
        field = self.pad_periodic(field)
        if self.convention == 'staggered_x':
            # For staggered grid we don't need ghost nodes in x
            field = field[:,1:-1,:,:]
        return field
    
    def return_inner_values(self, field):
        if self.convention == 'staggered_x':
            inner_field = field[:,:,1:-1,1:-1]
        else :
            inner_field = field[:,1:-1,1:-1,1:-1]
        return inner_field
    
    def init_field_from_numpy(self, array: np.ndarray):
        field = self.to_backend(array)
        field = self.expand_dim(field, 0)
        # field = self.pad_with_ghost_nodes(field)
        return field

    def export_field_to_numpy(self, field):
        inner = self.squeeze(self.return_inner_values(field), 0)
        return self.to_numpy(inner)
    
    def calc_field_average(self, field):
        # TODO: this currently only works for batch dimension of 1
        inner_field = self.return_inner_values(field)
        if self.convention == 'cell_center':
            average = self.lib.mean(inner_field)
        elif self.convention == 'staggered_x':
            # Count first and last slice as half cells
            average = self.lib.sum(inner_field[1:-1,:,:]) \
                      + 0.5*self.lib.sum(inner_field[ 0,:,:]) \
                      + 0.5*self.lib.sum(inner_field[-1,:,:])
            average /= self.size
        return average
    
    def _apply_yz_periodic(self, field):
        field = self.set(field, (__,__, 0,__), field[:,:,-2,:])
        field = self.set(field, (__,__,-1,__), field[:,:, 1,:])
        field = self.set(field, (__,__,__, 0), field[:,:,:,-2])
        field = self.set(field, (__,__,__,-1), field[:,:,:, 1])
        return field

    def apply_periodic_BC_cell_center(self, field):
        """
        Periodic boundary conditions in all directions.
        Consistent with cell centered grid.
        """
        field = self.set(field, (__, 0,__,__), field[:,-2,:,:])
        field = self.set(field, (__,-1,__,__), field[:, 1,:,:])
        return self._apply_yz_periodic(field)
    
    def apply_periodic_BC_staggered_x(self, field):
        raise NotImplementedError
    
    def apply_dirichlet_periodic_BC_cell_center(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with cell centered grid,
        but loss of 2nd order convergence.
        """
        field = self.set(field, (__, 0,__,__), 2.0*bc0 - field[:, 1,:,:])
        field = self.set(field, (__,-1,__,__), 2.0*bc1 - field[:,-2,:,:])
        return self._apply_yz_periodic(field)
    
    def apply_dirichlet_periodic_BC_staggered_x(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with staggered_x grid,
        maintains 2nd order convergence.
        """
        field = self.set(field, (__, 0,__,__), bc0)
        field = self.set(field, (__,-1,__,__), bc1)
        return self._apply_yz_periodic(field)
    
    def apply_zero_flux_periodic_BC_cell_center(self, field):
        field = self.set(field, (__, 0,__,__), field[:, 1,:,:])
        field = self.set(field, (__,-1,__,__), field[:,-2,:,:])
        return self._apply_yz_periodic(field)

    def apply_zero_flux_periodic_BC_staggered_x(self, field):
        """
        The following comes out of on interpolation polynomial p with
        p'(0) = 0, p(dx) = f(dx,...), p(2*dx) = f(2*dx,...)
        and then use p(0) for the ghost cell. 
        This should be of sufficient order of f'(0) = 0, and even better if
        also f'''(0) = 0 (as it holds for cos(k*pi*x)  )
        """
        fac1 =  4/3
        fac2 =  1/3
        field = self.set(field, (__, 0,__,__), fac1*field[:, 1,:,:] - fac2*field[:, 2,:,:])
        field = self.set(field, (__,-1,__,__), fac1*field[:,-2,:,:] - fac2*field[:,-3,:,:])
        return self._apply_yz_periodic(field)

    def calc_laplace(self, field):
        """
        Calculate laplace based on compact 2nd order stencil.
        Returned field has same shape as the input field (padded with zeros)
        """
        # Manual indexing is ~10x faster than conv3d with laplace kernel in torch
        laplace = \
            (field[:, 2:, 1:-1, 1:-1] + field[:, :-2, 1:-1, 1:-1]) * self.div_dx2[0] + \
            (field[:, 1:-1, 2:, 1:-1] + field[:, 1:-1, :-2, 1:-1]) * self.div_dx2[1] + \
            (field[:, 1:-1, 1:-1, 2:] + field[:, 1:-1, 1:-1, :-2]) * self.div_dx2[2] - \
             2 * field[:, 1:-1, 1:-1, 1:-1] * self.lib.sum(self.div_dx2)
        laplace = self.pad_zeros(laplace)
        return laplace


class VoxelGridTorch(VoxelGrid):
    def __init__(self, grid: Grid, precision='float32', device: str='cuda'):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch backend selected but 'torch' is not installed.")
        self.torch = torch

        # Handle torch device
        self.device = torch.device(device)
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn("CUDA not available, defaulting device to cpu. To avoid this warning, set device=torch.device('cpu')")
        torch.set_default_device(self.device)

        # Handle torch precision
        if precision == 'float32':
            self.precision = torch.float32
        if precision == 'float64':
            self.precision = torch.float64

        super().__init__(grid, torch)

    def to_backend(self, np_arr):
        return self.torch.tensor(np_arr, dtype=self.precision, device=self.device)

    def to_numpy(self, field):
        return field.cpu().numpy()

    def pad_periodic(self, field):
        return self.torch.nn.functional.pad(field, (1,1,1,1,1,1), mode='circular')

    def pad_zeros(self, field):
        return self.torch.nn.functional.pad(field, (1,1,1,1,1,1), mode='constant', value=0)

    def fftn(self, field):
        return self.torch.fft.fftn(field)

    def rfftn(self, field):
        return self.torch.fft.rfftn(field)

    def irfftn(self, field, shape):
        return self.torch.fft.irfftn(field, s=shape)

    def real_of_ifftn(self, field):
        return self.torch.real(self.torch.fft.ifftn(field))

    def expand_dim(self, field, dim):
        return field.unsqueeze(dim)

    def squeeze(self, field, dim):
        return self.torch.squeeze(field, dim)

    def set(self, field, index, value):
        field[index] = value
        return field


class VoxelGridJax(VoxelGrid):
    def __init__(self, grid: Grid, precision='float32'):
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX backend selected but 'jax' is not installed.")
        self.jnp = jnp
        self.precision = precision
        super().__init__(grid, jnp)

    def to_backend(self, np_arr):
        return self.jnp.array(np_arr, dtype=self.precision)

    def to_numpy(self, field):
        return np.array(field)

    def pad_periodic(self, field):
        pad_width = ((0, 0), (1, 1), (1, 1), (1, 1))
        return self.jnp.pad(field, pad_width, mode='wrap')

    def pad_zeros(self, field):
        pad_width = ((0, 0), (1, 1), (1, 1), (1, 1))
        return self.jnp.pad(field, pad_width, mode='constant', constant_values=0)

    def fftn(self, field):
        return self.jnp.fft.fftn(field)

    def rfftn(self, field):
        return self.jnp.fft.rfftn(field)

    def irfftn(self, field, shape):
        return self.jnp.fft.irfftn(field, s=shape)

    def real_of_ifftn(self, field):
        return self.jnp.fft.ifftn(field).real

    def expand_dim(self, field, dim):
        return self.jnp.expand_dims(field, dim)

    def squeeze(self, field, dim):
        return self.jnp.squeeze(field, axis=dim)

    def set(self, field, index, value):
        return field.at[index].set(value)

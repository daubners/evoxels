import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Any

# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

CENTER = (__,  _i_, _i_, _i_)
LEFT   = (__, slice(None,-2), _i_, _i_)
RIGHT  = (__, slice(2, None), _i_, _i_)
BOTTOM = (__, _i_, slice(None,-2), _i_)
TOP    = (__, _i_, slice(2, None), _i_)
BACK   = (__, _i_, _i_, slice(None,-2))
FRONT  = (__, _i_, _i_, slice(2, None))

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
        self.div_dx = 1/self.to_backend(np.array(self.spacing))
        self.div_dx2 = 1/self.to_backend(np.array(self.spacing))**2

        # Boundary conditions
        if self.convention == 'staggered_x':
            self.trim_boundary_nodes       = self.trim_boundary_nodes_staggered_x
            self.pad_dirichlet_periodic_BC = self.pad_dirichlet_periodic_BC_staggered_x
            self.pad_zero_flux_periodic_BC = self.pad_zero_flux_periodic_BC_staggered_x
            self.pad_periodic_BC           = self.pad_periodic_BC_staggered_x
            self.pad_zero_flux_BC          = self.pad_zero_flux_BC_staggered_x
        else:
            self.trim_boundary_nodes       = self.trim_boundary_nodes_cell_center
            self.pad_dirichlet_periodic_BC = self.pad_dirichlet_periodic_BC_cell_center
            self.pad_zero_flux_periodic_BC = self.pad_zero_flux_periodic_BC_cell_center
            self.pad_periodic_BC           = self.pad_periodic_BC_cell_center
            self.pad_zero_flux_BC          = self.pad_zero_flux_BC_cell_center
    
    # Operate on fields
    def to_backend(self, field):
        """Convert a NumPy array to the backend representation."""
        raise NotImplementedError

    def to_numpy(self, field):
        """Convert a backend array to ``numpy.ndarray``."""
        raise NotImplementedError

    def pad_periodic(self, field):
        """Pad a field with periodic boundary conditions."""
        raise NotImplementedError

    def pad_zeros(self, field):
        """Pad a field with zeros."""
        raise NotImplementedError

    def fftn(self, field):
        """Compute the n-dimensional discrete Fourier transform."""
        raise NotImplementedError

    def real_of_ifftn(self, field):
        """Return the real part of the inverse FFT."""
        raise NotImplementedError

    def expand_dim(self, field, dim):
        """Add a singleton dimension to ``field``."""
        raise NotImplementedError

    def squeeze(self, field, dim):
        """Remove ``dim`` from ``field``."""
        raise NotImplementedError

    def concatenate(self, fieldlist, dim):
        """Concatenate fields in ``fieldlist``."""
        raise NotImplementedError

    def set(self, field, index, value):
        """Set ``field[index]`` to ``value`` and return ``field``."""
        raise NotImplementedError
    
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
    
    def init_scalar_field(self, array):
        """Convert and pad a NumPy array for simulation."""
        field = self.to_backend(array)
        field = self.expand_dim(field, 0)
        return field
    
    def trim_boundary_nodes_cell_center(self, field):
        return field
    
    def trim_boundary_nodes_staggered_x(self, field):
        """Trim boundary nodes of ``field`` for staggered grids."""
        if field.shape[1] == self.shape[0]:
            return field[:,1:-1,:,:]
        else:
            raise ValueError(
                f"The provided field must have the shape {self.shape}."
            )
    
    def trim_ghost_nodes(self, field):
        if (self.convention == 'cell_center') and \
           (field[0,1:-1,1:-1,1:-1].shape == self.shape):
            return field[:,1:-1,1:-1,1:-1]
        elif (self.convention == 'staggered_x') and \
             (field[0,:,1:-1,1:-1].shape == self.shape):
            return field[:,:,1:-1,1:-1]
        else:
            raise ValueError(
                f"The provided field has the wrong shape {self.shape}."
            )

    def export_scalar_field_to_numpy(self, field):
        """Export backend field back to NumPy."""
        array = self.to_numpy(self.squeeze(field, 0))
        return array
    
    def calc_field_average(self, field):
        """Return the spatial average of ``field``."""
        if field.shape[1:] == self.shape:
            if self.convention == 'cell_center':
                average = self.lib.mean(field, (1,2,3))
            elif self.convention == 'staggered_x':
                # Count first and last slice as half cells
                average = self.lib.sum(field[:,1:-1,:,:], (1,2,3)) \
                        + 0.5*self.lib.sum(field[:, 0,:,:], (1,2,3)) \
                        + 0.5*self.lib.sum(field[:,-1,:,:], (1,2,3))
                average /= (self.shape[0]-1) * self.shape[1] * self.shape[2]
        else:
            raise ValueError(
                f"The provided field must have the shape {self.shape}."
            )
        return average

    def pad_periodic_BC_cell_center(self, field):
        """
        Periodic boundary conditions in all directions.
        Consistent with cell centered grid.
        """
        return self.pad_periodic(field)
    
    def pad_periodic_BC_staggered_x(self, field):
        raise NotImplementedError
    
    def pad_dirichlet_periodic_BC_cell_center(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with cell centered grid,
        but loss of 2nd order convergence.
        """
        padded = self.pad_periodic(field)
        padded = self.set(padded, (__, 0,__,__), 2.0*bc0 - padded[:, 1,:,:])
        padded = self.set(padded, (__,-1,__,__), 2.0*bc1 - padded[:,-2,:,:])
        return padded
    
    def pad_dirichlet_periodic_BC_staggered_x(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with staggered_x grid,
        maintains 2nd order convergence.
        """
        padded = self.pad_periodic(field)
        padded = self.set(padded, (__, 0,__,__), bc0)
        padded = self.set(padded, (__,-1,__,__), bc1)
        return padded
    
    def pad_zero_flux_periodic_BC_cell_center(self, field):
        padded = self.pad_periodic(field)
        padded = self.set(padded, (__, 0,__,__), padded[:, 1,:,:])
        padded = self.set(padded, (__,-1,__,__), padded[:,-2,:,:])
        return padded
    
    def pad_zero_flux_BC_cell_center(self, field):
        padded = self.pad_zeros(field)
        padded = self.set(padded, (__, 0,__,__), padded[:, 1,:,:])
        padded = self.set(padded, (__,-1,__,__), padded[:,-2,:,:])
        padded = self.set(padded, (__,__, 0,__), padded[:,:, 1,:])
        padded = self.set(padded, (__,__,-1,__), padded[:,:,-2,:])
        padded = self.set(padded, (__,__,__, 0), padded[:,:,:, 1])
        padded = self.set(padded, (__,__,__,-1), padded[:,:,:,-2])
        return padded

    def pad_zero_flux_BC_staggered_x(self, field):
        raise NotImplementedError

    def pad_zero_flux_periodic_BC_staggered_x(self, field):
        """
        The following comes out of on interpolation polynomial p with
        p'(0) = 0, p(dx) = f(dx,...), p(2*dx) = f(2*dx,...)
        and then use p(0) for the ghost cell. 
        This should be of sufficient order of f'(0) = 0, and even better if
        also f'''(0) = 0 (as it holds for cos(k*pi*x)  )
        """
        padded = self.pad_periodic(field)
        fac1 =  4/3
        fac2 =  1/3
        padded = self.set(padded, (__, 0,__,__), fac1*padded[:, 1,:,:] - fac2*padded[:, 2,:,:])
        padded = self.set(padded, (__,-1,__,__), fac1*padded[:,-2,:,:] - fac2*padded[:,-3,:,:])
        return padded
    
    def grad_x(self, field):
        return 0.5 * (field[RIGHT] - field[LEFT]) * self.div_dx[0]

    def grad_y(self, field):
        return 0.5 * (field[TOP] - field[BOTTOM]) * self.div_dx[1]

    def grad_z(self, field):
        return 0.5 * (field[FRONT] - field[BACK]) * self.div_dx[2]

    def calc_gradient_norm_squared(self, field):
        """Gradient norm squared"""
        return self.grad_x(field)**2 + self.grad_y(field)**2 + self.grad_z(field)**2

    def calc_laplace(self, field):
        r"""Calculate laplace based on compact 2nd order stencil.

        Laplace given as $\nabla\cdot(\nabla u)$ which in 3D is given by
        $\partial^2 u/\partial^2 x + \partial^2 u/\partial^2 y+ \partial^2 u/\partial^2 z$
        Returned field has same shape as the input field (padded with zeros)
        """
        # Manual indexing is ~10x faster than conv3d with laplace kernel in torch
        laplace = \
            (field[RIGHT] + field[LEFT]) * self.div_dx2[0] + \
            (field[TOP] + field[BOTTOM]) * self.div_dx2[1] + \
            (field[FRONT] + field[BACK]) * self.div_dx2[2] - \
             2 * field[CENTER] * self.lib.sum(self.div_dx2)
        return laplace
    
    def calc_normal_laplace(self, field):
        r"""Calculate the normal component of the laplacian

        which is identical to the full laplacian minus curvature.
        It is defined as $\partial^2_n u = \nabla\cdot(\nabla u\cdot n)\cdot n$
        where $n$ denotes the surface normal.
        In the context of phasefield models $n$ is defined as
        $\frac{\nabla u}{|\nabla u|}$.
        The calaculation is based on a compact 2nd order stencil.
        """
        n_laplace =\
            self.grad_x(field)**2 * (field[RIGHT] - 2*field[CENTER] + field[LEFT]) * self.div_dx2[0] +\
            self.grad_y(field)**2 * (field[TOP] - 2*field[CENTER] + field[BOTTOM]) * self.div_dx2[1]+\
            self.grad_z(field)**2 * (field[FRONT] - 2*field[CENTER] + field[BACK]) * self.div_dx2[2]+\
            0.5 * self.grad_x(field) * self.grad_y(field) *\
                  (field[:,2:,2:,1:-1] + field[:,:-2,:-2,1:-1] -\
                   field[:,:-2,2:,1:-1] - field[:,2:,:-2,1:-1]) * self.div_dx[0] * self.div_dx[1] +\
            0.5 *self.grad_x(field) * self.grad_z(field) *\
                  (field[:,2:,1:-1,2:] + field[:,:-2,1:-1,:-2] -\
                   field[:,:-2,1:-1,2:] - field[:,2:,1:-1,:-2]) * self.div_dx[0] * self.div_dx[2] +\
            0.5 * self.grad_y(field) * self.grad_z(field) *\
                  (field[:,1:-1,2:,2:] + field[:,1:-1,:-2,:-2] -\
                   field[:,1:-1,:-2,2:] - field[:,1:-1,2:,:-2]) * self.div_dx[1] * self.div_dx[2]
        norm2 = self.calc_gradient_norm_squared(field)
        bulk = self.lib.where(norm2 <= 1e-7)
        norm2 = self.set(norm2, bulk, 1.0)
        return n_laplace/norm2

class VoxelGridTorch(VoxelGrid):
    def __init__(self, grid: Grid, precision='float32', device: str='cuda'):
        """Create a torch backed grid.

        Args:
            grid: Grid description.
            precision: Floating point precision.
            device: Torch device string.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch backend selected but 'torch' is not installed.")
        self.torch = torch

        # Handle torch device
        self.device = torch.device(device)
        if torch.device(device).type.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            warnings.warn(
                "CUDA not available, defaulting device to cpu. "
                "To avoid this warning, set device=torch.device('cpu')",
            )
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

    def concatenate(self, fieldlist, dim):
        return self.torch.cat(fieldlist, dim=dim)

    def set(self, field, index, value):
        field[index] = value
        return field


class VoxelGridJax(VoxelGrid):
    def __init__(self, grid: Grid, precision='float32'):
        """Create a JAX backed grid.

        Args:
            grid: Grid description.
            precision: Floating point precision for arrays.
        """
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

    def concatenate(self, fieldlist, dim):
        return self.jnp.concatenate(fieldlist, axis=dim)

    def set(self, field, index, value):
        return field.at[index].set(value)

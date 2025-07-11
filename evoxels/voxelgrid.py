import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Any
from .fd_stencils import FDStencils
from .boundary_conditions import CellCenteredBCs, StaggeredXBCs


@dataclass
class Grid:
    """Handles most basic properties"""
    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    convention: str


@dataclass
class VoxelGrid(FDStencils):
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
        if self.convention == 'cell_center':
            self.bc = CellCenteredBCs(self)
        elif self.convention == 'staggered_x':
            self.bc = StaggeredXBCs(self)

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

    def fftn(self, field, shape):
        """Compute the n-dimensional discrete Fourier transform."""
        raise NotImplementedError

    def real_of_ifftn(self, field, shape):
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
        a_x, a_y, _ = self.fft_axes()
        _, _, a_z = self.rfft_axes()
        kx, ky, kz = self.lib.meshgrid(a_x, a_y, a_z, indexing='ij')
        return kx**2 + ky**2 + kz**2
    
    def fft_k_squared_nonperiodic(self):
        if self.convention == 'cell_center':
            a_x = 2*self.lib.pi*self.lib.fft.fftfreq(2*self.shape[0], d=self.spacing[0])
        else:   
            a_x = 2*self.lib.pi*self.lib.fft.fftfreq(2*self.shape[0]-2, d=self.spacing[0])
        _, a_y, _ = self.fft_axes()
        _, _, a_z = self.rfft_axes()
        kx, ky, kz = self.lib.meshgrid(a_x, a_y, a_z, indexing='ij')
        return kx**2 + ky**2 + kz**2
    
    def init_scalar_field(self, array):
        """Convert and pad a NumPy array for simulation."""
        field = self.to_backend(array)
        field = self.expand_dim(field, 0)
        return field

    def export_scalar_field_to_numpy(self, field):
        """Export backend field back to NumPy."""
        array = self.to_numpy(self.squeeze(field, 0))
        return array
    
    def average(self, field):
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

    def fftn(self, field, shape):
        return self.torch.fft.fftn(field, s=shape)

    def rfftn(self, field, shape):
        return self.torch.fft.rfftn(field, s=shape)

    def irfftn(self, field, shape):
        return self.torch.fft.irfftn(field, s=shape)

    def real_of_ifftn(self, field, shape):
        return self.torch.real(self.torch.fft.ifftn(field, s=shape))

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

    def fftn(self, field, shape):
        return self.jnp.fft.fftn(field, s=shape)

    def rfftn(self, field, shape):
        return self.jnp.fft.rfftn(field, s=shape)

    def irfftn(self, field, shape):
        return self.jnp.fft.irfftn(field, s=shape)

    def real_of_ifftn(self, field, shape):
        return self.jnp.fft.ifftn(field, s=shape).real

    def expand_dim(self, field, dim):
        return self.jnp.expand_dims(field, dim)

    def squeeze(self, field, dim):
        return self.jnp.squeeze(field, axis=dim)

    def concatenate(self, fieldlist, dim):
        return self.jnp.concatenate(fieldlist, axis=dim)

    def set(self, field, index, value):
        return field.at[index].set(value)

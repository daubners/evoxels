import numpy as np
import psutil
import torch
import torch.nn.functional as F
import warnings
from .fields import VoxelFields

class BaseSolver:
    def __init__(self, data: VoxelFields, device='cuda'):
        """
        Base solver class to handle common functionality for solvers.

        Args:
            data (VoxelFields): The voxel field data object containing spatial and field information.
            device (str): The device to perform computations ('cpu' or 'cuda').
        """
        self.data = data
        self.device = torch.device(device)
        # check device is available
        if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            warnings.warn("CUDA not available, defaulting device to cpu. To avoid this warning, set device=torch.device('cpu')")

        if data.precision == np.float32:
            self.precision = torch.float32
        if data.precision == np.float64:
            self.precision = torch.float64
        
        self.spacing = data.spacing
        # Precompute squares of inverse spacing
        self.div_dx2 = 1/torch.tensor(self.spacing, dtype=self.precision, device=self.device)**2
        self.grid = None
        self.frame = 0

    def init_tensor_from_voxel_field(self, data: VoxelFields, name: str):
        tensor = torch.tensor(data.fields[name], dtype=self.precision, device=self.device)
        tensor = tensor.unsqueeze(0)
        tensor = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
        if data.convention == 'staggered_x':
            # For staggered grid we don't need ghost nodes in x
            tensor = tensor[:,1:-1,:,:]
        return tensor

    def apply_periodic_BC_cell_center(self, tensor):
        """
        Periodic boundary conditions in all directions.
        Consistent with cell centered grid.
        """
        tensor[:, 0,:,:] = tensor[:,-2,:,:]
        tensor[:,-1,:,:] = tensor[:, 1,:,:]
        tensor[:,:, 0,:] = tensor[:,:,-2,:]
        tensor[:,:,-1,:] = tensor[:,:, 1,:]
        tensor[:,:,:, 0] = tensor[:,:,:,-2]
        tensor[:,:,:,-1] = tensor[:,:,:, 1]
        return tensor
    
    def apply_dirichlet_periodic_BC_cell_center(self, tensor, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with cell centered grid,
        but loss of 2nd order convergence.
        """
        tensor[:, 0,:,:] = 2.0*bc0 - tensor[:, 1,:,:]
        tensor[:,-1,:,:] = 2.0*bc1 - tensor[:,-2,:,:]
        tensor[:,:, 0,:] = tensor[:,:,-2,:]
        tensor[:,:,-1,:] = tensor[:,:, 1,:]
        tensor[:,:,:, 0] = tensor[:,:,:,-2]
        tensor[:,:,:,-1] = tensor[:,:,:, 1]
        return tensor
    
    def apply_dirichlet_periodic_BC_staggered_x(self, tensor, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with staggered_x grid,
        maintains 2nd order convergence.
        """
        tensor[:, 0,:,:] = bc0
        tensor[:,-1,:,:] = bc1
        tensor[:,:, 0,:] = tensor[:,:,-2,:]
        tensor[:,:,-1,:] = tensor[:,:, 1,:]
        tensor[:,:,:, 0] = tensor[:,:,:,-2]
        tensor[:,:,:,-1] = tensor[:,:,:, 1]
        return tensor
    
    def apply_zero_flux_periodic_BC_cell_center(self, tensor):
        tensor[:, 0,:,:] = tensor[:, 1,:,:]
        tensor[:,-1,:,:] = tensor[:,-2,:,:]
        tensor[:,:, 0,:] = tensor[:,:,-2,:]
        tensor[:,:,-1,:] = tensor[:,:, 1,:]
        tensor[:,:,:, 0] = tensor[:,:,:,-2]
        tensor[:,:,:,-1] = tensor[:,:,:, 1]
        return tensor

    def apply_zero_flux_periodic_BC_staggered_x(self, tensor):
        """
        The following comes out of on interpolation polynomial p with
        p'(0) = 0, p(dx) = f(dx,...), p(2*dx) = f(2*dx,...)
        and then use p(0) for the ghost cell. 
        This should be of sufficient order of f'(0) = 0, and even better if
        also f'''(0) = 0 (as it holds for cos(k*pi*x)  )
        """
        fac1 =  4/3
        fac2 =  1/3
        tensor[:, 0,:,:] = fac1*tensor[:, 1,:,:] - fac2*tensor[:, 2,:,:]
        tensor[:,-1,:,:] = fac1*tensor[:,-2,:,:] - fac2*tensor[:,-3,:,:]
        tensor[:,:, 0,:] = tensor[:,:,-2,:]
        tensor[:,:,-1,:] = tensor[:,:, 1,:]
        tensor[:,:,:, 0] = tensor[:,:,:,-2]
        tensor[:,:,:,-1] = tensor[:,:,:, 1]
        return tensor

    def calc_laplace(self, tensor):
        """
        Calculate laplace based on compact 2nd order stencil.
        Returned field has same shape as the input tensor (padded with zeros)
        """
        # Manual indexing is ~10x faster than conv3d with laplace kernel
        laplace = torch.zeros_like(tensor)
        laplace[:,1:-1,1:-1,1:-1] = \
            (tensor[:, 2:, 1:-1, 1:-1] + tensor[:, :-2, 1:-1, 1:-1]) * self.div_dx2[0] + \
            (tensor[:, 1:-1, 2:, 1:-1] + tensor[:, 1:-1, :-2, 1:-1]) * self.div_dx2[1] + \
            (tensor[:, 1:-1, 1:-1, 2:] + tensor[:, 1:-1, 1:-1, :-2]) * self.div_dx2[2] - \
             2 * tensor[:, 1:-1, 1:-1, 1:-1] * torch.sum(self.div_dx2)
        return laplace

    def solve(self):
        """
        Abstract method to be implemented by derived solver classes.
        Raises:
            NotImplementedError: If the method is not overwritten in a child class.
        """
        raise NotImplementedError("The solve method must be implemented in child classes")

    def print_memory_stats(self, start, end, iters):
        """
        Call in verbose mode to return
        Wall time and RAM requirements for simulation.
        """
        print(f'Wall time: {np.around(end - start, 4)} s after {iters} iterations ({np.around((end - start)/iters, 4)} s/iter)')
        if self.device.type == 'cuda':
            print(f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
        elif self.device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")
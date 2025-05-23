from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import sys
from timeit import default_timer as timer
import psutil
import torch
import torch.fft as fft
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
        self.grid = None
        self.frame = 0

    def print_memory_stats(self, start, end, iters):
        print(f'Wall time: {np.around(end - start, 4)} s ({np.around((end - start)/iters, 4)} s/iter)')
        if self.device.type == 'cuda':
            print(f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)")
        elif self.device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

    def solve(self):
        """
        Abstract method to be implemented by derived solver classes.
        Raises:
            NotImplementedError: If the method is not overridden in a child class.
        """
        raise NotImplementedError("The solve method must be implemented in child classes")


class CahnHilliardSolver(BaseSolver):
    """
    Cahn-Hilliard system with double-well potential energy
    c is dimensionless and can be interpreted as molar fraction or phase fraction
    \epsilon scales the diffuse interface width
    \sigma scales the interfacial energy (set to 1 here)
    F = int( sigma*eps*|grad(c)|^2 + 9*sigma*/eps * c^2 * (1-c)^2).

    The temporal evolution is given as
    dc/dt = div( D*c*(1-c) \nabla ( 18*c*(1-c)*(1-2c) - 2*laplace(c) )) + f(t) .

    The current implementation assumes isotropic voxels (dx=dy=dz).

    Attributes:
        fieldname: Field in the VoxelFields object to perform simulation with.
        diffusivity: Constant diffusion coefficent [m^2/s].
        epsilon: Parameter scaling the diffuse interface width [m]
        device: device for storing torch tensors
    """

    def __init__(self, data: VoxelFields, fieldname, diffusivity=1.0, epsilon=3, device='cuda'):
        """
        Solves the Cahn-Hilliard equation for two-phase system with double-well energy assuming periodic BC.
        """
        super().__init__(data, device=device)
        self.field = fieldname
        self.D = diffusivity
        self.eps = epsilon
        self.tensor = torch.tensor(data.fields[fieldname], dtype=self.precision, device=self.device)
        # Precompute squares of inverse spacing
        self.div_dx2 = 1/torch.tensor(self.spacing, dtype=self.precision, device=self.device)**2

    def handle_outputs(self, vtk_out, verbose, slice=0):
        self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
        if np.isnan(self.data.fields[self.field]).any():
            print(f"NaN detected in frame {self.frame} at time {self.time}. Aborting simulation.")
            sys.exit(1)  # Exit the program with an error status
        if vtk_out:
            filename = self.field+f"_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=[self.field])
        if verbose == 'plot':
            clear_output(wait=True)
            self.data.plot_slice(self.field, slice, time=self.time)

    def pad_periodic(self, tensor):
        return F.pad(tensor, (1,1,1,1,1,1), mode='circular')

    def calc_laplace(self, padded):
        # Manual indexing is ~10x faster than conv3d with laplace kernel
        laplace = (padded[:, 2:, 1:-1, 1:-1] + padded[:, :-2, 1:-1, 1:-1]) * self.div_dx2[0] + \
                  (padded[:, 1:-1, 2:, 1:-1] + padded[:, 1:-1, :-2, 1:-1]) * self.div_dx2[1] + \
                  (padded[:, 1:-1, 1:-1, 2:] + padded[:, 1:-1, 1:-1, :-2]) * self.div_dx2[2] - \
                   2 * padded[:, 1:-1, 1:-1, 1:-1] * torch.sum(self.div_dx2)
        return laplace

    def calc_divergence_variable_mobility(self, padded_mu, padded_c):
        divergence = ((padded_mu[:, 2:, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, :-2, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[0]

        divergence += ((padded_mu[:, 1:-1, 2:, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, :-2, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[1]

        divergence += ((padded_mu[:, 1:-1, 1:-1, 2:] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, :-2]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[2]
        return divergence

    def solve(self, time_increment=0.01, frames=10, max_iters=1000, variable_m = True, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using explicit timestepping.
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters/frames)
        self.frame = 0
        self.time = 0
        self.tensor = self.tensor.unsqueeze(0)
        slice = int(self.tensor.shape[-1]/2)
        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i*time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                padded_c = self.pad_periodic(self.tensor)
                laplace = self.calc_laplace(padded_c)
                mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*laplace
                padded_mu = self.pad_periodic(mu)
                if variable_m:
                    divergence = self.calc_divergence_variable_mobility(padded_mu, padded_c)
                else:
                    divergence = self.calc_laplace(padded_mu)
                self.tensor += time_increment * self.D * divergence

            end = timer()
            self.time = max_iters*time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)

    def initialise_FFT_wavenumbers(self):
        Nx, Ny, Nz = self.tensor.shape
        dx, dy, dz = self.spacing

        kx = 2 * torch.pi * fft.fftfreq(Nx, d=dx, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(Ny, d=dy, device=self.device)
        kz = 2 * torch.pi * fft.fftfreq(Nz, d=dz, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx, ky, kz, kx**2 + ky**2 + kz**2

    def solve_FFT(self, time_increment=0.01, frames=10, max_iters=1000, variable_m = True, A = 0.25, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation based on FFT and semi-implicit timestepping.
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters/frames)
        slice = int(self.tensor.shape[-1]/2)
        self.frame = 0
        self.time = 0
        if variable_m:
            kx, ky, kz, k_squared = self.initialise_FFT_wavenumbers()
        else:
            _, _, _, k_squared = self.initialise_FFT_wavenumbers()

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i*time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                dfhom_dc = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor)
                if variable_m:
                    c_hat = fft.fftn(self.tensor)
                    mobility = self.tensor*(1-self.tensor)
                    mu_hat = fft.fftn(dfhom_dc) + 2*self.eps*k_squared*c_hat
                    flux_x = mobility * torch.real(fft.ifftn(1j * kx * mu_hat))
                    flux_y = mobility * torch.real(fft.ifftn(1j * ky * mu_hat))
                    flux_z = mobility * torch.real(fft.ifftn(1j * kz * mu_hat))
                    flux_div = 1j * (kx*fft.fftn(flux_x) + ky*fft.fftn(flux_y) + kz*fft.fftn(flux_z))

                    # Update c_hat using semi-implicit scheme
                    c_hat += time_increment*self.D*flux_div / (1 + 2*self.eps*time_increment*self.D*k_squared**2*A)

                else:
                    # Update c_hat using semi-implicit scheme
                    c_hat = (fft.fftn(self.tensor) - time_increment*self.D*k_squared*fft.fftn(dfhom_dc)) / (1 + 2*self.eps*time_increment*self.D*k_squared**2)

                # Transform back to real space
                self.tensor = torch.real(fft.ifftn(c_hat))

            end = timer()
            self.time = max_iters*time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)

    def solve_FFT2(self, time_increment=0.01, frames=10, max_iters=1000, A = 0.25, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters/frames)
        slice = int(self.tensor.shape[-1]/2)
        self.frame = 0
        self.time = 0
        _, _, _, k_squared = self.initialise_FFT_wavenumbers()
        self.tensor = self.tensor.unsqueeze(0)

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i*time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                padded_c = self.pad_periodic(self.tensor)
                laplace = self.calc_laplace(padded_c)
                mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*laplace
                padded_mu = self.pad_periodic(mu)
                divergence = self.calc_divergence_variable_mobility(padded_mu, padded_c)
                divergence *= self.D
                flux_div = fft.fftn(divergence)

                c_hat = fft.fftn(self.tensor)
                # Update c_hat using semi-implicit scheme
                c_hat += time_increment*flux_div / (1 + 2*self.eps*time_increment*self.D*k_squared**2*A)

                # Transform back to real space
                self.tensor = torch.real(fft.ifftn(c_hat))

            end = timer()
            self.time = max_iters*time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)
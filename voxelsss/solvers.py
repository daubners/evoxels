from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import sys
from timeit import default_timer as timer
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
        self.precision = torch.float32
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
        self.laplace = (padded[:, 2:, 1:-1, 1:-1] + padded[:, :-2, 1:-1, 1:-1]) * self.div_dx2[0] + \
                       (padded[:, 1:-1, 2:, 1:-1] + padded[:, 1:-1, :-2, 1:-1]) * self.div_dx2[1] + \
                       (padded[:, 1:-1, 1:-1, 2:] + padded[:, 1:-1, 1:-1, :-2]) * self.div_dx2[2] - \
                       2 * padded[:, 1:-1, 1:-1, 1:-1] * torch.sum(self.div_dx2)

    def calc_divergence_variable_mobility(self, padded_mu, padded_c):
        self.laplace = ((padded_mu[:, 2:, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, :-2, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[0]

        self.laplace += ((padded_mu[:, 1:-1, 2:, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, :-2, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[1]

        self.laplace += ((padded_mu[:, 1:-1, 1:-1, 2:] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, :-2]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]))) \
                       * self.div_dx2[2]

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
                self.calc_laplace(padded_c)
                mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*self.laplace
                padded_mu = self.pad_periodic(mu)
                if variable_m:
                    self.calc_divergence_variable_mobility(padded_mu, padded_c)
                else:
                    self.calc_laplace(padded_mu)
                self.tensor += time_increment * self.D * self.laplace

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


class CahnHilliard4PhaseSolver(BaseSolver):
    """
    Specific Cahn Hilliard solver for a 4 phase system assuming periodic BC based on
    Zhang2021 https://doi.org/10.1021/acs.langmuir.1c00275,
    """

    def __init__(self, data: VoxelFields, field_names, binary_interactions, ternary_interactions, diffusivity=1.0, kappa=1, device='cuda'):
        """
        Solves the Cahn-Hilliard equation for two-phase system with double-well energy assuming periodic BC.
        """
        self.phase_count = 4

        super().__init__(data, device=device)
        if not isinstance(field_names, (list, tuple)) or len(field_names) != self.phase_count:
            raise ValueError(f"field_names must be a list or tuple with {self.phase_count} elements.")
        if not all(isinstance(x, str) for x in field_names):
            raise ValueError("All field names must be strings")
        self.field_names = field_names
        self.D = diffusivity
        self.kappa = kappa
        self.binary = torch.tensor(binary_interactions, dtype=self.precision, device=self.device)
        self.ternary = torch.tensor(ternary_interactions, dtype=self.precision, device=self.device)
        self.min = torch.tensor(1e-10, device=self.device)
        self.max = 1-torch.tensor(1e-10, device=self.device)
        self.c = torch.stack([torch.tensor(data.fields[field_names[i]], dtype=self.precision, device=self.device) for i in range(3)])

    def map_to_rgb(self, slice):
        """Plot 4 overlapping fields as an RGB image."""
        rgb_image = np.zeros((self.data.Ny, self.data.Nz, self.phase_count))  # Initialize an empty RGB image
        for i, name in enumerate(self.field_names):
            rgb_image[..., i] = self.data.fields[name][slice,:,:]
        # rgb_image[..., 3] = 1 - self.data.fields[self.field_names[-1]][slice,:,:] # Opacity
        return rgb_image

    def plot_4phase_RGB(self, slice):
        RGB_image = self.map_to_rgb(slice)
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        # Plot the scalar field with a colorbar
        im0 = ax0.imshow(self.data.fields[self.field_names[0]][slice, :, :], cmap='viridis')
        cbar = fig.colorbar(im0, ax=ax0)
        min_val = self.data.fields[self.field_names[0]][slice, :, :].min()
        max_val = self.data.fields[self.field_names[0]][slice, :, :].max()
        cbar.set_ticks([min_val, max_val])
        cbar.set_ticklabels([f"{min_val:.2f}", f"{max_val:.2f}"])
        ax0.set_title(f"Field {self.field_names[0]} at frame {self.frame}")
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")

        # Plot the RGB image
        ax1.imshow(RGB_image)
        ax1.set_title("RGB Image")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # Show the plots
        plt.tight_layout()
        plt.show()

    def handle_outputs(self, vtk_out, verbose, slice=0):
        for i, name in enumerate(self.field_names):
            self.data.fields[name] = self.c[i].squeeze().cpu().numpy()
            if np.isnan(self.data.fields[name]).any():
                print(f"NaN detected in frame {self.frame} at time {self.time}. Aborting simulation.")
                sys.exit(1)  # Exit the program with an error status
        if vtk_out:
            filename = f"4phase_CH_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=self.field_names)
        if verbose == 'plot':
            clear_output(wait=True)
            self.plot_4phase_RGB(slice)
            print(f'c0_av: {torch.mean(self.c[0]):.3f}')
            print(f'c1_av: {torch.mean(self.c[1]):.3f}')

    def solve(self, time_increment=0.01, frames=10, max_iters=1000, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        self.n_out = int(max_iters/frames)
        self.frame = 0
        slice = int(self.data.Nx/2)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)

        with torch.no_grad():
            start = timer()
            for iter in range(max_iters):
                if iter % self.n_out == 0:
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                c_pad = F.pad(self.c, (1,1,1,1,1,1), mode='circular')
                laplace = ( c_pad[:, 2:, 1:-1, 1:-1] + \
                            c_pad[:, :-2, 1:-1, 1:-1] + \
                            c_pad[:, 1:-1, 2:, 1:-1] + \
                            c_pad[:, 1:-1, :-2, 1:-1] + \
                            c_pad[:, 1:-1, 1:-1, 2:] + \
                            c_pad[:, 1:-1, 1:-1, :-2] - \
                            6*c_pad[:, 1:-1, 1:-1, 1:-1] )
                self.mu = -self.kappa*laplace + 2*self.c #torch.log(self.zero + self.c)

                for j in range(1,4):
                    factors = torch.diagonal(torch.roll(self.binary,j,dims=0)).view(-1, 1, 1, 1)
                    self.mu += factors * torch.roll(self.c, j, dims=0)
                    # for k in range(j+1,4):
                    #     if k != i:
                    #         mu += self.ternary[i,j,k] * getattr(self, f'c{j}') * getattr(self, f'c{k}')
                # self.mu[2] = torch.zeros_like(self.c[0], device=self.device)
                # self.mu[3] = torch.zeros_like(self.c[0], device=self.device)
                mu_pad = F.pad(self.mu, (1,1,1,1,1,1), mode='circular')

                c_av_x = 0.5*(c_pad[:, 1:, 1:-1, 1:-1] + c_pad[:, :-1, 1:-1, 1:-1])
                c_av_y = 0.5*(c_pad[:, 1:-1, 1:, 1:-1] + c_pad[:, 1:-1, :-1, 1:-1])
                c_av_z = 0.5*(c_pad[:, 1:-1, 1:-1, 1:] + c_pad[:, 1:-1, 1:-1, :-1])

                grad_x_mu = mu_pad[:, 1:, 1:-1, 1:-1] - mu_pad[:, :-1, 1:-1, 1:-1]
                grad_y_mu = mu_pad[:, 1:-1, 1:, 1:-1] - mu_pad[:, 1:-1, :-1, 1:-1]
                grad_z_mu = mu_pad[:, 1:-1, 1:-1, 1:] - mu_pad[:, 1:-1, 1:-1, :-1]

                divergence = grad_x_mu[:, 1:, :, :] * c_av_x[:, 1:, :, :] * (1-c_av_x[:, 1:, :, :])\
                            -grad_x_mu[:, :-1, :, :] * c_av_x[:, :-1, :, :] * (1-c_av_x[:, :-1, :, :])

                divergence += grad_y_mu[:, :, 1:, :] * c_av_y[:, :, 1:, :] * (1-c_av_y[:, :, 1:, :]) \
                             -grad_y_mu[:, :, :-1, :] * c_av_y[:, :, :-1, :] * (1-c_av_y[:, :, :-1, :])

                divergence += grad_z_mu[:, :, :, 1:] * c_av_z[:, :, :, 1:] * (1-c_av_z[:, :, :, 1:]) \
                             -grad_z_mu[:, :, :, :-1] * c_av_z[:, :, :, :-1] * (1-c_av_z[:, :, :, :-1])

                for j in range(1,4):
                    divergence -= torch.roll(grad_x_mu[:, 1:, :, :], j, dims=0) * \
                                  c_av_x[:, 1:, :, :] * torch.roll(c_av_x[:, 1:, :, :], j, dims=0)
                    divergence += torch.roll(grad_x_mu[:, :-1, :, :], j, dims=0) * \
                                  c_av_x[:, :-1, :, :] * torch.roll(c_av_x[:, :-1, :, :], j, dims=0)

                    divergence -= torch.roll(grad_y_mu[:, :, 1:, :], j, dims=0) *\
                                  c_av_y[:, :, 1:, :] * torch.roll(c_av_y[:, :, 1:, :], j, dims=0)
                    divergence += torch.roll(grad_y_mu[:, :, :-1, :], j, dims=0) *\
                                  c_av_y[:, :, :-1, :] * torch.roll(c_av_y[:, :, :-1, :], j, dims=0)

                    divergence -= torch.roll(grad_z_mu[:, :, :, 1:], j, dims=0) *\
                                  c_av_z[:, :, :, 1:] * torch.roll(c_av_z[:, :, :, 1:], j, dims=0)
                    divergence += torch.roll(grad_z_mu[:, :, :, :-1], j, dims=0) *\
                                  c_av_z[:, :, :, :-1] * torch.roll(c_av_z[:, :, :, :-1], j, dims=0)

                self.c += time_increment * self.D * divergence

            end = timer()
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)

    def initialise_FFT_wavenumbers(self):
        Nx, Ny, Nz = self.data.Nx, self.data.Ny, self.data.Nz
        dx, dy, dz = self.spacing

        kx = 2 * torch.pi * fft.fftfreq(Nx, d=dx, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(Ny, d=dy, device=self.device)
        kz = 2 * torch.pi * fft.fftfreq(Nz, d=dz, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx, ky, kz, kx**2 + ky**2 + kz**2

    def enforce_gibbs_simplex_contraint(self):
        self.c = torch.maximum(self.c, torch.tensor(self.zero, device=self.device))
        self.c /= torch.sum(self.c, dim=0, keepdim=True)

    def solve_FFT(self, time_increment=0.01, frames=10, max_iters=1000, A = 0.25, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters/frames)
        self.frame = 0
        slice = int(self.data.Nx/2)
        kx, ky, kz, k_squared = self.initialise_FFT_wavenumbers()

        with torch.no_grad():
            start = timer()
            for iter in range(max_iters):
                if iter % self.n_out == 0:
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                c_hat = fft.fftn(self.c, dim=(1, 2, 3))  # Compute FFT across spatial dimensions for all components
                dfhom_dc = torch.log(self.c)
                mu_hat = fft.fftn(dfhom_dc, dim=(1, 2, 3)) + self.kappa * k_squared * c_hat
                # mu_hat = (2 + self.kappa * k_squared) * c_hat
                for j in range(1,self.phase_count):
                    factors = torch.diagonal(torch.roll(self.binary,j,dims=0)).view(-1, 1, 1, 1)
                    mu_hat += factors * torch.roll(c_hat, j, dims=0)
                    for k in range(j+1,self.phase_count):
                        factors = torch.roll(self.ternary, shifts=(j, k), dims=(0, 1)).diagonal().diagonal().view(-1, 1, 1, 1)
                        mu_hat += factors * torch.roll(c_hat, j, dims=0) * torch.roll(c_hat, k, dims=0)

                grad_x_mu = torch.real(fft.ifftn(1j * kx * mu_hat, dim=(1, 2, 3)))
                grad_y_mu = torch.real(fft.ifftn(1j * ky * mu_hat, dim=(1, 2, 3)))
                grad_z_mu = torch.real(fft.ifftn(1j * kz * mu_hat, dim=(1, 2, 3)))

                flux_x = grad_x_mu
                flux_y = grad_y_mu
                flux_z = grad_z_mu

                for j in range(0,self.phase_count):
                    mobility = torch.roll(self.c, j, dims=0) #torch.minimum(torch.maximum(torch.roll(self.c, j, dims=0), self.min), self.max)
                    flux_x -= mobility * torch.roll(grad_x_mu, j, dims=0)
                    flux_y -= mobility* torch.roll(grad_y_mu, j, dims=0)
                    flux_z -= mobility * torch.roll(grad_z_mu, j, dims=0)

                mobility = self.c #torch.minimum(torch.maximum(self.c, self.min), self.max)
                flux_x = mobility * grad_x_mu
                flux_y = mobility * grad_y_mu
                flux_z = mobility * grad_z_mu

                flux_div = 1j * (kx * fft.fftn(flux_x, dim=(1, 2, 3)) + \
                                 ky * fft.fftn(flux_y, dim=(1, 2, 3)) + \
                                 kz * fft.fftn(flux_z, dim=(1, 2, 3)))

                # Update c_hat
                c_hat += time_increment * self.D * flux_div / (1 + self.kappa * time_increment * self.D * k_squared**2 * A)
                # Transform back to real space
                self.c = torch.real(fft.ifftn(c_hat, dim=(1, 2, 3)))
                # self.enforce_gibbs_simplex_contraint()

            end = timer()
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_memory_stats(start, end, max_iters)
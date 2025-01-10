from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import sys
from timeit import default_timer as timer
import torch
import torch.fft as fft
import torch.nn.functional as F
import torch_dct as dct
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
        self.precision = torch.float32
        self.spacing = data.spacing
        self.frame = 0

    # def check_convergence(self, current_error):
    #     """
    #     Check if the solution has converged.
    #     """
    #     return current_error < self.tolerance

    def print_GPU_stats(self, start, end, iters):
        print(f'Wall time: {np.around(end - start, 4)} s ({np.around((end - start)/iters, 4)} s/iter)')
        print(f"GPU-RAM currently allocated {torch.cuda.max_memory_allocated() / 1e6:.2f} MB ({torch.cuda.max_memory_reserved() / 1e6:.2f} MB reserved)")
        print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated() / 1e6:.2f} MB ({torch.cuda.max_memory_reserved() / 1e6:.2f} MB reserved)")

    def solve(self):
        """
        Abstract method to be implemented by derived solver classes.
        Raises:
            NotImplementedError: If the method is not overridden in a child class.
        """
        raise NotImplementedError("The solve method must be implemented in child classes")


class CahnHilliardSolver(BaseSolver):
    # Note: This is the simplest Cahn-Hilliard system for now with
    # F = int( eps*|grad(phi)|^2 + 9/eps * phi^2 * (1-phi)^2)
    # and dc/dt = D*laplace( 18*c*(1-c)*(1-2c) - 2*laplace(c) ).
    # A non-linear mobility e.g. M=D*c*(1-c) is notoriously more difficult to implement in Fourier space.

    # The current implementation assumes isotropic voxels (dx=dy=dz).
    # However, this could be easily changed.

    def __init__(self, data: VoxelFields, fieldname, diffusivity=1.0, epsilon=3, device='cuda'):
        """
        Solves the Cahn-Hilliard equation for two-phase system with double-well energy assuming periodic BC.
        """
        super().__init__(data, device)
        self.field = fieldname
        self.D = diffusivity
        self.eps = epsilon
        self.tensor = torch.tensor(data.fields[fieldname], dtype=self.precision, device=self.device)

    def handle_outputs(self, vtk_out, verbose, slice=0):
        self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
        if np.isnan(self.data.fields[self.field]).any():
            print(f"NaN detected in frame {self.frame}. Aborting simulation.")
            sys.exit(1)  # Exit the program with an error status
        if vtk_out:
            filename = self.field+f"_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=[self.field])
        if verbose == 'plot':
            clear_output(wait=True)
            self.data.plot_slice(self.field, slice)

    def calc_laplace(self, tensor):
        padded = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
        # Manual indexing is ~10x faster than conv3d with laplace kernel
        self.laplace = padded[:, 2:, 1:-1, 1:-1] + \
                       padded[:, :-2, 1:-1, 1:-1] + \
                       padded[:, 1:-1, 2:, 1:-1] + \
                       padded[:, 1:-1, :-2, 1:-1] + \
                       padded[:, 1:-1, 1:-1, 2:] + \
                       padded[:, 1:-1, 1:-1, :-2] - \
                     6*padded[:, 1:-1, 1:-1, 1:-1]

    def calc_divergence_variable_mobility(self, tensor):
        padded_mu = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
        padded_c = F.pad(self.tensor, (1,1,1,1,1,1), mode='circular')
        self.laplace = (padded_mu[:, 2:, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, :-2, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))

        self.laplace += (padded_mu[:, 1:-1, 2:, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, :-2, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))

        self.laplace += (padded_mu[:, 1:-1, 1:-1, 2:] - padded_mu[:, 1:-1, 1:-1, 1:-1]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1])) - \
                       (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, :-2]) *\
                       0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]) * \
                       (1 - 0.5*(padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]))

    def solve(self, time_increment=0.01, frames=10, max_iters=1000, variable_m = True, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        # Specific numerical implementation for Cahn-Hilliard equation
        self.n_out = int(max_iters/frames)
        self.frame = 0
        self.tensor = self.tensor.unsqueeze(0)
        slice = int(self.tensor.shape[-1]/2)
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                # Numerical increment
                self.calc_laplace(self.tensor)
                mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*self.laplace
                if variable_m:
                    self.calc_divergence_variable_mobility(mu)
                else:
                    self.calc_laplace(mu)
                self.tensor += time_increment * self.D * self.laplace

            end = timer()
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_GPU_stats(start, end, max_iters)

    def initialise_FFT_wavenumbers(self):
        Nx, Ny, Nz = self.tensor.shape
        dx, dy, dz = self.spacing

        kx = 2 * torch.pi * fft.fftfreq(Nx, d=dx, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(Ny, d=dy, device=self.device)
        kz = 2 * torch.pi * fft.fftfreq(Nz, d=dz, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx, ky, kz, kx**2 + ky**2 + kz**2

    def initialise_DCT_wavenumbers(self):
        Nx, Ny, Nz = self.tensor.shape
        dx, dy, dz = self.spacing

        kx = torch.pi * torch.arange(0,1,1/Nx, dtype=self.precision, device=self.device)
        ky = torch.pi * torch.arange(0,1,1/Ny, dtype=self.precision, device=self.device)
        kz = torch.pi * torch.arange(0,1,1/Nz, dtype=self.precision, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx, ky, kz, kx**2 + ky**2 + kz**2

    def solve_FFT(self, time_increment=0.01, frames=10, max_iters=1000, variable_m = True, A = 0.25, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        torch.cuda.reset_peak_memory_stats()
        self.n_out = int(max_iters/frames)
        slice = int(self.tensor.shape[-1]/2)
        self.frame = 0
        if variable_m:
            kx, ky, kz, k_squared = self.initialise_FFT_wavenumbers()
        else:
            _, _, _, k_squared = self.initialise_FFT_wavenumbers()

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
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
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_GPU_stats(start, end, max_iters)

    # TODO: Fix imaginary parts for cosine transformation
    def solve_FFT_neumann(self, time_increment=0.01, frames=10, max_iters=1000, A=0.25, verbose=True, vtk_out=True):
        """
        Solves the Cahn-Hilliard equation using Neumann boundary conditions.
        """
        self.n_out = int(max_iters / frames)
        slice = int(self.tensor.shape[-1] / 2)
        self.frame = 0
        kx, ky, kz, k_squared = self.initialise_DCT_wavenumbers()

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.handle_outputs(vtk_out, verbose, slice=slice)
                    self.frame += 1

                # Compute the derivative of the free energy
                dfhom_dc = 18 / self.eps * self.tensor * (1 - self.tensor) * (1 - 2 * self.tensor)

                # Transform to spectral space
                c_hat = dct.dct_3d(self.tensor)
                mobility = self.tensor * (1 - self.tensor)
                mu_hat = dct.dct_3d(dfhom_dc) + 2 * self.eps * k_squared * c_hat

                # Compute the flux divergence
                flux_x = mobility * torch.real(dct.idct_3d(1j * kx * mu_hat))
                flux_y = mobility * torch.real(dct.idct_3d(1j * ky * mu_hat))
                flux_z = mobility * torch.real(dct.idct_3d(1j * kz * mu_hat))
                flux_div = dct.dct_3d(flux_x) * 1j * kx + fft.dct_3d(flux_y) * 1j * ky + dct.dct_3d(flux_z) * 1j * kz

                # Update concentration in spectral space
                c_hat += time_increment * self.D * flux_div / (1 + 2 * self.eps * time_increment * self.D * k_squared**2 * A)

                # Transform back to real space
                self.tensor = torch.real(dct.idct_3d(c_hat))

            end = timer()
            self.handle_outputs(vtk_out, verbose, slice=slice)
            if verbose:
                self.print_GPU_stats(start, end, max_iters)


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

        super().__init__(data, device)
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
                print(f"NaN detected in frame {self.frame}. Aborting simulation.")
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
        torch.cuda.reset_peak_memory_stats()

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
                self.print_GPU_stats(start, end, max_iters)

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
        torch.cuda.reset_peak_memory_stats()
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
                self.print_GPU_stats(start, end, max_iters)


# TODO: Umschreiben auf torch from numpy and from 2D to 3D!!
class MultiPhaseSolver(BaseSolver):
    """
    Compute multiphase-field evolution based on Allen-Cahn equations.
    Each phase should be labelled with a unique integer in given array.
    Curvature effects are removed from evolution equation such that shape is preserved.
    Set dx=1 and mobility M=1 in de-dimensionalized equation for solution.
    """
    def __init__(self, data: VoxelFields, fieldname, eps=3, stabilize = 0.0):
        super().__init__(data, device)
        self.eps = eps
        self.stabilize = stabilize
        labelled_array = data.fields[fieldname]
        self.phase_count = np.unique(labelled_array).size
        self.phasefields = np.zeros((self.phase_count,*labelled_array.shape))
        self.labels = []
        for i, label in enumerate(np.unique(labelled_array)):
            self.phasefields[i] = (labelled_array == label).astype(float)
            self.labels.append(f"phi{int(label)}")

    def calc_functional_derivatives(self):
        # Define threshold close to zero to avoid division by zero
        zero = 1e-10
        df_dphi = np.zeros_like(self.phasefields)
        dfgrad_dphi = np.zeros_like(self.phasefields)
        sum_phi_squared = np.zeros_like(dfgrad_dphi[0])
        sum_dfgrad_dphi = np.zeros_like(dfgrad_dphi[0])

        # Construct slices for better readability
        # x-1: left, x+1: right, y-1: bottom, y+1: top
        center = np.s_[1:-1,1:-1]
        left   = np.s_[ :-2,1:-1]
        right  = np.s_[2:  ,1:-1]
        bottom = np.s_[1:-1, :-2]
        top    = np.s_[1:-1,2:  ]

        for i in range(self.phase_count):
            field = np.pad(self.phasefields[i], 1, mode='edge')

            norm2 = 0.25 * ((field[right] - field[left])**2) + 0.25 * ((field[top] - field[bottom])**2)
            # As we will divide by norm2, we need to take care of small values
            bulk = np.where(norm2 <= zero)
            norm2[bulk] = np.inf

            eLe = 0.25 * ((field[right] - field[left])**2) * (field[right] - 2*field[center] + field[left]) \
                + 0.25 * ((field[top] - field[bottom])**2) * (field[top]   - 2*field[center] + field[bottom]) \
                + 0.125 * (field[right] - field[left]) * (field[top] - field[bottom]) * (field[2:,2:] + field[:-2, :-2] - field[:-2,2:] - field[2:,:-2])

            laplace = field[right] - 2*field[center] + field[left] + field[top] - 2*field[center] + field[bottom]
            dfgrad_dphi[i] = self.eps*(self.stabilize*laplace + (1.0-self.stabilize)*eLe/norm2)
            sum_dfgrad_dphi += dfgrad_dphi[i]

            sum_phi_squared += field[center]*field[center]

        # Assemble derivatives of gradient and potential terms
        for i in range(self.phase_count):
            df_dphi[i] = sum_dfgrad_dphi-dfgrad_dphi[i] \
                + 27/2/self.eps*self.phasefields[i]*(sum_phi_squared - self.phasefields[i]*self.phasefields[i]) \
                + 9/2/self.eps*((self.phasefields[i])**3 - self.phasefields[i])

        return df_dphi

    def enforce_gibbs_simplex_contraint(self):
        sum = np.zeros_like(self.phasefields[0])
        for i in range(self.phase_count):
            self.phasefields[i] = np.maximum(self.phasefields[i], 0)
            sum += self.phasefields[i]

        # Normalize the fields to ensure their sum equals 1
        for i in range(self.phase_count):
            self.phasefields[i] /= sum

    def solve_without_curvature(self, steps=1000, frames=10, dt=0.02, convergence = 0.01, verbose=True):
        self.n_out = int(steps/frames)
        df_dphi = np.zeros_like(self.phasefields)

        for it in range(steps):
            if it % self.n_out == 0:
                for i in range(self.phase_count):
                    self.data.add_field(self.labels[i], self.phasefields[i])
                if verbose == 'plot':
                    clear_output(wait=True)
                    self.data.plot_field(self.labels[0])

            df_dphi = self.calc_functional_derivatives()
            sum_df_dphi = np.zeros_like(df_dphi[0])
            for i in range(self.phase_count):
                sum_df_dphi += df_dphi[i]

            for i in range(self.phase_count):
                self.phasefields[i] -= dt * (df_dphi[i] - sum_df_dphi/self.phase_count)

            self.enforce_gibbs_simplex_contraint()

        for i in range(self.phase_count):
            self.data.add_field(self.labels[i], self.phasefields[i])


class TortuositySolver(BaseSolver):
    def __init__(self, data, geometry, fieldname, diffusivity=1.0, omega1=1.0, omega2=1.0, device='cuda', frames=10, max_iters=1000):
        """
        Solver for the Laplace equation with Dirichlet boundary conditions.
        """
        super().__init__(data, device, frames, max_iters)
        self.D = diffusivity
        self.bottom_bc = 0.5
        self.top_bc = -0.5
        # overrelaxation factor
        self.omega1 = omega1 #2 - torch.pi / (1.5 * data.fields[fieldname].shape[1])
        self.omega2 = omega2 #2 - torch.pi / (1.5 * data.fields[fieldname].shape[1])
        self.geometry = geometry
        self.field = fieldname
        self.tensor = torch.tensor(data.fields[fieldname], dtype=self.precision, device=self.device)
        self.tensor = self.tensor.unsqueeze(0).unsqueeze(0)

        self.init_neighbours()
        self.initial_and_boundary_conditions()
        self.init_checkerboard()
        # self.mask = torch.tensor(data.fields[geometry], dtype=self.precision, device=self.device)

    def init_neighbours(self):
        even_neighbours_kernel = torch.tensor([[1, 1, 1],
                                               [1, 0, 1],
                                               [0, 1, 0]],
                                        dtype=self.precision, device=self.device)
        self.even_neighbours_kernel = even_neighbours_kernel.unsqueeze(0).unsqueeze(0)

        odd_neighbours_kernel = torch.tensor([[0, 1, 0],
                                              [1, 0, 1],
                                              [1, 1, 1]],
                                        dtype=self.precision, device=self.device)
        self.odd_neighbours_kernel = odd_neighbours_kernel.unsqueeze(0).unsqueeze(0)
        # Take periodicity in x into account
        mask = torch.tensor(self.data.fields[self.geometry], dtype=self.precision, device=self.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        padded = F.pad(mask, (1,1,1,1), mode='circular')
        even = F.conv2d(padded, self.even_neighbours_kernel)
        odd  = F.conv2d(padded, self.odd_neighbours_kernel)

        # Stitch even and odd columns for full neighbour field
        self.neighbours = torch.zeros_like(mask)
        self.neighbours[:, :, :, ::2] = even[:, :, :, ::2]
        self.neighbours[:, :, :, 1::2] = odd[:, :, :, 1::2]
        self.neighbours[mask==0] = torch.inf
        self.neighbours[self.neighbours==0] = torch.inf

    def initial_and_boundary_conditions(self):
        mask = torch.tensor(self.data.fields[self.geometry], dtype=self.precision, device=self.device)
        vec = torch.linspace(self.bottom_bc, self.top_bc, self.tensor.shape[-1], dtype=self.precision, device=self.device)
        vec = torch.unsqueeze(vec, 0)
        self.tensor[0,0,:,:] = vec.repeat(self.tensor.shape[-2], 1) * mask
        self.tensor[:,:,:,0] = self.bottom_bc * mask[:,0]
        self.tensor[:,:,:,-1] = self.top_bc * mask[:,-1]

    def init_checkerboard(self):
        checkers = torch.zeros_like(self.tensor)
        checkers[:,:,::3,::2] = 1
        checkers[:,:,1::3,1::2] = 1
        checkers[:,:,1::3,::2] = 2
        checkers[:,:,2::3,1::2] = 2
        checkers[:,:,:,0] = -1
        checkers[:,:,:,-1] = -1
        # self.checkers = checkers

        # checkers = checkers[:,:,:,1:-1]
        # Check this strange behaviour!
        self.checkers = [(checkers==0), (checkers==1), (checkers==2)]

    def solve(self):
        """
        Solves the Laplace equation using the specified numerical method.
        """
        self.frame = 0
        increment = torch.zeros_like(self.tensor)
        for i in range(self.max_iters):
            # self.write_frame(self, i)
            if i % self.n_plot == 0:
                self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
                filename = self.field+f"_{self.frame:03d}.vtk"
                self.data.export_to_vtk(filename=filename)
                self.frame += 1

            # Numerical increment
            padded = F.pad(self.tensor, (1,1,1,1), mode='circular')
            even = F.conv2d(padded, self.even_neighbours_kernel)
            odd  = F.conv2d(padded, self.odd_neighbours_kernel)

            # Stitch even and odd columns
            increment[:, :, :, ::2] = even[:, :, :, ::2]
            increment[:, :, :, 1::2] = odd[:, :, :, 1::2]

            increment /= self.neighbours
            increment -= self.tensor
            # self.tensor[:,:,:,1:-1] += self.omega * increment[:,:,:,1:-1]

            # Maybe checkerboarding
            idx = i % 4
            if idx > 0:
                increment = self.omega1 * increment * self.checkers[idx-1]
            self.tensor[:,:,:,1:-1] += self.omega2 * increment[:,:,:,1:-1]
            # self.tensor[self.checkers==idx] += self.omega * increment[self.checkers==idx]


            # Do the same with indexing -> faster?

            # Check for convergence
            # if self.check_convergence(current_error):
            #     print(f"Converged after {i} iterations.")
            #     break

        # Write final frame
        self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
        filename = self.field+f"_{self.frame:03d}.vtk"
        self.data.export_to_vtk(filename=filename)
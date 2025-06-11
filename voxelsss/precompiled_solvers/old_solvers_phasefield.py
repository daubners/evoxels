from IPython.display import clear_output
import numpy as np
import psutil
import sys
from timeit import default_timer as timer
import torch
import torch.fft as fft
import torch.nn.functional as F
import warnings
from ..voxelfields import VoxelFields


class BaseSolver:
    def __init__(self, data: VoxelFields, device="cuda"):
        """
        Base solver class to handle common functionality for solvers.

        Args:
            data (VoxelFields): The voxel field data object containing spatial and field information.
            device (str): The device to perform computations ('cpu' or 'cuda').
        """
        self.data = data
        self.device = torch.device(device)
        # check device is available
        if (
            torch.device(device).type.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            self.device = torch.device("cpu")
            warnings.warn(
                "CUDA not available, defaulting device to cpu. To avoid this warning, set device=torch.device('cpu')"
            )

        if data.precision == np.float32:
            self.precision = torch.float32
        if data.precision == np.float64:
            self.precision = torch.float64

        self.spacing = data.spacing
        # Precompute squares of inverse spacing
        self.div_dx2 = (
            1
            / torch.tensor(self.spacing, dtype=self.precision, device=self.device) ** 2
        )
        self.grid = None
        self.frame = 0

    def init_tensor_from_voxel_field(self, data: VoxelFields, name: str):
        tensor = torch.tensor(
            data.fields[name], dtype=self.precision, device=self.device
        )
        tensor = tensor.unsqueeze(0)
        tensor = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode="circular")
        if data.convention == "staggered_x":
            # For staggered grid we don't need ghost nodes in x
            tensor = tensor[:, 1:-1, :, :]
        return tensor

    def apply_periodic_BC_cell_center(self, tensor):
        """
        Periodic boundary conditions in all directions.
        Consistent with cell centered grid.
        """
        tensor[:, 0, :, :] = tensor[:, -2, :, :]
        tensor[:, -1, :, :] = tensor[:, 1, :, :]
        tensor[:, :, 0, :] = tensor[:, :, -2, :]
        tensor[:, :, -1, :] = tensor[:, :, 1, :]
        tensor[:, :, :, 0] = tensor[:, :, :, -2]
        tensor[:, :, :, -1] = tensor[:, :, :, 1]
        return tensor

    def apply_dirichlet_periodic_BC_cell_center(self, tensor, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with cell centered grid,
        but loss of 2nd order convergence.
        """
        tensor[:, 0, :, :] = 2.0 * bc0 - tensor[:, 1, :, :]
        tensor[:, -1, :, :] = 2.0 * bc1 - tensor[:, -2, :, :]
        tensor[:, :, 0, :] = tensor[:, :, -2, :]
        tensor[:, :, -1, :] = tensor[:, :, 1, :]
        tensor[:, :, :, 0] = tensor[:, :, :, -2]
        tensor[:, :, :, -1] = tensor[:, :, :, 1]
        return tensor

    def apply_dirichlet_periodic_BC_staggered_x(self, tensor, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with staggered_x grid,
        maintains 2nd order convergence.
        """
        tensor[:, 0, :, :] = bc0
        tensor[:, -1, :, :] = bc1
        tensor[:, :, 0, :] = tensor[:, :, -2, :]
        tensor[:, :, -1, :] = tensor[:, :, 1, :]
        tensor[:, :, :, 0] = tensor[:, :, :, -2]
        tensor[:, :, :, -1] = tensor[:, :, :, 1]
        return tensor

    def apply_zero_flux_periodic_BC_cell_center(self, tensor):
        tensor[:, 0, :, :] = tensor[:, 1, :, :]
        tensor[:, -1, :, :] = tensor[:, -2, :, :]
        tensor[:, :, 0, :] = tensor[:, :, -2, :]
        tensor[:, :, -1, :] = tensor[:, :, 1, :]
        tensor[:, :, :, 0] = tensor[:, :, :, -2]
        tensor[:, :, :, -1] = tensor[:, :, :, 1]
        return tensor

    def apply_zero_flux_periodic_BC_staggered_x(self, tensor):
        """
        The following comes out of on interpolation polynomial p with
        p'(0) = 0, p(dx) = f(dx,...), p(2*dx) = f(2*dx,...)
        and then use p(0) for the ghost cell.
        This should be of sufficient order of f'(0) = 0, and even better if
        also f'''(0) = 0 (as it holds for cos(k*pi*x)  )
        """
        fac1 = 4 / 3
        fac2 = 1 / 3
        tensor[:, 0, :, :] = fac1 * tensor[:, 1, :, :] - fac2 * tensor[:, 2, :, :]
        tensor[:, -1, :, :] = fac1 * tensor[:, -2, :, :] - fac2 * tensor[:, -3, :, :]
        tensor[:, :, 0, :] = tensor[:, :, -2, :]
        tensor[:, :, -1, :] = tensor[:, :, 1, :]
        tensor[:, :, :, 0] = tensor[:, :, :, -2]
        tensor[:, :, :, -1] = tensor[:, :, :, 1]
        return tensor

    def calc_laplace(self, tensor):
        """
        Calculate laplace based on compact 2nd order stencil.
        Returned field has same shape as the input tensor (padded with zeros)
        """
        # Manual indexing is ~10x faster than conv3d with laplace kernel
        laplace = torch.zeros_like(tensor)
        laplace[:, 1:-1, 1:-1, 1:-1] = (
            (tensor[:, 2:, 1:-1, 1:-1] + tensor[:, :-2, 1:-1, 1:-1]) * self.div_dx2[0]
            + (tensor[:, 1:-1, 2:, 1:-1] + tensor[:, 1:-1, :-2, 1:-1]) * self.div_dx2[1]
            + (tensor[:, 1:-1, 1:-1, 2:] + tensor[:, 1:-1, 1:-1, :-2]) * self.div_dx2[2]
            - 2 * tensor[:, 1:-1, 1:-1, 1:-1] * torch.sum(self.div_dx2)
        )
        return laplace

    def solve(self):
        """
        Abstract method to be implemented by derived solver classes.
        Raises:
            NotImplementedError: If the method is not overwritten in a child class.
        """
        raise NotImplementedError(
            "The solve method must be implemented in child classes"
        )

    def print_memory_stats(self, start, end, iters):
        """
        Call in verbose mode to return
        Wall time and RAM requirements for simulation.
        """
        print(
            f"Wall time: {np.around(end - start, 4)} s after {iters} iterations ({np.around((end - start) / iters, 4)} s/iter)"
        )
        if self.device.type == "cuda":
            print(
                f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=self.device) / 1e6:.2f} MB reserved)"
            )
            print(
                f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=self.device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=self.device) / 1e6:.2f} MB reserved)"
            )
        elif self.device.type == "cpu":
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")


class PeriodicCahnHilliardSolver(BaseSolver):
    """
    Cahn-Hilliard system with double-well potential energy
    c is dimensionless and can be interpreted as molar fraction or phase fraction
    \epsilon scales the diffuse interface width
    \sigma scales the interfacial energy (set to 1 here)
    F = int( sigma*eps*|grad(c)|^2 + 9*sigma*/eps * c^2 * (1-c)^2).

    The temporal evolution is given as
    dc/dt = div( D*c*(1-c) grad( 18*c*(1-c)*(1-2c) - 2*laplace(c) )) + f(t) .

    This class assumes periodic boundary conditions in all directions.

    Attributes:
        fieldname: Field in the VoxelFields object to perform simulation with.
        diffusivity: Constant diffusion coefficent [m^2/s].
        epsilon: Parameter scaling the diffuse interface width [m]
        device: device for storing torch tensors
    """

    def __init__(
        self,
        data: VoxelFields,
        fieldname,
        diffusivity=1.0,
        epsilon=3,
        rhs_in=None,
        device="cuda",
    ):
        """
        Solves the Cahn-Hilliard equation for two-phase system with double-well energy assuming periodic BC.
        """
        super().__init__(data, device=device)
        self.fieldname = fieldname
        self.D = diffusivity
        self.eps = epsilon
        self.field = self.init_tensor_from_voxel_field(data, fieldname)
        self.rhs = rhs_in
        # Stuff to use the grid
        if rhs_in:
            self.data.add_grid()
            self.grid = torch.stack(
                (
                    torch.from_numpy(self.data.grid[0]),
                    torch.from_numpy(self.data.grid[1]),
                    torch.from_numpy(self.data.grid[2]),
                ),
                dim=0,
            ).to(dtype=self.precision, device=self.device)

    def calc_divergence_variable_mobility(self, padded_mu, padded_c):
        divergence = (
            (padded_mu[:, 2:, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1])
            * 0.5
            * (padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, 2:, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))
            - (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, :-2, 1:-1, 1:-1])
            * 0.5
            * (padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, :-2, 1:-1, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))
        ) * self.div_dx2[0]

        divergence += (
            (padded_mu[:, 1:-1, 2:, 1:-1] - padded_mu[:, 1:-1, 1:-1, 1:-1])
            * 0.5
            * (padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, 1:-1, 2:, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))
            - (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, :-2, 1:-1])
            * 0.5
            * (padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, 1:-1, :-2, 1:-1] + padded_c[:, 1:-1, 1:-1, 1:-1]))
        ) * self.div_dx2[1]

        divergence += (
            (padded_mu[:, 1:-1, 1:-1, 2:] - padded_mu[:, 1:-1, 1:-1, 1:-1])
            * 0.5
            * (padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, 1:-1, 1:-1, 2:] + padded_c[:, 1:-1, 1:-1, 1:-1]))
            - (padded_mu[:, 1:-1, 1:-1, 1:-1] - padded_mu[:, 1:-1, 1:-1, :-2])
            * 0.5
            * (padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1])
            * (1 - 0.5 * (padded_c[:, 1:-1, 1:-1, :-2] + padded_c[:, 1:-1, 1:-1, 1:-1]))
        ) * self.div_dx2[2]
        return divergence

    def initialise_FFT_wavenumbers(self):
        dx, dy, dz = self.spacing
        kx = 2 * torch.pi * fft.fftfreq(self.data.Nx, d=dx, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(self.data.Ny, d=dy, device=self.device)
        kz = 2 * torch.pi * fft.fftfreq(self.data.Nz, d=dz, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx, ky, kz, kx**2 + ky**2 + kz**2

    def enforce_gibbs_simplex_contraint(self):
        self.field = torch.minimum(
            torch.maximum(self.field, torch.tensor(0.0, device=self.device)),
            torch.tensor(1.0, device=self.device),
        )

    def solve(
        self,
        time_increment=0.1,
        frames=10,
        max_iters=100,
        method="mixed_FFT",
        A=0.25,
        verbose=True,
        vtk_out=True,
        plot_bounds=None,
    ):
        """
        Solves the Cahn-Hilliard equation using specified timestepping method:
         - 'explicit': explicit Euler
         - 'full_FFT': semi-implicit timestepping in Fourier space
         - 'mixed_FFT': divergence in real space, update in Fourier space
        """
        if self.data.convention != "cell_center":
            raise ValueError(
                "The fully periodic solver assumes cell centered grid convention."
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters / frames)
        self.frame = 0
        self.time = 0
        self.plot_bounds = plot_bounds
        slice = int(self.data.Nz / 2)

        if method == "full_FFT":
            kx, ky, kz, k_squared = self.initialise_FFT_wavenumbers()
        # elif method == 'mixed_FFT':
        else:
            _, _, _, k_squared = self.initialise_FFT_wavenumbers()
        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i * time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice, iter=i)
                    self.frame += 1

                if method == "explicit":
                    self.apply_periodic_BC_cell_center(self.field)
                    laplace = self.calc_laplace(self.field)
                    mu = (
                        18
                        / self.eps
                        * self.field
                        * (1 - self.field)
                        * (1 - 2 * self.field)
                        - 2 * self.eps * laplace
                    )
                    self.apply_periodic_BC_cell_center(mu)
                    divergence = self.calc_divergence_variable_mobility(mu, self.field)
                    self.field[:, 1:-1, 1:-1, 1:-1] += (
                        time_increment * self.D * divergence
                    )
                    if self.rhs:
                        self.field[:, 1:-1, 1:-1, 1:-1] += time_increment * self.rhs(
                            time_increment * i, *self.grid
                        )

                elif method == "full_FFT":
                    c_inner = self.field[:, 1:-1, 1:-1, 1:-1]
                    dfhom_dc = (
                        18 / self.eps * c_inner * (1 - c_inner) * (1 - 2 * c_inner)
                    )
                    c_hat = fft.fftn(c_inner)
                    mobility = c_inner * (1 - c_inner)
                    mu_hat = fft.fftn(dfhom_dc) + 2 * self.eps * k_squared * c_hat

                    flux_x = mobility * torch.real(fft.ifftn(1j * kx * mu_hat))
                    flux_y = mobility * torch.real(fft.ifftn(1j * ky * mu_hat))
                    flux_z = mobility * torch.real(fft.ifftn(1j * kz * mu_hat))
                    flux_div = 1j * (
                        kx * fft.fftn(flux_x)
                        + ky * fft.fftn(flux_y)
                        + kz * fft.fftn(flux_z)
                    )

                    if self.rhs:
                        rhs_term = fft.fftn(self.rhs(time_increment * i, *self.grid))
                        c_hat += (
                            time_increment
                            * (self.D * flux_div + rhs_term)
                            / (
                                1
                                + 2
                                * self.eps
                                * time_increment
                                * self.D
                                * k_squared**2
                                * A
                            )
                        )
                    else:
                        # Update c_hat using semi-implicit scheme
                        c_hat += (
                            time_increment
                            * self.D
                            * flux_div
                            / (
                                1
                                + 2
                                * self.eps
                                * time_increment
                                * self.D
                                * k_squared**2
                                * A
                            )
                        )

                    # Transform back to real space
                    self.field[:, 1:-1, 1:-1, 1:-1] = torch.real(fft.ifftn(c_hat))
                    # self.enforce_gibbs_simplex_contraint

                elif method == "mixed_FFT":
                    self.apply_periodic_BC_cell_center(self.field)
                    laplace = self.calc_laplace(self.field)
                    mu = (
                        18
                        / self.eps
                        * self.field
                        * (1 - self.field)
                        * (1 - 2 * self.field)
                        - 2 * self.eps * laplace
                    )
                    self.apply_periodic_BC_cell_center(mu)
                    divergence = self.calc_divergence_variable_mobility(mu, self.field)
                    divergence *= self.D
                    if self.rhs:
                        divergence += self.rhs(time_increment * i, *self.grid)
                    # Compute update (in Fourier space) and transform back
                    dc_fft = fft.fftn(divergence)
                    dc_fft *= time_increment / (
                        1 + 2 * self.eps * time_increment * self.D * k_squared**2 * A
                    )
                    self.field[:, 1:-1, 1:-1, 1:-1] += torch.real(fft.ifftn(dc_fft))
                    # self.enforce_gibbs_simplex_contraint

            end = timer()
            self.time = max_iters * time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice, iter=i)
            if verbose:
                self.print_memory_stats(start, end, max_iters)

    def handle_outputs(self, vtk_out, verbose, slice=0, iter=None):
        if self.data.convention == "staggered_x":
            self.data.fields[self.fieldname] = (
                self.field[:, :, 1:-1, 1:-1].squeeze(0).cpu().numpy()
            )
        else:
            self.data.fields[self.fieldname] = (
                self.field[:, 1:-1, 1:-1, 1:-1].squeeze(0).cpu().numpy()
            )
        if np.isnan(self.data.fields[self.fieldname]).any():
            if iter:
                print(
                    f"NaN detected in frame {self.frame} at time {self.time} in iteration {iter}. Aborting simulation."
                )
            else:
                print(
                    f"NaN detected in frame {self.frame} at time {self.time}. Aborting simulation."
                )
            sys.exit(1)  # Exit the program with an error status
        if vtk_out:
            filename = self.fieldname + f"_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=[self.fieldname])
        if verbose == "plot":
            clear_output(wait=True)
            self.data.plot_slice(
                self.fieldname, slice, time=self.time, value_bounds=self.plot_bounds
            )


class MixedCahnHilliardSolver(PeriodicCahnHilliardSolver):
    """
    Cahn-Hilliard system as described in the parent class.
    This solver takes mixed boundary conditions into account.

    Attributes:
        fieldname: Field in the VoxelFields object to perform simulation with.
        diffusivity: Constant diffusion coefficent [m^2/s].
        epsilon: Parameter scaling the diffuse interface width [m]
        device: device for storing torch tensors
    """

    def __init__(
        self,
        data: VoxelFields,
        fieldname,
        diffusivity=1.0,
        epsilon=3,
        mode="dirichlet",
        rhs_in=None,
        device="cuda",
    ):
        super().__init__(
            data,
            fieldname,
            diffusivity=diffusivity,
            epsilon=epsilon,
            rhs_in=rhs_in,
            device=device,
        )
        self.mode = mode

        # Decision tree for correct boundary conditions
        if mode == "dirichlet":
            if data.convention == "staggered_x":
                self.apply_c_bc = self.apply_dirichlet_periodic_BC_staggered_x
                self.apply_mu_bc = self.apply_zero_flux_periodic_BC_staggered_x
                self.apply_flip_bc = self.flip_zero_dirichlet_periodic_staggered_x
            else:
                self.apply_c_bc = self.apply_dirichlet_periodic_BC_cell_center
                self.apply_mu_bc = self.apply_zero_flux_periodic_BC_cell_center
                self.apply_flip_bc = self.flip_zero_dirichlet_periodic_cell_center
        elif mode == "neumann":
            if data.convention != "cell_center":
                raise ValueError("Neumann requires cell_center.")
            zero_flux = self.apply_zero_flux_periodic_BC_cell_center
            # wrap it to accept (tensor, bc0, bc1) but ignore bc0, bc1
            self.apply_c_bc = lambda tensor, bc0, bc1: zero_flux(tensor)
            self.apply_mu_bc = self.apply_zero_flux_periodic_BC_cell_center
            self.apply_flip_bc = self.flip_neumann_periodic_cell_center
        else:
            raise ValueError("Mode must be dirichlet or neumann.")

    def flip_zero_dirichlet_periodic_cell_center(self, tensor):
        pad_right = -tensor[:, :, :, :]
        return torch.cat([tensor, pad_right], dim=1)

    def flip_zero_dirichlet_periodic_staggered_x(self, tensor):
        zeros = torch.zeros(
            tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], device=self.device
        )
        pad_right = -tensor[:, :, :, :]
        return torch.cat([zeros, tensor, zeros, pad_right], dim=1)

    def flip_neumann_periodic_cell_center(self, tensor):
        flipped = torch.cat((tensor, tensor.flip(1)), dim=1)
        return flipped

    def initialise_FFT_wavenumbers(self, convention):
        dx, dy, dz = self.spacing
        if convention == "staggered_x":
            kx = (
                2
                * torch.pi
                * fft.fftfreq(2 * self.data.Nx - 2, d=dx, device=self.device)
            )
        else:
            kx = 2 * torch.pi * fft.fftfreq(2 * self.data.Nx, d=dx, device=self.device)
        ky = 2 * torch.pi * fft.fftfreq(self.data.Ny, d=dy, device=self.device)
        kz = 2 * torch.pi * fft.fftfreq(self.data.Nz, d=dz, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        return kx**2 + ky**2 + kz**2

    def solve(
        self,
        time_increment=0.1,
        frames=10,
        max_iters=100,
        method="mixed_FFT",
        A=0.25,
        bcs=(0.0, 0.0),
        verbose=True,
        vtk_out=True,
        plot_bounds=None,
    ):
        """
        Solves the Cahn-Hilliard equation using specified timestepping method:
         - 'explicit': explicit Euler
         - 'mixed_FFT': divergence in real space, update in Fourier space
        """
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.n_out = int(max_iters / frames)
        self.frame = 0
        self.time = 0
        self.plot_bounds = plot_bounds
        slice = int(self.data.Nz / 2)

        # Will be ignored in Neumann case
        self.bc0 = bcs[0]
        self.bc1 = bcs[1]

        if method == "mixed_FFT":
            k_squared = self.initialise_FFT_wavenumbers(self.data.convention)

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                if i % self.n_out == 0:
                    self.time = i * time_increment
                    self.handle_outputs(vtk_out, verbose, slice=slice, iter=i)
                    self.frame += 1

                self.apply_c_bc(self.field, self.bc0, self.bc1)
                laplace = self.calc_laplace(self.field)
                mu = (
                    18 / self.eps * self.field * (1 - self.field) * (1 - 2 * self.field)
                    - 2 * self.eps * laplace
                )
                self.apply_mu_bc(mu)
                divergence = self.D * self.calc_divergence_variable_mobility(
                    mu, self.field
                )
                if self.rhs:
                    divergence += self.rhs(time_increment * i, *self.grid)
                    # divergence += self.rhs(time_increment*i, *self.grid)[1:-1,:,:]

                if method == "explicit":
                    self.field[:, 1:-1, 1:-1, 1:-1] += time_increment * divergence

                elif method == "mixed_FFT":
                    divergence_flip = self.apply_flip_bc(divergence)
                    div_fft = fft.fftn(divergence_flip)
                    # Compute update (in Fourier space) and transform back
                    dc_fft = (
                        time_increment
                        * div_fft
                        / (
                            1
                            + 2 * self.eps * time_increment * self.D * k_squared**2 * A
                        )
                    )
                    if self.data.convention == "cell_center":
                        self.field[:, 1:-1, 1:-1, 1:-1] += torch.real(
                            fft.ifftn(dc_fft)
                        )[:, : self.data.Nx, :, :]
                    else:
                        self.field[:, :, 1:-1, 1:-1] += torch.real(fft.ifftn(dc_fft))[
                            :, : self.data.Nx, :, :
                        ]

            end = timer()
            self.time = max_iters * time_increment
            self.handle_outputs(vtk_out, verbose, slice=slice, iter=i)
            if verbose:
                self.print_memory_stats(start, end, max_iters)

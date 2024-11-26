### Voxel-based Solvers (HexSers)

# In a world of cubes and blocks,
# Where reality takes voxel knocks,
# Every shape and form we see,
# Is a pixelated mystery.

# Mountains rise in jagged peaks,
# Rivers flow in blocky streaks.
# So embrace the charm of this edgy place,
# Where every voxel finds its space

# In einer Welt aus Würfeln und Blöcken,
# in der die Realität in Voxelform erscheint,
# ist jede Form, die wir sehen,
# ein verpixeltes Rätsel.

# Berge erheben sich in gezackten Gipfeln,
# Flüsse fließen in blockförmigen Adern.
# Also lass dich vom Charme dieses kantigen Ortes verzaubern,
# wo jedes Voxel seinen Platz findet.

### Simon Daubner (s.daubner@imperial.ac.uk)
### Dyson School of Design Engineering
### Imperial College London

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pyvista as pv
import sys
from timeit import default_timer as timer
import torch
import torch.nn.functional as F


class voxelFields:
    def __init__(self, num_x: int, num_y: int, num_z: int, spacing: float):
        """
        Initialize voxel grid with given dimensions and cell size.
        """
        self.Nx = num_x
        self.Ny = num_y
        self.Nz = num_z

        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ValueError("spacing must be a list or tuple with three elements (dx, dy, dz)")
        if not all(isinstance(x, (int, float)) for x in spacing):
            raise ValueError("All elements in spacing must be integers or floats")
        if (np.max(spacing)/np.min(spacing) > 10):
            warnings.warn("Simulations become very questionable for largely different spacings e.g. dz >> dx.")
        self.spacing = spacing
        self.origin = (0, 0, 0)
        self.fields = {}

    def add_field(self, name: str, array=None):
        """
        Initializes field data associated given name.
        If an array is provided, it checks if it is a numpy array
        with the correct shape; otherwise, it initializes a zero array.
        """
        if array is not None:
            # Check if array is a numpy array and if it has the right shape
            if isinstance(array, np.ndarray):
                if array.shape == (self.Nx, self.Ny, self.Nz):
                    self.fields[name] = array
                else:
                    raise ValueError(f"The provided array must have the shape ({self.num_x}, {self.num_y}, {self.num_z}).")
            else:
                raise TypeError("The provided array must be a numpy array.")
        else:
            self.fields[name] = np.zeros((self.Nx, self.Ny, self.Nz))

    def export_to_vtk(self, filename="output.vtk", field_names=None):
        """
        Export a 3D numpy array to VTK format for visualization in VisIt or ParaView.
        """
        # Create a structured grid from the array
        grid = pv.ImageData()
        grid.dimensions = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        grid.spacing = self.spacing
        grid.origin = self.origin

        if field_names is not None:
            names = field_names
        else:
            names = list(self.fields.keys())

        for name in names:
            grid.cell_data[name] = self.fields[name].flatten(order="F")  # Fortran order flattening
        grid.save(filename)

    def plotSlice(self, fieldname, slice_index, direction='z', dpi=200, colormap='viridis'):
        plt.figure(figsize=(5, 5), dpi=dpi)
        # Colormaps
        # linear: viridis, Greys
        # diverging: seismic
        # levels: tab20, flag
        # gradual: turbo
        if direction == 'x':
            slice = np.s_[slice_index,:,:]
            end1 = self.spacing[1]
            end2 = self.spacing[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            slice = np.s_[:,slice_index,:]
            end1 = self.spacing[0]
            end2 = self.spacing[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            slice = np.s_[:,:,slice_index]
            end1 = self.spacing[0]
            end2 = self.spacing[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        plt.imshow(self.fields[fieldname][slice], cmap=colormap, origin='lower', extent=[0, end1, 0, end2])
        plt.xlabel(label1)
        plt.ylabel(label2)
        title = fieldname + ' in ' + direction
        plt.title(title)
        plt.show()

    def plotFieldInteractive(self, fieldname, direction='x', dpi=200, colormap='viridis'):
        if direction == 'x':
            axes = (0,1,2)
            end1 = self.spacing[1]
            end2 = self.spacing[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            axes = (1,0,2)
            end1 = self.spacing[0]
            end2 = self.spacing[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            axes = (2,0,1)
            end1 = self.spacing[0]
            end2 = self.spacing[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        field = np.transpose(self.fields[fieldname], axes)
        max_id = np.max(np.unique(field))
        # Create the initial plot
        fig, ax = plt.subplots(dpi=dpi)
        im = ax.imshow(np.transpose(field[0]), cmap=colormap, origin='lower', extent=[0, end1, 0, end2], vmin=0, vmax=max_id)

        # Add a slider for changing timeframes
        layer = field.shape[0]-1
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        axis_slider = Slider(ax_slider, 'slice', 0, layer, valinit=0, valstep=1)

        # Function to update the plot based on the slider value
        def update(val):
            x = int(axis_slider.val)
            im.set_array(field[x].T)
            ax.set_title(f'Slice {x} in ' + direction + '-direction of '+fieldname)
            fig.canvas.draw_idle()

        # Connect the slider to the update function
        axis_slider.on_changed(update)

        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        return axis_slider


class BaseSolver:
    def __init__(self, data: voxelFields, device='cuda'):
        """
        Base solver class to handle common functionality for different solvers.
        Args:
            data (HexDiscretisation): The hexagonal object containing the field data.
            device (str): The device to perform computations ('cpu' or 'cuda').
            tolerance (float): Convergence tolerance.
            max_iters (int): Maximum number of iterations.
        """
        self.data = data
        self.device = torch.device(device)
        self.precision = torch.float32

    # def write_frame(self, data, fieldname, current_iteration: int):
    #     if current_iteration%self.n_plot == 0
    #         data.fields[fieldname] = field.cpu().numpy()
    #         filename = fieldname+f"_{frame:03d}.vtk"
    #         data.export_to_vtk(filename=filename)
    #         self.frame += 1

    # def check_convergence(self, current_error):
    #     """
    #     Check if the solution has converged.
    #     """
    #     return current_error < self.tolerance

    def solve(self):
        """
        Abstract method to be implemented in child classes.
        """
        raise NotImplementedError("The solve method must be implemented in child classes")


class CahnHilliardSolver(BaseSolver):
    # Note: This is the simplest Cahn-Hilliard system for now with
    # F = int( eps*|grad(phi)|^2 + 9/eps * phi^2 * (1-phi)^2)
    # and dc/dt = D*laplace( 18*c*(1-c)*(1-2c) - 2*laplace(c) ).
    # A non-linear mobility e.g. M=D*c*(1-c) is notoriously more difficult to implement in Fourier space.

    # The current implementation assumes isotropic voxels (dx=dy=dz).
    # However, this could be easily changed.

    def __init__(self, data: voxelFields, fieldname, device='cuda'):
        """
        Solves the Cahn-Hilliard equation for two-phase system with double-well energy assuming periodic BC.
        """
        super().__init__(data, device)
        self.field = fieldname
        self.tensor = torch.tensor(data.fields[fieldname], dtype=self.precision, device=self.device)
        self.tensor = self.tensor.unsqueeze(0).unsqueeze(0)

        laplace_kernel =torch.tensor([[[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]],

                                      [[0, 1, 0],
                                       [1, -6, 1],
                                       [0, 1, 0]],

                                      [[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]]],
                                    dtype=self.precision, device=device)
        self.laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)

    def calc_laplace(self, tensor):
        padded = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
        self.laplace = F.conv3d(padded, self.laplace_kernel)

    def calc_laplace2(self, tensor):
        tensor = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
        self.laplace = F.conv3d(padded, self.laplace_kernel)
        tensor = tensor[:, :, 1:-1, 1:-1, 1:-1]

    # def calc_laplace2(self, tensor):
    #     padded = F.pad(tensor, (1,1,1,1,1,1), mode='circular')
    #     self.laplace = padded[:, :, 2:, 1:-1, 1:-1] + \
    #                    padded[:, :, :-2, 1:-1, 1:-1] + \
    #                    padded[:, :, 1:-1, 2:, 1:-1] + \
    #                    padded[:, :, 1:-1, :-2, 1:-1] + \
    #                    padded[:, :, 1:-1, 1:-1, 2:] + \
    #                    padded[:, :, 1:-1, 1:-1, :-2] - \
    #                  6*padded[:, :, 1:-1, 1:-1, 1:-1]

    def solve(self, diffusivity=1.0, time_increment=0.01, epsilon=3, frames=10, max_iters=1000, verbose=True):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        # Specific numerical implementation for Cahn-Hilliard equation
        self.n_plot = int(max_iters/frames)
        self.frame = 0
        self.D = diffusivity
        self.eps = epsilon

        if verbose == 'plot':
            slice = int(self.tensor.shape[-1]/2)
            fig, ax = plt.subplots(dpi=200)
            im = ax.imshow(self.tensor[0,0,:,:,slice].cpu().numpy(), cmap='turbo', origin='lower', vmin=0, vmax=1)
            ax.set_title("Cahn-Hilliard Simulation")
            fig.colorbar(im, ax=ax)
            plt.ion()  # Turn on interactive mode

        with torch.no_grad():
            start = timer()
            for i in range(max_iters):
                # self.write_frame(self, i)
                if i % self.n_plot == 0:
                    self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
                    filename = self.field+f"_{self.frame:03d}.vtk"
                    self.data.export_to_vtk(filename=filename, field_names=[self.field])
                    if verbose == 'plot':
                        im.set_data(self.data.fields[self.field][:,:,slice])
                        ax.set_title(f"Step {self.frame}")
                        plt.draw()  # Redraw the figure
                        plt.pause(0.01)
                    if np.isnan(self.data.fields[self.field]).any():
                        print(f"NaN detected in frame {self.frame}. Aborting simulation.")
                        sys.exit(1)  # Exit the program with an error status
                    self.frame += 1

                # Numerical increment
                self.calc_laplace(self.tensor)
                mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*self.laplace
                self.calc_laplace(mu)
                self.tensor += time_increment * self.D * self.laplace

            if verbose:
                print(f'Wall time: {np.around(timer() - start, 4)} s ({np.around((timer() - start)/max_iters, 4)} s/iter)')
            if verbose == 'plot':
                plt.ioff()
                plt.show()

            self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
            filename = self.field+f"_{self.frame:03d}.vtk"
            self.data.export_to_vtk(filename=filename, field_names=[self.field])

    def solve2(self, diffusivity=1.0, time_increment=0.01, epsilon=3, frames=10, max_iters=1000):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        # Specific numerical implementation for Cahn-Hilliard equation
        self.n_plot = int(max_iters/frames)
        self.frame = 0
        self.D = diffusivity
        self.eps = epsilon

        for i in range(max_iters):
            # self.write_frame(self, i)
            if i % self.n_plot == 0:
                self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
                filename = self.field+f"_{self.frame:03d}.vtk"
                self.data.export_to_vtk(filename=filename, field_names=[self.field])
                self.frame += 1

            # Numerical increment
            self.calc_laplace2(self.tensor)
            mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*self.laplace
            self.calc_laplace2(mu)
            self.tensor += time_increment * self.D * self.laplace

        self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
        filename = self.field+f"_{self.frame:03d}.vtk"
        self.data.export_to_vtk(filename=filename, field_names=[self.field])

    def solveFFT(self, diffusivity=1.0, time_increment=0.01, epsilon=3, frames=10, max_iters=1000):
        """
        Solves the Cahn-Hilliard equation using the specified numerical method.
        """
        # Specific numerical implementation for Cahn-Hilliard equation
        self.n_plot = int(max_iters/frames)
        self.frame = 0
        self.D = diffusivity
        self.eps = epsilon

        for i in range(max_iters):
            # self.write_frame(self, i)
            if i % self.n_plot == 0:
                self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
                filename = self.field+f"_{self.frame:03d}.vtk"
                self.data.export_to_vtk(filename=filename, field_names=[self.field])
                self.frame += 1

            # Numerical increment
            self.calc_laplace2(self.tensor)
            mu = 18/self.eps*self.tensor*(1-self.tensor)*(1-2*self.tensor) - 2*self.eps*self.laplace
            self.calc_laplace2(mu)
            self.tensor += time_increment * self.D * self.laplace

        self.data.fields[self.field] = self.tensor.squeeze().cpu().numpy()
        filename = self.field+f"_{self.frame:03d}.vtk"
        self.data.export_to_vtk(filename=filename, field_names=[self.field])


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


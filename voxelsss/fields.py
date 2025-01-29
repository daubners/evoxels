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

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pyvista as pv
import warnings

class VoxelFields:
    """
    Represents a 3D voxel grid with fields for simulation and visualization.

    Attributes:
        Nx, Ny, Nz: Number of voxels along the x, y and z-axis.
        domain_size (tuple): Length of physical domain (Lx, Ly, Lz)
        spacing (tuple): Grid spacing along each axis (dx, dy, dz).
        origin (tuple): Position of lower left corner (for vtk export)
        fields (dict): Dictionary to store named 3D fields.
    """

    def __init__(self, num_x: int, num_y: int, num_z: int, domain_size = (1,1,1), convention='cell_center'):
        """
        Initializes the voxel grid with specified dimensions and domain size.

        Raises:
            ValueError: If spacing is not a list or tuple with three elements or contains non-numeric values.
            Warning: If spacings differ significantly, a warning is issued.
        """
        self.Nx = num_x
        self.Ny = num_y
        self.Nz = num_z

        if not isinstance(domain_size, (list, tuple)) or len(domain_size) != 3:
            raise ValueError("spacing must be a list or tuple with three elements (dx, dy, dz)")
        if not all(isinstance(x, (int, float)) for x in domain_size):
            raise ValueError("All elements in domain_size must be integers or floats")
        self.domain_size = domain_size
        self.spacing = (domain_size[0]/num_x, domain_size[1]/num_y, domain_size[2]/num_z)
        if (np.max(self.spacing)/np.min(self.spacing) > 10):
            warnings.warn("Simulations become very questionable for largely different spacings e.g. dz >> dx.")
        self.origin = (0, 0, 0)
        if convention == 'cell_center':
            self.origin = (self.spacing[0]/2, self.spacing[1]/2, self.spacing[2]/2)
        self.grid = None
        self.fields = {}

    def add_field(self, name: str, array=None):
        """
        Adds a field to the voxel grid.

        Args:
            name (str): Name of the field.
            array (numpy.ndarray, optional): 3D array to initialize the field. If None, initializes with zeros.

        Raises:
            ValueError: If the provided array does not match the voxel grid dimensions.
            TypeError: If the provided array is not a numpy array.
        """
        if array is not None:
            if isinstance(array, np.ndarray):
                if array.shape == (self.Nx, self.Ny, self.Nz):
                    self.fields[name] = array
                else:
                    raise ValueError(f"The provided array must have the shape ({self.num_x}, {self.num_y}, {self.num_z}).")
            else:
                raise TypeError("The provided array must be a numpy array.")
        else:
            self.fields[name] = np.zeros((self.Nx, self.Ny, self.Nz))

    def add_grid(self):
        """
        Creates the meshgrid for a regular voxel grid layout if it doesn't already exist.
        """
        if self.grid is None:
            x_lin = np.arange(0, self.Nx, dtype=np.float32) * self.spacing[0] + self.origin[0]
            y_lin = np.arange(0, self.Ny, dtype=np.float32) * self.spacing[1] + self.origin[1]
            z_lin = np.arange(0, self.Nz, dtype=np.float32) * self.spacing[2] + self.origin[2]
            x, y, z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
            self.grid = (x, y, z)

    def export_to_vtk(self, filename="output.vtk", field_names=None):
        """
        Exports fields to a VTK file for visualization (e.g. VisIt or ParaView).

        Args:
            filename (str): Name of the output VTK file.
            field_names (list, optional): List of field names to export. Exports all fields if None.
        """
        grid = pv.ImageData()
        grid.dimensions = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        grid.spacing = self.spacing
        grid.origin = self.origin

        names = field_names if field_names else list(self.fields.keys())
        for name in names:
            grid.cell_data[name] = self.fields[name].flatten(order="F")  # Fortran order flattening
        grid.save(filename)

    def plot_slice(self, fieldname, slice_index, direction='z', time=None, colormap='viridis'):
        """
        Plots a 2D slice of a field along a specified direction.

        Args:
            fieldname (str): Name of the field to plot.
            slice_index (int): Index of the slice to plot.
            direction (str): Normal direction of the slice ('x', 'y', or 'z').
            dpi (int): Resolution of the plot.
            colormap (str): Colormap to use for the plot.

        Raises:
            ValueError: If an invalid direction is provided.
        """
        # Colormaps
        # linear: viridis, Greys
        # diverging: seismic
        # levels: tab20, flag
        # gradual: turbo
        if direction == 'x':
            slice = np.s_[slice_index,:,:]
            end1, end2 = self.domain_size[1], self.domain_size[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            slice = np.s_[:,slice_index,:]
            end1, end2 = self.domain_size[0], self.domain_size[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            slice = np.s_[:,:,slice_index]
            end1, end2 = self.domain_size[0], self.domain_size[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        plt.figure()
        im = plt.imshow(self.fields[fieldname][slice].T, cmap=colormap, origin='lower', extent=[0, end1, 0, end2])
        plt.colorbar(im)
        plt.xlabel(label1)
        plt.ylabel(label2)
        if time:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction} at time {time}')
        else:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction}')
        plt.show()

    def plot_field_interactive(self, fieldname, direction='x', colormap='viridis'):
        """
        Creates an interactive plot for exploring slices of a 3D field.

        Args:
            fieldname (str): Name of the field to plot.
            direction (str): Direction of slicing ('x', 'y', or 'z').
            dpi (int): Resolution of the plot.
            colormap (str): Colormap to use for the plot.

        Raises:
            ValueError: If an invalid direction is provided.
        """
        if direction == 'x':
            axes = (0,1,2)
            end1, end2 = self.spacing[1], self.spacing[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            axes = (1,0,2)
            end1, end2 = self.spacing[0], self.spacing[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            axes = (2,0,1)
            end1, end2 = self.spacing[0], self.spacing[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        field = np.transpose(self.fields[fieldname], axes)
        max_id = np.max(np.unique(field))
        fig, ax = plt.subplots()
        im = ax.imshow(field[0].T, cmap=colormap, origin='lower', extent=[0, end1, 0, end2], vmin=0, vmax=max_id)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'Slice 0 in {direction}-direction of {fieldname}')
        plt.colorbar(im, ax=ax)

        # Add a slider for changing timeframes
        position = plt.axes([0.2, 0.0, 0.6, 0.02])
        ax_slider = Slider(position, 'Slice', 0, field.shape[0]-1, valinit=0, valstep=1)

        def update(val):
            slice_idx = int(ax_slider.val)
            im.set_array(field[slice_idx].T)
            ax.set_title(f'Slice {slice_idx} in ' + direction + '-direction of ' + fieldname)
            fig.canvas.draw_idle()

        ax_slider.on_changed(update)
        return ax_slider
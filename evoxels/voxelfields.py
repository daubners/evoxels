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

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from typing import Tuple
import warnings
from .voxelgrid import Grid

class VoxelFields:
    """Manage 3D voxel grids for simulation, visualization, and I/O.

    This class provides a uniform, cell‐centered or staggered‐x voxel grid,
    handles spacing and origin, and stores any number of named 3D fields.

    Args:
        shape (tuple[int, int, int]): Number of voxels ``(Nx, Ny, Nz)``.
        domain_size (tuple[float, float, float], optional):
            Physical dimensions (Lx, Ly, Lz). Defaults to (1, 1, 1).
        convention (str, optional):
            Grid convention, either 'cell_center' or 'staggered_x'.
            Defaults to 'cell_center'.

    Raises:
        ValueError: If `domain_size` is not length 3 or contains non-numeric values.
        ValueError: If `convention` is not one of 'cell_center' or 'staggered_x'.
        Warning: If the spacing ratio max/min > 10, a warning is issued.

    Attributes:
        shape (tuple[int, int, int]): Number of voxels ``(Nx, Ny, Nz)``.
        domain_size (tuple[float, float, float]): Physical domain lengths.
        spacing (tuple[float, float, float]): Grid spacing (dx, dy, dz).
        origin (tuple[float, float, float]):
            Coordinates of the (0, 0, 0) corner for cell-centered or staggered grids.
        convention (str): Either 'cell_center' or 'staggered_x'.
        precision (type): NumPy floating-point type for grid coordinates.
        grid (tuple[np.ndarray, np.ndarray, np.ndarray] or None):
            Meshgrid arrays (x, y, z) once created by `add_grid()`, else None.
        fields (dict[str, np.ndarray]): Mapping field names to 3D arrays.

    Example:
        >>> vf = VoxelFields((100, 100, 100), domain_size=(1, 1, 1))
        >>> vf.add_field('temperature', np_array)
        >>> x, y, z = vf.plot_slice('temperature', 10)
    """

    def __init__(self, shape: Tuple[int, int, int], domain_size=(1, 1, 1), convention='cell_center'):
        """Create a voxel grid with ``shape`` cells."""
        if not (
            isinstance(shape, (list, tuple))
            and len(shape) == 3
            and all(isinstance(n, (int, np.integer)) for n in shape)
        ):
            raise ValueError("shape must be a tuple of three integers")
        self.shape = tuple(int(n) for n in shape)
        num_x, num_y, num_z = self.shape
        self.precision = 'float32'  # float64
        self.convention = convention

        if not isinstance(domain_size, (list, tuple)) or len(domain_size) != 3:
            raise ValueError("domain_size must be a list or tuple with three elements (dx, dy, dz)")
        if not all(isinstance(x, (int, float)) for x in domain_size):
            raise ValueError("All elements in domain_size must be integers or floats")
        self.domain_size = domain_size

        if convention == 'cell_center':
            self.spacing = (domain_size[0]/num_x, domain_size[1]/num_y, domain_size[2]/num_z)
            self.origin = (self.spacing[0]/2, self.spacing[1]/2, self.spacing[2]/2)
        elif convention == 'staggered_x':
            self.spacing = (domain_size[0]/(num_x-1), domain_size[1]/num_y, domain_size[2]/num_z)
            self.origin = (0, self.spacing[1]/2, self.spacing[2]/2)
        else:
            raise ValueError("Chosen convention must be cell_center or staggered_x.")

        if (np.max(self.spacing)/np.min(self.spacing) > 10):
            warnings.warn("Simulations become very questionable for largely different spacings e.g. dz >> dx.")
        self.grid = None
        self.fields = {}

    @property
    def Nx(self) -> int:  # backward compatibility
        return self.shape[0]

    @property
    def Ny(self) -> int:
        return self.shape[1]

    @property
    def Nz(self) -> int:
        return self.shape[2]

    def __str__(self):
        """Return a human readable description of the voxel grid."""
        return (
            f"Domain with size {self.domain_size} and "
            f"{self.shape} grid points on "
            f"{self.convention} position."
        )

    def grid_info(self):
        """Return a :class:`Grid` dataclass describing this domain."""
        grid = Grid(self.shape, self.origin, self.spacing, self.convention)
        return grid

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
                if array.shape == self.shape:
                    self.fields[name] = array
                else:
                    raise ValueError(
                        f"The provided array must have the shape {self.shape}."
                    )
            else:
                raise TypeError("The provided array must be a numpy array.")
        else:
            self.fields[name] = np.zeros(self.shape)

    def set_voxel_sphere(self, name: str, center, radius, label: int | float = 1):
        """Create a voxelized representation of a sphere in 3D
        
        Fill voxels within given ``radius`` around the given ``center``
        with value provided by ``label``.
        """
        x, y, z = np.ogrid[:self.Nx, :self.Ny, :self.Nz]
        distance_squared = (x * self.spacing[0] + self.origin[0] - center[0])**2 +\
                           (y * self.spacing[1] + self.origin[1] - center[1])**2 +\
                           (z * self.spacing[2] + self.origin[2] - center[2])**2
        mask = distance_squared <= radius**2
        self.fields[name][mask] = label

    def average(self, name: str):
        """Return the average value of a stored field."""
        if self.convention == 'cell_center':
            average = np.mean(self.fields[name])
        elif self.convention == 'staggered_x':
            # Count first and last slice as half cells
            average = np.sum(self.fields[name][1:-1,:,:]) \
                      + 0.5*np.sum(self.fields[name][ 0,:,:]) \
                      + 0.5*np.sum(self.fields[name][-1,:,:])
            average /= ((self.Nx - 1) * self.Ny * self.Nz)
        return average
    
    def axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns the 1D coordinate arrays along each axis. """
        return tuple(
            np.arange(0, n, dtype=self.precision) * self.spacing[i] + self.origin[i]
            for i, n in enumerate(self.shape)
        )
    
    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Returns full 3D mesh grids for each axis. """
        ax = self.axes()
        # indexing='ij' makes Ax[i,j,k] = x-coordinate at (i,j,k), etc.
        return tuple(np.meshgrid(*ax, indexing='ij'))

    def export_to_vtk(self, filename="output.vtk", field_names=None):
        """
        Exports fields to a VTK file for visualization (e.g. VisIt or ParaView).

        Args:
            filename (str): Name of the output VTK file.
            field_names (list, optional): List of field names to export. Exports all fields if None.
        """
        import pyvista as pv
        names = field_names if field_names else list(self.fields.keys())
        grid = pv.ImageData()
        grid.spacing = self.spacing
        grid.dimensions = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        grid.origin = (self.origin[0] - self.spacing[0]/2, \
                        self.origin[1] - self.spacing[1]/2, \
                        self.origin[2] - self.spacing[2]/2)
        for name in names:
            grid.cell_data[name] = self.fields[name].flatten(order="F")  # Fortran order flattening
        grid.save(filename)

    def plot_slice(self, fieldname, slice_index, direction='z', time=None, colormap='viridis', value_bounds=None):
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
            start1, start2 = self.origin[1]-self.spacing[1]/2, self.origin[2]-self.spacing[2]/2
            end1, end2 = self.domain_size[1]-start1, self.domain_size[2]-start2
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            slice = np.s_[:,slice_index,:]
            start1, start2 = self.origin[0]-self.spacing[0]/2, self.origin[2]-self.spacing[2]/2
            end1, end2 = self.domain_size[0]-start1, self.domain_size[2]-start2
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            slice = np.s_[:,:,slice_index]
            start1, start2 = self.origin[0]-self.spacing[0]/2, self.origin[1]-self.spacing[1]/2
            end1, end2 = self.domain_size[0]-start1, self.domain_size[1]-start2
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        plt.figure()
        if value_bounds is not None:
            im = plt.imshow(self.fields[fieldname][slice].T, cmap=colormap,\
                            origin='lower', extent=[start1, end1, start2, end2],\
                                vmin=value_bounds[0], vmax=value_bounds[1])
        else:
            im = plt.imshow(self.fields[fieldname][slice].T, cmap=colormap, \
                            origin='lower', extent=[start1, end1, start2, end2])

        ratio = np.clip((end2-start2)/(end1-start1), 0, 1)
        plt.colorbar(im, shrink=ratio)
        plt.xlabel(label1)
        plt.ylabel(label2)
        if time:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction} at time {time}')
        else:
            plt.title(f'Slice {slice_index} of {fieldname} in {direction}')
        plt.show()

    def plot_field_interactive(self, fieldname, direction='x', colormap='viridis', value_bounds=None):
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
            end1, end2 = self.domain_size[1], self.domain_size[2]
            label1, label2 = ['Y', 'Z']
        elif direction == 'y':
            axes = (1,0,2)
            end1, end2 = self.domain_size[0], self.domain_size[2]
            label1, label2 = ['X', 'Z']
        elif direction == 'z':
            axes = (2,0,1)
            end1, end2 = self.domain_size[0], self.domain_size[1]
            label1, label2 = ['X', 'Y']
        else:
            raise ValueError("Given direction must be x, y or z")

        field = np.transpose(self.fields[fieldname], axes)
        fig, ax = plt.subplots()
        if value_bounds is None:
            value_bounds = (np.min(field), np.max(field))
        im = ax.imshow(
            field[0].T,
            cmap=colormap,
            origin="lower",
            extent=[0, end1, 0, end2],
            vmin=value_bounds[0],
            vmax=value_bounds[1],
        )
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

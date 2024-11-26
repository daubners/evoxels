
### Some helper functions for handling data
import numpy as np

def add_voxel_sphere(array, center_x, center_y, center_z, radius, label=1):
    """
    Create a voxelized representation of a sphere in 3D array based on
    given midpoint and radius in terms of pixel resolution.
    """
    nx, ny, nz = array.shape
    x, y, z = np.ogrid[:nx, :ny, :nz]

    distance_squared = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2 + (z - center_z + 0.5)**2
    mask = distance_squared <= radius**2
    array[mask] = label
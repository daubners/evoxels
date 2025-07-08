"""Tests for field handling with VoxelFields class."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import evoxels as evo

def test_voxelFields_init():
    N = 10
    vf = evo.VoxelFields((N, N, N))
    assert vf.shape == (N, N, N)
    assert (vf.Nx, vf.Ny, vf.Nz) == (N, N, N)

def test_voxelFields_init_domain():
    N = 10
    vf = evo.VoxelFields((N, N, N), (N, N, N))
    assert (vf.domain_size, vf.spacing) == ((N, N, N), (1, 1, 1))

def test_voxelFields_grid_cell_centered():
    Nx, Ny, Nz = 10, 5, 7
    vf = evo.VoxelFields((Nx, Ny, Nz), (Nx, Ny, Nz))
    (x,y,z) = vf.meshgrid()
    assert x[-1,0,0] == Nx-vf.spacing[0]/2

def test_voxelFields_grid_staggered_x():
    Nx, Ny, Nz = 10, 5, 7
    vf = evo.VoxelFields((Nx + 1, Ny, Nz), (Nx, Ny, Nz), convention='staggered_x')
    (x,y,z) = vf.meshgrid()
    assert (x[-1,0,0] == Nx) and (y[0,-1,0] == Ny-vf.spacing[1]/2)

def test_voxelFields_init_fields():
    N = 10
    vf = evo.VoxelFields((N, N, N), (N, N, N))
    vf.add_field("c", 0.123*np.ones((N, N, N)))

    assert (vf.fields['c'][1,2,3], *vf.fields['c'].shape) == (0.123, N, N, N)

@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_plot_functions():
    vf = evo.VoxelFields((3, 3, 3))
    vf.add_field("f", np.arange(27).reshape(vf.shape))
    with pytest.raises(ValueError):
        vf.plot_slice("f", 0, direction="bad")
    vf.plot_slice("f", 0, direction="x")
    vf.plot_slice("f", 1, direction="y")
    vf.plot_slice("f", 2, direction="z")
    slider = vf.plot_field_interactive("f", direction="y")
    slider.set_val(1)
    plt.close("all")

def test_mock_vtk_export():
    vf = evo.VoxelFields((3, 3, 3))
    vf.add_field('test1')
    vf.add_field('test2')
    with pytest.raises(ValueError):
        vf.export_to_vtk('bad_name')

def test_sphere_volume_fraction():
    vf = evo.VoxelFields((6, 5, 5), convention='staggered_x')
    vf.add_field("sphere")
    vf.set_voxel_sphere("sphere", center=(0.5, 0.5, 0.5), radius=0.31, label=1)
    assert np.count_nonzero(vf.fields["sphere"] == 1) == 20
    assert vf.average('sphere') == 0.16

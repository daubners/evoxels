"""Tests for field handling with VoxelFields class."""

import numpy as np
import voxelsss as vox


def test_voxelFields_init():
    N = 10
    sim = vox.VoxelFields(N, N, N)
    assert (sim.Nx, sim.Ny, sim.Nz) == (N, N, N)


def test_voxelFields_init_domain():
    N = 10
    sim = vox.VoxelFields(N, N, N, (N, N, N))
    assert (sim.domain_size, sim.spacing) == ((N, N, N), (1, 1, 1))


def test_voxelFields_grid_cell_centered():
    Nx, Ny, Nz = 10, 5, 7
    sim = vox.VoxelFields(Nx, Ny, Nz, (Nx, Ny, Nz))
    (x, y, z) = sim.meshgrid()
    assert x[-1, 0, 0] == Nx - sim.spacing[0] / 2


def test_voxelFields_grid_staggered_x():
    Nx, Ny, Nz = 10, 5, 7
    sim = vox.VoxelFields(Nx + 1, Ny, Nz, (Nx, Ny, Nz), convention="staggered_x")
    (x, y, z) = sim.meshgrid()
    assert (x[-1, 0, 0] == Nx) and (y[0, -1, 0] == Ny - sim.spacing[1] / 2)


def test_voxelFields_init_fields():
    N = 10
    sim = vox.VoxelFields(N, N, N, (N, N, N))
    sim.add_field("c", 0.123 * np.ones((N, N, N)))

    assert (sim.fields["c"][1, 2, 3], *sim.fields["c"].shape) == (0.123, N, N, N)

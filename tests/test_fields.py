"""Tests for field handling with VoxelFields class."""

import numpy as np
import voxelsss as vox

def test_voxelFields_init():
    N = 10
    sim = vox.VoxelFields(N, N, N)
    assert (sim.Nx, sim.Ny, sim.Nz) == (N, N, N)

def test_voxelFields_init_domain():
    N = 10
    sim = vox.VoxelFields(N, N, N, (N,N,N))
    assert (sim.domain_size, sim.spacing) == ((N, N, N),(1, 1, 1))

def test_voxelFields_init_grid():
    N = 10
    sim = vox.VoxelFields(N, N, N, (N,N,N))
    sim.add_grid()
    (x,y,z) = sim.grid

    assert x[-1,0,0] == N-sim.spacing[0]/2

def test_voxelFields_init_fields():
    N = 10
    sim = vox.VoxelFields(N, N, N, (N,N,N))
    sim.add_field("c", 0.123*np.ones((N, N, N)))

    assert (sim.fields['c'][1,2,3], *sim.fields['c'].shape) == (0.123, N, N, N)
"""Tests for spatial finite difference discretizations."""

import numpy as np
import voxelsss as vox
from numpy import sin, cos, pi

### Generalized test case
def grid_convergence_test(
    test_function,      # a callable(vgrid, field)->torch.Tensor (shape [x,y,z])
    init_fun,           # exact init_fun(x,y,z)->np.ndarray
    exact_fun,          # exact_fun(x,y,z)->np.ndarray
    convention='cell_center',
    dtype='float32',
    powers = np.array([3,4,5,6,7]),
    backend = 'torch'
):
    dx     = np.zeros(len(powers))
    errors = np.zeros(len(powers))

    for i, p in enumerate(powers):
        if convention == 'cell_center':
            vf = vox.VoxelFields(2**p, 2**p, 2**p, (1,1,1), convention=convention)
        elif convention == 'staggered_x':
            vf = vox.VoxelFields(2**p+1, 2**p, 2**p, (1,1,1), convention=convention)
        vf.precision = dtype
        grid = vf.meshgrid()
        init_data = init_fun(*grid)

        if backend == 'torch':
            vg = vox.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device='cpu')
        elif backend == 'jax':
            vg = vox.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)
        field = vg.init_field_from_numpy(init_data)

        # Compute solutions
        comp = vg.export_field_to_numpy(test_function(vg, field))
        exact = exact_fun(*grid)
        if convention == 'staggered_x':
            comp = comp[1:-1,:,:]
            exact = exact[1:-1,:,:]

        # Error norm
        diff = comp - exact
        errors[i] = np.linalg.norm(diff)/np.linalg.norm(exact)
        dx[i]     = vf.spacing[0]

    # Fit slope
    slope, _ = np.polyfit(np.log(dx), np.log(errors),1)
    assert abs(slope - 2) < 0.1, f"expected order 2, got {slope:.2f}"


# Test 1: Laplacian stencil with periodic boundary data
def initial_periodic(x,y,z): 
    return sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) * (2**2 + 4**2 + 6**2)**(-1) *pi**(-2)

def laplace_periodic(x,y,z):
    return -sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z)

def calc_laplace_periodic(vg, field):
    field = vg.apply_periodic_BC(field)
    laplace = vg.calc_laplace(field)
    return laplace

def test_periodic_laplace_torch():
    grid_convergence_test(
        test_function = calc_laplace_periodic,
        init_fun      = initial_periodic,
        exact_fun     = laplace_periodic,
        convention    = 'cell_center',
        dtype         = 'float32',
        backend       = 'torch'
    )

def test_periodic_laplace_jax():
    grid_convergence_test(
        test_function = calc_laplace_periodic,
        init_fun      = initial_periodic,
        exact_fun     = laplace_periodic,
        convention    = 'cell_center',
        dtype         = 'float32',
        backend       = 'jax'
    )

# Test 2: Laplace with zero BC!
def initial_zero_dirichlet(x,y,z):
    return (x*(1-x))**2

def laplace_zero_dirichlet(x,y,z):
    return 2 - 12*x + 12*x**2

def calc_laplace_zero_dirichlet(vg, field):
    field = vg.apply_dirichlet_periodic_BC(field)
    laplace = vg.calc_laplace(field)
    return laplace

def test_laplace_zero_dirichlet():
    grid_convergence_test(
        test_function = calc_laplace_zero_dirichlet,
        init_fun      = initial_zero_dirichlet,
        exact_fun     = laplace_zero_dirichlet,
        convention    = 'staggered_x',
        dtype         = 'float32',
        backend       = 'torch'
    )


# Test 3: Laplace with non-zero BC!
bc_l, bc_r = (1,-1)

def initial_nonzero_dirichlet(x,y,z):
    return cos(pi*x)**3

def laplace_nonzero_dirichlet(x,y,z):
    return  3*pi**2*(2*sin(pi*x)**2 - cos(pi*x)**2)*cos(pi*x)

def calc_laplace_nonzero_dirichlet(vg, field):
    field = vg.apply_dirichlet_periodic_BC(field, bc0=bc_l, bc1=bc_r)
    laplace = vg.calc_laplace(field)
    return laplace

def test_laplace_nonzero_dirichlet():
    grid_convergence_test(
        test_function = calc_laplace_nonzero_dirichlet,
        init_fun      = initial_nonzero_dirichlet,
        exact_fun     = laplace_nonzero_dirichlet,
        convention    = 'staggered_x',
        dtype         = 'float32',
        backend       = 'torch'
    )


# Test 4: Laplace with zero Neumann!
# Both cell_center and staggered_x convention preserve second order
def initial_zero_flux_BC(x,y,z):
    return cos(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) / (2**2 + 4**2 + 6**2) / pi**2

def laplace_zero_flux_BC(x,y,z):
    return -cos(2*pi*x)*sin(4*pi*y)*sin(6*pi*z)

def calc_laplace_zero_flux_BC(vg, field):
    field = vg.apply_zero_flux_periodic_BC(field)
    laplace = vg.calc_laplace(field)
    return laplace

def test_laplace_zero_flux_cell_center():
    grid_convergence_test(
        test_function = calc_laplace_zero_flux_BC,
        init_fun      = initial_zero_flux_BC,
        exact_fun     = laplace_zero_flux_BC,
        convention    = 'cell_center',
        dtype         = 'float32',
        backend       = 'torch'
    )

def test_laplace_zero_flux_staggered():
    grid_convergence_test(
        test_function = calc_laplace_zero_flux_BC,
        init_fun      = initial_zero_flux_BC,
        exact_fun     = laplace_zero_flux_BC,
        convention    = 'staggered_x',
        dtype         = 'float32',
        backend       = 'torch'
    )
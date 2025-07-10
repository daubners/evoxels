"""Tests for spatial finite difference discretizations."""

import sympy as sp
import sympy.vector as spv
import importlib.util
import pytest
from evoxels.problem_definition import ReactionDiffusion
from evoxels.utils import rhs_convergence_test

jax_available = importlib.util.find_spec("jax") is not None

CS = spv.CoordSys3D('CS')
def forcing(u,t):
    return 0


# Test 1: Laplacian stencil with periodic boundary data
test_fun1 = sp.sin(2*sp.pi*CS.x)*sp.sin(4*sp.pi*CS.y)*sp.sin(6*sp.pi*CS.z) \
            * (2**2 + 4**2 + 6**2)**(-1) *sp.pi**(-2)

def test_periodic_laplace_torch():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'periodic'},
        test_function  = test_fun1,
        convention     = 'cell_center',
        dtype          = 'float32',
        backend       = 'torch'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

@pytest.mark.skipif(not jax_available, reason="jax not installed")
def test_periodic_laplace_jax():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'periodic'},
        test_function  = test_fun1,
        convention     = 'cell_center',
        dtype          = 'float32',
        backend       = 'jax'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"


# Test 2: Laplace with zero BC!
test_fun2 = (CS.x*(1-CS.x))**2

def test_laplace_zero_dirichlet():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'dirichlet', 'bcs': (0,0)},
        test_function  = test_fun2,
        convention     = 'staggered_x',
        dtype          = 'float32',
        backend       = 'torch'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"


# Test 3: Laplace with non-zero BC!
test_fun3 = sp.cos(sp.pi*CS.x)**3

def test_laplace_nonzero_dirichlet():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'dirichlet', 'bcs': (1,-1)},
        test_function  = test_fun3,
        convention     = 'staggered_x',
        dtype          = 'float32',
        backend       = 'torch'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"


# Test 4: Laplace with zero Neumann!
# Both cell_center and staggered_x convention preserve second order
test_fun = sp.cos(2*sp.pi*CS.x)*sp.sin(4*sp.pi*CS.y)*sp.sin(6*sp.pi*CS.z) \
           / (2**2 + 4**2 + 6**2) / sp.pi**2

def test_laplace_zero_flux_cell_center():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'neumann'},
        test_function  = test_fun3,
        convention     = 'cell_center',
        dtype          = 'float32',
        backend       = 'torch'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

def test_laplace_zero_flux_staggered():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusion,
        problem_kwargs = {'D': 1.0, 'f': forcing, \
                          'BC_type': 'neumann'},
        test_function  = test_fun3,
        convention     = 'staggered_x',
        dtype          = 'float32',
        backend       = 'torch'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

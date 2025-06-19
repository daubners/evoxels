"""Tests for rhs of problem definition."""

import sympy as sp
import sympy.vector as spv
from voxelsss.problem_definition import PeriodicCahnHilliard, AllenCahnEquation
from voxelsss.utils import rhs_convergence_test

CS = spv.CoordSys3D('CS')
test_fun = 0.4 + 0.1 * sp.sin(2*sp.pi*CS.x)

def test_Cahn_Hilliard_divergence_for_given_c():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = PeriodicCahnHilliard,
        problem_kwargs = {'eps': 3.0, 'D': 1.0, 'A': 0.25},
        test_function  = test_fun,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

test_fun2 = 0.5 + 0.3 * sp.cos(4*sp.pi*CS.x)\
                         * sp.cos(2*sp.pi*CS.y)\
                         * (CS.z**2/2 - CS.z**3/3)

def test_Allen_Cahn_divergence_for_given_c():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = AllenCahnEquation,
        problem_kwargs = {'eps': 3.0, 'curvature': 0.5},
        test_function  = test_fun2,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

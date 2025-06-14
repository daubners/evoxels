"""Tests for rhs of problem definition."""

import sympy as sp
import sympy.vector as spv
from voxelsss.problem_definition import PeriodicCahnHilliard
from voxelsss.utils import rhs_convergence_test

alpha = 0.4
beta = 0.1

CS = spv.CoordSys3D('CS')
test_fun = alpha + beta * sp.sin(2*sp.pi*CS.x)

def test_Cahn_Hilliard_divergence_for_given_c():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = PeriodicCahnHilliard,
        problem_kwargs = {'eps': 3.0, 'D': 1.0, 'A': 0.25},
        test_function  = test_fun,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

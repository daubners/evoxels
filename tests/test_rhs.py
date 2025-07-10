"""Tests for rhs of problem definition."""

import sympy as sp
import sympy.vector as spv
from evoxels.problem_definition import PeriodicCahnHilliard, \
    AllenCahnEquation, CoupledReactionDiffusion, ReactionDiffusionSBM
from evoxels.utils import rhs_convergence_test

CS = spv.CoordSys3D('CS')
test_fun_ch = 0.4 + 0.1 * sp.sin(2*sp.pi*CS.x)

def test_Cahn_Hilliard_rhs():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = PeriodicCahnHilliard,
        problem_kwargs = {'eps': 3.0, 'D': 1.0, 'A': 0.25},
        test_function  = test_fun_ch,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

# Construct test function which fulfills zero flux BC
# and has significant curvature in the domain
test_fun_ac = 0.5 + 0.3 * sp.cos(4*sp.pi*CS.x)\
                         * sp.cos(2*sp.pi*CS.y)\
                         * (CS.z**2/2 - CS.z**3/3)

def test_Allen_Cahn_rhs():
    _ ,_ , slope, order = rhs_convergence_test(
        ODE_class      = AllenCahnEquation,
        problem_kwargs = {'eps': 3.0, 'curvature': 0.5, 'force': 1},
        test_function  = test_fun_ac,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"


test_funs_rd = (0.5 + 0.3 * sp.cos(4*sp.pi*CS.x) * sp.cos(4*sp.pi*CS.y)**3,
             0.4 + 0.1 * sp.sin(2*sp.pi*CS.x) * sp.cos(4*sp.pi*CS.z) )

def test_coupled_reaction_diffusion_rhs():
    _, _, slopes, order = rhs_convergence_test(
        ODE_class      = CoupledReactionDiffusion,
        problem_kwargs = {"D_A": 1.0, "D_B": 0.5},
        test_function  = test_funs_rd,
        convention     = 'cell_center',
        dtype          = 'float64'
    )
    assert all(abs(s - order) < 0.1 for s in slopes),\
        f"expected order {order}, got {slopes[0]:.2f} and {slopes[1]:.2f}"
    

test_fun_sbm = CS.x + (CS.x*(1-CS.x))
mask_fun = 0.5 + 0.3*sp.cos(4*sp.pi*CS.x) * sp.cos(2*sp.pi*CS.y)

def test_reaction_diffusion_smoothed_boundary_rhs():
    _, _, slope, order = rhs_convergence_test(
        ODE_class      = ReactionDiffusionSBM,
        problem_kwargs = {"D": 1.0, "BC_type": 'dirichlet', "bcs": (0,1),},
        test_function  = test_fun_sbm,
        mask_function  = mask_fun,
        convention     = 'staggered_x',
        dtype          = 'float32'
    )
    assert abs(slope - order) < 0.1, f"expected order {order}, got {slope:.2f}"

"""Tests for spatial finite difference discretizations."""

import numpy as np
import voxelsss as vox
import torch
from numpy import sin, cos, pi

### Generalized test case
def run_convergence_test(
    solver_call,        # a callable(vf, fieldname)->solver
    test_function,      # a callable(solver)->torch.Tensor   (shape [x,y,z])
    init_fun,           # exact init_fun(x,y,z)->np.ndarray
    exact_fun,          # exact_fun(x,y,z)->np.ndarray
    convention='cell_center',
    dtype=np.float32,
    powers = np.array([3,4,5,6,7])
):
    dx     = np.zeros(len(powers))
    errors = np.zeros(len(powers))

    for i, p in enumerate(powers):
        if convention == 'cell_center':
            vf = vox.VoxelFields(2**p, 2**p, 2**p, (1,1,1), convention=convention)
        elif convention == 'staggered_x':
            vf = vox.VoxelFields(2**p+1, 2**p, 2**p, (1,1,1), convention=convention)
        vf.precision = dtype
        vf.add_grid()
        init_data = init_fun(*vf.grid)
        vf.add_field("c", init_data)

        # Compute solutions
        solver = solver_call(vf, "c", device='cpu')
        comp = test_function(solver).squeeze(0).cpu().numpy()
        exact = exact_fun(*vf.grid)
        if convention == 'staggered_x':
            exact = exact[1:-1,:,:]

        # Error norm
        diff = comp - exact
        errors[i] = np.linalg.norm(diff)/np.linalg.norm(exact)
        dx[i]     = vf.spacing[0]

    # Fit slope
    slope, _ = np.polyfit(np.log(dx), np.log(errors),1)
    assert abs(slope - 2) < 0.1, f"expected order 2, got {slope:.2f}"


# Test 1: Laplacian stencil with periodic boundary data
def init_fun_1(x,y,z): 
    return sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) * (2**2 + 4**2 + 6**2)**(-1) *pi**(-2)

def laplace_init_fun_1(x,y,z):
    return -sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z)

def test_periodic_laplace():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.PeriodicCahnHilliardSolver(vf,f,device=device),
        test_function  = lambda s: s.calc_laplace(s.apply_periodic_BC_cell_center(s.field))[:,1:-1,1:-1,1:-1],
        init_fun       = init_fun_1,
        exact_fun      = laplace_init_fun_1,
        convention     = 'cell_center',
        dtype          = np.float32
    )


# Test 2: 2nd order convergence of div(c*(1-c)*grad(mu))
alpha = 0.4
beta = 0.1

def init_fun_2(x,y,z): 
    return alpha + beta*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)

def init_fun_mu_2(x,y,z):
    return sin(2*pi*x)

def result_fun_2(x,y,z):
    return 4*pi**2*(-beta*(alpha + beta*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z))*sin(2*pi*y)*sin(2*pi*z)*cos(2*pi*x)**2 - beta*(alpha + beta*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) - 1)*sin(2*pi*y)*sin(2*pi*z)*cos(2*pi*x)**2 + (alpha + beta*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z))*(alpha + beta*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) - 1)*sin(2*pi*x))

def test_divergence_for_given_mu():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.PeriodicCahnHilliardSolver(vf,f,device=device),
        # Test calc_divergence_variable_mobility(padded_mu, padded_c)
        test_function  = lambda s: s.calc_divergence_variable_mobility(
                                s.apply_periodic_BC_cell_center(init_mu := torch.from_numpy(np.pad(init_fun_mu_2(*s.data.grid), 1)).unsqueeze(0)),
                                s.apply_periodic_BC_cell_center(s.field)
                            ),
        init_fun       = init_fun_2,
        exact_fun      = result_fun_2,
        convention     = 'cell_center',
        dtype          = np.float32
    )


# Test 3: 2nd order convergence of div(c*(1-c)*grad(mu)) with mu computed numerically
# IMPORTANT: fails with float32!!
alpha = 0.4
beta = 0.1

def init_fun_3(x,y,z):
    return  alpha + beta*sin(2*pi*x)
    
def result_fun_3(x,y,z):
    return pi**2*beta*(-2*beta*(alpha + beta*sin(2*pi*x))*(72.0*alpha**2 + 144.0*alpha*beta*sin(2*pi*x) - 72.0*alpha - 72.0*beta**2*(1 - cos(2*pi*x))**2 - 144.0*beta**2*cos(2*pi*x) + 144.0*beta**2 - 72.0*beta*sin(2*pi*x) + 12.0 + 48.0*pi**2)*cos(2*pi*x)**2 - 2*beta*(alpha + beta*sin(2*pi*x) - 1)*(72.0*alpha**2 + 144.0*alpha*beta*sin(2*pi*x) - 72.0*alpha - 72.0*beta**2*(1 - cos(2*pi*x))**2 - 144.0*beta**2*cos(2*pi*x) + 144.0*beta**2 - 72.0*beta*sin(2*pi*x) + 12.0 + 48.0*pi**2)*cos(2*pi*x)**2 + (alpha + beta*sin(2*pi*x))*(alpha + beta*sin(2*pi*x) - 1)*(144.0*alpha**2*sin(2*pi*x) - 576.0*alpha*beta*(1 - cos(2*pi*x))**2 - 1152.0*alpha*beta*cos(2*pi*x) + 864.0*alpha*beta - 144.0*alpha*sin(2*pi*x) - 432.0*beta**2*(1 - cos(2*pi*x))**2*sin(2*pi*x) + 576.0*beta**2*sin(2*pi*x) - 432.0*beta**2*sin(4*pi*x) + 288.0*beta*(1 - cos(2*pi*x))**2 + 576.0*beta*cos(2*pi*x) - 432.0*beta + 24.0*sin(2*pi*x) + 96.0*pi**2*sin(2*pi*x)))

def test_divergence_for_given_c():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.PeriodicCahnHilliardSolver(vf,f,device=device),
        # Test calc_divergence_variable_mobility(padded_mu, padded_c)
        test_function  = lambda s: s.calc_divergence_variable_mobility(
                                s.apply_periodic_BC_cell_center(
                                  (mu:= 18/s.eps*s.field*(1-s.field)*(1-2*s.field) 
                                   -2*s.eps*s.calc_laplace(s.apply_periodic_BC_cell_center(s.field)))), \
                                s.apply_periodic_BC_cell_center(s.field)
                                ),
        init_fun       = init_fun_3,
        exact_fun      = result_fun_3,
        convention     = 'cell_center',
        dtype          = np.float64
    )


# Test 4: Laplace with zero BC!
def init_fun_4(x,y,z):
    return sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z)  *  ( 2**2 + 4**2 + 6**2)**(-1) *pi**(-2)

def laplace_init_fun_4(x,y,z):
    return sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) * (-1)

def test_laplace_with_dirichlet_BC():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.MixedCahnHilliardSolver(vf,f,device=device),
        test_function  = lambda s: s.calc_laplace(s.apply_dirichlet_periodic_BC_staggered_x(s.field))[:,1:-1,1:-1,1:-1],
        init_fun       = init_fun_4,
        exact_fun      = laplace_init_fun_4,
        convention     = 'staggered_x',
        dtype          = np.float64
    )


# Test 5: Laplace with non-zero BC!
bc_l, bc_r = (0.7,0.7)

def init_fun_5(x,y,z):
    b = bc_l
    return b + (x*(1-x))**2 + sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) * ( 2**2 + 4**2 + 6**2)**(-1) *pi**(-2)

def laplace_init_fun_5(x,y,z):
    return  2*(x**2 + 4*x*(x - 1) + (x - 1)**2) + sin(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) *  -1

def test_laplace_with_nonzero_dirichlet_BC():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.MixedCahnHilliardSolver(vf,f,device=device),
        test_function  = lambda s: s.calc_laplace(s.apply_dirichlet_periodic_BC_staggered_x(s.field, bc0=bc_l, bc1=bc_r))[:,1:-1,1:-1,1:-1],
        init_fun       = init_fun_5,
        exact_fun      = laplace_init_fun_5,
        convention     = 'staggered_x',
        dtype          = np.float64
    )


# Test 6: Laplace with zero Neumann!
def init_fun_6(x,y,z):
    return   cos(2*pi*x)*sin(4*pi*y)*sin(6*pi*z)  *  ( 2**2 + 4**2 + 6**2)**(-1) *pi**(-2)

def laplace_init_fun_6(x,y,z):
    return  cos(2*pi*x)*sin(4*pi*y)*sin(6*pi*z) *  (-1) 

def test_laplace_with_neumann_BC():
    run_convergence_test(
        solver_call    = lambda vf, f, device='cpu': 
                             vox.MixedCahnHilliardSolver(vf,f,device=device),
        test_function  = lambda s: s.calc_laplace(s.apply_zero_flux_periodic_BC_cell_center(s.field))[:,1:-1,1:-1,1:-1],
        init_fun       = init_fun_6,
        exact_fun      = laplace_init_fun_6,
        convention     = 'cell_center',
        dtype          = np.float64
    )
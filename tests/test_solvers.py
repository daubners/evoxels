"""Tests for solver functionality."""

import importlib.util
import numpy as np
import pytest
import evoxels as evo
from evoxels.solvers import TimeDependentSolver

jax_available = importlib.util.find_spec("jax") is not None

def test_time_solver_multiple_fields():
    """Test calling custom step function and multiple fields"""
    vf = evo.VoxelFields((4, 4, 4))
    vf.add_field("a", np.ones(vf.shape))
    vf.add_field("b", np.zeros(vf.shape))

    def step(u, t):
        return u + 1

    solver = TimeDependentSolver(vf, ["a", "b"], backend="torch", step_fn=step, device="cpu")
    solver.solve(frames=1, max_iters=1, verbose=False, jit=False)

    assert np.allclose(vf.fields["a"], 2)
    assert np.allclose(vf.fields["b"], 1)

@pytest.mark.skipif(not jax_available, reason="jax not installed")
def test_1D_analytical_tanh_profile():
    """1D analytical phase-field solution
    
    The 1D equilibrium solution of the double well potential
    is a  tanh profile. This is valid for both the Allen-Cahn
    equation and the Cahn-Hilliard equation.
    """
    Nx = 16
    vf = evo.VoxelFields((Nx, 1, 1), domain_size=(Nx, 1, 1))
    phi = np.zeros((Nx, 1, 1), dtype=np.float32)
    phi[: Nx // 2] = 1.0
    vf.add_field("phi1", phi)
    vf.add_field("phi2", phi)

    eps = 3.0
    evo.run_allen_cahn_solver(
        vf,
        "phi1",
        backend="torch",
        device="cpu",
        frames=1,
        max_iters=13,
        time_increment=0.1,
        eps=eps,
        jit=False,
        verbose=False,
    )

    evo.run_cahn_hilliard_solver(
        vf,
        "phi2",
        backend="jax",
        frames=1,
        max_iters=30,
        time_increment=1,
        eps=eps,
        jit=True,
        verbose=False,
    )

    phi1_numeric = vf.fields["phi1"].squeeze()
    phi2_numeric = vf.fields["phi2"].squeeze()

    x = np.arange(Nx) + 0.5
    phi_analytic = 0.5 - 0.5*np.tanh(3*(x - 0.5*Nx) / 2 / eps)
    L2_error1 = np.linalg.norm(phi1_numeric - phi_analytic)
    L2_error2 = np.linalg.norm(phi2_numeric[(x>5) & (x<11)] -\
                               phi_analytic[(x>5) & (x<11)] )
    
    assert L2_error1 < 0.05,\
        f"Allen-Cahn error for 1D profile is > 5% ({L2_error1:.2f})"
    assert L2_error2 < 0.05,\
        f"Cahn-Hilliard error for 1D profile is > 5% ({L2_error2:.2f})"

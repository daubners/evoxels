"""Tests for solver functionality."""

import importlib.util
import numpy as np
import pytest
import evoxels as evo
from evoxels.problem_definition import TwoPhaseAllenCahn, ReactionDiffusion
from evoxels.solvers import TimeDependentSolver
from evoxels.timesteppers import ExponentialEuler
from evoxels.voxelgrid import VoxelGridTorch

jax_available = importlib.util.find_spec("jax") is not None

def test_time_solver_multiple_fields():
    """Test calling custom step function and multiple fields"""
    vf = evo.VoxelFields((4, 4, 4))
    vf.add_field("a", np.ones(vf.shape))
    vf.add_field("b", np.zeros(vf.shape))

    def step(t, u):
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
    vf.add_field("phi1", phi.copy())
    vf.add_field("phi2", phi.copy())

    eps = 3.0
    evo.run_allen_cahn_solver(
        vf,
        "phi1",
        backend="torch",
        device="cpu",
        frames=1,
        max_iters=10,
        time_increment=0.5,
        eps=eps,
        jit=False,
        verbose=False,
    )

    evo.run_cahn_hilliard_solver(
        vf,
        "phi2",
        backend="jax",
        frames=1,
        max_iters=10,
        time_increment=0.5,
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


def test_reaction_diffusion_normalizes_bc():
    vf = evo.VoxelFields((4, 4, 4))
    vg = VoxelGridTorch(vf.grid_info(), device="cpu")
    with pytest.warns(
        UserWarning,
        match="Applying Dirichlet BCs on a cell_center grid reduces the spatial order of convergence to 0.5!",
    ):
        problem = ReactionDiffusion(
            vg,
            D=1.0,
            bc=(('dirichlet', (1, -1)), 'periodic', 'periodic'),
        )

    assert problem.bc == (
        ("dirichlet", (1, -1)),
        ("periodic", None),
        ("periodic", None),
    )


def test_reaction_diffusion_mixed_bc_uses_generic_padding_fallback():
    vf = evo.VoxelFields((2, 2, 2))
    vg = VoxelGridTorch(vf.grid_info(), device="cpu")
    with pytest.warns(
        UserWarning,
        match="Applying Dirichlet BCs on a cell_center grid reduces the spatial order of convergence to 0.5!",
    ):
        problem = ReactionDiffusion(
            vg,
            D=1.0,
            bc=(('dirichlet', (10.0, 20.0)), 'neumann', 'periodic'),
        )
    field = vg.init_scalar_field(np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2))

    padded = vg.to_numpy(problem.pad_bc(field))[0]
    expected = np.pad(np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2), 1, mode='wrap')
    expected[0, :, :] = 2.0 * 10.0 - expected[1, :, :]
    expected[-1, :, :] = 2.0 * 20.0 - expected[-2, :, :]
    expected[:, 0, :] = expected[:, 1, :]
    expected[:, -1, :] = expected[:, -2, :]

    assert np.allclose(padded, expected)


def test_reaction_diffusion_dirichlet_periodic_keeps_specialized_padding():
    vf = evo.VoxelFields((2, 2, 2))
    vg = VoxelGridTorch(vf.grid_info(), device="cpu")
    with pytest.warns(
        UserWarning,
        match="Applying Dirichlet BCs on a cell_center grid reduces the spatial order of convergence to 0.5!",
    ):
        problem = ReactionDiffusion(
            vg,
            D=1.0,
            bc=(('dirichlet', (1.0, -1.0)), 'periodic', 'periodic'),
        )
    field = vg.init_scalar_field(np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2))

    padded = problem.pad_bc(field)
    expected = vg.bc.pad_dirichlet_periodic(field, 1.0, -1.0)

    assert np.allclose(vg.to_numpy(padded), vg.to_numpy(expected))


def test_reaction_diffusion_neumann_periodic_keeps_specialized_padding():
    vf = evo.VoxelFields((2, 2, 2))
    vg = VoxelGridTorch(vf.grid_info(), device="cpu")
    problem = ReactionDiffusion(
        vg,
        D=1.0,
        bc=('neumann', 'periodic', 'periodic'),
    )
    field = vg.init_scalar_field(np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2))

    padded = problem.pad_bc(field)
    expected = vg.bc.pad_zero_flux_periodic(field)

    assert np.allclose(vg.to_numpy(padded), vg.to_numpy(expected))


def test_exponential_euler_rejects_full_neumann_semilinear_problem():
    vf = evo.VoxelFields((4, 4, 4))
    vg = VoxelGridTorch(vf.grid_info(), device="cpu")
    problem = TwoPhaseAllenCahn(vg)

    with pytest.raises(ValueError, match="periodic boundary conditions in y and z"):
        ExponentialEuler(problem, 0.1)

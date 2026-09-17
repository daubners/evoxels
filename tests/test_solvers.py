"""Tests for solver functionality."""

import importlib.util

import numpy as np
import pytest

import evoxels as evo
from evoxels.pdes import ReactionDiffusion, TwoPhaseAllenCahn
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


def test_precompiled_multi_phase_solver_labels_mode():
    vf = evo.VoxelFields((4, 4, 1))
    labels = np.zeros(vf.shape, dtype=np.int32)
    labels[2:, :, :] = 1
    vf.add_field("labels", labels)

    evo.run_multi_phase_solver(
        vf,
        "labels",
        backend="torch",
        device="cpu",
        frames=1,
        max_iters=1,
        jit=False,
        verbose=False,
    )

    assert set(np.unique(vf.fields["labels"])).issubset({0, 1})


def test_precompiled_multi_phase_solver_validates_bulk_driving_forces():
    vf = evo.VoxelFields((4, 4, 1))
    labels = np.zeros(vf.shape, dtype=np.int32)
    labels[2:, :, :] = 1
    vf.add_field("labels", labels)

    with pytest.raises(ValueError, match="one value per phase"):
        evo.run_multi_phase_solver(
            vf,
            "labels",
            backend="torch",
            device="cpu",
            bulk_driving_forces=(0.0,),
        )

@pytest.mark.skipif(not jax_available, reason="jax not installed")
def test_1D_analytical_tanh_profile():
    """1D analytical phase-field solution
    
    The 1D equilibrium solution of the double-well potential is a tanh
    profile for two-phase Allen-Cahn, Cahn-Hilliard, and two-channel
    multiphase Allen-Cahn.
    """
    Nx = 16
    vf = evo.VoxelFields((Nx, 1, 1), domain_size=(Nx, 1, 1))
    phi = np.zeros((Nx, 1, 1), dtype=np.float32)
    phi[: Nx // 2] = 1.0
    vf.add_field("phi1", phi.copy())
    vf.add_field("phi2", phi.copy())
    vf.add_field("phia", phi.copy())
    vf.add_field("phib", (1 - phi).copy())

    eps = 3.0
    steps = 20
    bcs = ("neumann", "periodic", "periodic")
    evo.run_allen_cahn_solver(
        vf,
        "phi1",
        backend="torch",
        device="cpu",
        frames=1,
        max_iters=steps,
        time_increment=0.5,
        eps=eps,
        curvature=1.0,
        bc=bcs,
        jit=False,
        verbose=False,
    )

    evo.run_cahn_hilliard_solver(
        vf,
        "phi2",
        backend="jax",
        frames=1,
        max_iters=steps,
        time_increment=0.5,
        eps=eps,
        bc=bcs,
        jit=True,
        verbose=False,
    )
    evo.run_multi_phase_solver(
        vf,
        ("phia", "phib"),
        backend="torch",
        device="cpu",
        from_labels=False,
        time_increment=0.5,
        frames=1,
        max_iters=steps,
        eps=eps,
        M=1.0,
        bc=bcs,
        jit=False,
        verbose=False,
    )

    phi1_numeric = vf.fields["phi1"].squeeze()
    phi2_numeric = vf.fields["phi2"].squeeze()
    phi3_numeric = vf.fields["phia"].squeeze()

    x = np.arange(Nx) + 0.5
    phi_analytic = 0.5 - 0.5*np.tanh(3*(x - 0.5*Nx) / 2 / eps)
    L2_error1 = np.linalg.norm(phi1_numeric - phi_analytic)
    L2_error2 = np.linalg.norm(phi2_numeric - phi_analytic)
    L2_error3 = np.linalg.norm(phi3_numeric - phi_analytic)
    
    assert L2_error1 < 0.03,\
        f"Allen-Cahn error for 1D profile is > 3% ({L2_error1:.2f})"
    assert L2_error2 < 0.03,\
        f"Cahn-Hilliard error for 1D profile is > 3% ({L2_error2:.2f})"
    assert L2_error3 < 0.03,\
        f"Multiphase error for 1D profile is > 3% ({L2_error3:.2f})"
    np.testing.assert_allclose(phi1_numeric, phi3_numeric, rtol=0, atol=1e-6)


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

    with pytest.raises(ValueError, match="support at most one non-periodic axis"):
        ExponentialEuler(problem, 0.1)

"""Focused diffrax adapter coverage for JAX-compatible problems and steppers."""

import importlib.util

import numpy as np
import pytest
import evoxels as evo
from evoxels.diffrax_adapter import DiffraxTimeStepperAdapter
from evoxels.problem_definition import (
    CahnHilliard,
    CoupledReactionDiffusion,
    MultiPhaseAllenCahn,
    ReactionDiffusion,
    ReactionDiffusionSBM,
    TwoPhaseAllenCahn,
)
from evoxels.timesteppers import (
    ExponentialEuler,
    ForwardEuler,
    PseudoSpectralIMEX,
    RKC1,
    RKC2,
    RungeKutta4,
)
from evoxels.voxelgrid import VoxelGridJax

diffrax_available = importlib.util.find_spec("diffrax") is not None


def _solve_with_adapter(problem, y0, stepper_cls, dt=0.05):
    import diffrax as dfx
    import jax.numpy as jnp

    solver = DiffraxTimeStepperAdapter(stepper_cls(problem, dt))
    saveat = dfx.SaveAt(ts=jnp.array([0.0, dt], dtype=jnp.float32))
    solution = dfx.diffeqsolve(
        dfx.ODETerm(lambda t, y, args: problem.rhs(t, y)),
        solver,
        t0=saveat.subs.ts[0],
        t1=saveat.subs.ts[-1],
        dt0=dt,
        y0=y0,
        saveat=saveat,
        max_steps=16,
        throw=False,
        adjoint=dfx.ForwardMode(),
    )
    return solution.ys


def _scalar_state(vg, data):
    return vg.init_scalar_field(data.astype(np.float32))


def _multifield_state(vg, arrays):
    fields = [vg.init_scalar_field(arr.astype(np.float32)) for arr in arrays]
    return vg.concatenate(fields, 0)


@pytest.mark.skipif(not diffrax_available, reason="diffrax not installed")
@pytest.mark.parametrize(
    "stepper_cls",
    [ForwardEuler, RungeKutta4, PseudoSpectralIMEX, ExponentialEuler, RKC1, RKC2],
)
def test_reaction_diffusion_sbm_supports_all_step_dt_adapters(stepper_cls):
    vf = evo.VoxelFields((4, 4, 4), domain_size=(4, 4, 4))
    vg = VoxelGridJax(vf.grid_info(), precision=vf.precision)
    problem = ReactionDiffusionSBM(vg, D=1.0)
    coords = np.indices(vf.shape, dtype=np.float32)
    y0 = 0.4 + 0.05 * np.sin(coords[0] + coords[1] + coords[2])
    sol = _solve_with_adapter(problem, _scalar_state(vg, y0), stepper_cls)

    assert sol.shape == (2, 1, 4, 4, 4)
    assert np.all(np.isfinite(np.asarray(sol)))


@pytest.mark.skipif(not diffrax_available, reason="diffrax not installed")
@pytest.mark.parametrize(
    ("problem_cls", "problem_kwargs", "state_factory"),
    [(  ReactionDiffusion,
        {"D": 1.0},
        lambda vg, shape: _scalar_state(vg,
            0.5 + 0.05 * np.sin(np.indices(shape, dtype=np.float32)[0]),
        )),
     (  ReactionDiffusionSBM,
        {"D": 1.0},
        lambda vg, shape: _scalar_state(vg,
            0.5 + 0.05 * np.cos(np.indices(shape, dtype=np.float32)[1]),
        )),
     (  CahnHilliard,
        {"eps": 3.0, "D": 1.0},
        lambda vg, shape: _scalar_state(vg,
            0.5 + 0.05 * np.sin(np.indices(shape, dtype=np.float32)[0]),
        )),
     (  TwoPhaseAllenCahn,
        {"eps": 3.0, "bc": ("periodic", "periodic", "periodic")},
        lambda vg, shape: _scalar_state(vg,
            0.5 + 0.05 * np.cos(np.indices(shape, dtype=np.float32)[0]),
        )),
     (  MultiPhaseAllenCahn,
        {"eps": 3.0},
        lambda vg, shape: _multifield_state(vg,
            (np.full(shape, 0.2, dtype=np.float32),
             np.full(shape, 0.3, dtype=np.float32),
             np.full(shape, 0.5, dtype=np.float32)),
        )),
     (  CoupledReactionDiffusion,
        {"D_A": 1.0, "D_B": 0.5},
        lambda vg, shape: _multifield_state(vg,
            (0.5 + 0.05 * np.sin(np.indices(shape, dtype=np.float32)[0]),
             0.25 + 0.05 * np.cos(np.indices(shape, dtype=np.float32)[1])),
        )),
    ],
)
def test_exponential_euler_supports_all_problem_definitions(
    problem_cls, problem_kwargs, state_factory
):
    vf = evo.VoxelFields((4, 4, 4), domain_size=(4, 4, 4))
    vg = VoxelGridJax(vf.grid_info(), precision=vf.precision)
    problem = problem_cls(vg, **problem_kwargs)
    y0 = state_factory(vg, vf.shape)
    sol = _solve_with_adapter(problem, y0, ExponentialEuler)

    assert sol.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(sol)))

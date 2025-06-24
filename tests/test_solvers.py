"""Tests for solver functionality."""

import numpy as np
import voxelsss as vox
from voxelsss.solvers import TimeDependendSolver

def test_time_solver_multiple_fields():
    vf = vox.VoxelFields((4, 4, 4))
    vf.add_field("a", np.ones(vf.shape))
    vf.add_field("b", np.zeros(vf.shape))

    def step(u, t):
        return u + 1

    solver = TimeDependendSolver(vf, ["a", "b"], backend="torch", step_fn=step, device="cpu")
    solver.solve(frames=1, max_iters=1, verbose=False, jit=False)

    assert np.allclose(vf.fields["a"], 2)
    assert np.allclose(vf.fields["b"], 1)

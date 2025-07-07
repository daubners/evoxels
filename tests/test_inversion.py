"""Test inversion model class"""

import importlib.util
import numpy as np
import pytest
import evoxels as evo
from evoxels.inversion import InversionModel
from evoxels.problem_definition import PeriodicCahnHilliard

diffrax_available = importlib.util.find_spec("diffrax") is not None

def test_train_validates_sequence_length():
    vf = evo.VoxelFields((4, 4, 4))
    model = InversionModel(vf, PeriodicCahnHilliard)
    data = {
        "ts": np.array([0.0, 1.0, 2.0]),
        "ys": np.zeros((3, 4, 4, 4), dtype=np.float32),
    }
    inds = [[0, 1, 2], [0, 1]]
    with pytest.raises(ValueError):
        model.train({"D": 1.0}, data, inds)

@pytest.mark.skipif(not diffrax_available, reason="diffrax not installed")
def test_inversion_forward_solve_constant_solution():
    import diffrax as dfx
    import jax.numpy as jnp

    vf = evo.VoxelFields((4, 4, 4))
    vf.add_field('c', np.full((4, 4, 4), 0.5, dtype=np.float32))
    model = InversionModel(vf, PeriodicCahnHilliard, {'eps': 3.0})
    saveat = dfx.SaveAt(ts=jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32))
    sol = model.forward_solve({'D': 1.0}, 'c', saveat, dt0=0.1, verbose=False)
    assert sol.shape == (3, 4, 4, 4)
    np.testing.assert_allclose(np.array(sol), 0.5, atol=1e-6)

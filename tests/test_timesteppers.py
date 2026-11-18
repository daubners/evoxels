"""Numerical contracts for the fixed-stage stabilized explicit schemes."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial.legendre import legval

from evoxels.timesteppers import RKL1, RKL2, ROCK2


STEPPERS = (RKL1, RKL2, ROCK2)
STAGE_CASES = [(RKL1, 1), (RKL2, 2), (ROCK2, 3)] + [
    (cls, 4) for cls in STEPPERS
]


def _problem(rhs):
    return SimpleNamespace(rhs=rhs)


def _rkl_amplification(cls, s, z):
    if cls is RKL1:
        return legval(1 + 2*z/(s*(s + 1)), [0]*s + [1])
    b = (s*s + s - 2) / (2*s*(s + 1))
    return 1 - b + b*legval(1 + 4*z/(s*s + s - 2), [0]*s + [1])


@pytest.mark.parametrize("cls,s", STAGE_CASES)
def test_step_contract_and_rhs_count(cls, s):
    times = []

    def rhs(t, u):
        times.append(t)
        return -u

    stepper = cls(_problem(rhs), 0.1, polygrad=s)
    u = np.array([0.25, 0.75], dtype=np.float64)
    original = u.copy()
    result = stepper.step(1.3, u)
    assert stepper.order == (1 if cls is RKL1 else 2)
    assert len(times) == s
    assert times[0] == 1.3
    np.testing.assert_array_equal(u, original)
    np.testing.assert_allclose(result, stepper.step_dt(1.3, 0.1, u), atol=0, rtol=0)
    np.testing.assert_allclose(stepper.step_dt(1.3, 0.0, u), u, atol=0, rtol=0)
    shorter = stepper.step_dt(1.3, 0.025, u)
    np.testing.assert_allclose(shorter, cls(stepper.problem, 0.025, s).step(1.3, u))
    assert not np.allclose(shorter, result)
    stepper = cls(_problem(lambda t, u: 0*u), 0.1, polygrad=s)
    np.testing.assert_array_equal(stepper.step(1.3, u), u)


@pytest.mark.parametrize("cls,s", [
    (RKL1, True), (RKL1, 4.0), (RKL1, 0),
    (RKL2, True), (RKL2, 4.0), (RKL2, 1),
    (ROCK2, True), (ROCK2, 4.0), (ROCK2, 2), (ROCK2, 23), (ROCK2, 201),
])
def test_invalid_stage_counts(cls, s):
    with pytest.raises(ValueError, match="polygrad"):
        cls(_problem(lambda t, u: -u), 0.1, polygrad=s)


@pytest.mark.parametrize("cls", STEPPERS)
def test_global_convergence_order(cls):
    t0, duration = 0.3, 0.5
    problem = _problem(lambda t, u: -(1 + t)*u*u)
    exact = 1 / (1 + duration + ((t0 + duration)**2 - t0*t0)/2)
    errors = []
    for n in (20, 40, 80):
        dt = duration / n
        stepper = cls(problem, dt)
        u = 1.0
        for k in range(n):
            u = stepper.step(t0 + k*dt, u)
        errors.append(abs(u - exact))
    expected = 2**stepper.order
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], expected, rtol=0.12)


@pytest.mark.parametrize("cls", (RKL1, RKL2))
def test_legendre_amplification_and_stability(cls):
    s = 9
    extent = s*(s + 1) if cls is RKL1 else (s*s + s - 2)/2
    z = np.linspace(-extent, 0, 257)
    actual = cls(_problem(lambda t, u: z*u), 1.0, s).step(0, np.ones_like(z))
    np.testing.assert_allclose(actual, _rkl_amplification(cls, s, z), atol=2e-12, rtol=2e-12)
    assert np.max(np.abs(actual)) <= 1 + 2e-12
    outside = cls(_problem(lambda t, u: -1.2*extent*u), 1.0, s).step(0, 1.0)
    assert abs(outside) > 1


# Computed with 70-digit Decimal arithmetic directly from the pinned SciML
# source table, evaluating its recurrence polynomial times the finishing factor
# 1 + 2*sigma*z + sigma*(sigma+tau)*z**2 (independent of step_dt).
@pytest.mark.parametrize("s,z,expected", [
    (3, -1.0, 0.436695915684274728269202216712239),
    (4, -2.0, 0.425716868390266262212440388628723),
    (35, -400.0, -0.398434350547062591810499415895878),
    (200, -16000.0, -0.732669378761272856478466850257915),
])
def test_rock2_amplification_and_stability(s, z, expected):
    actual = ROCK2(_problem(lambda t, u: z*u), 1.0, s).step(0, 1.0)
    assert actual == pytest.approx(expected, abs=2e-11, rel=2e-11)
    # Conservative bound used by the reference implementation's stage selector.
    extent = 0.811*(s - 1)**2 - 1.5
    z = np.linspace(-extent, 0, 513)
    actual = ROCK2(_problem(lambda t, u: z*u), 1.0, s).step(0, np.ones_like(z))
    assert np.max(np.abs(actual)) <= 1 + 2e-11


@pytest.mark.parametrize("cls", STEPPERS)
def test_torch_values_and_gradients(cls):
    torch = pytest.importorskip("torch")
    u = torch.tensor([0.25, 0.75], dtype=torch.float32, requires_grad=True)
    stepper = cls(_problem(lambda t, u: -20*u), 0.1)
    actual = stepper.step(0.3, u)
    expected = (0.4257168683902662622 if cls is ROCK2
                else _rkl_amplification(cls, 4, -2.0))
    np.testing.assert_allclose(actual.detach().numpy(), u.detach().numpy()*expected, atol=2e-6)
    actual.sum().backward()
    np.testing.assert_allclose(u.grad.numpy(), expected, atol=2e-6)
    assert actual.dtype == u.dtype and actual.device == u.device


@pytest.mark.parametrize("cls", STEPPERS)
def test_torch_compilation(cls):
    torch = pytest.importorskip("torch")
    stepper = cls(_problem(lambda t, u: -u*u + t), 0.1)
    u = torch.tensor([0.25, 0.75])
    compiled = torch.compile(stepper.step_dt, fullgraph=True)
    for dt in (0.1, 0.025, 0.0):
        torch.testing.assert_close(compiled(0.3, dt, u), stepper.step_dt(0.3, dt, u))


@pytest.mark.parametrize("cls", STEPPERS)
def test_jax_values_gradients_and_traced_dt(cls):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    stepper = cls(_problem(lambda t, u: -20*u), 0.1)
    u = jnp.array([0.25, 0.75], dtype=jnp.float32)
    compiled = jax.jit(stepper.step_dt)
    for dt in (0.1, 0.025, 0.0):
        expected = (stepper.step_dt(0.3, dt, 1.0) if cls is ROCK2
                    else _rkl_amplification(cls, 4, -20*dt))
        actual = compiled(jnp.asarray(0.3), jnp.asarray(dt), u)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(u)*expected, atol=2e-6)
        grad = jax.grad(lambda y: jnp.sum(compiled(0.3, jnp.asarray(dt), y)))(u)
        np.testing.assert_allclose(np.asarray(grad), expected, atol=2e-6)
        assert actual.dtype == u.dtype and actual.device == u.device

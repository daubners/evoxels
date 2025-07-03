import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    class DummyJax:
        @staticmethod
        def jit(f):
            return f
    class DummyJnp:
        @staticmethod
        def ones_like(x):
            return np.ones_like(x)
        @staticmethod
        def exp(x):
            return np.exp(x)

    jax = DummyJax()
    jnp = DummyJnp()

import dataclasses

@dataclasses.dataclass
class DiffusionLegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        self.leg_poly = ExpLegendrePolynomials(self.max_degree)

    def __call__(self, params, inputs):
        return self.leg_poly(params, 2.0 * inputs - 1.0)


@dataclasses.dataclass
class ChemicalPotentialLegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        self.leg_poly = LegendrePolynomialRecurrence(self.max_degree)

    def __call__(self, params, inputs):
        return self.leg_poly(params, 2.0 * inputs - 1.0)


@dataclasses.dataclass
class ExpLegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        leg_poly = LegendrePolynomialRecurrence(self.max_degree)
        self.func = jax.jit(lambda p, x: jnp.exp(leg_poly(p, x)))

    def __call__(self, params, inputs):
        return self.func(params, inputs)

# TODO: This can be made more efficient
@dataclasses.dataclass
class LegendrePolynomialRecurrence:
    max_degree: int

    def __post_init__(self):
        # Create a JIT-compiled function that computes the Legendre polynomial sum
        def compute_polynomial_sum(params, x):
            result = params[0] * self.T0(x)
            for i in range(1, self.max_degree + 1):
                result += params[i] * self._compute_legendre(i, x)
            return result

        self.func = jax.jit(compute_polynomial_sum)

    def __call__(self, params, inputs):
        return self.func(params, inputs)

    def T0(self, x):
        return 1.0 * jnp.ones_like(x)

    def _compute_legendre(self, n, x):
        """Compute the nth Legendre polynomial using the three-term recurrence relation."""
        if n == 0:
            return self.T0(x)
        elif n == 1:
            return x

        # Initialize P₀ and P₁
        p_prev = self.T0(x)  # P₀
        p_curr = x  # P₁

        # Compute Pₙ using the recurrence relation
        for i in range(1, n):
            p_next = ((2 * i + 1) * x * p_curr - i * p_prev) / (i + 1)
            p_prev = p_curr
            p_curr = p_next

        return p_curr
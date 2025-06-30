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


@dataclasses.dataclass
class LegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        if self.max_degree == 0:
            self.func = jax.jit(lambda p, x: p[0] * self.T0(x))
        elif self.max_degree == 1:
            self.func = jax.jit(lambda p, x: p[0] * self.T0(x) + p[1] * self.T1(x))
        elif self.max_degree == 2:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x) + p[1] * self.T1(x) + p[2] * self.T2(x)
            )
        elif self.max_degree == 3:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
            )
        elif self.max_degree == 4:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
            )
        elif self.max_degree == 5:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
            )
        elif self.max_degree == 6:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
            )
        elif self.max_degree == 7:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
            )
        elif self.max_degree == 8:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
            )
        elif self.max_degree == 9:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
                + p[9] * self.T9(x)
            )
        elif self.max_degree == 10:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
                + p[9] * self.T9(x)
                + p[10] * self.T10(x)
            )

    def __call__(self, params, inputs):
        return self.func(params, inputs)

    def T0(self, x):
        return 1.0 * jnp.ones_like(x)

    def T1(self, x):
        return x

    def T2(self, x):
        return 0.5 * (3 * x**2 - 1.0)

    def T3(self, x):
        return 0.5 * (5 * x**3 - 3 * x)

    def T4(self, x):
        return 0.125 * (35 * x**4 - 30 * x**2 + 3)

    def T5(self, x):
        return 0.125 * (63 * x**5 - 70 * x**3 + 15 * x)

    def T6(self, x):
        return 0.0625 * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)

    def T7(self, x):
        return 0.0625 * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)

    def T8(self, x):
        return 0.0078125 * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)

    def T9(self, x):
        return 0.0078125 * (
            12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x
        )

    def T10(self, x):
        return 0.00390625 * (
            46189 * x**10
            - 109395 * x**8
            + 90090 * x**6
            - 30030 * x**4
            + 3465 * x**2
            - 63
        )


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
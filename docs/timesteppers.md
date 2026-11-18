# Appendix: Timestppers

All schemes live in `evoxels.timesteppers`. Construct a timestepper with an ODE
problem exposing `rhs(t, u)` and a timestep `dt`. `step(t, u)` returns the state
at `t + dt`; `step_dt(t, dt, u)` uses the supplied increment instead. Neither
method advances a stored clock. The `order` property gives temporal accuracy,
independently of the spatial discretization's order.

| Class | Order | Work per step | Main use |
| --- | ---: | --- | --- |
| `ForwardEuler` | 1 | 1 RHS evaluation | Simple explicit baseline |
| `RungeKutta4` | 4 | 4 RHS evaluations | Nonstiff problems |
| `PseudoSpectralIMEX` | 1 | 1 RHS evaluation and a forward/inverse transform pair | Semilinear problems with a stiff spectral linear part |
| `ExponentialEuler` | 1 | 1 RHS evaluation and a forward/inverse transform pair | Exponential integration of the spectral linear part |
| `RKC1` | 1 | `polygrad` RHS evaluations | Explicit stabilized Chebyshev integration |
| `RKC2` | 2 | `polygrad` RHS evaluations | Explicit stabilized Chebyshev integration |
| `ROCK2` | 2 | `polygrad` RHS evaluations | Orthogonal Chebyshev integration with a two-stage finish |
| `RKL1` | 1 | `polygrad` RHS evaluations | Explicit Legendre super-time-stepping |
| `RKL2` | 2 | `polygrad` RHS evaluations | Explicit Legendre super-time-stepping |

The state arithmetic supports Torch tensors and JAX arrays. Existing spatial
operators and Fourier schemes use the problem's voxel grid for backend, device,
precision, and boundary handling. The new RKL and ROCK classes cache scalar
coefficients and use only state arithmetic and `rhs`, without transferring the
state between devices. They do not update the supplied state in place.

## Usage

```python
import numpy as np
from functools import partial

from evoxels import VoxelFields
from evoxels.pdes import ReactionDiffusion
from evoxels.solvers import TimeDependentSolver
from evoxels.timesteppers import RKL1, RKL2, ROCK2
from evoxels.voxelgrid import VoxelGridTorch

vf = VoxelFields((16, 16, 16))
vf.add_field("c", np.full(vf.shape, 0.5, dtype=np.float32))
vg = VoxelGridTorch(vf.grid_info(), precision=vf.precision, device="cpu")
problem = ReactionDiffusion(vg, D=1.0)
u = vg.init_scalar_field(vf.fields["c"])

stepper = RKL2(problem, dt=0.1, polygrad=8)
u = stepper.step(0.0, u)
u = stepper.step_dt(0.1, 0.025, u)

# The solver constructs its timestepper with (problem, time_increment).
solver = TimeDependentSolver(
    vf, "c", backend="torch", device="cpu",
    problem_cls=ReactionDiffusion,
    timestepper_cls=partial(ROCK2, polygrad=8),
)
solver.solve(
    time_increment=0.1, frames=1, max_iters=10,
    problem_kwargs={"D": 1.0}, jit=False, verbose=False,
)
```

The constant field makes this a minimal usage example, rather than an accuracy
or performance benchmark. Select a timestep suitable for your actual problem.

## Euler and classical Runge–Kutta

`ForwardEuler` uses $u_{n+1}=u_n+hF(t_n,u_n)$.
`RungeKutta4` uses the classical four-stage formula:

$$
\begin{aligned}
k_1 &= F(t,u), \\
k_2 &= F(t+h/2,u+hk_1/2), \\
k_3 &= F(t+h/2,u+hk_2/2), \\
k_4 &= F(t+h,u+hk_3), \\
u_{n+1} &= u+h(k_1+2k_2+2k_3+k_4)/6.
\end{aligned}
$$

These explicit schemes require the usual timestep stability restrictions for
the discretized problem.

## Fourier IMEX and exponential Euler

Both Fourier schemes require a `SemiLinearODE` with a `fourier_symbol` and
compatible FFT boundary configuration. Writing $L$ for this spectral linear
operator, their implemented updates are

$$
u_{n+1}=u_n+h(I-hL)^{-1}F(t_n,u_n)
\qquad\text{(`PseudoSpectralIMEX`)},
$$

$$
u_{n+1}=u_n+h\varphi_1(hL)F(t_n,u_n),\qquad
\varphi_1(z)=\frac{e^z-1}{z}
\qquad\text{(`ExponentialEuler`)}.
$$

The complete RHS is evaluated in physical space, padded for the configured
boundary conditions, transformed, multiplied by the spectral factor, transformed
back, and cropped. Fixed-step spectral factors are cached; `step_dt` recomputes
them for the supplied increment. Exponential Euler evaluates $\varphi_1$ near
zero using the existing degree-six Padé approximation to avoid cancellation.
The remaining nonlinear terms still constrain stability and accuracy.

See [Zhu et al. (1999)](https://doi.org/10.1103/PhysRevE.60.3564) for the
semi-implicit Fourier spectral method, and
[Hochbruck, Lubich, and Selhofer (1998)](https://doi.org/10.1137/S1064827595295337)
for exponential integration.

## Stabilized explicit schemes

RKC, ROCK, and RKL methods perform several explicit stages within a single
step of length $h$. Their stability intervals along the negative real axis grow
quadratically with the stage count $s$, while RHS work grows linearly. This is
useful for diffusion-dominated problems. It does not make the methods implicit
or unconditionally stable.

All five classes default to `polygrad=4`. Stage count stays fixed when calling
`step_dt` with another increment. Choose a new timestepper if a different stage
count is required; do not mutate cached stage configuration. No spectral-radius
estimate or automatic stage selection is performed. Accuracy can require a much
smaller timestep than stability permits, especially for large stage counts.

### RKC1 and RKC2

The existing RKC implementations follow
[Verwer, Hundsdorfer, and Sommeijer (1990)](https://doi.org/10.1007/BF01386405).
They cache Chebyshev values and derivatives at
$w_0=1+\mathrm{damping}/s^2$ using the problem's backend.
`RKC1` defaults to `damping=0.05`, and `RKC2` to `damping=2/13`.
Use at least one stage for RKC1 and three for the current RKC2 implementation;
the existing constructors do not explicitly validate these limits. Positive
damping is required by their current derivative formulas.

For RKC1, $b_j=1/T_j(w_0)$ and $w_1=T_s(w_0)/T_s'(w_0)$.
For RKC2, $b_j=T_j''(w_0)/(T_j'(w_0))^2$ for $j\geq2$,
$b_0=b_1=b_2$, and $w_1=T_s'(w_0)/T_s''(w_0)$.
Each scheme advances a three-term recurrence. RKC2 also retains the initial
RHS for its second-order correction. The existing implementations and defaults
are preserved.

### RKL1 and RKL2

The Legendre schemes follow
[Meyer, Balsara, and Aslam (2014)](https://doi.org/10.1016/j.jcp.2013.08.021).
The RKL2 coefficients are also given explicitly in
[Vaidya et al. (2017), Appendix A](https://arxiv.org/html/1702.05487#A1).
There is no configurable damping parameter.

Let $P_j$ denote the ordinary Legendre polynomial, $z=h\lambda$, and $s$ the
stage count. The final stability polynomials are

$$
R_s(z)=P_s(1+w_1z),\qquad w_1=\frac{2}{s(s+1)}
\qquad\text{(RKL1)},
$$

$$
R_s(z)=1-b_s+b_sP_s(1+w_1z),\quad
b_s=\frac{s^2+s-2}{2s(s+1)},\quad
w_1=\frac{4}{s^2+s-2}
\qquad\text{(RKL2)}.
$$

Both start from $Y_0=u_n$ and $Y_1=u_n+\widetilde\mu_1 hF(t_n,u_n)$,
then use

$$
\begin{aligned}
Y_j={}&\mu_jY_{j-1}+\nu_jY_{j-2}+(1-\mu_j-\nu_j)u_n \\
 &+\widetilde\mu_jhF(t_n+c_{j-1}h,Y_{j-1})
  +\widetilde\gamma_jhF(t_n,u_n),\qquad j=2,\ldots,s.
\end{aligned}
$$

For RKL1, $b_j=1$, $\widetilde\mu_1=w_1$, and
$\widetilde\gamma_j=0$. For RKL2, use $b_0=b_1=1/3$ and the expression above
for $b_j$ when $j\geq2$, with $\widetilde\mu_1=b_1w_1$. The other
coefficients are

$$
\mu_j=\frac{2j-1}{j}\frac{b_j}{b_{j-1}},\quad
\nu_j=-\frac{j-1}{j}\frac{b_j}{b_{j-2}},\quad
\widetilde\mu_j=\mu_jw_1,\quad
\widetilde\gamma_j=-(1-b_{j-1})\widetilde\mu_j.
$$

Stage times are $c_1=\widetilde\mu_1$ and
$c_j=b_jj(j+1)w_1/2$ for $j\geq2$. These times matter for a time-dependent
RHS. The implementation caches the coefficients as Python scalars and uses
differences from $u_n$ in the state recurrence to preserve zero-RHS states.

`RKL1` accepts integer stage counts $s\geq1$ (one stage is Forward Euler).
`RKL2` accepts $s\geq2$. Booleans, nonintegers, and smaller values raise
`ValueError`. Both even and odd stage counts are supported.

For real nonpositive eigenvalues with largest magnitude $\rho$, sufficient
linear stability conditions are

$$
h\rho\leq s(s+1)\quad\text{(RKL1)},\qquad
h\rho\leq\frac{s^2+s-2}{2}\quad\text{(RKL2)}.
$$

Equivalently, relative to the Forward Euler limit $h_{\rm FE}=2/\rho$, the
allowed factors are $s(s+1)/2$ and $(s^2+s-2)/4$. These are scalar/normal linear
operator bounds, not guarantees for arbitrary nonlinear or nonnormal systems.

### ROCK2

ROCK2 implements the orthogonal-polynomial method of
[Abdulle and Medovikov (2001)](https://doi.org/10.1007/s002110100292).
It uses the reference coefficient tables rather than RKC2 coefficients.
The data and complete MIT attribution are included in
`evoxels/_rock2_coefficients.py`, pinned to OrdinaryDiffEq.jl revision
`23cd1cd81406ed59555312893136e85922eeba17`:
[tables](https://github.com/SciML/OrdinaryDiffEq.jl/blob/23cd1cd81406ed59555312893136e85922eeba17/lib/OrdinaryDiffEqStabilizedRK/src/rkc_tableaus_rock2.jl),
[reference algorithm](https://github.com/SciML/OrdinaryDiffEq.jl/blob/23cd1cd81406ed59555312893136e85922eeba17/lib/OrdinaryDiffEqStabilizedRK/src/rkc_perform_step.jl).

Here `polygrad` counts the total number of stages, including the two-stage
finish. The recurrence degree is $m=s-2$. Supported total counts are:

```text
3–22 (every integer),
24, 26, 28, 30, 32, 35, 38, 41, 45, 49, 53, 58, 63, 68, 74, 80,
87, 95, 104, 114, 125, 137, 150, 165, 182, 200.
```

Unsupported values, booleans, and nonintegers raise `ValueError` with the
supported counts. Values are never silently rounded.

Starting with $Y_0=u_n$ and $Y_1=u_n+\mu_1hF(t_n,u_n)$, the recurrence is

$$
Y_j=Y_{j-1}+\kappa_j(Y_{j-1}-Y_{j-2})
    +\mu_jhF(t_n+c_{j-1}h,Y_{j-1}),\qquad j=2,\ldots,m,
$$

where $c_0=0$, $c_1=\mu_1$, and
$c_j=c_{j-1}+\kappa_j(c_{j-1}-c_{j-2})+\mu_j$.
The finishing coefficients $\sigma,\tau$ give

$$
\begin{aligned}
f_0 &= F(t_n+c_mh,Y_m), \\
V &= Y_m+\sigma h f_0, \\
f_1 &= F(t_n+(c_m+\sigma)h,V), \\
u_{n+1} &= V+\sigma h f_1+\tau h(f_1-f_0).
\end{aligned}
$$

A conservative negative-real bound derived from the reference stage selector is
$h\rho\leq0.811(s-1)^2-1.5$. This deliberately lies inside the reference
family's stability extent and does not describe its full complex stability
region. Coefficients and stage times are independent of $h$ and cached at
initialization. A step performs exactly $s$ RHS evaluations without the extra
endpoint RHS used by the reference solver's interpolation/error machinery.

## Differentiation, compilation, and limitations

The RKL/ROCK stage counts are static during Torch compilation or JAX tracing.
`step_dt` can receive traced time and timestep values. A zero timestep returns
the original state numerically for finite RHS evaluations; the stage evaluations
still occur. Changing `dt` does not rebuild coefficients.

The existing `DiffraxTimeStepperAdapter` calls `step_dt` and reports the scheme's
order. Its existing error estimate is unchanged; adapter compatibility alone
does not validate adaptive error control for these schemes. The adapter also
evaluates an initial RHS in addition to the stepper's own stage evaluations.

These methods do not add clipping, positivity enforcement, or per-stage
projections. Problem-specific `post_step` handling remains the outer solver's
responsibility. Monotonicity results have specific spatial/boundary assumptions;
see [Tan, Aslam, and Bertozzi (2025)](https://arxiv.org/html/2507.17062v2).
No general positivity guarantee is made for phase-field, anisotropic diffusion,
or other arbitrary nonlinear problems. Time-dependent boundaries and nonlinear
stiffness require problem-specific accuracy checks.

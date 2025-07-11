from functools import partial
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Type, Optional
from evoxels.timesteppers import PseudoSpectralIMEX_dfx

try:
    import diffrax as dfx
    import optimistix as optx
    import jax.numpy as jnp
    import jax
except ImportError:
    dfx = None
    optx = None
    jnp = None
    jax = None

DIFFRAX_AVAILABLE = dfx is not None

@dataclass
class InversionModel:
    """Inverse modeling using JAX and diffrax.

    This small helper class wraps the differentiable solver implementation and
    provides utilities to fit material parameters via gradient based
    optimization.  It is intentionally lightweight so that new users can easily
    follow the individual steps: solving the PDE, computing residuals and
    running a least squares optimiser.
    """
    vf: Any  # VoxelFields object
    problem_cls: Type
    pos_params: Optional[list[str]] = None
    problem_kwargs: Optional[dict[str, Any]] = None
    backend: str = 'jax'

    def __post_init__(self):
        """Initialize backend specific components."""
        self.problem_kwargs = self.problem_kwargs or {}
        if self.backend == 'jax':
            from evoxels.voxelgrid import VoxelGridJax
            from .profiler import JAXMemoryProfiler
            self.vg = VoxelGridJax(self.vf.grid_info(), precision=self.vf.precision)
            self.profiler = JAXMemoryProfiler()

            if not DIFFRAX_AVAILABLE:
                raise ImportError(
                    "CahnHilliardInversionModel requires the optional JAX"
                    " dependencies (jax, diffrax)."
                )
            
    def solve(self, parameters, y0, saveat, adjoint=dfx.ForwardMode(), dt0=0.1):
        """Integrate the Cahn--Hilliard equation for a given parameter set.

        Args:
            parameters (dict): Dictionary containing the material parameters to
                solve with.
            y0 (array-like): Initial concentration field.
            saveat (:class:`diffrax.SaveAt`): Time points at which the solution
                should be stored.
            adjoint: Differentiation mode used by :func:`diffrax.diffeqsolve`.
            dt0 (float): Initial step size for the time integrator.

        Returns:
            jax.Array: Array of saved concentration fields with shape
            ``(len(saveat.ts), Nx, Ny, Nz)``.
        """
        u = self.vg.init_scalar_field(y0)
        u = self.vg.bc.trim_boundary_nodes(u)
        if self.pos_params:
            parameters = {k: jnp.exp(v) if k in self.pos_params else v for k, v in parameters.items()}
        problem = self.problem_cls(self.vg, **self.problem_kwargs, **parameters)
        solver = PseudoSpectralIMEX_dfx(problem.fourier_symbol)

        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: problem.rhs(y, t)),
            solver,
            t0=saveat.subs.ts[0],
            t1=saveat.subs.ts[-1],
            dt0=dt0,
            y0=u,
            saveat=saveat,
            max_steps=100000,
            throw=False,
            adjoint=adjoint,
        )
        padded = problem.pad_bc(solution.ys[:, 0])
        out = self.vg.bc.trim_ghost_nodes(padded)
        return out
    
    def forward_solve(self, parameters, fieldname, saveat, dt0=0.1, verbose=True):
        start = timer()
        u0 = self.vf.fields[fieldname]
        if self.pos_params:
            parameters = {k: jnp.log(v) if k in self.pos_params else v for k, v in parameters.items()}
        sol = self.solve(parameters, u0, saveat, dt0=dt0)
        end = timer()

        self.vf.fields[fieldname] = self.vg.to_numpy(sol[-1])
        if verbose:
            iterations = int(saveat.subs.ts[-1] // dt0)
            self.profiler.print_memory_stats(start, end, iterations)

        return sol
    
    def residuals(self, parameters, y0s__values__saveat, adjoint=dfx.ForwardMode()):
        """Calculate residuals between measured and simulated states.

        Args:
            parameters (dict): Current estimate of the model parameters.
            y0s__values__saveat (tuple): Tuple ``(y0s, values, saveat)`` where
                ``y0s`` contains the initial states for each sequence, ``values``
                contains the observed states and ``saveat`` specifies the time
                points of these observations.
            adjoint: Differentiation mode for :func:`solve`.

        Returns:
            jax.Array: Array of residuals with shape matching ``values``.
        """
        y0s, values, saveat = y0s__values__saveat
        solve_ = partial(self.solve, adjoint=adjoint)
        batch_solve = jax.vmap(solve_, in_axes=(None, 0, None))
        pred_values = batch_solve(parameters, y0s, saveat)
        residuals = values - pred_values[:, 1:]
        return residuals

    def train(
        self,
        initial_parameters,
        data,
        inds,
        adjoint=dfx.ForwardMode(),
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
        max_steps=1000,
    ):
        """Fit ``parameters`` so that the model matches observed data.

        This method assembles the observed sequences into a format suitable for
        :func:`optimistix.least_squares` and then runs a Levenberg--Marquardt
        optimisation to minimise the residuals returned by :func:`residuals`.

        Args:
            initial_parameters (dict): Initial guess for the parameters to be
                optimised.
            data (dict): Dictionary containing ``"ts"`` (time stamps) and
                ``"ys"`` (concentration fields) as produced by :func:`solve`.
            inds (list[list[int]]): For each sequence, the indices in ``data``
                that should be used for training. All sequences must have the
                same spacing.
            adjoint: Differentiation mode used when evaluating the residuals.
            rtol, atol (float): Tolerances for the optimiser.
            verbose (bool): If ``True``, prints optimisation progress.
            max_steps (int): Maximum number of optimisation steps.

        Returns:
            optimistix.State: The optimiser state after termination.
        """
        # Get length of first sequence to use as reference
        ref_len = len(inds[0])
        if ref_len < 2:
            raise ValueError("Each sequence in inds must have at least 2 elements")

        # Get reference spacing from first sequence
        ref_spacing = [inds[0][i + 1] - inds[0][i] for i in range(ref_len - 1)]

        # Validate all other sequences
        for i, sequence in enumerate(inds):
            if len(sequence) != ref_len:
                raise ValueError(
                    f"Sequence {i} has different length than first sequence"
                )

            # Check spacing
            spacing = [sequence[j + 1] - sequence[j] for j in range(len(sequence) - 1)]
            if spacing != ref_spacing:
                raise ValueError(
                    f"Sequence {i} has different spacing than first sequence"
                )

        # TODO: make data a voxelgrid or voxelfield object
        y0s = jnp.array([data["ys"][ind[0]] for ind in inds])
        values = jnp.array(
            [
                jnp.array([data["ys"][ind[i]] for i in range(1, len(ind))])
                for ind in inds
            ]
        )
        saveat = dfx.SaveAt(
            ts=jnp.array(
                [0.0]
                + [
                    data["ts"][inds[0][i]] - data["ts"][inds[0][0]]
                    for i in range(1, len(inds[0]))
                ]
            )
        )

        args = (y0s, values, saveat)
        residuals_ = partial(self.residuals, adjoint=adjoint)

        if self.pos_params:
            # Ensure parameters are positive and take log
            for key in self.pos_params:
                if initial_parameters[key] <= 0:
                    raise ValueError(f"Parameter {key} must be positive")
                initial_parameters[key] = jnp.log(initial_parameters[key])

        solver = optx.LevenbergMarquardt(
            rtol=rtol,
            atol=atol,
            verbose=frozenset(
                {"step", "accepted", "loss", "step_size"} if verbose else None
            ),
        )

        sol = optx.least_squares(
            residuals_,
            solver,
            initial_parameters,
            args=args,
            max_steps=max_steps,
            throw=False,
        )

        res = sol.value

        if self.pos_params:
            # Ensure parameters are positive and take exp
            for key in self.pos_params:
                res[key] = jnp.exp(res[key])

        return res

try:
    import diffrax as dfx
    import equinox as eqx
    import optimistix as optx
    import jax.numpy as jnp
    import jax
except Exception:
    dfx = None
    eqx = None
    optx = None
    jnp = None
    jax = None
JAX_AVAILABLE = jax is not None

import numpy as np
from functools import partial
from .voxelfields import VoxelFields
from .timesteppers import SemiImplicitFourierSpectral
from .problem_definition import PeriodicCahnHilliard
from .voxelgrid import VoxelGridJax


class CahnHilliardInversionModel:
    def __init__(
        self,
        Nx=128,
        Ny=128,
        Nz=128,
        Lx=0.01,
        Ly=0.01,
        Lz=0.01,
        eps=3.0,
        A=0.25,
    ):
        if not JAX_AVAILABLE:
            raise ImportError(
                "CahnHilliardInversionModel requires the optional JAX"
                " dependencies (jax, diffrax, equinox, optimistix)."
            )
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.eps, self.A = eps, A

    def solve(self, parameters, y0, saveat, adjoint=dfx.ForwardMode(), dt0=0.1):
        vf = VoxelFields(
            self.Nx, self.Ny, self.Nz, domain_size=(self.Lx, self.Ly, self.Lz)
        )
        vg = VoxelGridJax(vf.grid_info())
        u = vg.init_field_from_backend(jnp.array(y0))
        problem = PeriodicCahnHilliard(vg, self.eps, parameters["D"], self.A)

        solver = SemiImplicitFourierSpectral(
            spectral_factor=problem.spectral_factor,
            rfftn=vg.rfftn,
            irfftn=vg.irfftn,
        )

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
        return solution.ys[:,0]

    def residuals(self, parameters, y0s__values__saveat, adjoint=dfx.ForwardMode()):
        y0s, values, saveat = y0s__values__saveat
        solve_ = partial(self.solve, adjoint=adjoint)
        batch_solve = jax.vmap(
            solve_, in_axes=(None, 0, None)
        )
        pred_values = batch_solve(parameters, y0s, saveat)
        residuals = values - pred_values[:, 1:]
        return residuals, jax.tree.map(lambda x: 10.0 * x, parameters)

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
        
        # Get length of first sequence to use as reference
        ref_len = len(inds[0])
        if ref_len < 2:
            raise ValueError("Each sequence in inds must have at least 2 elements")
            
        # Get reference spacing from first sequence
        ref_spacing = [inds[0][i+1] - inds[0][i] for i in range(ref_len-1)]
        
        # Validate all other sequences
        for i, sequence in enumerate(inds):
            if len(sequence) != ref_len:
                raise ValueError(f"Sequence {i} has different length than first sequence")
            
            # Check spacing
            spacing = [sequence[j+1] - sequence[j] for j in range(len(sequence)-1)]
            if spacing != ref_spacing:
                raise ValueError(f"Sequence {i} has different spacing than first sequence")

        # TODO: make data a voxelgrid or voxelfield object
        y0s = jnp.array([data["ys"][ind[0]] for ind in inds])
        values = jnp.array([jnp.array([data["ys"][ind[i]] for i in range(1, len(ind))]) for ind in inds])
        saveat = dfx.SaveAt(ts=jnp.array([0.0] + [data["ts"][inds[0][i]] - data["ts"][inds[0][0]] for i in range(1, len(inds[0]))]))

        args = (y0s, values, saveat)
        residuals_ = partial(self.residuals, adjoint=adjoint)

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

        return sol

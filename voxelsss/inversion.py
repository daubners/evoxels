import diffrax as dfx
import jax
import jax.numpy as jnp
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
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.eps, self.A = eps, A

    def solve(self, parameters, y0, saveat, adjoint=dfx.ForwardMode(), dt0=0.1):

        vf = VoxelFields(self.Nx, self.Ny, self.Nz, domain_size=(self.Lx, self.Ly, self.Lz))
        vf.add_field("c", y0)
        vg = VoxelGridJax(vf.grid_info())
        u = vg.init_field_from_numpy(vf.fields["c"])
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
        return solution.ys

    # def residuals(self, parameters, y0s__values__saveat, adjoint=dfx.ForwardMode()):
    #     y0s, values, saveat = y0s__values__saveat
    #     solve_ = partial(self.solve, adjoint=adjoint)
    #     batch_solve = eqx.filter_vmap(solve_, in_axes=(None, 0, None))
    #     pred_values = batch_solve(parameters, y0s, saveat)
    #     return values - pred_values[:, -1]

    # def residuals_full_trajectory(
    #     self, parameters, y0__values__saveat, adjoint=dfx.ForwardMode()
    # ):
    #     y0, values, saveat = y0__values__saveat
    #     pred_values = self.solve(parameters, y0, saveat, adjoint=adjoint)
    #     print(pred_values.shape)
    #     print(values.shape)
    #     return values - pred_values

    # def loss(
    #     self,
    #     parameters,
    #     y0s__values__saveat__lambda__weights,
    #     adjoint=dfx.ForwardMode(),
    # ):
    #     y0s, values, saveat, lambda_reg, weights = y0s__values__saveat__lambda__weights
    #     return jnp.sum(
    #         self.residuals(parameters, (y0s, values, saveat), adjoint) ** 2
    #     ) + l2_regularization(parameters, lambda_reg, weights)

    # def loss_full_trajectory(
    #     self,
    #     parameters,
    #     y0__values__saveat__lambda__weights,
    #     adjoint=dfx.ForwardMode(),
    # ):
    #     y0, values, saveat, lambda_reg, weights = y0__values__saveat__lambda__weights
    #     return jnp.sum(
    #         self.residuals_full_trajectory(parameters, (y0, values, saveat), adjoint)
    #         ** 2
    #     ) + l2_regularization(parameters, lambda_reg, weights)

    # def train(self, config_path, data_input, inds, use_full_trajectory=False):
    #     config = load_config(config_path)
    #     data = data_input

    #     if use_full_trajectory:
    #         # Use the first point as initial condition and all points for comparison
    #         y0 = data["ys"][inds[0]]
    #         values = data["ys"][inds]
    #         print(data["ts"][np.array(inds)] - data["ts"][inds[0]])
    #         saveat = dfx.SaveAt(ts=data["ts"][np.array(inds)] - data["ts"][inds[0]])
    #         print(saveat)
    #     else:
    #         # Original approach: split into pairs of consecutive points
    #         y0s = data["ys"][inds][:-1]
    #         values = data["ys"][inds][1:]
    #         saveat = dfx.SaveAt(
    #             ts=jnp.array([0.0, data["ts"][inds[1]] - data["ts"][inds[0]]])
    #         )

    #     adjoint = (
    #         dfx.RecursiveCheckpointAdjoint()
    #         if config["adjoint"] == "reverse"
    #         else dfx.ForwardMode()
    #     )

    #     if use_full_trajectory:
    #         args = (
    #             (y0, values, saveat)
    #             if config["problem_type"] == "least_squares"
    #             else (y0, values, saveat, config["lambda_reg"], config["weights"])
    #         )
    #         residuals_ = partial(self.residuals_full_trajectory, adjoint=adjoint)
    #         loss_ = partial(self.loss_full_trajectory, adjoint=adjoint)
    #     else:
    #         args = (
    #             (y0s, values, saveat)
    #             if config["problem_type"] == "least_squares"
    #             else (y0s, values, saveat, config["lambda_reg"], config["weights"])
    #         )
    #         residuals_ = partial(self.residuals, adjoint=adjoint)
    #         loss_ = partial(self.loss, adjoint=adjoint)

    #     # Check if using optax optimizer
    #     if config.get("use_optax", False):
    #         # Use optax optimizer (e.g., LBFGS)
    #         optimizer = getattr(optax, config["optimizer"])
    #         opt_config = config.get("optax_config", {})
    #         opt = optimizer(**opt_config)

    #         # Initialize optimizer state
    #         opt_state = opt.init(self.parameters)

    #         # Define value and gradient function
    #         if config["problem_type"] == "least_squares":

    #             def value_fn(params):
    #                 residuals = residuals_(params, args)
    #                 return jnp.sum(residuals**2)

    #         else:
    #             value_fn = lambda params: loss_(params, args)

    #         value_and_grad_fn = jax.value_and_grad(value_fn)

    #         # Optimization loop
    #         params = self.parameters
    #         best_params = params
    #         best_loss = float("inf")

    #         for step in range(config.get("max_steps", 100)):
    #             value, grad = value_and_grad_fn(params)

    #             if value < best_loss:
    #                 best_loss = value
    #                 best_params = params

    #             updates, opt_state = opt.update(
    #                 grad, opt_state, params, value=value, grad=grad, value_fn=value_fn
    #             )
    #             params = optax.apply_updates(params, updates)

    #             if config.get("verbose", False) and step % 10 == 0:
    #                 print(f"Step {step}, Loss: {value:.6e}")

    #         self.parameters = best_params
    #         return {"value": best_params, "state": {"loss": best_loss}}
    #     else:
    #         # Use optimistix optimizer (original implementation)
    #         optimizer = getattr(optx, config["optimizer"])
    #         solver = optimizer(
    #             rtol=config["rtol"],
    #             atol=config["atol"],
    #             verbose=config["verbose"],
    #         )
    #         problem_type = getattr(optx, config["problem_type"])

    #         sol = problem_type(
    #             {"least_squares": residuals_, "minimise": loss_}[
    #                 config["problem_type"]
    #             ],
    #             solver,
    #             self.parameters,
    #             options=config["options"],
    #             args=args,
    #             throw=config["throw"],
    #             max_steps=config["max_steps"],
    #         )
    #         self.parameters = sol.value
    #         return sol
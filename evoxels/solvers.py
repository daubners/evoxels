from IPython.display import clear_output
from dataclasses import dataclass
from typing import Callable, Any, Type
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import sys
from .problem_definition import ODE
from .timesteppers import TimeStepper

@dataclass
class BaseSolver(ABC):
    """Generic wrapper for solving one or more fields with a time stepper."""
    vf: Any  # VoxelFields object
    fieldnames: str | list[str]
    backend: str
    problem_cls: Type[ODE] | None = None
    timestepper_cls: Type[TimeStepper] | None = None
    step_fn: Callable | None = None
    device: str='cuda'

    def __post_init__(self):
        """Initialize backend specific components."""
        if self.backend == 'torch':
            from .voxelgrid import VoxelGridTorch
            from .profiler import TorchMemoryProfiler
            grid = self.vf.grid_info()
            self.vg = VoxelGridTorch(grid, precision=self.vf.precision, device=self.device)
            self.profiler = TorchMemoryProfiler(self.vg.device)

        elif self.backend == 'jax':
            from .voxelgrid import VoxelGridJax
            from .profiler import JAXMemoryProfiler
            self.vg = VoxelGridJax(self.vf.grid_info(), precision=self.vf.precision)
            self.profiler = JAXMemoryProfiler()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        if isinstance(self.fieldnames, str):
            self.fieldnames = [self.fieldnames]
        else:
            self.fieldnames = list(self.fieldnames)

    def _init_fields(self):
        """Initialize fields in the voxel grid."""
        u_list = [self.vg.init_scalar_field(self.vf.fields[name]) for name in self.fieldnames]
        u = self.vg.concatenate(u_list, 0)
        u = self.vg.bc.trim_boundary_nodes(u)
        return u
    
    def _init_stepper(self, time_increment, problem_kwargs, jit):
        problem_kwargs = problem_kwargs or {}
        if self.step_fn is not None:
            self.problem = None
            step = self.step_fn
        else:
            if self.problem_cls is None or self.timestepper_cls is None:
                raise ValueError("Either provide step_fn or both problem_cls and timestepper_cls")
            self.problem = self.problem_cls(self.vg, **problem_kwargs)
            timestepper = self.timestepper_cls(self.problem, time_increment)
            step = timestepper.step

        # Make use of just-in-time compilation
        if jit and self.backend == 'jax':
            import jax
            step = jax.jit(step)
        elif jit and self.backend == 'torch':
            import torch
            step = torch.compile(step)

        return step
    
    @abstractmethod
    def _run_loop(self, u, step, time_increment, frames, max_iters,
                  vtk_out, verbose, plot_bounds, colormap):
        """Abstract method for running the time integration loop."""
        raise NotImplementedError("Subclasses must implement _run_loop method.")

    def solve(
        self,
        time_increment=0.1,
        frames=10,
        max_iters=100,
        problem_kwargs=None,
        jit=True,
        verbose=True,
        vtk_out=False,
        plot_bounds=None,
        colormap='viridis'
        ):
        """Run the time integration loop.

        Args:
            time_increment (float): Size of a single time step.
            frames (int): Number of output frames (for plotting, vtk, checks).
            max_iters (int): Number of time steps to compute.
            problem_kwargs (dict | None): Problem-specific input arguments.
            jit (bool): Create just-in-time compiled kernel if ``True`` 
            verbose (bool | str): If ``True`` prints memory stats, ``'plot'``
                updates an interactive plot.
            vtk_out (bool): Write VTK files for each frame if ``True``.
            plot_bounds (tuple | None): Optional value range for plots.
        """
        u = self._init_fields()
        step = self._init_stepper(time_increment, problem_kwargs, jit)

        start = timer()
        u = self._run_loop(u, step, time_increment, frames, max_iters,
                           vtk_out, verbose, plot_bounds, colormap)
        end = timer()
        self.computation_time = end - start
        if verbose:
            self.profiler.print_memory_stats(start, end, max_iters)

    def _handle_outputs(self, u, frame, time, slice_idx, vtk_out, verbose, plot_bounds, colormap):
        """Store results and optionally plot or write them to disk."""
        if getattr(self, 'problem', None) is not None:
            u_out = self.vg.bc.trim_ghost_nodes(self.problem.pad_bc(u))
        else:
            u_out = self.vg.bc.trim_ghost_nodes(self.vg.pad_zeros(u))

        for i, name in enumerate(self.fieldnames):
            self.vf.set_field(name, self.vg.export_scalar_field_to_numpy(u_out[i:i+1]))

        if verbose:
            self.profiler.update_memory_stats()

        if self.vg.lib.isnan(u_out).any():
            print(f"NaN detected in frame {frame} at time {time}. Aborting simulation.")
            sys.exit(1)

        if vtk_out:
            filename = self.problem_cls.__name__ + "_" +\
                       self.fieldnames[0] + f"_{frame:03d}.vtk"
            self.vf.export_to_vtk(filename=filename, field_names=self.fieldnames)

        if verbose == 'plot':
            clear_output(wait=True)
            self.vf.plot_slice(self.fieldnames[0], slice_idx, time=time, colormap=colormap, value_bounds=plot_bounds)

@dataclass
class TimeDependentSolver(BaseSolver):
    """Solver for time-dependent problems."""
    def _run_loop(self, u, step, time_increment, frames, max_iters,
                  vtk_out, verbose, plot_bounds, colormap):
        n_out = max_iters // frames
        frame = 0
        slice_idx = self.vf.Nz // 2

        for i in range(max_iters):
            time = i * time_increment
            if i % n_out == 0:
                self._handle_outputs(u, frame, time, slice_idx, vtk_out,
                                     verbose, plot_bounds, colormap)
                frame += 1

            u = step(time, u)
        time = max_iters * time_increment
        self._handle_outputs(u, frame, time, slice_idx, vtk_out,
                             verbose, plot_bounds, colormap)
        return u

@dataclass
class SteadyStatePseudoTimeSolver(BaseSolver):
    """Solver for steady-state problems."""
    conv_crit: float = 1e-6
    check_freq: int = 10
    
    def _run_loop(self, u, step, time_increment, frames, max_iters,
                  vtk_out, verbose, plot_bounds, colormap):
        slice_idx = self.vf.Nz // 2
        self.converged = False
        self.iter = 0

        while not self.converged and self.iter < max_iters:
            time = self.iter * time_increment
            diff = u - step(time, u)
            u = step(time, u)

            if self.iter % self.check_freq == 0:
                self.converged = self.check_convergence(diff, verbose)

        self._handle_outputs(u, 0, time, slice_idx, vtk_out,
                             verbose, plot_bounds, colormap)
        return u
    
    def check_convergence(self, diff, verbose):
        """Check for convergence based on relative change in fields."""
        converged = True
        for i, name in enumerate(self.fieldnames):
            # Check if Frobenius norm of change is below threshold
            rel_change = self.vg.lib.linalg.norm(diff[i]) / \
                         self.vg.lib.sqrt(self.vf.Nx * self.vf.Ny * self.vf.Nz)
            if rel_change > self.conv_crit:
                converged = False
            if verbose:
                print(f"Iter {self.iter}: Field '{name}' relative change: {rel_change:.2e}")

        if converged and verbose:
            print(f"Converged after {self.iter} iterations.")

        return converged

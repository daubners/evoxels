from IPython.display import clear_output
from dataclasses import dataclass
from typing import Callable, Any, Type
from timeit import default_timer as timer
import sys

@dataclass
class OneVariableTimeDependendSolver:
    """Generic wrapper for solving a single field with a time stepper."""

    vf: Any  # VoxelFields object
    fieldname: str
    backend: str
    problem_cls: Type | None = None
    timestepper_fn: Callable | None = None
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

    def solve(
        self,
        time_increment=0.1,
        frames=10,
        max_iters=100,
        problem_kwargs=None,
        jit=True,
        verbose=True,
        vtk_out=False,
        plot_bounds=None
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

        problem_kwargs = problem_kwargs or {}
        u = self.vg.init_scalar_field(self.vf.fields[self.fieldname])
        u = self.vg.trim_boundary_nodes(u)

        if self.step_fn is not None:
            step_fn = self.step_fn
            self.problem = None
        else:
            if self.problem_cls is None or self.timestepper_fn is None:
                raise ValueError("Either provide step_fn or both problem_cls and timestepper_fn")
            self.problem = self.problem_cls(self.vg, **problem_kwargs)
            step_fn = self.timestepper_fn(self.problem, time_increment)

        # Make use of just-in-time compilation
        if jit and self.backend == 'jax':
            import jax
            step_fn = jax.jit(step_fn)
        elif jit and self.backend == 'torch':
            import torch
            step_fn = torch.compile(step_fn)

        n_out = max_iters // frames
        frame = 0
        slice_idx = self.vf.Nz // 2

        start = timer()
        for i in range(max_iters):
            time = i * time_increment
            if i % n_out == 0:
                self._handle_outputs(u, frame, time, slice_idx, vtk_out, verbose, plot_bounds)
                frame += 1

            u = step_fn(u, time)

        end = timer()
        time = max_iters * time_increment
        self._handle_outputs(u, frame, time, slice_idx, vtk_out, verbose, plot_bounds)

        if verbose:
            self.profiler.print_memory_stats(start, end, max_iters)

    def _handle_outputs(self, u, frame, time, slice_idx, vtk_out, verbose, plot_bounds):
        """Store results and optionally plot or write them to disk."""
        if getattr(self, 'problem', None) is not None:
            u_out = self.vg.trim_ghost_nodes(self.problem.pad_boundary_conditions(u))
        else:
            u_out = u
        self.vf.fields[self.fieldname] = self.vg.export_scalar_field_to_numpy(u_out)

        if verbose:
            self.profiler.update_memory_stats()

        if self.vg.lib.isnan(u_out).any():
            print(f"NaN detected in frame {frame} at time {time}. Aborting simulation.")
            sys.exit(1)

        if vtk_out:
            filename = self.fieldname+f"_{frame:03d}.vtk"
            self.vf.export_to_vtk(filename=filename, field_names=[self.fieldname])

        if verbose == 'plot':
            clear_output(wait=True)
            self.vf.plot_slice(self.fieldname, slice_idx, time=time, value_bounds=plot_bounds)

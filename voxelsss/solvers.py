from IPython.display import clear_output
from dataclasses import dataclass
from typing import Callable, Any, Type
from timeit import default_timer as timer
import sys

@dataclass
class OneVariableTimeDependendSolver:
    vf: Any # VoxelFields object
    fieldname: str
    problem_cls: Type
    timestepper_fn: Callable
    backend: str
    device: str='cuda'

    def __post_init__(self):
        if self.backend == 'torch':
            from .voxelgrid import VoxelGridTorch
            from .profiler import TorchMemoryProfiler
            grid = self.vf.grid_info()
            self.vg = VoxelGridTorch(grid, device=self.device)
            self.profiler = TorchMemoryProfiler(self.vg.device)

        elif self.backend == 'jax':
            from .voxelgrid import VoxelGridJax
            from .profiler import JAXMemoryProfiler
            self.vg = VoxelGridJax(self.vf.grid_info())
            self.profiler = JAXMemoryProfiler()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def solve(
        self,
        time_increment=0.1,  # Time-step size for update
        frames=10,           # Amount of frames (plotting, vtk, checks)
        max_iters=100,       # Amount of iterations
        problem_kwargs=None, # problem-specific input
        jit=True,            # just-in-time compilation
        verbose=True,        # ='plot' for visual output
        vtk_out=False,       # save vtk each frame
        plot_bounds=None     # bounds for visual output
        ):

        problem_kwargs = problem_kwargs or {}
        problem = self.problem_cls(self.vg, **problem_kwargs)
        u = self.vg.init_field_from_numpy(self.vf.fields[self.fieldname])
        step_fn = self.timestepper_fn(problem, time_increment)
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
        self.vf.fields[self.fieldname] = self.vg.export_field_to_numpy(u)

        if verbose:
            self.profiler.update_memory_stats()

        if self.vg.lib.isnan(u).any():
            print(f"NaN detected in frame {frame} at time {time}. Aborting simulation.")
            sys.exit(1)

        if vtk_out:
            filename = self.fieldname+f"_{frame:03d}.vtk"
            self.vf.export_to_vtk(filename=filename, field_names=[self.fieldname])

        if verbose == 'plot':
            clear_output(wait=True)
            self.vf.plot_slice(self.fieldname, slice_idx, time=time, value_bounds=plot_bounds)

from dataclasses import dataclass
from typing import Callable, Any, Type
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
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

    def _export_fields(self, u_out):
        for i, name in enumerate(self.fieldnames):
            self.vf.set_field(name, self.vg.export_scalar_field_to_numpy(u_out[i:i+1]))

    def _plot_frame(self, fields, time, colormap, plot_bounds):
        """Plot a single frame of the specified field."""
        slice_idx = self.vf.Nz // 2
        slice = self.vg.to_numpy(fields[0, :, :, slice_idx])

        if not hasattr(self, "_fig"):
            self._fig, self._ax0 = plt.subplots(nrows=1, ncols=1)
            start1 = self.vf.origin[0] - self.vf.spacing[0] / 2
            start2 = self.vf.origin[1] - self.vf.spacing[1] / 2
            end1 = self.vf.domain_size[0] - start1
            end2 = self.vf.domain_size[1] - start2
            extent = [start1, end1, start2, end2]

            if plot_bounds is not None:
                self._im0 = self._ax0.imshow(slice.T, cmap=colormap, origin='lower', extent=extent,\
                                             vmin=plot_bounds[0], vmax=plot_bounds[1])
            else:
                self._im0 = self._ax0.imshow(slice.T, cmap=colormap, origin='lower', extent=extent)
            ratio = np.clip((end2 - start2) / (end1 - start1), 0, 1)
            self._cbar = self._fig.colorbar(self._im0, ax=self._ax0, shrink=ratio)
            self._ax0.set_xlabel("X")
            self._ax0.set_ylabel("Y")
        else:
            self._im0.set_data(slice.T)
            if plot_bounds is None:
                self._im0.set_clim(float(slice.min()), float(slice.max()))
                self._cbar.update_normal(self._im0)

        self._ax0.set_title(f"Slice {slice_idx} of {self.fieldnames[0]} in z at time {time}")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _handle_outputs(self, u, frame, time, vtk_out, verbose, plot_bounds, colormap):
        """Store results and optionally plot or write them to disk."""
        if getattr(self, 'problem', None) is not None:
            u_out = self.vg.bc.trim_ghost_nodes(self.problem.pad_bc(u))
        else:
            u_out = self.vg.bc.trim_ghost_nodes(self.vg.pad_zeros(u))
        self._export_fields(u_out)

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
            if not plt.isinteractive():
                plt.ion()

            self._plot_frame(u, time, colormap, plot_bounds)
            try:
                from IPython.display import display
                if not hasattr(self, "_disp"):
                    self._disp = display(self._fig, display_id=True)
                    if not hasattr(self, "_closed"):
                        plt.close(self._fig)
                        self._closed = True
                else:
                    self._disp.update(self._fig)
            except Exception:
                pass

@dataclass
class TimeDependentSolver(BaseSolver):
    """Solver for time-dependent problems."""
    def _run_loop(self, u, step, time_increment, frames, max_iters,
                  vtk_out, verbose, plot_bounds, colormap):
        n_out = max_iters // frames
        frame = 0

        for i in range(max_iters):
            time = i * time_increment
            if i % n_out == 0:
                self._handle_outputs(u, frame, time, vtk_out,
                                     verbose, plot_bounds, colormap)
                frame += 1

            u = step(time, u)
        time = max_iters * time_increment
        self._handle_outputs(u, frame, time, vtk_out,
                             verbose, plot_bounds, colormap)
        return u

@dataclass
class SteadyStatePseudoTimeSolver(BaseSolver):
    """Solver for steady-state problems."""
    conv_crit: float = 1e-6
    check_freq: int = 10
    
    def _run_loop(self, u, step, time_increment, frames, max_iters,
                  vtk_out, verbose, plot_bounds, colormap):
        self.converged = False
        self.iter = 0

        while not self.converged and self.iter < max_iters:
            time = self.iter * time_increment
            diff = u - step(time, u)
            u = step(time, u)

            if self.iter % self.check_freq == 0:
                self.converged = self.check_convergence(diff, verbose)

        self._handle_outputs(u, 0, time, vtk_out,
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

@dataclass
class MultiPhaseSolver(TimeDependentSolver):
    """
    Multiphase-specific solver wrapper.

    Two modes:
    (A) labels-mode (default):
        from_labels=True and fieldnames="labels" (or ["labels"])
        -> reads a labeled integer array, builds one-hot phase tensor (P, Nx, Ny, Nz),
           runs the solver, and exports a single label field by argmax over phases.

    (B) fields-mode:
        fieldnames = ["phi0","phi1",...]
        -> stacks them to u with shape (P, Nx, Ny, Nz) and exports back to the same fields.

    Safety:
        If P > max_phases, exits with a warning (dense arrays become RAM-critical).
    """
    from_labels: bool = True
    output_label_fieldname: str | None = None
    max_phases: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.phase_labels = None   # numpy array of unique labels (labels-mode)
        self.phase_count = None

        # Determine phase count early for memory warning
        if self.from_labels:
            if len(self.fieldnames) != 1:
                raise ValueError("labels-mode expects a single fieldname containing the labeled array.")
            lab = self.vf.fields[self.fieldnames[0]]
            uniq = np.unique(lab)
            self.phase_labels = uniq
            self.phase_count = int(uniq.size)
        else:
            self.phase_count = int(len(self.fieldnames))

        if self.phase_count > self.max_phases:
            self._warn_too_many_phases_and_exit(self.phase_count)

    def _warn_too_many_phases_and_exit(self, P: int):
        # Rough RAM estimate for dense float storage (ignores ghost nodes and overhead)
        Nx, Ny, Nz = int(self.vf.Nx), int(self.vf.Ny), int(self.vf.Nz)
        bytes_per = 8 if str(self.vf.precision).endswith("64") else 4
        est_gb = P * Nx * Ny * Nz * bytes_per / (1024**3)
        warnings.warn(
            f"MultiPhaseTimeDependentSolver: P={P} phases detected (> {self.max_phases}). "
            f"This dense multiphase approach is not suited for too many phases and becomes RAM critical. "
            f"Rough field storage alone: ~{est_gb:.2f} GB (excluding ghost nodes/overhead). "
            f"Aborting."
        )
        sys.exit(1)

    def _init_fields(self):
        if not self.from_labels:
            # Use normal stacking behavior: each field is one phase channel
            return super()._init_fields()

        # labels-mode: build dense phase tensor
        # NOTE: for large P this is RAM-heavy (guarded above).
        labels_np = self.vf.fields[self.fieldnames[0]]
        phis = (labels_np[None, ...] == self.phase_labels[:, None, None, None])
        u_list = [self.vg.init_scalar_field(phis[p]) for p in range(self.phase_count)]
        u = self.vg.concatenate(u_list, 0)
        u = self.vg.bc.trim_boundary_nodes(u)
        return u
    
    def _export_fields(self, u_out):
        self.phiindex = self.vg.argmax(u_out, dim=0, keepdim=True)

        if not self.from_labels:
            return super()._export_fields(u_out)

        labels_np = self.vg.export_scalar_field_to_numpy(self.phiindex)
        out_name = self.output_label_fieldname or self.fieldnames[0]
        self.vf.set_field(out_name, labels_np)

    def phases_to_rgb(self, phis_slice):
        """Map phase-fields to RGB image."""
        P = phis_slice.shape[0]

        # phase 0 = white; phases 1.. = distinct-ish colors (some are RGB mixes)
        colors = np.array([
            [1.0, 1.0, 1.0],  # 0 white
            [1.0, 0.0, 0.0],  # 1 red
            [0.0, 1.0, 0.0],  # 2 green
            [0.0, 0.0, 1.0],  # 3 blue
            [1.0, 1.0, 0.0],  # 4 yellow
            [1.0, 0.0, 1.0],  # 5 magenta
            [0.0, 1.0, 1.0],  # 6 cyan
            [1.0, 0.5, 0.0],  # 7 orange
            [0.5, 0.0, 1.0],  # 8 purple
            [0.5, 1.0, 0.0],  # 9 lime
            [0.0, 0.5, 1.0],  # 10 azure
        ], dtype=np.float32)

        # rgb_image = np.zeros((H, W, 4))  # Initialize an empty RGB image
        rgb_image = np.tensordot(phis_slice, colors[:P], axes=(0, 0))
        return np.clip(rgb_image, 0.0, 1.0)

    def _plot_frame(self, phis, time, colormap, plot_bounds):
        slice_idx = self.vf.Nz // 2
        phi_label_slice = self.vg.to_numpy(self.phiindex[0, :, :, slice_idx])
        phis_slice = self.vg.to_numpy(phis[:, :, :, slice_idx])
        rgb_slice = self.phases_to_rgb(phis_slice)

        if not hasattr(self, "_fig"):
            self._fig, (self._ax0, self._ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            start1 = self.vf.origin[0] - self.vf.spacing[0] / 2
            start2 = self.vf.origin[1] - self.vf.spacing[1] / 2
            end1 = self.vf.domain_size[0] - start1
            end2 = self.vf.domain_size[1] - start2
            extent = [start1, end1, start2, end2]
            self._im0 = self._ax0.imshow(phi_label_slice.T, cmap=colormap, origin='lower', extent=extent)
            ratio = np.clip((end2 - start2) / (end1 - start1), 0, 1)
            self._cbar = self._fig.colorbar(self._im0, ax=self._ax0, shrink=ratio)
            self._ax0.set_title("Phase Labels")
            self._ax0.set_xlabel("X")
            self._ax0.set_ylabel("Y")

            self._im1 = self._ax1.imshow(np.transpose(rgb_slice, (1, 0, 2)), origin='lower', extent=extent)
            self._ax1.set_title("Phases in RGB")
            self._ax1.set_xlabel("X")
            self._ax1.set_ylabel("Y")
        else:
            self._im0.set_data(phi_label_slice.T)
            self._im1.set_data(np.transpose(rgb_slice, (1, 0, 2)))

        self._fig.suptitle(f"Slice {slice_idx} of phase-fields at time {time}")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

import itertools, time, io, sys, gc, os
import numpy as np
import torch, psutil
from contextlib import redirect_stdout
import evoxels as evo

from evoxels.voxelgrid import VoxelGridTorch
from evoxels.problem_definition import PeriodicCahnHilliard
from evoxels.profiler import TorchMemoryProfiler
from timeit import default_timer as timer
import torchode as to

OUTFILE = 'torchode_tsit5_benchmark_results.txt'

def run_cahn_hilliard_torchode(
    voxelfields,
    fieldname: str,
    device: str = "cuda",
    jit: bool = "True",
    time_increment: float = 0.1,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 3.0,
    diffusivity: float = 1.0
):
    vg = VoxelGridTorch(voxelfields.grid_info(), precision=voxelfields.precision, device=device)
    problem = PeriodicCahnHilliard(vg, eps=eps, D=diffusivity)
    profiler = TorchMemoryProfiler(vg.device)

    y0_3d = vg.init_scalar_field(voxelfields.fields[fieldname])
    batch, Nx, Ny, Nz = y0_3d.shape
    y0 = y0_3d.flatten(start_dim=1)

    t_end = time_increment * max_iters
    dt0 = torch.tensor(time_increment, dtype=vg.precision, device=vg.device)
    t_eval = torch.linspace(0.0, t_end, frames+1,
        dtype=vg.precision, device=vg.device).unsqueeze(0)

    def rhs(t, y_flat):
        y3d = y_flat.view(batch, Nx, Ny, Nz)
        dy3d = problem.rhs(y3d, t)
        return dy3d.flatten(start_dim=1)

    term = to.ODETerm(rhs)
    # controller = to.FixedStepController()
    controller = to.PIDController(atol=1e-5, rtol=1e-3, pcoeff=0.2, icoeff=0.5, dcoeff=0.0, term=term)
    method = to.Tsit5(term=term) #Dopri5(term=term)
    adjoint = to.AutoDiffAdjoint(method, controller, max_steps=None)

    if jit:
        adjoint = torch.compile(adjoint)

    batch_size = y0.shape[0]
    if t_eval is not None and t_eval.ndim == 1:
        t_eval = t_eval.expand((batch_size, -1))

    t_start, t_end = t_eval[:, 0], t_eval[:, -1]
    if t_start.ndim == 0:
        t_start = t_start.expand(batch_size)
    if t_end.ndim == 0:
        t_end = t_end.expand(batch_size)

    ivp = to.InitialValueProblem(y0, t_start, t_end, t_eval)
    start = timer()
    sol = adjoint.solve(ivp, term, dt0=dt0, args=None)
    end = timer()

    # Update field with final state for convenience
    voxelfields.fields[fieldname] = vg.export_scalar_field_to_numpy(
                                        sol.ys[0, -1].view(batch, Nx, Ny, Nz)
                                    )
    profiler.print_memory_stats(start, end, max_iters)
    return sol

def write_header_if_missing():
    """Create file and write header if it doesn't exist."""
    if not os.path.exists(OUTFILE):
        with open(OUTFILE, 'w') as f:
            f.write(f"{'N':>4} {'dev':>4} {'jit':>3} "
                    f"{'Ttime(s)':>9} {'Wtime(s)':>9} {'CPU':>8} "
                    f"{'Nv(cur)':>8} {'Nv(max)':>8} "
                    f"{'T(cur)':>8} {'T(max)':>8} {'T(res)':>8}\n")
            f.write("=" * 83 + "\n")

def append_row_to_file(r: dict):
    """Format a single result dict `r` and append as one line to OUTFILE."""
    line = (f"{r['N']:4d} {r['device'][:4]:>4} {str(r['jit']):>3} "
            f"{r['total_time']:9.3f} {r['solve_time']:9.3f}"
            f"{r['cpu_ram']:8.1f} {r['nvidia_cur']:8.1f} {r['nvidia_max']:8.1f} "
            f"{r['torch_cur']:8.2f} {r['torch_max']:8.2f} {r['torch_res']:8.2f}\n")
    with open(OUTFILE, 'a') as f:
        f.write(line)

def run_benchmark(N, device, jit, rows):
    dt = 1
    end_time = 1000
    max_iters = int(end_time / dt)

    # Setup
    vf = evo.VoxelFields((N, N, N), domain_size=(N, N, N))
    noise = 0.5 + 0.1*(0.5-np.random.rand(N, N, N))
    vf.add_field('c', noise)

    # Pre-cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
    torch._dynamo.reset()
    gc.collect()

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024**2

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Run solver
    buf = io.StringIO()
    with redirect_stdout(buf):
        sol = run_cahn_hilliard_torchode(
            vf, 'c', device=device, jit=jit,
            time_increment=dt, frames=10, max_iters=max_iters,
            diffusivity=1.0, eps=3.0)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / 1024**2
    out = buf.getvalue().splitlines()

    wall_line = next(line for line in out if line.startswith("Wall time:"))
    nvidia_line = next(line for line in out if "GPU-RAM (nvidia-smi)" in line)
    torch_line = next(line for line in out if "GPU-RAM (torch)" in line)

    # Parse wall time
    wall_time = float(wall_line.split()[2])

    # Parse nvidia-smi line
    parts = nvidia_line.replace('(', '').replace(')', '').split()
    nvidia_cur = float(parts[3])
    nvidia_max = float(parts[5])

    # Parse torch line
    parts = torch_line.replace('(', '').replace(')', '').replace(',', '').split()
    torch_cur = float(parts[3])
    torch_max = float(parts[5])
    torch_res = float(parts[8])

    r = {
        'N': N, 'device': device, 'jit': jit,
        'total_time': end_time - start_time,
        'solve_time': wall_time,
        'cpu_ram': end_mem - start_mem,
        'nvidia_cur': nvidia_cur, 'nvidia_max': nvidia_max,
        'torch_cur': torch_cur, 'torch_max': torch_max, 'torch_res': torch_res
    }
    rows.append(r)
    append_row_to_file(r)

def main():
    Ns = [100, 128, 200, 256, 300, 384, 400, 500, 512, 600, 700] #, 800, 900, 1000, 1024]
    devices = ['cuda'] #['cpu', 'cuda']
    jits = [False]
    rows = []

    write_header_if_missing()
    for N, device, jit in itertools.product(Ns, devices, jits):
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Skipping N={N} on CUDA (not available)")
            continue
        run_benchmark(N, device, jit, rows)

if __name__ == '__main__':
    main()

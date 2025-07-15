import itertools, time, io, sys, gc, os
import numpy as np
import jax, psutil
from contextlib import redirect_stdout
import evoxels as evo

from evoxels.voxelgrid import VoxelGridJax
from evoxels.problem_definition import PeriodicCahnHilliard
from evoxels.profiler import JAXMemoryProfiler
from evoxels.timesteppers import PseudoSpectralIMEX_dfx
from timeit import default_timer as timer
import diffrax as dfx

OUTFILE = 'diffrax_tsit5_benchmark_results.txt'

def run_cahn_hilliard_diffrax(
    voxelfields,
    fieldname: str,
    dt: float = 0.1,
    frames: int = 10,
    max_iters: int = 100,
    eps: float = 3.0,
    diffusivity: float = 1.0
):
    vg = VoxelGridJax(voxelfields.grid_info(), precision=voxelfields.precision)
    problem = PeriodicCahnHilliard(vg, eps=eps, D=diffusivity)
    profiler = JAXMemoryProfiler()

    u0 = vg.init_scalar_field(voxelfields.fields[fieldname])
    t_end = max_iters*dt
    saveat = dfx.SaveAt(ts=np.linspace(0, t_end, frames+1))
    # solver = PseudoSpectralIMEX_dfx(problem.fourier_symbol)
    solver = dfx.Tsit5() #Dopri5() #Tsit5() # ImplicitEuler()
    stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-5)

    start = timer()
    sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: problem.rhs(y, t)),
            solver,
            t0=saveat.subs.ts[0],
            t1=saveat.subs.ts[-1],
            dt0=dt,
            y0=u0,
            saveat=saveat,
            max_steps=100000,
            throw=False,
            adjoint=dfx.ForwardMode(),
            stepsize_controller=stepsize_controller,
        )
    padded = problem.pad_bc(sol.ys[:, 0])
    out = vg.bc.trim_ghost_nodes(padded)
    end = timer()

    # Update field with final state for convenience
    voxelfields.fields[fieldname] = vg.to_numpy(out[-1])
    iterations = int(saveat.subs.ts[-1] // dt)
    profiler.print_memory_stats(start, end, iterations)
    return out

def write_header_if_missing():
    """Create file and write header if it doesn't exist."""
    if not os.path.exists(OUTFILE):
        with open(OUTFILE, 'w') as f:
            f.write(f"{'N':>4} {'dev':>4} {'jit':>3} "
                    f"{'Ttime(s)':>9} {'Wtime(s)':>9} {'CPU':>8} "
                    f"{'Nv(cur)':>8} {'Nv(max)':>8}\n")
            f.write("=" * 61 + "\n")

def append_row_to_file(r: dict):
    """Format a single result dict `r` and append as one line to OUTFILE."""
    line = (f"{r['N']:4d} {r['device'][:4]:>4} {str(r['jit']):>3} "
            f"{r['total_time']:9.3f} {r['solve_time']:9.3f}"
            f"{r['cpu_ram']:8.1f} {r['nvidia_cur']:8.1f} {r['nvidia_max']:8.1f}\n")
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
    gc.collect()

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024**2

    start_time = time.perf_counter()

    # Run solver
    buf = io.StringIO()
    with redirect_stdout(buf):
        sol = run_cahn_hilliard_diffrax(
            vf, 'c', dt=dt, frames=10, max_iters=max_iters,
            diffusivity=1.0, eps=3.0)

        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / 1024**2
    out = buf.getvalue().splitlines()

    wall_line = next(line for line in out if line.startswith("Wall time:"))
    nvidia_line = next(line for line in out if "GPU-RAM (nvidia-smi)" in line)

    # Parse wall time
    wall_time = float(wall_line.split()[2])

    # Parse nvidia-smi line
    parts = nvidia_line.replace('(', '').replace(')', '').split()
    nvidia_cur = float(parts[3])
    nvidia_max = float(parts[5])

    r = {
        'N': N, 'device': device, 'jit': jit,
        'total_time': end_time - start_time,
        'solve_time': wall_time,
        'cpu_ram': end_mem - start_mem,
        'nvidia_cur': nvidia_cur, 'nvidia_max': nvidia_max
    }
    rows.append(r)
    append_row_to_file(r)

def main():
    Ns = [100, 128, 200, 256, 300, 384, 400, 500, 512, 600, 700, 800, 900, 1000, 1024]
    devices = ['cuda']
    jits = [True]
    rows = []

    write_header_if_missing()
    for N, device, jit in itertools.product(Ns, devices, jits):
        run_benchmark(N, device, jit, rows)

if __name__ == '__main__':
    main()

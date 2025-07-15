import itertools, time, io, sys, gc, os
import numpy as np
import jax, psutil
from contextlib import redirect_stdout
import evoxels as evo
import jax.profiler
import re
import subprocess

OUTFILE = 'jax_benchmark_results.txt'

def write_header_if_missing():
    """Create file and write header if it doesn't exist."""
    if not os.path.exists(OUTFILE):
        with open(OUTFILE, 'w') as f:
            f.write(f"{'N':>4} {'dev':>4} {'jit':>3} "
                    f"{'Ttime(s)':>9} {'Wtime(s)':>9} {'CPU':>8} "
                    f"{'Nv(cur)':>8} {'Nv(max)':>8} {'Jax(max)':>8}\n")
            f.write("=" * 61 + "\n")

def append_row_to_file(r: dict):
    """Format a single result dict `r` and append as one line to OUTFILE."""
    line = (f"{r['N']:4d} {r['device'][:4]:>4} {str(r['jit']):>3} "
            f"{r['total_time']:9.3f} {r['solve_time']:9.3f}"
            f"{r['cpu_ram']:8.1f} {r['nvidia_cur']:8.1f} {r['nvidia_max']:8.1f}"
            f"{r['jax_max']:8.1f}\n")
    with open(OUTFILE, 'a') as f:
        f.write(line)

def extract_peak_from_pprof(prof_path: str) -> float:
    # Call pprof and capture its output
    result = subprocess.run(
        ["pprof", "--text", prof_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        text=True  # decode stdout as text
    )
    text = result.stdout

    # Look for the line like: "Showing nodes accounting for 4628.79kB"
    match = re.search(r"Showing nodes accounting for\s+([\d\.]+)kB", text)
    if not match:
        raise ValueError("Could not find peak memory line in pprof output")
    
    peak_kb = float(match.group(1))
    return peak_kb/1024 # return in MB

def run_benchmark(N, device, jit, rows):
    dt = 1
    end_time = 1000
    max_iters = int(end_time / dt)

    # Force JAX backend (must be set before any computation)
    # os.environ["JAX_PLATFORM_NAME"] = device

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
        evo.run_cahn_hilliard_solver(
            vf, 'c', 'jax', jit=jit, device=device,
            time_increment=dt, frames=10, max_iters=max_iters)

        # Wait for JAX to finish
        jax.block_until_ready(vf.fields['c'])
        # This is experimental
        jax.profiler.save_device_memory_profile("test.prof")

        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / 1024**2
    # This is experimental
    peak = extract_peak_from_pprof("test.prof")

    out = buf.getvalue().splitlines()
    wall_line = next(line for line in out if line.startswith("Wall time:"))
    # trace_line = next(line for line in out if "CPU-RAM (tracemalloc)" in line)
    # psutil_line = next(line for line in out if "CPU-RAM (psutil)" in line)
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
        'nvidia_cur': nvidia_cur, 'nvidia_max': nvidia_max,
        'jax_max': peak
    }
    rows.append(r)
    append_row_to_file(r)

def main():
    Ns = [100, 128, 200, 256, 300, 384, 400, 500, 512, 600, 700, 800, 900, 1000]
    devices = ['cuda'] #['cpu', 'cuda']
    jits = [False, True]
    rows = []

    write_header_if_missing()
    for N, device, jit in itertools.product(Ns, devices, jits):
        run_benchmark(N, device, jit, rows)

if __name__ == '__main__':
    main()

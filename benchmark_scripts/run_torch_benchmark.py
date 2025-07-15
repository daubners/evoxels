import itertools, time, io, sys, gc, os
import numpy as np
import torch, psutil
from contextlib import redirect_stdout
import evoxels as evo

OUTFILE = 'torch_benchmark_results.txt'

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
        evo.run_cahn_hilliard_solver(
            vf, 'c', 'torch', jit=jit, device=device,
            time_increment=dt, frames=10, max_iters=max_iters)

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
    Ns = [100, 128, 200, 256, 300, 384, 400, 500, 512, 600, 700, 800, 900, 1000, 1024]
    devices = ['cuda'] #['cpu', 'cuda']
    jits = [False, True]
    rows = []

    write_header_if_missing()
    for N, device, jit in itertools.product(Ns, devices, jits):
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Skipping N={N} on CUDA (not available)")
            continue
        run_benchmark(N, device, jit, rows)


if __name__ == '__main__':
    main()

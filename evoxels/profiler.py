import numpy as np
import psutil
import os
import subprocess
import tracemalloc
from abc import ABC, abstractmethod

class MemoryProfiler(ABC):
    """Base interface for tracking host and device memory usage."""
    def get_cuda_memory_from_nvidia_smi(self):
        """Return currently used CUDA memory in megabytes."""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            return int(output.strip().split('\n')[0])
        except Exception as e:
            print(f"Error tracking memory with nvidia-smi: {e}")

    def update_memory_stats(self):
        process = psutil.Process(os.getpid())
        used_cpu = process.memory_info().rss / 1024**2
        self.max_used_cpu = np.max((self.max_used_cpu, used_cpu))
        used = self.get_cuda_memory_from_nvidia_smi()
        self.max_used_gpu = np.max((self.max_used_gpu, used))

    @abstractmethod
    def print_memory_stats(self, start: float, end: float, iters: int):
        """Print profiling summary after a simulation run."""
        pass

class TorchMemoryProfiler(MemoryProfiler):
    def __init__(self, device):
        """Initialize the profiler for a given torch device."""
        import torch
        self.torch = torch
        self.device = device
        tracemalloc.start()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=device)
        self.max_used_gpu = 0
        self.max_used_cpu = 0

    def print_memory_stats(self, start, end, iters):
        """Print usage statistics for the Torch backend."""
        print(f'Wall time: {np.around(end - start, 4)} s after {iters} iterations '
              f'({np.around((end - start)/iters, 4)} s/iter)')
        
        if self.device.type == 'cpu':
            current, peak = tracemalloc.get_traced_memory()
            print(f"CPU-RAM (tracemalloc) current: {current / 1024**2:.2f} MB ({peak / 1024**2:.2f} MB max)")
            tracemalloc.stop()

            process = psutil.Process(os.getpid())
            current = process.memory_info().rss / 1024**2
            print(f"CPU-RAM (psutil)      current: {current:.2f} MB ({self.max_used_cpu:.2f} MB max)")

        elif self.device.type == 'cuda':
            self.update_memory_stats()
            used = self.get_cuda_memory_from_nvidia_smi()
            print(f"GPU-RAM (nvidia-smi)  current: {used} MB ({self.max_used_gpu} MB max)")
            print(f"GPU-RAM (torch)       current: "
                  f"{self.torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB "
                  f"({self.torch.cuda.max_memory_allocated(self.device) / 1024**2:.2f} MB max, "
                  f"{self.torch.cuda.max_memory_reserved(self.device) / 1024**2:.2f} MB reserved)")

class JAXMemoryProfiler(MemoryProfiler):
    def __init__(self):
        """Initialize the profiler for JAX."""
        import jax
        self.jax = jax
        self.max_used_gpu = 0
        self.max_used_cpu = 0
        tracemalloc.start()

    def print_memory_stats(self, start, end, iters):
        """Print usage statistics for the JAX backend."""
        print(f'Wall time: {np.around(end - start, 4)} s after {iters} iterations '
              f'({np.around((end - start)/iters, 4)} s/iter)')

        current, peak = tracemalloc.get_traced_memory()
        print(f"CPU-RAM (tracemalloc) current: {current / 1024**2:.2f} MB ({peak / 1024**2:.2f} MB max)")
        tracemalloc.stop()

        process = psutil.Process(os.getpid())
        current = process.memory_info().rss / 1024**2
        print(f"CPU-RAM (psutil)      current: {current:.2f} MB ({self.max_used_cpu:.2f} MB max)")

        if self.jax.default_backend() == 'gpu':
            self.update_memory_stats()
            used = self.get_cuda_memory_from_nvidia_smi()
            print(f"GPU-RAM (nvidia-smi)  current: {used} MB ({self.max_used_gpu} MB max)")
import numpy as np
import psutil
import subprocess
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
        """Update the maximum observed device memory usage."""
        used = self.get_cuda_memory_from_nvidia_smi()
        self.max_used = np.max((self.max_used, used))

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
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=device)
        self.max_used = 0

    def print_memory_stats(self, start, end, iters):
        """Print usage statistics for the Torch backend."""
        print(f'Wall time: {np.around(end - start, 4)} s after {iters} iterations '
              f'({np.around((end - start)/iters, 4)} s/iter)')
        
        if self.device.type == 'cuda':
            self.update_memory_stats()
            used = self.get_cuda_memory_from_nvidia_smi()
            print(f"GPU-RAM currently allocated: "
                  f"{self.torch.cuda.memory_allocated(self.device) / 1e6:.2f} MB "
                  f"({self.torch.cuda.memory_reserved(self.device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM maximally allocated: "
                  f"{self.torch.cuda.max_memory_allocated(self.device) / 1e6:.2f} MB "
                  f"({self.torch.cuda.max_memory_reserved(self.device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM nvidia-smi current:  {used} MB ({self.max_used} MB max)")
        else:
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

class JAXMemoryProfiler(MemoryProfiler):
    def __init__(self):
        """Initialize the profiler for JAX."""
        import jax
        self.jax = jax
        self.max_used = 0

    def print_memory_stats(self, start, end, iters):
        """Print usage statistics for the JAX backend."""
        print(f'Wall time: {np.around(end - start, 4)} s after {iters} iterations '
              f'({np.around((end - start)/iters, 4)} s/iter)')

        if self.jax.default_backend() == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")
        elif self.jax.default_backend() == 'gpu':
            self.update_memory_stats()
            used = self.get_cuda_memory_from_nvidia_smi()
            print(f"GPU-RAM nvidia-smi current:  {used} MB ({self.max_used} MB max)")

[![Python package](https://github.com/daubners/evoxels/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/daubners/evoxels/actions/workflows/python-package.yml)

# evoxels
A differentiable physics framework for voxel-based microstructure simulations

For more detailed information about the code [read the docs](https://evoxels.readthedocs.io).

<p align="center">
  <img src="docs/evoxels-graphical-abstract.png" width="80%"></img>
</p>

```
In a world of cubes and blocks,
Where reality takes voxel knocks,
Every shape and form we see,
Is a pixelated mystery.

Mountains rise in jagged peaks,
Rivers flow in blocky streaks.
So embrace the charm of this edgy place,
Where every voxel finds its space
```
## Vision

This package provides a unified voxel-based framework that integrates segmented 3D microscopy data, physical simulations, inverse modeling, and machine learning - without any mesh generation. It represents microstructures as dense PyTorch/JAX tensors (e.g. $200^3$ - $1000^3$ voxels) and leverages GPU/CPU-parallel FFT kernels for advanced time stepping schemes applied to phase-field, reaction-diffusion, and transport simulations. By operating entirely within PyTorch’s autodiff graph, it enables end-to-end gradient-based parameter estimation and surrogate training straight from image data. Advanced FFT-based solvers and low-RAM in-place updates scale to hundreds of millions of cells on commodity hardware.

## Description

This package contains a generic ``VoxelFields`` class which is able to handle multiple fields on the same voxel grid.
Basic attributes are the grid size (Nx, Ny, Nz), the length of physical domain (Lx, Ly, Lz), the resulting grid spacing along each axis (dx, dy, dz) as well as the origin (position of lower left corner) for vtk export.
The class comes with some comfort features like adding a corresponding meshgrid, plotting slices, interactive plotting of the whole 3D structure and data export to vtk for further visualization.

The solvers are currently under development.
At this stage, the binary Cahn-Hilliard solver can be used as a reference for computational performance. The precompiled version achieves high computational performance thanks to optimized time-stepping in combination with fast matrix operations on GPU backends.

## Installation

TL;DR
```bash
conda create --name voxenv python=3.12
conda activate voxenv
pip install evoxels[torch,jax,dev,notebooks]
pip install --upgrade "jax[cuda12]"
```

The package is available on pypi but can also be installed by cloning the repository
```
git clone git@github.com:daubners/evoxels.git
```

and then locally installing in editable mode.
It is recommended to install the package inside a Python virtual environment so
that the dependencies do not interfere with your system packages. Create and
activate a virtual environment e.g. using miniconda

```bash
conda create --name myenv python=3.12
conda activate myenv
```
Navigate to the evoxels folder, then
```
pip install -e .[torch] # install with torch backend
pip install -e .[jax]   # install with jax backend
pip install -e .[dev, notebooks] # install testing and notebooks
```
Note that the default `[jax]` installation is only CPU compatible. To install the corresponding CUDA libraries check your CUDA version with
```bash
nvidia-smi
```
then install the CUDA-enabled JAX backend via (in this case for CUDA version 12)
```bash
pip install -U "jax[cuda12]"
```
To install both backends within one environment it is important to install torch first and then upgrade the `jax` installation e.g.
```bash
pip install evoxels[torch, jax, dev, notebooks]
pip install --upgrade "jax[cuda12]"
```
To work with the example notebooks install Jupyter and all notebook related dependencies via
```
pip install -e .[notebooks]
```
Launch the notebooks with
```
jupyter notebook
```
If you are using VSCode open the Command Palette and select
"Jupyter: Create New Blank Notebook" or open an existing notebook file.


## Usage

Example of creating a voxel field object and running a Cahn-Hilliard simulation based on a semi-implicit FFT approach

```
import evoxels as vox
import numpy as np

nx, ny, nz = [100, 100, 100]

vf = vox.VoxelFields((nx, ny, nz), (nx,ny,nz))
noise = 0.5 + 0.1*np.random.rand(nx, ny, nz)
vf.add_field("c", noise)

dt = 0.1
final_time = 100
steps = int(final_time/dt)

vox.run_cahn_hilliard_solver(
    vf, 'c', 'torch', jit=True, device='cuda',
    time_increment=dt, frames=10, max_iters=steps,
    verbose='plot', vtk_out=False, plot_bounds=(0,1)
  )
```
As the simulation is running, the "c" field will be overwritten each frame. Therefore, ``vf.fields["c"]`` will give you the last frame of the simulation. This code design has been chosen specifically for large data such that the RAM requirements are rather low.
For visual inspection of your simulation results, you can plot individual slices (e.g. slice=10) for a given direction (e.g. x)
```
sim.plot_slice("c", 10, direction='x', colormap='viridis')
```
or use the following code for interactive plotting with a slider to go through the volume
```
%matplotlib widget
sim.plot_field_interactive("c", direction='x', colormap='turbo')
```

## License
This code has been published under the MIT licence.

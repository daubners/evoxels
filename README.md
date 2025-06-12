[![Python package](https://github.com/daubners/voxelsss/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/daubners/voxelsss/actions/workflows/python-package.yml)

# voxelsss
A differentiable physics framework for voxel-based microstructure simulations

For more detailed information about the code [read the docs](https://voxelsss.readthedocs.io).

<p align="center">
  <img src="voxelsss-graphical.png" width="70%"></img>
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
At this stage, the binary Cahn-Hilliard solver can be used as a reference for computational performance. Especially the ``method = 'mixed_FFT'`` achieves high computational performance thanks to optimized time-stepping in combination with fast matrix operations from pytorch.

## Installation

Can be installed by cloning the repository
```
git clone git@github.com:daubners/voxelsss.git
```

and then locally installing in editable mode.
It is recommended to install the package inside a Python virtual environment so
that the dependencies do not interfere with your system packages. Create and
activate a virtual environment e.g. using miniconda

```bash
conda create --name myenv python=3.11
conda activate myenv
```
Navigate to the voxelsss folder, then
```
pip install -e .[torch] # install with torch backend
pip install -e .[jax]   # install with jax backend
pip install -e .[torch, dev, notebooks] # also install testing and notebooks
```
To work with the example notebooks install Jupyter as well install all notebook related dependencies via
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
import voxelsss as vox
import numpy as np

nx, ny, nz = [100, 100, 100]

vf = vox.VoxelFields((nx, ny, nz), (nx,ny,nz))
noise = 0.5 + 0.1*np.random.rand(nx, ny, nz)
vf.add_field("c", noise)

dt = 0.1
final_time = 100
iter = int(final_time/dt)
sim = vox.PeriodicCahnHilliardSolver(vf, "c", device='cuda')
sim.solve(method='mixed_FFT', time_increment=dt, frames=10, max_iters=iter, \
          verbose='plot', vtk_out=False, plot_bounds=(0,1))
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

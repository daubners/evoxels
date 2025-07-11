[![Python package](https://github.com/daubners/evoxels/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/daubners/evoxels/actions/workflows/python-package.yml)

# evoxels
A differentiable physics framework for voxel-based microstructure simulations

For more detailed information about the code [read the docs](https://evoxels.readthedocs.io).

![Evoxels overview](https://raw.githubusercontent.com/daubners/evoxels/main/evoxels.png)

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

## Description
**evoxels are not static — they evolve, adapt, and reveal.**
Whether you're modeling phase transitions, predicting effective properties, or coupling imaging and simulation — evoxels is the GPU-native, differentiable core that keeps pace with your science.

Materials science inherently spans disciplines: experimentalists use advanced microscopy to uncover micro- and nanoscale structure, while theorists and computational scientists develop models that link processing, structure, and properties. Bridging these domains is essential for inverse material design where you start from desired performance and work backwards to optimal microstructures and manufacturing routes. Integrating high-resolution imaging with predictive simulations and data‐driven optimization accelerates discovery and deepens understanding of process–structure–property relationships

From a high-level perspective, evoxels is organized around two core abstractions: ``VoxelFields`` and ``VoxelGrid``. VoxelFields provides a uniform, NumPy-based container for any number of 3D fields on the same regular grid, maximizing interoperability with image I/O libraries (e.g. tifffile, h5py, napari, scikit-image) and visualization tools (PyVista, VTK). VoxelGrid couples these fields to either a PyTorch or JAX backend, offering pre-defined boundary conditions, finite difference stencils and FFT libraries.

The evoxels package enables large-scale forward and inverse simulations on uniform voxel grids, ensuring direct compatibility with microscopy data and harnessing GPU-optimized FFT and tensor operations.
This design supports forward modeling of transport and phase evolution phenomena, as well as backpropagation-based inverse problems such as parameter estimation and neural surrogate training - tasks which are still difficult to achieve with traditional FEM-based solvers.
This differentiable‐physics foundation makes it easy to embed voxel‐based solvers as neural‐network layers, train generative models for optimal microstructures, or jointly optimize processing and properties via gradient descent. By keeping each simulation step fast and fully backpropagatable, evoxels enables data‐driven materials discovery and high‐dimensional design‐space exploration.

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
import evoxels as evo
import numpy as np

nx, ny, nz = [100, 100, 100]

vf = evo.VoxelFields((nx, ny, nz), (nx,ny,nz))
noise = 0.5 + 0.1*np.random.rand(nx, ny, nz)
vf.add_field("c", noise)

dt = 0.1
final_time = 100
steps = int(final_time/dt)

evo.run_cahn_hilliard_solver(
    vf, 'c', 'torch', jit=True, device='cuda',
    time_increment=dt, frames=10, max_iters=steps,
    verbose='plot', vtk_out=False, plot_bounds=(0,1)
  )
```
As the simulation is running, the "c" field will be overwritten each frame. Therefore, ``vf.fields["c"]`` will give you the last frame of the simulation. This code design has been chosen specifically for large data such that the RAM requirements are rather low.
For visual inspection of your simulation results, you can plot individual slices (e.g. slice=10) for a given direction (e.g. x)
```
vf.plot_slice("c", 10, direction='x', colormap='viridis')
```
or use the following code for interactive plotting with a slider to go through the volume
```
%matplotlib widget
vf.plot_field_interactive("c", direction='x', colormap='turbo')
```

## License
This code has been published under the MIT licence.

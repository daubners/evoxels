# voxelsss (voxel-based structure simulation solvers)

<p align="center">
  <img src="voxelsss.png" width="50%"></img>
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

## Description

This package contains a generic ``VoxelFields`` class which is able to handle multiple fields on the same voxel grid.
Basic attributes are the grid size (Nx, Ny, Nz), the length of physical domain (Lx, Ly, Lz), the resulting grid spacing along each axis (dx, dy, dz) as well as the origin (position of lower left corner) for vtk export.
The class comes with some comfort features like adding a corresponding meshgrid, plotting slices, interactive plotting of the whole 3D structure and data export to vtk for further visualization.

The solvers are currently under development.
At this stage, the binary Cahn-Hilliard solver can be used as a reference for computational performance. Especially the ``solve_FFT`` achieves high computational performance thanks to optimized time-stepping in combination with fast matrix operations from pytorch.

## Installation

Can be installed by cloning the repository

`git clone git@github.com:daubners/voxelsss.git`

and then locally installing in editable mode. Navigate to the voxelsss folder, then

`pip install -e .`

To utilize all plotting functionality in jupyter notebooks make sure to
```
pip install ipywidgets
pip install ipympl
```

## Usage

Example of creating a voxel field object and running a Cahn-Hilliard simulation based on a semi-implicit FFT approach

```
import voxelsss as vox
import numpy as np

nx, ny, nz = [100, 100, 100]

vf = vox.VoxelFields(nx, ny, nz, (nx,ny,nz))
noise = 0.5 + 0.1*np.random.rand(nx, ny, nz)
vf.add_field("c", noise)

dt = 0.1
final_time = 100
iter = int(final_time/dt)

sim = vox.CahnHilliardSolver(vf, "c", device='cuda')
sim.solve_FFT(time_increment=dt, max_iters=iter, frames=10, verbose='plot', vtk_out=False)
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
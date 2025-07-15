---
title: 'evoxels: A differentiable physics framework for voxel-based microstructure simulations'
tags:
  - Python
  - materials science
  - differentiable physics
  - phase-field method
  - microstructure
authors:
  - name: Simon Daubner
    orcid: 0000-0002-7944-6026
    corresponding: true
    affiliation: 1
  - name: Alexander E. Cohen
    affiliation: 2
  - name: Benjamin Dörich
    orcid: 0000-0001-5840-2270
    affiliation: 3
  - name: Samuel J. Cooper
    orcid: 0000-0003-4055-6903
    affiliation: 1
affiliations:
 - name: Imperial College London, United Kingdom
   index: 1
 - name: Massachusetts Institute of Technology, United States
   index: 2
 - name: Karlsruhe Institute of Technology, Germany
   index: 3
date: 15 July 2025
bibliography: paper.bib
---

# Summary

Materials science inherently spans disciplines: experimentalists use advanced microscopy to uncover micro- and nanoscale structure, while theorists and computational scientists develop models that link processing, structure, and properties. Bridging these domains is essential for inverse material design where you start from desired performance and work backwards to optimal microstructures and manufacturing routes. Integrating high-resolution imaging with predictive simulations and data‐driven optimization accelerates discovery and deepens understanding of process–structure–property relationships.

![Artwork visualizing the core idea of evoxels: a Python-based differentiable physics framework for simulating and analyzing 3D voxelized microstructures.\label{fig:evoxels}](evoxels.png)

The differentiable physics framework **evoxels** is based on a fully Pythonic, unified voxel-based approach that integrates segmented 3D microscopy data, physical simulations, inverse modeling, and machine learning.

- At its core is a voxel grid representation compatible with both pytorch and jax to leverage massive parallelization on CPU, GPU and TPU for large microstructures.
- Both backends naturally provide high computational performance based on just-in-time compiled kernels and end-to-end gradient-based parameter learning through automatic differentiation.
- The solver design based on advanced Fourier spectral time-stepping and low-RAM in-place updates enables scaling to hundreds of millions of DOFs on commodity hardware (e.g. forward Cahn-Hilliard simulation with $400^3$ voxels on NVIDIA RTX 500 Ada Laptop GPU with 4GB memory) and billions of DOFs on high end data-center GPUs ($1024^3$ voxels on NVIDIA RTX A6000; more details see Figure \autoref{fig:benchmark}).
- Its modular design includes comprehensive convergence tests to ensure the right order of convergence and robustness for various combinations of boundary conditions, grid conventions and stencils during rapid prototyping of new PDEs.

While not intended to replace general finite-element or multi-physics platforms, it fills a unique niche for high-resolution voxel workflows, rapid prototyping for structure simulations and materials design, and fully open, reproducible pipelines that bridge imaging, modeling, and data-driven optimization.

From a high-level perspective, evoxels is organized around two core abstractions: VoxelFields and VoxelGrid. VoxelFields provides a uniform, NumPy-based container for any number of 3D fields on the same regular grid, maximizing interoperability with image I/O libraries (e.g. tifffile, h5py, napari, scikit-image) and visualization tools (PyVista, VTK). VoxelGrid couples these fields to either a PyTorch or JAX backend, offering pre-defined boundary conditions, finite difference stencils and FFT libraries. The implemented solvers leverage advanced Fourier spectral timesteppers (e.g. semi-implicit, exponential integrators), on-the-fly plotting and integrated wall-time and RAM profiling. A suite of predefined PDE “problems” (e.g. Cahn–Hilliard, reaction-diffusion, multi-phase evolution) can be solved out of the box or extended via user-defined ODEs classes with custom right-hand sides. Integrated convergence tests ensure each discretization achieves the expected order before it ever touches real microscopy data.

![Visualisation of package concept. The VoxelFields class acts as the user interface for organising 3D fields on a regular grid including plotting and export functions. Solvers are assembled in a modular fashion. The chosen timestepper and ODE class are just-in-time compiled (green becomes one kernel) based on the given VoxelGrid backend.\label{fig:evoxels-code-logic}](evoxels-code-logic.png)

**evoxels** is aimed squarely at researchers who need a “plug-in-your-image, get-your-answer” workflow for digital materials science and inverse design. Experimentalists can feed segmented FIB-SEM or X-ray tomograms directly into high-performance simulations; computational scientists and modelers benefit from a truly open, reproducible framework. It speaks to anyone who wants special-purpose solvers for representative volume elements - without the overhead of mesh generation - while still offering the flexibility to develop new solvers, test boundary conditions, and incorporate machine-learning-driven optimization. evoxels provides both the turnkey usability of a specialized package and the extensibility of a low-level research toolkit for e.g. benchmarking tortuosity, fitting diffusion coefficients, or prototyping novel phase-field models.

# Statement of need

Understanding the link between microstructure and material properties is a central challenge in materials science which increasingly relies on high-resolution 3D imaging, large-scale simulations, and data-driven optimization. Despite the growing availability of segmented volumes from FIB-SEM, X-ray CT, or synchrotron tomography, and data augmentation through generative AI [@Kench2021;@Finegan2022] the pipeline from experimental data to simulation remains fragmented. Existing simulation tools rarely operate directly on voxelized microscopy data, instead requiring costly meshing or complex preprocessing.
While boundary-conforming meshes (finite element/finite volume method) can better capture complex geometries, voxel-based methods (finite difference and Fourier pseudospectral methods)  -- especially in combination with smoothed boundary techniques [@Yu2012; @Daubner2024_micro] -- offer a robust and practical alternative for computing effective material properties.
In many materials science applications, small numerical or geometric errors (e.g. 5–10%) are acceptable, as modeling assumptions are often approximate and the goal is to capture the correct order of magnitude or understand factors like tortuosity or relative transport rates -- that is, how much better or worse a given microstructure performs.
Furthermore, many commercial codes rely on proprietary data formats, complicating data exchange and reproducibility.
In addition to these technical hurdles, significant domain expertise is typically required to configure simulations i.e. choosing appropriate time-stepping schemes, numerical discretizations, and boundary conditions.
Even for well-studied problems such as the Cahn–Hilliard equation [@Cahn1958; @Zhu1999], no scalable 3D Python implementation exists, highlighting a broader lack of open, reusable simulation frameworks in the field.
These gaps in data interoperability, code availability, and accessible expertise continue to hinder progress in understanding process-structure-property relationships and limit the practical deployment of inverse design methodologies.

The evoxels package enables large-scale forward and inverse simulations on uniform voxel grids, ensuring direct compatibility with microscopy data and harnessing GPU-optimized FFT and tensor operations.
This design supports forward modeling of transport and phase evolution phenomena, as well as backpropagation-based inverse problems such as parameter estimation and neural surrogate training - tasks which are still difficult to achieve with traditional FEM-based solvers.
This differentiable‐physics foundation makes it easy to embed voxel‐based solvers as neural‐network layers, train generative models for optimal microstructures, or jointly optimize processing and properties via gradient descent. By keeping each simulation step fast and fully backpropagatable, evoxels enables data‐driven materials discovery and high‐dimensional design‐space exploration.

There remains significant untapped potential in applying FFT-based semi-implicit schemes [@Zhu1999] and exponential integrators [@Hochbruck2010] across the broader landscape of digital materials science. Although these methods are well-established in areas such as spectral homogenization and phase-field modeling, their adoption has largely been limited to specialized research codes.
For example in [@Caliari2022], a ``C++``-CUDA implementation  of exponential integrators combined with FFT on a GPU was shown to outperform state-of-the-art exponential integrators implementations by fully exploiting the tensor structure of the spatial discretizations.
However, few open-source frameworks incorporate these methods into modern simulation pipelines that support automatic differentiation and GPU acceleration—capabilities increasingly critical for inverse design and data-driven workflows.

To evaluate performance against state-of-the-art python libraries, we benchmark the stiff, fourth‐order Cahn–Hilliard spinodal‐decomposition problem using torchode and Diffrax. As shown in Figure 3, evoxels’ native pseudo‐spectral IMEX solver achieves runtimes one to two orders of magnitude shorter than general‐purpose ODE integrators and requires substantially less GPU memory. By contrast, the TSIT5 integrator with PID‐controlled timestepping which is available in both torchode and Diffrax demands finer timesteps, increasing both computation time and memory use to impractical levels for parameter optimization or inverse‐design tasks. We also provide a custom Diffrax pseudo‐spectral IMEX implementation fully integrated into the evoxels framework; while its wall time matches the native evoxels solver, it incurs higher memory overhead. Finally, fully implicit schemes (e.g., Diffrax’s Implicit Euler) exhaust GPU memory on moderate‐sized 3D grids (even $<100^3$), underlining their unsuitability for high-resolution microstructure simulations.

![Comparison of wall time and maximum GPU memory usage for the Cahn-Hilliard (CH) problem. Wall time for solving 1000 timesteps with fixed stepsize $\Delta t=1$ based on pseudo-spectral IMEX scheme with evoxels-torch (blue) and evoxels-jax (red) - both with and without just-in-time (jit) compilation; pseudo spectral IMEX scheme as custom diffrax solver (orange); and tsit5 scheme in combination with a PID timestep controller in torchode (green) and diffrax (purple). Vertical lines denote maximum problem size on Nvidia RTX A6000 for reference. Black datapoints refer to spectral element simulation of CH using MATLAB on Nvidia A100 [@XinyuLiu2024]. GPU memory footprint of all pytorch-based simulations shown in  b) shows linear scaling with amount of voxels.\label{fig:benchmark}](benchmark.png)

evoxels positions itself as a lightweight, accessible, and rigorously tested tool for prototyping voxel-based PDE solvers. Compared to domain-specific tools like taufactor [@Kench2023] and magnum.np [@Bruckner2023], evoxels supports a broader range of problems, boundary conditions, and numerical methods while maintaining a modular, user-friendly interface for imaging-driven workflows.
At the same time, it is more specialized and efficient for problems on uniform grids with fixed physics than general-purpose solvers like FiPy or FEniCS.
evoxels is not intended to replace multiphysics platforms such as COMSOL or MOOSE, but to complement them by filling a niche in high-resolution, imaging-driven, and differentiable simulations.

Building on prior advances in microstructure characterization [@Daubner2024_micro], phase-field modeling for battery materials [@Daubner2025] and the inverse learning of physics from image data [@Zhao2023], evoxels integrates these capabilities into a unified, extensible codebase. It is currently being used by researchers and students alike to advance inverse-learning capabilities and to develop advanced time integration methods.
In a field where open-source simulation tools remain underdeveloped, it provides a practical blueprint for reproducible digital materials science, helping to democratize capabilities that have long been confined to specialist groups or proprietary codebases.


# Acknowledgements

We acknowledge computational resources and support provided by the Imperial College Research Computing Service (http://doi.org/10.14469/hpc/2232). This work has received financial support from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101069726 (SEATBELT project). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or CINEA. Neither the European Union nor the granting authority can be held responsible for them.

# References
import numpy as np
import sympy as sp
import sympy.vector as spv
import voxelsss as vox

### Generalized test case
def rhs_convergence_test(
    ODE_class,       # an ODE class with callable rhs(field, t)->torch.Tensor (shape [x,y,z])
    problem_kwargs,  # problem parameters to instantiate ODE
    test_function,   # exact init_fun(x,y,z)->np.ndarray
    convention='cell_center',
    dtype='float32',
    powers = np.array([3,4,5,6,7]),
    backend = 'torch'
):
    dx     = np.zeros(len(powers))
    errors = np.zeros(len(powers))
    CS = spv.CoordSys3D('CS')

    for i, p in enumerate(powers):
        if convention == 'cell_center':
            vf = vox.VoxelFields((2**p, 2**p, 2**p), (1, 1, 1), convention=convention)
        elif convention == 'staggered_x':
            vf = vox.VoxelFields((2**p + 1, 2**p, 2**p), (1, 1, 1), convention=convention)
        vf.precision = dtype
        grid = vf.meshgrid()
        init_fun = sp.lambdify((CS.x, CS.y, CS.z), test_function, "numpy")
        init_data = init_fun(*grid)

        if backend == 'torch':
            vg = vox.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device='cpu')
        elif backend == 'jax':
            vg = vox.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)
        field = vg.init_scalar_field(init_data)
        field = vg.trim_boundary_nodes(field)
        ODE = ODE_class(vg, **problem_kwargs)

        # Compute solutions
        comp = vg.export_scalar_field_to_numpy(ODE.rhs(field, 0))
        exact_fun = ODE.rhs_analytic(test_function, 0)
        exact_fun = sp.lambdify((CS.x, CS.y, CS.z), exact_fun, "numpy")
        exact = exact_fun(*grid)
        if convention == 'staggered_x':
            exact = exact[1:-1,:,:]

        # Error norm
        diff = comp - exact
        errors[i] = np.linalg.norm(diff)/np.linalg.norm(exact)
        dx[i] = vf.spacing[0]

        # Fit slope
    slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)
    order = ODE.order

    return dx, errors, slope, order
    
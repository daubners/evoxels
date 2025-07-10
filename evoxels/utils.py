import numpy as np
import sympy as sp
import sympy.vector as spv
import evoxels as evo
from evoxels.problem_definition import SmoothedBoundaryODE

### Generalized test case
def rhs_convergence_test(
    ODE_class,       # an ODE class with callable rhs(field, t)->torch.Tensor (shape [x,y,z])
    problem_kwargs,  # problem parameters to instantiate ODE
    test_function,   # exact init_fun(x,y,z)->np.ndarray
    mask_function=None, 
    convention="cell_center",
    dtype="float32",
    powers = np.array([3,4,5,6,7]),
    backend = "torch"
):
    """Evaluate spatial order of an ODE right-hand side.

    ``test_function`` can be a single sympy expression or a list of
    expressions representing multiple variables. The returned error and
    slope arrays have one entry for each provided function.

    Args:
        ODE_class: an ODE class with callable rhs(field, t).
        problem_kwargs: problem-specific parameters to instantiate ODE.
        test_function: single sympy expression or a list of expressions.
        mask_function: static mask for smoothed boundary method.
        convention: grid convention.
        dtype: floate precision (``float32`` or ``float64``).
        powers: refine grid in powers of two (i.e. ``Nx = 2**p``).
        backend: use ``torch`` or ``jax`` for testing.
    """
    # Verify mask_function only used with SmoothedBoundaryODE
    if mask_function is not None and not issubclass(ODE_class, SmoothedBoundaryODE):
        raise TypeError(
            f"Mask function provided but {ODE_class.__name__} "
            "is not a SmoothedBoundaryODE."
        )
    CS = spv.CoordSys3D('CS')
    # Prepare lambdified mask if needed
    mask = (
        sp.lambdify((CS.x, CS.y, CS.z), mask_function, "numpy")
        if mask_function is not None
        else None
    )

    if isinstance(test_function, (list, tuple)):
        test_functions = list(test_function)
    else:
        test_functions = [test_function]
    n_funcs = len(test_functions)

    # Multiply test functions with mask for SBM testing
    if mask is not None:
        temp_list = []
        for func in test_functions:
            temp_list.append(func*mask_function)
        test_functions = temp_list

    dx     = np.zeros(len(powers))
    errors = np.zeros((n_funcs, len(powers)))

    for i, p in enumerate(powers):
        if convention == 'cell_center':
            vf = evo.VoxelFields((2**p, 2**p, 2**p), (1, 1, 1), convention=convention)
        elif convention == 'staggered_x':
            vf = evo.VoxelFields((2**p + 1, 2**p, 2**p), (1, 1, 1), convention=convention)
        vf.precision = dtype
        grid = vf.meshgrid()
        if backend == 'torch':
            vg = evo.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device='cpu')
        elif backend == 'jax':
            vg = evo.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)
    
        # Initialise fields
        u_list = []
        for func in test_functions:
            init_fun = sp.lambdify((CS.x, CS.y, CS.z), func, "numpy")
            init_data = init_fun(*grid)
            u_list.append(vg.init_scalar_field(init_data))

        u = vg.concatenate(u_list, 0)
        u = vg.bc.trim_boundary_nodes(u)

        # Init mask if smoothed boundary ODE
        if mask is not None:
            problem_kwargs["mask"] = mask(*grid)

        ODE = ODE_class(vg, **problem_kwargs)
        rhs_numeric = ODE.rhs(u, 0)

        if n_funcs > 1 and mask is not None:
            rhs_analytic = ODE.rhs_analytic(mask_function, test_functions, 0)
        elif n_funcs > 1 and mask is None:
            rhs_analytic = ODE.rhs_analytic(test_functions, 0)
        elif n_funcs == 1 and mask is not None:
            rhs_analytic = [ODE.rhs_analytic(mask_function, test_functions[0], 0)]
        else:
            rhs_analytic = [ODE.rhs_analytic(test_functions[0], 0)]

        # Compute solutions
        for j, func in enumerate(test_functions):
            comp = vg.export_scalar_field_to_numpy(rhs_numeric[j:j+1])
            exact_fun = sp.lambdify((CS.x, CS.y, CS.z), rhs_analytic[j], "numpy")
            exact = exact_fun(*grid)
            if convention == "staggered_x":
                exact = exact[1:-1, :, :]

            # Error norm
            diff = comp - exact
            errors[j, i] = np.linalg.norm(diff) / np.linalg.norm(exact)
        dx[i] = vf.spacing[0]

    # Fit slope after loop
    slopes = np.array(
        [np.polyfit(np.log(dx), np.log(err), 1)[0] for err in errors]
    )
    if slopes.size == 1:
        slopes = slopes[0]
    order = ODE.order

    return dx, errors if errors.shape[0] > 1 else errors[0], slopes, order
    
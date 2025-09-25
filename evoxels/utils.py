import numpy as np
import sympy as sp
import sympy.vector as spv
import evoxels as evo
from evoxels.problem_definition import SmoothedBoundaryODE
from evoxels.solvers import TimeDependentSolver
import contextlib
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.patches import Patch

### Generalized test case
def rhs_convergence_test(
    ODE_class,
    problem_kwargs,
    test_function,
    mask_function = None,
    convention = "cell_center",
    dtype = "float32",
    powers = np.array([3,4,5,6,7]),
    backend = "torch"
):
    """Evaluate spatial order of an ODE right-hand side.

    ``test_function`` can be a single sympy expression or a list of
    expressions representing multiple variables. The returned error and
    slope arrays have one entry for each provided function.

    Args:
        ODE_class: ODE class with callable rhs(t, u).
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
    # Assumed to be static i.e. no function of t
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
        dx[i] = vf.spacing[0]

        if backend == 'torch':
            vg = evo.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device='cpu')
        elif backend == 'jax':
            vg = evo.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)

        # Init mask if smoothed boundary ODE
        numpy_grid = vf.meshgrid()
        if mask is not None:
            problem_kwargs["mask"] = mask(*numpy_grid)

        # Initialise fields
        u_list = []
        for func in test_functions:
            init_fun = sp.lambdify((CS.x, CS.y, CS.z), func, "numpy")
            init_data = init_fun(*numpy_grid)
            u_list.append(vg.init_scalar_field(init_data))

        u = vg.concatenate(u_list, 0)
        u = vg.bc.trim_boundary_nodes(u)

        ODE = ODE_class(vg, **problem_kwargs)
        rhs_numeric = ODE.rhs(0, u)

        if n_funcs > 1 and mask is not None:
            rhs_analytic = ODE.rhs_analytic(0, test_functions, mask_function)
        elif n_funcs > 1 and mask is None:
            rhs_analytic = ODE.rhs_analytic(0, test_functions)
        elif n_funcs == 1 and mask is not None:
            rhs_analytic = [ODE.rhs_analytic(0, test_functions[0], mask_function)]
        else:
            rhs_analytic = [ODE.rhs_analytic(0, test_functions[0])]

        # Compute solutions
        for j, func in enumerate(test_functions):
            comp = vg.export_scalar_field_to_numpy(rhs_numeric[j:j+1])
            exact_fun = sp.lambdify((CS.x, CS.y, CS.z), rhs_analytic[j], "numpy")
            exact = exact_fun(*numpy_grid)
            if convention == "staggered_x":
                exact = exact[1:-1, :, :]

            # Error norm
            diff = comp - exact
            errors[j, i] = np.linalg.norm(diff) / np.linalg.norm(exact)

    # Fit slope after loop
    slopes = np.array(
        [np.polyfit(np.log(dx), np.log(err), 1)[0] for err in errors]
    )
    if slopes.size == 1:
        slopes = slopes[0]
    order = ODE.order

    return dx, errors if errors.shape[0] > 1 else errors[0], slopes, order


def mms_convergence_test(
    ODE_class,       # an ODE class with callable rhs(field, t)->torch.Tensor (shape [x,y,z])
    problem_kwargs,  # problem parameters to instantiate ODE
    test_function,   # exact init_fun(x,y,z)->np.ndarray
    mask_function=None,
    timestepper_cls=None,
    convention="cell_center",
    dtype="float32",
    mode = 'temporal',
    g_powers = np.array([3,4,5,6,7]),
    t_powers = np.array([3,4,5,6,7]),
    t_final = 1,
    backend = "jax",
    device = 'cpu'
):
    """Evaluate temporal and spatial order of ODE solution.

    ``test_function`` can be a single sympy expression or a list of
    expressions representing multiple variables. The returned error and
    slope arrays have one entry for each provided function.

    Args:
        ODE_class: ODE class with callable rhs(t, u).
        problem_kwargs: problem-specific parameters to instantiate ODE.
        test_function: single sympy expression or a list of expressions.
        mask_function: static mask for smoothed boundary method.
        timestepper_cls: timestepper class with callable step(t, u).
        convention: grid convention.
        dtype: floate precision (``float32`` or ``float64``).
        mode: Use ``temporal`` or ``spatial`` to construct MMS forcing.
        g_powers: refine grid in powers of two (i.e. ``Nx = 2**p``).
        t_powers: refine time increment in powers of two (i.e. ``dt = 2**p``).
        t_final: End time for evaluation. Should be order of L^2/D.
        backend: use ``torch`` or ``jax`` for testing.
        device: use ``cpu`` or ``cuda`` for testing in torch.
    """
    # Verify mask_function only used with SmoothedBoundaryODE
    if mask_function is not None and not issubclass(ODE_class, SmoothedBoundaryODE):
        raise TypeError(
            f"Mask function provided but {ODE_class.__name__} "
            "is not a SmoothedBoundaryODE."
        )
    CS = spv.CoordSys3D('CS')
    t = sp.symbols('t', real=True)

    # Prepare lambdified mask if needed
    # Assumed to be static i.e. no function of t
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

    if mode == 'temporal':
        u_list = [sp.lambdify((t, CS.x, CS.y, CS.z),
                                sp.N(func), backend) \
                    for func in test_functions]
    u_t_list = [sp.lambdify((t, CS.x, CS.y, CS.z),
                            sp.N(sp.diff(func, t)), backend) \
                for func in test_functions]

    dx     = np.zeros(len(g_powers))
    dt     = np.zeros(len(t_powers))
    errors = np.zeros((n_funcs, len(t_powers), len(g_powers)))

    for i, p in enumerate(g_powers):
        if convention == 'cell_center':
            vf = evo.VoxelFields((2**p, 2**p, 2**p), (1, 1, 1), convention=convention)
        elif convention == 'staggered_x':
            vf = evo.VoxelFields((2**p + 1, 2**p, 2**p), (1, 1, 1), convention=convention)
        vf.precision = dtype
        dx[i] = vf.spacing[0]
    
        if backend == 'torch':
            vg = evo.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device=device)
        elif backend == 'jax':
            vg = evo.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)

        # Init mask if smoothed boundary ODE
        numpy_grid = vf.meshgrid()
        if mask is not None:
            problem_kwargs["mask"] = mask(*numpy_grid)

        ODE = ODE_class(vg, **problem_kwargs)
        rhs_orig = ODE.rhs
        grid = vg.meshgrid()

        # Construct new rhs including forcing term from MMS
        if mode == 'temporal':
            def mms_rhs(t, u):
                """Manufactured solution rhs
                with numerical evaluation of rhs in forcing, i.e.
                forcing = du/dt_exact(t,grid) - rhs_num(t, u_exact(t,grid))
                """
                rhs = rhs_orig(t, u)
                t_ = vg.to_backend(t)
                u_ex_list = []
                for j, func in enumerate(test_functions):
                    u_ex_list.append(vg.expand_dim(u_list[j](t_, *grid), 0))
                    rhs = vg.set(rhs, j, rhs[j] + u_t_list[j](t_, *grid))
                u_ex = vg.concatenate(u_ex_list, 0)
                u_ex = vg.bc.trim_boundary_nodes(u_ex)
                rhs -= rhs_orig(t, u_ex)
                return rhs

        elif mode == 'spatial':
            if n_funcs > 1 and mask is not None:
                rhs_func = ODE.rhs_analytic(t, test_functions, mask_function)
                rhs_analytic = [sp.lambdify((t, CS.x, CS.y, CS.z), sp.N(func), backend) for func in rhs_func]
            elif n_funcs > 1 and mask is None:
                rhs_func = ODE.rhs_analytic(t, test_functions)
                rhs_analytic = [sp.lambdify((t, CS.x, CS.y, CS.z), sp.N(func), backend) for func in rhs_func]
            elif n_funcs == 1 and mask is not None:
                rhs_func = ODE.rhs_analytic(t, test_functions[0], mask_function)
                rhs_analytic = [sp.lambdify((t, CS.x, CS.y, CS.z), sp.N(rhs_func), backend)]
            else:
                rhs_func = ODE.rhs_analytic(t, test_functions[0])
                rhs_analytic = [sp.lambdify((t, CS.x, CS.y, CS.z), sp.N(rhs_func), backend)]

            def mms_rhs(t, u):
                """Manufactured solution rhs
                with analytical evaluation of rhs in forcing, i.e.
                forcing = du/dt_exact(t,grid) - rhs_exact(t, grid)
                """
                rhs = rhs_orig(t, u)
                t_ = vg.to_backend(t)
                for j, func in enumerate(test_functions):
                    rhs = vg.set(rhs, j, rhs[j] - rhs_analytic[j](t_, *grid))
                    rhs = vg.set(rhs, j, rhs[j] + u_t_list[j](t_, *grid))
                return rhs
        else:
            raise ValueError("Mode must be 'temporal' or 'spatial'.")

        # Over-write original rhs with contructed mms_rhs
        ODE.rhs = mms_rhs

        # Loop over time refinements
        for k, q in enumerate(t_powers):
            # Initialise fields
            field_names = []
            for j, func in enumerate(test_functions):
                fun = sp.lambdify((t, CS.x, CS.y, CS.z), func, "numpy")
                init_data = fun(0, *numpy_grid)
                final_data = fun(t_final, *numpy_grid)
                vf.add_field(f'u{j}', init_data)
                vf.add_field(f'u{j}_final', final_data)
                field_names.append(f'u{j}')

            # Init time increment and step function
            dt[k] = t_final / 2**q
            timestepper = timestepper_cls(ODE, dt[k])
            step = timestepper.step

            # Init solver
            solver = TimeDependentSolver(
                vf, field_names,
                backend, device=device,
                step_fn=step
            )

            # Wrap solve to capture NaN exit
            nan_hit = False
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    solver.solve(dt[k], 8, int(2**q), problem_kwargs, verbose=False)
                except SystemExit:
                    nan_hit = True

            if nan_hit:
                errors[:, k, i] = np.nan
                continue
        
            # Compute relative L2 error
            for j, func in enumerate(test_functions):
                exact = vf.fields[f'u{j}_final']
                diff = vf.fields[f'u{j}'] - exact
                errors[j, k, i] = np.linalg.norm(diff) / np.linalg.norm(exact)

    # Fit slope after loop
    def calc_slope(x, y):
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]

    t_slopes = np.array([calc_slope(dt, err[:,0]) for err in errors])
    g_slopes = np.array([calc_slope(dx, err[-1,:]) for err in errors])

    results = {
        'dt': dt,
        'dx': dx,
        'error': errors if n_funcs > 1 else errors[0],
        't_slopes': t_slopes if n_funcs > 1 else t_slopes[0],
        'g_slopes': g_slopes if n_funcs > 1 else g_slopes[0],
        'n_funcs': n_funcs,
        't_order': timestepper.order,
        'g_order': ODE.order,
    }
    return results


def plot_error_surface(series, log_axes=(True, True, True), z_max=0, title=None, alpha=0.4):
    """
    Plot one or more 3D surfaces z(x, y) with semi-transparent tiles and solid mesh lines.

    Parameters
    ----------
    series : tuple[list] of dict
        Each dict must have:
          - 'dt': 1D array-like of dt-values  (length Nx)
          - 'dx': 1D array-like of dx-values  (length Ny)
          - 'error': 2D array-like of values Z(X, Y) with shape (Nx, Ny)
          - 'name': (optional) label for legend
    log_axes : tuple(bool, bool, bool)
        (log_x, log_y, log_z): apply log10 to respective axis data when True.
        For Z, nonpositive values are masked to NaN before log10.
    title : str or None
        Plot title.
    alpha : float
        Face transparency for surfaces.
    """
    if not isinstance(series, (list, tuple)) or len(series) == 0:
        raise ValueError("`series` must be a non-empty tuple/list of dictionaries.")

    log_x, log_y, log_z = log_axes

    # Distinct colors
    base_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:gray',
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:orange',
                   'tab:olive', 'tab:cyan']

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    legend_patches = []

    count = 0
    for i, s in enumerate(series):
        if not isinstance(s, dict) or not all(k in s for k in ('dt', 'dx', 'error')):
            raise ValueError(f"Item {i} must be a dict with keys 'dt', 'dx', 'error' (and optional 'name').")

        x_in = np.asarray(s['dt'])
        y_in = np.asarray(s['dx'])
        Z = np.asarray(s['error'])
        Z = np.expand_dims(Z, axis=0) if Z.ndim == 2 else Z
        name = s.get('name', f'[{i}]')

        # Handle (1D,1D,2D) or (2D,2D,2D)
        if x_in.ndim == 1 and y_in.ndim == 1:
            X, Y = np.meshgrid(x_in, y_in, indexing='ij')  # (Nx, Ny)
        else:
            raise ValueError(f"Item {i}: dt and dx must both be 1D grids.")

        if Z.shape[1:] != X.shape:
            raise ValueError(f"Item {i}: z.shape {Z.shape} must match x/y grid shape {X.shape}.")

        # Apply log scaling
        Xp = np.log10(X) if log_x else X
        Yp = np.log10(Y) if log_y else Y
        if log_z:
            Z = np.where(Z > 0, Z, np.nan)
            Zp = np.log10(Z)
        else:
            Zp = Z

        for j in range(s['n_funcs']):
            color = base_colors[count % len(base_colors)]
            ax.plot_surface(
                Xp, Yp, Zp[j],
                color=color,         # uniform color per surface
                alpha=alpha,         # semi-transparent tiles
                edgecolor=color,     # solid mesh lines
                linewidth=0.6,
                antialiased=True,
                shade=False
            )
            label = name + f"_u{j}" if j > 0 else name
            legend_patches.append(Patch(facecolor=color, edgecolor=color, alpha=alpha, label=label))
            count += 1

    # Axis labels reflect log choice
    ax.set_xlabel('log10(dt)' if log_x else 'dt')
    ax.set_ylabel('log10(dx)' if log_y else 'dx')
    ax.text2D(0.0, 0.8, 'log10(error)' if log_z else 'error',
              transform=ax.transAxes, va="top", ha="left")
    ax.set_zlim(top=z_max)
    ax.set_title(title or 'Error Surfaces')
    ax.view_init(elev=25., azim=-145, roll=0)

    ax.legend(handles=legend_patches, loc='best')
    fig.tight_layout()
    plt.show()

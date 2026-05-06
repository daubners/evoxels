import warnings

from .timesteppers import TimeStepper

try:
    import diffrax as dfx

    class DiffraxTimeStepperAdapter(dfx.AbstractSolver):
        """Wrap an ``evoxels`` timestepper as a diffrax solver."""

        timestepper: TimeStepper
        term_structure = dfx.ODETerm
        interpolation_cls = dfx.LocalLinearInterpolation

        def order(self, terms):
            del terms
            return self.timestepper.order

        def init(self, terms, t0, t1, y0, args):
            del terms, t0, t1, y0, args
            return None

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del terms, args, solver_state, made_jump
            dt = t1 - t0
            f0 = self.timestepper.problem.rhs(t0, y0)
            y1 = self.timestepper.step_dt(t0, dt, y0)
            y_error = y1 - (y0 + dt * f0)
            dense_info = dict(y0=y0, y1=y1)
            return y1, y_error, dense_info, None, dfx.RESULTS.successful

        def func(self, terms, t0, y0, args):
            del terms, args
            return self.timestepper.problem.rhs(t0, y0)

except ImportError:
    DiffraxTimeStepperAdapter = None
    warnings.warn("Diffrax not found. 'DiffraxTimeStepperAdapter' will not be available.")

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any


class ODE(ABC):
    @property
    @abstractmethod
    def order(self) -> int:
        """Spatial order of convergence for numerical right-hand side."""

    @abstractmethod
    def rhs_analytic(self, t, u):
        """Sympy expression of the problem right-hand side.

        Args:
            t (float): Current time.
            u : Sympy function of current state.

        Returns:
            Sympy function of problem right-hand side.
        """

    @abstractmethod
    def rhs(self, t, u):
        """Numerical right-hand side of the ODE system.

        Args:
            t (float): Current time.
            u (array): Current state.

        Returns:
            Same type as ``u`` containing the time derivative.
        """

    def pad_bc(self, u):
        """Function to pad and impose boundary conditions.

        Enables applying boundary conditions on u within and
        outside of the right-hand-side function.

        Args:
            u : field

        Returns:
            Field padded with boundary values.
        """
        return self._pad_bc(u)

    def initialize_boundary_conditions(self):
        bc = getattr(self, "bc", None)
        if bc is None or bc == 'fully_periodic':
            self.bc = (('periodic', None), ('periodic', None), ('periodic', None))
            self._pad_bc = self.vg.bc.pad_periodic
            return

        if len(bc) != 3:
            raise ValueError("bc must contain exactly three axis entries ordered as (x, y, z).")

        normalized_bc = []
        for axis_bc in bc:
            if isinstance(axis_bc, str):
                if axis_bc == 'dirichlet':
                    raise ValueError("Dirichlet BCs require explicit values ('dirichlet', (left, right)).")
                if axis_bc not in {'periodic', 'neumann'}:
                    raise ValueError(f"Unsupported BC type: {axis_bc}")
                normalized_bc.append((axis_bc, None))
                continue

            if len(axis_bc) != 2:
                raise ValueError(
                    "Each axis boundary specification must be either a string or "
                    "a tuple like ('dirichlet', (left, right))."
                )

            kind, values = axis_bc
            if kind not in {'periodic', 'dirichlet', 'neumann'}:
                raise ValueError(f"Unsupported BC type: {kind}")

            if kind == 'dirichlet':
                if values is None or len(values) != 2:
                    raise ValueError("Dirichlet BCs require two boundary values.")
                if self.vg.convention == 'cell_center':
                    warnings.warn(
                        "Applying Dirichlet BCs on a cell_center grid "
                        "reduces the spatial order of convergence to 0.5!"
                        )
                normalized_bc.append((kind, tuple(values)))
            else:
                if values is not None:
                    raise ValueError(f"{kind} BCs do not accept boundary values.")
                normalized_bc.append((kind, None))
        self.bc = tuple(normalized_bc)

        if self.bc_type == ('periodic','periodic','periodic'):
            self._pad_bc = self.vg.bc.pad_periodic

        elif self.bc_type == ('dirichlet','periodic','periodic'):
            self._pad_bc = lambda field: self.vg.bc.pad_dirichlet_periodic(
                field, self.bc[0][1][0], self.bc[0][1][1]
            )
        elif self.bc_type == ('neumann','periodic','periodic'):
            self._pad_bc = self.vg.bc.pad_zero_flux_periodic
        else:
            self._pad_bc = lambda field: self.vg.bc.pad_bc(field, self.bc)

    @property
    def bc_type(self):
        """Boundary-condition metadata for the current problem."""
        return tuple(axis_bc[0] for axis_bc in self.bc)


class SemiLinearODE(ODE):
    @property
    @abstractmethod
    def fourier_symbol(self):
        """Symbol of the highest order spatial operator
        
        The symbol of an operator is its representation in the
        Fourier (spectral) domain. For instance the:
        - Laplacian operator $\nabla^2$ has a symbol $-k^2$,
        - diffusion operator $D\nabla^2$ corresponds to $-k^2D$
        
        The symbol is required for pseudo-spectral timesteppers.
        """

    def verify_fft_bc_config(self):
        x_bc, _, _ = self.bc_type
        nonperiodic_axes = tuple(
            axis for axis, kind in zip(('x', 'y', 'z'), self.bc_type)
            if kind != 'periodic'
        )

        if len(nonperiodic_axes) > 1:
            raise ValueError(
                "FFT-based timesteppers currently support at most one non-periodic axis, "
                f"got {self.bc_type}."
            )

        if len(nonperiodic_axes) == 1 and nonperiodic_axes[0] != 'x':
            raise NotImplementedError(
                "FFT-based timesteppers currently only implement the single non-periodic axis "
                f"case for x; got {self.bc_type}. Axis permutation is not implemented yet."
            )

        if x_bc == 'periodic':
            self._pad_fft_bc = self.vg.bc.pad_fft_periodic
        elif x_bc == 'dirichlet':
            self._pad_fft_bc = self.vg.bc.pad_fft_dirichlet_periodic
        elif x_bc == 'neumann':
            self._pad_fft_bc = self.vg.bc.pad_fft_zero_flux_periodic
        else:
            raise ValueError(
                "FFT-based timesteppers only support periodic, dirichlet, or "
                f"neumann boundary conditions in x, got {x_bc}."
            )
        
    def k_squared(self):
        """Helper to choose k^2 for fourier symbol based on BCs."""
        if self.bc_type[0] in {'dirichlet', 'neumann'}:
            return self.vg.rfft_k_squared_nonperiodic()
        else:
            # Note: this is technically not correct for non-periodic BCs,
            # but this is irrelevant if timestepper does not use FFT, while
            # timesteppers which use FFT call verify_fft_bc_config()
            return self.vg.rfft_k_squared()

    def k_squared_fd(self):
        """Helper to choose finite difference k^2 based on BCs."""
        if self.bc_type[0] in {'dirichlet', 'neumann'}:
            return self.vg.rfft_k_squared_nonperiodic_fd()
        else:
            return self.vg.rfft_k_squared_fd()

    def pad_fft_bc(self, u):
        return self._pad_fft_bc(u)


class SmoothedBoundaryODE(ODE):
    @property
    @abstractmethod
    def mask(self) -> Any | float:
        """A field (same shape as the state) that remains fixed."""

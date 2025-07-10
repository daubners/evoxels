# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

class CellCenteredBCs:
    def __init__(self, vg):
        self.vg = vg

    def pad_periodic(self, field):
        """
        Periodic boundary conditions in all directions.
        Consistent with cell centered grid.
        """
        return self.vg.pad_periodic(field)
    
    def pad_dirichlet_periodic(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with cell centered grid,
        but loss of 2nd order convergence.
        """
        padded = self.vg.pad_periodic(field)
        padded = self.vg.set(padded, (__, 0,__,__), 2.0*bc0 - padded[:, 1,:,:])
        padded = self.vg.set(padded, (__,-1,__,__), 2.0*bc1 - padded[:,-2,:,:])
        return padded

    def pad_zero_flux_periodic(self, field):
        padded = self.vg.pad_periodic(field)
        padded = self.vg.set(padded, (__, 0,__,__), padded[:, 1,:,:])
        padded = self.vg.set(padded, (__,-1,__,__), padded[:,-2,:,:])
        return padded
    
    def pad_zero_flux(self, field):
        padded = self.vg.pad_zeros(field)
        padded = self.vg.set(padded, (__, 0,__,__), padded[:, 1,:,:])
        padded = self.vg.set(padded, (__,-1,__,__), padded[:,-2,:,:])
        padded = self.vg.set(padded, (__,__, 0,__), padded[:,:, 1,:])
        padded = self.vg.set(padded, (__,__,-1,__), padded[:,:,-2,:])
        padded = self.vg.set(padded, (__,__,__, 0), padded[:,:,:, 1])
        padded = self.vg.set(padded, (__,__,__,-1), padded[:,:,:,-2])
        return padded
    
    def pad_fft_periodic(self, field):
        """Periodic field needs no fft padding."""
        return field
    
    def pad_fft_dirichlet_periodic(self, field):
        """Pad with inverse of flipped field in x direction."""
        return self.vg.concatenate((field, -self.vg.lib.flip(field, [0])), 1)
    
    def pad_fft_zero_flux_periodic(self, field):
        """Pad with flipped field in x direction."""
        return self.vg.concatenate((field, self.vg.lib.flip(field, [0])), 1)

    def trim_boundary_nodes(self, field):
        return field

    def trim_ghost_nodes(self, field):
        if field[0,_i_,_i_,_i_].shape == self.vg.shape:
            return field[:,_i_,_i_,_i_]
        else:
            raise ValueError(
                f"The provided field has the wrong shape {self.vg.shape}."
            )


class StaggeredXBCs:
    def __init__(self, vg):
        self.vg = vg

    def pad_periodic_BC_staggered_x(self, field):
        """
        If field is fully periodic it should be in
        cell center convention!
        """
        raise NotImplementedError

    def pad_dirichlet_periodic(self, field, bc0=0, bc1=0):
        """
        Homogenous Dirichlet boundary conditions in x-drection,
        periodic in y- and z-direction. Consistent with staggered_x grid,
        maintains 2nd order convergence.
        """
        padded = self.vg.pad_periodic(field)
        padded = self.vg.set(padded, (__, 0,__,__), bc0)
        padded = self.vg.set(padded, (__,-1,__,__), bc1)
        return padded

    def pad_zero_flux_periodic(self, field):
        """
        The following comes out of on interpolation polynomial p with
        p'(0) = 0, p(dx) = f(dx,...), p(2*dx) = f(2*dx,...)
        and then use p(0) for the ghost cell. 
        This should be of sufficient order of f'(0) = 0, and even better if
        also f'''(0) = 0 (as it holds for cos(k*pi*x)  )
        """
        padded = self.vg.pad_periodic(field)
        fac1 =  4/3
        fac2 =  1/3
        padded = self.vg.set(padded, (__, 0,__,__), fac1*padded[:, 1,:,:] - fac2*padded[:, 2,:,:])
        padded = self.vg.set(padded, (__,-1,__,__), fac1*padded[:,-2,:,:] - fac2*padded[:,-3,:,:])
        return padded

    def pad_zero_flux(self, field):
        raise NotImplementedError
    
    def pad_fft_periodic(self, field):
        """
        If field is fully periodic it should be in
        cell center convention!
        """
        raise NotImplementedError
    
    def pad_fft_dirichlet_periodic(self, field):
        """Pad with inverse of flipped field in x direction."""
        bc = self.vg.lib.zeros_like(field[:,0:1])
        return self.vg.concatenate((field, bc, -self.vg.lib.flip(field, [0]), bc), 1)
    
    def pad_fft_zero_flux_periodic(self, field):
        """Pad with flipped field in x direction."""
        raise NotImplementedError

    def trim_boundary_nodes(self, field):
        """Trim boundary nodes of ``field`` for staggered grids."""
        if field.shape[1] == self.vg.shape[0]:
            return field[:,_i_,:,:]
        else:
            raise ValueError(
                f"The provided field must have the shape {self.vg.shape}."
            )

    def trim_ghost_nodes(self, field):
        if field[0,:,_i_,_i_].shape == self.vg.shape:
            return field[:,:,_i_,_i_]
        else:
            raise ValueError(
                f"The provided field has the wrong shape {self.vg.shape}."
            )
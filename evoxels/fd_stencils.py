# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

CENTER = (__,  _i_, _i_, _i_)
LEFT   = (__, slice(None,-2), _i_, _i_)
RIGHT  = (__, slice(2, None), _i_, _i_)
BOTTOM = (__, _i_, slice(None,-2), _i_)
TOP    = (__, _i_, slice(2, None), _i_)
BACK   = (__, _i_, _i_, slice(None,-2))
FRONT  = (__, _i_, _i_, slice(2, None))

class FDStencils:
    """Class wrapper for finite difference stencils
    
    Is inherited by the VoxelGrid to apply stencils to
    backend arrays.
    """

    def to_x_face(self, field):
        """Interpolate to face position staggered in x"""
        return 0.5 * (field[:,1:,:,:] + field[:,:-1,:,:])

    def to_y_face(self, field):
        """Interpolate to face position staggered in y"""
        return 0.5 * (field[:,:,1:,:] + field[:,:,:-1,:])

    def to_z_face(self, field):
        """Interpolate to face position staggered in z"""
        return 0.5 * (field[:,:,:,1:] + field[:,:,:,:-1])

    def grad_x_face(self, field):
        """Gradient at face position staggered in x"""
        return (field[:,1:,:,:] - field[:,:-1,:,:]) * self.div_dx[0]

    def grad_y_face(self, field):
        """Gradient at face position staggered in y"""
        return (field[:,:,1:,:] - field[:,:,:-1,:]) * self.div_dx[1]

    def grad_z_face(self, field):
        """Gradient at face position staggered in z"""
        return (field[:,:,:,1:] - field[:,:,:,:-1]) * self.div_dx[2]

    def grad_x_center(self, field):
        """Gradient in x at cell center"""
        return 0.5 * (field[RIGHT] - field[LEFT]) * self.div_dx[0]

    def grad_y_center(self, field):
        """Gradient in x at cell center"""
        return 0.5 * (field[TOP] - field[BOTTOM]) * self.div_dx[1]

    def grad_z_center(self, field):
        """Gradient in x at cell center"""
        return 0.5 * (field[FRONT] - field[BACK]) * self.div_dx[2]

    def gradient_norm_squared(self, field):
        """Gradient norm squared at cell centers"""
        return self.grad_x_center(field)**2 +\
               self.grad_y_center(field)**2 + \
               self.grad_z_center(field)**2

    def laplace(self, field):
        r"""Calculate laplace based on compact 2nd order stencil.

        Laplace given as $\nabla\cdot(\nabla u)$ which in 3D is given by
        $\partial^2 u/\partial^2 x + \partial^2 u/\partial^2 y+ \partial^2 u/\partial^2 z$
        Returned field has same shape as the input field (padded with zeros)
        """
        # Manual indexing is ~10x faster than conv3d with laplace kernel in torch
        laplace = \
            (field[RIGHT] + field[LEFT]) * self.div_dx2[0] + \
            (field[TOP] + field[BOTTOM]) * self.div_dx2[1] + \
            (field[FRONT] + field[BACK]) * self.div_dx2[2] - \
             2 * field[CENTER] * self.lib.sum(self.div_dx2)
        return laplace

    def normal_laplace(self, field):
        r"""Calculate the normal component of the laplacian

        which is identical to the full laplacian minus curvature.
        It is defined as $\partial^2_n u = \nabla\cdot(\nabla u\cdot n)\cdot n$
        where $n$ denotes the surface normal.
        In the context of phasefield models $n$ is defined as
        $\frac{\nabla u}{|\nabla u|}$.
        The calaculation is based on a compact 2nd order stencil.
        """
        n_laplace =\
            self.grad_x_center(field)**2 * (field[RIGHT] - 2*field[CENTER] + field[LEFT]) * self.div_dx2[0] +\
            self.grad_y_center(field)**2 * (field[TOP] - 2*field[CENTER] + field[BOTTOM]) * self.div_dx2[1]+\
            self.grad_z_center(field)**2 * (field[FRONT] - 2*field[CENTER] + field[BACK]) * self.div_dx2[2]+\
            0.5 * self.grad_x_center(field) * self.grad_y_center(field) *\
                  (field[:,2:,2:,1:-1] + field[:,:-2,:-2,1:-1] -\
                   field[:,:-2,2:,1:-1] - field[:,2:,:-2,1:-1]) * self.div_dx[0] * self.div_dx[1] +\
            0.5 *self.grad_x_center(field) * self.grad_z_center(field) *\
                  (field[:,2:,1:-1,2:] + field[:,:-2,1:-1,:-2] -\
                   field[:,:-2,1:-1,2:] - field[:,2:,1:-1,:-2]) * self.div_dx[0] * self.div_dx[2] +\
            0.5 * self.grad_y_center(field) * self.grad_z_center(field) *\
                  (field[:,1:-1,2:,2:] + field[:,1:-1,:-2,:-2] -\
                   field[:,1:-1,:-2,2:] - field[:,1:-1,2:,:-2]) * self.div_dx[1] * self.div_dx[2]
        norm2 = self.gradient_norm_squared(field)
        bulk = self.lib.where(norm2 <= 1e-7)
        norm2 = self.set(norm2, bulk, 1.0)
        return n_laplace/norm2
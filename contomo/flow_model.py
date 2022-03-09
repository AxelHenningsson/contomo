import numpy as np
import sys
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

class FlowModel(object):
    """Finite Volumes MUSCL scheme reconstruction type flow model.

    This object defines a numerical flow model to approximate a flow model operator
    :math:`\\mathcal{F}[\\cdot]` such that

    .. math::
        \\mathcal{F}[ \\rho, v ] \\approx \\dfrac{\\partial \\rho}{\\partial t}

    where :math:`\\partial \\rho/\\partial t` is the temporal derivative of a density field :math:`\\rho(x,t)`,
    :math:`v(x,t)` is a velocity field, :math:`t` denotes time and :math:`x` is a spatial coordinate.

    Given a finite basis approximation of :math:`v(x,t)` and a current density field :math:`\\rho` the
    flow model defines an approximation to :math:`\\partial \\rho/\\partial t` by making use of a 
    MUSCL finite volumes scheme. For an introduction to MUSCL finite volumes schemes see `this wikipedia article`_:

    .. _this wikipedia article: https://en.wikipedia.org/wiki/MUSCL_scheme

    Specificaly this implementeation uses a superbee limiter function and a mesh with equal cubic cells placed
    on an equdistant grid, such that the entire cell mesh forms also a cube in space.

    Args:
        rho (:obj:`numpy array`): Per cell density field of shape=(N,N,N). Note that the mesh coordinate arrays are reduced
            in size since an implicit assumption of zero density boundary on rho is assumed.
        origin (tuple): Spatial coordinate of cell located at index (0,0,0) in mesh coordinate 
            arrays ``Xi``, ``Yi``, ``Zi`` .
        dx (float): Cell side lenght.
        velocity_basis (:obj:`Basis`): Velocity field basis that can be called to render vertex velocities.

    Attributes:
        rho (:obj:`numpy array`): per cell density field of shape=(N,N,N). Note that the mesh coordinate arrays
            ``Xi``, ``Yi``, ``Zi`` are reduced in size since an implicit assumption of zero density boundary on 
            rho is assumed. Make sure to pad rho with 2 cells of zero density if this is not the case.
        Xi,Yi,Zi (:obj:`numpy array`): Cell mesh x,y,z-coordinates. E.g Xi[i,j,k] is the x-coordinate of cell i,j,k.
            Compared to any input density field of shape=(N,N,N) the mesh coordinate array has shape shape=(N-3,N-3,N-3).
            For an input density field only cells with indices (2:-2,2:-2,2:-2) will be treated as nonzero.
        origin (tuple): Spatial coordinate of cell located at index (0,0,0) in mesh coordinate 
            arrays ``Xi``, ``Yi``, ``Zi`` .
        dx (float): Cell side lenght.
        vertex_gradients (:obj:`list` of :obj:`numpy array`): Precomputed vertex gradients of shape=(N-4,N-4,N-4). These may be
            set via :obj:`fixate_density_field()` and change only with density and not velocity.
        velocity_basis (:obj:`Basis`): a velocity field basis that can be called to render vertex velocities.
        vertex_velocity_matrices (:obj:`list` of :obj:`list` of :obj:`scipy.sparse.csr_matrix`): Sparse precomputed
            matrices enabling fast computation of vertex velocities. Can be set via :obj:`fixate_velocity_basis()` to
            avoid repeated slow interpolation through the ``velocity_basis`` . First index in vertex_velocity_matrices is
            for dimension and second for positive or negative cell vertex. E.g vertex_velocity_matrices[0][0] gives
            a matrix that can render the negative vertex velocity in x-dimension while vertex_velocity_matrices[0][1]
            renders the positive vertex velocity  in x-dimension.

    """

    def __init__(self, rho, origin, dx, velocity_basis ):
        self.rho = rho
        self.dx  = dx
        self.origin = origin
        self.velocity_basis = velocity_basis

        # Interior cell centre coordinates (the mesh) backwards padded to allow fast vertex renderings.
        xcell = self.origin[0] + np.linspace(self.dx,  (self.rho.shape[0]-3)*self.dx, self.rho.shape[0]-3 )
        ycell = self.origin[1] + np.linspace(self.dx,  (self.rho.shape[1]-3)*self.dx, self.rho.shape[1]-3 )
        zcell = self.origin[2] + np.linspace(self.dx,  (self.rho.shape[2]-3)*self.dx, self.rho.shape[2]-3 )
        self.Xi, self.Yi, self.Zi = np.meshgrid(  xcell, ycell, zcell, indexing='ij' )

        self.vertex_gradients = None
        self.vertex_velocity_matrices = None

    def fixate_velocity_basis( self, verbose=True ):
        """Precompute vertex velocity rendering matrices.
        
        This function sets the attribute ``vertex_velocity_matrices``. This
        computation can be slow but results in great speedups if the same velocity 
        basis is to be used with the flow model repeatedly.

        Args:
            verbose (boolean): Print progress of computation.
        """

        data = [[[],[]] for _ in range(3)]
        rows = [[[],[]] for _ in range(3)]
        cols = [[[],[]] for _ in range(3)]
        self.vertex_velocity_matrices = [[[],[]] for _ in range(3)]

        if verbose: 
            print("Precomputing vertex velocity rendering matrices for ",self.velocity_basis.nodal_coordinates.shape[0]," nodes")

        for basis_indx in range(self.velocity_basis.nodal_coordinates.shape[0]):
            ri = self.velocity_basis.get_bounding_sphere_radius( basis_indx )
            nx,ny,nz = self.velocity_basis.nodal_coordinates[basis_indx,:]
            for axis in range(3):

                if verbose:
                    prog_int = int(basis_indx/(self.velocity_basis.nodal_coordinates.shape[0]//50))
                    prog = np.round(100*basis_indx/self.velocity_basis.nodal_coordinates.shape[0],1)
                    sys.stdout.write("\r{0}>".format("="*prog_int)+" "+str(prog)+"%")
                    sys.stdout.flush()
                    time.sleep(0.0001)

                X = [self.Xi.flatten(), self.Yi.flatten(), self.Zi.flatten()]
                X[axis] += self.dx/2.
                mask = np.where( (X[0]-nx)**2 + (X[1]-ny)**2 + (X[2]-nz)**2 < ri**2 )

                rendered_flat_sparse_basis =  self.velocity_basis.get_basis(X[0][mask], X[1][mask], X[2][mask], basis_indx)
                rendered_flat_dense_basis  = np.zeros( X[0].shape )
                rendered_flat_dense_basis[mask] = rendered_flat_sparse_basis[:]
                rendered_dense_basis = rendered_flat_dense_basis.reshape( self.Xi.shape )

                vn,vp = self._get_vertex_velocity_from_padded_array(rendered_dense_basis, axis=axis)

                for i,v in enumerate([vn.flatten(), vp.flatten()]):
                    nonzero_columns = np.where( v != 0 )[0]
                    if len(nonzero_columns)>0:
                        data[axis][i].extend( v[nonzero_columns] )
                        cols[axis][i].extend( nonzero_columns )
                        rows[axis][i].extend( [basis_indx]*len(nonzero_columns) )

        if verbose:
            sys.stdout.write("\r{0}>".format("="*prog_int)+" "+str(100.0)+"%")
            sys.stdout.flush()
            time.sleep(0.0001)
            print(" ")
        shape = (self.velocity_basis.nodal_coordinates.shape[0], len(self.Xi[1:,1:,1:].flatten()) )
        for axis in range(3):
            self.vertex_velocity_matrices[axis][0] = csr_matrix((data[axis][0], (rows[axis][0], cols[axis][0])), shape=shape)
            self.vertex_velocity_matrices[axis][1] = csr_matrix((data[axis][1], (rows[axis][1], cols[axis][1])), shape=shape)

    def fixate_density_field( self, rho ):
        """Set the density field values and precompute vertex density gradients.
        """
        self.rho = rho
        self.vertex_gradients = [ self._get_vertex_gradients( self.rho, axis ) for axis in range(3) ]

    def _get_vertex_velocity_from_padded_array(self, padded_velocity, axis):
        """Return positive and negative vertex velocity from a backwards padded velocity array.
        """
        if axis==0:
            vn = padded_velocity[ 0:-1,  1:,  1: ]
            vp = padded_velocity[ 1:  ,  1:,  1: ]
        elif axis==1:
            vn = padded_velocity[ 1:,  0:-1,  1: ]
            vp = padded_velocity[ 1:,  1:,  1:   ]
        elif axis==2:
            vn = padded_velocity[ 1:,  1:,  0:-1 ]
            vp = padded_velocity[ 1:,  1:,  1:   ]
        return vn, vp

    def _get_vertex_gradients(self, f, axis):
        """Compute the vertex gradients based on the axis dimension and limiter function.
        """ 
        df = self._get_density_gradient( f, axis )
        rn, ri, rp = self._r( df, axis )
        phin = self._limiter_function( rn )
        phii = self._limiter_function( ri )
        phip = self._limiter_function( rp )

        if axis==0:
            rho_L_p = f[2:-2,2:-2,2:-2]  +  0.5*phii*df[2:-2,2:-2,2:-2]
            rho_L_n = f[1:-3,2:-2,2:-2]  +  0.5*phin*df[1:-3,2:-2,2:-2]
            rho_R_p = f[3:-1,2:-2,2:-2]  -  0.5*phip*df[3:-1,2:-2,2:-2]
            rho_R_n = f[2:-2,2:-2,2:-2]  -  0.5*phii*df[2:-2,2:-2,2:-2]
        elif axis==1:
            rho_L_p = f[2:-2,2:-2,2:-2]  +  0.5*phii*df[2:-2,2:-2,2:-2]
            rho_L_n = f[2:-2,1:-3,2:-2]  +  0.5*phin*df[2:-2,1:-3,2:-2]
            rho_R_p = f[2:-2,3:-1,2:-2]  -  0.5*phip*df[2:-2,3:-1,2:-2]
            rho_R_n = f[2:-2,2:-2,2:-2]  -  0.5*phii*df[2:-2,2:-2,2:-2]
        elif axis==2:
            rho_L_p = f[2:-2,2:-2,2:-2]  +  0.5*phii*df[2:-2,2:-2,2:-2]
            rho_L_n = f[2:-2,2:-2,1:-3]  +  0.5*phin*df[2:-2,2:-2,1:-3]
            rho_R_p = f[2:-2,2:-2,3:-1]  -  0.5*phip*df[2:-2,2:-2,3:-1]
            rho_R_n = f[2:-2,2:-2,2:-2]  -  0.5*phii*df[2:-2,2:-2,2:-2]

        return rho_L_p, rho_L_n, rho_R_p, rho_R_n

    def _limiter_function( self, r ):
        """Superbee limiter function.

        Args:
            r (float): Local density gradient (f[i]-f[i-1])/(f[i+1]-f[i]).

        Returns:
            float: Limiter function value

        """
        min1 = np.minimum( 2*r,   1   )
        min2 = np.minimum(   r,   2   )
        max1 = np.maximum( min1, min2 )
        max2 = np.maximum(   0,  max1 )
        return max2

    def _r( self, df, axis ):
        """Local density gradient, input to limiter function.

        Args:
            f (:obj:`numpy array`): Discretized density field as a numpy 3d array.
            i (int): Cell index in dimension x.
            j (int): Cell index in dimension y.
            k (int): Cell index in dimension z.
            axis (int): axis dimension of gradient.

        Returns:
            float: Local density gradient value at index (i, j, k) along axis.

        """
        old_settings  = np.seterr(all='ignore')
        if axis==0:
            rn = ( df[0:-4,2:-2,2:-2]   ) / ( df[1:-3,2:-2,2:-2] )
            ri = ( df[1:-3,2:-2,2:-2]   ) / ( df[2:-2,2:-2,2:-2] )
            rp = ( df[2:-2,2:-2,2:-2]   ) / ( df[3:-1,2:-2,2:-2] ) 
        elif axis==1:
            rn = ( df[2:-2,0:-4,2:-2]   ) / ( df[2:-2,1:-3,2:-2] )
            ri = ( df[2:-2,1:-3,2:-2]   ) / ( df[2:-2,2:-2,2:-2] )
            rp = ( df[2:-2,2:-2,2:-2]   ) / ( df[2:-2,3:-1,2:-2] ) 
        elif axis==2:
            rn = ( df[2:-2,2:-2,0:-4]   ) / ( df[2:-2,2:-2,1:-3] )
            ri = ( df[2:-2,2:-2,1:-3]   ) / ( df[2:-2,2:-2,2:-2] )
            rp = ( df[2:-2,2:-2,2:-2]   ) / ( df[2:-2,2:-2,3:-1] ) 
        np.seterr(**old_settings)

        self._map_to_intervall([rn,ri,rp], 0.0, 2.0)

        return rn,ri,rp

    def _map_to_intervall(self, arrays, low, high):
        """Map np.nan and np.inf low and high valeus in a list of numpy arrays.

        NOTE: In place mutation of all input arrays are performed.

        Args:
            low  (float): value to replace np.nan with.
            high (float): value to replace np.inf with.
            arrays (array like): Iterable of numpy arrays.

        """
        for arr in arrays:
            arr[np.isnan(arr)] = low
            arr[np.isinf(arr)] = high

    def _get_vertex_velocity(self, axis):
        """Evaluate the vertex velocities on the 2-padded interior of the cell mesh.
        """
        if self.vertex_velocity_matrices is None:

            if axis==0:
                padded_velocity  = self.velocity_basis(self.Xi + self.dx/2., self.Yi, self.Zi, dim=axis)
            elif axis==1:
                padded_velocity  = self.velocity_basis(self.Xi, self.Yi + self.dx/2., self.Zi , dim=axis)
            elif axis==2:
                padded_velocity  = self.velocity_basis(self.Xi, self.Yi, self.Zi + self.dx/2., dim=axis)

            vn, vp = self._get_vertex_velocity_from_padded_array( padded_velocity, axis=axis )

        else:
            vn = self.vertex_velocity_matrices[axis][0].T.dot( self.velocity_basis.coefficents[ :, axis ] ).reshape(self.Xi[1:,1:,1:].shape)
            vp = self.vertex_velocity_matrices[axis][1].T.dot( self.velocity_basis.coefficents[ :, axis ] ).reshape(self.Xi[1:,1:,1:].shape)

        return vn, vp

    def _get_density_gradient(self, rho, axis):
        """Forward first order finite difference on cell centres"""
        padshape = list(rho.shape)
        padshape[axis] = 1
        rhopad = np.append(rho, np.zeros( tuple(padshape) ), axis=axis)
        return np.diff( rhopad, n=1, axis=axis )

    def _sign(self, x):
        """Positive sign mapper (0 => 1)
        """
        s = np.sign(x)
        s[s==0]=1
        return s

    def get_temporal_density_derivatives( self, get_vertex_velocity_derivatives=False ):
        """Compute per cell density derivatives with respect to time.

        The option ``get_vertex_velocity_derivatives`` allows for computing the derivatives w.r.t
        the vertex velocities.

        .. math::
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{x-}},
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{x+}},
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{y-}},
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{y+}},
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{z-}},
             \\dfrac{\\partial \\mathcal{F}[ \\rho, v ]}{\\partial v_{z+}},

        These are usefull to construct further derivaitves of the flow model w.r.t the 
        velocity basis coefficents.

        Note:
            Computation is based on the attributes ``rho`` and ``velocity_basis`` that defines
            density and velocity field. These values are meant to be set before calling this 
            function. By using the ``fixate_density_field`` and ``fixate_velocity_basis``
            speedups can be achived.

        Args:
            get_vertex_velocity_derivatives (:obj:`boolean`): Also compute derivatie w.r.t vertex velocity.

        Returns:
            Per cell density derivative with respect to time as a :obj:`numpy array` of shape=(N,N,N), and, 
            optionally the flow model derivative w.r.t vertex velocities as a :obj:`numpy array` of 
            shape=(N,N,N,3,2), where index (i,j,k,axis,sign) gives the vertex velcoity derivative in dimension
            axis of sign sign. i.e (i,j,k,2,1) gives the derivative in cell i,j,k w.r.t :math:`y+`.

        """

        # NOTE: The cell-mesh is assumed to be padded with a edge of 2 cells 
        # containing zero density. The relevant region where dfbardt!=0 is 
        # thus in the indices (2:-2, 2:-2, 2:-2) along the three axis.
        assert np.alltrue( self.rho[:,:,0:2]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'
        assert np.alltrue( self.rho[:,0:2,:]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'
        assert np.alltrue( self.rho[0:2,:,:]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'
        assert np.alltrue( self.rho[:,:,-2:]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'
        assert np.alltrue( self.rho[:,-2:,:]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'
        assert np.alltrue( self.rho[-2:,:,:]==0 ), 'The cell-mesh is not padded by 2 cells containing zero density.'

        # To be output derivatives
        drhodt = np.zeros(self.rho.shape)

        if get_vertex_velocity_derivatives:
            ddrhodtdv = np.zeros(self.rho.shape+(3,2,))

        # For each spatial dimension axis=(x,y,z)
        for axis in range(3):

            # Calculate cell vertex density gradients
            if self.vertex_gradients is not None:
                rho_L_p, rho_L_n, rho_R_p, rho_R_n = self.vertex_gradients[axis]
            else:
                rho_L_p, rho_L_n, rho_R_p, rho_R_n = self._get_vertex_gradients( self.rho, axis )

            # Calculate cell vertex velocities
            vn, vp = self._get_vertex_velocity( axis )

            # Calculate numerical fluxes
            Fn = 0.5*( vn*( rho_R_n + rho_L_n ) - np.abs( vn )*(rho_R_n - rho_L_n) )
            Fp = 0.5*( vp*( rho_R_p + rho_L_p ) - np.abs( vp )*(rho_R_p - rho_L_p) )

            # Add contributions to derivatives
            drhodt[2:-2, 2:-2, 2:-2] += (-1./self.dx)*(Fp - Fn)

            if get_vertex_velocity_derivatives:
                dFndu = 0.5*( 1.0*( rho_R_n + rho_L_n ) - self._sign(vn)*(rho_R_n - rho_L_n) )
                dFpdu = 0.5*( 1.0*( rho_R_p + rho_L_p ) - self._sign(vp)*(rho_R_p - rho_L_p) )
                ddrhodtdv[2:-2, 2:-2, 2:-2, axis, 0] = dFndu*(-1./self.dx)
                ddrhodtdv[2:-2, 2:-2, 2:-2, axis, 1] = dFpdu*(-1./self.dx)

        if get_vertex_velocity_derivatives:
            return drhodt, ddrhodtdv
        else:          
            return drhodt

    def max_cfl(self, dt):
        """Return maximum Courant–Friedrichs–Lewy number.
        
        The Courant–Friedrichs–Lewy number, :math:`C`, is defined as

        .. math:: C = \\dfrac{\\Delta t ( \\| v_x \\|+\\| v_y \\|+\\| v_z \\| )}{\\Delta x}

        where :math:`\\Delta t` is a time increment,  :math:`\\Delta x` the size of the cells in the
        mesh and :math:`v_x`, :math:`v_y`, :math:`v_z` are the 3d velocities. The Courant–Friedrichs–Lewy number
        must always be less than unity for the possibility of stability to exist. For more info 
        see this `wikipedia article`_:

        .. _wikipedia article: https://en.wikipedia.org/wiki/Courant-Friedrichs-Lewy_condition
       
        NOTE: 
            It is assumed that the ``velocity_basis`` defines a velocity basis with maximum
            Courant–Friedrichs–Lewy number at the basis nodes. This is the case for a Finite
            Element type basis. Such a basis property greatly simplifies the computation of 
            the Courant–Friedrichs–Lewy number, since it is then enough to evaluate the above
            formulat at a small finite number of points in space.

        Args:
            dt (float): Size of timestep.

        Returns:
            (float) maximum Courant–Friedrichs–Lewy number.

        """
        return np.max( np.sum( np.abs( self.velocity_basis.coefficents ), axis = 1 )*dt / self.dx )
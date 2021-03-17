import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyevtk.hl import gridToVTK
import meshio
import vtk
from vtk.util import numpy_support
from . import ray_model

"""A set of utility functions used across package.

This module defines a set of callable functions that are shared between models and objects in
the package. Utility functions for scripting with the package is also keept in the utility module.
"""

def downsample_sinogram( sinogram, sampling ):
    """Downsample a sinogram array by a fixed factor by forming local means.

    Args:
        sinogram (:obj:`numpy array`): Sinogram to be downsampled.
        sampling (:obj:`int`): Downsampling factor.
    
    Returns:
        :obj:`numpy array` downsampled sinogram.

    """
    shape = (sinogram.shape[0]//sampling, sinogram.shape[2]//sampling)
    downsampled_sinogram = np.zeros((shape[0],sinogram.shape[1],shape[1]))
    for i in range(sinogram.shape[1]):
        sh = shape[0],sinogram[:,i,:].shape[0]//shape[0],shape[1],sinogram[:,i,:].shape[1]//shape[1]
        downsampled_sinogram[:,i,:] = sinogram[:,i,:].reshape(sh).mean(-1).mean(1)
    return downsampled_sinogram

def tomographic_density_field_reconstruction( angles, sinogram, maxiter):
    """Solve a classic tomography problem using positivity constraints and zero paddings.
    
    Reconstructs a volume from sinograms enforcing positive voxel values in the reconstruction
    and a zero padding along the 2 depth edge of the reconstruction. The scipy L-BFGS-B
    implementation is used for solving the problem. This method is meant to be used for retrieveing
    an initial volume from a full refrence scan from which further reconstruction with sparse angles
    can take place.

    Args:
        angles (:obj:`numpy array`): Sinogram angles in degrees.
        sinogram (:obj:`numpy array`): Sinograms.
        maxiter (:obj:`int`): Maximum number of L-BFGS-B iterations. 

    Returns:
        :obj:`numpy array` Reconstructed volume. 

    """
    x0 = np.zeros((sinogram.shape[0],sinogram.shape[0],sinogram.shape[0]))
    raymodel = ray_model.RayModel(x0.shape, sinogram.shape[0], angles)

    def cost_and_jac( x ): 
        res = raymodel.forward_project(x.reshape(x0.shape)) - sinogram
        jac = 2*raymodel.backward_project( res ).flatten().astype(np.float64)
        cost = np.linalg.norm( res )**2
        return cost, jac

    def callback(x,res=None):
        cost,_ = cost_and_jac( x )
        print(cost)
        return False

    # Ensure the padding is zero in the reconstruction.
    bv = np.ones(x0.shape)
    bv[:,:,0:2] = 0
    bv[:,0:2,:] = 0
    bv[0:2,:,:] = 0
    bv[:,:,-2:] = 0
    bv[:,-2:,:] = 0
    bv[-2:,:,:] = 0
    bounds = [  ]
    for b in bv.flatten():
        if b==0: bounds.append( (0,0) )
        if b==1: bounds.append( (0,None) )

    res = minimize( cost_and_jac, x0.flatten(), method='L-BFGS-B',  \
                    jac=True, bounds=bounds,     \
                    callback=callback, options={'disp': True, 'maxiter':maxiter, 'gtol': 1e-16, 'ftol':1e-16, 'maxls': 200, 'maxcor': 100} )
    
    return res.x.reshape(x0.shape)

def euler_step(dydt, t0, y0, dt):
    """Perform a single Euler Forward step.

    Args:
        dydt (:obj:`callable`): Derivative of target w.r.t time. ``dydt(t,y)`` should return the
            temporal derivative of a Parial Differential Equation (PDE) at time t for spatial field y.
        t0 (:obj:`float`): Time at which to start integration.
        y0 (:obj:`numpy array`): Spatial discretized field at ``t0``.
        dt (:obj:`callable`): Integration steplength.

    Returns:
        :obj:`numpy array` Spatial discretized field at time ``t0+dt``

    """
    t,y = t0,y0
    return y + dt*dydt(t, y)

def TVD_RK3_step(dydt, t0, y0, dt):
    """Perform a single Third-order Strong Stability Preserving Runge-Kutta step, (SSPRK3).

    This is a convex combination of euler forward steps which results 
    in a Total Variational Diminishing (TVD) step if a single euler forward
    step of size dt is also TVD.

    citation:
        `Total Variation Diminishing Runge-Kutta Schemes,
        Sigal Gottlieb and Chi-Wang Shu,
        Methematics of Computation,
        Volume 67, Number 221, January 1998, Pages 73–85`_:

    .. _`Total Variation Diminishing Runge-Kutta Schemes,
        Sigal Gottlieb and Chi-Wang Shu,
        Methematics of Computation,
        Volume 67, Number 221, January 1998, Pages 73–85`: https://doi.org/10.1090/S0025-5718-98-00913-2

    Butcher tableau can be `found at wikipedia`_: 

    .. _`found at wikipedia`: https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods

    Args:
        dydt (:obj:`callable`): Derivative of target w.r.t time. ``dydt(t,y)`` should return the
            temporal derivative of a Parial Differential Equation (PDE) at time t for spatial field y.
        t0 (:obj:`float`): Time at which to start integration.
        y0 (:obj:`numpy array`): Spatial discretized field at ``t0``.
        dt (:obj:`callable`): Integration steplength.

    Returns:
        :obj:`numpy array` Spatial discretized field at time ``t0+dt``

    """
    t,y = t0,y0
    k1 = dydt(t, y)
    k2 = dydt(t + dt, y + dt*k1)
    k3 = dydt(t + dt/2., y + dt*( (1./4)*k1 + (1./4)*k2 ) )
    return y + dt*(1./6)*( k1 + k2 + 4*k3 )

def get_total_variation(volume):
    """Compute Total Variation of a discretized density volume.

    The Total Variation, :math:`TVD`, is defined as

    .. math::
        TVD = \\int \\|\\dfrac{\\partial \\rho}{\\partial x}\\|_1 + 
        \\|\\dfrac{\\partial \\rho}{\\partial y}\\|_1 + 
        \\|\\dfrac{\\partial \\rho}{\\partial z}\\|_1

    where :math:`\\rho` is a spatial field and :math:`\\| \\cdot \\|_1` denotes
    the absoulte value (:math:`L1` norm). The parial derivaitves are implemented
    as forward finite differences.

    Args:
        volume (:obj:`numpy array`): Density field.
    
    Returns:
        :obj:`float` Total Variation of input volume.

    """ 
    dx = np.diff(volume, axis=0)[:,0:-1,0:-1]
    dy = np.diff(volume, axis=1)[0:-1,:,0:-1]
    dz = np.diff(volume, axis=2)[0:-1,0:-1,:]
    return np.sum( np.abs(dx) + np.abs(dy) + np.abs(dz) )

def save_as_vtk_particles(file, coordinates, velocities, radii):
    """Save numpy arrays with particle information to paraview readable format.

    Args:
        file (:obj:`string`): Absolute path ending with desired filename. 
        coordinates (:obj:`numpy array`): Coordinates of particle ensemble, shape=(N,3).
        velocities (:obj:`numpy array`): Velocities of particle ensemble, shape=(N,3).
        radii (:obj:`numpy array`): Radii of particle ensemble, shape=(N,).

    """
    cells = [("vertex", np.array([[i] for i in range(coordinates.shape[0])]) )]
    meshio.Mesh(
        coordinates,
        cells,
        point_data={"radii" : radii, "velocity": velocities},
        ).write(file)

def save_as_vtk_voxel_volume(file, voxel_volume):
    """Save numpy array with voxel information to paraview readable format.

    Args:
        file (:obj:`string`): Absolute path ending with desired filename. 
        voxel_volume (:obj:`numpy array`): Per voxel density values in a 3d array.

    """ 
    x = np.arange(0, voxel_volume.shape[0]+1, dtype=np.int32)
    y = np.arange(0, voxel_volume.shape[1]+1, dtype=np.int32)
    z = np.arange(0, voxel_volume.shape[2]+1, dtype=np.int32)
    gridToVTK(file, x, y, z, cellData = {'voxel_volume': voxel_volume})

def load_vtk_point_data(file):
    """Load a vtk file containing a liggghts dem output of moving particles.

    Args:
        file (:obj:`string`): 

    Returns:
        :obj:`numpy arrays` with per particle, ``coordinates``, ``radius`` and ``velocity`` 
        the format of these arrays is the same as in :meth:`save_as_vtk_particles`.

    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()

    coordinates = np.array( [ data.GetPoint(i) for i in range(data.GetNumberOfPoints()) ] )
    
    radius      = numpy_support.vtk_to_numpy(data.GetPointData().GetArray("radius"))
    density     = numpy_support.vtk_to_numpy(data.GetPointData().GetArray("mass")) / ( (4./3)*(radius**3)*np.pi )
    velocity    = numpy_support.vtk_to_numpy(data.GetPointData().GetArray("v"))
    
    ids         = numpy_support.vtk_to_numpy(data.GetPointData().GetArray("id"))
    order       = np.argsort(ids)

    return coordinates[order], radius[order], velocity[order], density[order]

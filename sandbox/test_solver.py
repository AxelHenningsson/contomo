import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/axel/workspace/contomo/')
import utils
from FVM import FiniteVolumes
from solver import FVMSolver
from sinogram_interpolator import SinogramSplineInterpolator
from basis import LinearTetraMeshBasis, TetraMesh
from analytical_phantom import AnalyticalPhantom
from projector import Projector

ap = AnalyticalPhantom.load('/home/axel/workspace/contomo/dem_inversion/dem_phantom')

N = ap.dynamic_sinograms[0].shape[0]
volume_shape = (N,N,N)
reference_projector = Projector( (N,N,N), N, N, ap.static_projection_angles[0] )

initial_volume = np.load('astra_recon_initial_volume.npy')
final_volume = np.load('astra_recon_final_volume.npy') 

assert np.sum(initial_volume<0)==0


projector = Projector( (N,N,N), N, N, ap.dynamic_projection_angles[0] )


current_volume  = initial_volume.copy()
sample_times    = np.array(ap.dynamic_sample_times)
sinograms       = np.array(ap.dynamic_sinograms)
spline_interp   = SinogramSplineInterpolator(sample_times, sinograms, smoothness=0, order=2)
spline_interp.save_state()

dt_sampling       = (sample_times[1]-sample_times[0])         # time between two sinograms (assumed to be uniform)
nbr_of_substeps   = 1                                         # number of integration steps between sample times
dt                = dt_sampling/nbr_of_substeps               # time between two integration steps
integration_times = np.arange(sample_times[0], sample_times[-3] + dt, dt)


volumes = [current_volume.copy()]

assert ap.static_projection_angles[0][0]==ap.dynamic_projection_angles[0][0]
assert np.linalg.norm(  ap.static_sinograms[0][:,0,:] - sinograms[0,:,0,:] )==0
print('##############################################################################')
print(' R A Y    M O D E L    E R R O R ')
ray_model_error =  np.linalg.norm( projector.forward_project( initial_volume ) - sinograms[0])
print( ray_model_error )
print('##############################################################################')

# Define a finite basis for velocities.
dx = ap.pixel_size
esize = N/8.0
velocity_basis = TetraMesh()
velocity_basis.generate_mesh_from_numpy_array( np.ones((N,N,N),dtype=np.uint8), dx, max_cell_circumradius=esize*dx, max_facet_distance=esize*dx)
velocity_basis.expand_mesh_data()

print('Nelm:  ',velocity_basis.enod.shape[0])
print('Nnods: ',velocity_basis.coord.shape[0])
velocity_basis.to_xdmf('dem_phantom_mesh')

# We define a scaled set of meshes to be used in the velocity recovery step
origin = (0,0,0)
dx_scaled  = 1.0
scaled_flow_model = FiniteVolumes( origin, dx_scaled, current_volume.shape )

scaled_velocity_basis = TetraMesh()
scaled_velocity_basis.generate_mesh_from_numpy_array( np.ones((N,N,N),dtype=np.uint8), dx, max_cell_circumradius=esize*dx, max_facet_distance=esize*dx)
scaled_velocity_basis.coord = scaled_velocity_basis.coord/dx
scaled_velocity_basis.nodal_coordinates = scaled_velocity_basis.nodal_coordinates/dx
scaled_velocity_basis.expand_mesh_data()

scaled_velocity_basis.to_xdmf('dem_scaled_phantom_mesh')

scaled_velocity_basis.coefficents = np.zeros( scaled_velocity_basis.nodal_coordinates.shape )

scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi + scaled_flow_model.dx/2., scaled_flow_model.Yi, scaled_flow_model.Zi, label='xp' )
scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi - scaled_flow_model.dx/2., scaled_flow_model.Yi, scaled_flow_model.Zi, label='xn' )

scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi, scaled_flow_model.Yi + scaled_flow_model.dx/2., scaled_flow_model.Zi, label='yp' )
scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi, scaled_flow_model.Yi - scaled_flow_model.dx/2., scaled_flow_model.Zi, label='yn' )

scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi, scaled_flow_model.Yi, scaled_flow_model.Zi + scaled_flow_model.dx/2., label='zp' )
scaled_velocity_basis.precompute_basis( scaled_flow_model.Xi, scaled_flow_model.Yi, scaled_flow_model.Zi - scaled_flow_model.dx/2., label='zn' )

solver = FVMSolver( None, scaled_flow_model, projector, scaled_velocity_basis, None ) 
solver.x0 = np.zeros( scaled_velocity_basis.coefficents.shape )
solver.print_frequency = 1

res2_norm = np.inf

sinogram_residual_history = []
recon_times = []

#i=0
for current_time in integration_times:
    #i+=1
    #if i==2: raise

    current_time_indx = np.round( (current_time + dt)/dt_sampling ).astype(int)

    print(' ')
    print('############################## T A R G E T    T I M E    I N D E X  :  '+str((current_time + dt)/dt_sampling)+'  (t='+str(current_time)+') ############################################')
    verbose = True
    def dydt(t,y):

        # Interpolate measurements for target time; t interpolating with projections from the current state y.
        spline_interp.add_points( [t], [projector.forward_project( y )] , resolution = dt_sampling*1e-8  )
        dgdt = spline_interp( [t], derivative=1 )[0,:,:,:]
        spline_interp.reset()

        # Rescale units of the problem to give better numerical properties.
        dt_sampling_scaled = dt_sampling
        dgdt_scaled = dgdt*dt_sampling_scaled

        scaled_flow_model.fixate_density_field( y )

        # Nonlinear optimization step
        solver.density_field = y
        solver.second_member = -dgdt_scaled
        solver.iteration = 0
        solver.set_CFL_scaling_dimensions( (dx/dt_sampling_scaled)*dt/dx ) 

        #solver.set_cfl_constraint( solver.flow_model.dx, 1.0 )

        #solver.set_cfl_constraint( solver.flow_model.dx, 1.0, sensitivity_fix_threshold=0, verbose=True )

        #solver.check_jac(solver.x0.flatten())
        #solver.check_hessian(solver.x0.flatten())

        #solver.set_uniform_bounds( -(dx/dt_sampling_scaled)*dx/dt, (dx/dt_sampling_scaled)*dx/dt )

        # TODO: some more investioagtions of the convergence and the negative denseties
        # seems the hessian lsq solution is not perhaps the "right" solution to pick.
        # if CFL < 1 one would expect that no negative denseties would appear.... 

        solver.solve( maxiter=10, verbose=verbose, method='L-BFGS-B' )
        
        #solver.solve( maxiter=5, verbose=True, method='Builtin Newton' )

        solver.x0 = solver.x

        print(' ')

        # Optimal velocity solution 
        velocity_basis.coefficents = solver.x*(dx/dt_sampling_scaled) # add units

        #velocity_basis.coefficents = np.zeros( scaled_velocity_basis.coefficents.shape )
        #velocity_basis.coefficents[:,2] = 0.1

        fvm_propagator    = FiniteVolumes( origin, dx, y.shape )

        # If the CFL is too large it makes sense to restart the iteration with a smaller dt.
        maxCFL = fvm_propagator.max_CFL(velocity_basis, y, dt)
        assert maxCFL<1, 'The reconstructed velocities lead to unfesible local CFL='+str(maxCFL)+'>1'

        vertex_velocity = []
        negtags = ['xn','yn','zn']
        postags = ['xp','yp','zp']
        sx,sy,sz = y.shape
        for axis,(n,p) in enumerate( zip(negtags,postags) ):
            vn = scaled_velocity_basis.rendered_basis[n].T.dot( velocity_basis.coefficents[ :, axis ] ).reshape(sx-4,sy-4,sz-4)
            vp = scaled_velocity_basis.rendered_basis[p].T.dot( velocity_basis.coefficents[ :, axis ] ).reshape(sx-4,sy-4,sz-4)
            vertex_velocity.append( (vn, vp) )

        return fvm_propagator.get_dfbardt(vertex_velocity, y)

    old_TV = utils.get_total_variation(current_volume)
    old_mass = np.sum(current_volume)

    # Check that dt is strong stability preserving for a single forward euler step
    x0_copy = solver.x0.copy()
    verbose = False
    euler_volume = utils.euler_step(dydt, current_time, current_volume.copy(), dt)
    verbose = True
    solver.x0 = x0_copy
    euler_TV = utils.get_total_variation(euler_volume)

    #current_volume = utils.RK4_step(dydt, current_time, current_volume.copy(), dt)
    #current_volume = utils.RK3_step(dydt, current_time, current_volume.copy(), dt)
    current_volume = utils.TVD_RK3_step(dydt, current_time, current_volume.copy(), dt)
    #current_volume = utils.euler_step(dydt, current_time, current_volume.copy(), dt)
    current_TV = utils.get_total_variation(current_volume)
    current_mass = np.sum(current_volume)

    print('Euler TV diff = ', euler_TV - old_TV )
    print('TV diff = ', current_TV - old_TV )
    print('mass fraction diff = '+str(np.abs(current_mass - old_mass)/old_mass))
    print('min density  = ', np.min(current_volume) )

    # assert that the total vartiation is diminished, the masss preserved and the denseties positive
    #assert current_TV - old_TV <  0, 'TV diff = '+str(current_TV - old_TV)
    #assert np.abs(current_mass - old_mass)/old_mass <  1e-4, 'mass fraction diff = '+str(np.abs(current_mass - old_mass)/old_mass)
    #assert np.sum(current_volume<-1e-8)==0, str(current_volume[current_volume<-1e-8].flatten())

    if current_time_indx!=0 and np.abs( current_time_indx - ((current_time+dt)/dt_sampling) ) < (dt/2.):

        volumes.append(current_volume.copy())
        res1 = projector.forward_project(volumes[-2])  - sinograms[current_time_indx]
        res2 = projector.forward_project(volumes[-1])  - sinograms[current_time_indx]
        res1_norm = np.linalg.norm(res1)
        res2_norm = np.linalg.norm(res2)
        print( 'Sino res before: ',res1_norm)
        print( 'Sino res after : ',res2_norm)
        print('(Inherent ray model error: ', ray_model_error,')')
        sinogram_residual_history.append( res2_norm )
        recon_times.append( current_time )

        print('#####################################################################################################################')
        print(' ')

print('Final real space volume residual: ', np.linalg.norm(final_volume - current_volume))
print('to be compared to: ', np.linalg.norm(final_volume - initial_volume))

np.save( 'reconstructed_volumes/sinogram_residual_history.npy', np.array(sinogram_residual_history))
np.save( 'reconstructed_volumes/recon_times.npy', np.array(recon_times))

os.system('rm reconstructed_volumes/reconstructed_*')
for i,vol in enumerate(volumes):
    sino_res        = np.abs( projector.forward_project( vol ) - sinograms[i] )
    sino_res_images = np.vstack( [sino_res[:,i,:] for i in range(sino_res.shape[1])] )
    sino_res_images = np.expand_dims(sino_res_images, axis=2)
    np.save('reconstructed_volumes/reconstructed_volume_'+str(i)+'.npy', vol)
    utils.save_as_vtk_voxel_volume('reconstructed_volumes/reconstructed_volume_'+str(i), vol)
    utils.save_as_vtk_voxel_volume('reconstructed_volumes/reconstructed_sino_residual_'+str(i), sino_res_images)

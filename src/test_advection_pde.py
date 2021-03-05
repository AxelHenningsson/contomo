import numpy as np
import matplotlib.pyplot as plt
import phantom
import projected_advection_pde
import flow_model
import velocity_solver
import basis
import ray_model
import sinogram_interpolator

ph = phantom.Voxels.load( "/home/axel/Downloads/voxelphantom.ph" )
initial_volume = ph.inital_volume
dx = ph.detector_pixel_size

sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles()
fig,ax = plt.subplots(1,1)

vb = basis.TetraMesh.generate_mesh_from_numpy_array(   np.ones(initial_volume.shape, dtype=np.uint8), 
                                                       voxel_size=dx, 
                                                       max_cell_circumradius=7*dx, 
                                                       max_facet_distance=7*dx  )
print( 'Number of elements: ', vb.enod.shape[0] )

rm = ray_model.RayModel(  initial_volume.shape,
                           ph.number_of_detector_pixels,
                           ph.number_of_detector_pixels,
                           angles[0] )

fvm = flow_model.FlowModel( origin = (0,0,0), 
                                dx = dx, 
                                shape_field = initial_volume.shape )
fvm.fixate_velocity_basis( vb )

si = sinogram_interpolator.SinogramInterpolator( sample_times, sinograms, smoothness=0, order=2)

pde = advection_pde.ProjectedAdvectionPDE( flow_model = fvm, 
                                           ray_model = rm,
                                           sinogram_interpolator = si )

pde.propagate_from_initial_value( initial_volume,
                                  start_time=sample_times[0],
                                  stepsize=(sample_times[1]-sample_times[0]),
                                  number_of_timesteps=len(sample_times),
                                  velocity_recovery_iterations = 10,
                                  verbose = True,
                                  save_intermediate_volumes = True )

pde.save_volumes_as_vtk("/home/axel/Downloads/pde/pdevolumes")

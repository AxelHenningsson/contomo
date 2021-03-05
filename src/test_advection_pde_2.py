import numpy as np
import matplotlib.pyplot as plt
import phantom
import projected_advection_pde
import flow_model
import velocity_solver
import basis
import ray_model
import sinogram_interpolator
import utils

ph = phantom.Voxels.load( "/home/axel/Downloads/spherephantom.phantom" )
dx = ph.detector_pixel_size

sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles(labels="Dynamic scan")
reference_sinogram, reference_time, reference_angles = ph.get_sorted_sinograms_times_and_angles(labels="Reference scan")
final_sinogram, final_time, final_angles = ph.get_sorted_sinograms_times_and_angles(labels="Final scan")

si = sinogram_interpolator.SinogramInterpolator( sample_times, sinograms, smoothness=0, order=2)

initial_volume = utils.tomographic_density_field_reconstruction( angles = reference_angles, 
                                                                 sinogram = reference_sinogram, 
                                                                 maxiter = 50 )
utils.save_as_vtk_voxel_volume("/home/axel/Downloads/inital_volume", initial_volume)

final_volume = utils.tomographic_density_field_reconstruction( angles = final_angles, 
                                                                 sinogram = final_sinogram, 
                                                                 maxiter = 50 )
utils.save_as_vtk_voxel_volume("/home/axel/Downloads/final_volume", final_volume)

vb = basis.TetraMesh.generate_mesh_from_numpy_array(   np.ones(initial_volume.shape, dtype=np.uint8), 
                                                       voxel_size=dx, 
                                                       max_cell_circumradius=3*dx, 
                                                       max_facet_distance=3*dx  )
print( 'Number of elements: ', vb.enod.shape[0] )

rm = ray_model.RayModel(    initial_volume.shape,
                            ph.number_of_detector_pixels,
                            angles[0] )

fvm = flow_model.FlowModel( rho=initial_volume,
                            origin = (0,0,0), 
                            dx = dx, 
                            velocity_basis = vb )

fvm.fixate_velocity_basis( vb )

pde = projected_advection_pde.ProjectedAdvectionPDE( flow_model = fvm, 
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

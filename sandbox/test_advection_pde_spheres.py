import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from contomo import phantom
from contomo import projected_advection_pde
from contomo import flow_model
from contomo import velocity_solver
from contomo import basis
from contomo import ray_model
from contomo import sinogram_interpolator
from contomo import utils

ph = phantom.Spheres.load( "/home/axel/Downloads/spherephantom.phantom" )
dx = ph.detector_pixel_size

sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles(labels="Dynamic scan")

si = sinogram_interpolator.SinogramInterpolator( sample_times, sinograms, smoothness=0, order=2)


initial_volume = np.load("/home/axel/Downloads/intermediate_volume_0000.npy")
#max_cell_circumradius = np.max( 2*ph.get_radii(sample_times[0]) ) # Tetra elements of the same size as grains
max_cell_circumradius = 9*dx
max_facet_distance    = max_cell_circumradius
vb = basis.TetraMesh.generate_mesh_from_numpy_array(   np.ones(initial_volume.shape, dtype=np.uint8), 
                                                       voxel_size=dx, 
                                                       max_cell_circumradius=max_cell_circumradius, 
                                                       max_facet_distance=max_cell_circumradius  )

print( 'Number of elements: ', vb.enod.shape[0] )
equation_ration = len(sinograms.flatten()) / float(vb.enod.shape[0]*3)
assert equation_ration < 1.0 , " More velocity variables than pixels available. "

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

pde.save("/home/axel/Downloads/projected_advection_pde")

save_path = "/home/axel/Downloads/spheres"
if os.path.isdir(save_path):
    shutil.rmtree(save_path + "/volumes")
    shutil.rmtree(save_path + "/projections")
    shutil.rmtree(save_path + "/velocity")
    os.remove(save_path + "/times.npy")
    os.remove(save_path + "/meta_data.npy")
    os.rmdir(save_path)

pde.propagate_from_initial_value( initial_volume,
                                  start_time=sample_times[0],
                                  stepsize=(sample_times[1]-sample_times[0]),
                                  number_of_timesteps=len(sample_times),
                                  velocity_recovery_iterations = 10,
                                  verbose = True,
                                  save_path = "/home/axel/Downloads/spheres" )


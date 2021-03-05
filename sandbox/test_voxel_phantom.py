import numpy as np
import matplotlib.pyplot as plt
from contomo import phantom
from contomo import projected_advection_pde
from contomo import flow_model
from contomo import velocity_solver
from contomo import basis
from contomo import ray_model
from contomo import sinogram_interpolator
from contomo import utils

N  = 32

R1  = 9
cx1 = R1 + 4
cy1 = R1 + 4
cz1 = R1 + 4

grid_coords = np.linspace(0,N-1,N)
X,Y,Z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')

initial_volume = (  (( (X-cx1)**2 + (Y-cy1)**2 + (Z-cz1)**2 ) < R1**2) ).astype(np.float64)
initial_volume[initial_volume!=0]=1.0

dx                  = 0.38754
origin              = (0,0,0)
number_of_timesteps = 30
t0                  = 0
dt                   = 0.45*1e-3

# Speckle the phantom sphere (10% uniform variablity)
np.random.seed(1)
randvol = (0.5 - np.random.rand( initial_volume.shape[0], initial_volume.shape[1], initial_volume.shape[2] ))/5.
initial_volume[initial_volume>0] = initial_volume[initial_volume>0] + randvol[initial_volume>0]

velocity_basis = basis.TetraMesh.generate_mesh_from_numpy_array(  np.ones(initial_volume.shape, dtype=np.uint8), 
                                                                  voxel_size = dx, 
                                                                  max_cell_circumradius = 7*dx, 
                                                                  max_facet_distance = 7*dx  )
print( 'Number of elements: ', velocity_basis.enod.shape[0] )

ph = phantom.Voxels( detector_pixel_size = dx, 
                     number_of_detector_pixels = N, 
                     voxel_volume = initial_volume, 
                     integration_stepsize = dt, 
                     velocity_basis = velocity_basis )


def v(t):
    return 250*dx*np.ones((ph.flow_model.velocity_basis.coefficents.shape))

ph.set_velocity_coefficent_function( v )


ph.save("/home/axel/Downloads/voxelphantom.ph")
ph = phantom.Voxels.load("/home/axel/Downloads/voxelphantom.ph")

times = np.arange(t0, number_of_timesteps*dt, dt)

max_cfl = ph.get_max_cfl(times, dt, dx)
print("max_cfl : ", max_cfl)
assert max_cfl<1.0


for i,time in enumerate(times):
    ph.add_measurement( time=time, angles=np.array([0, 45, 90, 135]), label=str(time) )
    utils.save_as_vtk_voxel_volume( "/home/axel/Downloads/voxel_volume_"+str(i), ph.voxel_volume )
    print(time)

ph.save("/home/axel/Downloads/voxelphantom.ph")
ph = phantom.Voxels.load("/home/axel/Downloads/voxelphantom.ph")

fig,ax = plt.subplots(1,1)
for sino in ph.get_sinograms_by_time(times):
    ax.clear()
    ax.imshow(sino[:,1,:])
    plt.pause(0.05)
plt.show()




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


# construct a sphere phantom form a DEM simulation and check CFL
ref_angles = np.linspace(0.,180.,180)
static_angles = np.array([-45, 0, 45])

number_of_detector_pixels = 128
dem_timestepsize = 1e-5
dem_particle_radius = 0.004
dx = dem_particle_radius/(number_of_detector_pixels*2./32)
hypersampling = 3
translation  = (0, 0, 2.0*dem_particle_radius)

ph = phantom.Spheres.from_DEM_liggghts( "/home/axel/workspace/contomo_old/DEM_simulations/post/",
                                    pattern = ["silo_",".vtk"],
                                    timestepsize = dem_timestepsize, 
                                    translation  = translation,
                                    detector_pixel_size = dx, 
                                    number_of_detector_pixels = number_of_detector_pixels,
                                    hypersampling = hypersampling )

times = np.array(ph._dem_vtk_timeseries['sorted_timesteps'])*dem_timestepsize
t0 = np.min(times)
t1 = np.max(times)
times = np.linspace( t0, t1 - (t1-t0)*0.2, 240)
dt = times[1]-times[0]

max_cfl = ph.get_max_cfl(times, dt, dx)
print("max_cfl : ", max_cfl)
assert max_cfl<1.0

sino = ph.get_sinogram( time=times[0], angles=np.array([0,90]))
fig,ax = plt.subplots(1,2)
ax[0].imshow(sino[:,0,:])
ax[1].imshow(sino[:,1,:])

sino = ph.get_sinogram( time=times[-1], angles=np.array([0,90]))
fig,ax = plt.subplots(1,2)
ax[0].imshow(sino[:,0,:])
ax[1].imshow(sino[:,1,:])
plt.show()


d = number_of_detector_pixels/2.
ph.to_vtk( "/home/axel/Downloads/dem_spherephantom", times, scale=dx, translation=np.array([d,d,d]) )




# Reconstruct normal tomographies at selected times for comparison:
for i,time in enumerate(times): 
    if i==0 or i==len(times)-1 or i%120==0:
        sino = ph.get_sinogram( time=time, angles=ref_angles)
        intermediate_volume = utils.tomographic_density_field_reconstruction( angles   = ref_angles, 
                                                                              sinogram = sino, 
                                                                              maxiter  = 30 )
        np.save("/home/axel/Downloads/intermediate_volume_"+str(i).zfill(4)+".npy", intermediate_volume)
        utils.save_as_vtk_voxel_volume("/home/axel/Downloads/intermediate_volume_"+str(i).zfill(4), intermediate_volume)


# Add the measurments with sparse angles
for i,time in enumerate(times):
    ph.add_measurement( time=time, angles=static_angles, label="Dynamic scan" )
    print("time: ", time)

ph.save("/home/axel/Downloads/spherephantom")

# Plot the projections
sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles(labels="Dynamic scan")

fig,ax = plt.subplots(1,3)
for k,sino in enumerate(sinograms):
    for i,a in enumerate(ax):
        a.clear()
        a.imshow(sino[:,i,:], cmap='gray')
        a.set_xticks([])
        a.set_yticks([])
    plt.pause(0.1)
plt.show()


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
dem_timestepsize = 1e-5
dem_particle_radius = 0.004
dx = dem_particle_radius/2.
number_of_detector_pixels = 32
ph = phantom.Spheres.from_DEM_liggghts( "/home/axel/workspace/contomo_old/DEM_simulations/post/",
                                    pattern = ["silo_",".vtk"],
                                    timestepsize = dem_timestepsize, 
                                    translation  = (0, 0, 2.5*0.004),
                                    detector_pixel_size = dx, 
                                    number_of_detector_pixels = number_of_detector_pixels,
                                    hypersampling = 3 )

times = np.array(ph._dem_vtk_timeseries['sorted_timesteps'])*1e-5
t0 = np.min(times)
t1 = np.max(times)
times = np.linspace( t0, t1 - (t1-t0)*0.2, 120)
dt = times[1]-times[0]

max_cfl = ph.get_max_cfl(times, dt, dx)
print("max_cfl : ", max_cfl)
assert max_cfl<1.0

d = number_of_detector_pixels/2.
ph.to_vtk( "/home/axel/Downloads/dem_spherephantom", times, scale=dx, translation=np.array([d,d,d]) )


ph.add_measurement( time=times[0], angles=np.linspace(0.,180.,180), label="Reference scan" )
ph.add_measurement( time=times[-1], angles=np.linspace(0.,180.,180), label="Final scan" )

for i,time in enumerate(times):
    ph.add_measurement( time=time, angles=np.array([-45, 0, 45]), label="Dynamic scan" )
    print(time)

ph.save("/home/axel/Downloads/spherephantom")
ph = phantom.Spheres.load("/home/axel/Downloads/spherephantom.phantom")

fig,ax = plt.subplots(1,1)
sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles(labels="Dynamic scan")

for sino in sinograms:
    ax.clear()
    ax.imshow(sino[:,0,:])
    plt.pause(0.1)
plt.show()

ph.to_vtk("/home/axel/Downloads/spherephantom", times, scale=1.0, translation=np.array([0,0,0]))

raise

ph = phantom.Spheres( detector_pixel_size = 1.0, 
                      number_of_detector_pixels = 64,
                      hypersampling=3 )

ph.add_sphere(  lambda t: 2*t,
                lambda t: 3*t,
                lambda t: 4*t,
                radii=15.0, 
                density=2.21 )

ph.add_sphere(  lambda t: -2*t,
                lambda t: -3*t,
                lambda t: -4*t,
                radii=12.0,
                density=1.21 )

ph.save("/home/axel/Downloads/spherephantom")
ph = phantom.Spheres.load("/home/axel/Downloads/spherephantom.phantom")

times = np.arange(0, 0.1, 0.05)
for i,time in enumerate(times):
    ph.add_measurement( time=time, angles=np.array([-45, 0, 45]), labels="Dynamic scan") 
    print(time)

ph.save("/home/axel/Downloads/spherephantom")
ph = phantom.Spheres.load("/home/axel/Downloads/spherephantom.phantom")

fig,ax = plt.subplots(1,1)
sinograms, sample_times, angles = ph.get_sorted_sinograms_times_and_angles(labels="Dynamic scan")

for sino in sinograms:
    ax.clear()
    ax.imshow(sino[:,0,:])
    ax.set_title()
    plt.pause(0.1)
plt.show()
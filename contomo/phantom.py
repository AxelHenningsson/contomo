import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import copy
import glob
from . import utils
from . import ray_model
from . import flow_model

class DynamicPhantom(object):
    """Density field phantom defined in space and time (4d).

    This class defines phantoms that can be measured through a
    ray model (:obj:`RayModel`) at arbitrary time points. Phantoms
    meta data and measurment can be saved and loaded easily to and from 
    disc via pickling.

    Args:
        detector_pixel_size (:obj:`int`): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)

    Attributes:
        detector_pixel_size (:obj:`int`): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
        measurements (:obj:`list` of :obj:`dictionary`): List of recorded measurements as dictionaries
            with string keys

            ``"label"``     mapping to a custom string label for the measurment.

            ``"time"``      mapping to a float time at which the measurement was recorded.

            ``"angles"``    mapping to a :obj:`numpy array` of angles in degrees at which projections where recorded.

            ``"sinogram"``  mapping to a :obj:`numpy array` sinogram of shape=(n,k,n), where k is number of projections.

    """

    def __init__(self, detector_pixel_size, number_of_detector_pixels):
        self.detector_pixel_size = detector_pixel_size
        self.number_of_detector_pixels = number_of_detector_pixels
        self.measurements = []

    def add_measurement( self, time, angles, label ):
        """Add a measurement to the ``measurements`` list with a custom label.

        Args:
            label (:obj:`str`): Label of measurement.
            time (:obj:`float`): Time at which to evaluate the phantom density field.
            angles (:obj:`numpy array`): Projection angles in degrees.

        """
        self.measurements.append( 
                            { "label"    : label,
                              "time"     : time,
                              "angles"   : angles,
                              "sinogram" : self.get_sinogram(time, angles) }
                               )

    def get_sinogram(self, time, angles):
        """Compute and return sinogram for a given time and angles.

        Args:
            time (:obj:`float`): Time at which to evaluate the phantom density field.
            angles (:obj:`numpy array`): Projection angles in degrees.

        Returns:
            :obj:`numpy array` sinogram of `shape=(m,len(angles),n)`

        """
        raise NotImplementedError("The method get_sinogram() method must be implemented by the subclass")

    def get_measurements_by_label( self, labels ):
        """Return all measurments that have one of the input labels.

        Args:
            labels (:obj:`list` of str): Labels of measurements to return.

        Returns:
            All measurments that have one of the specififed labels.

        """
        measurements = [m for m in self.measurements if m["label"] in labels]
        return measurements

    def get_measurements_by_time( self, times ):
        """Return all measurments that occured at one of the input times.

        Args:
            times (:obj:`iterable` of float): Times of measurements to return.

        Returns:
            All measurments that occured at one of the input times.

        """
        measurements = [m for m in self.measurements if m["time"] in times]
        return measurements

    def get_sinograms_by_label( self, labels ):
        """Return all sinograms that have one of the input labels.

        Args:
            labels (:obj:`list` of str): Labels of sinograms to return.

        Returns:
            All sinograms that have one of the specififed labels.

        """
        sinograms = [m["sinogram"] for m in self.measurements if m["label"] in labels]
        return sinograms

    def get_sinograms_by_time( self, times ):
        """Return all sinograms that occured at one of the input times.

        Args:
            times (:obj:`iterable` of float): Times of sinograms to return.

        Returns:
            All sinograms that occured at one of the input times.

        """
        sinograms = [m["sinogram"] for m in self.measurements if m["time"] in times]
        return sinograms

    def get_sorted_sinograms_times_and_angles(self, labels):
        """Return, sorted by times, all measurments that have one of the input labels.

        Args:
            labels (:obj:`list` of str): Labels of measurements to return.

        Returns:
            sorted by time :obj:`numpy arrays` of all measurments that have one of the specified labels.

        """
        measurements = self.get_measurements_by_label(labels)
        times     =  [m["time"] for m in measurements]
        sinograms =  [m["sinogram"] for m in measurements]
        angles    =  [m["angles"] for m in measurements]
        indx      =  np.argsort(times)
        sorted_times = np.array(times)[indx]      
        sorted_sinograms = np.array(sinograms)[indx] 
        sorted_angles = np.array(angles)[indx] 
        if len(sorted_times)==1:
            sorted_times = sorted_times[0]
            sorted_sinograms = sorted_sinograms[0]
            sorted_angles = sorted_angles[0]
        return sorted_sinograms, sorted_times, sorted_angles

    def get_max_cfl( self, times, dt, dx ):
        """Return phantom maximum Courant–Friedrichs–Lewy number over a series of times.
        
        The Courant–Friedrichs–Lewy number, :math:`C`, is defined as

        .. math:: C = \\dfrac{\\Delta t ( \\| v_x \\|+\\| v_y \\|+\\| v_z \\| )}{\\Delta x}

        where :math:`\\Delta t` is a time increment,  :math:`\\Delta x` the size of the cells in the
        mesh and :math:`v_x`, :math:`v_y`, :math:`v_z` are 3d velocities. The Courant–Friedrichs–Lewy number
        must always be less than unity for the possibility of stability to exist. For more info 
        see this `wikipedia article`_:

        .. _wikipedia article: https://en.wikipedia.org/wiki/Courant-Friedrichs-Lewy_condition

        Args:
            times (:obj:`numpy array`): Times at which to evaluate the cfl number.
            dt (:obj:`float`): Size of timestep.
            dx (:obj:`float`): Size of cells.

        Returns:
            (:obj:`float`) maximum Courant–Friedrichs–Lewy number.

        """
        raise NotImplementedError("The method get_max_cfl() method must be implemented by the subclass")

    def save(self, file):
        """Save the phantom by pickling it to disc.

        Args:
            file (str): Absolute file path ending with the desired filename and no extensions.

        """
        with open(file+".phantom", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file):
        """Load a phantom from a pickled file. 

        Args:
            file (str): Absolute file path ending with the full filename. The extension
                should be ".phantom".

        """
        with open(file, 'rb') as output:
            return pickle.load(output)

class Voxels(DynamicPhantom):
    """Phantom constructed from a set of cubic voxels.

    This phantom is constructed from quanteties used in later reconstructions, these are the
    flow and ray model (:obj:`FlowModel`) (:obj:`RayModel`). It is usefull for defining
    arbitrary density fields and flows.

    Args:
        detector_pixel_size (int): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (int): Number of pixels comprising the detector side (square detector)
        voxel_volume (:obj:`numpy array`): Per cell density field values, shape=(N,N,N).
        integration_stepsize (float): timestepsize for propagating phantom in time numerically.
        velocity_basis (:obj:`Basis`): Velocity basis defining the time evolution flow of the phantom.

    Attributes:
        detector_pixel_size (int): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (int): Number of pixels comprising the detector side (square detector)
        voxel_volume (:obj:`numpy array`): Per cell density field values, shape=(N,N,N).
        integration_stepsize (float): timestepsize for propagating phantom in time numerically.
        velocity_basis (:obj:`Basis`): Velocity basis defining the time evolution flow of the phantom.
        flow_model (:obj:`FlowModel`): Flow model defining the approximate density temporal derivative.

    """

    def __init__( self, detector_pixel_size, number_of_detector_pixels, voxel_volume, integration_stepsize, velocity_basis ):
        super().__init__(detector_pixel_size, number_of_detector_pixels)
        self.time = 0.
        self.inital_volume = voxel_volume.copy()
        self.voxel_volume = voxel_volume
        self.integration_stepsize = integration_stepsize
        self.flow_model = flow_model.FlowModel( (0,0,0), detector_pixel_size, self.voxel_volume.shape )
        self.flow_model.fixate_velocity_basis( velocity_basis )        

    def get_sinogram(self, time, angles):
        """This method implements :meth:`get_sinogram()` of superclass :obj:`Phantom`.
        """
        self._propagate_volume( time )
        rm = ray_model.RayModel( self.voxel_volume.shape,
                                  self.number_of_detector_pixels,
                                  self.number_of_detector_pixels,
                                  angles )
        return rm.forward_project( self.voxel_volume )

    def set_velocity_coefficent_function(self, function):
        """Set a function for describing the temporal evolution of the velocity basis coefficents.
        
        The flow model velocity basis coefficents, ``flow_model.velocity_basis.coefficents``, are
        set by the provided callable function whenever velocities are needed.

        Args:
            function (:obj:`callable`): ``function(time)`` should return a :obj:`numpy array` with velocity 
                basis coefficents of shape ``flow_model.velocity_basis.coefficents.shape``. 
        """
        self.velocity_coefficent_function = function

    def _velocity_coefficent_function(self, t):
        """Stores the provided velocity coefficent function.
        """ 
        raise ValueError( "The velocity coefficents function must be set before called" )

    def _drhodt(self, time, rho):
        """Return temporal derivative of density field as modeled by the flow model.
        """
        self.flow_model.velocity_basis.coefficents = self._velocity_coefficent_function(time)
        self.flow_model.rho = rho
        return self.flow_model.get_temporal_density_derivatives()

    def _propagate_volume( self, time ):
        """Move voxel density volume forward in time by using a Runge-Kutta integration scheme.
        """
        assert self.time <= time, "Attempted backwards time propagation not supported"

        while( self.time + self.integration_stepsize < time ):
            self.voxel_volume = utils.TVD_RK3_step(self._drhodt, self.time, self.voxel_volume, self.integration_stepsize)
            self.time += self.integration_stepsize
        self.voxel_volume = utils.TVD_RK3_step(self._drhodt, self.time, self.voxel_volume, time - self.time )
        self.time = time

    def get_max_cfl( self, times, dt, dx ):
        """This method overrides :meth:`Phantom.get_max_cfl`.
        """
        return np.max([ np.sum( np.abs(self.velocity_coefficent_function(time)), axis=1 )*dt/dx for time in times ])

class Spheres(DynamicPhantom):
    """Phantom constructed from a set of analytical spherical objects.

    This phantom is constructed by describing a set of spheres and their analytical 
    trajectory through space and time. Measurments are created by considering the 
    analytical radon transform of a sphere.

    Args:
        detector_pixel_size (:obj:`int`): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
        hypersampling (:obj:`int`, optional): Number of hypersampling points per detector pixel to use when creating 
            analytical sinograms. For instance, ``hypersampling=3`` will use a 3 by 3 grid of points
            at each detector pixel at which the radon transform is evaluated. The resulting projected 
            value at the detector pixel is then set to the average of these nine evaluated points.
            Defaults to unity.

    Attributes:
        detector_pixel_size (:obj:`int`): Size of a single detector pixel (square pixels).
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
        hypersampling (:obj:`int`): Number of hypersampling points per detector pixel to use when creating 
            analytical sinograms. For instance, ``hypersampling=3`` will use a 3 by 3 grid of points
            at each detector pixel at which the radon transform is evaluated. The resulting projected 
            value at the detector pixel is then set to the average of these nine evaluated points.

    """

    def __init__( self, detector_pixel_size, number_of_detector_pixels, hypersampling=1 ):
        super().__init__(detector_pixel_size, number_of_detector_pixels)
        self.hypersampling = hypersampling
        self._spheres = []
        self._dem_vtk_timeseries = None

    def add_sphere(self, x, y, z, radii, density):
        """Add a spherical particle to the ensemble defining the phantom.

        Args:
            x (:obj:`callable` or :obj:`float`): Spatial x-coordinate of sphere. If callable, ``x(time)`` should return the
                x-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
            y (:obj:`callable` or :obj:`float`): Spatial y-coordinate of sphere. If callable, ``y(time)`` should return the
                y-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
            z (:obj:`callable` or :obj:`float`): Spatial z-coordinate of sphere. If callable, ``z(time)`` should return the
                z-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
            radii (:obj:`callable` or :obj:`float`): Radius of sphere. If callable, ``radii(time)`` should return the
               radii of the sphere at time ``time``. If :obj:`float` the sphere radii is assumed constant in time.
            density (:obj:`callable` or :obj:`float`): Homogenious density of sphere. If callable, ``radii(time)`` should 
                return the density of the sphere at time ``time``. If :obj:`float` the density is assumed constant in time.

        """
        self._spheres.append( _Sphere( x, y, z, radii, density )  )

    def _radon_on_grid(self, time, eta, xi, angles):
        """Compute radon transform of sphere ensemble over a grid.
        """
        sinogram = 0
        for s in self._spheres:
            sinogram += s.radon(time, eta, xi, angles)
        return sinogram

    def _radon(self, time, angles):
        """Construct detector grid and compute radon transform of sphere ensemble.

        We design detector xi and eta grids s.t the projection will be in "matplotlibstyle", i.e when 
        plotted, it will appear as if one is sitting on the sample and looking at the detector.
        With the xrays along the x-axis and rotations around the z-axis.

        """
        dp = self.hypersampling*self.number_of_detector_pixels

        xi, eta = np.mgrid[0.5:-0.5:dp*1j, 0.5:-0.5:dp*1j]
        xi  = self.detector_pixel_size * self.number_of_detector_pixels * xi
        eta = self.detector_pixel_size * self.number_of_detector_pixels * eta

        sinogram = self._radon_on_grid( time, eta, xi, angles )
        if self.hypersampling>1:
            sinogram = utils.downsample_sinogram(sinogram, self.hypersampling)
        return sinogram

    def get_coordinates( self, time ):
        """Get all sphere coordinates at specified time.

        Args:
            time (:obj:`float`): time at which to evaluate the sphere ensemble coordinates.
        
        Returns:
            :obj:`numpy array` of shape=(N,3) with per sphere x,y,z coordinates.
        """
        coordinates = [[ s.x(time), s.y(time), s.z(time)] for s in self._spheres]
        return np.asarray(coordinates)

    def get_densities( self, time ):
        """Get all sphere densities at specified time.

        Args:
            time (:obj:`float`): time at which to evaluate the sphere ensemble coordinates.
        
        Returns:
            :obj:`numpy array` of shape=(N,) with per sphere density.
        """
        densities = [ s.density(time) for s in self._spheres]
        return np.asarray(densities)

    def get_radii( self, time ):
        """Get all sphere radii at specified time.

        Args:
            time (:obj:`float`): time at which to evaluate the sphere ensemble coordinates.
        
        Returns:
            :obj:`numpy array` of shape=(N,) with per sphere radius.
        """
        radii = [ s.radius(time) for s in self._spheres]
        return np.asarray(radii)

    def get_max_cfl( self, times, dt, dx ):
        """This method overrides :meth:`Phantom.get_max_cfl`.
        """
        CFLs = []
        h = 1e-6
        for i,time in enumerate(times):
            if self._dem_vtk_timeseries is not None:
                self._set_sphere_ensemble_from_dem(time)
            coordinates =  self.get_coordinates(time)
            velocities  =  ( self.get_coordinates(time+h) - coordinates ) / h
            CFLs.append(  np.max( np.sum( np.abs(velocities) ,axis=1 ) )*dt / dx )
        return np.max(CFLs)

    @classmethod
    def from_DEM_liggghts(  cls, 
                            absolute_path_to_vtk_files,
                            pattern,
                            timestepsize, 
                            translation, 
                            detector_pixel_size, 
                            number_of_detector_pixels, 
                            hypersampling=1 ):
        """Define a sphere phantom from a series of Discrete Element liggghts simulation output files.

        This method allows instantiation of a sphere phantom by providing a directory in which
        Discrete Elements liggghts simulations have been saved in .vtk format. 
        For more information in liggghts DEM simulations see `this link`_:

        .. _this link: https://www.cfdem.com/media/DEM/docu/Manual.html

        To provide a continous phantom format, liner interpolation between provided DEM timesteps 
        is preformed when the phantom is called to be evaluated at an arbitrary timepoint.

        Args:
            absolute_path_to_vtk_files (:obj:`string`): Absolute path to directory in which the DEM liggghts
                .vtk output files are stored.
            pattern (:obj:`list` of :obj:`string`): Pattern to match in directory absolute_path_to_vtk_files. The files read are
                assumed to match the pattern ``pattern[0]+"[0-9]*"+pattern[1]``. For instance, the pattern 
                ``pattern=["my_filename_","some_file_ending.vtk"]`` will read files such as 
                ``my_filename_3810some_file_ending.vtk`` and infer the timestep as ``3810``.
            timestepsize (:obj:`float`): Time between two timesteps in the DEM simulation.
            translation (:obj:`numpy array`): Desired coordinate translation of sphere ensemble.
            detector_pixel_size (:obj:`int`): Size of a single detector pixel (square pixels).
            number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
            hypersampling (:obj:`int`, optional): Number of hypersampling points per detector pixel to use when creating 
                analytical sinograms. For instance, ``hypersampling=3`` will use a 3 by 3 grid of points
                at each detector pixel at which the radon transform is evaluated. The resulting projected 
                value at the detector pixel is then set to the average of these nine evaluated points.
                Defaults to unity.

        """

        dem_vtk_timeseries = {}
        for file in glob.glob(absolute_path_to_vtk_files+pattern[0]+"[0-9]*"+pattern[1]):
            timestep = int( file.split(pattern[0])[1].split(pattern[1])[0] )
            dem_vtk_timeseries[timestep] = file

        dem_vtk_timeseries['sorted_timesteps'] = np.sort(list(dem_vtk_timeseries.keys()))
        dem_vtk_timeseries['timestepsize'] = timestepsize
        dem_vtk_timeseries['translation'] = translation
        spheres = cls(detector_pixel_size, number_of_detector_pixels, hypersampling)
        spheres._dem_vtk_timeseries = dem_vtk_timeseries
        return spheres

    def _set_sphere_ensemble_from_dem( self, time ):
        """Set the ensemble according to DEM simulation  by linear interpolation between timesteps.
        """
        t0,t1,c0,c1,r0,r1,v0,v1 = self._get_particle_state_from_file(time)
        self._spheres = []
        for particle in range(c0.shape[0]):
            x,y,z = self._get_interp_functions(c0, c1, particle, t0, t1)
            self.add_sphere( copy.copy(x), copy.copy(y), copy.copy(z), r0[particle], density=1.0 )

    def _get_particle_state_from_file(self, time):
        """Extract particle states of neighbouring DEM timesteps enclosing the specified time.
        """
        dump_times = self._map_time_to_timestep(time)
        c0, r0, v0 = utils.load_vtk_point_data( self._dem_vtk_timeseries[dump_times[0]] )
        c1, r1, v1 = utils.load_vtk_point_data( self._dem_vtk_timeseries[dump_times[1]] )
        t0 = dump_times[0]*self._dem_vtk_timeseries['timestepsize']
        t1 = dump_times[1]*self._dem_vtk_timeseries['timestepsize']
        c0 += self._dem_vtk_timeseries['translation']
        c1 += self._dem_vtk_timeseries['translation']
        return t0,t1,c0,c1,r0,r1,v0,v1

    def _map_time_to_timestep(self, time):
        """Find neighbouring DEM timesteps enclosing a specified time.
        """
        dt = self._dem_vtk_timeseries['timestepsize']
        dump_timesteps = self._dem_vtk_timeseries['sorted_timesteps']
        dump_times = np.sort( dump_timesteps[ np.argsort( np.abs( dump_timesteps*dt - time ) )[0:2] ] )
        assert dump_times[1] <= dump_timesteps[-1] or  dump_timesteps[-1] <= dump_times[0]
        return dump_times

    def _get_interp_functions(self, c0, c1, particle, t0, t1):
        """Setup linear interpolation functions between two particle states.
        """
        interp_funcs = []
        for dim in range(3):
            c0d, c1d = c0[particle, dim], c1[particle, dim]
            interp_funcs.append( lambda t, t0=t0, t1=t1, c0d=c0d, c1d=c1d: c0d + ( (t - t0) / (t1 - t0) )*( c1d - c0d ) )
        return interp_funcs

    def get_sinogram(self, time, angles):
        """This method implements :meth:`get_sinogram()` of superclass :obj:`Phantom`.
        """
        if self._dem_vtk_timeseries is not None: 
            self._set_sphere_ensemble_from_dem( time )
        sinogram = self._radon( time, angles )
        return sinogram

    def to_vtk(self, file, times, scale=1.0, translation=np.array([0,0,0]) ):
        """Save analytical phantom to vtk, i.e paraview readable format.

        Args:
            file (:obj:`string`): Absolute file path ending with desired filename.
            times (:obj:`numpy array`): Timepoints at which to save the phantom particle ensemble.
            scale (:obj:`float`, optional): Scales the coordinates of the output. Defaults to unity. 
            translation (:obj:`numpy array`): Translates the output ensemble, shape=(3,). Defaults to zero.

        """
        h = 1e-6
        for i,time in enumerate(times):
            if self._dem_vtk_timeseries is not None:
                self._set_sphere_ensemble_from_dem(time)
            coordinates0 =  (self.get_coordinates(time)/scale) + translation
            coordinates1 =  (self.get_coordinates(time+h)/scale) + translation
            velocities  =  ( coordinates1 - coordinates0 ) / h
            radii       =  self.get_radii(time)/scale
            utils.save_as_vtk_particles(file+'_timestep_'+str(i)+'.vtk', coordinates0, velocities, radii )

class _Sphere( object ):
    """Represent a simple spherical object and it's radon transform.

    Args:
        x (:obj:`callable` or :obj:`float`): Spatial x-coordinate of sphere. If callable, ``x(time)`` should return the
            x-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
        y (:obj:`callable` or :obj:`float`): Spatial y-coordinate of sphere. If callable, ``y(time)`` should return the
            y-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
        z (:obj:`callable` or :obj:`float`): Spatial z-coordinate of sphere. If callable, ``z(time)`` should return the
            z-coordinate of the sphere at time ``time``. If :obj:`float` the sphere coordinate is assumed stationary.
        radii (:obj:`callable` or :obj:`float`): Radius of sphere. If callable, ``radii(time)`` should return the
            radii of the sphere at time ``time``. If :obj:`float` the sphere radii is assumed constant in time.
        density (:obj:`callable` or :obj:`float`): Homogenious density of sphere. If callable, ``radii(time)`` should 
            return the density of the sphere at time ``time``. If :obj:`float` the density is assumed constant in time.

    Attributes:
        x (:obj:`callable`): Spatial x-coordinate of sphere.
        y (:obj:`callable`): Spatial y-coordinate of sphere.
        z (:obj:`callable`): Spatial z-coordinate of sphere.
        radii (:obj:`callable`): Radius of sphere.
        density (:obj:`callable`): Homogenious density of sphere.

    """

    def __init__(self, x, y, z, radius, density ):
        self.x = self._make_callable( x )
        self.y = self._make_callable( y )
        self.z = self._make_callable( z )
        self.radius = self._make_callable( radius )
        self.density = self._make_callable( density )

    def _make_callable( self, variable ):
        """Convert input to callable if float, otherwise do nothing.
        """
        if callable(variable): 
            return variable
        else:
            return lambda time, variable=variable : variable

    def radon( self, time, eta, xi, angles ):
        """Compute radon tansform of a sphere at multiple detector locations (eta, xi).

        Args:
            time (:obj:`float`): time at which to record the radon transform.
            eta (:obj:`numpy array`): 2d detector grid eta coordinate.
            xi (:obj:`numpy array`): 2d detector grid xi coordinate.
            angles (:obj:`numpy array`): Projection angles in degrees.

        Returns:
            (:obj:`numpy array`): sinogram of ``shape=(xi.shape[0], len(angles), xi.shape[1])``
        """

        density  = self.density(time)
        position = np.array([[self.x(time)],[self.y(time)],[self.z(time)]])
        radius   = self.radius(time)

        # NOTE: we need the minus sign, since we are rotating the projection plane 
        # and not the spheres !
        c = np.cos( np.radians(-angles) )
        s = np.sin( np.radians(-angles) )
        Rtheta = np.zeros((len(angles),3,3))
        Rtheta[:,0,0] =  c
        Rtheta[:,0,1] = -s
        Rtheta[:,1,0] =  s
        Rtheta[:,1,1] =  c
        Rtheta[:,2,2] =  1

        nhat = Rtheta.dot( np.array([1,0,0]) ).T
        P = position - position.T.dot(nhat)*nhat
        p_eta = np.sum( P*Rtheta.dot( np.array([0,1,0]) ).T, axis=0 )
        p_xi  = P.T.dot( np.array([0,0,1]) )
        sinogram = np.zeros((eta.shape[0],len(angles),eta.shape[1]))

        for i in range( len( angles ) ):
            sinogram[:,i,:] =  2*density*np.real( np.sqrt( radius**2 - (eta - p_eta[i])**2  - (xi - p_xi[i])**2 + 0j  ) ) 

        return sinogram
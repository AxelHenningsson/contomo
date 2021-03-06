
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import os
from contomo import utils
from contomo import velocity_solver

class ProjectedAdvectionPDE(object):
    """Advection partial differential equation (PDE) projected into sinogram space.

    This object defines the partial differential equation to be solved as an 
    inital value problem. Mathematically, the system of equations to solve are

    .. math::
        \\dfrac{\\partial \\rho}{\\partial t} = \\mathcal{F}[ \\rho, v ] \\quad\\quad (1)
    .. math::
        \\mathcal{P}[ \\mathcal{F}[ \\rho, v ] ] = \\dfrac{\\partial g}{\\partial t} \\quad\\quad (2)

    where :math:`\\partial \\rho/\\partial t` is the temporal derivative of a density field :math:`\\rho(x,t)`, 
    :math:`\\mathcal{P}[\\cdot]` defines a projection operator, and :math:`\\partial g/\\partial t` is the temporal
    derivative of some measured projections, :math:`g(s,t)`. Here :math:`x` is a real space coordinate and :math:`s` a
    sinogram space coordinate, :math:`t` denotes time. The operator :math:`\\mathcal{F}[\\cdot]` is a
    flow model approximation to :math:`\\partial \\rho/\\partial t`, driven by and underlying velocity field
    :math:`v(x,t)`, which in turn is decomposed on a finite basis.

    Given the inital value of the field :math:`\\rho(x,t=0)=\\rho_0` and a set of measured
    projections :math:`g(s, t)`, this class progagates equation (1) in time using some
    provided ray model for :math:`\\mathcal{P}[\\cdot]` and flow model :math:`\\mathcal{F}[\\cdot]`.

    Args:
        flow_model (:obj:`FlowModel`): Object defining the derivatives of density w.r.t time.
        ray_model (:obj:`RayModel`): Object defining the transformation from real space to sinogram space.
        sinogram_interpolator (:obj:`RayModel`): Object defining the continous derivatives in time of measured sinograms.

    Attributes:
        flow_model (:obj:`FlowModel`): Object defining the derivatives of density w.r.t time.
        ray_model (:obj:`RayModel`): Object defining the transformation from real space to sinogram space.
        sinogram_interpolator (:obj:`SinogramInterpolator`): Object defining the continous derivatives in time of measured sinograms.

    """

    def __init__( self, 
                  flow_model, 
                  ray_model,
                  sinogram_interpolator ):
        self.flow_model            =  flow_model
        self.ray_model             =  ray_model
        self.sinogram_interpolator =  sinogram_interpolator

    def get_density_derivative(self, time, rho):
        """Compute right hand side of advection PDE (1) by solving for v(x,t) through (2).

        The procedure to approximate the density field temporal derivative can be described
        in two steps. First the velocities are recovered by solving, P[ F[ rho, v ] ] = dgdt,
        next, F[ rho, v ] can be computed based on the retrived v.

        Args:
            time (float): time.
            rho (:obj:`numpy array`): real space density field.

        Returns:
            drhodt (:obj:`numpy array`): density field temporal derivative at time ``time``, ``shape=rho.shape``.

        """

        # self.sinogram_interpolator.soft_spline_reset()
        # self.sinogram_interpolator.add_sinograms( [time], 
        #                                           [self.ray_model.forward_project( rho )], 
        #                                           resolution = self.stepsize/1000.,
        #                                           update_splines=False )
        # bc_indx = np.argmin( np.abs( self.sinogram_interpolator.sample_times - time  ) ) + 1
        # self.sinogram_interpolator.set_bc_index(bc_indx)
        # self.sinogram_interpolator._set_splines(local=time)

        ##DEBUG:
        if 0:
            t0 = np.max([0,time - 2.1*self.stepsize])
            fig,ax = self.sinogram_interpolator.show_fit( [32], [0], [32], np.arange(t0, time + 2.1*self.stepsize, self.stepsize/30.) )
            plt.show()
        ##

        dgdt = self.sinogram_interpolator( time, derivative=1 )[:,:,:]
        
        self._velocity_solver.flow_model.fixate_density_field( rho )

        self._velocity_solver.second_member = dgdt

        # TODO: Probably slightly faster to have the inital guess only be redefined at the measurement points
        # and not for each RK integration point ...
        if self._velocity_solver.optimal_coefficents is not None:
            inital_guess = self._velocity_solver.optimal_coefficents 
        else:
            inital_guess = np.zeros( self.flow_model.velocity_basis.coefficents.shape )

        self._velocity_solver.solve( dgdt, 
                                    inital_guess,
                                    maxiter=self.maxiter, 
                                    verbose=self.verbose, 
                                    print_frequency=1 )

        self._velocity_solver.initial_guess_coefficents = self._velocity_solver.optimal_coefficents
        self.flow_model.velocity_basis.coefficents = self._velocity_solver.optimal_coefficents
        drhodt = self.flow_model.get_temporal_density_derivatives()

        return drhodt

    def propagate_from_initial_value( self, 
                                      initial_volume,
                                      start_time,
                                      end_time,
                                      stepsize,
                                      velocity_recovery_iterations = 10,
                                      reinterpolation="mutated moving bc",
                                      verbose = True,
                                      save_path = None ):
        """Propagate the target advection equation in time.

        Args:
            initial_volume (:obj:`numpy array`): Density field at starting time.
            start_time (:obj:`float`): Time at which the initial density field exists.
            end_time (:obj:`float`): Time at which to stop time integration.
            stepsize (:obj:`float`): Duration of time between two integration steps. 
            number_of_timesteps (:obj:`int`): Number of integration steps to execute.
            velocity_recovery_iterations (:obj:`numpy array`): Number of allowed iterations for recovering velocities 
                in the projected sub-problem.
            reinterpolation (:obj:`str`): One of ```mutated moving bc```, ```mutated static bc``` or
                ```original static bc```. Defines the strategy for computing projected derivatives during 
                RK time integration Defaults to ```mutated moving bc```.
            verbose (:obj:`bool`, optional): Print progress and convergence metrics. Defaults to True.
            save_path (:obj:`string`, optional): Save reconstructed density fields and sinograms to the given
                absolute path ending with desired folder name. Defaults to None.

        """
        self.stepsize = stepsize

        self._velocity_solver = velocity_solver.VelocitySolver( self.flow_model, 
                                                                self.ray_model,
                                                                dt = stepsize )
        self._velocity_solver.x0 = np.zeros(self._velocity_solver.flow_model.velocity_basis.coefficents.shape)

        self.maxiter = velocity_recovery_iterations
        self.verbose = verbose

        current_time = start_time
        self.current_volume = initial_volume.copy()

        if save_path is not None:
            self._instantiate_save_folders( save_path )

        # Compute inherent erros due to ray model etc..
        interpolated_sinogram  = self.sinogram_interpolator( start_time, original=True )[:,:,:]
        reconstructed_sinogram = self.ray_model.forward_project( initial_volume )
        residual = interpolated_sinogram - reconstructed_sinogram
        iL2 =  np.linalg.norm( interpolated_sinogram - reconstructed_sinogram )
        iMaxAbs = np.max(np.abs(residual))
        iRMSE = np.sqrt( np.sum( residual**2 ) / len(residual.flatten()) )
        iMAE = np.sum( np.abs(residual)  / len(residual.flatten()) )

        if verbose:
            print("##############################################################################")
            print(" R A Y    M O D E L    E R R O R ")
            print( "L2 Norm: ", iL2 )
            print( "Max.Abs: ", iMaxAbs )
            print( "RMSE: ", iRMSE )
            print( "MAE: ", iMAE )

            print("##############################################################################")
            print(" ")
            print("Starting propagation of density volume in time")

        print(end_time, start_time, stepsize)
        number_of_timesteps = int( np.ceil( (end_time - start_time)/stepsize ) )
        for step in range(number_of_timesteps - 2): # we cannot integrate for the last step as the spline ends

            if self.verbose:
                print(" ")
                print("@ time = ", current_time, "s ,i.e, @timestep = ", step, "  integrating for timestep = ", step+1)
                print("The total number of temporal data points is ", number_of_timesteps)

            # Reinterpolation strategy
            if reinterpolation=="original static bc":
                pass
            elif reinterpolation=="mutated moving bc":
                self.sinogram_interpolator.soft_spline_reset()
                self.sinogram_interpolator.add_sinograms( [current_time], 
                                                        [self.ray_model.forward_project( self.current_volume )], 
                                                        resolution = self.stepsize/1000.,
                                                        update_splines=False )
                bc_indx = np.argmin( np.abs( self.sinogram_interpolator.sample_times - current_time  ) ) + 1
                self.sinogram_interpolator.set_bc_index(bc_indx)
                self.sinogram_interpolator._set_splines(local=current_time)
            elif reinterpolation=="mutated static bc":
                self.sinogram_interpolator.soft_spline_reset()
                self.sinogram_interpolator.add_sinograms( [current_time], 
                                                        [self.ray_model.forward_project( self.current_volume )], 
                                                        resolution = self.stepsize/1000.,
                                                        update_splines=False )
                self.sinogram_interpolator.set_bc_index(0)
                self.sinogram_interpolator._set_splines()

            previous_volume = self.current_volume.copy()

            adapted_stepsize = np.min([ stepsize, end_time - current_time] ) # To handle the last step

            self.current_volume = utils.TVD_RK3_step( self.get_density_derivative, 
                                                      current_time, 
                                                      self.current_volume.copy(), 
                                                      adapted_stepsize )

            current_time += adapted_stepsize

            if verbose:
                original_sinogram  = self.sinogram_interpolator( current_time, original=True )[:,:,:]
                starting_reconstructed_sinogram = self.ray_model.forward_project( previous_volume )
                reconstructed_sinogram = self.ray_model.forward_project( self.current_volume )
                residual = reconstructed_sinogram - original_sinogram
                print("Original default sinogram L2 error to current time (no motion): ", np.linalg.norm( original_sinogram - starting_reconstructed_sinogram) )
                print("Resulting sinogram L2 error to current time (by applying the motion): ", np.linalg.norm( original_sinogram - reconstructed_sinogram), "  (inherent L2", iL2, ")" )
                print("Sinogram residual Max.Abs error: ", np.max(np.abs(residual)), "  (inherent Max.Abs", iMaxAbs, ")" )
                print("Sinogram residual RMSE error: ", np.sqrt( np.sum( residual**2 ) / len(residual.flatten()) ), "  (inherent RMSE", iRMSE, ")" )
                print("Sinogram residual MAE error: ", np.sum( np.abs(residual)  / len(residual.flatten()) ), "  (inherent MAE", iMAE, ")" )

            if save_path is not None:
                self._save_integration_step( save_path, self.current_volume, current_time, step )


    def _instantiate_save_folders(self, save_path):
        """Setup a folder structure to save reconstruction progress.

        """
        os.mkdir(save_path)
        for folder in ["volumes", "projections", "velocity"]:
            if folder not in os.listdir(save_path):
                os.mkdir(save_path + "/" + folder)

        np.save(save_path+"/times", np.array([]))

        meta_data = { "detector dimension"                 : self.ray_model.number_of_detector_pixels,
                      "angles"                             : self.ray_model.angles,
                      "number of velocity basis functions" : self.flow_model.velocity_basis.coefficents.shape[0],
                      "Finite volume cell size"            : self.flow_model.dx }

        np.save(save_path+"/meta_data.npy", meta_data)

    def _save_integration_step(self, save_path, current_volume, current_time, step ):
        """Save volume and projections to disc in save_path directory.

        """
        interpolated_sinogram  = self.sinogram_interpolator( current_time, original=True )[:,:,:]
        reconstructed_sinogram = self.ray_model.forward_project( current_volume )
        times = np.load(save_path+"/times.npy")

        np.save( save_path + "/volumes/volume_"+str(step).zfill(4)+".npy", current_volume )

        utils.save_as_vtk_voxel_volume(save_path + "/volumes/volume_"+str(step).zfill(4) , current_volume, 3*[self.flow_model.dx], self.flow_model.origin)

        np.save( save_path + "/projections/reconstructed_sinogram_"+str(step).zfill(4)+".npy", reconstructed_sinogram )
        np.save( save_path + "/projections/interpolated_sinogram_"+str(step).zfill(4)+".npy", interpolated_sinogram )

        np.save( save_path + "/times", np.concatenate( [times, np.array([current_time]) ]) )

        np.save(save_path + "/velocity/basis_coefficents_"+str(step).zfill(4)+".npy", self._velocity_solver.optimal_coefficents)

    def save(self, file):
        """Save the projected advection pde problem by pickling it to disc.

        Args:
            file (str): Absolute file path ending with the desired filename and no extensions.

        """
        with open(file+".papde", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file):
        """Load a projected advection pde problem from a pickled file. 

        Args:
            file (str): Absolute file path ending with the full filename. The extension
                should be ".papde".

        """
        with open(file, 'rb') as output:
            return pickle.load(output)

import numpy as np
import matplotlib.pyplot as plt
import utils
import velocity_solver
import utils

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
        intermediate_volumes (:obj:`list`): Density fields recovered at integration timepoints, only filled if 
            the boolean save_intermediate_volumes is True in propagate_from_initial_value()

    """

    def __init__( self, 
                  flow_model, 
                  ray_model,
                  sinogram_interpolator ):
        self.flow_model            =  flow_model
        self.ray_model             =  ray_model
        self.sinogram_interpolator =  sinogram_interpolator
        import copy
        self.s2 = copy.deepcopy(sinogram_interpolator)
        self.intermediate_volumes  = []

    def get_interpolated_sinogram_derivative(self, time, rho):
        """Compute sinogram temporal derivatives considering current density state and measured data.

        To approximate the right hand side of equation (2), when P[rho] does not match the
        interpolated values of the data, the projection of the current density field, y, is added
        to the interpolated sinogram series and reinterpolation is executed as defined by the 
        sinogram_interpolator object. This approximation ensures that time integration will consistently
        strive to return to the original interpolation path defined by the measured data. 

        Args:
            time (float): time.
            rho (:obj:`numpy array`): real space density field.

        Returns:
            dgdt (:obj:`numpy array`): sinogram temporal derivative at time=t, shape=(det_pix, num_proj, det_pix).

        """
        self.sinogram_interpolator.add_sinograms( [time], [self.ray_model.forward_project( rho )] )
        dgdt = self.sinogram_interpolator( [time], derivative=1 )[0,:,:,:]

        # TODO: Is it good or bad to reset the splines? If not reset, then the previous
        # errors might serve to build up the correction more strongly. Some results
        # do indicate that this is a more consistent interpretation of the off line
        # derivatives.

        # self.sinogram_interpolator.reset() 

        return dgdt

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
        dgdt = self.get_interpolated_sinogram_derivative( time, rho )
        self.velocity_solver.flow_model.fixate_density_field( rho )
        self.velocity_solver.second_member = dgdt

        if self.velocity_solver.optimal_coefficents is not None:
            inital_guess = self.velocity_solver.optimal_coefficents 
        else:
            inital_guess = np.zeros( self.flow_model.velocity_basis.coefficents.shape )

        self.velocity_solver.solve( dgdt, 
                                    inital_guess,
                                    maxiter=self.maxiter, 
                                    verbose=self.verbose, 
                                    print_frequency=1 )

        self.velocity_solver.initial_guess_coefficents = self.velocity_solver.optimal_coefficents
        self.flow_model.velocity_basis.coefficents = self.velocity_solver.optimal_coefficents
        drhodt = self.flow_model.get_temporal_density_derivatives()

        return drhodt

    def propagate_from_initial_value( self, 
                                      initial_volume,
                                      start_time,
                                      stepsize,
                                      number_of_timesteps,
                                      velocity_recovery_iterations = 10,
                                      verbose = True,
                                      save_intermediate_volumes = True ):
        """Propagate the target advection equation in time.

        Args:
            initial_volume (:obj:`numpy array`): Density field at starting time.
            start_time (float): Time at which the initial density field exists.
            stepsize (float): Duration of time between two integration steps. 
            number_of_timesteps (int): Number of integration steps to execute.
            velocity_recovery_iterations (:obj:`numpy array`): Number of allowed iterations for recovering velocities 
                in the projected sub-problem.
            verbose (:obj:`bool`, optional): Print progress and convergence metrics. Defaults to True.
            save_intermediate_volumes (:obj:`bool`, optional): Save all reconstructed density fields. Defaults to True.

        """
        if verbose:
            print('##############################################################################')
            print(' R A Y    M O D E L    E R R O R ')
            ray_model_error =  np.linalg.norm( self.ray_model.forward_project( initial_volume ) - self.sinogram_interpolator([0])[0])
            print( ray_model_error )
            print('##############################################################################')

        self.velocity_solver = velocity_solver.VelocitySolver( self.flow_model, 
                                                               self.ray_model,
                                                               dt = stepsize )
        self.velocity_solver.x0 = np.zeros(self.velocity_solver.flow_model.velocity_basis.coefficents.shape)


        self.maxiter = velocity_recovery_iterations
        self.verbose = verbose

        current_time = start_time
        self.current_volume = initial_volume

        if save_intermediate_volumes:
            self.intermediate_volumes.append( self.current_volume )

        if save_intermediate_volumes:
            self.intermediate_volumes.append( self.current_volume )

        if self.verbose:
            print("Starting propagation of density volume in time")

        for step in range(number_of_timesteps):
            if self.verbose:
                print(" ")
                print("time = ", current_time, "s   timestep = ", step, "  out of total ", number_of_timesteps, " steps")

            self.current_volume = utils.TVD_RK3_step( self.get_density_derivative, 
                                                      current_time, 
                                                      self.current_volume.copy(), 
                                                      stepsize )
            current_time += stepsize

            if verbose:
                interpolated_sinogram  = self.sinogram_interpolator( [current_time], original=True )[0,:,:,:]
                if save_intermediate_volumes:
                    starting_reconstructed_sinogram = self.ray_model.forward_project( self.intermediate_volumes[-1] )
                    print("Original Siogram residual: ", np.linalg.norm( interpolated_sinogram - starting_reconstructed_sinogram) )
                reconstructed_sinogram = self.ray_model.forward_project( self.current_volume )
                print("Siogram residual: ", np.linalg.norm( interpolated_sinogram - reconstructed_sinogram) )

            if save_intermediate_volumes:
                self.intermediate_volumes.append( self.current_volume )

    def save_volumes_as_vtk(self, file):
        """Write all saved density volume to disc in a paraview readable format.

        Args:
            file (str): abosulte path, ending with filename wihtout extensions.

        """
        for i in range(len(self.intermediate_volumes)):
            utils.save_as_vtk_voxel_volume(file+"_"+str(i) , self.intermediate_volumes[i])
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class VelocitySolver(object):
    """Iterative solver of the projected advection PDE subproblem in which velocity is recovered.

    The system of equations to solve for a fixed time, :math:`t=t_n`, is

    .. math::
        \\mathcal{P}[ \\mathcal{F}[ \\rho, v ] ] = \\dfrac{\\partial g}{\\partial t}

    where :math:`\\rho(x,t)` is a density field, :math:`\\mathcal{P}[\\cdot]` is a projection operator defined
    by a ray model, :math:`\\partial g/\\partial t` are the temporal derivatives of some measured projections, :math:`g(s,t)`. 
    Here :math:`x` is a real space coordinate and :math:`s` a  sinogram space coordinate, :math:`t` denotes time. 
    The operator :math:`\\mathcal{F}[\\cdot]` is a flow model approximation to :math:`\\partial \\rho/\\partial t`, 
    driven by and underlying velocity field :math:`v(x,t)`, which in turn is decomposed on a finite basis. This velocity 
    decomposition is parameterised by some coefficents, :math:`\\alpha_{xi},\\alpha_{yi},\\alpha_{zi}` which are the 
    sought values of the problem.

    The problem is solved practically by minimising the least squares scalar cost function :math:`C`

    .. math::
        C = \\int_s \\bigg(\\mathcal{P}[ \\mathcal{F}[ \\rho, v ] ] - \\dfrac{\\partial g}{\\partial t}\\bigg)^2

    The used solver is the L-BFGS-B as implemented in the :obj:`scipy.optimize.minimize`. For further
    details on the solver `see the scipy docs`_.
    
    .. _`see the scipy docs`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

    Note that it is possible to override the :obj:`solve` method if another method is preffered.

    Args:
        flow_model (:obj:`FlowModel`): Object defining the derivatives of density w.r.t time.
        ray_model (:obj:`RayModel`): Object defining the transformation from real space to sinogram space.
        dt (:obj:`float`): Intended timestep of the integrator. This is used to check CFL numbers and
            defines a feasible domain for the recovered velocity.

    Attributes:
        flow_model (:obj:`FlowModel`): Object defining the derivatives of density w.r.t time.
        ray_model (:obj:`RayModel`): Object defining the transformation from real space to sinogram space.
        dt (:obj:`float`): Intended timestep of the integrator. This is used to check CFL numbers and
            defines a feasible domain for the recovered velocity.
        optimal_coefficents (:obj:`numpy array`): Optimal velocity basis coefficents in a LSQ sense.
            ``shape=flow_model.velocity_basis.coefficents.shape``

    Note:
        Current implementation handles CFL violations in the solution by rescaling the final optimal
        coefficents, :math:`\\alpha_{xi},\\alpha_{yi},\\alpha_{zi}`, such that at any one basis location
        the CFL is no longer violated. The return to the feasible domain is executed along lines defined
        by the vectors :math:`\\alpha_{xi},\\alpha_{yi},\\alpha_{zi}`, i.e, it is not the closest return
        which would be the projection unto the :math:`L1` ball. Note also that if the basis does not have 
        maximum CFL at the basis nodal locations, this mehod is invalid.

    """

    def __init__(self, flow_model, ray_model, dt ):
        self.flow_model = flow_model
        self.ray_model  = ray_model
        self.dt = dt
        self.optimal_coefficents = None
        self._second_member = None

    def cost(self, x):
        """Evaluate the cost function for a given state.

        Args:
            x (:obj:`numpy array`): Velocity basis coefficents in flattened format, i.e
                ``shape=flow_model.velocity_basis.coefficents.flatten().shape``

        Returns:
            :obj:`float` cost

        """
        self.flow_model.velocity_basis.coefficents = x.reshape( self.flow_model.velocity_basis.coefficents.shape )
        drhodt = self.flow_model.get_temporal_density_derivatives(get_vertex_velocity_derivatives=False)
        res = self.ray_model.forward_project( drhodt ) - self._second_member
        return np.linalg.norm(res)**2

    def cost_and_gradient(self, x):
        """Evaluate both cost and gradient of cost for a given state.

        The sought derivatives are

        .. math::
            \\dfrac{ \\partial C }{ \\partial \\alpha_{xi} },\quad
            \\dfrac{ \\partial C }{ \\partial \\alpha_{yi} },\quad
            \\dfrac{ \\partial C }{ \\partial \\alpha_{zi} }

        where :math:`C` is the cost function and :math:`\\alpha` the velocity basis coefficents.

        Args:
            x (:obj:`numpy array`): Velocity basis coefficents in flattened format, i.e
                ``shape=flow_model.velocity_basis.coefficents.flatten().shape``
        
        Returns:
            :obj:`float` cost and :obj:`numpy array` gradient matching the input, ``shape=x.shape``.

        """
        self.flow_model.velocity_basis.coefficents = x.reshape( self.flow_model.velocity_basis.coefficents.shape )
        drhodt, ddrhodtdv = self.flow_model.get_temporal_density_derivatives(get_vertex_velocity_derivatives=True)

        res = self.ray_model.forward_project( drhodt ) - self._second_member
        cost       = np.linalg.norm(res)**2

        gradient = np.zeros(self.flow_model.velocity_basis.coefficents.shape)
        bpres = self.ray_model.backward_project(res[:,:,:])

        for axis in range(gradient.shape[1]):

            product_field_n = (bpres[2:-2,2:-2,2:-2] * ddrhodtdv[2:-2, 2:-2, 2:-2, axis, 0])
            product_field_p = (bpres[2:-2,2:-2,2:-2] * ddrhodtdv[2:-2, 2:-2, 2:-2, axis, 1])

            phin = self.flow_model.vertex_velocity_matrices[axis][0]
            phip = self.flow_model.vertex_velocity_matrices[axis][1]

            gradient[:,axis] = 2 * ( phip.dot( product_field_p.flatten() ) - phin.dot( product_field_n.flatten() ) )

        gradient = gradient.flatten()

        return cost, gradient

    def check_gradient( self, x ):
        """Plot and print numerical and analytical gradients.

        Args:
            x (:obj:`numpy array`): Velocity basis coefficents in flattened format, i.e
                ``shape=flow_model.velocity_basis.coefficents.flatten().shape``
        
        """
        _, analytical_gradient = self.cost_and_gradient(x)
        numerical_gradient  = self._get_numerical_gradient(x)
        print()
        print('numerical_gradient'  , numerical_gradient )
        print('analytical_gradient' , analytical_gradient)
        print('')
        print(x)
        plt.plot(analytical_gradient, 'ro', label='analytical')
        plt.plot(numerical_gradient, 'ko', label='numerical')
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_numerical_gradient(self, x):
        """Compute numerical gradient by finite difference.

        The used approximation is

        .. math::
            \\dfrac{d f}{d x}  = -\\dfrac{f(x+2h)-4f(x+h)+3f(x)}{2h}

        Args:
            x (:obj:`numpy array`): Velocity basis coefficents in flattened format, i.e
                ``shape=flow_model.velocity_basis.coefficents.flatten().shape``

        Returns:
            :obj:`numpy array` numerical approximation of gradient matching the input, ``shape=x.shape``.

        """
        numerical_gradient = np.zeros((len(x),))
        for i in range(len(x)):
            h  = np.zeros((len(x),))
            h[i] += 1e-5
            f1,_ = self.cost_and_gradient( x        )
            f2,_ = self.cost_and_gradient( x  +  h  )
            f3,_ = self.cost_and_gradient( x  + 2*h )
            numerical_gradient[i] = -(f3 - 4*f2 + 3*f1)/(2*np.max(h))
        return numerical_gradient

    def _get_nodal_cfl(self, x):
        nodal_velocity = x.reshape(self.flow_model.velocity_basis.coefficents.shape)
        nodal_cfl = np.sum( np.abs(nodal_velocity), axis=1 ) * self.dt / self.flow_model.dx 
        return nodal_cfl

    def _scale_to_cfl_constraint(self, x):
        """Scale the solution vector x such that the CFL<1.0.

        Each nodal basis velocity vector magnitude is scaled until 
        CFL==0.995 the direction of flow is thus preserved in the nodes.

        """
        nodal_cfl = self._get_nodal_cfl( x)
        cfl_mask = ( nodal_cfl >= 1.0 )
        nodal_velocity = x.reshape(self.flow_model.velocity_basis.coefficents.shape)
        nodal_velocity[cfl_mask,:] *= 0.995 * (1/np.expand_dims(nodal_cfl[cfl_mask],axis=1))
        assert np.max( self._get_nodal_cfl( x ) ) < 1.0, np.max( self._get_nodal_cfl( x ) )
        x = nodal_velocity.flatten()
        return x

    def solve(self, dgdt, inital_guess, maxiter, verbose=True, print_frequency=1):
        """Minimise cost function by L-BFGS-B.

        This method runs the :obj:`scipy.opimize.minimize` L-BFGS-B code
        and sets the ``optimal_coefficents`` attribute to the optimal solution.

        Args:
            dgdt (:obj:`numpy array`): Sinogram temporal derivatives at target time, ``shape=(m,k,n)``.
            inital_guess (:obj:`numpy array`): Initial guess velocity basis coefficents 
                ``shape=flow_model.velocity_basis.coefficents.shape``
            maxiter (:obj:`int`): Maximum number of L-BFGS-B iterations. 
            verbose (:obj:`boolean`, optional): Print convergence information. Defaults to True.
            print_frequency (:obj:`int`, optional): Only if ``verbose=True``. Print convergence information
                at every ``print_frequency iteration`` step. Defaults to 1.

        """
        self._iteration = 0
        self._print_frequency = 1
        self._second_member = dgdt

        if verbose:
            callback = self._print_current_optim_state
        else:
            def callback(xk, state=None, cost=None): return False

        callback( inital_guess.flatten(), state=None, cost=None)
        result = minimize( self.cost_and_gradient,
                           inital_guess.flatten(),
                           method='L-BFGS-B',
                           jac=True,
                           callback=callback,
                           options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol':1e-16, 'maxls':25} )

        optimal_coefficents = self._scale_to_cfl_constraint( result.x )
        
        callback( optimal_coefficents.flatten(), state=None, cost=None)
        self.optimal_coefficents  = optimal_coefficents.reshape( self.flow_model.velocity_basis.coefficents.shape )

    def _print_current_optim_state(self, xk, state=None, cost=None):
        """Print cost, max CFL and iteration count as the minimizer progress.
        """

        if self._iteration==0 or self._iteration%self._print_frequency==0:

            if self._iteration==0:
                print(''.ljust(1),'Iteration'.ljust(25),'Cost'.ljust(20),'max CFL')

            if cost is None:
                cost = self.cost(xk)
            max_cfl = np.max( self._get_nodal_cfl(xk) )

            print( ''.ljust(4),str(self._iteration).ljust(11),''.ljust(4), str(cost).ljust(24), str(max_cfl))

        self._iteration += 1

        return False
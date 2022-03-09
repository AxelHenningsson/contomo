import numpy as np

class PiecewiseQuadratic(object):
    """Interpolate by piecewise quadratic polynomials data from a univariate function.

    This class represents a piecewise quadratic polynomial that passes through a series
    of (x,y) data points. The resulting continous function will have a continous derivative 
    with prescribed value of first derivative at the inital data point (x_0,y_0).

    Args:
        times (:obj:`numpy array`): Time points at which the data was collected, `shape=(N,)`.
        data (:obj:`numpy array`): The scalar data values collected at the timepoints, `shape=(N,)`.
        bc_indx (:obj:`float`): Index of data point at which to enforce derivative value ```bc_deriv```. Defaults to zero.
        bc_deriv (:obj:`float`): Value of first derivative to enforce at times[```bc_indx```]. Defaults to zero.

    """

    def __init__(self, times, data, bc_indx=0, bc_deriv=0):
        assert np.allclose( np.sort(times), times ), "Input times needs to be sorted"
        self._times = times
        self._data  = data
        self._Mi = np.array([ [-1,  1,  -1],
                             [ 0,  0,   1],
                             [ 1,  0,   0]] )
        self._point_derivatives = self._get_point_derivatives( times, data, bc_indx, bc_deriv )

    def _get_point_derivatives(self, times, data, bc_indx, bc_deriv):
        """Find the first derivatives at the given timepoints to speed up later interpolation evaluation.

        The idea is to solve for the values as a chain starting with setting the first derivative to `bc_start`, 
        fitting the first quadratic. The derivative of this interval at the endpoint is then passed along 
        to fit the next quadratic and so on. Only the derivatives are saved rather than the polynomial
        coefficents to save memory. The resulting point derivatives are expressed in the local basis, i.e
        each polynomial exist on a [0,1] intervall mapping.

        Args: 
            times (:obj:`numpy array`): Time points at which the data was collected, `shape=(N,)`.
            data (:obj:`numpy array`): The scalar data values collected at the timepoints, `shape=(N,)`.

        Returns:
            point_derivatives (:obj:`numpy array`): Derivatives of the piecewise polynomial at input times, `shape=(N,)`.

        """

        point_derivatives = np.zeros((len(times),))
        
        point_derivatives[bc_indx] = bc_deriv * (times[bc_indx+1] - times[bc_indx]) # local basis scale
        # NOTE: The point derivatives are in the local basis scale in a forward sense. i.e point_derivatives[i]
        # is in the scale of region i, which has lenght dt = times[i+1] - times[i].

        base = 2*( data[1:] - data[:-1] )
        for region in range(bc_indx, len(times)-2, 1): # forward marching +
            # Here we may first solve, and then convert to the local scale of derivative
            point_derivatives[region+1] = (base[region] - point_derivatives[region]) * ( (times[region+2] - times[region+1]) / (times[region+1] - times[region]) )
            
        for region in range(bc_indx, 0, -1 ): # backwards marching -
            # Here we must first convert to local derivative scale and then solve.
            point_derivatives[region-1] = base[region-1] - point_derivatives[region] * ( (times[region] - times[region-1]) / (times[region+1] - times[region]) )

        return point_derivatives

    def _time_to_intervall(self, time):
        """Find integer index of polynomial interval containing input time.
        """
        indx = np.argsort(np.abs(self._times-time))[0]
        if self._times[0] < time <= self._times[indx]:
            indx = indx - 1
        return indx

    def __call__(self, t, derivative=0):
        """Evaluate the piecewise polynomial or any of its derivatives at input points.

        Args:
            t (:obj:`iterable`): Time points at which to evaluate the piecewise polynomial.
            derivative (:obj:`int`): Order of desired derivative to evaluate. Default is 0.

        Returns:
            (:obj:`numpy array`) the polynomial values at input times.

        """
        out = np.zeros((len(t),))
        for i in range(len(t)):
            
            indx = self._time_to_intervall(t[i])
            c = self._Mi.dot( np.array([self._data[indx], self._data[indx+1], self._point_derivatives[indx]]) )
            xpar = (t[i] - self._times[indx]) / (self._times[indx+1] - self._times[indx])

            assert xpar>=0.0 and xpar<=1.0, "xpar="+str(xpar)+" _times[indx+2]="+str(self._times[indx+2])+" _times[indx+1]="+str(self._times[indx+1])+" _times[indx]="+str(self._times[indx])+" t[i]="+str(t[i])

            if derivative==0:
                out[i] = c[0]*xpar**2 + c[1]*xpar + c[2]
            elif derivative==1:
                out[i] = (2*c[0]*xpar + c[1]) / (self._times[indx+1] - self._times[indx])
            elif derivative==2:
                out[i] = 2*c[0] / (self._times[indx+1] - self._times[indx])**2
            else:
                out[i] = 0

        return out
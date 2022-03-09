import numpy as np
import matplotlib.pyplot as plt
import copy
from contomo.piecewise_quadratic import PiecewiseQuadratic

## wrapper to allow comparison to scipy...
# from scipy.interpolate import UnivariateSpline
# class unispline(object):
#     def __init__(self, x,y,k=3,s=None):
#         self.spline = UnivariateSpline(x,y,k=k,s=s)
#     def __call__(self, times, derivative=0):
#         return self.spline(times,nu=derivative)

class SinogramInterpolator(object):
    """Interpolate a timeseries of sinograms by piecewise quadratic polynomials.

    Given a set of pixelated sinograms with fixed dimension in time, this
    class defines an interpolation scheme, such that both the sinogram value and
    temporal derivatives of any one pixel can be evaluated for an arbitrary timepoint.

    Interpolation is executed through :obj:`contomo.piecewise_quadratic.PiecewiseQuadratic` with `bc_start=0`.

    Args:
        sample_times (:obj:`numpy array`): Times at which the sinograms where recorded.
        sinogram_timeseries (:obj:`numpy array`): Sinogram timeseries of ``shape=(T,m,n,d)`` where ``axis=2``
            indexes projections ``axis=0`` indexes time, and ``m`` and ``d`` indexes the pixels of the sinograms.

    Attributes:
        sample_times (:obj:`numpy array`): Times at which the sinograms where recorded.
        sinogram_timeseries (:obj:`numpy array`): Sinogram timeseries of ``shape=(T,m,n,d)`` where ``axis=2`` 
            indexes projections ``axis=0`` indexes time, and ``m`` and ``d`` indexes the pixels of the sinograms.

    """

    def __init__(self, sample_times, sinogram_timeseries):
        self.sample_times = sample_times
        self.sinogram_timeseries = sinogram_timeseries
        self._bc_indx  = 0
        self._bc_deriv = np.zeros(self.sinogram_timeseries[0].shape)
        self._set_splines()

        self._original_splines = copy.deepcopy( self.splines )
        self._original_sample_times = copy.deepcopy( self.sample_times )
        self._original_sinogram_timeseries = copy.deepcopy( self.sinogram_timeseries )

    def add_sinograms(self, times, sinograms, resolution=1e-8, update_splines=True):
        """Add projections at a series of timepoints points into the spline. 
        
        If the specified ``times`` are closer than ``resolution`` to already existing time
        points in the ``sinogram_timeseries`` existing sinograms will be replaced.

        To avoid poluting the data series with noise arising from ray model errors and the like
        it is possible to pass along a fixed scalar tolerance that defines a minimum distance
        from the original data path below which the passed sinogram values will not be used but
        the original data interpolation will be left untouched.  

        Args:
            times (:obj:`iterable` of :obj:`float`): Times at which ``sinograms`` are recorded.
            sinograms (:obj:`iterable` of :obj:`numpy array`): Sinograms to insert into the timeseries.
            resolution (:obj:`float`, optional): Threshold distance in time from existing 
                sinograms below which existing sinograms will be replaced by new ones. 
                Defaults to 1e-8.
            tolerance (:obj:`float`, optional): Only add sinogram pixel values into the timeseries that
                exceeds a predefined threshold in comparison to the original interpolation. i.e only pixels
                with absolute value difference to the original interpolation > tolerance will be added to the
                timeseries. Remaining pixels will be assigned the values of the original interpolation defined
                upon instantiation of the sinogram interpolator object. Defaults to 0, i.e all passed values
                are used by default.

        """
        for t,s in zip(times,sinograms):

            if np.min(np.abs(self.sample_times-t))<resolution:
                indx = np.argmin(np.abs(self.sample_times-t))
                self.sinogram_timeseries[indx,:,:,:] = s[:,:,:]
                self.sample_times[indx] = t
            else:
                self.sample_times = np.append( self.sample_times, [t] )
                self.sinogram_timeseries = np.append( self.sinogram_timeseries, [s], axis=0 )
        indx = np.argsort( self.sample_times )
        self.sample_times = self.sample_times[indx]
        self.sinogram_timeseries = self.sinogram_timeseries[indx]
        if update_splines: self._set_splines()

    def soft_spline_reset(self):
        self.sample_times = self._original_sample_times.copy()
        self.sinogram_timeseries = self._original_sinogram_timeseries.copy()

    def reset_splines(self):
        self.splines = copy.deepcopy( self._original_splines )
        self.sample_times = copy.deepcopy( self._original_sample_times )
        self.sinogram_timeseries = copy.deepcopy( self._original_sinogram_timeseries )
        self._set_splines()

    def set_bc_index( self, bc_indx ):
        """Set the spline derivaties to be equal to the original series derivatives at bc_indx. 
        """
        self._bc_indx  = bc_indx
        self._bc_deriv = self.__call__( self.sample_times[bc_indx], derivative=1, original=True )

    def _set_splines(self, local=None):
        """Set the per pixel splines for the timeseries data.
        """
        if local is None:
            indx1 = 0
            indx2 = len(self.sample_times)
            bc_indx = self._bc_indx
        else:
            indx1 = np.max( [0, np.argmin( np.abs(self.sample_times - local ) ) - 2 ] )
            indx2 = np.min( [len(self.sample_times), indx1+5] )
            bc_indx = self._bc_indx - indx1

        self.splines = []
        for i in range( self.sinogram_timeseries.shape[1] ):
            self.splines.append([])
            for p in range( self.sinogram_timeseries.shape[2] ):
                self.splines[i].append([])
                for j in range( self.sinogram_timeseries.shape[3] ):

                    spline = PiecewiseQuadratic( self.sample_times[indx1:indx2], 
                                                 self.sinogram_timeseries[indx1:indx2,i,p,j], 
                                                 bc_indx=bc_indx, 
                                                 bc_deriv=self._bc_deriv[i,p,j] )

                    #spline = unispline(self.sample_times, self.sinogram_timeseries[:,i,p,j], k=3, s=0)
                    self.splines[i][p].append( spline )

    def __call__(self, times, derivative=0, original=False):
        """Evaluate the interpolation at an arbitrary time.

        Args:
            times (:obj:`iterable` of :obj:`float` or :obj:`float` or :obj:`int`): Times at which to evaluate the interpolation.
            derivative (:obj:`int`, optional): Order of temporal derivative to evaluate. Defaults to 0.
            original (:obj:`boolean`, optional): Use the orignal unmutated sinogram data. This disregards
                any sinograms added via :meth:`add_sinograms()` Defaults to False.

        Returns
            (:obj:`numpy array`) interpolated sinograms at times ``time`` of ``shape=(len(times),m,n,d)``
            or ``shape=(m,n,d)`` of times is a scalar.

        """

        if isinstance(times, int):
            times = float(times)

        if isinstance(times, float):
            times = np.array([times])

        if original:
            splines = self._original_splines
        else:
            splines = self.splines

        T,m,n,d = self.sinogram_timeseries.shape
        interpolated_array = np.zeros( (len(times),m,n,d) )
        for i in range( self.sinogram_timeseries.shape[1] ):
            for p in range( self.sinogram_timeseries.shape[2] ):
                for j in range( self.sinogram_timeseries.shape[3] ):
                    interpolated_array[:, i, p, j] = splines[i][p][j](times, derivative=derivative)
        
        if interpolated_array.shape[0]==1:
            interpolated_array = interpolated_array[0,:,:,:]

        return interpolated_array

    def show_fit(self, row, projection, col, times):
        """Displays a simple plot of the spline interpolation for selected pixels.

        Args:
            row (:obj:`list` of int): Pixel rows to plot in the sinogram.
            projection (:obj:`list` of int): Projections to plot.
            col  (:obj:`list` of int): Pixel columns to plot in the sinogram.
            times  (:obj:`numpy array`): timepoint at which to evaluate and plot the splines.

        Returns
            matplotlib.pyplot figure and axis. Can be showned by calling the ```show()``` command.

        """

        fig,ax = plt.subplots(1,2,figsize=(12,6))
        mask1 = (self.sample_times<=np.max(times))*(self.sample_times>=np.min(times))
        mask2 = (self._original_sample_times<=np.max(times))*(self._original_sample_times>=np.min(times))
        mask3 = (times<=self.sample_times[-1])*(times>=self.sample_times[0])

        for i in row:
            for p in projection:
                for j in col:
                    spl  =  self.splines[i][p][j](times[mask3], derivative=0)
                    splderiv = self.splines[i][p][j](times[mask3], derivative=1)
                    ax[0].plot(times[mask3],spl,label="Spline")
                    ax[0].plot(self.sample_times[mask1], self.sinogram_timeseries[mask1,i,p,j], 'r^', markersize=8, label="Reinterpolation data")
                    ax[0].plot(self._original_sample_times[mask2], self._original_sinogram_timeseries[mask2,i,p,j], 'bo', markersize=4, label="Original data")
                    ax[1].plot(times[mask3],splderiv)
        ax[0].grid(True)
        ax[0].set_title('Spline fit')
        ax[1].grid(True)
        ax[1].set_title('Spline Derivative')
        ax[0].legend()
        ax[1].legend()
        return fig, ax
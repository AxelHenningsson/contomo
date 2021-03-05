import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import copy

class SinogramInterpolator(object):
    """Interpolate a timeseries of sinograms by univariate splines.

    Given a set of pixelated sinograms with fixed dimension in time, this
    class defines an interpolation scheme, such that both the sinogram value and
    temporal derivatives of any one pixel can be evaluated for an arbitrary timepoint.

    Support for inserting, replacing and resetting the interpolation splines is given.

    Interpolation is executed through :obj:`scipy.interpolate.UnivariateSpline` for 
    further documentation and details on the interpolation `see the scipy docs`_.
    
    .. _`see the scipy docs`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

    Args: 

        sample_times (:obj:`numpy array`): Times at which the sinograms where recorded.
        sinogram_timeseries (:obj:`numpy array`): Sinogram timeseries of ``shape=(T,m,n,d)`` where ``axis=2``
            indexes projections ``axis=0`` indexes time, and ``m`` and ``d`` indexes the pixels of the sinograms.
        smoothness (:obj:`float`, optional): Required smoothness of spline. Defaults to 0 (snugg fit to data).
        order (:obj:`int`): Order of the univariate spline to use for interpolation. Defaults to 2 (quadratic).

    Attributes:
        sample_times (:obj:`numpy array`): Times at which the sinograms where recorded.
        sinogram_timeseries (:obj:`numpy array`): Sinogram timeseries of ``shape=(T,m,n,d)`` where ``axis=2`` 
            indexes projections ``axis=0`` indexes time, and ``m`` and ``d`` indexes the pixels of the sinograms.
        smoothness (:obj:`float`, optional): Required smoothness of spline. Defaults to 0 (snugg fit to data).
        order (:obj:`int`): Order of the univariate spline to use for interpolation. Defaults to 2 (quadratic).

    """

    def __init__(self, sample_times, sinogram_timeseries, smoothness=0, order=2):
        self.sample_times = sample_times
        self.sinogram_timeseries = sinogram_timeseries
        self.order = order
        self.smoothness = smoothness

        self._set_splines()

        self._original_splines = copy.deepcopy( self.splines )
        self._original_sample_times = copy.deepcopy( self.sample_times )
        self._original_sinogram_timeseries = copy.deepcopy( self.sinogram_timeseries )

    def reset(self):
        """Reset the spline to it's original state existing upon creation.
        """
        self.splines = copy.deepcopy( self._original_splines )
        self.sample_times = copy.deepcopy( self._original_sample_times )
        self.sinogram_timeseries = copy.deepcopy( self._original_sinogram_timeseries )

    def add_sinograms(self, times, sinograms, resolution=1e-8 ):
        """Add projections at a series of timepoints points into the spline. 
        
        If the specified ``times`` are closer than ``resolution`` to already existing time
        points in the ``sinogram_timeseries`` existing sinograms will be replaced.  

        Args:
            times (:obj:`iterable` of :obj:`float`): Times at which ``sinograms`` are recorded.
            sinograms (:obj:`iterable` of :obj:`numpy array`): Sinograms to insert into the timeseries.
            resolution (:obj:`float`, optional): Threshold distance in time from existing 
                sinograms below which existing sinograms will be replaced by new ones. 
                Defaults to 1e-8.

        """
        for t,s in zip(times,sinograms):
            if np.min(np.abs(self.sample_times-t))<resolution:
                indx = np.argmin(np.abs(self.sample_times-t))
                self.sample_times[indx] = t
                self.sinogram_timeseries[indx,:,:,:] = s[:,:,:]
            else:
                self.sample_times = np.append( self.sample_times, [t] )
                self.sinogram_timeseries = np.append( self.sinogram_timeseries, [s], axis=0 )
        indx = np.argsort( self.sample_times )
        self.sample_times = self.sample_times[indx]
        self.sinogram_timeseries = self.sinogram_timeseries[indx]
        self._set_splines()

    def _set_splines(self):
        """Set the per pixel splines for the timeseries data.
        """
        self.splines = []
        for i in range( self.sinogram_timeseries.shape[1] ):
            self.splines.append([])
            for p in range( self.sinogram_timeseries.shape[2] ):
                self.splines[i].append([])
                for j in range( self.sinogram_timeseries.shape[3] ):
                    spline = UnivariateSpline( self.sample_times, self.sinogram_timeseries[:,i,p,j], k=self.order, s=self.smoothness )
                    self.splines[i][p].append( spline )

    def __call__(self, times, derivative=0, original=False):
        """Evaluate the interpolation at an arbitrary time.

        Args:
            times (:obj:`iterable`): Times at which to evaluate the interpolation.
            derivative (:obj:`int`, optional): Order of temporal derivative to evaluate. Defaults to 0.
            original (:obj:`boolean`, optional): Use the orignal unmutated sinogram data. This disregards
                any sinograms added via :meth:`add_sinograms()` Defaults to False.

        Returns
            (:obj:`numpy array`) interpolated sinograms at times ``time`` of ``shape=(len(times),m,n,d)``

        """

        if original:
            splines = self._original_splines
        else:
            splines = self.splines

        T,m,n,d = self.sinogram_timeseries.shape
        interpolated_array = np.zeros( (len(times),m,n,d) )
        for i in range( self.sinogram_timeseries.shape[1] ):
            for p in range( self.sinogram_timeseries.shape[2] ):
                for j in range( self.sinogram_timeseries.shape[3] ):
                    interpolated_array[:, i, p, j] = splines[i][p][j](times, nu=derivative)
        return interpolated_array

    def show_fit(self, row, projection, col):
        """Displays a simple plot of the spline interpolation for selected pixels.

        Args:
            row (:obj:`list` of int): Pixel rows to plot in the sinogram.
            projection (:obj:`list` of int): Projections to plot.
            col  (:obj:`list` of int): Pixel columns to plot in the sinogram.

        """
        t = np.linspace( np.min(self.sample_times), np.max(self.sample_times), 500 )

        spl = self.__call__(t,derivative=0)
        splderiv = self.__call__(t,derivative=1)

        fig,ax = plt.subplots(1,2,figsize=(12,6))
        for i in row:
            for p in projection:
                for j in col:
                    ax[0].plot(t,spl[:,i,p,j])
                    ax[0].plot(self.sample_times, self.sinogram_timeseries[:,i,p,j], 'ro')
                    ax[1].plot(t,splderiv[:,i,p,j])
        ax[0].grid(True)
        ax[0].set_title('Spline fit')
        ax[1].grid(True)
        ax[1].set_title('Spline Derivative')
        plt.show()
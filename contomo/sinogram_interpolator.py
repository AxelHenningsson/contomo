import numpy as np
import matplotlib.pyplot as plt
import copy
from contomo.piecewise_quadratic import PiecewiseQuadratic

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

        self._set_splines()

        self._original_splines = copy.deepcopy( self.splines )
        self._original_sample_times = copy.deepcopy( self.sample_times )
        self._original_sinogram_timeseries = copy.deepcopy( self.sinogram_timeseries )

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
                    spline = PiecewiseQuadratic( self.sample_times, self.sinogram_timeseries[:,i,p,j] )
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
                    interpolated_array[:, i, p, j] = splines[i][p][j](times, derivative=derivative)
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
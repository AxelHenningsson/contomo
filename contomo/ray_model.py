import numpy as np
import matplotlib.pyplot as plt
import astra

class RayModel(object):
    """Numerical ray model for projecting 3d volumes, implementation is a wrapper of `the ASTRA-toolbox`_:
            
    .. _the ASTRA-toolbox : https://www.astra-toolbox.com/index.html

    The selected model is parallel3d_vec in the ASTRA documentation and allows for fast GPU accelerated projections.

    Args:
        volume_shape (:obj:`tuple`): Fixed shape of the :obj:`numpy arrays` to be traced by the ray model.
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
        angles (:obj:`numpy array`): Projection angles in degrees.

    Attributes:
        number_of_detector_pixels (:obj:`int`): Number of pixels comprising the detector side (square detector)
        angles (:obj:`numpy array`): Projection angles in degrees.

    NOTE: 
        This implementation uses the indexing convention that an array has
        x dimension along axis=0, y along axis=1, z along axis=2 with ascending coordinate
        corresponding to ascending index. X-ray is along x (axis=0) and rotation around
        z (axis=2). (This is not the default of ASTRA.) Projections plotted with matplotlib
        will be as if one sits on the sample and looks at the detector.

    """

    def __init__(self, volume_shape, number_of_detector_pixels, angles ):
        self.number_of_detector_pixels = number_of_detector_pixels
        self.angles = angles
        self._volume_shape = volume_shape
        self._dcols = number_of_detector_pixels
        self._drows = number_of_detector_pixels
        self._vec_geom = self._angles_to_vectors( np.radians(angles) )
        dx,dy,dz = volume_shape 
        self._vol_geom = astra.creators.create_vol_geom( dy, dz, dx  )
        self._proj_geom = astra.creators.create_proj_geom("parallel3d_vec", self._drows, self._dcols, self._vec_geom)

    def _angles_to_vectors( self, angles ):
        """Convert array of in plane angles to astra vector geometry.
        """
        proj_geom = astra.creators.create_proj_geom('parallel3d', 1, 1,\
                                        self._drows, self._dcols, angles )
        vec_geom = astra.functions.geom_2vec( proj_geom )['Vectors']
        return vec_geom

    def _astra_to_index_notation( self, astra_volume ):
        """Convert volume from ASTRA notation to index notation.
        """
        index_volume = astra_volume[:,:,::-1]
        index_volume = index_volume[:,::-1,:]
        index_volume = index_volume[::-1,:,:]
        index_volume = np.swapaxes(index_volume,0,2)
        index_volume = np.swapaxes(index_volume,0,1)
        return index_volume

    def _index_notation_to_astra( self, index_volume ):
        """Convert volume from index notation to ASTRA notation.
        """
        astra_volume = np.swapaxes(index_volume,0,1)
        astra_volume = np.swapaxes(astra_volume,0,2)
        astra_volume = astra_volume[::-1,:,:]
        astra_volume = astra_volume[:,::-1,:]
        astra_volume = astra_volume[:,:,::-1]
        return astra_volume

    def forward_project(self, data, pos=(0,0,0)):
        """Compute a forward projection of data translating it by pos.

        Args:
            data (:obj:`numpy array`): Data to be projected.
            pos (:obj:`tuple`, optional): Translation of volume. Defaults to ``(0,0,0)``.
                This argument is usefull for projecting subvolumes slightly faster.

        Returns:
            :obj:`numpy array` sinogram.

        """
        self._vol_geom = astra.creators.create_vol_geom( data.shape )
        self._vol_geom = astra.functions.move_vol_geom( self._vol_geom, pos)
        astra_volume = self._index_notation_to_astra( data )
        idn, sino = astra.creators.create_sino3d_gpu( astra_volume , self._proj_geom, self._vol_geom )
        astra.data3d.delete(idn)
        return sino

    def backward_project(self, data):
        """Compute backprojection from sinogram data.

        Args:
            data (:obj:`numpy array`): Sinogram to be backprojected, must have same shape 
                as the output sinogram of :meth:`forward_project`.

        Returns:
            :obj:`numpy array` backprojection.
        """
        idn, astra_volume = astra.creators.create_backprojection3d_gpu( data, self._proj_geom, self._vol_geom)
        astra.data3d.delete(idn)
        index_volume = self._astra_to_index_notation( astra_volume ) 
        return index_volume
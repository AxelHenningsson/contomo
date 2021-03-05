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

def padd(array, value, pad):
    new_arr = np.zeros( tuple(np.array(array.shape)+2*pad) )
    new_arr[pad:-pad,pad:-pad,pad:-pad] = array[:,:,:]
    return new_arr

voxel_volume = np.ones((32,32,32))
voxel_volume = padd(voxel_volume, 0, 6)

voxel_size = 1.0
velocity_basis = basis.TetraMesh.generate_mesh_from_numpy_array(  np.ones(tuple(np.array(voxel_volume.shape)+8), dtype=np.uint8), 
                                                                  voxel_size=1.0, 
                                                                  max_cell_circumradius=10., 
                                                                  max_facet_distance=10.  )
displacement = -np.ones(velocity_basis.coord.shape)*4*voxel_size
velocity_basis.move( displacement )

print( 'Number of elements: ', velocity_basis.enod.shape[0] )
print( 'Number of nodes: ',velocity_basis.nodal_coordinates.shape[0])

fvm = FVM.FiniteVolumes( origin=(0., 0., 0.,), dx=1.0, shape_field=voxel_volume.shape )
fvm.fixate_velocity_basis( velocity_basis )
fvm.velocity_basis.coefficents = fvm.velocity_basis.coord**2
dfbardt = fvm.get_dfbardt( velocity=None, f=voxel_volume, get_ddfbardtdu=False, debug=False )

vn1,vp1 = fvm._get_vertex_velocity( velocity=velocity_basis, axis=0 ) 
vn2,vp2 = fvm._get_vertex_velocity( velocity=None, axis=0 ) 

assert np.allclose(vn1, vn2)
assert np.allclose(vp1, vp2)

for i in range(vn1.shape[2]):
    fig,ax = plt.subplots(1,2) 
    ax[0].imshow(vn1[:,:,i],vmin=0, vmax=np.max(vn1))
    ax[1].imshow(vn2[:,:,i],vmin=0, vmax=np.max(vn1))
    plt.show()



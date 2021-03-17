import numpy as np
import os
import contomo
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def rmse( arr1, arr2 ):
    n = len(arr1.flatten())
    return np.sqrt( np.sum( (arr1 - arr2)**2 ) / n )

def mae( arr1, arr2 ):
    n = len(arr1.flatten())
    return np.sqrt( np.sum( np.abs(arr1 - arr2) ) / n )

def maxabs( arr1, arr2 ):
    return np.max(np.abs(arr1 - arr2))

def match( target, predictions, th ):
    return np.sum( np.abs(target-predictions)<th ) / len(target.flatten())

save_path = "/home/axel/Downloads/spheres"

times           =  np.load(save_path+"/times.npy")
meta_data       =  np.load(save_path+"/meta_data.npy", allow_pickle=True).item()
print("Analysing reconstruction with meta parameters: ")
for key in meta_data:
    print(key, " : ", meta_data[key])




################################################################################################################################
# Volume matching timeseries analysis, plot perecentual match for segmenting thresholds as 3d surface
thresholds = np.linspace(0.0,1.0,1000)
tomo_times = []
overlaps  = []
norm_factor = np.max( np.load("/home/axel/Downloads/intermediate_volume_0000.npy") )
for i, time in enumerate(times):

    try:
        tomo_volume = np.load("/home/axel/Downloads/intermediate_volume_"+str(i).zfill(4)+".npy")
    except:
        continue
    
    flow_volume = np.load( save_path+"/volumes/volume_"+str(i).zfill(4)+ ".npy" )

    tomo_volume = tomo_volume/norm_factor
    flow_volume = flow_volume/norm_factor

    overlaps.append( [match( tomo_volume, flow_volume, th ) for th in thresholds] )
    tomo_times.append(time)

tomo_times = np.asarray(tomo_times)
overlaps = np.asarray(overlaps)
X,Y = np.meshgrid( np.array(tomo_times), thresholds,   indexing="ij" )
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.set_title("Evolution of Per Voxel Match of Normalised Volumes", fontsize=20)
ax.set_xlabel("Time [s]",fontsize=20)
ax.set_ylabel("Gray value threshold",fontsize=20)
surf = ax.plot_surface(X, Y, overlaps, cmap=cm.RdBu,
                       linewidth=0, antialiased=False)
ax.set_zlim(0.0, 1.1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
fig.colorbar(surf, shrink=0.5, aspect=5)

fig,ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(thresholds, thresholds, c='#0072b2', linewidth=1.5, linestyle='--', label="Match for random uniform draw")

n = len(tomo_times)
ax.plot(thresholds, overlaps[0,:], c='C1' , linewidth=3, linestyle='-', label="Match for reconstructed field t="+str(np.round(tomo_times[0],3))+"s")
ax.plot(thresholds, overlaps[n//2,:], c='C2' , linewidth=3, linestyle='-', label="Match for reconstructed field t="+str(np.round(tomo_times[n//2],3))+"s")
ax.plot(thresholds, overlaps[-1,:], c='C3' , linewidth=3, linestyle='-', label="Match for reconstructed field t="+str(np.round(tomo_times[-1],3))+"s")

ax.axhline(y=1.0, xmin=0.0, xmax=1.0, c='k', linewidth=1.5, linestyle='--', label="Maximum attainable match")
ax.set_title("Per Voxel Match of Normalised Volumes at end time", fontsize=20)
ax.set_ylabel("Fraction of overlap", fontsize=20)
ax.set_xlabel("Gray value threshold", fontsize=20)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.legend(fontsize=15)

################################################################################################################################








################################################################################################################################
# Volume residual analysis, plot rmse, mae and maxabs as they evolve with time.
volume_residual_maxabs = []
volume_residual_rmse   = []
volume_residual_mae    = []
tomo_times = []
norm_factor = np.max( np.load("/home/axel/Downloads/intermediate_volume_0000.npy") )

for i, time in enumerate(times):

    try:
        tomo_volume = np.load("/home/axel/Downloads/intermediate_volume_"+str(i).zfill(4)+".npy")
    except:
        continue
    
    reconstructed_volume   =  np.load(save_path + "/volumes/volume_" + str(i).zfill(4) + ".npy")

    tomo_volume = tomo_volume/norm_factor
    reconstructed_volume = reconstructed_volume/norm_factor

    volume_residual_maxabs.append( maxabs(tomo_volume, reconstructed_volume) )
    volume_residual_rmse.append( rmse(tomo_volume, reconstructed_volume) )
    volume_residual_mae.append( mae(tomo_volume, reconstructed_volume) )
    tomo_times.append( time )


data   = [volume_residual_maxabs, volume_residual_rmse, volume_residual_mae]
titles = ["Max absolute value of volume residual", "RMSE of volume residual", "MAE of volume residual"]
colors = ['#d55e00',  '#0072b2',  '#009e73']
fig,ax = plt.subplots(1,3, figsize=(15,5))
for i,(c,d,t) in enumerate(zip(colors,data,titles)):
    ax[i].plot( tomo_times, d, c=c,  linestyle='--', marker='o' )
    ax[i].set_title(t, fontsize=15)
    ax[i].grid(True)
    ax[i].tick_params(axis='both', which='major', labelsize=11)
    ax[i].tick_params(axis='both', which='minor', labelsize=11)
    ax[i].set_xlabel("time [s]", fontsize=15)

################################################################################################################################










################################################################################################################################
# Sinogram residual analysis, plot rmse, mae and maxabs as they evolve with time.
sinogram_residual_maxabs = np.zeros((len(times),))
sinogram_residual_rmse   = np.zeros((len(times),))
sinogram_residual_mae    = np.zeros((len(times),))


norm_factor  =  np.max( np.load(save_path + "/projections/interpolated_sinogram_0000.npy") )
for i,time in enumerate(times):
    interpolated_sinogram  =  np.load(save_path + "/projections/interpolated_sinogram_" + str(i).zfill(4) + ".npy")
    reconstructed_sinogram =  np.load(save_path + "/projections/reconstructed_sinogram_" + str(i).zfill(4) + ".npy")

    interpolated_sinogram = interpolated_sinogram/norm_factor
    reconstructed_sinogram = reconstructed_sinogram/norm_factor

    sinogram_residual_maxabs[i] =  maxabs(interpolated_sinogram, reconstructed_sinogram)
    sinogram_residual_rmse[i]   =  rmse(interpolated_sinogram, reconstructed_sinogram)
    sinogram_residual_mae[i]    =  mae(interpolated_sinogram, reconstructed_sinogram)

data   = [sinogram_residual_maxabs, sinogram_residual_rmse, sinogram_residual_mae]
titles = ["Max absolute value of sinogram residual", "RMSE of sinogram residual", "MAE of sinogram residual"]
colors = ['#d55e00',  '#0072b2',  '#009e73']
fig,ax = plt.subplots(1,3, figsize=(15,5))
for i,(c,d,t) in enumerate(zip(colors,data,titles)):
    ax[i].plot( times, d, c=c,  linestyle='--', marker='o' )
    ax[i].set_title(t, fontsize=15)
    ax[i].grid(True)
    ax[i].tick_params(axis='both', which='major', labelsize=11)
    ax[i].tick_params(axis='both', which='minor', labelsize=11)
    ax[i].set_xlabel("time [s]", fontsize=15)
################################################################################################################################







################################################################################################################################
# Phantom velocity analysis, plot rmse, mae and maxabs as they evolve with time.
velocity_residual_maxabs = np.zeros((len(times),3))
velocity_residual_rmse   = np.zeros((len(times),3))
velocity_residual_mae    = np.zeros((len(times),3))

pde = contomo.projected_advection_pde.ProjectedAdvectionPDE.load("/home/axel/Downloads/projected_advection_pde.papde")
ph  = contomo.phantom.Spheres.load( "/home/axel/Downloads/spherephantom.phantom" )

for i,time in enumerate(times):

    coeffs = np.load(save_path + "/velocity/basis_coefficents_" + str(i).zfill(4) + ".npy")
    pde.flow_model.velocity_basis.coefficents = coeffs
    
    radii = ph.get_radii(time)
    dr = radii/2.

    c1 = ph.get_coordinates(time)
    c2 = ph.get_coordinates(time + np.min(radii)*1e-5)
    v_phantom  = (c2-c1)/(np.min(radii)*1e-5)

    vc  = pde.flow_model.velocity_basis( c1[:,0], c1[:,1], c1[:,2], dim="all" )

    vxp = pde.flow_model.velocity_basis( c1[:,0]+dr, c1[:,1], c1[:,2], dim="all" )
    vxn = pde.flow_model.velocity_basis( c1[:,0]-dr, c1[:,1], c1[:,2], dim="all" )
        
    vyp = pde.flow_model.velocity_basis( c1[:,0], c1[:,1]+dr, c1[:,2], dim="all" )
    vyn = pde.flow_model.velocity_basis( c1[:,0], c1[:,1]-dr, c1[:,2], dim="all" )

    vzp = pde.flow_model.velocity_basis( c1[:,0], c1[:,1], c1[:,2]+dr, dim="all" )
    vzn = pde.flow_model.velocity_basis( c1[:,0], c1[:,1], c1[:,2]-dr, dim="all" )

    v_recon = (vc + vxp + vxn + vyp + vyn + vzp + vzn ) / 7.

    for k in range(3):
        velocity_residual_maxabs[i,k] =  maxabs(v_phantom[:,k], v_recon[:,k])
        velocity_residual_rmse[i,k]   =  rmse(v_phantom[:,k], v_recon[:,k])
        velocity_residual_mae[i,k]    =  mae(v_phantom[:,k], v_recon[:,k])

data   = [velocity_residual_maxabs, velocity_residual_rmse, velocity_residual_mae]
titles = ["Max absolute value of velocity residual", "RMSE of velocity residual", "MAE of velocity residual"]
fig,ax = plt.subplots(1,3, figsize=(15,5))
for i,(c,d,t) in enumerate(zip(colors,data,titles)):
    ax[i].plot( times, d[:,0], c='#d55e00',  linestyle='--', marker='o', label='$v_x$' )
    ax[i].plot( times, d[:,1], c='#0072b2',  linestyle='--', marker='o', label='$v_y$' )
    ax[i].plot( times, d[:,2], c='#009e73',  linestyle='--', marker='o', label='$v_z$' )
    ax[i].set_title(t, fontsize=15)
    ax[i].grid(True)
    ax[i].tick_params(axis='both', which='major', labelsize=11)
    ax[i].tick_params(axis='both', which='minor', labelsize=11)
    ax[i].set_xlabel("time [s]", fontsize=15)
    ax[i].legend( fontsize=15 )
################################################################################################################################




plt.show()




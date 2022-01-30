
# Import Packages
import numpy as np
import scipy.linalg

#TODO: Commenting


def POD1(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    param = U[s_ind:e_ind, :]
    param_mean = np.mean(param, axis=0)[np.newaxis, ...]

    param_flux = param - param_mean
    flux_copy = np.copy(param_flux)

    # Snapshots
    snap_shots = np.matmul(param_flux, param_flux.T)
    eig_vals, eig_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eig_vals = eig_vals[eig_vals.shape[0]::-1]
    eig_vecs = eig_vecs[:, eig_vals.shape[0]::-1]

    spatial_modes = np.matmul(param_flux.T, eig_vecs[:,:modes])/np.sqrt(eig_vals[:modes])
    temporal_coefficients = np.matmul(param_flux,spatial_modes)

    return spatial_modes,temporal_coefficients,eig_vals,flux_copy

def POD2(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    #Parameters
    param1 = U[ s_ind:e_ind,:,:,0]
    param2 = U[s_ind:e_ind,:,:, 1]

    param1_mean = np.mean(param1, axis=0)[np.newaxis, ...]
    param2_mean = np.mean(param2, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    param1_flux = param1 - param1_mean
    param2_flux = param2 - param2_mean
    
    # return copies
    flux1_copy = np.copy(param1_flux)
    flux2_copy = np.copy(param2_flux)

    #Reshape spatial_dims for Snapshots
    shape = param1_flux.shape
    param1_flux = param1_flux.reshape(shape[0], shape[1] * shape[2])
    param2_flux = param2_flux.reshape(shape[0], shape[1] * shape[2])
    stacked_flux = np.hstack((param1_flux, param2_flux))

    # Snapshot Method:
    snap_shots = np.matmul(stacked_flux, stacked_flux.T) # YY^T
    eigen_vals, eigen_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eigen_vals = eigen_vals[eigen_vals.shape[0]::-1]
    eigen_vecs = eigen_vecs[:, eigen_vals.shape[0]::-1] #unit norm

    spatial_modes = np.matmul(stacked_flux.T, eigen_vecs[:, :modes]) / np.sqrt(eigen_vals[:modes]) # is this unit norm?
    temporal_coefficients = np.matmul(stacked_flux, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, eigen_vals, flux1_copy, flux2_copy


def POD3(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """

    # density in x
    S_ux = U[0,:, s_ind:e_ind]
    S_ux = np.moveaxis(S_ux,[0, 1], [1, 0])

    # velocity in x
    S_uy = U[1,:, s_ind:e_ind]
    S_uy = np.moveaxis(S_uy, [0, 1], [1, 0])
    
    # velocity in x
    S_uz = U[2,:, s_ind:e_ind]
    S_uz = np.moveaxis(S_uz, [0, 1], [1, 0])

    # taking the temporal mean of snapshots
    S_uxm = np.mean(S_ux, axis=0)[np.newaxis, ...]
    S_uym = np.mean(S_uy, axis=0)[np.newaxis, ...]
    S_uzm = np.mean(S_uy, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    Ux = S_ux - S_uxm
    Uy = S_uy - S_uym
    Uz = S_uz-S_uzm
        
    # return copies
    Uxr = np.copy(Ux)
    Uyr = np.copy(Uy)
    Uzr = np.copy(Uz)

    # Reshaping to create snapshot matrix Y
    Y = np.hstack((Ux, Uy, Uz))

    # Snapshot Method:
    Cs = np.matmul(Y, Y.T)    

    # L:eigvals, As:eigvecs
    Lv, As = scipy.linalg.eigh(Cs)

    # descending order
    Lv = Lv[Lv.shape[0]::-1]
    As = As[:, Lv.shape[0]::-1]

    spatial_modes = np.matmul(Y.T, As[:, :modes]) / np.sqrt(Lv[:modes])    
    temporal_coefficients = np.matmul(Y, spatial_modes)

    return spatial_modes, temporal_coefficients, Lv, Uxr, Uyr, Uzr



def PODKPP(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    # velocity in x
    S_ux = U[:, :, s_ind:e_ind]
    S_ux = np.moveaxis(S_ux, [0, 1, 2], [1, 2, 0])

    # taking the temporal mean of snapshots
    S_uxm = np.mean(S_ux, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    Ux = S_ux - S_uxm
    
    # return copies
    Uxr = np.copy(Ux)

    # Reshaping to create snapshot matrix Y
    shape = Ux.shape
    Y = Ux.reshape(shape[0], shape[1] * shape[2])

    # Snapshot Method:
    Cs = np.matmul(Y, Y.T)

    # L:eigvals, As:eigvecs
    Lv, As = scipy.linalg.eigh(Cs)

    # descending order
    Lv = Lv[Lv.shape[0]::-1]
    As = As[:, Lv.shape[0]::-1]

    spatial_modes = np.matmul(Y.T, As[:, :modes]) / np.sqrt(Lv[:modes])
    temporal_coefficients = np.matmul(Y, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, Lv, Uxr
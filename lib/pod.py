
# Import Packages
import numpy as np
import scipy.linalg

#TODO: Commenting


def POD1(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    # velocity in x
    S_ux = U[s_ind:e_ind, :]
    
    # taking the temporal mean of snapshots
    S_uxm = np.mean(S_ux, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    Ux = S_ux - S_uxm
    
    # return copies
    Uxr = np.copy(Ux)

    # Snapshot Method:
    Cs = np.matmul(Ux, Ux.T)

    # L:eigvals, As:eigvecs
    Lv, As = scipy.linalg.eigh(Cs)

    # descending order
    Lv = Lv[Lv.shape[0]::-1]
    As = As[:, Lv.shape[0]::-1]

    spatial_modes = np.matmul(Ux.T, As[:, :modes]) / np.sqrt(Lv[:modes])
    temporal_coefficients = np.matmul(Ux, spatial_modes)

    return spatial_modes, temporal_coefficients, Lv, Uxr

def POD2(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    # velocity in x
    S_ux = U[:, :, s_ind:e_ind, 0]
    S_ux = np.moveaxis(S_ux, [0, 1, 2], [1, 2, 0])

    # velocity in y
    S_uy = U[:, :, s_ind:e_ind, 1]
    S_uy = np.moveaxis(S_uy, [0, 1, 2], [1, 2, 0])

    # taking the temporal mean of snapshots
    S_uxm = np.mean(S_ux, axis=0)[np.newaxis, ...]
    S_uym = np.mean(S_uy, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    Ux = S_ux - S_uxm
    Uy = S_uy - S_uym
    
    # return copies
    Uxr = np.copy(Ux)
    Uyr = np.copy(Uy)

    # Reshaping to create snapshot matrix Y
    shape = Ux.shape
    Ux = Ux.reshape(shape[0], shape[1] * shape[2])
    Uy = Uy.reshape(shape[0], shape[1] * shape[2])
    Y = np.hstack((Ux, Uy))

    # Snapshot Method:
    Cs = np.matmul(Y, Y.T) # YY^T

    # L:eigvals, As:eigvecs
    Lv, As = scipy.linalg.eigh(Cs)

    # descending order
    Lv = Lv[Lv.shape[0]::-1]
    As = As[:, Lv.shape[0]::-1] #unit norm

    spatial_modes = np.matmul(Y.T, As[:, :modes]) / np.sqrt(Lv[:modes]) # is this unit norm?
    temporal_coefficients = np.matmul(Y, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, Lv, Uxr, Uyr


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
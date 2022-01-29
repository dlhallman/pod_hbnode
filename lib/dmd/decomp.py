import numpy as np
import scipy.linalg



def DMD(X,Xp,modes):

    U,Sigma,V = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    Ur = U[:,:modes]
    Sigmar = np.diag(Sigma[:modes]    )
    Vr = V[:modes,:].T

    invSigmar = np.linalg.inv(Sigmar)


    Atilde = Ur.T@Xp@Vr@invSigmar
    Lambda, W = np.linalg.eig(Atilde)
    Lambda = np.diag(Lambda)

    Phi = Xp@(Vr@invSigmar)@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = np.linalg.solve(W@Lambda,alpha1)

    return X, Atilde, Ur, Phi, Lambda, Sigma, b

def DMD1(data,s_ind,e_ind,modes):

    var = data[s_ind:e_ind,:]
    var1_mean = np.mean(var, axis=0)[np.newaxis, ...]

    var = var-var1_mean

    X = data[s_ind:e_ind-1,:].T
    Xp = data[s_ind+1:e_ind,:].T

    return DMD(X,Xp,modes)


def DMD2(data,s_ind,e_ind,modes):

    var1 = data[:, :, s_ind:e_ind, 0]
    var1 = np.moveaxis(var1, [0, 1, 2], [1, 2, 0])

    var2 = data[:, :, s_ind:e_ind, 1]
    var2 = np.moveaxis(var2, [0, 1, 2], [1, 2, 0])

    var1_mean = np.mean(var1, axis=0)[np.newaxis, ...]
    var2_mean = np.mean(var2, axis=0)[np.newaxis, ...]

    var1 = var1-var1_mean
    var2 = var2-var2_mean


    shape = var1.shape
    var1 = var1.reshape(shape[0], shape[1] * shape[2])
    var2 = var2.reshape(shape[0], shape[1] * shape[2])
    Y = np.hstack((var1, var2))

    X = Y[:-1,:].T
    Xp = Y[1:,:].T

    return DMD(X,Xp,modes)


def DMD3(data,s_ind,e_ind,modes):

    var1 = data[0,:, s_ind:e_ind]
    var1 = np.moveaxis(var1,[0, 1], [1, 0])

    var2 = data[1,:, s_ind:e_ind]
    var2 = np.moveaxis(var2,[0, 1], [1, 0])

    var3 = data[2,:, s_ind:e_ind]
    var3 = np.moveaxis(var3,[0, 1], [1, 0])

    Y = np.hstack((var1, var2, var3))

    X = Y[:-1,:]
    Xp = Y[1:,:]

    return DMD(X,Xp,modes)

def DMDKPP(data,s_ind,e_ind,modes):

    var = data[:,:, s_ind:e_ind]
    var = np.moveaxis(var,[0, 1, 2], [1, 2, 0])

    var_mean = np.mean(var, axis=0)[np.newaxis, ...]
    var = var-var_mean

    shape = var.shape
    var = var.reshape(shape[0], shape[1] * shape[2])

    X = var[:-1,:]
    Xp = var[1:,:]

    return DMD(X,Xp,modes)
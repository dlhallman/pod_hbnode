import numpy as np
import scipy.linalg


def DMD1(data,s_ind,e_ind,modes):

    X = data[s_ind:e_ind-1,:].T
    Xp = data[s_ind+1:e_ind,:].T

    U,Sigma,V = np.linalg.svd(X, full_matrices=False)
    Ur = U[:,:modes]
    Sigmar = Sigma[:modes]    
    Vr = V[:,:modes]

    invSigmar = np.linalg.inv(np.diag(Sigmar))

    Atilde = Ur.T@Xp.T@Vr@invSigmar
    W,Lambda = np.linalg.eig(Atilde)

    Phi = Xp.T@Vr@invSigmar@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = (W@Lambda)/alpha1

    return Atilde, Ur, Phi, Lambda, b


def DMD2(data,s_ind,e_ind,modes):

    var1 = data[:, :, s_ind:e_ind, 0]
    var1 = np.moveaxis(var1, [0, 1, 2], [1, 2, 0])

    var2 = data[:, :, s_ind:e_ind, 1]
    var2 = np.moveaxis(var2, [0, 1, 2], [1, 2, 0])

    shape = var1.shape
    var1 = var1.reshape(shape[0], shape[1] * shape[2])
    var1 = var1.reshape(shape[0], shape[1] * shape[2])
    Y = np.hstack((var1, var2))

    X = Y[:-1,:]
    Xp = Y[1:,:]

    U,Sigma,V = np.linalg.svd(X.T, full_matrices=False)
    Ur = U[:,:modes]
    Sigmar = Sigma[:modes]    
    Vr = V[:,:modes]

    invSigmar = np.linalg.inv(np.diag(Sigmar))

    Atilde = Ur.T@Xp.T@Vr@invSigmar
    W,Lambda = np.linalg.eig(Atilde)

    Phi = Xp.T@Vr@invSigmar@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = (W@Lambda)/alpha1

    return Atilde, Ur, Phi, Lambda, b

def DMD3(data,s_ind,e_ind,modes):

    var1 = data[0,:, s_ind:e_ind]
    var1 = np.moveaxis(var1,[0, 1], [1, 0])

    var2 = data[1,:, s_ind:e_ind]
    var2 = np.moveaxis(var1,[0, 1], [1, 0])

    var3 = data[2,:, s_ind:e_ind]
    var3 = np.moveaxis(var1,[0, 1], [1, 0])

    shape = var1.shape
    var1 = var1.reshape(shape[0], shape[1] * shape[2])
    var1 = var1.reshape(shape[0], shape[1] * shape[2])
    Y = np.hstack((var1, var2, var3))

    X = Y[:-1,:]
    Xp = Y[1:,:]

    U,Sigma,V = np.linalg.svd(X.T, full_matrices=False)
    Ur = U[:,:modes]
    Sigmar = Sigma[:modes]    
    Vr = V[:,:modes]

    invSigmar = np.linalg.inv(np.diag(Sigmar))

    Atilde = Ur.T@Xp.T@Vr@invSigmar
    W,Lambda = np.linalg.eig(Atilde)

    Phi = Xp.T@Vr@invSigmar@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = (W@Lambda)/alpha1

    return Atilde, Ur, Phi, Lambda, b

def DMDKPP(data,s_ind,e_ind,modes):

    var = data[0,:, s_ind:e_ind]
    var = np.moveaxis(var,[0, 1, 2], [1, 2, 0])


    shape = var.shape
    var = var.reshape(shape[0], shape[1] * shape[2])

    X = var[:-1,:]
    Xp = var[1:,:]

    U,Sigma,V = np.linalg.svd(X.T, full_matrices=False)
    Ur = U[:,:modes]
    Sigmar = Sigma[:modes]    
    Vr = V[:,:modes]

    invSigmar = np.linalg.inv(np.diag(Sigmar))

    Atilde = Ur.T@Xp.T@Vr@invSigmar
    W,Lambda = np.linalg.eig(Atilde)

    Phi = Xp.T@Vr@invSigmar@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = (W@Lambda)/alpha1

    return Atilde, Ur, Phi, Lambda, b
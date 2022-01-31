from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('./')
from lib.datasets import VKS_DAT, KPP_DAT

plt.style.use('classic')


"""
VKS EXAMPLE

"""
k = 100
modes = 8
s_ind = 0
e_ind = 101
vks = VKS_DAT('./data/VKS.pkl')

shape = vks.shape
time_len = shape[0]
domain_len = shape[1]*shape[2]
domain_shape = shape[1:3]
component_len = shape[-1]

var1 = vks[s_ind:e_ind,:,:,0]
var2 = vks[s_ind:e_ind,:,:,1]

domain_len = var1.shape[1]*var1.shape[2]
time_len = var1.shape[0]

var1_mean = np.mean(var1, axis=0)[np.newaxis, ...]
var2_mean = np.mean(var2, axis=0)[np.newaxis, ...]

var1_flux = var1-var1_mean
var2_flux = var2-var2_mean

var1_flux = var1_flux.reshape(time_len,domain_len)
var2_flux = var2_flux.reshape(time_len,domain_len)

stacked_flux = np.hstack((var1_flux, var2_flux))

X = stacked_flux[:-1,:].T
Xp = stacked_flux[1:,:].T

U,Sigma,Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)

Ur = U[:,:modes]
Sigmar = np.diag(Sigma[:modes])
Vr = Vh[:modes,:].T

invSigmar = np.linalg.inv(Sigmar)

Atilde = Ur.T@Xp@Vr@invSigmar
Lambda, W = np.linalg.eig(Atilde)
Lambda = np.diag(Lambda)

Phi = Xp@(Vr@invSigmar)@W

alpha1 = Sigmar@(Vr[0,:].T)
b = np.linalg.solve(W@Lambda,alpha1)

print(Phi.shape, Lambda.shape, b.shape)
X0 = Xp[-1,:]

fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(domain_shape),
                origin='upper', vmin=-1, vmax=1, aspect='auto')]

def run(vks_t):
    Lambda_k = np.linalg.matrix_power(Lambda,vks_t)
    Xk=Phi@Lambda_k@b
    var1_xk = np.real(Xk[:domain_len].reshape(domain_shape))
    var2_xk = np.real(Xk[domain_len:].reshape(domain_shape))
    lines[0].set_data(var1_xk.T)
    lines[1].set_data(var2_xk.T)
    return lines

ani = animation.FuncAnimation(fig, run, blit=True, interval=k,
    repeat=False)

ani.save('./out/vks_dmd_recon.gif', "PillowWriter", fps=6)


"""
KPP EXAMPLE

"""
k = 100
modes = 8
s_ind = 0
e_ind = 101
kpp = KPP_DAT('./data/KPP.npz')
xv =  np.tile(np.linspace(-2,2,kpp.shape[1]),(kpp.shape[1],1))
yv = np.tile(np.linspace(-2.4,1.4,kpp.shape[2]),(kpp.shape[2],1)).T

var = kpp[s_ind:e_ind,:,:]
shape = var.shape
time_len = shape[0]
domain_len = shape[1]*shape[2]
domain_shape = shape[1:]

var_mean = np.mean(var, axis=0)[np.newaxis, ...]
var_flux = var-var_mean

var_flux = var_flux.reshape(time_len, domain_len)

X = var_flux[:-1,:].T
Xp = var_flux[1:,:].T

U,Sigma,Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
Ur = U[:,:modes]
Sigmar = np.diag(Sigma[:modes])
Vr = Vh[:modes,:].T

invSigmar = np.linalg.inv(Sigmar)


Atilde = Ur.T@Xp@Vr@invSigmar
Lambda, W = np.linalg.eig(Atilde)
Lambda = np.diag(Lambda)

Phi = Xp@(Vr@invSigmar)@W

alpha1 = Sigmar@(Vr[0,:].T)
b = np.linalg.solve(W@Lambda,alpha1)

plt.style.use('default')
fig = plt.figure(figsize=(12,12), tight_layout=True)
ax1 = fig.add_subplot(projection='3d')
ax1.set_zlim(0, 10)
lines =[ax1.plot_surface(xv, yv, np.ones(domain_shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)]

def run(kpp_t):
    ax1.clear()
    ax1.set_zlim(0, 10)
    Lambda_k = np.linalg.matrix_power(Lambda,kpp_t)
    Xk=np.real((Phi@Lambda_k@b).reshape(domain_shape))
    lines =[ax1.plot_surface(xv, yv, Xk, cmap=cm.coolwarm, linewidth=0, antialiased=False)]
    return lines

ani = animation.FuncAnimation(fig, run, blit=True, interval=k,
    repeat=False)
ani.save('./out/kpp_dmd_recon.gif', 'PillowWriter', fps=6)

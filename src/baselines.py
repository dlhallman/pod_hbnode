#IMPORTS
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

#SELF IMPORTS
import sys
sys.path.append('./')

from lib.decomp.dmd import *
from lib.decomp.pod import *
from lib.datasets import DMD_DATASET, VKS_DAT, KPP_DAT, EE_DAT


"""
VKS EXAMPLE

"""
padding=15

vks = VKS_DAT('./data/VKS.pkl')
domain_shape=vks.shape[1:3]
domain_len=vks.shape[1]*vks.shape[2]
time_len=vks.shape[0]

plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros((vks.shape[1:3])), origin='upper', vmin =-.4,vmax =.4, aspect='auto')]

def run_vks_lines(vks_t):
    lines[0].set_data(vks[vks_t,:,:,0].T)
    lines[1].set_data(vks[vks_t,:,:,1].T)
    return lines
ani = animation.FuncAnimation(fig, run_vks_lines, blit=False, interval=vks.shape[0]-1,
    repeat=False)
ani.save('./out/vks_init_data.gif', "PillowWriter", fps=6)

#STATIC PLOT
plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=36,pad=padding)
ax.imshow(vks[150,:,:,0].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=36,pad=padding)
ax.imshow(vks[150,:,:,1].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_init_data.pdf', format="pdf", bbox_inches="tight")


"""POD Decomp Transient"""
#GIF
s_ind=0
e_ind=time_len
num_pod_modes=4
spatial_modes, pod_modes, eigenvalues, vx_mnstrt, vy_mnstrt = POD2(vks, s_ind, e_ind, num_pod_modes)
fig, axes = plt.subplots(2,1, tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(vks.shape[1:3]), origin='upper',vmin=-1, vmax=1, aspect='auto')]
def vks_mnstrt_anim(vks_t):
    lines[0].set_data(vx_mnstrt[vks_t,:,:].T)
    lines[1].set_data(vy_mnstrt[vks_t,:,:].T)
    return lines
ani = animation.FuncAnimation(fig, vks_mnstrt_anim, blit=True, interval=vks.shape[0]-1,
    repeat=False)
ani.save('./out/vks_mnstrt_data.gif', "PillowWriter", fps=6);

#STATIC PLOT
plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=36,pad=padding)
ax.imshow(vx_mnstrt[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=36,pad=padding)
ax.imshow(vy_mnstrt[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_mnstrt_data.pdf', format="pdf", bbox_inches="tight")

"""POD Reconstruction"""
vks_reconstructed = pod_modes@spatial_modes.T
ux_reconstructed = vks_reconstructed[:,:domain_len].reshape(vks.shape[:-1])
uy_reconstructed = vks_reconstructed[:,domain_len:].reshape(vks.shape[:-1])
fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(domain_shape), origin='upper',vmin=-1, vmax=1, aspect='auto')]

def vks_pod_recon_anim(vks_t):
    lines[0].set_data(ux_reconstructed[vks_t,:,:].T)
    lines[1].set_data(uy_reconstructed[vks_t,:,:].T)
    return lines

ani = animation.FuncAnimation(fig, vks_pod_recon_anim, blit=True, interval=vks.shape[0]-1,
    repeat=False)
ani.save('./out/vks_pod_recon_full.gif', "PillowWriter", fps=6);

#STATIC PLOT
plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=48,pad=padding)
ax.imshow(ux_reconstructed[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=48,pad=padding)
ax.imshow(uy_reconstructed[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_pod_recon_full.pdf', format="pdf", bbox_inches="tight")

"""POD Modes"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
m=2
n = num_pod_modes//m
plt.figure(tight_layout=True)
for i in trange(num_pod_modes,desc='POD mode plot'):
    plt.subplot(num_pod_modes//2,2,i+1)
    plt.plot(pod_modes[:,i], 'k')
    plt.xlabel("Time $(t)$",fontsize=36)
    plt.ylabel('$\\alpha_{}(t)$'.format(i+1),fontsize=36)
plt.savefig('./out/vks_pod_modes_full.pdf', format="pdf", bbox_inches="tight")

"""POD Decay"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
s_ind=0
e_ind=time_len
num_pod_modes=time_len
spatial_modes, pod_modes, eigenvalues, vx_mnstrt, vy_mnstrt = POD2(vks, s_ind, e_ind, num_pod_modes)
total = sum(eigenvalues)
cumulative=[1]

for eig in tqdm(eigenvalues,desc='POD decay plot'):
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))
#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.yscale('log')
plt.savefig('./out/vks_pod_decay_full.pdf', format="pdf", bbox_inches="tight") 
print("Reconstructed value is {:.5f} for {} modes.".format(1-cumulative[4],4))


"""POD Decomp Non-Transient"""
#GIF
s_ind=100
e_ind=time_len
num_pod_modes=4
spatial_modes, pod_modes, eigenvalues, vx_mnstrt, vy_mnstrt = POD2(vks, s_ind, e_ind, num_pod_modes)
shape = [time_len-s_ind]+list(vks.shape[1:-1])
"""POD Reconstruction"""
vks_reconstructed = pod_modes@spatial_modes.T
ux_reconstructed = vks_reconstructed[:,:domain_len].reshape(shape)
uy_reconstructed = vks_reconstructed[:,domain_len:].reshape(shape)
fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(domain_shape), origin='upper',vmin=-1, vmax=1, aspect='auto')]

def vks_pod_recon_anim(vks_t):
    lines[0].set_data(ux_reconstructed[vks_t,:,:].T)
    lines[1].set_data(uy_reconstructed[vks_t,:,:].T)
    return lines

ani = animation.FuncAnimation(fig, vks_pod_recon_anim, blit=True, interval=shape[0]-1,
    repeat=False)
ani.save('./out/vks_pod_recon_nonT.gif', "PillowWriter", fps=6);

#STATIC PLOT
plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=48,pad=padding)
ax.imshow(ux_reconstructed[50].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=48,pad=padding)
ax.imshow(uy_reconstructed[50].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_pod_recon_nonT.pdf', format="pdf", bbox_inches="tight")

"""POD Modes"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
m=2
n = num_pod_modes//m
plt.figure(tight_layout=True)
for i in trange(num_pod_modes,desc='POD mode plot'):
    plt.subplot(num_pod_modes//2,2,i+1)
    plt.plot(pod_modes[:,i], 'k')
    plt.xlabel("Time $(t)$",fontsize=36)
    plt.ylabel('$\\alpha_{}(t)$'.format(i+1),fontsize=36)
plt.savefig('./out/vks_pod_modes_nonT.pdf', format="pdf", bbox_inches="tight")

"""POD Decay"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
s_ind=100
e_ind=time_len
num_pod_modes=time_len-s_ind
spatial_modes, pod_modes, eigenvalues, vx_mnstrt, vy_mnstrt = POD2(vks, s_ind, e_ind, num_pod_modes)
total = sum(eigenvalues)
cumulative=[1]

for eig in tqdm(eigenvalues,desc='POD decay plot'):
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))
#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.yscale('log')
plt.savefig('./out/vks_pod_decay_nonT.pdf', format="pdf", bbox_inches="tight") 
print("Reconstructed value is {:.5f} for {} modes.".format(1-cumulative[4],4))

"""DMD Decomposition Transient"""
s_ind = 0
e_ind = 398
pred = vks.shape[0]
num_dmd_modes=16
X, Atilde,Ur,Phi,Lambda,Sigma,b = DMD2(vks, s_ind, e_ind, num_dmd_modes)
dmd_modes = Phi.reshape([2]+list(vks.shape[1:-1])+[num_dmd_modes])
m=2
n = num_dmd_modes//m
plt.figure(tight_layout=True)
for i in trange(num_dmd_modes,desc='DMD modes'):
    plt.subplot(num_dmd_modes,4,4*i+1)
    plt.imshow(np.real(dmd_modes[0,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+2)
    plt.imshow(np.imag(dmd_modes[0,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+3)
    plt.imshow(np.real(dmd_modes[1,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+4)
    plt.imshow(np.imag(dmd_modes[1,:,:,i].T),aspect='auto')
plt.savefig('./out/vks_dmd_modes_full.pdf', format="pdf", bbox_inches="tight")
fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(domain_shape),
                origin='upper', vmin=-.4, vmax=.4, aspect='auto')]

def run(vks_t):
    Lambda_k = np.linalg.matrix_power(Lambda,vks_t)
    Xk=Phi@Lambda_k@b
    var1_xk = np.real(Xk[:domain_len].reshape(domain_shape))
    var2_xk = np.real(Xk[domain_len:].reshape(domain_shape))
    lines[0].set_data(var1_xk.T)
    lines[1].set_data(var2_xk.T)
    return lines

ani = animation.FuncAnimation(fig, run, blit=True, interval=pred,
    repeat=False)
ani.save('./out/vks_dmd_recon_full.gif', "PillowWriter", fps=6)

total = sum(Sigma)
cumulative=[1]

for eig in tqdm(Sigma,desc='POD decay plot'):
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))
#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.yscale('log')
plt.savefig('./out/vks_dmd_decay_full.pdf', format="pdf", bbox_inches="tight") 
print("Reconstructed value is {:.5f} for {} modes.".format(1-cumulative[16],16))


"""DMD Decomposition Non-Transient"""
s_ind = 100
e_ind = 398
pred = vks.shape[0]
num_dmd_modes=16
X, Atilde,Ur,Phi,Lambda,Sigma,b = DMD2(vks, s_ind, e_ind, num_dmd_modes)
dmd_modes = Phi.reshape([2]+list(vks.shape[1:-1])+[num_dmd_modes])
m=2
n = num_dmd_modes//m
plt.figure(tight_layout=True)
for i in trange(num_dmd_modes,desc='DMD modes'):
    plt.subplot(num_dmd_modes,4,4*i+1)
    plt.imshow(np.real(dmd_modes[0,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+2)
    plt.imshow(np.imag(dmd_modes[0,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+3)
    plt.imshow(np.real(dmd_modes[1,:,:,i].T),aspect='auto')
    plt.subplot(num_dmd_modes,4,4*i+4)
    plt.imshow(np.imag(dmd_modes[1,:,:,i].T),aspect='auto')
plt.savefig('./out/vks_dmd_modes_nonT.pdf', format="pdf", bbox_inches="tight")
fig, axes = plt.subplots(2,1,tight_layout=True)
lines = []
for ax in axes.flatten():
    lines = lines + [ax.imshow(np.zeros(domain_shape),
                origin='upper', vmin=-.4, vmax=.4, aspect='auto')]

def run(vks_t):
    Lambda_k = np.linalg.matrix_power(Lambda,vks_t)
    Xk=Phi@Lambda_k@b
    var1_xk = np.real(Xk[:domain_len].reshape(domain_shape))
    var2_xk = np.real(Xk[domain_len:].reshape(domain_shape))
    lines[0].set_data(var1_xk.T)
    lines[1].set_data(var2_xk.T)
    return lines

ani = animation.FuncAnimation(fig, run, blit=True, interval=pred,
    repeat=False)
ani.save('./out/vks_dmd_recon_nonT.gif', "PillowWriter", fps=6)

total = sum(Sigma)
cumulative=[1]

for eig in tqdm(Sigma,desc='POD decay plot'):
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))
#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.yscale('log')
plt.savefig('./out/vks_dmd_decay_nonT.pdf', format="pdf", bbox_inches="tight") 
print("Reconstructed value is {:.5f} for {} modes.".format(1-cumulative[16],16))


"""
KPP EXAMPLE

"""
kpp = KPP_DAT('./data/KPP.npz')
xv =  np.tile(np.linspace(-2,2,kpp.shape[1]),(kpp.shape[2],1))
yv = np.tile(np.linspace(-2.4,1.4,kpp.shape[2]),(kpp.shape[1],1)).T
domain_shape=kpp.shape[1:]
domain_len=kpp.shape[1]*kpp.shape[2]
time_len=kpp.shape[0]
"""Initial Data"""
plt.style.use('default')
plt.rcParams['font.family']='Times New Roman'
# plt.rcParams['font.size']=28
fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(projection='3d')
ax1.set_zlim(0, 10)
lines =[ax1.plot_surface(xv, yv, np.ones((20,20)), cmap=cm.coolwarm, linewidth=0)]

def kpp_init_anim(kpp_t):
    ax1.clear()
    ax1.set_zlim(0, 10)
    lines =[ax1.plot_surface(xv, yv, kpp[kpp_t,:,:], cmap=cm.coolwarm, linewidth=0)]
    return lines

ani = animation.FuncAnimation(fig, kpp_init_anim, blit=True, interval=kpp.shape[0]-1,
    repeat=False)
ani.save('./out/kpp_init_data.gif', "PillowWriter", fps=6);
"""POD Decomposition"""
s_ind = 0
e_ind = time_len
num_pod_modes = 4
spatial_modes, pod_modes, eigenvalues, kpp_mnstrt= PODKPP(kpp, s_ind, e_ind, num_pod_modes)
plt.style.use('default')
plt.rcParams['font.family']='Times New Roman'
# plt.rcParams['font.size']=28
fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(projection='3d')
min_,max_ = np.min(kpp_mnstrt), np.max(kpp_mnstrt)
val = max(abs(min_),abs(max_))
ax1.set_zlim(-val,val)
lines =[ax1.plot_surface(xv, yv, np.zeros(domain_shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)]

def kpp_mnstrt_anim(kpp_t):
    ax1.clear()
    ax1.set_zlim(-val,val)
    data = kpp_mnstrt[kpp_t,:].reshape(domain_shape)
    lines =[ax1.plot_surface(xv,yv,data,cmap=cm.coolwarm,linewidth=0)]
    return lines

ani = animation.FuncAnimation(fig, kpp_mnstrt_anim, blit=True, interval=kpp.shape[0]-1,
    repeat=False)
ani.save('./out/kpp_mnstrt_data.gif', "PillowWriter", fps=6);

"""POD Reconstruction"""
kpp_reconstructed = pod_modes@spatial_modes.T
kpp_reconstructed = kpp_reconstructed.reshape(kpp.shape)
plt.style.use('default')
plt.rcParams['font.family']='Times New Roman'
fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(projection='3d')
min_,max_ = np.min(kpp_reconstructed), np.max(kpp_reconstructed)
val = max(abs(min_),abs(max_))
ax1.set_zlim(-val,val)
lines =[ax1.plot_surface(xv, yv, np.zeros(domain_shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)]

def kpp_pod_lines(kpp_t):
    ax1.clear()
    ax1.set_zlim(-val,val)
    data = kpp_reconstructed[kpp_t,:]
    lines =[ax1.plot_surface(xv,yv,data,cmap=cm.coolwarm,linewidth=0)]
    return lines

ani = animation.FuncAnimation(fig, kpp_pod_lines, blit=True, interval=kpp.shape[0]-1,
    repeat=False)
ani.save('./out/kpp_pod_recon.gif', "PillowWriter", fps=6)

"""POD Modes"""
m=2
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
n = num_pod_modes//m
plt.figure(tight_layout=True)
for i in trange(num_pod_modes,desc='POD mode plot'):
    plt.subplot(num_pod_modes//2,2,i+1)
    plt.plot(pod_modes[:,i], 'k')
    plt.xlabel("Time $(t)$",fontsize=36)
    # plt.ylabel("Value",fontsize=36)
    plt.ylabel('$\\alpha_{}(t)$'.format(i+1),fontsize=36)
plt.savefig('./out/kpp_pod_modes.pdf', format="pdf", bbox_inches="tight")

"""POD Decay"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
s_ind=0
e_ind=time_len
num_pod_modes=time_len
spatial_modes, pod_modes, eigenvalues, kpp_mnstrt= PODKPP(kpp, s_ind, e_ind, num_pod_modes)
total = sum(eigenvalues)
cumulative=[1]

for eig in tqdm(eigenvalues,desc='POD decay'):
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))

#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.savefig('./out/kpp_pod_decay.pdf', format="pdf", bbox_inches="tight")

"""DMD Decomposition"""
plt.style.use('default')
plt.rcParams['font.family']='Times New Roman'
s_ind = 0
e_ind = 101
num_dmd_modes=4
X, Atilde,Ur,Phi,Lambda,Sigma,b = DMDKPP(kpp, s_ind, e_ind, num_dmd_modes)
dmd_modes = Phi.reshape(list(domain_shape)+[num_dmd_modes])
m=2
n = num_dmd_modes//m
fig = plt.figure(tight_layout=True)
for i in range(num_dmd_modes):
  ax1 = fig.add_subplot(num_dmd_modes,2,i*2+1,projection='3d')
  ax1.plot_surface(xv,yv,np.real(dmd_modes[:,:,i]),cmap=cm.coolwarm,linewidth=0)
  ax2 = fig.add_subplot(num_dmd_modes,2,i*2+2,projection='3d')
  ax2.plot_surface(xv,yv,np.imag(dmd_modes[:,:,i]),cmap=cm.coolwarm,linewidth=0)
plt.savefig('./out/kpp_dmd_modes.pdf', format="pdf", bbox_inches="tight")

"""
EE EXAMPLE

"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
ee=EE_DAT('./data/EulerEqs.npz')
x = np.linspace(-5,5,ee.shape[1])
domain_shape=[ee.shape[1]]
domain_len=ee.shape[1]
time_len=ee.shape[0]

fig, axes = plt.subplots(3,1, tight_layout=True)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
lims = [(-2,5), (-5,5),(-5,11)]
lines = []
for i,ax in enumerate(axes.flatten()):
    ax.set_ylim(lims[i])
    lines = lines + ax.plot(x,np.zeros(domain_shape), 'k')
def run_ee_init(ee_t):
    lines[0].set_ydata(ee[ee_t,:,0])
    lines[1].set_ydata(ee[ee_t,:,1])
    lines[2].set_ydata(ee[ee_t,:,2])
    return lines
ani = animation.FuncAnimation(fig, run_ee_init, blit=False, interval=time_len-1,
    repeat=False)
ani.save('./out/ee_init_data.gif',"PillowWriter", fps=6)

"""POD Decompositioin"""
s_ind=0
e_ind=time_len
num_pod_modes=4
spatial_modes, pod_modes, eigenvalues, rho_mnstrt, v_mnstrt, e_mnstrt = POD3(ee, s_ind, e_ind, num_pod_modes)
ee_mnstrt = np.moveaxis(np.array([rho_mnstrt, v_mnstrt, e_mnstrt]),0,-1)
fig, axes = plt.subplots(3,1, tight_layout=True)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
lims = [(-2,5), (-5,5),(-5,11)]
lines = []
for i,ax in enumerate(axes.flatten()):
    ax.set_ylim(lims[i])
    lines = lines + ax.plot(x,np.zeros(domain_shape), 'k')
def run_ee_mnstrt(ee_t):
    lines[0].set_ydata(ee_mnstrt[ee_t,:,0])
    lines[1].set_ydata(ee_mnstrt[ee_t,:,1])
    lines[2].set_ydata(ee_mnstrt[ee_t,:,2])
    return lines
ani = animation.FuncAnimation(fig, run_ee_mnstrt, blit=False, interval=time_len-1,
    repeat=False)
ani.save('./out/ee_mnstrt_data.gif',"PillowWriter", fps=6)

"""POD Reconstruction"""
ee_reconstructed = pod_modes@spatial_modes.T
rho_reconstructed = ee_reconstructed[:,:domain_len]
v_reconstructed = ee_reconstructed[:,domain_len:2*domain_len]
e_reconstructed = ee_reconstructed[:,2*domain_len:]
ee_reconstructed = np.moveaxis(np.array([rho_reconstructed, v_reconstructed, e_reconstructed]),0,-1)
fig, axes = plt.subplots(3,1, tight_layout=True)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
lims = [(-2,5), (-5,5),(-5,11)]
lines = []
for i,ax in enumerate(axes.flatten()):
    ax.set_ylim(lims[i])
    lines = lines + ax.plot(x,np.zeros(domain_shape), 'k')
def run_ee_pod_recon(ee_t):
    lines[0].set_ydata(ee_mnstrt[ee_t,:,0])
    lines[1].set_ydata(ee_mnstrt[ee_t,:,1])
    lines[2].set_ydata(ee_mnstrt[ee_t,:,2])
    return lines
ani = animation.FuncAnimation(fig, run_ee_pod_recon, blit=False, interval=time_len-1,
    repeat=False)
ani.save('./out/ee_pod_recon.gif',"PillowWriter", fps=6)

"""POD Modes"""
m=2
n = num_pod_modes//m
plt.figure(tight_layout=True)
for i in range(num_pod_modes):
    plt.subplot(num_pod_modes//2,2,i+1)
    plt.plot(pod_modes[:,i], 'k')
    plt.xlabel("Time $(t)$",fontsize=36)
    # plt.ylabel("Value",fontsize=36)
    plt.ylabel('$\\alpha_{}(t)$'.format(i+1),fontsize=36)
plt.savefig('./out/ee_pod_modes.pdf', format="pdf", bbox_inches="tight")

"""POD Decay"""
plt.style.use('classic')
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['xtick.minor.size']=0
plt.rcParams['ytick.minor.size']=0
s_ind=0
e_ind=time_len
num_pod_modes=time_len
spatial_modes, pod_modes, eigenvalues, rho_mnstrt, v_mnstrt, e_mnstrt = POD3(ee, s_ind, e_ind, num_pod_modes)
total = sum(eigenvalues)
cumulative=[1]

for eig in eigenvalues:
    val = eig/total
    cumulative = cumulative + [cumulative[-1]-val]

x2 = np.arange(0,len(cumulative))
#CUMULATIVE PLOT
plt.figure(tight_layout=True)
plt.plot(x2,cumulative, 'k')
plt.xlabel('Mode $(\\alpha_i)$',fontsize=36)
plt.ylabel('Mode Decay $(\eta)$',fontsize=36)
plt.savefig('./out/ee_pod_decay.pdf', format="pdf", bbox_inches="tight") 

"""DMD Decomposition"""
s_ind = 0
e_ind = 101
pred = time_len
num_dmd_modes=4
X, Atilde,Ur,Phi,Lambda,Sigma,b = DMD3(ee, s_ind, e_ind, num_dmd_modes)
dmd_modes = Phi.reshape([3]+list(domain_shape)+[num_dmd_modes])
m=2
n = num_dmd_modes//m
plt.figure(figsize=(20,5*num_dmd_modes), tight_layout=True)
for i in range(num_dmd_modes):
    plt.subplot(num_dmd_modes,6,6*i+1)
    plt.plot(np.real(dmd_modes[0,:,i].T))
    plt.subplot(num_dmd_modes,6,6*i+2)
    plt.plot(np.imag(dmd_modes[0,:,i].T))
    plt.subplot(num_dmd_modes,6,6*i+3)
    plt.plot(np.real(dmd_modes[1,:,i].T))
    plt.subplot(num_dmd_modes,6,6*i+4)
    plt.plot(np.imag(dmd_modes[0,:,i].T))
    plt.subplot(num_dmd_modes,6,6*i+5)
    plt.plot(np.real(dmd_modes[2,:,i].T))
    plt.subplot(num_dmd_modes,6,6*i+6)
    plt.plot(np.imag(dmd_modes[2,:,i].T))
plt.savefig('./out/ee_dmd_modes.pdf', format="pdf", bbox_inches="tight")

x = np.linspace(-5,5,ee.shape[1])
fig, axes = plt.subplots(3,1, tight_layout=True)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
lims = [(-2,5), (-5,5),(-5,11)]
lines = []
for i,ax in enumerate(axes.flatten()):
    ax.set_ylim(lims[i])
    lines = lines + ax.plot(x,np.zeros(domain_shape), 'k')
def run_ee_dmd_recon(ee_t):
    Lambda_k = np.linalg.matrix_power(Lambda,ee_t)
    Xk=Phi@Lambda_k@b
    var1_xk = np.real(Xk[:domain_len])
    var2_xk = np.real(Xk[domain_len:2*domain_len])
    var3_xk = np.real(Xk[2*domain_len:])
    lines[0].set_ydata(var1_xk)
    lines[1].set_ydata(var2_xk)
    lines[2].set_ydata(var3_xk)
    return lines

ani = animation.FuncAnimation(fig, run_ee_dmd_recon, blit=True, interval=time_len,
    repeat=False)
ani.save('./out/ee_dmd_recon.gif', "PillowWriter", fps=6)
#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

#SELF IMPORTS
import sys
sys.path.append('./')

from lib.datasets import VKS_DAT
from lib.decomp.pod import POD2
from lib.utils.misc import set_outdir

plt.style.use('classic')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
padding=15

"""LOAD DATA"""
vks = VKS_DAT('./data/VKS.pkl')
domain_shape=vks.shape[1:3]
domain_len=vks.shape[1]*vks.shape[2]
time_len=vks.shape[0]

"""MEAN SUBTR DATA"""
param1 = vks[:,:,:,0]
param2 = vks[:,:,:, 1]

param1_mean = np.mean(param1, axis=0)[np.newaxis, ...]
param2_mean = np.mean(param2, axis=0)[np.newaxis, ...]

# fluctuating components: taking U-Um
param1_flux = param1 - param1_mean
param2_flux = param2 - param2_mean

"""TRANSIENT PLOT"""
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=36,pad=padding)
ax.imshow(param1_flux[50].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=36,pad=padding)
ax.imshow(param2_flux[50].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_transient.pdf', format="pdf", bbox_inches="tight")

"""QUASI PERIODIC PLOT"""
fig=plt.figure(tight_layout=True)
ax = plt.subplot(211)
ax.set_title('$u\'_x$',fontsize=36,pad=padding)
ax.imshow(param1_flux[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
ax = plt.subplot(212)
ax.set_title('$u\'_y$',fontsize=36,pad=padding)
ax.imshow(param2_flux[150].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
plt.savefig('./out/vks_quasi_periodic.pdf', format="pdf", bbox_inches="tight")

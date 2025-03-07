import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np







######################
##### ANIMATIONS #####
######################
def vks_animate(data,args):
    plt.style.use('classic')

    fig, axes = plt.subplots(2,1,tight_layout=True)
    lines = []
    for ax in axes.flatten():
        lines = lines + [ax.imshow(np.zeros((data.shape[1:3])), origin='upper', vmin =-.4,vmax =.4, aspect='auto')]

    def run_vks_lines(vks_t):
        lines[0].set_data(data[vks_t,:,:,0].T)
        lines[1].set_data(data[vks_t,:,:,1].T)
        return lines
    ani = animation.FuncAnimation(fig, run_vks_lines, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    return 1
    
def kpp_animate(data,args):
    plt.style.use('default')
    xv =  np.tile(np.linspace(-2,2,data.shape[1]),(data.shape[2],1))
    yv = np.tile(np.linspace(-2.4,1.4,data.shape[2]),(data.shape[1],1)).T

    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(projection='3d')
    val = 5
    ax1.set_zlim(-val,val)
    lines =[ax1.plot_surface(xv, yv, np.ones(data.shape[1:]), cmap=cm.coolwarm, linewidth=0, antialiased=False)]
    
    def run_kpp_lines(kpp_t):
        ax1.clear()
        ax1.set_zlim(-val,val)
        lines =[ax1.plot_surface(xv, yv, data[kpp_t], cmap=cm.coolwarm, linewidth=0, antialiased=False)]
        return lines

    ani = animation.FuncAnimation(fig, run_kpp_lines, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    return 1

def ee_animate(data,args):
    x = np.linspace(-5,5,data.shape[1])

    fig, axes = plt.subplots(3,1, figsize=(12,12), tight_layout=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    lims = [(-2,5), (-5,5),(-5,11)]
    lines = []
    for i,ax in enumerate(axes.flatten()):
        ax.set_ylim(lims[i])
        lines = lines + ax.plot(x,np.zeros(data.shape[1]), 'k')

    def run_ee_lines(ee_t):
        lines[0].set_ydata(data[ee_t,:,0,0])
        lines[1].set_ydata(data[ee_t,:,0,1])
        lines[2].set_ydata(data[ee_t,:,0,2])
        return lines

    ani = animation.FuncAnimation(fig, run_ee_lines, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    return 1


def fiber_animate(data,args):
    # run_vae.py adds an extra dimension to the data. We check for that here
    if data.ndim > 2:
        data = data[0,:,:]
    
    plt.style.use('classic')

    fig, axes = plt.subplots(1,1,tight_layout=True)
    axes.set_ylim([-2,2])
    x = np.linspace(0,5,data.shape[1])
    lines = axes.plot(x,np.zeros(data.shape[1]), 'k')
    
    def run_fiber_lines(fiber_t):
        lines[0].set_ydata(data[fiber_t,:])
        return lines
    
    ani = animation.FuncAnimation(fig, run_fiber_lines, frames = data.shape[0]-1, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'tinkleDick.gif', "PillowWriter", fps=12)
    return 1

def fiber_param_animate(data, args):
    plt.style.use('classic')

    fig, axes = plt.subplots(2,2,tight_layout=True)
    axes[0,0].set_ylim([-0.5,2])
    axes[0,1].set_ylim([-0.5,2])
    axes[1,0].set_ylim([-0.5,2])
    axes[1,1].set_ylim([-0.5,2])
    x = np.linspace(0,5,data.shape[2])
    lines1 = axes[0,0].plot(x,np.zeros(data.shape[2]), 'k')
    lines2 = axes[0,1].plot(x,np.zeros(data.shape[2]), 'k')
    lines3 = axes[1,0].plot(x,np.zeros(data.shape[2]), 'k')
    lines4 = axes[1,1].plot(x,np.zeros(data.shape[2]), 'k')
    
    def run_fiber_lines(fiber_t):
        lines1[0].set_ydata(data[0,fiber_t,:])
        lines2[0].set_ydata(data[1,fiber_t,:])
        lines3[0].set_ydata(data[2,fiber_t,:])
        lines4[0].set_ydata(data[3,fiber_t,:])
        return lines1, lines2, lines3, lines4
    
    ani = animation.FuncAnimation(fig, run_fiber_lines, frames = data.shape[1]-1, blit=False, interval=data.shape[1]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_reconSDFSDFSDF').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=12)
    return 1


"""
ANIMATION HEADER

"""

ANIM = {'VKS':vks_animate,'KPP':kpp_animate, 'EE':ee_animate, 'FIB': fiber_animate}
def data_animation(data,args):

    ANIM[args.dataset](data,args)
    # print(ani)
    # end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    # ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    # if args.verbose: plt.show()

    return 1

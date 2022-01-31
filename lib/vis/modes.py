#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

""" KNOWN USEFUL """
def pod_mode_to_true(dataloader,modes, args):
    DL = dataloader
    spatial_modes = DL.spatial_modes
    normalized = (modes*DL.std_data + DL.mean_data)
    true = np.matmul(normalized, spatial_modes.T)

    if args.dataset == "VKS":
        if len(true.shape)>2:
            true = true[-1]
        pod_x = true[:, :DL.Nxy]
        pod_y = true[:, DL.Nxy:]

        shape = [true.shape[0], DL.data_init.shape[0], DL.data_init.shape[1]]
        true_x = pod_x.reshape(pod_x.shape[0], shape[1], shape[2])
        true_x = true_x.swapaxes(0,-1)
        true_y = pod_y.reshape(pod_y.shape[0], shape[1], shape[2])
        true_y = true_y.swapaxes(0,-1)

        true = np.array([true_x,true_y])
    return true.T


"""  EIGEN VALUE DECAY PLOT """
def eig_decay(dataloader,args):
    #INITIALIZE
    DL = dataloader
    total = DL.lv.sum()
    decay=[1]
    true_switch = False #prevent overflow that occurs from dividing by total
    #CALCULATE DECAY
    # for eig in DL.lv[:args.modes]:
    for eig in DL.lv:
        if eig < 1e-14:
            true_switch=True
        if true_switch:
            val = 0
        else:
            val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    x = np.arange(0,len(DL.lv)+1)
    #GENERATE-Y TICKS
    yticks = []
    i=0
    power = 1/min(decay[np.where(decay>0)])
    while(10**(i-1)<power):
        yticks=yticks+[10**(-i)]
        i=i+1
    num = len(yticks)//5

    plt.figure(tight_layout=True)
    plt.plot(x,decay, 'k')
    #AXES
    plt.yscale('log')
    plt.yticks(yticks[::num])
    # num = (args.modes-1)//5
    num = (len(DL.lv)+1)//5
    plt.xticks(list(x[::num])+[x[-1]])
    #TITLES
    plt.xlabel('Mode ($r$)')
    plt.ylabel('Decay ($\\varepsilon$)')
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_decay').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    print("Reconstructed value is {:.5f} for {} modes.".format(1-decay[args.modes],args.modes))
    
    return 1

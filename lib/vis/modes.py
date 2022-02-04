#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

""" KNOWN USEFUL """
def pod_mode_to_true(dataset,modes,args):
    print(modes.shape)
    data = dataset
    spatial_modes = data.spatial_modes
    normalized = (modes*data.std_data+data.mean_data)
    true = np.matmul(normalized,spatial_modes.T)

    if args.dataset == "VKS":
        if len(true.shape)>2:
            true = true[0]
        pod_x = true[:, :data.domain_len]
        pod_y = true[:, data.domain_len:]

        shape = [true.shape[0]] + list(data.domain_shape)
        true_x = pod_x.reshape(shape)
        true_y = pod_y.reshape(shape)
        true = np.array([true_x.T,true_y.T])
    return true.T


"""  EIGEN VALUE DECAY PLOT """
def eig_decay(dataset,args):
    plt.style.use('classic')
    #INITIALIZE
    data = dataset
    total = data.lv.sum()
    decay=[1]
    true_switch = False #prevent overflow that occurs from dividing by total
    #CALCULATE DECAY
    # for eig in data.lv[:args.modes]:
    for eig in data.lv:
        if eig < 1e-14:
            true_switch=True
        if true_switch:
            val = 0
        else:
            val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    x = np.arange(0,len(data.lv)+1)
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
    num = (len(data.lv)+1)//5
    plt.xticks(list(x[::num])+[x[-1]])
    #TITLES
    plt.xlabel('Mode ($r$)',fontsize=28)
    plt.ylabel('Decay ($\\varepsilon$)',fontsize=28)
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_decay').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    print("Reconstructed value is {:.5f} for {} modes.".format(1-decay[args.modes],args.modes))
    
    return 1


def plot_modes(dataset,predictions,true,times,args):
    plt.style.use('classic')
    data = dataset
    for j in range(3):
        plt.figure(tight_layout=True)
        for i,node in enumerate(predictions.T):
            plt.subplot(args.modes//2,2,i+1)
            plt.plot(node, 'r', label='Prediction')
            plt.plot(true.cpu().T[i], 'k--', label='True')
            plt.xlabel("Time")
            plt.ylabel("$\\alpha_{}$".format(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
        #OUTPUT
        end_str = str(args.dataset+'_'+args.model+'_modes').lower()
        plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
        if args.verbose: plt.show()

    return 1

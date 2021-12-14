#IMPORTS
from torchdiffeq import odeint
import torch.optim as optim
from tqdm import tqdm,trange


import sys
sys.path.append('../')

from sci.lib.loader import *
from sci.lib.vae.models import *
from sci.lib.vae.parser import *
from sci.lib.utils import *
from sci.lib.vis.vae import *

def main(parse=None):

    #ARGS INPUT
    try:
        sys_args = sys.argv[1:]
    except:
        sys_args = []

    args = parse_args(sys_args) if parse==None else parse
    if args.verbose:
        print('Parsed Arguments')
        for arg in vars(args):
            print('\t',arg, getattr(args, arg))

    #FORMAT OUTDIR
    set_outdir(args.out_dir, args)

    #DATA LOADER
    DL = STD_LOADER(args)

    #MODEL DIMENSIONS
    print('Generating ...\t Model: NODE ')
    obs_dim = DL.train_data.shape[2]
    latent_dim = obs_dim - args.latent_dim
    layers_node = [latent_dim] + args.layers_node + [latent_dim]

    MODELS = {'NODE' : NODE(df = LatentODE(layers_node)),
            'HBNODE' : hbnode(LatentODE(layers_node))}

    if args.model == "HBNODE":
        latent_dim = latent_dim*2
    #NETWORKS
    enc = Encoder(latent_dim, obs_dim, args.units_enc, args.layers_enc)
    node = MODELS[args.model]
    dec = Decoder(latent_dim, obs_dim, args.units_dec, args.layers_dec)
    params = (list(enc.parameters()) + list(node.parameters()) + list(dec.parameters()))

    #LEARNING UTILITIES
    optimizer = optim.AdamW(params, lr= args.lr)
    loss_meter_t = RunningAverageMeter()
    meter_train = RunningAverageMeter()
    meter_valid = RunningAverageMeter()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.99, patience=5, verbose=False, threshold=1e-5,
                                                    threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
    criterion = torch.nn.MSELoss()
    lossTrain = []
    lossVal = []

    #TRAINING
    print('Training ... \t Iterations: {}'.format(args.epochs))
    for itr in trange(1, args.epochs + 1):

        optimizer.zero_grad()

        #SCHEDULE
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        scheduler.step(metrics=loss_meter_t.avg)

        #FORWARD STEP
        out_enc = enc.forward(DL.obs_t)
        qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size())
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        zt = odeint(node, z0, DL.train_times, method='rk4').permute(1, 0, 2)
        output_vae_t = dec(zt)

        # LOSS
        pz0_mean = pz0_logvar = torch.zeros(z0.size())
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        kl_loss = torch.mean(analytic_kl, dim=0)
        loss = criterion(output_vae_t, DL.train_data) + kl_loss

        #BACK PROP
        loss.backward()
        optimizer.step()
        loss_meter_t.update(loss.item())
        meter_train.update(loss.item() - kl_loss.item())
        lossTrain.append(meter_train.avg)

        #VALIDATION
        with torch.no_grad():

            enc.eval()
            node.eval()
            dec.eval()

            zv = odeint(node, z0, DL.valid_times, method='rk4').permute(1, 0, 2)
            output_vae_v = dec(zv)

            loss_v = criterion(output_vae_v[:, DL.tr_ind:],
                                DL.valid_data[:, DL.tr_ind:])

            meter_valid.update(loss_v.item())
            lossVal.append(meter_valid.avg)

            enc.train()
            node.train()
            dec.train()

        #OUTPUT
        if itr % 100 == 0:
            output_vae = (output_vae_v.cpu().detach().numpy()) * DL.std_data + DL.mean_data
        if itr == args.epochs:
            plotNODE(output_vae, DL.data[:DL.val_ind, :], lossTrain, lossVal, itr, DL.tr_ind, args.out_dir, args)

        if np.isnan(lossTrain[itr - 1]):
            break


    #SAVE MODEL DATA
    torch.save(enc.state_dict(), args.out_dir + './pth/enc.pth')
    torch.save(node.state_dict(), args.out_dir + './pth/node.pth')
    torch.save(dec.state_dict(), args.out_dir + './pth/dec.pth')

    #FORWARD STEP TEST DATA
    with torch.no_grad():

        enc.eval()
        node.eval()
        dec.eval()

        ze = odeint(node, z0, DL.eval_times, method='rk4').permute(1, 0, 2)
        output_vae_e = dec(ze)

        enc.train()
        node.train()
        dec.train()

    #SAVE TEST DATA
    data_NODE = (output_vae_e.cpu().detach().numpy()) * DL.std_data + DL.mean_data
    with open(args.out_dir + './pth/data_node8.pth', 'wb') as f:
        pickle.dump(data_NODE, f)

if __name__ == "__main__":
    main()
#IMPORTS
import argparse
from torchdiffeq import odeint
import torch.optim as optim
from tqdm import tqdm,trange
import numpy as np


import sys
sys.path.append('./')

from lib.datasets import VAE_DATASET
from lib.models.vae import *
from lib.utils.misc import *
from lib.vis.modes import *
from lib.vis.animate import data_animation
from lib.vis.reconstruct import data_reconstruct


"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
#DATA PARAMS
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE].')
data_parser.add_argument('--data_dir', type=str, default='data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='out/',
                    help='Directory of output from cwd: sci.')
data_parser.add_argument('--modes', type = int, default =8,
                    help = 'POD reduction modes.\nNODE model parameters.')
data_parser.add_argument('--tstart', type = int, default=100,
                    help='Start time for reduction along time axis.')
data_parser.add_argument('--tstop', type=int, default=400,
                    help='Stop time for reduction along time axis.' )
data_parser.add_argument('--tr_ind', type = int, default=75,
                    help='Time index for training data.')
data_parser.add_argument('--val_ind', type=int, default=100,
                    help='Time index for validation data.' )
#MODEL PARAMS
model_parser = parser.add_argument_group('Model Parameters')
model_parser.add_argument('--model', type=str, default='NODE',
                    help='Dataset types: [NODE , HBNODE].')
model_parser.add_argument('--epochs', type=int, default=500,
                    help='Training epochs.')
model_parser.add_argument('--latent_dim', type=int, default=6,
                    help = 'Size of latent dimension')
model_parser.add_argument('--layers_enc', type=int, default=4,
                help='Encoder Layers.')
model_parser.add_argument('--units_enc', type=int, default=10,
                    help='Encoder units.')
model_parser.add_argument('--layers_node', type=list, default=[12],
                help='NODE Layers.')
model_parser.add_argument('--units_dec', type=int, default=41,
                    help='Training iterations.')
model_parser.add_argument('--layers_dec', type=int, default=4,
                help='Encoder Layers.')
model_parser.add_argument('--lr', type=float, default=0.00153,
                    help = 'Initial learning rate.')
model_parser.add_argument('--factor', type=float, default=0.95,
                    help = 'Factor for reducing learning rate.')
#UNIQUE PARAMS
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', type=bool, default=False,
                help='Display full NN and all plots.')
#PARSE
args, unknown = parser.parse_known_args()
if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))

"""INITIALIZE"""
#FORMAT OUTDIR
set_outdir(args.out_dir, args)
#LOAD DATA
vae = VAE_DATASET(args)

"""GENERATE MODEL"""
print('Generating ...\t Model: VAE '+args.model)
obs_dim = vae.train_data.shape[2]
latent_dim = obs_dim - args.latent_dim
layers_node = [latent_dim] + args.layers_node + [latent_dim]

MODELS = {'NODE' : NODE(df = LatentODE(layers_node)),
        'HBNODE' : HBNODE(LatentODE(layers_node))}

if args.model == "HBNODE":
    latent_dim = latent_dim*2
#NETWORKS
enc = Encoder(latent_dim, obs_dim, args.units_enc, args.layers_enc)
node = MODELS[args.model]
dec = Decoder(latent_dim, obs_dim, args.units_dec, args.layers_dec)
params = (list(enc.parameters()) + list(node.parameters()) + list(dec.parameters()))

"""GENERATE TRAINING UTILITIES"""
optimizer = optim.AdamW(params, lr= args.lr)
loss_meter_t = RunningAverageMeter()
meter_train = RunningAverageMeter()
meter_valid = RunningAverageMeter()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                factor=args.factor, patience=5, verbose=False, threshold=1e-5,
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
    out_enc = enc.forward(vae.obs_t)
    qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size())
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    zt = odeint(node, z0, vae.train_times, method='rk4').permute(1, 0, 2)
    output_vae_t = dec(zt)

    # LOSS
    pz0_mean = pz0_logvar = torch.zeros(z0.size())
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1)
    kl_loss = torch.mean(analytic_kl, dim=0)
    loss = criterion(output_vae_t, vae.train_data) + kl_loss

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

        zv = odeint(node, z0, vae.valid_times, method='rk4').permute(1, 0, 2)
        output_vae_v = dec(zv)

        loss_v = criterion(output_vae_v[:, args.tr_ind:],
                            vae.valid_data[:, args.tr_ind:])

        meter_valid.update(loss_v.item())
        lossVal.append(meter_valid.avg)

        enc.train()
        node.train()
        dec.train()

    #OUTPUT
    if itr % args.epochs == 0:
        output_vae = (output_vae_v.cpu().detach().numpy()) * vae.std_data + vae.mean_data
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

    ze = odeint(node, z0, vae.eval_times, method='rk4').permute(1, 0, 2)
    output_vae_e = dec(ze)

    enc.train()
    node.train()
    dec.train()

#SAVE TEST DATA
data_NODE = (output_vae_e.cpu().detach().numpy()) * vae.std_data + vae.mean_data
with open(args.out_dir + './pth/vae_'+args.model+'_modes.pth', 'wb') as f:
    pickle.dump(data_NODE, f)

eig_decay(vae,args)
#INVERT OVER TIME
idx = [i for i in range(vae.valid_data.size(0) - 1, -1, -1)]
idx = torch.LongTensor(idx)
obs_t = vae.valid_data.index_select(0, idx)
out_enc = enc.forward(obs_t)
qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
epsilon = torch.randn(qz0_mean.size())
z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
zt = odeint(node, z0, vae.valid_times, method='rk4').permute(1, 0, 2)
predictions = dec(zt).detach().numpy()
normalized = (predictions*vae.std_data+vae.mean_data)
mode_prediction(vae,predictions,vae.valid_data,np.arange(args.tstart,args.val_ind),args)
val_recon = pod_mode_to_true(vae,predictions,args)
data_reconstruct(val_recon,-1,args)
data_animation(val_recon,args)

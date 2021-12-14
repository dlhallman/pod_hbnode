# Import Packages
import torch.nn as nn
import torch

from sci.lib.utils import *

class Encoder(nn.Module):

    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Encoder, self).__init__()

        #VAE
        self.rnn = nn.GRU(obs_dim, hidden_units, hidden_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_units, latent_dim * 2)
        
    def forward(self, x):
        y, _ = self.rnn(x)
        y = y[:, -1, :]
        y = self.h2o(y)
        return y
class LatentODE(nn.Module):

    def __init__(self, layers):
        super(LatentODE, self).__init__()

        self.act = nn.Tanh()
        self.layers = layers

        #FEED FORWARD
        arch = []
        for ind_layer in range(len(self.layers) - 2):
            layer = nn.Linear(self.layers[ind_layer], self.layers[ind_layer + 1])
            nn.init.xavier_uniform_(layer.weight)
            arch.append(layer)
        layer = nn.Linear(self.layers[-2], self.layers[-1])
        layer.weight.data.fill_(0)
        arch.append(layer)

        #LIN OUT
        self.linear_layers = nn.ModuleList(arch)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        for ind in range(len(self.layers) - 2):
            x = self.act(self.linear_layers[ind](x))
        y = self.linear_layers[-1](x)
        return y

class Decoder(nn.Module):

    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Decoder, self).__init__()

        self.act = nn.Tanh()
        #SEQ2SEQ
        self.rnn = nn.GRU(latent_dim, hidden_units, hidden_layers, batch_first=True)
        self.h1 = nn.Linear(hidden_units, hidden_units - 5)
        self.h2 = nn.Linear(hidden_units - 5, obs_dim)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.h1(y)
        y = self.act(y)
        y = self.h2(y)
        return y


# Neural ODES
class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
        self.elem_t = None
    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)
    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)



class hbnode(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        # Activation for dh, GHBNODE only
        self.actv_h = nn.Identity() if actv_h is None else actv_h
    def forward(self, t, x):
        self.nfe += 1
        h, m = torch.split(x, 6, dim=1)
        dh = self.actv_h(- m)
        dm = self.df(t, h) - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)#.to(device)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)
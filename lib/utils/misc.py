# Import Packages
import os
import pickle
import torch
import torch.nn as nn


# LEARNING UTILS
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# MODEL UTILS
class NLayerNN(nn.Module):
    def __init__(self, *args, actv=nn.ReLU()):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(args)):
            self.linears.append(nn.Linear(args[i], args[i+1]))
        self.actv = actv
    def forward(self, x):
        for i in range(self.layer_cnt):
            x = self.linears[i](x)
            if i < self.layer_cnt - 1:
                x = self.actv(x)
        return x

class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)
zeronet = Zeronet()

class TVnorm(nn.Module):
    def __init__(self):
        super(TVnorm, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        return torch.norm(v, 1)

class NormAct(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
    def forward(self, x):
        x = x - self.bound + 1
        x = self.relu(x) * self.elu(-x) + 1
        return x

class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen
    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param
    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True
    def unfreeze(self):
        self.frozen = False
    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())

# DIRECTORY UTILS
def set_outdir(OUTPUT_DIR, args):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR+'./pth/'):
        os.makedirs(OUTPUT_DIR+'./pth/')
    top_str = args.model+'_'+args.dataset+'_'
    with open(OUTPUT_DIR + '/pth/'+top_str+'.pth', 'wb') as f:
        pickle.dump({' Model Arguments' : args}, f)




class NLayerNN(nn.Module):
    def __init__(self, *args, actv=nn.ReLU()):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(args)):
            self.linears.append(nn.Linear(args[i], args[i+1]))
        self.actv = actv

    def forward(self, x):
        for i in range(self.layer_cnt):
            x = self.linears[i](x)
            if i < self.layer_cnt - 1:
                x = self.actv(x)
        return x


class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

zeronet = Zeronet()

class TVnorm(nn.Module):
    def __init__(self):
        super(TVnorm, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        return torch.norm(v, 1)


class NormAct(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = x - self.bound + 1
        x = self.relu(x) * self.elu(-x) + 1
        return x


class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())
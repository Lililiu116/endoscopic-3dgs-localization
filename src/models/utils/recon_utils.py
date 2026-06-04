# code from https://github.com/qq456cvb/UKPGAN/blob/pytorch/model_utils.py
import torch.nn as nn
import math
import numpy as np


def mlp(input_dim, layer_dims, bn=None):
    layer_dims.insert(0, input_dim)
    layers = nn.ModuleList()
    for i in range(len(layer_dims) - 2):
        layers.append(
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.BatchNorm1d(layer_dims[i + 1]) if bn else nn.Identity(),
                nn.ReLU()
            )
        )
    
    layers.append(
        nn.Linear(layer_dims[-2], layer_dims[-1])
    )
    return nn.Sequential(*layers)


def mlp_conv(input_dim, layer_dims, bn=None):
    layer_dims.insert(0, input_dim)
    layers = nn.ModuleList()
    for i in range(len(layer_dims) - 2):
        layers.append(
            nn.Sequential(
                nn.Conv1d(layer_dims[i], layer_dims[i + 1], kernel_size=1),
                nn.BatchNorm1d(layer_dims[i + 1]) if bn else nn.Identity(),
                nn.ReLU()
            )
        )
    
    layers.append(
        nn.Conv1d(layer_dims[-2], layer_dims[-1], kernel_size=1)
    )
    return nn.Sequential(*layers)


# Number of children per tree levels for 2048 output points
def get_arch(nlevels, npts):
    tree_arch = {}
    tree_arch[2] = [32, 64]
    tree_arch[4] = [4, 8, 8, 8]
    # tree_arch[6] = [2, 4, 4, 4, 4, 4]
    tree_arch[6] = [2, 4, 4, 4, 4, 2]
    tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

    # logmult = int(math.log2(npts/2048))
    # assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    # while logmult > 0:
    #     last_min_pos = np.where(arch==np.min(arch))[0][-1]
    #     arch[last_min_pos]*=2
    #     logmult -= 1
    return arch

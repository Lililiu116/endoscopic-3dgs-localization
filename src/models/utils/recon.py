import numpy as np
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import torch.nn as nn
from .recon_utils import get_arch, mlp, mlp_conv



def get_topnet_params():
    return SimpleNamespace(
    nlevels=6,
    nfeat=64,
    code_nfts=128,
    )


# code from https://github.com/qq456cvb/UKPGAN/blob/pytorch/model_utils.py
class TopNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.nfeat = opt.topnet.nfeat
        self.code_nfts = opt.topnet.code_nfts
        self.nin = opt.topnet.nfeat + opt.topnet.code_nfts
        self.nout = opt.topnet.nfeat
        self.npoints = opt.gs_mae.npoints
        self.tarch = get_arch(opt.topnet.nlevels, self.npoints)
        level0 = nn.Sequential(
            mlp(self.code_nfts, [256, 64, self.nfeat * int(self.tarch[0])], bn=False),
            nn.Tanh()
        )
        self.levels = nn.ModuleList([level0])
        nin = self.nin
        for i in range(1, len(self.tarch)):
            if i == len(self.tarch) - 1:
                nout = 3
                bn = False
            else:
                nout = self.nout
                bn = False
                
            level = nn.Sequential(
                self.create_level(i, nin, nout, bn),
                nn.Tanh()
            )
            self.levels.append(level)
            # nin = nout * int(self.tarch[i]) + self.code_nfts
            # print(nin, nout * int(self.tarch[i]))
    
    def create_level(self, level, input_channels, output_channels, bn):
        return mlp_conv(input_channels, [input_channels, int(input_channels / 2),
                                    int(input_channels / 4), int(input_channels / 8),
                                    output_channels * int(self.tarch[level])], bn)
    
    def forward(self, code : torch.Tensor):
        nlevels = len(self.tarch)
        level0 = self.levels[0](code).reshape(-1, self.nfeat, int(self.tarch[0]))
        outs = [level0, ]
        for i in range(1, nlevels):
            if i == len(self.tarch) - 1:
                nout = 3
            else:
                nout = self.nout
            inp = outs[-1]
            y = torch.cat([inp, code[:, :, None].expand(-1, -1, inp.shape[2])], 1)
            outs.append(self.levels[i](y).reshape(y.shape[0], nout, -1))
            
        reconstruction = outs[-1]
        return reconstruction
    
class Recon(nn.Module):
    def __init__(self, opt):
        super().__init__()
        opt.topnet = get_topnet_params() 
        self.fc_topnet = nn.Linear(opt.feature_dim * 2, opt.topnet.code_nfts)
        self.topnet = TopNet(opt)

    
    def forward(self, kpt_feature, sigmas):
        """ kpt_feature: B feature_dim M
            sigma(uncertainty): B M
            returns   : reconstruction B 3 N
        """

        # turn uncertainty sigmas into confident z
        kpt_feature = kpt_feature.transpose(1,2)
        tau = sigmas.detach().median(dim=1, keepdim=True)[0]  # B x 1
        tau = tau + 1e-8
        z = torch.exp(-sigmas / tau)
        print("tau:", tau[0][:10])     # 打印前 10 个 tau（通常为 B x 1）
        print("z:", z[0][:10]) 
        emb = F.normalize(kpt_feature, dim=-1)
        # import pdb; pdb.set_trace()
        code = torch.cat([torch.max(torch.relu(emb) * z[..., None], 1)[0], torch.max(torch.relu(-emb) * z[..., None], 1)[0]], -1)
        code = self.fc_topnet(code)
        
        # topnet
        rec = self.topnet(code)
        
        return rec


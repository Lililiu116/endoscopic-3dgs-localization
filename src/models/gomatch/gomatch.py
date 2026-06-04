# modified from https://github.com/dvl-tum/gomatch/blob/main/gomatch/models/gomatch.py
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from models.gomatch.gomatch_utils import *
from models.network import GSFeat


class OTMatcherDesc(nn.Module):
    """
    使用外部 desc2d/desc3d；坐标 pts2d/pts3d 仅用于 attention。
    输出格式与 GoMatch OTMatcher 类似：List[Tensor]，每个 sample 一个 (n3d+1, n2d+1) 的 scores。
    """
    def __init__(
        self,
        opt,
        d2: int,          # 输入的 desc2d 维度 D1
        out_dim: int = 128,   # 投影到统一维度 C
        att_layers: Iterable[str] = ("self", "cross", "self"),
    ) -> None:
        super().__init__()
        self.opt = opt
        # Initialize OT
        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.ot = RegularisedOptimalTransport()
        self.attention = SCAtt2D3D(att_layers) if att_layers else None
        
        # 2D desc proj to align feature dim
        self.proj2 = nn.Identity() if d2 == out_dim else nn.Linear(d2, out_dim, bias=False)
        # 3d encoder
        self.kpt3d_enc = GSFeat(opt)
        self.out_dim = out_dim

    def encode_gs(self, gs3d):
        BN, C = gs3d.shape
        N = self.opt.gs_mae.npoints
        assert C == 14
        assert BN % N == 0
        B = BN // N
        gs_data = (gs3d
            .view(B, N, C)          # (B, N, 14)
            .permute(0, 2, 1)       # (B, 14, N)
            .contiguous()
        ) # B 14 N
        E_point = self.kpt3d_enc.forward_feat(gs_data)  # B D2 N
        B, D2, N = E_point.shape
        desc3d = (
            E_point
            .permute(0, 2, 1)   # (B, N, D2)
            .reshape(B * N, D2)  # (BN, D2)
            .contiguous()
        )
       
        return desc3d

    def ot_match(self, desc2d: torch.Tensor, desc3d: torch.Tensor) -> torch.Tensor:
        # Matching distances
        idists = pairwiseL2Dist(desc3d.transpose(-2, -1), desc2d.transpose(-2, -1))

        # Optimal transport
        cost, mu, nu = init_couplings_and_marginals(idists, bin_cost=self.bin_score)
        iscores = self.ot(cost, mu, nu).squeeze(0)
        return iscores

    def forward_sample(
        self,
        desc2d: torch.Tensor,
        desc3d: torch.Tensor,
        pts2d: torch.Tensor,
        pts3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        

        if self.attention is not None:
            desc2d, desc3d = self.attention(desc2d, desc3d, pts2d, pts3d)  # N, C

        # Reshape descriptors
        desc2d = desc2d.T.unsqueeze(0)  # B, C, N
        desc3d = desc3d.T.unsqueeze(0)
        # L2 Normalise
        desc2d = nn.functional.normalize(desc2d, p=2, dim=1)
        desc3d = nn.functional.normalize(desc3d, p=2, dim=1)

        # OT matching scores
        iscores = self.ot_match(desc2d, desc3d)
        return iscores, desc2d, desc3d

    def forward(self, pts2d, idx2d, desc2d, gs3d, idx3d):

        desc2d = self.proj2(desc2d)  # (N2, D1)
        desc3d = self.encode_gs(gs3d)  # BN, D2
        
        pts3d = gs3d[:,:3].contiguous() # BN 3


        # Iterate each sample
        nb = len(torch.unique_consecutive(idx2d))
        
        scores_b = []
        for ib in range(nb):
            mask2d = ib == idx2d
            mask3d = ib == idx3d

            # Descriptor matching
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            idesc2d, idesc3d = desc2d[mask2d], desc3d[mask3d]
            iscores, _, _ = self.forward_sample(idesc2d, idesc3d, ipts2d, ipts3d)
            scores_b.append(iscores)
        return scores_b


class OTMatcherDescCls(nn.Module):
    def __init__(self, opt, d2, out_dim=128, att_layers=("self","cross","self")):
        super().__init__()
        self.opt= opt
        self.raw_matcher = OTMatcherDesc(opt=opt, d2=d2,out_dim=out_dim, att_layers=att_layers)
        self.classifier = MatchCls2D3D(kp_feat_dim=out_dim)

    def classify_sample(self, idesc2d, idesc3d, iscores):
        # idesc2d/idesc3d: (1,C,N) after forward_sample return
        match_mask = torch.tensor(
            mutual_assignment(iscores),
            device=iscores.device,
            dtype=torch.bool
        )
        i3d, i2d = torch.where(match_mask[:-1, :-1])

        if len(i3d) == 1:
            i3d = i3d.expand(2)
            i2d = i2d.expand(2)

        f3d = idesc3d[:, :, i3d]  # (1,C,M)
        f2d = idesc2d[:, :, i2d]

        probs = self.classifier(f2d, f3d)

        match_probs = -1.0 * torch.ones_like(iscores)
        match_probs[i3d, i2d] = probs
        return match_probs

    def forward(self, pts2d, idx2d, desc2d, gs3d, idx3d):
        # 投影在 raw_matcher 内部做
        desc2d = self.raw_matcher.proj2(desc2d)
        desc3d = self.raw_matcher.encode_gs(gs3d)
        pts3d = gs3d[:,:3].contiguous() # BN 3

        nb = len(torch.unique_consecutive(idx2d))
        scores_b, match_probs_b = [], []

        for ib in range(nb):
            mask2d = ib == idx2d
            mask3d = ib == idx3d

            # Predict raw matches
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            idesc2d, idesc3d = desc2d[mask2d], desc3d[mask3d]
            iscores, idesc2d, idesc3d = self.raw_matcher.forward_sample(
                idesc2d, idesc3d, ipts2d, ipts3d
            )
            scores_b.append(iscores)

            # Classify inlier/outlier matches
            ipts2d, ipts3d = pts2d[mask2d], pts3d[mask3d]
            match_probs = self.classify_sample(idesc2d, idesc3d, iscores)
            match_probs_b.append(match_probs)

        return scores_b, match_probs_b

from typing import Iterable, Tuple, Optional, List, Sequence, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path

from models.gomatch.point_resnet import PointResNet
from models.gomatch.batch_ops import batchify_tile_b, flatten_b

Device = Union[torch.device, str, None]
PathT = Union[str, Path]
TensorOrArray = Union[torch.Tensor, np.ndarray]
def square_distance(
    src: torch.Tensor, dst: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalized:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
def get_graph_feature(
    coords: torch.Tensor, feats: torch.Tensor, k: int = 10
) -> torch.Tensor:
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    k = min(k, N - 1)  # There are cases the input data points are fewer than k
    dist = square_distance(coords.transpose(1, 2), coords.transpose(1, 2))
    idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True)[
        1
    ]  # [B, N, K+1], here we ignore the smallest element as it's the query itself
    idx = idx[:, :, 1:]  # [B, N, K]

    idx = idx.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)  # [B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1, 1, 1, k)

    feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)

    return feats_cat
class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, k: int = 10) -> None:
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(
            feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3
def MLP(channels: Sequence[int], do_bn: bool = True) -> nn.Sequential:
    """Multi-layer perceptron"""
    n = len(channels)
    layers: List[nn.Module] = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class PointResNetEncoder(PointResNet):
    def forward(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # type: ignore
        # batchify with tiling
        points_b = batchify_tile_b(points, idx)
        out = super().forward(points_b.transpose(-1, -2)).transpose(-2, -1)
        out = flatten_b(out, idx)
        return out
    
class SCAttention(nn.Module):
    """Predator + SuperGlue Self-Cross Attention Implementation"""

    def __init__(
        self,
        layer_names: Iterable[str],
        num_head: int = 4,
        feature_dim: int = 128,
        k: int = 10,
    ) -> None:
        super().__init__()
        self.names = layer_names
        layers: List[nn.Module] = []
        for atten_type in layer_names:
            if atten_type == "self":
                layers.append(SelfAttention(feature_dim, k))
            else:
                layers.append(AttentionalPropagation(feature_dim, num_head))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        coords0: torch.Tensor,
        coords1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: descs [B, C, N] *coords: [B, D, N]
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                desc0 = desc0 + layer(desc0, desc1)
                desc1 = desc1 + layer(desc1, desc0)
            elif name == "self":
                desc0 = layer(coords0, desc0)
                desc1 = layer(coords1, desc1)
        return desc0, desc1
    
class SCAtt2D3D(nn.Module):
    """Self and cross attention for 2D3D matching."""

    def __init__(self, att_layers: Iterable[str] = ("self", "cross", "self")) -> None:
        super().__init__()
        self.att = SCAttention(att_layers)

    def forward(
        self,
        desc2d: torch.Tensor,
        desc3d: torch.Tensor,
        pts2d: torch.Tensor,
        pts3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input descs: [N, C]   pts: [N, D]
        desc2d = desc2d.unsqueeze(0).permute(0, 2, 1)
        desc3d = desc3d.unsqueeze(0).permute(0, 2, 1)
        pts2d = pts2d.unsqueeze(0).permute(0, 2, 1)
        pts3d = pts3d.unsqueeze(0).permute(0, 2, 1)

        # Perform attention
        desc2d, desc3d = self.att(desc2d, desc3d, coords0=pts2d, coords1=pts3d)
        desc2d = desc2d.permute(0, 2, 1).squeeze()
        desc3d = desc3d.permute(0, 2, 1).squeeze()
        return desc2d, desc3d
    
    
    
    
def init_couplings_and_marginals(
    cost: torch.Tensor, bin_cost: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, m, n = cost.shape
    one = cost.new([1])
    couplings = cost

    # Append dustbins to couplings  (SuperGlue version)
    m_bins = bin_cost.expand(b, m, 1)
    n_bins = bin_cost.expand(b, 1, n)
    last_bin = bin_cost.expand(b, 1, 1)
    couplings = torch.cat(
        [torch.cat([cost, m_bins], -1), torch.cat([n_bins, last_bin], -1)], 1
    )

    # Uniform marginals with dustbin
    mu = torch.cat([one.expand(m), one * n]) / (m + n)
    nu = torch.cat([one.expand(n), one * m]) / (m + n)
    mu = mu.unsqueeze(0)  # 1, m
    nu = nu.unsqueeze(0)  # 1, n
    return couplings, mu, nu



def mutual_assignment(match_scs: TensorOrArray) -> np.ndarray:
    if isinstance(match_scs, torch.Tensor):
        match_scs = match_scs.cpu().data.numpy()

    # match_scs: (n3d+1, n2d+1)
    scores = match_scs[:-1, :-1]
    n3d, n2d = scores.shape

    # Extract matches
    nn12 = scores.argmax(1)
    nn21 = scores.argmax(0)
    m12 = np.dstack([np.arange(n3d), nn12]).squeeze()
    m21 = np.dstack([nn21, np.arange(n2d)]).squeeze()

    # Mutually matched
    match_ids = np.concatenate([m12, m21], axis=0)
    _, ids, counts = np.unique(match_ids, axis=0, return_index=True, return_counts=True)
    match_ids_ = match_ids[ids[counts > 1]]

    # To avoid pose failure cases as much as possible
    if len(match_ids_) <= 4:
        match_ids_ = match_ids[ids]

    # Construct assignment mask for metric calculation
    assign_mask = np.zeros_like(match_scs)
    if len(match_ids_.shape) == 2:  # In cases we have no mutual matches
        ids1, ids2 = match_ids_[:, 0], match_ids_[:, 1]
        assign_mask[ids1, ids2] = 1
    dust_col = 1 - assign_mask.sum(1)
    dust_row = 1 - assign_mask.sum(0)
    assign_mask[:, -1] = dust_col
    assign_mask[-1, :] = dust_row
    assign_mask[-1, -1] = 0
    assign_mask = assign_mask.astype(bool)
    return assign_mask


class RegularisedOptimalTransport(torch.nn.Module):
    def __init__(
        self, max_iters: int = 20, thresh: float = 1e-6, eps: float = 0.1
    ) -> None:
        super().__init__()

        # Initialize params
        self.max_iters = max_iters
        self.thresh = thresh
        self.eps = eps

    def forward(
        self, cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        P = sinkhorn_log(
            cost,
            mu,
            nu,
            self.eps,
            self.max_iters,
            self.thresh,
        )
        return P

def sinkhorn_log(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    eps: float = 0.1,
    max_iters: int = 50,
    thresh: float = 1e-6,
    acc_factor: Optional[float] = None,
) -> torch.Tensor:
    """Sinkhorn algorithm for regularized optimal transport in log space.
    Codes are adapated from https://github.com/gpeyre/SinkhornAutoDiff.

    Args:
        cost: cost matrices, (b, m, n)
        mu, nu: row-wise & column-wise target marginals
        eps: regularization factor; eps -> 0, closer to original ot problem
        thresh: sinkhorn stopping criteria
        max_iters: maximal number of sinkhorn iterations
        accelerate: bool, specify True to accelerate the unbalanced transport
    Return:
        P: optimal transport plan, (b, m, n)
    """

    if acc_factor:
        # To accelerate unbalanced transport
        lam = 0.5 ** 2 / (0.5 ** 2 + eps)
        tau = -acc_factor

    def ave(u: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u_prev

    def M(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

    # Sinkhorn iterations
    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    log_mu, log_nu = torch.log(mu + 1e-8), torch.log(nu + 1e-8)
    for _ in range(max_iters):
        u_prev = u

        # accelerated unbalanced iterations
        if acc_factor:
            u = ave(u, lam * (eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u))
            v = ave(
                v,
                lam
                * (
                    eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1))
                    + v
                ),
            )
        else:
            # Fixed point updates
            u = eps * (log_mu - torch.logsumexp(M(u, v), dim=-1)) + u
            v = eps * (log_nu - torch.logsumexp(M(u, v).transpose(-2, -1), dim=-1)) + v

        # Stopping criteria
        err = (u - u_prev).norm(dim=-1).mean()
        if err < thresh:
            break

    # Transport plan P = diag(a)*K*diag(b)
    P = torch.exp(M(u, v))
    return P


class MatchCls2D3D(nn.Module):
    """Match classification for 2D3D matching."""

    def __init__(
        self, kp_feat_dim: int = 128, feat_dim: int = 128, num_layers: int = 4
    ) -> None:
        super().__init__()
        self.encoder = PointResNet(
            in_channel=kp_feat_dim * 2,
            num_layers=num_layers,
            feat_channel=feat_dim,
            mid_channel=feat_dim,
        )
        self.conv_out = nn.Conv1d(feat_dim, 1, 1)

    def forward(self, f2d: torch.Tensor, f3d: torch.Tensor) -> torch.Tensor:
        # f2d, f3d: (B, C, N)

        # Feature fusion
        mfeat = torch.cat([f2d, f3d], dim=1)  # B=1, C, N

        # Predict probs
        mfeat = self.encoder(mfeat)
        logits = self.conv_out(mfeat).squeeze()  # N,
        probs = torch.sigmoid(logits)
        return probs


def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets

    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res
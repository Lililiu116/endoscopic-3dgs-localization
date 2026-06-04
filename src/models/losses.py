import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== detector loss ============== begin ======
class ChamferLoss_Brute(nn.Module):
    def __init__(self, lambda_color: float = 0.5):
        super(ChamferLoss_Brute, self).__init__()
        self.dimension = 3
        self.lambda_color = lambda_color

    def forward(self, pc_src_input, pc_dst_input, 
                sigma_src=None, sigma_dst=None):
        '''
        :param pc_src_input: Bx3xM Tensor in GPU
        :param pc_dst_input: Bx3xN Tensor in GPU
        :param sigma_src: BxM Tensor in GPU
        :param sigma_dst: BxN Tensor in GPU

        :return:
        '''
        
        B, C, M = pc_src_input.size()[0], pc_src_input.size()[1], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]
        
        pc_src_xyz = pc_src_input[:, :3, :]          # B x 3 x M
        pc_dst_xyz = pc_dst_input[:, :3, :]          # B x 3 x N
        
        
        # -------- 使用 torch.cdist 计算几何距离矩阵 diff：B x M x N --------
        # pc_src_xyz: B x 3 x M → B x M x 3
        src_xyz = pc_src_xyz.transpose(1, 2)  # B x M x 3
        dst_xyz = pc_dst_xyz.transpose(1, 2)  # B x N x 3

                

        diff = torch.cdist(src_xyz.contiguous(), dst_xyz.contiguous(), p=2)  # B x M x N
        
        if sigma_src is not None and sigma_dst is not None:

            # pc_src vs selected pc_dst, M
            src_dst_min_dist, src_dst_I = torch.min(diff, dim=2, keepdim=False)  # BxM, BxM
            selected_sigma_dst = torch.gather(sigma_dst, dim=1, index=src_dst_I)  # BxN -> BxM
            sigma_src_dst = (sigma_src + selected_sigma_dst) / 2
            forward_loss = (torch.log(sigma_src_dst) + src_dst_min_dist / sigma_src_dst).mean()
           
            # pc_dst vs selected pc_src, N
            dst_src_min_dist, dst_src_I = torch.min(diff, dim=1, keepdim=False)  # BxN, BxN
            selected_sigma_src = torch.gather(sigma_src, dim=1, index=dst_src_I)  # BxM -> BxN
            sigma_dst_src = (sigma_dst + selected_sigma_src) / 2  
            backward_loss = (torch.log(sigma_dst_src) + dst_src_min_dist / sigma_dst_src).mean()

            
            # loss that do not involve in optimization
            chamfer_loss = forward_loss + backward_loss
            # chamfer_pure = (src_dst_min_dist.mean() + dst_src_min_dist.mean()).detach()
            # weight_src_dst = (1.0/sigma_src_dst) / torch.mean(1.0/sigma_src_dst)
            # weight_dst_src = (1.0/sigma_dst_src) / torch.mean(1.0/sigma_dst_src)
            # chamfer_weighted = ((weight_src_dst * src_dst_min_dist).mean() +
            #                     (weight_dst_src * dst_src_min_dist).mean()).detach()
                    
            
        else:
            # pc_src vs selected pc_dst, M
            src_dst_min_dist, src_dst_I = torch.min(diff, dim=2, keepdim=False)  # BxM
            forward_loss = src_dst_min_dist.mean() 
            

            # pc_dst vs selected pc_src, N
            dst_src_min_dist, dst_src_I = torch.min(diff, dim=1, keepdim=False)  # BxN
            backward_loss = dst_src_min_dist.mean() 

            chamfer_loss= forward_loss + backward_loss
            # chamfer_pure = forward_loss + backward_loss
            # chamfer_weighted = chamfer_pure
                         

        return chamfer_loss

def zscore(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)


def sinkhorn_region_loss(
    scores,
    tau=0.1,
    epsilon=0.05,
    n_iters=3,
):
    """
    Args:
        scores:  (M, K)  region similarity logits
        tau:     softmax temperature for prediction
        epsilon: sinkhorn temperature
        n_iters: sinkhorn normalization iterations

    Returns:
        loss_region: scalar
    """

    M, K = scores.shape

    # ---------- Sinkhorn-Knopp (balanced targets) ----------
    with torch.no_grad():
        Q = torch.exp(scores / epsilon).t()      # (K, M)
        Q = Q / (Q.sum() + 1e-6)

        r = torch.ones(K, device=scores.device) / K
        c = torch.ones(M, device=scores.device) / M

        for _ in range(n_iters):
            Q = Q * (r / (Q.sum(dim=1) + 1e-6)).unsqueeze(1)
            Q = Q * (c / (Q.sum(dim=0) + 1e-6)).unsqueeze(0)

        Q = (Q / (Q.sum(dim=0, keepdim=True) + 1e-6)).t()  # (M, K)

    # ---------- prediction ----------
    logp = F.log_softmax(scores / tau, dim=-1)   # (M, K)

    # ---------- cross entropy ----------
    loss_region = -(Q * logp).sum(dim=-1).mean()

    return loss_region



class KeypointOnPCLoss(nn.Module):
    def __init__(self):
        super(KeypointOnPCLoss, self).__init__()
        self.single_side_chamfer = SingleSideChamferLoss_Brute()

    def forward(self, keypoint, pc, sn=None):

        loss = self.single_side_chamfer(keypoint, pc)

        return loss


class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self):
        super(SingleSideChamferLoss_Brute, self).__init__()
        self.dimension = 3

    def forward(self, gs_src_input, gs_dst_input):
        '''
        :param gs_src_input: Bx3xM Variable in GPU, keypoints
        :param gs_dst_input: Bx3xN Variable in GPU, original point cloud
        :return:
        '''
        
        B, M = gs_src_input.size()[0], gs_src_input.size()[2]
        gs_dst_input = gs_dst_input[:, :3, :]
        N = gs_dst_input.size()[2]

        gs_src_input_expanded = gs_src_input.unsqueeze(3).expand(B, 3, M, N)
        gs_dst_input_expanded = gs_dst_input.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(gs_src_input_expanded - gs_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM

        return src_dst_min_dist

def ReconLoss(recon, pc):
    """
    input: B 3 M, B 3 N
    """
    pc = pc.transpose(-1,-2)
    recon = recon.transpose(-1,-2)
    dist = torch.cdist(pc, recon)
    loss_recon = torch.mean(torch.min(dist, -1)[0] + torch.min(dist, -2)[0])
    return loss_recon
    
    
class PointOnSurfaceLoss(nn.Module):
    def __init__(self):
        super(PointOnSurfaceLoss, self).__init__()


    def forward(self, keypoint, pc, sn):
        '''

        :param keypoint: Bx3xM
        :param pc: Bx3xN
        :param sn: Bx3xN
        :return:
        '''

        B, M = keypoint.size()[0], keypoint.size()[2]
        N = pc.size()[2]

        keypoint_expanded = keypoint.unsqueeze(3).expand(B, 3, M, N)
        pc_expanded = pc.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(keypoint_expanded - pc_expanded, dim=1, keepdim=False)  # BxMxN

        # keypoint vs selected pc, M
        keypoint_pc_min_dist, keypoint_pc_min_I = torch.min(diff, dim=2, keepdim=False)  # BxM
        pc_selected = torch.gather(pc, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM
        sn_selected = torch.gather(sn, dim=2, index=keypoint_pc_min_I.unsqueeze(1).expand(B, 3, M))  # Bx3xM

        # keypoint on surface loss
        keypoint_minus_pc = keypoint - pc_selected  # Bx3xM
        keypoint_minus_pc_norm = torch.norm(keypoint_minus_pc, dim=1, keepdim=True)  # Bx1xM
        keypoint_minus_pc_normalized = keypoint_minus_pc / (keypoint_minus_pc_norm + 1e-7)  # Bx3xM

        sn_selected = sn_selected.permute(0, 2, 1)  # BxMx3
        keypoint_minus_pc_normalized = keypoint_minus_pc_normalized.permute(0, 2, 1)  # BxMx3

        loss = torch.matmul(sn_selected.unsqueeze(2), keypoint_minus_pc_normalized.unsqueeze(3)) ** 2  # BxMx1x3 * BxMx3x1 -> BxMx1x1 -> 1

        return loss
    
def soft_snap(pcs1_b3n, pcs2_b3m, k=32, alpha=50.0):
    """
    pcs1_b3n: (B,3,N) predicted keypoints (N=64)
    pcs2_b3m: (B,3,M) anchor/surface points (M=16384)
    return:
      pcs1_snap: (B,3,N)
    """
    pcs1 = pcs1_b3n.transpose(1, 2).contiguous()  # (B,N,3)
    pcs2 = pcs2_b3m.transpose(1, 2).contiguous()  # (B,M,3)

    # (B,N,M) squared dist
    d2 = torch.cdist(pcs1, pcs2, p=2) ** 2

    # kNN
    d2_knn, idx = torch.topk(d2, k=k, dim=-1, largest=False)  # (B,N,k)

    # gather neighbors: (B,N,k,3)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    pcs2_expand = pcs2.unsqueeze(1).expand(-1, pcs1.shape[1], -1, -1)  # (B,N,M,3)
    nbr = torch.gather(pcs2_expand, 2, idx_exp)

    # soft weights
    w = torch.softmax(-alpha * d2_knn, dim=-1)  # (B,N,k)

    pcs1_snap = torch.sum(w.unsqueeze(-1) * nbr, dim=2)  # (B,N,3)
    return pcs1_snap.transpose(1, 2).contiguous()        # (B,3,N)

def hard_snap(pcs1_b3n, pcs2_b3m):
    pcs1 = pcs1_b3n.transpose(1, 2).contiguous()  # (B,N,3)
    pcs2 = pcs2_b3m.transpose(1, 2).contiguous()  # (B,M,3)

    d2 = torch.cdist(pcs1, pcs2, p=2) ** 2        # (B,N,M)
    idx = torch.argmin(d2, dim=-1)                # (B,N)

    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)  # (B,N,1,3)
    pcs2_expand = pcs2.unsqueeze(1).expand(-1, pcs1.shape[1], -1, -1)
    snapped = torch.gather(pcs2_expand, 2, idx_exp).squeeze(2)      # (B,N,3)
    return snapped.transpose(1, 2).contiguous()    

import torch
import sys
sys.path.append('/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/ext/PCN-PyTorch')
from extensions.chamfer_distance.chamfer_distance import ChamferDistance


CD = ChamferDistance()


def cd_loss_L1(pcs1, pcs2, lambda_color: float = 0.0):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, 3, N): ours
        pcs2 (torch.tensor): (B, 3, M): gt
    """
    B, C, N = pcs1.shape
    _, _, M = pcs2.shape
    # print(pcs1.shape, pcs2.shape)
    # (B, C, N) -> (B, N, C)
    pcs1 = pcs1.to(dtype=torch.float32)
    pcs2 = pcs2.to(dtype=torch.float32)
    pcs1_t = pcs1.transpose(1, 2).contiguous()   # B x N x C
    pcs2_t = pcs2.transpose(1, 2).contiguous()   # B x M x C

    # --- split xyz / color ---
    xyz1 = pcs1_t[:,:, :3]          # B x N x 3
    xyz2 = pcs2_t[:,:, :3]          # B x M x 3


    

    has_color = (
        lambda_color > 0.0 and 
        pcs1_t.size(-1) >= 6 and 
        pcs2_t.size(-1) >= 6
    )
    if has_color:
        col1 = pcs1_t[..., -3:]          # B x N x 3
        col2 = pcs2_t[..., -3:]          # B x M x 3
    

        

    # --- geometric chamfer using CUDA ---
    # chamfer_3DFunction gives squared distances
    dist1, dist2, idx1, idx2 = CD(xyz1.contiguous(), xyz2.contiguous())
    idx1 = idx1.long()
    idx2 = idx2.long()

    # L1-ish: sqrt of squared distances
    dist1_sqrt = torch.sqrt(dist1 + 1e-8)
    dist2_sqrt = torch.sqrt(dist2 + 1e-8)

    geo_mean = (dist1_sqrt.mean() + dist2_sqrt.mean()) / 2.0

    # ---- fixed tail penalty: top 10% + weight 0.5 ----
    def tail_mean(d):
        k = max(1, int(d.shape[1] * 0.10))  # fixed 10%
        return torch.topk(d, k=k, dim=1, largest=True).values.mean()

    geo_tail = (tail_mean(dist1_sqrt) + tail_mean(dist2_sqrt)) / 2.0
    geo_loss = geo_mean + 0.5 * geo_tail  # fixed weight

    # --- no color: return pure geometry ---
    if not has_color:

        return geo_loss

    # --- color chamfer using NN indices ---

    # src -> dst color
    # idx1: B x N  (index into M)
    idx1_exp = idx1.unsqueeze(-1).expand(-1, -1, 3)    # B x N x 3
    dst_nn_col = torch.gather(col2, 1, idx1_exp)       # B x N x 3
    color_fwd_loss = F.smooth_l1_loss(col1, dst_nn_col, reduction="mean")
    # dst -> src color
    idx2_exp = idx2.unsqueeze(-1).expand(-1, -1, 3)    # B x M x 3
    src_nn_col = torch.gather(col1, 1, idx2_exp)       # B x M x 3
    color_bwd_loss = F.smooth_l1_loss(col2, src_nn_col, reduction="mean")

    color_loss = color_fwd_loss + color_bwd_loss
    
    with torch.no_grad():
        scale = (geo_loss.detach() / (color_loss.detach() + 1e-6)).clamp(0.1, 10.0)


    return geo_loss + lambda_color * color_loss * scale
# ============== detector loss ============== end ======


# ============== matcher loss ============== begin ======
from collections import defaultdict
from typing import Dict, Mapping, Tuple, Union
from datasets.geom_match import project3d_normalized
from evaluations.metrics import io_metric

def compute_loss(
    data: Mapping[str, torch.Tensor],
    preds: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    opt_inliers_only: bool = False,
    cls: bool = False,
    rpthres: float = 1,
) -> Dict[str, torch.Tensor]:
    # preds: ot_scores, *match_probs_b
    bids = torch.unique_consecutive(data["idx2d"])
    i = 0
    losses = defaultdict(list)
    if cls:
        ot_scores_b, match_probs_b = preds
    else:
        ot_scores_b = preds
    for bid in bids:
        mask2d = data["idx2d"] == bid
        mask3d = data["idx3d"] == bid
        n2d = mask2d.sum()
        n3d = mask3d.sum()
        total = (n2d + 1) * (n3d + 1)

        # Load gt matches
        matches_gt = data["matches_bin"][i : i + total].view(n3d + 1, n2d + 1)
        loss = ot_scores_b[bid].new_tensor(0.0)

        # OT log loss
        scores = ot_scores_b[bid]
        if opt_inliers_only:
            matches_gt = matches_gt[:n3d, :n2d]
            scores = scores[:n3d, :n2d]
        loss_log = torch.mean(-scores[matches_gt].log())
        loss += loss_log
        losses["loss_log"].append(loss_log)
        

        # Classification loss
        if cls:
            device = loss.device
            K = data["K"][bid]
            R_gt = data["R"][bid]
            t_gt = data["t"][bid]
            pts2d = data["pts2d"][mask2d].to(device)
            pts3d = data["pts3d"][mask3d].to(device)

            # Compute reprojection err
            i3d, i2d = torch.where(match_probs_b[bid][:-1, :-1] > -1)
            kps3d, kps2d = pts3d[i3d], pts2d[i2d, :2]
            kps2d_proj = project3d_normalized(R_gt, t_gt, kps3d)
            reproj_err = (kps2d - kps2d.new_tensor(kps2d_proj)).norm(dim=1)

            # Balanced BCE loss
            cls_preds = match_probs_b[bid][i3d, i2d]
            pos_mask = (reproj_err < rpthres).float()

            # Measure here instead of inside metric for convience
            cls_metrics = io_metric(cls_preds > 0.5, reproj_err < rpthres)
            losses["cls_recall"].append(cls_metrics["recall"])
            losses["cls_prec"].append(cls_metrics["precision"])

            # Record reproj errors
            losses["matched_reproj_err"].append(reproj_err.mean())
            losses["cls_pos_reproj_err"].append(reproj_err[cls_preds > 0.5].mean())

            # Only when there are positive samples
            if pos_mask.sum() > 0:
                neg_mask = 1 - pos_mask
                pwei = neg_mask.sum() / pos_mask.sum() * pos_mask + neg_mask
                loss_cls = nn.functional.binary_cross_entropy(
                    cls_preds, pos_mask, reduction="none"
                )
                loss_cls = (pwei * loss_cls).mean()
            else:
                loss_cls = cls_preds.new_tensor(0.0)
            loss += loss_cls
            losses["loss_cls"].append(loss_cls)
            if len(i3d) > 0:
                losses["cls_pos_rate"].append(pos_mask.sum() / len(i3d))

        losses["loss"].append(loss)
        i += total
    mean_losses = {k: torch.mean(torch.stack(v)) for k, v in losses.items()}
    return mean_losses


# ============== matcher loss ============== end ======

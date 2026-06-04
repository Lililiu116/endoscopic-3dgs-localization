import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import pytorch3d.utils
import torch
import torch.nn as nn
from .cage2kpt import deform_with_MVC
from .influence_network import Linear, PointNetfeat
import torch.nn.functional as F
from util.vis_tools import plot_keypoint_correspondence



def reindex(src,dst, apply = None,squared = True):
    ...
    """
    Reindex `dst` (and optionally `apply`) to best match `src` under L2 (or L2^2) using Hungarian per batch.

    Args:
        src:   (B, 3, K)  source group centers / keypoints
        dst:   (B, 3, K)  destination group centers / keypoints (to be reindexed)
        apply: (B, C, K)  optional tensor to reindex with the same permutation (e.g., dst_keypoints)
        squared: if True use squared L2 in the cost matrix; else use L2

    Returns:
        dst_reindexed: (B, 3, K) torch.Tensor on the same device as inputs
        perms:         list[np.ndarray] of length B, each shape (K,) (the permutation indices)
        apply_reindexed (optional): (B, C, K) only if `apply` is not None
    """
    assert src.dim() == 3 and dst.dim() == 3, "src and dst must be (B,3,K)"
    assert src.shape[:2] == dst.shape[:2] and src.shape[1] == 3, "src/dst must be (B,3,K)"
    assert src.shape[2] == dst.shape[2], "src and dst must have the same K"

    device = src.device
    B, _, K = src.shape
    
    apply_list = []
    if apply is not None:
            if isinstance(apply, (list, tuple)):
                apply_list = list(apply)
            else:
                apply_list = [apply]
            for t in apply_list:
                assert t.dim() == 3, "`apply` tensors must be (B,C,K)"
                assert t.shape[0] == B and t.shape[2] == K, "`apply` must share batch size B and last dim K with src/dst"
    # Output tensors
    dst_reindexed = torch.empty_like(dst)
    apply_reindexed_list = [torch.empty_like(t) for t in apply_list]
    perms: list[np.ndarray] = []

    # Move small copies to CPU/NumPy for Hungarian
    src_cpu = src.detach().cpu().numpy()   # (B,3,K)
    dst_cpu = dst.detach().cpu().numpy()   # (B,3,K)

    for b in range(B):
        # (K,3)
        src_b = src_cpu[b].T
        dst_b = dst_cpu[b].T

        # Cost matrix C[i,j] = ||src_i - dst_j|| or ||.||^2
        diff = src_b[:, None, :] - dst_b[None, :, :]  # (K,K,3)
        if squared:
            C = np.sum(diff ** 2, axis=-1)
        else:
            C = np.linalg.norm(diff, axis=-1)

        row_ind, col_ind = linear_sum_assignment(C)
        perm = col_ind  # reorder dst indices to match src
        perms.append(perm)

        # Apply permutation
        dst_reindexed[b] = torch.from_numpy(dst_b[perm].T).to(device)
        p = torch.as_tensor(perm, device=device, dtype=torch.long)

        for i, t in enumerate(apply_list):
            apply_reindexed_list[i][b] = t[b][:, p]
            
    # Return apply_reindexed in same structure as input
    if apply is None:
        return dst_reindexed, perms
    elif isinstance(apply, (list, tuple)):
        return dst_reindexed, perms, apply_reindexed_list
    else:
        return dst_reindexed, perms, apply_reindexed_list[0]


def apply_transformation(points, R, s, t):
    """
    Apply SIM(3) transform to a batch of point sets.

    Args:
        points: (B, 3, N)    # e.g., cage or keypoints
        R:      (B, 3, 3)    # rotation
        s:      scalar or (B,) or (B,1,1)  # scale
        t:      (B, 3, 1)    # translation

    Returns:
        (B, 3, N) transformed points
    """
    # Ensure s is a tensor of shape (B,1,1) on the right device/dtype
    if not torch.is_tensor(s):
        s = torch.tensor(s, dtype=points.dtype, device=points.device)
    if s.dim() == 0:
        s = s.expand(points.shape[0])
    if s.dim() == 1:
        s = s.view(-1, 1, 1)  # (B,1,1)

    # Make sure R,t are on same device/dtype
    R = R.to(points.device, points.dtype)
    t = t.to(points.device, points.dtype)

    out = torch.matmul(R, points)          # (B,3,3) x (B,3,N) -> (B,3,N)
    out = out * s                          # broadcast (B,1,1)
    out = out + t                          # broadcast (B,3,1)
    return out

class CageSkinning(nn.Module):
    def __init__(self, opt):
        """
        From https://github.com/tomasjakab/keypoint_deformer
        """
        super(CageSkinning, self).__init__()
        self.opt = opt
        self.dim = 3 # for default, means 3D
        
        self.vertices, self.faces = self.create_cage() # (1,3,V), (1,F,3)
        self.init_influence(self.dim)
        
    # ---------- cage & utils ----------    
    @torch.no_grad()
    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.opt.ico_sphere_div, device=self.opt.device)
        V = mesh.verts_padded()
        F = mesh.faces_padded()
        V = V / (V.norm(dim=-1, keepdim=True).max() + 1e-8)
        V = V.transpose(1,2)
        return V, F
    
    @torch.no_grad()
    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    

    @torch.no_grad()
    def src_cage(self, src_gs, margin=1.15):
        """
        根据 src 点云放置紧贴的 cage
        src_gs: (B,3,N)
        return: src_cage (B,3,V), center (B,3,1), r_src (B,1,1)

        """
        B = src_gs.shape[0]
        center = src_gs.mean(dim=2, keepdim=True)  # (B,3,1)
        rel = src_gs - center                      # (B,3,N)
        r_src = torch.sqrt((rel**2).sum(dim=1, keepdim=True)).max(dim=2, keepdim=True)[0]  # (B,1,1)
        radius = margin * r_src

        cage0 = self.vertices.expand(B, -1, -1)   # (B,3,V) canonical
        src_cage = cage0 * radius + center  # (B,3,V)
        
        return src_cage

    
    
    # ---------- influence ------------
    def init_influence(self, dim):
        # influence predictor
        influence_size = self.opt.n_keypoints * self.vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=dim, num_points=self.opt.gs_mae.npoints, bottleneck_size=influence_size),
            Linear(influence_size, influence_size, activation="lrelu", normalization=None))
        dencoder_influence = nn.Sequential(
                Linear(influence_size, influence_size, activation="lrelu", normalization=None),
                Linear(influence_size, influence_size, activation=None, normalization=None))
        self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)
    
    
    
    def forward(self, src_gs, src_keypoints, dst_keypoints, src_sigmas, dst_sigmas, src_group_center_transformed, dst_group_center):
        """
        src_gs (B,14,N)
        src_keypoints (B,3,M)
        dst_keypoints (B,3,M) 
    
        src_group_center + dst_group_center -> reindexed dst
        
        """
        
        B, _, _ = src_gs.shape
        src_xyz = src_gs[:,:3,:]


        # reindex to align src and dst     
  
        _, _, (dst_reindex_keypoints, dst_reindex_sigmas )= reindex(src_group_center_transformed, dst_group_center, apply=[dst_keypoints,dst_sigmas.unsqueeze(1)])
 
        # 1) initialize influence
        influence_logits = self.influence_predictor(src_xyz)
        influence_logits = influence_logits.view(B, self.opt.n_keypoints, self.vertices.shape[2])
        
        # 2) cage close to src_gs
        src_cage = self.src_cage(src_xyz, margin= 1.3) # (B,3,V)
        src_cage = self.optimize_cage(src_cage, src_xyz)


        # 3) src_kpt influence on cage
        distance = torch.sum((src_keypoints[..., None] - src_cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.opt.n_influence_ratio)
        n_influence = max(1, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        
 
        
        # 4) Masked Softmax：权重归一化（不动点云坐标） ------------------
        
        neg_inf = torch.finfo(influence_logits.dtype).min
        masked_logits = torch.where(keep, influence_logits, torch.full_like(influence_logits, neg_inf))
        
        # add log σ from USIP
        eps = 1e-6
        # 确保 σ>0（如果上游还没做，可以在这里兜底）
        src_sigmas = torch.clamp(src_sigmas, min=eps)
        dst_reindex_sigmas = dst_reindex_sigmas.squeeze(1)
        dst_reindex_sigmas = torch.clamp(dst_reindex_sigmas, min=eps)
        sigma_pair = 0.5 * (src_sigmas + dst_reindex_sigmas)     # (B,M)
        sigma_pair = torch.clamp(sigma_pair, min=eps)

        log_prior = -torch.log(sigma_pair + eps)[..., None]         # (B, M, 1)
        masked_logits = masked_logits + log_prior        
        # weights: (B, M, V)，沿 M 维 softmax，使每个 V 列的权重和为 1
        weights = F.softmax(masked_logits, dim=1)
        assert torch.isfinite(weights).all()


        # 兜底：极端情况下某些列全被 mask（数值上等价 -inf），softmax 会 NaN/全 0。
        # 我们给这种列回退到“最近点 one-hot”： self.faces.e
        with torch.no_grad():
            bad_col = ~torch.isfinite(weights).any(dim=1, keepdim=True)  # (B,1,V)
            if bad_col.any():
                nn_idx = distance.argmin(dim=1, keepdim=True)            # (B,1,V)
                one_hot = torch.zeros_like(weights).scatter_(1, nn_idx, 1.0)
                weights = torch.where(bad_col, one_hot, weights)

        
        # 5) apply Umeyama for global transformation
        R, s, t = predict_Rst(src_keypoints, dst_reindex_keypoints, with_scale=False)


        src_transformed_kpt = apply_transformation(src_keypoints, R, s, t)
        src_transformed_cage = apply_transformation(src_cage, R, s, t)
        
        # 5) Keypoints offset to Cage offset
        keypoints_offset = dst_reindex_keypoints - src_transformed_kpt
        # keypoints_offset = dst_reindex_keypoints - src_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * weights[:, None, :, :], dim=2)  # (B,3,V)
        cage_offset = getattr(self.opt, 'cage_offset_scale', 1) * cage_offset
        # apply (R,s,t) on cage: from src_cage to dst_cage
        
  
        new_cage = src_transformed_cage + cage_offset
        # new_cage = src_cage + cage_offset
        new_cage = new_cage.transpose(1, 2) # B V 3

        deformed_shapes, weights, _ = deform_with_MVC(
            src_cage.transpose(1, 2), new_cage, self.faces.expand(B, -1, -1), src_xyz.transpose(1, 2), verbose=True)

        

        return deformed_shapes.transpose(1, 2), new_cage.transpose(1, 2), self.faces, src_cage
    



def umeyama(src_kpt, dst_kpt, with_scale=True, eps=1e-8):
    # Expect (B,3,M)
    assert src_kpt.dim() == 3 and dst_kpt.dim() == 3
    assert src_kpt.shape == dst_kpt.shape
    B, C, M = src_kpt.shape
    assert C == 3

    # Centroids (B,3,1)
    mu_x = src_kpt.mean(dim=2, keepdim=True)
    mu_y = dst_kpt.mean(dim=2, keepdim=True)

    # Centralize (B,3,M)
    Xc = src_kpt - mu_x
    Yc = dst_kpt - mu_y

    # Covariance (B,3,3)
    Sigma = (Yc @ Xc.transpose(1, 2)) / float(M)

    # SVD
    U, S, Vh = torch.linalg.svd(Sigma)  # S: (B,3)

    # Fix reflection
    det = torch.det(U @ Vh)             # (B,)
    sign = torch.ones_like(det)
    sign[det < 0] = -1.0
    Sfix = torch.eye(3, device=src_kpt.device, dtype=src_kpt.dtype).unsqueeze(0).repeat(B,1,1)
    Sfix[:, -1, -1] = sign

    # Rotation (B,3,3)
    R = U @ Sfix @ Vh

    # Scale (B,1,1)
    if with_scale:
        var_x = (Xc.pow(2).sum(dim=(1,2)) / float(M)).clamp_min(eps)  # (B,)
        num = S[:, 0] + S[:, 1] + sign * S[:, 2]                      # (B,)
        s = (num / var_x).reshape(B, 1, 1)                            # (B,1,1)
    else:
        s = torch.ones((B,1,1), device=src_kpt.device, dtype=src_kpt.dtype)

    # Translation (B,3,1)
    t = mu_y - s * (R @ mu_x)
    


    return R, s, t




def predict_Rst(src_kpt, dst_kpt, with_scale=True):
    """
    Predict transformation from src_kpt to dst_kpt
    src_kpt: (B, 3, M)
    dst_kpt: (B, 3, M)

    """

    R, s, t = umeyama(src_kpt, dst_kpt, with_scale=with_scale)
    return R, s, t
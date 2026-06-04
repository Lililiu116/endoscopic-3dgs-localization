import torch
import torch.nn as nn
import torch.nn.functional as F

class KCN(nn.Module):
    """
    输入:
      kpt_xyz:  (B, 3, K)
      kpt_feat: (B, C, K)

    输出:
      coarse_xyz: (B, 3, M)  # 你也可以改成返回 coarse 骨架点(如 K*M 或 K*M*S)
      fine_xyzrgb:(B, 6, N)  # xyz+rgb
    """

    def __init__(self, feat_dim=256, grid_size=4, M=3, seed_range=0.05, fine_ratio=0.25):
        super().__init__()
        assert M == 3, "你说 M=3，这里先固定实现为 3（想改也很容易）"
        self.M = M
        self.grid_size = grid_size
        self.S = grid_size ** 2
        self.seed_range = seed_range
        self.fine_ratio = fine_ratio

        # 每个 kpt 输出 3 个方向向量 -> (B,K,3,3)
        self.dir_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3 * self.M),
        )

        # 每个 kpt、每个分支输出尺度 (sx, sy) -> (B,K,3,2)
        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 2 * self.M),
        )

        # 每个 kpt、每个分支输出沿方向的长度系数（可选，但很好用）-> (B,K,3,1)
        self.len_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1 * self.M),
        )

        # fine：在 (t1, t2, d) 局部基底里输出小偏移（xyz）
        # 输入 = kpt_feat + seed2d + branch_id_onehot(3)
        self.fine_xyz_head = nn.Sequential(
            nn.Linear(feat_dim + 2 + self.M, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3),
        )

        # color：同样基于 coarse frame（避免颜色完全绕过几何）
        self.color_head = nn.Sequential(
            nn.Linear(feat_dim + 2 + self.M, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3),
        )

        # 2D folding seed
        a = torch.linspace(-seed_range, seed_range, steps=grid_size).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-seed_range, seed_range, steps=grid_size).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        seed = torch.cat([a, b], dim=0).view(1, 2, self.S)  # (1,2,S)
        self.register_buffer("folding_seed", seed)
        # 1D seed for growth along direction d (1, 1, S)
        s1d = torch.linspace(0.0, 1.0, steps=self.S, dtype=torch.float).view(1, 1, self.S)
        self.register_buffer("seed_1d", s1d)


        # 分支 one-hot（常量，干净）
        eye = torch.eye(self.M).view(1, 1, self.M, self.M)  # (1,1,M,M)
        self.register_buffer("branch_onehot", eye)

    def _make_frame_from_dir(self, d, device):
        """
        d: (B, K, M, 3)  -> returns t1,t2: same shape
        这里就是你之前不理解那段：用 ref 与 d 做叉乘构造正交基
        """
        B, K, M, _ = d.shape
        ref = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3).expand(B, K, M, 3)
        dot = (d * ref).sum(-1).abs()  # (B,K,M)
        near = (dot > 0.95).unsqueeze(-1)  # (B,K,M,1)
        ref2 = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 1, 3).expand(B, K, M, 3)
        ref = torch.where(near, ref2, ref)

        t1 = torch.cross(ref, d, dim=-1)
        t1 = F.normalize(t1, dim=-1)
        t2 = torch.cross(d, t1, dim=-1)
        return t1, t2

    def forward(self, kpt_xyz, kpt_feat):

        kpt_xyz = kpt_xyz.transpose(1, 2).contiguous()   # -> (B,K,3)
        kpt_feat = kpt_feat.transpose(1, 2).contiguous() # -> (B,K,C)

        B, K, _ = kpt_xyz.shape
        C = kpt_feat.shape[-1]
        device = kpt_xyz.device


        # -------- coarse: dirs + scales (+ length) --------
        d = self.dir_head(kpt_feat).view(B, K, self.M, 3)
        d = F.normalize(d, dim=-1)  # (B,K,3,3)

        scale = self.scale_head(kpt_feat).view(B, K, self.M, 2)
        scale = F.softplus(scale) + 1e-6  # sx,sy positive

        # 沿分支方向的长度（允许“走远”，但还是结构化的标量）
        # 用 softplus 保证为正
        length = self.len_head(kpt_feat).view(B, K, self.M, 1)
        length = F.softplus(length) + 1e-6  # (B,K,3,1)

        t1, t2 = self._make_frame_from_dir(d, device=device)  # (B,K,3,3)

        # -------- expand seeds to (B,K,M,S,2) --------
        seed = self.folding_seed.expand(B, -1, -1).permute(0, 2, 1).contiguous()  # (B,S,2)
        seed = seed.view(B, 1, 1, self.S, 2).expand(B, K, self.M, self.S, 2)      # (B,K,M,S,2)
        ux = seed[..., 0:1]  # (B,K,M,S,1)
        uy = seed[..., 1:2]

        # 分支 onehot -> (B,K,M,S,M)
        onehot = self.branch_onehot.expand(B, K, -1, -1).unsqueeze(3).expand(B, K, self.M, self.S, self.M)

        # kpt_feat -> (B,K,M,S,C)
        feat = kpt_feat.view(B, K, 1, 1, C).expand(B, K, self.M, self.S, C)

        # -------- coarse skeleton points (B,K,M,S,3) --------
        base = kpt_xyz.view(B, K, 1, 1, 3).expand(B, K, self.M, self.S, 3)

        sx = scale[..., 0:1].unsqueeze(3)  # (B,K,M,1,1)
        sy = scale[..., 1:2].unsqueeze(3)
        L  = length.unsqueeze(3)           # (B,K,M,1,1)

        s1d = self.seed_1d.view(1, 1, 1, self.S,1).expand(B, K, self.M, self.S, 1)

        coarse_pts = base \
            + (ux * sx).expand(-1, -1, -1, -1, 3) * t1.unsqueeze(3) \
            + (uy * sy).expand(-1, -1, -1, -1, 3) * t2.unsqueeze(3) \
            + (s1d * L).expand(-1, -1, -1, -1, 3)  * d.unsqueeze(3)

        # -------- fine xyz (local) --------
        fine_in = torch.cat([feat, seed, onehot], dim=-1)  # (B,K,M,S,C+2+M)

        local_off = self.fine_xyz_head(fine_in)  # (B,K,M,S,3)
        local_off = torch.tanh(local_off)

        # 限制 fine：与 coarse 尺度相关（防止作弊大跳）
        # 用 min(sx,sy) 做局部半径基准
        r0 = self.fine_ratio * torch.min(sx, sy)  # (B,K,M,1,1)
        local_off = local_off * r0                # (B,K,M,S,3)

        dx = local_off[..., 0:1]
        dy = local_off[..., 1:2]
        dz = local_off[..., 2:3]

        fine_off_world = dx.expand(-1, -1, -1, -1, 3) * t1.unsqueeze(3) \
                       + dy.expand(-1, -1, -1, -1, 3) * t2.unsqueeze(3) \
                       + dz.expand(-1, -1, -1, -1, 3) * d.unsqueeze(3)

        fine_xyz = coarse_pts + fine_off_world  # (B,K,M,S,3)

        # -------- color (rgb) --------
        # color is [-1,1]
        rgb = torch.tanh(self.color_head(fine_in))  # (B,K,M,S,3)

        # coarse: just output keypoints as (B,3,K)
        Nc = K * self.M * self.S
        coarse = coarse_pts.reshape(B, Nc, 3).transpose(1, 2).contiguous()  # (B,3,Nc)


        # fine xyzrgb full: (B,6,N_full)
        N_full = K * self.M * self.S

        fine = torch.cat([fine_xyz, rgb], dim=-1)                       # (B,K,M,S,6)
        fine = fine.reshape(B, N_full, 6).transpose(1, 2).contiguous()  # (B,6,N_full)

        return coarse, fine

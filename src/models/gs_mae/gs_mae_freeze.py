
# import sys
# sys.path.append("/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/ext/ShapeSplat-Gaussian_MAE")
# from segmentation_gs.models.pointnet2_utils import PointNetFeaturePropagation
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from knn_cuda import KNN
import math





# Liwen modify
class GS_MAE_Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.attribute = opt.gs_mae.attribute               # ['xyz','opacity','scale','rotation','sh']
        self.trans_dim = opt.gs_mae.encoder_dims            # 384
        self.depth = opt.gs_mae.depth                       # 12
        self.drop_path_rate = opt.gs_mae.drop_path_rate     # 0.1
        self.num_heads=opt.gs_mae.num_heads                 # 6
        self.npoints = opt.gs_mae.npoints                   # 1024



        
        # 1. 局部编码模块
        self.encoder = Encoder(encoder_channel=self.trans_dim, attribute=self.attribute, npoints=self.npoints)

        # 2. 位置编码
        self.pos_feature_dim = []
        self.pos_feature_dim.extend([0,1,2])

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        
        # 3. Transformer encoder
        self.attn_geo = None
        self.attn_thresh = None #0.02 set the attention threshold to be 0.02 based on the analysis in exploration 9_20251024
        self.attn_topk = None
        # print the not None options
        if self.attn_geo is not None:
            print(f"Using geometric attention mask with k={self.attn_geo}")
        if self.attn_topk is not None:
            print(f"Using top-k attention with k={self.attn_topk}")
        if self.attn_thresh is not None:
            print(f"Using attention threshold: {self.attn_thresh}")
    
        
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=self.drop_path_rate ,
            num_heads=self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        
    def load_encoder_from_ckpt(self, ckpt_path):
        
        assert ckpt_path, "[Encoder] ckpt_path is empty."
        ckpt = torch.load(ckpt_path, map_location="cpu")
        base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }


        model_dict = self.state_dict()
        filtered_ckpt = {k: v for k, v in base_ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        for k, v in base_ckpt.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    print(f"Shape mismatch for {k}: ckpt {v.shape} vs model {model_dict[k].shape}")


        # 加载匹配的权重
        self.load_state_dict(filtered_ckpt, strict=False)
        print(f"[Encoder] Loaded {len(filtered_ckpt)} matching parameters from {ckpt_path}")


      
    def forward(self, neighborhood, center_xyz, epoch=None):
        '''
        :param:
            neighborhood: B G M 14
            center xyz:  B G 3
        :return: point embed from frozen gs-mae B N C(1024)
        '''

        
        group_input_tokens = self.encoder(neighborhood)  # group feature: B G trans_dim 
        pos = self.pos_embed(center_xyz) # B x G x trans_dim
        
        if self.attn_geo is not None:
            attn_geo = build_attn_mask(center_xyz, k_mask = int(self.attn_geo)) # B G G
        else:
            attn_geo = None
            

        # === 2. Transformer编码 ===
        feature_list = self.blocks(group_input_tokens, pos, attn_geo =attn_geo, attn_topk=self.attn_topk, attn_thresh=self.attn_thresh)  # List of B x G x trans_dim

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() # B trans_dim G
                        for x in feature_list]

        # feature
        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # B x C(1152) x G 


        return x




class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel, attribute=['xyz'], npoints=1024):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.attribute = attribute
        input_dim = 0
        if 'xyz' in attribute:
            input_dim += 3
        if 'opacity' in attribute:
            input_dim += 1
        if 'sh' in attribute:
            input_dim += 3
        if 'scale' in attribute:
            input_dim += 3
        if 'rotation' in attribute:
            input_dim += 4
        if npoints == 1024:
            self.first_conv = nn.Sequential(
                nn.Conv1d(input_dim, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(256, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.encoder_channel, 1)
            )
        elif npoints == 4096:
            self.first_conv = nn.Sequential(
                nn.Conv1d(input_dim, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, self.encoder_channel, 1)
            )            

    def forward(self, point_groups):
        '''
            point_groups : B G N 3, gs B G N K
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        # choose the attribute we want
        # print("point_groups", point_groups.shape)
        attribute_index = [0, 1, 2]
        if 'opacity' in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if 'scale' in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if 'rotation' in self.attribute:
            rotation_index = [7,8,9,10]
            attribute_index.extend(rotation_index)
        if 'sh' in self.attribute:
            sh_index = [11,12,13]
            attribute_index.extend(sh_index)
        
        # choose the attribute we want
        # print("org point_groups", point_groups.shape)
        point_groups = point_groups[..., attribute_index]

        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        # print("point_groups", point_groups.shape)
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        # print("feature",  feature.shape)  ([8192, 256, 32])
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print("feature_global",  feature_global.shape) ([8192, 256, 1])
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_geo=None, attn_topk=None, attn_thresh=None):
        """
        x: B, N, C
        attn_geo: (optional) bool mask, shape [B, N, N] or [1, N, N]
                        True for allowed positions, False for masked ones
        topk: (optional) int, keep only top-k attention per query token
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        
        # -------- Option A: geometric neighbor --------
        if attn_geo is not None:
            if attn_geo.dim() == 3:
                attn_geo = attn_geo[:, None, :, :]   # [B,1,N,N]
            attn = attn.masked_fill(~attn_geo, float('-inf'))
        
        # -------- Option B: attention top-k --------
        if attn_topk is not None and attn_topk < attn.size(-1):
            vals, idx = attn.topk(attn_topk, dim=-1)
            sparse = torch.full_like(attn, float('-inf'))
            sparse.scatter_(-1, idx, vals)
            attn = sparse
        
        attn = attn.softmax(dim=-1) # [B, num_heads, N, N]

        # -------- Option C: attention threshold --------
        if attn_thresh is not None:
            # apply threshold
            attn = attn.masked_fill(attn < attn_thresh, 0.0)
            denom = attn.sum(dim=-1, keepdim=True)
            attn = attn / (denom + 1e-12)
        
        attn = self.attn_drop(attn)
        
        
        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, attn_geo=None, attn_topk=None, attn_thesh=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_geo=attn_geo, attn_thresh=attn_thesh, attn_topk=attn_topk))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(
                    drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos,attn_geo=None, attn_topk=None, attn_thresh=None):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos, attn_geo=attn_geo, attn_topk=attn_topk, attn_thesh=attn_thresh)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list




@torch.no_grad()
def build_attn_mask(center_xyz, k_mask: int, include_self: bool = True):
    """
    center_xyz: [B, G, 3]  每个 group 的几何中心
    k_mask:     每个 query 允许关注的“几何最近”邻居数（与注意力 topk 无关）
    include_self: 是否强制把自己包含在可见集合中
    return: attn_mask [B, G, G]，bool，True=允许注意
    """
    B, G, _ = center_xyz.shape

    # pairwise 距离 [B,G,G]
    # torch>=1.1 有 cdist
    dist = torch.cdist(center_xyz, center_xyz, p=2)  # Euclidean

    # 为了从“最近”取 topk，我们把距离取负号当做“分数”
    scores = -dist  # 距离越小 -> 分数越大

    # 如需保证“自己”一定可见，可先把对角线干预（或后面 OR 回来）
    if include_self:
        # 防止“自己”被 topk 选走后再覆盖出错，先把对角线设成最小值，这样我们最后手动 OR 回来
        eye = torch.eye(G, device=center_xyz.device)[None]  # [1,G,G]
        scores = scores.masked_fill(eye.bool(), float('-inf'))

    keep_k = min(k_mask, G)  # 防越界
    # 每个 query 取 top-k 最近邻（不含自己；自己我们稍后单独加）
    topk_vals, topk_idx = scores.topk(k=keep_k, dim=-1)  # [B,G,k]

    # 构造布尔掩码
    attn_mask = torch.zeros(B, G, G, dtype=torch.bool, device=center_xyz.device)
    attn_mask.scatter_(-1, topk_idx, True)  # 把 topk 的列置 True

    if include_self:
        eye = torch.eye(G, dtype=torch.bool, device=center_xyz.device)[None]
        attn_mask = attn_mask | eye

    return attn_mask

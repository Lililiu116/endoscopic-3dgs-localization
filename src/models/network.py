import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from models.gs_mae.gs_mae_freeze import GS_MAE_Encoder
from models.gs_mae.pointnet2_utils import PointNetFeaturePropagation
from config.path import external_ckpt_path

class GSFeat(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.npoints = opt.gs_mae.npoints                   # 1024
        self.num_group = opt.gs_mae.num_group               # 64
        self.group_size = opt.gs_mae.group_size             # 32
        self.group_attribute = opt.gs_mae.group_attribute   # ['xyz']
        self.trans_dim = opt.gs_mae.encoder_dims            # 384
        self.seed = opt.seed
        self.n_keypoints= opt.n_keypoints
        self.feature_dim = opt.feature_dim 
        self.proj_dim = 128
        self.region_tau = 0.1
        
        # --- Ablation parameter ---
        mode = opt.encoder_mode
        self.prop_mode = opt.prop_mode
        self.fusion_mode = opt.fusion_mode
        self.local_size = opt.local_size
        
        # 1. grouping
        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            seed=self.seed,
            attribute=self.group_attribute,
        )
        # 2. gs mae encoder
        self.gs_encoder = GS_MAE_Encoder(opt)
        # load pretrained
        ckpt =  str(external_ckpt_path())
        print('[Encoder]: loading pretrained weight from:', ckpt)
        self.gs_encoder.load_encoder_from_ckpt(ckpt)
        print('pretrained weight loaded finish!') 
        
        print('---Ablation---')
        if  mode == 'freeze':
            for p in self.gs_encoder.parameters():p.requires_grad = False
            print('[A1] pretrained module freezed!')        
        elif mode == 'finetune_all':
            for p in self.gs_encoder.parameters(): p.requires_grad = True
            print('[A2] pretrained module unfreeze! Finetune all')       
        elif mode == "finetune_last":
            for p in self.gs_encoder.parameters(): p.requires_grad = False
            # unfreeze last block(s) – adjust names to your encoder
            k = getattr(opt, "finetune_last_k", 1)  # 默认解冻最后 1 个 block
            for blk in self.gs_encoder.blocks.blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.gs_encoder.norm.parameters(): p.requires_grad = True   
            print('[A3] pretrained module unfreeze the last two layers! Finetune last')     
        
        if self.local_size == 8:    
            print(f'[D] using local size = {self.local_size}')
        elif self.local_size == 4: 
            print(f'[D1] using local size = {self.local_size}')
        elif self.local_size == 12: 
            print(f'[D2] using local size = {self.local_size}')
            
                                    
        if self.fusion_mode == 'local_only':
            print('[C1] using E_local only')
        elif self.fusion_mode == 'sem_only':
            print('[C2] using E_sem only')
        else:
            print('[C] using E_sem and E_local')
    
        # 3. propagate feature
        input_channels = self.trans_dim * 3

        self.propagation_all = PointNetFeaturePropagation(in_channel=input_channels+3,
                                                        mlp=[self.trans_dim * 4, 1024])
        # # map group-token (1152) -> point_embed (1024) for nearest-group baseline
        # self.group2point = nn.Sequential(
        #     nn.Conv1d(input_channels, 1024, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(1024, 1024, 1),
        # )

        # 4. score for each points
        self.score= FeatureScore(embed_dim=1024, out_dim=self.feature_dim, local_size = self.local_size, fusion_mode =self.fusion_mode)
        
        # self.kpt_head =KeypointHead( n_kpt=self.n_keypoints)


    def forward_feat(self, gs_data,  epoch=None):
        '''
        :param gs_data: B x C x N
        :return: center of cluster, keypoints, sigmas, descriptors
        '''
        B, C, N = gs_data.shape
        gs_data = gs_data.transpose(-1, -2).contiguous()  
        gs_xyz = gs_data[..., :3].contiguous()# B N 3
        
        # 1. grouping
        neighborhood, center, group_idx = self.group_divider(gs_data)
        center_xyz = center[..., :3].contiguous()
        
        # 2. frozen gs-encoder
        x = self.gs_encoder( neighborhood, center_xyz)  # B x C(1152) x G 
        
        # 3. propagate to all points --> semantic field for all points
        if self.prop_mode == "fp":
            f_level_0 = self.propagation_all(
                gs_xyz.transpose(-1, -2), center_xyz.transpose(-1, -2),gs_xyz.transpose(-1, -2), x) # B x C(1024) x N
            
            point_embed = f_level_0.transpose(1, 2).contiguous()  # (B,N,1024)
        # elif self.prop_mode == "nearest_group":
        #     # distances: [B,N,G]
        #     dist2 = torch.cdist(gs_xyz.float(), center_xyz.float(), p=2)  # might be heavy but clean
        #     gid = dist2.argmin(dim=-1)                    # [B,N]

        #     # gather group token for each point
        #     # x_points: [B,1152,N]
        #     Cg = x.shape[1] 
        #     x_perm = x.permute(0,2,1).contiguous()        # [B,G,1152]
        #     x_points = x_perm.gather(1, gid.unsqueeze(-1).expand(-1,-1,Cg))  # [B,N,1152]
        #     x_points = x_points.permute(0,2,1).contiguous()                    # [B,1152,N]

            # # small learnable mapping to 1024
            # point_embed = self.group2point(x_points).transpose(1,2).contiguous()  # [B,N,1024]            

        
        # 4. local saliency score based on geometry, color, and little of semantic
        E_point = self.score(gs_data, point_embed)
        


        return E_point
                

    
def fps_gs(data, number, attribute=['xyz'], return_idx = False, random_start = False, random_seed = None):
    '''
        data B N K
        number int
    '''
    fps_index = []
    if 'xyz' in attribute:
        fps_index.extend([0,1,2])
    if 'opacity' in attribute:
        fps_index.extend([3])
    if 'scale' in attribute:
        fps_index.extend([4,5,6])
    if 'rotation' in attribute:
        fps_index.extend([7,8,9,10])
    if 'sh' in attribute:
        fps_index.extend([11,12,13])

    data_fps = data.clone()[...,fps_index].contiguous()
    B, N, F = data_fps.shape
    device = data.device
    # print("data_fps", data_fps.shape)
    if random_start:
        # 1) random number between 0 to N, each Batch, one number
        fps_gen = torch.Generator(device=device)
        fps_gen.manual_seed(int(random_seed) if random_seed is not None else 0)

        start_idx = torch.randint(0, N, (B,), device=device, dtype=torch.long, generator=fps_gen)  # B


        # 2) 构造排列 perm，并把每行的 start_idx 与位置 0 交换（向量化，无 for）
        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone() # B x N (original order)
        rows = torch.arange(B, device=device)          # B

        # 交换第0列和随机起点列（仅交换索引，不动属性）
        tmp = perm[rows, 0].clone()
        perm[rows, 0] = perm[rows, start_idx]
        perm[rows, start_idx] = tmp


        # 3) 在仅交换首位后的序列上做 FPS（等价于随机起点）
        data_fps_shuf = torch.gather(data_fps, 1, perm.unsqueeze(-1).expand(-1, -1, F)).contiguous()
        idx_shuf = pointnet2_utils.furthest_point_sample(data_fps_shuf, number)   # B x M, int32

        # 4) 映射回原始索引（批量索引）
        bidx = torch.arange(B, device=device).unsqueeze(1)                         # B x 1
        fps_idx = perm[bidx, idx_shuf.long()].to(torch.int32).contiguous()  
    else:
        fps_idx = pointnet2_utils.furthest_point_sample(data_fps, number)
        
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2)
        
    if return_idx:
        return fps_data, fps_idx
    # print("fps_idx", fps_idx.shape, data.transpose(1, 2).contiguous().shape)


    return fps_data

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size,seed, attribute=['xyz']):
        # attribute use to group
        # xyz is popular
        # let's try xyz + rotation

        super().__init__()
        self.num_group = num_group
        self.group_size = (group_size) 
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.attribute = attribute
        self.seed = seed
        attribute_index = []
        if 'xyz' in self.attribute:
            xyz_index = [0,1,2]
            attribute_index.extend(xyz_index)
        if 'opacity' in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if 'scale' in self.attribute:
            scale_index = [4,5,6]
            attribute_index.extend(scale_index)
        if 'rotation' in self.attribute:
            rotation_index = [7,8,9,10]
            attribute_index.extend(rotation_index)
        if 'sh' in self.attribute:
            sh_index = [11,12,13]
            attribute_index.extend(sh_index)

        self.attribute_index = attribute_index

    def forward(self, xyz):
        '''
            input: B N 3 or B N C
            ---------------------------
            output: B G M C
            center : B G C
        '''

        batch_size, num_points, feature_dim = xyz.shape

        center = fps_gs(xyz, self.num_group, self.attribute, random_start=True, random_seed=self.seed) # B G K
        # choose center based on xyz
        # choose neighbor based on new attribute
        center_group = center[...,self.attribute_index]
        xyz_group = xyz[...,self.attribute_index]

        _, idx = self.knn(xyz_group.contiguous(), center_group.contiguous()) # B G M
        group_idx = idx
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size          
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.contiguous().view(-1)
        neighborhood = xyz.contiguous().view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.contiguous().view(batch_size, self.num_group, self.group_size, -1)
        neighborhood[...,:3] = neighborhood[...,:3] - center.unsqueeze(2)[...,:3]


        return neighborhood, center, group_idx
    

 
    
class FeatureScore(nn.Module):
    def __init__(self, embed_dim=1024, out_dim=256, local_size = 8, fusion_mode='full'):
        super().__init__()

        C1 = min(256,embed_dim//4  )
        C2 = min(256,embed_dim//4  )
        self.fusion_mode = fusion_mode
        self.score_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim //2 ),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim //2 , C1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.local_size = local_size
        self.out_dim =out_dim
        self.knn = KNN(k=self.local_size, transpose_mode=True)
        self.first_conv = nn.Sequential(
            nn.Conv2d(6, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
        )

        self.second_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, C2, 1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(C1 + C2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_dim, 1),

        )            
        
        self.fuse_sem = nn.Sequential(
            nn.Conv1d(C1, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_dim, 1),
        )

        self.fuse_local = nn.Sequential(
            nn.Conv1d(C2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_dim, 1),
        )
                
    def forward(self, gs_data, point_embed):
        '''
        gs_data: used for geometry(:3) and color(-3:) information # B N 14
        point_embed: from GS-MAE token, then propagate to each point, 'E_sem' # B N 1024
        return:
            score_local: score is basically calculate from geometry and color
            E_point: fused feature of E_sem and E_local
        
        '''
        B, N, C = gs_data.shape
        xyz = gs_data[:,:,:3].contiguous()
        rgb = gs_data[:, :, -3:].contiguous()
        
        # --- semantic feature: E_sem, semantic global saliency
        E_sem = self.score_proj(point_embed)     # B N C1
        E_sem = E_sem.permute(0, 2, 1).contiguous() # [B,C1,N]

        # --- E_local for color and geometry, need itself
        D, idx = self.knn(ref=xyz, query=xyz)  # B N M(local_size)
        K = idx.shape[-1]
        assert K == self.local_size           
        idx_full = idx                                              # [B,N,K]
        
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, K)  # (B,N,K)

        nb_xyz_full = xyz[batch_idx, idx_full, :]           # (B,N,K,3)

        nb_rgb_full = rgb[batch_idx, idx_full, :]          # (B,N,K,3)

        rel_xyz_full = nb_xyz_full - xyz.unsqueeze(2)                # [B,N,K,3]
        rel_rgb_full = nb_rgb_full - rgb.unsqueeze(2)  # [B,N,K,3]
        local_in = torch.cat([rel_xyz_full, rel_rgb_full], dim=-1).permute(0, 3, 1, 2).contiguous()  # [B,6,N,K]

        feat = self.first_conv(local_in)                             # [B,128,N,K]
        feat_local = feat.max(dim=-1, keepdim=True)[0]               # [B,128,N,1]
        feat = torch.cat([feat_local.expand(-1, -1, -1, K), feat], dim=1)  # [B,256,N,K]
        feat = self.second_conv(feat)                                # [B,C2,N,K]
        E_local = feat.max(dim=-1)[0]                                # [B,C2,N]

        # fuse E_semantic and E_local
        if self.fusion_mode == "sem_only":
            E_point = self.fuse_sem(E_sem)
        elif self.fusion_mode == "local_only":
            E_point = self.fuse_local(E_local)
        else:
            E_point = self.fuse(torch.cat([E_sem, E_local], dim=1)) # [B,out_dim,N]
  

        return E_point


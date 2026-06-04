# code refer https://github.com/qinglew/PCN-PyTorch/blob/master/models/pcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 256
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=256, grid_size=4):
        super().__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Conv1d(512, 3, 1) # xyz + color
        )
        # geometry head: 3 channels
        self.xyz_head = nn.Conv1d(512, 3, 1)

        # color head: 3 channels
        self.color_head = nn.Sequential(
                nn.Conv1d(512, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 3, 1),
            )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # (1, 2, S)

    def forward(self, kpt_xyz, kpt_feature=None, sigmas=None):
        """
        kpt_xyz: B 3 64 
        kpt_feature: B 128 64 (or need to change to 256?)
        sigmas: B 64
        
        return:
        coarse: B 3 1024
        fine: B 6 16384
        """
        

        B, _, N = kpt_xyz.shape
        
        # # encoder
        kpt_xyz = kpt_xyz - kpt_xyz.mean(dim=2, keepdim=True) # relative xyz
        feature_geo = self.first_conv(kpt_xyz)                # (B, 256, N)
        assert kpt_xyz.shape[2] == kpt_feature.shape[2]
        feature = torch.cat([feature_geo, kpt_feature], dim=1)   # (B, 512, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature_global = F.dropout(feature_global, p=0.5, training=self.training)
 
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)

        
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        feature_global = F.dropout(feature_global, p=0.5, training=self.training)

        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.to(kpt_xyz.device).unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)

        # shared trunk
        f = self.final_conv(feat)                                                  # (B, 6, num_fine)
        # geometry branch
        delta_xyz = self.xyz_head(f)               # (B, 3, num_fine)
        fine_xyz  = delta_xyz + point_feat         # (B, 3, num_fine)

        # color branch
        color_logits = self.color_head(f)          # (B, 3, num_fine)
        fine_color = torch.tanh(color_logits)    # (B, 3, num_fine), in [-1,1]

        fine = torch.cat([fine_xyz, fine_color], dim=1)  # (B, 6, num_fine)
        
        # print("fine_xyz range:",
        # fine_xyz.min().item(), fine_xyz.max().item())

        
        return coarse.transpose(1, 2), fine





# ------------------------------------------------------------------------------------------------------------
def fps_gs(data, number, attribute=['xyz'], return_idx = False, random_start = False):
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
        start_idx = torch.randint(0, N, (B,), device=device, dtype=torch.long)  # B


        # 2) 构造排列 perm，并把每行的 start_idx 与位置 0 交换（向量化，无 for）
        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).clone() # B x N (original order)
        rows = torch.arange(B, device=device)          # B

        # 交换第0列和随机起点列（仅交换索引，不动属性）
        perm[rows, 0], perm[rows, start_idx] = perm[rows, start_idx], perm[rows, 0].clone()

        # 3) 在仅交换首位后的序列上做 FPS（等价于随机起点）
        data_fps_shuf = torch.gather(data_fps, 1, perm.unsqueeze(-1).expand(-1, -1, F)).contiguous()
        idx_shuf = pointnet2_utils.furthest_point_sample(data_fps_shuf, number)   # B x M, int32

        # 4) 映射回原始索引（批量索引）
        bidx = torch.arange(B, device=device).unsqueeze(1)                         # B x 1
        fps_idx = perm[bidx, idx_shuf.long()].to(torch.int32).contiguous()  
    else:
        fps_idx = pointnet2_utils.furthest_point_sample(data_fps, number)
    if return_idx:
        return fps_idx
    # print("fps_idx", fps_idx.shape, data.transpose(1, 2).contiguous().shape)
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()


    return fps_data



def show_point_cloud(pc, title="Point Cloud"):
    """
    pc: torch.Tensor or numpy array, shape [N, 3]
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    
if __name__ == "__main__":
    import random
    from pathlib import Path
    from config.path import dataset_dir
    import matplotlib.pyplot as plt
    from config.options import BaseOptions
    import numpy as np
    from datasets.dataset import GSDataset, GS_Detector
    import torch.optim as Optim
    from tqdm import tqdm
    from pointnet2_ops import pointnet2_utils
    from models.losses import cd_loss_L1
    opt = BaseOptions().parse()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    
    # ----- Random seed setup -----
    # If args.seed is given, use it; otherwise generate one randomly
    if hasattr(opt, "seed") and opt.seed is not None:
        seed = opt.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    
    print(f"[INFO] Using random seed: {seed}")

    # Apply seed everywhere for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: make cuDNN deterministic (slower but repeatable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Also store it in args so other parts can access it
    opt.seed = seed
    
    gs_root_path = str(dataset_dir("gs_root"))
    raw_train_dataset = GSDataset(opt,gs_root=gs_root_path, split='train', return_params=False)
    trainset = GS_Detector(raw_train_dataset,mode = 'train')
    # caseA:
    trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)
    # # caseB:
    # trainDataLoader = torch.utils.data.DataLoader(raw_train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)

    
    '''MODEL LOADING'''
    model = PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(opt.device)

    
    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    iters_per_epoch = (len(trainset) + opt.batch_size - 1) // opt.batch_size
    train_step=0
    for epoch in range(501):
        epoch_iter = 0
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif train_step < 50000:
            alpha = 0.5
        else:
            alpha = 1.0
        # training
        model.train()
        with tqdm(trainDataLoader, desc=f"Epoch {epoch+1}/501", unit="batch", total=iters_per_epoch) as tepoch:
            for i, data in enumerate(tepoch):   
                src_gs, dst_gs, R, scale, shift, c = data 
                # c = src_gs[:,:3,:].transpose(1,2).contiguous().to(opt.device).float()
                c = c.float().to(opt.device)
                c_xyz =  c[:,:3,:].transpose(1,2).contiguous().to(opt.device).float()

                
                epoch_iter += len(c[0])  
                pp= fps_gs(c_xyz, 64,attribute=['xyz'], random_start=True)             
                optimizer.zero_grad()

                # forward propagation
                
                coarse_pred, dense_pred,_ = model(pp.transpose(1,2))


                # loss function
                loss1 = cd_loss_L1(coarse_pred, c)
                loss2 = cd_loss_L1(dense_pred, c)

                loss = loss1  #+ alpha * loss2


                # back propagation
                loss.backward()
                optimizer.step()

        train_step += 1
        lr_schedual.step()
        if epoch >0 and epoch % 10 == 0:   # 每10个epoch显示一次
            # 取 batch 中第 1 个点云进行可视化
            input_pc = pp[0].detach().cpu().numpy()            # 输入 FPS 后的点
            coarse_pc = coarse_pred[0].t().detach().cpu().numpy() # coarse 1024 点
            dense_pc  = dense_pred[0].t().detach().cpu().numpy()  # dense 重建点
            print(f"Epoch {epoch} Iter {i} Loss {loss.item():.6f}") 

            show_point_cloud(input_pc,  title=f"Epoch {epoch} - Input (Partial)")
            show_point_cloud(coarse_pc, title=f"Epoch {epoch} - Coarse Output")
            show_point_cloud(dense_pc,  title=f"Epoch {epoch} - Dense Output")


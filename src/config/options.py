from config.pretrain import read_model_params
import torch
from types import SimpleNamespace
import configargparse



# ==== 懒加载缓存，只读一次 ====
_GS_MAE_CACHE = None
def get_gs_mae_params() -> SimpleNamespace:
    global _GS_MAE_CACHE
    if _GS_MAE_CACHE is None:
        mp = read_model_params()
        _GS_MAE_CACHE = SimpleNamespace(**mp)
    return _GS_MAE_CACHE

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None
        self.unknown = []

    def initialize(self, parser: configargparse.ArgumentParser):


        # info
        parser.add_argument('--exp', type=str, default='joint', help='experiment group name, e.g.detector/ joint')
        parser.add_argument('--name', type=str, default='training', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--server', type=int, default=14, help='used workstation, for visualization')
        parser.add_argument('--save', action='store_true', help='whether to save the trained model')
        parser.add_argument('--seed', type=int, default=123, help='for reproducibility')
        parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training') # has to be 50 when using encoder_USIP
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
        parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
        parser.add_argument('--resume_dir', type=str, default=None, help='resume training')
        # training setup
        parser.add_argument('--max_epoch', type=int, default=50, help='max_epoch')
        parser.add_argument('--save_every', type=int, default=10, help='save_every')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--desired_number', type=int, default=None, help='number for training, None for all; val is set to 1/100 of training')
        parser.add_argument("--finetune_ckpt", type=str, default=None)
        parser.add_argument("--finetune_strict", action="store_true") # optional
        
        # detector
        parser.add_argument('--dataset', type=str, default='shapesplat', choices=["shapesplat", "scared"])
        parser.add_argument('--sampling_rule', type=str, default='random', help='gs sampling rule: [random, opacity, scale, mixed]')
        # parser.add_argument('--encoder_freeze', type=bool, default=True)
        parser.add_argument('--feature_dim', type=int, default=128, help='dimension of descriptor')
        parser.add_argument('--n_keypoints', type=int, default=None, help='number of keypoints')
        
        # matcher
        parser.add_argument("--rpthres", type=float, help="", default=0.01)
        parser.add_argument("--inliers_only", type=bool, help="", default=False)
        parser.add_argument("--matcher_type", type=str, default="OT", choices=["OT", "Cls", 'GoMatch'])
        
        # ablation 
        parser.add_argument("--encoder_mode", type=str, default="freeze", choices=["freeze", "finetune_all", "finetune_last"])
        parser.add_argument("--prop_mode", type=str, default="fp", choices=["fp", "nearest_group"])
        parser.add_argument("--fusion_mode", type=str, default="full", choices=["full", "sem_only", "local_only"])
        parser.add_argument('--local_size', type=int, default=8, help='local size for point feature')
        parser.add_argument("--finetune_last_k", type=int, default=1)
        

    
        self.initialized = True
        return parser
    
    def gather_options(self, args=None, unknown_ok=False):
        # 1) 初始化 parser（并把状态持久化）
        if not self.initialized:
            parser = configargparse.ArgumentParser(
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
            self.parser = parser
            self.initialized = True
        else:
            # 如果已经初始化过，就复用同一个 parser
            parser = self.parser

        # 2) 解析参数（可选放行未知参数）
        if unknown_ok:
            opt, unknown = parser.parse_known_args(args=args)
        else:
            opt = parser.parse_args(args=args)
            unknown = []

        # 3) 缓存解析结果，便于其他方法使用
        self.opt = opt
        self.unknown = unknown

        # 4) 返回结果（保持两个返回值的约定，和你现在一致）
        return opt, unknown
    
    
    def parse(self,args = None, unknown_ok = False):
        opt, unknown = self.gather_options(args=args, unknown_ok=unknown_ok)

        # === 旧逻辑 ===
        if opt.name == "USIP":
            opt.batch_size = 50

        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === ✅ 自动挂载预训练模型参数 ===
        opt.gs_mae = get_gs_mae_params()
        if opt.n_keypoints == None:
            opt.n_keypoints = opt.gs_mae.num_group

        self.opt = opt
        if unknown_ok:
            return opt, unknown
        
        # # --- modify learning rate
        # if opt.learning_rate is None:
        #     if opt.encoder_mode == "freeze":
        #         opt.learning_rate = 1e-3
        #     elif opt.encoder_mode == "finetune_last":
        #         opt.learning_rate = 3e-4
        #     else:  # finetune_all
        #         opt.learning_rate = 1e-4        
        
        
        return opt    
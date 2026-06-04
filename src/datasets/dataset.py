import numpy as np
import math
import h5py
import os
import traceback
from typing import Dict, List, Tuple, Optional
from plyfile import PlyData
from torch.utils.data import Dataset
import torch
import zlib
import glob
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import get_worker_info
import random

from datasets.augmentation import *
from datasets.geom_match import points2d_to_bearing_vector, project3d_normalized
from config.path import split_file
from config.pretrain import read_model_params
from util.gs_tools import *
from util.util import stem_dir, make_pose_rng




class GSDataset(Dataset):
    """
    On-demand GS loader with scene->path index + LRU cache.

    - ShapeSplatDataset uses: get_by_scene(scene, need_full=False) -> gs_sparse
    - GS_Detector uses:      get_by_scene(scene, need_full=True)  -> gs_sparse + gs_full
    """
    def __init__(self, 
                 opt,
                 gs_root='./data/gs_data', 
                 npoints=None,
                 split=None, 
                 return_params = False,
                 full_points=16384,
                 sampling_rule=None
                 ):
        # config
        self.opt = opt
        self.gs_root = gs_root
        self.npoints = int(opt.gs_mae.npoints)
        self.split = split
        self.full_points = int(full_points)
        self.return_params = bool(return_params)
        
        self.attribute = ['xyz','opacity','scale','rotation','sh']
        self.target_taxonomy = None
        
        if sampling_rule is None:
            self.sampling_rule = getattr(opt, "sampling_rule", "random")
        else:
            self.sampling_rule = sampling_rule


        self.desired_number = getattr(opt, "desired_number", None)
        self.rng = np.random.default_rng(getattr(opt, "seed", 123))
        
        
        # ----- build file list -----
        split_txt = split_file(self.split)
        if not os.path.exists(split_txt):
            raise FileNotFoundError(f"Cannot find split list: {split_txt}")

        with open(split_txt, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # （可选）按 taxonomy 过滤
        if self.target_taxonomy:
            lines = [ln for ln in lines if ln.split('-')[0] == self.target_taxonomy]
            
        
        # # 若需要再限制数量/随机抽样（沿用你现有 random/desired_number 语义）
        if self.desired_number is not None:
            if self.split == 'val': 
                self.desired_number= max(1,self.desired_number//6)



        candidate_paths = []
        for line in lines:
            taxonomy_id = line.split('-')[0]
            candidate_paths.append(os.path.join(self.gs_root, taxonomy_id, line))

        if len(candidate_paths) == 0:
            raise RuntimeError(f"No valid files found for split={self.split} (from train.txt) "
                                f"and taxonomy={self.target_taxonomy}")

        # preload and guarantee desired_number
        self.samples = []
        self.datapath = []
        self.scene_to_index = {}
        
        target = self.desired_number if self.desired_number is not None else len(candidate_paths)
        
        
        for path in candidate_paths:
            if self.desired_number is not None and len(self.samples) >= target:
                break

            try:
                pack = self.gs_pack(path)
                scene_id = pack["scene"]
                


                # if duplicates ever happen, keep the first
                if scene_id in self.scene_to_index:
                    continue

                self.scene_to_index[scene_id] = len(self.samples)
                self.samples.append(pack)
                self.datapath.append(path)

            except Exception as e:
                print(f"[WARN] skip bad GS: {path}, error: {e}")
                print(traceback.format_exc())
                try:
                    with open("bad_samples.txt", "a", encoding="utf-8") as f:
                        f.write(f"[GS preload] path={path}\n  need_full={self.need_full}\n  error={repr(e)}\n---\n")
                except Exception:
                    pass

        # strict desired ALWAYS
        if self.desired_number is not None and len(self.samples) < target:
            raise RuntimeError(
                f"desired_number={target}, but only {len(self.samples)} valid GS loaded "
                f"from {len(candidate_paths)} candidates. (need_full={self.need_full})"
            )

        # dataset length: exactly desired_number if set; else all valid
        self.dataset_len = target if self.desired_number is not None else len(self.samples)
 
    # ---------------- Dataset API ----------------
    def __getitem__(self, index):
        return self.samples[index]
        
    
    def __len__(self):
        return self.dataset_len

    # --------- scene index ---------        
    def get_path_by_scene(self, scene: str) -> str:
        scene_id = stem_dir(scene)
        idx = self.scene_to_index.get(scene_id, None)
        if idx is None:
            raise KeyError(f"Scene not found: {scene} (stem={scene_id})")
        return self.samples[idx]["path"]
    
    def get_by_scene(self, scene: str):
        scene_id = stem_dir(scene)
        idx = self.scene_to_index.get(scene_id, None)
        if idx is None:
            raise KeyError(f"Scene not found: {scene} (stem={scene_id})")
        return self.samples[idx]
    
    # --------- core processing ---------
    def pc_norm_gs(self, pc, attribute= ['xyz','opacity','scale','rotation','sh'], extra_pc=None, return_params=False):
        """ pc: NxC, return NxC """
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
        pc_xyz = pc_xyz / m
        # inside a sphere
        pc[..., :3] = pc_xyz
        pc[..., 4:7] = pc[..., 4:7] / m 

        if 'opacity' in attribute:
            # normalize to a -1 to 1 range
            min_opacity = 0
            max_opacity = 1
            pc[..., 3] = (pc[..., 3] - min_opacity) / \
                (max_opacity - min_opacity) * 2 - 1

        if 'scale' in attribute:
            # normalize to a -1 to 1 range
            s_center = np.mean(pc[..., 4:7], axis=0)
            pc[..., 4:7] = pc[..., 4:7] - s_center
            s_m = np.max(np.sqrt(np.sum(pc[..., 4:7]**2, axis=1)))
            pc[..., 4:7] = pc[..., 4:7] / s_m
        else:
            s_center = np.zeros(3)
            s_m = 1

        if 'sh'  in attribute:
            sh = pc[...,11:14]
            sh = sh * 0.28209479177387814 
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)  
            pc[...,11:14] = sh


        params = dict(centroid=centroid, m=m, s_center=s_center, s_m=s_m)

        return (pc, params) if return_params else pc
    

    def gs_pack(self, path: str):
        """Read one .ply, sample 1024 (+ optional 16384), normalize, return dict."""
        gs = IO.get(path)
        vertex = gs['vertex']
        gs_original = read_gaussian_attribute(vertex, self.attribute)
        

            
        # sample by scene id
        def make_scene_rng(base_seed: int, scene: str):
            scene_hash = zlib.adler32(scene.encode("utf-8"))  # 稳定 hash
            seed = (base_seed + scene_hash) % (2**32)
            return np.random.default_rng(seed)
        scene_id = stem_dir(path)
        rng_scene = make_scene_rng(getattr(self.opt, "seed", 123), scene_id)
        # 采样 1024 / npoints

        # if 'train' not in self.split:
        idx_sparse = sample_gs(
                gs_original, self.npoints,
                rule=self.sampling_rule, replace=True, rng=rng_scene)
        gs_sparse = gs_original[idx_sparse, :]
        gs_sparse, params = self.pc_norm_gs(gs_sparse, attribute=self.attribute, return_params=True)

        pack = {
            "scene": stem_dir(path),
            "path": path,
            "gs_sparse": gs_sparse.astype(np.float32),
        }        
            
        # else:
        #     idx_full = sample_gs(
        #         gs_original, self.full_points,
        #         rule=self.sampling_rule, replace=True, rng=rng_scene
        #     )
        #     gs_full = gs_original[idx_full, :]
            

            
        #     # pack["gs_full"] = gs_full
        #     # pack["gs_full"] = self.pc_norm_gs(gs_full, return_params=False).astype(np.float32)
            
        #     # # debug
        #     # idx_sparse = sample_gs(
        #     #         gs_original, self.npoints,
        #     #         rule=self.sampling_rule, replace=True, rng=rng_scene)
        #     # gs_sparse = gs_original[idx_sparse, :]
        #     # gs_sparse, params = self.pc_norm_gs(gs_sparse, attribute=self.attribute, return_params=True)
            
        #     pack = {
        #         "scene": stem_dir(path),
        #         "path": path,
        #         "gs_full": gs_full.astype(np.float32),
        #         # "gs_sparse": gs_sparse.astype(np.float32),
        #     }                    
        #     params = None
            
        if self.return_params:
            pack["params"] = params

        return pack




def build_joint_items(
    base_dataset: Dataset,
    match_root: str,
    xfeat_root: str,
    *,
    throttle_sleep: float = 0.002,
) -> List[Tuple[int, str, str, str, str]]:
    """
    Build list of (base_i, scene, qk, xfeat_path, match_path).
    Run this in MAIN PROCESS before DataLoader starts.
    """
    items: List[Tuple[int, str, str, str, str]] = []

    scan_open_kwargs = dict(retry_forever=False, retries=2000, sleep_max=10.0)

    for base_i in range(len(base_dataset)):
        pack = base_dataset[base_i]
        p = Path(pack["path"])
        cat = p.parent.name
        scene = p.stem

        xfeat_path = str(Path(xfeat_root) / cat / f"{scene}.h5")
        match_path = str(Path(match_root) / cat / f"{scene}.h5")

        try:
            with h5_open_retry(xfeat_path, "r", **scan_open_kwargs) as fx:
                qks = sorted(fx["queries"].keys())
        except Exception as e:
            print(f"[WARN] build_joint_items: skip scene={scene} ({xfeat_path}) err={repr(e)}")
            continue

        for qk in qks:
            items.append((base_i, scene, qk, xfeat_path, match_path))

        if throttle_sleep and throttle_sleep > 0:
            time.sleep(throttle_sleep)

    return items

class JointDataset(Dataset):
    def __init__(
        self,
        opt,
        base_dataset: Dataset,
        items: List[Tuple[int, str, str, str, str]],
        *,
        h5_cache_size: int = 16,
        drop_on_transient_error: bool = True,
        is_training:bool = False,
    ):
        """
        drop_on_transient_error=True:
          if network FS glitches (errno 112 etc.), return empty sample instead of crashing job.
        """
        self.base_dataset = base_dataset
        self.items = items
        self.npoints = int(opt.gs_mae.npoints)
        self.matcher_type = opt.matcher_type

        self._h5_cache_by_worker = {}
        self._h5_cache_size = int(h5_cache_size)
        self.drop_on_transient_error = bool(drop_on_transient_error)
        self.is_training = False
        self.opt = opt

    def __len__(self):
        return len(self.items)

    def _worker_id(self) -> int:
        wi = get_worker_info()
        return wi.id if wi is not None else 0

    def _get_h5(self, path: str):
        wid = self._worker_id()
        cache: OrderedDict = self._h5_cache_by_worker.get(wid)
        if cache is None:
            cache = OrderedDict()
            self._h5_cache_by_worker[wid] = cache

        f = cache.get(path)
        if f is not None:
            # 句柄可能因网络抖动变“坏”，这里做一个轻量 check
            try:
                _ = f.id.valid
                cache.move_to_end(path)
                return f
            except Exception:
                try:
                    f.close()
                except Exception:
                    pass
                cache.pop(path, None)

        # open with retry (training stage: you want to keep trying)
        f = h5_open_retry(path, "r", retry_forever=False, sleep_max=2.0)
        cache[path] = f
        cache.move_to_end(path)

        while len(cache) > self._h5_cache_size:
            _, old_f = cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        return f

    def __getitem__(self, idx: int) -> dict:
        try:
        
            base_i, scene, qk, xfeat_path, match_path = self.items[idx]

            # xfeat
            f_xfeat = self._get_h5(xfeat_path)
            g = f_xfeat["queries"][qk]
            K = g["intrinsics"][...].astype(np.float32)
            R_c2w = g["rotation"][...].astype(np.float32)
            C = g["position"][...].astype(np.float32)
            R = R_c2w.T.astype(np.float32)
            t = (-R @ C).astype(np.float32)

            width = int(g.attrs["width"])
            height = int(g.attrs["height"])
            pts2d_pix = g["kptuv"][...].astype(np.float32)
            if self.matcher_type == 'GoMatch':
                desc2d = None
            else:
                desc2d = g["feat"][...].astype(np.float32)
            # conf2d = g["conf"][...].astype(np.float32)
            
            # from GS_Dataset
            pack = self.base_dataset[base_i]
            gs_np = pack["gs_sparse"].astype(np.float32)  # (N,C)
            params = pack.get("params", None)

            # gt match
            fgt = self._get_h5(match_path)
            gq = fgt["queries"][qk]
            pts3d_norm = gq["pts3d_norm"][...].astype(np.float32)
            centroid = gq["centroid"][...].astype(np.float32)
            m = float(np.asarray(gq["m"][...]).reshape(()))
            pts3d = (pts3d_norm * m + centroid ).astype(np.float32)
            mb = gq["matches_bin"][...]
            matches_bin = mb.astype(np.uint8) if mb.dtype != np.uint8 else mb

            # # sanity check: 
            # # 1. params vs. centroid, m
            # # 2. gs_np vs. pts3d_norm
            # # 3. matches_bin

            # xyz_gs = gs_np[:, :3]
            # xyz_gt = pts3d_norm

            # centroid_diff = np.max(np.abs(params["centroid"] - centroid))
            # m_diff = abs(float(params["m"]) - float(m))
            # xyz_diff = np.max(np.abs(xyz_gs - xyz_gt))
            # xyz_ok = np.allclose(xyz_gs, xyz_gt, atol=1e-4)

            # print(
            #     f"[DEBUG] {scene}/{qk} "
            #     f"centroid_diff={centroid_diff:.10e} "
            #     f"m_diff={m_diff:.10e} "
            #     f"xyz_diff={xyz_diff:.10e} "
            #     f"xyz_ok={xyz_ok} "
            #     f"params_m={float(params['m']):.10e} "
            #     f"h5_m={float(m):.10e}",
            #     flush=True
            # )

            # assert np.allclose(
            #     params["centroid"], centroid, atol=1e-6
            # ), f"[{scene}/{qk}] centroid mismatch"

            # assert np.allclose(xyz_gs, xyz_gt, atol=1e-4), \
            #     f"[{scene}/{qk}] gs xyz not aligned with gt"

            # assert np.isclose(float(params["m"]), float(m), rtol=1e-5, atol=1e-6), \
            #     f"[{scene}/{qk}] m mismatch: params={params['m']}, h5={m}"

            # normalized pts2d_pix
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            pts2d = np.stack([(pts2d_pix[:,0] - cx) / fx,
                            (pts2d_pix[:,1] - cy) / fy], axis=1).astype(np.float32)
            if self.matcher_type == 'GoMatch':
                pts2dm = points2d_to_bearing_vector(pts2d_pix, K, vec_dim=2).astype(np.float32)
                pts3dm = project3d_normalized(R, t, pts3d).astype(np.float32)
            else:
                pts2dm = pts2d
                pts3dm = gs_np
            
            
            
            data = dict(
                name=f"{scene}/{qk}",
                # 2D
                pts2d=pts2d,
                pts2d_pix=pts2d_pix,
                pts2dm=pts2dm,
                # desc2d=desc2d ,
                # 3D
                # gs3d=gs_np.astype(np.float32),
                pts3d = pts3d,
                pts3dm = pts3dm,


                matches_bin=matches_bin,
                R=R,
                t=t,
                K=K,
                width=width,
                height=height,
            )
            
            if self.matcher_type != 'GoMatch':
                data["desc2d"] = desc2d

            return data
        
        except Exception as e:
            if self.drop_on_transient_error and (_is_transient_oserror(e) or isinstance(e, RuntimeError)):
                return {"name": None, "pts2d": np.zeros((0, 2), np.float32)}
            raise

import errno
import time

# 常见网络/分布式文件系统瞬态错误
_TRANSIENT_ERRNOS = {
    errno.EAGAIN,        # 11
    errno.EHOSTDOWN,     # 112 Host is down
    errno.ETIMEDOUT,     # 110
    errno.EIO,           # 5
    errno.ESTALE,        # 116 stale file handle (NFS)
    errno.ECONNRESET,    # 104
    errno.ENETDOWN,      # 100
    errno.ENETUNREACH,   # 101
    errno.ECONNABORTED,  # 103
}


def _is_transient_oserror(e: Exception) -> bool:
    if isinstance(e, OSError):
        en = getattr(e, "errno", None)
        if en in _TRANSIENT_ERRNOS:
            return True
        # 有的 h5py 错误 errno 可能是 None，但 message 里有 Host is down
        if "Host is down" in str(e):
            return True
    return False
def h5_open_retry(
    path: str,
    mode: str = "r",
    *,
    sleep_min: float = 0.05,
    sleep_max: float = 2.0,
    backoff: float = 1.7,
    print_every: int = 50,
    retry_forever: bool = True,
    retries: int = 120,   # when retry_forever=False
):
    """
    Robust open for HDF5 on unstable network FS.

    - retry_forever=True: never give up (may stall if storage stays down)
    - retry_forever=False: retry limited times (good for index-building stage)
    """
    attempt = 0
    delay = sleep_min

    while True:
        try:
            return h5py.File(path, mode)
        except FileNotFoundError:
            # real missing file: do not hide
            raise
        except BlockingIOError as e:
            err = getattr(e, "errno", errno.EAGAIN)
        except OSError as e:
            err = getattr(e, "errno", None)
            if (err is not None) and (err not in _TRANSIENT_ERRNOS):
                # non-transient => raise immediately
                raise

        attempt += 1
        if (attempt % print_every) == 1:
            print(f"[h5_open_retry pid={os.getpid()}] fail#{attempt} errno={err} sleep~{delay:.2f}s: {path}", flush = True)

        if (not retry_forever) and (attempt >= retries):
            raise RuntimeError(f"Failed to open HDF5 after {retries} retries: {path} (errno={err})")

        jitter = random.uniform(0.8, 1.2)
        time.sleep(min(delay * jitter, sleep_max))
        delay = min(delay * backoff, sleep_max)




# if __name__ == "__main__":
#     import argparse
#     import random
#     from pathlib import Path
#     from config.path import dataset_dir
#     import shutil
#     from plyfile import PlyData, PlyElement
#     import math
#     import matplotlib.pyplot as plt

#     # 保存 Gaussian Splatting 数据为 ply 文件（结构同原始）

#     # --- 仅 main 内部用的最小工具 ---
#     def write_from_array(save_path: Path, arr: np.ndarray):
#         """将 (N,14) 数组写成标准 GS PLY 字段"""
#         assert arr.ndim == 2 and arr.shape[1] == 14
#         N = arr.shape[0]
#         dtype = [
#             ('x','f4'),('y','f4'),('z','f4'),
#             ('opacity','f4'),
#             ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
#             ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
#             ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
#         ]
#         verts = np.empty(N, dtype=dtype)
#         verts['x']=arr[:,0]; verts['y']=arr[:,1]; verts['z']=arr[:,2]
#         verts['opacity']=arr[:,3]
#         verts['scale_0']=arr[:,4]; verts['scale_1']=arr[:,5]; verts['scale_2']=arr[:,6]
#         verts['rot_0']=arr[:,7]; verts['rot_1']=arr[:,8]; verts['rot_2']=arr[:,9]; verts['rot_3']=arr[:,10]
#         verts['f_dc_0']=arr[:,11]; verts['f_dc_1']=arr[:,12]; verts['f_dc_2']=arr[:,13]
#         el = PlyElement.describe(verts, 'vertex')
#         PlyData([el], text=False).write(str(save_path))


#     def np_logit(p, eps=1e-6):
#         p = np.clip(p, eps, 1-eps)
#         return np.log(p/(1-p))


#     def invert_pc_norm_gs(pc_norm, params, attribute=['xyz']):
#         """把 pc_norm_gs 的归一化反过来（需要你修改后的 pc_norm_gs 返回的 params）"""
#         pc = pc_norm.copy()
#         centroid = params['centroid']; m = params['m']; s_center = params['s_center']; s_m = params['s_m']
#         if 'scale' in attribute:
#             pc[:,4:7] = pc[:,4:7]*s_m + s_center
#         pc[:,4:7] = pc[:,4:7]*(m + 1e-12)
#         pc[:,:3]  = pc[:,:3]*(m + 1e-12) + centroid
#         if 'opacity' in attribute:
#             pc[:,3] = (pc[:,3] + 1.0)*0.5
#         if 'sh' in attribute:
#             sh = pc[:,11:14]
#             sh = sh*(math.sqrt(3)/2.0)
#             sh = sh/0.28209479177387814
#             pc[:,11:14] = sh
#         return pc

#     def invert_read_domain(arr, attribute=['xyz','opacity','scale','rotation','sh']):
#         """把 read_gaussian_attribute 输出域 → 原 PLY 存储域（用于写盘）"""
#         out = arr.copy()
#         if 'opacity' in attribute:
#             out[:,3] = np_logit(out[:,3])              # [0,1] → raw
#         if 'scale' in attribute:
#             out[:,4:7] = np.log(np.maximum(out[:,4:7], 1e-12))  # 正数 → log域
#         # rotation/sh 直接保留
#         return out

#     # ---------------- 参数解析 ----------------
#     parser = argparse.ArgumentParser(description="随机读取一个 Gaussian Splatting 数据并导出 src/dst")
#     parser.add_argument("--sampling_rule", type=str, default="random", help="choose from random, mixed, opacity, ")
#     parser.add_argument("--idx", type=int, default=None, help="固定索引，如果为 None 就随机选一个")
#     parser.add_argument("--seed", type=int, default=11, help="随机种子")
#     parser.add_argument("--mode", type=str, default="eval", help="传给 GS_Detector 的 mode")
#     args = parser.parse_args()



#     out_dir = Path('/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/localdata/check_data')
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print('sampling rule: ', args.sampling_rule)

#     # ---------------- random seed ----------------
#     seed = int(args.seed)   # 来自命令行参数
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # 如用到 cudnn，可以加下面两行保障确定性（略有性能损耗）
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # ---------------- Build dataset ----------------
#     gs_root = str(dataset_dir("gs_root"))
#     base = GSDataset(
#         args,
#         gs_root=gs_root,
#         # desired_number=1,          # use full split list
#         split='train',
#         target_taxonomy=None,
#         random=False,
#         return_params = True,
#         sampling_rule=args.sampling_rule
#     )
#     baseset = GS_Detector(base,mode = 'train')

#     if len(baseset) == 0:
#         raise RuntimeError("数据集为空")



#     for idx, data in enumerate(baseset):
#         src_gs, dst_gs, R, scale, shift, params = data

#         chosen_path = base.datapath[idx]
#         stem = Path(chosen_path).stem

#         # 2) original：原样复制
#         original_out = out_dir / f"{stem}.original.ply"
#         shutil.copyfile(chosen_path, original_out)

#         # 1) tensors (C,N) → numpy (N,14)
#         src_np = src_gs.t().cpu().numpy()
#         dst_np = dst_gs.t().cpu().numpy()

#         # 2) invert dataset normalization to original metric space
#         #    (uses the per-sample params returned by pc_norm_gs)
#         src_denorm = invert_pc_norm_gs(src_np, params, attribute=base.attribute)
#         dst_denorm = invert_pc_norm_gs(dst_np, params, attribute=base.attribute)

#         # 3) convert attribute domains back to PLY storage domain
#         #    (opacity: sigmoid^-1, scale: log, rot/sh as-is)
#         src_to_write = invert_read_domain(src_denorm, attribute=base.attribute)
#         dst_to_write = invert_read_domain(dst_denorm, attribute=base.attribute)

#         # 4) write .ply
#         src_out = out_dir / f"{stem}.src.ply"
#         dst_out = out_dir / f"{stem}.dst.ply"
#         write_from_array(src_out, src_to_write.astype(np.float32))
#         write_from_array(dst_out, dst_to_write.astype(np.float32))
        
#         # See the point cloud
#         coords = src_denorm[:, :3]
#         c = coords[:,2]

#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         ax.scatter(coords[:,0], coords[:,1], coords[:,2],
#                 s=8, c='grey', edgecolors='none', alpha=0.9)

#         # 自动设置范围
#         mins = coords.min(axis=0); maxs = coords.max(axis=0)
#         ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])

#         ax.view_init(elev=20, azim=30)
#         plt.tight_layout()

#         # 强烈建议同时保存，方便在非交互后端查看
#         # plt.savefig("gs_sample_rgb.png", dpi=200)
#         plt.show()

#         print(f"[OK] Src     : {src_out}  (N={src_to_write.shape[0]})")
#         print(f"[OK] Dst     : {dst_out}  (N={dst_to_write.shape[0]})")
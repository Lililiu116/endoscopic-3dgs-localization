# Specifically dealing our GS data!

import numpy as np
import os
from plyfile import PlyData
import h5py
from scipy.spatial import cKDTree
from typing import Optional


def sample_gs(gs_data, npoints, rule="random", replace=True, rng = None):
    """
    从 Gaussian Splat 数据中选择子集（Salient rule）

    参数:
        gs_data : np.ndarray, shape (N, 14)
            Gaussian splat 数据，通道定义:
              0:3   -> 坐标
              3     -> opacity
              4:7   -> scale
              7:11  -> rotation (quaternion)
              11:14 -> color (RGB)
        npoints : int
            采样数量
        rule : str
            采样规则:
              "random"  -> 均匀随机
              "opacity" -> 按 opacity 权重
              "scale"   -> 小尺度优先 (1/scale)
              "mixed"   -> opacity / scale
        replace : bool
            是否允许有放回采样

    返回:
        choice_idx : np.ndarray
            被选择的 GS 索引，shape (npoints,)
    """

    N = len(gs_data)
    if rng is None:
    # backward compatible fallback
        rng_choice = np.random.choice
    else:
        rng_choice = rng.choice


    if rule == "random":
        choice_idx = rng_choice(N, npoints, replace=replace)

    elif rule == "opacity":
        opacities = gs_data[:, 3]
        weights = opacities + 1e-8
        weights /= weights.sum()
        choice_idx = rng_choice(N, npoints, replace=replace, p=weights)
        
   
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return choice_idx




def _normalize_to_prob(x, eps=1e-12):
    w = x.astype(np.float64)
    w[~np.isfinite(w)] = eps
    s = w.sum()
    return np.full_like(w, 1.0/len(w)) if (s <= 0 or not np.isfinite(s)) else (w / s)





class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.ply']:
            return cls._read_ply(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
    
    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

   
def read_gaussian_attribute(vertex, attribute):
    # assert 'xyz' in attribute, 'At least need xyz attribute'
    if 'xyz' in attribute:
        x = vertex['x'].astype(np.float32)
        y = vertex['y'].astype(np.float32)
        z = vertex['z'].astype(np.float32)
        data = np.stack((x, y, z), axis=-1) # [n, 3]

    if 'opacity' in attribute:
        opacity = vertex['opacity'].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        # opacity range from 0 to 1
        data = np.concatenate((data, opacity), axis=-1)
        # print("data", data.shape)

    if 'scale' in attribute and 'rotation' in attribute:
        scale_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        
        scales = np.exp(scales)  # scale normalization
        
        # print("scales", scales.min(), scales.max())

        rot_names = [
            p.name for p in vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)
 
    if 'sh' in attribute:
        # sphere homrincals to rgb
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex['f_dc_0'].astype(np.float32)
        features_dc[:, 1, 0] = vertex['f_dc_1'].astype(np.float32)
        features_dc[:, 2, 0] = vertex['f_dc_2'].astype(np.float32)
  
        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)


    return data


def sample_view_based_rays(
    Xw_full: np.ndarray,         # (N,3) world points
    K: np.ndarray,
    R_c2w: np.ndarray,
    C_world: np.ndarray,
    n: int,
    rng: np.random.Generator,
    width: int = 400,
    height: int = 400,
    zmin: float = 1e-3,
    max_candidates: Optional[int] = None,  # None = 用全部可见候选，最稳定
    topk: int = 1,                       # 1=最近点；>1=最近topk随机(更鲁棒)
):
    """
    返回: idx (n,) int64
    repeatable: 只要 rng 固定，返回固定
    """
    Xw = np.asarray(Xw_full, np.float32).reshape(-1, 3)
    N = Xw.shape[0]
    C = np.asarray(C_world, np.float32).reshape(3,)
    R_c2w = np.asarray(R_c2w, np.float32).reshape(3, 3)
    K = np.asarray(K, np.float32).reshape(3, 3)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # --- world->cam 过滤候选（只保留前方 + 在frame内） ---
    R_wc = R_c2w.T
    t_wc = (-R_wc @ C).astype(np.float32)
    Xc = (R_wc @ Xw.T + t_wc.reshape(3, 1)).T
    z = Xc[:, 2]

    in_front = z > zmin
    u = fx * (Xc[:, 0] / (z + 1e-6)) + cx
    v = fy * (Xc[:, 1] / (z + 1e-6)) + cy
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    ok = in_front & in_frame
    if not np.any(ok):
        # 退化：全随机，但仍可重复
        return rng.choice(N, size=n, replace=(N < n)).astype(np.int64)

    Xw_ok = Xw[ok]
    idx_ok = np.nonzero(ok)[0]

    # 可选：为了速度抽子集（用同一个 rng，仍可重复）
    if (max_candidates is not None) and (Xw_ok.shape[0] > max_candidates):
        sel = rng.choice(Xw_ok.shape[0], size=max_candidates, replace=False)
        Xcand = Xw_ok[sel]
        Icand = idx_ok[sel]
    else:
        Xcand = Xw_ok
        Icand = idx_ok

    M = Xcand.shape[0]

    # --- 采样更多射线以减少重复 ---
    # oversample factor：候选少/结构简单时重复会多，适当多采一些 ray
    oversample = 3
    n_rays = max(n * oversample, n)

    u_r = rng.uniform(0, width, size=n_rays).astype(np.float32)
    v_r = rng.uniform(0, height, size=n_rays).astype(np.float32)

    # camera rays
    x = (u_r - cx) / fx
    y = (v_r - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=1)
    d_cam /= (np.linalg.norm(d_cam, axis=1, keepdims=True) + 1e-8)

    # to world
    d_world = (R_c2w @ d_cam.T).T
    d_world /= (np.linalg.norm(d_world, axis=1, keepdims=True) + 1e-8)

    # --- 点到射线距离 ---
    # V: (n_rays, M, 3)
    V = Xcand[None, :, :] - C[None, None, :]
    s = np.sum(V * d_world[:, None, :], axis=2)          # (n_rays, M)
    in_front2 = s > zmin

    V_perp = V - s[:, :, None] * d_world[:, None, :]
    dist = np.linalg.norm(V_perp, axis=2)                # (n_rays, M)
    dist = np.where(in_front2, dist, np.inf)

    if topk <= 1:
        j = np.argmin(dist, axis=1)                      # (n_rays,)
        idx_r = Icand[j].astype(np.int64)
    else:
        # 每条 ray 取 topk 个最近点，然后随机选1个（更鲁棒、减少过拟合到某个点）
        topk = int(topk)
        j_topk = np.argpartition(dist, kth=topk-1, axis=1)[:, :topk]   # (n_rays, topk)
        # 对 topk 内按距离做 softmin 概率
        d_top = np.take_along_axis(dist, j_topk, axis=1)               # (n_rays, topk)
        # 防止全 inf
        bad = ~np.isfinite(d_top).any(axis=1)
        # 对有效行做概率
        d_top = np.where(np.isfinite(d_top), d_top, 1e9)
        w = np.exp(-d_top / (np.min(d_top, axis=1, keepdims=True) + 1e-6))
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        pick = np.array([rng.choice(topk, p=w[i]) for i in range(n_rays)], dtype=np.int64)
        j = j_topk[np.arange(n_rays), pick]
        idx_r = Icand[j].astype(np.int64)
        # bad 行退化随机
        if np.any(bad):
            idx_r[bad] = rng.choice(Icand, size=int(bad.sum()), replace=True)

    # --- 去重并补齐到 n（保证输出长度固定 + 可重复） ---
    idx_unique = np.unique(idx_r)
    if idx_unique.shape[0] >= n:
        # 为了可重复：对 unique 再用 rng 选 n 个（不要直接切片，因为 unique 会按排序改变分布）
        return rng.choice(idx_unique, size=n, replace=False).astype(np.int64)
    else:
        # 不够：把剩下的从候选里补齐（不重复优先）
        need = n - idx_unique.shape[0]
        # 可补的集合
        mask = np.ones(M, dtype=bool)
        # Icand 是候选原索引
        # 用集合差：避免 Python set（慢），这里用 isin
        mask = ~np.isin(Icand, idx_unique)
        pool = Icand[mask]
        if pool.shape[0] >= need:
            extra = rng.choice(pool, size=need, replace=False).astype(np.int64)
        else:
            extra = rng.choice(Icand, size=need, replace=True).astype(np.int64)
        return np.concatenate([idx_unique.astype(np.int64), extra], axis=0)


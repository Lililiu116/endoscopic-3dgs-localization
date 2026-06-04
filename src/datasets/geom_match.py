from argparse import Namespace
import os
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from scipy.spatial import cKDTree

import numpy as np
import torch
from torch import Tensor


TensorOrArray = Union[torch.Tensor, np.ndarray]
TensorOrArrayOrList = Union[torch.Tensor, np.ndarray, List]


def cal_mutual_nn_dists_kdtrees(
    nn12: np.ndarray,
    nn21: np.ndarray,
    dist12: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Mutual nearest matches wrt min distances
    ids1 = np.arange(0, len(nn12))
    mutual_mask = ids1 == nn21[nn12]
    ids1 = ids1[mutual_mask]
    ids2 = nn12[mutual_mask]
    match_ids = np.stack([ids1, ids2]).T
    match_dists = dist12[mutual_mask]
    if threshold:
        thres_mask = match_dists < threshold
        match_ids = match_ids[thres_mask]
        match_dists = match_dists[thres_mask]
    return match_ids, match_dists

def align_points2d(
    pts1: np.ndarray, pts2: np.ndarray, dist_thres: Optional[float] = None
) -> np.ndarray:
    # Measure pair-wise distances
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    dist12, nn12 = tree2.query(pts1)
    dist21, nn21 = tree1.query(pts2)

    # Define inliers with the nearest mutual matches
    aligned_ids, _ = cal_mutual_nn_dists_kdtrees(
        nn12, nn21, dist12, threshold=dist_thres
    )
    return aligned_ids


def undistort(k: float, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dtype = u.dtype
    u, v = u.astype(np.float64), v.astype(np.float64)

    NUM_ITERATIONS = 100
    MAX_STEP_NORM = 1e-10
    REL_STEP_SIZE = 1e-6
    DOUBLE_EPS = np.finfo(np.float64).eps

    u0 = np.copy(u)
    v0 = np.copy(v)

    for _ in range(NUM_ITERATIONS):
        step0 = np.maximum(DOUBLE_EPS, np.abs(REL_STEP_SIZE * u))
        step1 = np.maximum(DOUBLE_EPS, np.abs(REL_STEP_SIZE * v))
        du, dv = distort(k, u, v)
        du_0b, dv_0b = distort(k, u - step0, v)
        du_0f, dv_0f = distort(k, u + step0, v)
        du_1b, dv_1b = distort(k, u, v - step1)
        du_1f, dv_1f = distort(k, u, v + step1)

        # fmt: off
        J = np.stack([
            np.stack([
                1 + (du_0f - du_0b) / (2 * step0),
                (du_1f - du_1b) / (2 * step1),
            ], axis=-1),
            np.stack([
                (dv_0f - dv_0b) / (2 * step0),
                1 + (dv_1f - dv_1b) / (2 * step1),
            ], axis=-1),
        ], axis=1)
        # fmt: on

        x = np.stack(
            [
                u + du - u0,
                v + dv - v0,
            ],
            axis=-1,
        )[..., None]
        step_u, step_v = np.linalg.solve(J, x).transpose(1, 0, 2).squeeze(-1)

        u -= step_u
        v -= step_v

        if np.max(step_u * step_u + step_v * step_v) < MAX_STEP_NORM:
            break

    u, v = u.astype(dtype), v.astype(dtype)
    return u, v


def points2d_to_bearing_vector(
    pts2d: np.ndarray, K: np.ndarray, vec_dim: int = 2, radial: Optional[float] = None
) -> np.ndarray:
    pts2d_homo = np.concatenate([pts2d, np.ones((len(pts2d), 1))], axis=-1)
    bvecs = np.linalg.solve(K, pts2d_homo.T)

    if radial is not None:
        bvecs[:2] = np.stack(undistort(radial, bvecs[0], bvecs[1]))
    bvecs = bvecs[:vec_dim].T
    return bvecs.astype(pts2d.dtype)


def align_2d3d_points_normalized(
    pts2d: np.ndarray,
    pts3d_proj: np.ndarray,
    K: np.ndarray,
    dist_thres: Optional[float],
    radial: Optional[float] = None,
) -> np.ndarray:
    # Convert pts2d to normalized coordinates
    pts2d_normed = points2d_to_bearing_vector(pts2d, K, vec_dim=2, radial=radial)
    pts3d_proj_normed = points2d_to_bearing_vector(
        pts3d_proj, K, vec_dim=2, radial=radial
    )
    aligned_ids = align_points2d(pts2d_normed, pts3d_proj_normed, dist_thres=dist_thres)
    return aligned_ids



def project_points3d(
    K: TensorOrArrayOrList,
    R: TensorOrArrayOrList,
    t: TensorOrArrayOrList,
    pts3d: TensorOrArrayOrList,
    radial: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D points using extrinsics and intrinsics
    Args:
        - K: camera intrinc matrix (3, 3)
        - R: world to camera rotation (3, 3)
        - t: world to camera translation (3,)
        - pts3d: 3D points (N, 3)
        - radial: single distortion coefficient. It presents the coefficient k1 in https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    Return:
        - pts2d: projected 2D points (N, 2)
        - valid: bool mask, indicates the 3d points that are projected in front of the camera
    """

    K = get_numpy(K)
    R = get_numpy(R)
    t = get_numpy(t)
    pts3d = get_numpy(pts3d)

    pts2d_norm, valid = project3d_normalized(
        R, t, pts3d, radial=radial, return_valid=True
    )
    assert isinstance(pts2d_norm, np.ndarray)
    assert isinstance(valid, np.ndarray)
    pts3d_norm = np.concatenate([pts2d_norm, np.ones((len(pts2d_norm), 1))], axis=1)

    # Transform to pixel space. Last column is already guaranteed to be set to 1
    pixels = pts3d_norm @ K.T
    return pixels[:, :2], valid

def get_numpy(data: TensorOrArrayOrList) -> np.ndarray:
    if isinstance(data, Tensor):
        out = data.cpu().data.numpy()
    elif isinstance(data, list):
        out = np.array(data)
    else:
        out = data

    assert isinstance(out, np.ndarray)
    return out



def project3d_normalized(
    R: TensorOrArray,
    t: TensorOrArray,
    pts3d: TensorOrArray,
    radial: Optional[float] = None,
    return_valid: bool = False,
) -> Union[TensorOrArray, Tuple[TensorOrArray, TensorOrArray]]:
    # Move points to camera space
    pts3d_cam = pts3d @ R.T + t

    # Bring it to the normalized image plane at z=1
    pts3d_norm = pts3d_cam / pts3d_cam[:, -1, None]

    # Distort if needed
    if radial is not None:
        assert isinstance(
            pts3d_norm, np.ndarray
        ), "Unable to apply radial distortion with torch.Tensor arguments"
        du, dv = distort(radial, pts3d_norm[:, 0], pts3d_norm[:, 1])
        pts3d_norm = pts3d_norm + np.stack([du, dv, np.zeros_like(dv)], axis=1)

    if return_valid:
        # only consider points in front of the camera
        valid = pts3d_cam[:, -1] >= 0
        return pts3d_norm[:, :2], valid
    else:
        return pts3d_norm[:, :2]
    

def distort(k: float, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """COLMAP - src/base/camera_models.h:747"""
    u2 = u * u
    v2 = v * v
    r2 = u2 + v2
    radial = k * r2
    du = radial * u
    dv = radial * v
    return du, dv


def compute_gt_2d3d_match(
    pts2d: np.ndarray,
    pts3d: TensorOrArrayOrList,
    K: np.ndarray,
    R: TensorOrArrayOrList,
    t: TensorOrArrayOrList,
    inls_thres: Optional[float] = 1,
    normalize: bool = False,
    radial: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Project 3d keypoints onto image plane
    pts2d_proj, valid = project_points3d(K, R, t, pts3d, radial=radial)

    # Align 2d points with 3d projections
    if normalize:
        # NN matching based on normalized distances
        matches = align_2d3d_points_normalized(
            pts2d, pts2d_proj[valid], K, dist_thres=inls_thres, radial=radial
        )
    else:
        # NN matching based on image pixel distances
        matches = align_points2d(pts2d, pts2d_proj[valid], dist_thres=inls_thres)
    matches = align_points2d(pts2d, pts2d_proj[valid], dist_thres=inls_thres)
    i3d_map = np.where(valid)[0]
    i2ds, i3ds = matches[:, 0], i3d_map[matches[:, 1]]

    # Everything not inliers as outliers
    n2d, n3d = len(pts2d), len(valid)
    inls_mask2d = np.zeros(n2d, dtype=bool)
    inls_mask2d[i2ds] = True
    inls_mask3d = np.zeros(n3d, dtype=bool)
    inls_mask3d[i3ds] = True
    o2ds = np.where(~inls_mask2d)[0]
    o3ds = np.where(~inls_mask3d)[0]
    return i2ds, i3ds, o2ds, o3ds


def compute_gt_2d3d_match(
    n2d: int,                 # 本 query 的 2D 点数量
    pts3d_pred: np.ndarray,   # (M,3) 在线 3D detector 预测点（世界系）
    xyz_idx: np.ndarray,      # (S,) 这些是“有 GT 的 2D 点”的 local 索引（指向该 query 的 pts2d）
    kptxyz: np.ndarray,       # (S,3) 对应的 GT 3D 世界坐标
    dist_th: float = 0.05,    # 3D 最近邻阈值（米，自己调）
    mutual: bool = True,      # mutual NN 更稳一点
):
    """
    输出与原 compute_gt_2d3d_match 类似的四类 index：
      i2ds, i3ds: inlier 对
      o2ds: 2D outliers（没被匹配到的 2D）
      o3ds: 3D outliers（没被匹配到的 3D）
    注意：
      - xyz_idx / i2ds 都是 “sample 内 local 2D 索引”
      - i3ds / o3ds 是 “pred 3D 索引”
    """
    xyz_idx = np.asarray(xyz_idx, dtype=np.int64)
    kptxyz  = np.asarray(kptxyz, dtype=np.float32)
    pts3d_pred = np.asarray(pts3d_pred, dtype=np.float32)

    M = int(pts3d_pred.shape[0])
    S = int(kptxyz.shape[0])

    # 边界情况
    if n2d <= 0:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int64), np.arange(M, dtype=np.int64)
    if M == 0:
        # 没有任何 3D pred：所有 2D 都是 outlier
        return np.array([], np.int64), np.array([], np.int64), np.arange(n2d, dtype=np.int64), np.array([], np.int64)
    if S == 0:
        # 没有 GT backproj：无法给 inlier，对所有 2D / 3D 判 outlier
        return np.array([], np.int64), np.array([], np.int64), np.arange(n2d, dtype=np.int64), np.arange(M, dtype=np.int64)

    # pairwise squared dist: (M,S)
    diff = pts3d_pred[:, None, :] - kptxyz[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)

    # 对每个 GT 3D（每一列）找最近 pred 3D
    nn_pred_for_gt = np.argmin(d2, axis=0)  # (S,)
    best_d2 = d2[nn_pred_for_gt, np.arange(S)]

    # mutual NN（可选）
    if mutual:
        nn_gt_for_pred = np.argmin(d2, axis=1)  # (M,)
        mutual_mask = (nn_gt_for_pred[nn_pred_for_gt] == np.arange(S))
    else:
        mutual_mask = np.ones((S,), dtype=bool)

    th2 = dist_th * dist_th
    ok = (best_d2 <= th2) & mutual_mask

    gt_ids = np.where(ok)[0]  # 通过阈值的 GT 条目
    i3ds = nn_pred_for_gt[gt_ids].astype(np.int64)   # pred 3D indices
    i2ds = xyz_idx[gt_ids].astype(np.int64)          # local 2D indices

    # 处理 collision：多个 GT 可能指向同一个 pred 3D
    # 这里简单策略：同一 pred 3D 只保留距离最小的那条
    if len(i3ds) > 1:
        # 为每条匹配取距离
        ds = best_d2[gt_ids]
        order = np.argsort(ds)  # 先小后大
        used3d = set()
        keep = []
        for k in order:
            j3d = int(i3ds[k])
            if j3d in used3d:
                continue
            used3d.add(j3d)
            keep.append(k)
        keep = np.array(keep, dtype=np.int64)
        i2ds = i2ds[keep]
        i3ds = i3ds[keep]

    # outliers
    matched_2d = set(i2ds.tolist())
    matched_3d = set(i3ds.tolist())

    o2ds = np.array([u for u in range(n2d) if u not in matched_2d], dtype=np.int64)
    o3ds = np.array([v for v in range(M)   if v not in matched_3d], dtype=np.int64)

    return i2ds, i3ds, o2ds, o3ds

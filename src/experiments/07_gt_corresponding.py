#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import h5py
import torch
from tqdm import tqdm
import glob
from scipy.spatial.ckdtree import cKDTree

from config.options import BaseOptions
from config.path import dataset_dir
from datasets.dataset import GSDataset
# from util.gs_tools import sample_view_based_rays
# from util.util import make_pose_rng
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ============================================================
# project like GoMatch + real coords filter
# ============================================================

def project3d_normalized(R, t, pts3d, return_valid=False):
    # Move points to camera space
    pts3d_cam = pts3d @ R.T + t

    # Bring it to the normalized image plane at z=1
    pts3d_norm = pts3d_cam / pts3d_cam[:, -1, None]

    if return_valid:
        # only consider points in front of the camera
        valid = pts3d_cam[:, -1] >= 0
        return pts3d_norm[:, :2], valid
    else:
        return pts3d_norm[:, :2]
    
def project_points3d(K, R, t, pts3d):
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
    pts2d_norm, valid = project3d_normalized(
        R, t, pts3d, return_valid=True
    )
    assert isinstance(pts2d_norm, np.ndarray)
    assert isinstance(valid, np.ndarray)
    pts3d_norm = np.concatenate([pts2d_norm, np.ones((len(pts2d_norm), 1))], axis=1)

    # Transform to pixel space. Last column is already guaranteed to be set to 1
    pixels = pts3d_norm @ K.T
    return pixels[:, :2], valid

def points2d_to_bearing_vector(pts2d, K, vec_dim):
    pts2d_homo = np.concatenate([pts2d, np.ones((len(pts2d), 1))], axis=-1)
    bvecs = np.linalg.solve(K, pts2d_homo.T)
    bvecs = bvecs[:vec_dim].T
    return bvecs.astype(pts2d.dtype)
def cal_mutual_nn_dists_kdtrees(nn12,nn21,dist12,threshold= None) :
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
    pts1, pts2, dist_thres):
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

def align_2d3d_points_normalized(
    pts2d,
    pts3d_proj,
    K,
    dist_thres,
):
    # Convert pts2d to normalized coordinates
    pts2d_normed = points2d_to_bearing_vector(pts2d, K, vec_dim=2)
    pts3d_proj_normed = points2d_to_bearing_vector(
        pts3d_proj, K, vec_dim=2)
    aligned_ids = align_points2d(pts2d_normed, pts3d_proj_normed, dist_thres=dist_thres)
    return aligned_ids

def debug_plot_3d_sampling(Xw_full, Xw_samp, C_world, title="3D sampling", max_bg=20000):
    """
    Xw_full: (N,3) full GS xyz in WORLD
    Xw_samp: (n,3) sampled xyz in WORLD
    C_world: (3,) camera center in WORLD
    """
    Xw_full = np.asarray(Xw_full).reshape(-1, 3)
    Xw_samp = np.asarray(Xw_samp).reshape(-1, 3)
    C_world = np.asarray(C_world).reshape(3,)

    # optional downsample background for speed
    if Xw_full.shape[0] > max_bg:
        step = max(1, Xw_full.shape[0] // max_bg)
        Xbg = Xw_full[::step]
    else:
        Xbg = Xw_full

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # background (light)
    ax.scatter(Xbg[:, 0], Xbg[:, 1], Xbg[:, 2], s=1, alpha=0.08)

    # sampled (highlight)
    ax.scatter(Xw_samp[:, 0], Xw_samp[:, 1], Xw_samp[:, 2], s=8, alpha=0.9)

    # camera center
    ax.scatter([C_world[0]], [C_world[1]], [C_world[2]], s=80, marker="^", alpha=1.0)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # keep equal-ish aspect (matplotlib 3D isn't perfect, but helps)
    all_pts = np.vstack([Xbg, Xw_samp, C_world[None, :]])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.show()

def project_world_to_image(Xw, K, R_wc, t_wc):
    """
    Xw: (N,3) in WORLD
    K: (3,3)
    R_wc: (3,3) world->cam
    t_wc: (3,)
    returns uv (N,2), z (N,)
    """
    Xw = np.asarray(Xw).reshape(-1, 3).astype(np.float32)
    K = np.asarray(K).astype(np.float32)
    R_wc = np.asarray(R_wc).astype(np.float32)
    t_wc = np.asarray(t_wc).reshape(3,).astype(np.float32)

    Xc = (R_wc @ Xw.T + t_wc.reshape(3, 1)).T
    z = Xc[:, 2]
    z_safe = z + 1e-6

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (Xc[:, 0] / z_safe) + cx
    v = fy * (Xc[:, 1] / z_safe) + cy
    uv = np.stack([u, v], axis=1)
    return uv, z


def debug_plot_2d_overlay(Xw_samp, pts2d, K, R_wc, t_wc, width=400, height=400,
                         title="sampled 3D -> 2D", show_pts2d=True):
    """
    Xw_samp: (n,3) sampled points in WORLD
    pts2d:   (M,2) 2D keypoints (optional overlay)
    """
    uv, z = project_world_to_image(Xw_samp, K, R_wc, t_wc)

    in_front = z > 1e-3
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    keep = in_front & in_frame

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # empty canvas (if you don't have the actual image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # invert y to match image coords
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nkeep={keep.sum()}/{len(keep)}")

    # sampled projections
    ax.scatter(uv[keep, 0], uv[keep, 1], s=10, alpha=0.9, label="sampled 3D proj")
    ax.scatter(uv[~keep, 0], uv[~keep, 1], s=10, alpha=0.15, label="out (behind/outside)")

    # optional overlay of 2D keypoints
    if show_pts2d and pts2d is not None:
        pts2d = np.asarray(pts2d).reshape(-1, 2)
        ax.scatter(pts2d[:, 0], pts2d[:, 1], s=8, alpha=0.35, label="pts2d")

    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def compute_gt_2d3d_match(
    pts2d, pts3d, # world
    xyz_idx, 
    kptxyz_world,
    K,R_c2w,C,
    th_3d_world=0.05,
    inls_thres=0.01,
):
    pts2d = np.asarray(pts2d, np.float32)
    pts3d = np.asarray(pts3d, np.float32)
    xyz_idx = np.asarray(xyz_idx, np.int64).reshape(-1)
    kptxyz_world = np.asarray(kptxyz_world, np.float32).reshape(-1, 3)
   
    
    # Project 3d keypoints onto image plane
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C    
    pts2d_proj, valid = project_points3d(K, R_w2c, t_w2c, pts3d)


    # 1. GoMatch style NN matching based on normalized distances
    matches = align_2d3d_points_normalized(
        pts2d, pts2d_proj[valid], K, dist_thres=inls_thres
    )

    i3d_map = np.where(valid)[0]
    i2ds_all, i3ds_all = matches[:, 0], i3d_map[matches[:, 1]]
    i2ds, i3ds = matches[:, 0], i3d_map[matches[:, 1]]

    
    # use xyz_idx for those having 3d points
    row_of = np.full((len(pts2d),), -1, dtype=np.int64)
    row_of[xyz_idx] = np.arange(len(xyz_idx), dtype=np.int64)
    # all pts2d has 3d point
    # row_of = np.arange(len(pts2d), dtype=np.int64)

    rows = row_of[i2ds_all]          # (num_matches,)
    has_xyz = rows >= 0              # only those 2D with kptxyz

    # If you want "no kptxyz => outlier", then just drop them here:
    i2ds = i2ds_all[has_xyz]
    i3ds = i3ds_all[has_xyz]
    rows = rows[has_xyz]

    # 3D distance check
    d3 = np.linalg.norm(pts3d[i3ds] - kptxyz_world[rows], axis=1)


    keep3 = d3 < float(th_3d_world)

    i2ds = i2ds[keep3]
    i3ds = i3ds[keep3]

    # Everything not inliers as outliers
    n2d, n3d = len(pts2d), len(valid)
    inls_mask2d = np.zeros(n2d, dtype=bool)
    inls_mask2d[i2ds] = True
    inls_mask3d = np.zeros(n3d, dtype=bool)
    inls_mask3d[i3ds] = True
    o2ds = np.where(~inls_mask2d)[0]
    o3ds = np.where(~inls_mask3d)[0]
    return i2ds, i3ds, o2ds, o3ds



# ============================================================
# read one XFeat query
# ============================================================
def read_xfeat_query(h5_path: str, qk: str):
    with h5py.File(h5_path, "r") as f:
        g = f["queries"][qk]
  
        # kpt from 2D
        pts2d = g["kptuv"][...].astype(np.float32)          # (N2,2)
        xyz_idx = g["xyz_idx"][...].astype(np.int64)        # (M,)
        kptxyz = g["kptxyz"][...].astype(np.float32)        # (M,3) world
        # query camera info
        C = g["position"][...].astype(np.float32) 
        R_c2w = g["rotation"][...].astype(np.float32) 
        K = g["intrinsics"][...].astype(np.float32)

      
    return pts2d,  kptxyz, xyz_idx, K, R_c2w, C


# ============================================================
# save scene GT
# ============================================================
def save_scene_gt_h5_qkwise(
    out_h5_path: str,
    scene: str,
    matches_by_qk: dict,      # {qk: matches_bin}
    pts3d_norm_by_qk: dict,   # {qk: (N3,3)}
    centroid_by_qk: dict,     # {qk: (3,)}
    m_by_qk: dict,            # {qk: float}
    overwrite: bool = False,
    compression: str = "gzip",
    compression_opts: int = 4,
):
    ensure_dir(os.path.dirname(out_h5_path))
    if (not overwrite) and os.path.exists(out_h5_path):
        return out_h5_path
    if os.path.exists(out_h5_path):
        os.remove(out_h5_path)

    with h5py.File(out_h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["scene"] = scene
        meta.attrs["num_queries"] = int(len(matches_by_qk))

        qroot = f.create_group("queries")

        for qk in sorted(matches_by_qk.keys()):
            mb = np.asarray(matches_by_qk[qk], dtype=np.uint8)  # 存 uint8 更稳
            pts3d_norm = np.asarray(pts3d_norm_by_qk[qk], np.float32).reshape(-1, 3)
            centroid = np.asarray(centroid_by_qk[qk], np.float32).reshape(3,)
            m = float(m_by_qk[qk])

            n3d_plus1, n2d_plus1 = mb.shape
            n3d = n3d_plus1 - 1
            n2d = n2d_plus1 - 1

            if n3d != pts3d_norm.shape[0]:
                raise ValueError(f"[{scene} {qk}] matches_bin n3d={n3d} != pts3d_norm n3d={pts3d_norm.shape[0]}")

            g = qroot.create_group(qk)
            g.create_dataset("pts3d_norm", data=pts3d_norm,
                             compression=compression, compression_opts=compression_opts)
            g.create_dataset("centroid", data=centroid)
            g.create_dataset("m", data=np.array(m, dtype=np.float32))
            g.create_dataset("matches_bin", data=mb,
                             compression=compression, compression_opts=compression_opts)

            g.attrs["n2d"] = int(n2d)
            g.attrs["n3d"] = int(n3d)
            g.attrs["num_matches"] = int(mb[:n3d, :n2d].sum())

    return out_h5_path

# ============================================================
# GEN: build offline GT
# ============================================================
def build_offline_gt(args, opt, debug =False):
    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu")
    print('loading dataset...')
    gs_ds = GSDataset(
        opt=opt,                 # 用你的 BaseOptions（含 seed/sampling_rule/npoints 等）
        gs_root=args.gs_root,
        split=args.split,
        return_params=True,
    )
    
    print(f"Total scenes in dataset: {len(gs_ds)}")

    th_3d_world = float(args.th_3d)
    matched_counts = []
    
    pbar = tqdm(gs_ds.samples, desc=f"Offline GT ({args.split})", unit="scene")
    for pack in pbar:
        scene = pack["scene"]
        gs_path = pack["path"]
        cat = os.path.basename(os.path.dirname(gs_path))
        

        xfeat_h5 = os.path.join(args.xfeat_root, cat, f"{scene}.h5")
        if not os.path.exists(xfeat_h5):
            raise FileNotFoundError(f"XFeat h5 not found: {xfeat_h5}")

        with h5py.File(xfeat_h5, "r") as f:
            qkeys = sorted(list(f["queries"].keys()))
            
        matches_by_qk = {}
        pts3d_norm_qk = {}
        centroid_qk = {}
        m_qk = {}
        
        for qk in qkeys:

            pts2d, kptxyz_world, xyz_idx, K, R_c2w, C = read_xfeat_query(xfeat_h5, qk)      
            R_wc = R_c2w.T
            t_wc = (-R_wc @ C).astype(np.float32)
            

                                       
            # if 'train' in args.split:
                # gs_full = pack["gs_full"].astype(np.float32) # (16384,C)
                # rng_pose = make_pose_rng(getattr(opt, "seed", 123), scene, qk)

                # idx_sparse = sample_view_based_rays(
                #     Xw_full=gs_full[:, :3],
                #     K=K,
                #     R_c2w=R_c2w,
                #     C_world=C,
                #     n=1024,
                #     rng=rng_pose,
                #     width=400,
                #     height=400,
                #     max_candidates=None,  # 最稳定（候选用全部可见点）
                #     topk=1                # 先用 1；如果重复/不稳定再试 3~5
                # )
                # gs_sparse = gs_full[idx_sparse]
                # gs_sparse, params = gs_ds.pc_norm_gs(gs_sparse, attribute=gs_ds.attribute, return_params=True)
                # gs_sparse = gs_sparse.astype(np.float32) # (N,C)   



            # else:    
            gs_sparse = pack["gs_sparse"].astype(np.float32)
            params = pack.get("params", None)
            
            
            
            pts3d_norm_np = gs_sparse[:, :3].astype(np.float32)
            n3d = int(pts3d_norm_np.shape[0])

        
            if params is None:
                raise RuntimeError(f"Missing params for scene={scene}. Need return_params=True in GSDataset.")

            centroid = np.asarray(params["centroid"], dtype=np.float32).reshape(3,)
            m = float(params["m"])

            pts3d = pts3d_norm_np*(m + 1e-12) + centroid
                
                    
            i2ds, i3ds, o2ds, o3ds = compute_gt_2d3d_match(
                                        pts2d,pts3d, xyz_idx,
                                        kptxyz_world,
                                        K,R_c2w,C, 
                                        th_3d_world=th_3d_world,  
                                        inls_thres=args.inls2d_thres)
            n2d, n3d = len(pts2d), len(pts3d)
            matches_bin = np.zeros((n3d + 1, n2d + 1), dtype=bool)
            matches_bin[i3ds, i2ds] = True
            matches_bin[o3ds, -1] = True
            matches_bin[-1, o2ds] = True
            
            num_matches = int(matches_bin[:-1, :-1].sum())
            matched_counts.append(num_matches)
            
            matches_by_qk[qk] = matches_bin
            pts3d_norm_qk[qk] = pts3d_norm_np
            centroid_qk[qk] = centroid
            m_qk[qk] = m
            if debug:
                
                # pts3d_norm_np2 = gs_random[:, :3].astype(np.float32)
                # n3d2 = int(pts3d_norm_np2.shape[0])    
                # centroid2 = np.asarray(params_random["centroid"], dtype=np.float32).reshape(3,)
                # m2 = float(params_random["m"])

                # pts3d2 = pts3d_norm_np2*(m2 + 1e-12) + centroid2   
                # i2ds2, i3ds2, o2ds2, o3ds2 = compute_gt_2d3d_match(
                #                             pts2d,pts3d2,
                #                             kptxyz_world,
                #                             K,R_c2w,C, 
                #                             th_3d_world=th_3d_world,  
                #                             inls_thres=args.inls2d_thres)
                # n2d, n3d2 = len(pts2d), len(pts3d2)
                # matches_bin2 = np.zeros((n3d2 + 1, n2d + 1), dtype=bool)
                # matches_bin2[i3ds2, i2ds2] = True
                # matches_bin2[o3ds2, -1] = True
                # matches_bin2[-1, o2ds2] = True                             

                # debug_plot_2d_overlay(gs_full[idx_sparse, :3], pts2d, K, R_wc, t_wc, width=400, height=400,
                #     title=f"Pose Sampled: {qk} 3D -> 2D")
                # print(gs_random.shape)
                # debug_plot_2d_overlay(pts3d2[:, :3], pts2d, K, R_wc, t_wc, width=400, height=400,
                #                     title=f"Random Sampled: {qk} 3D -> 2D")
                plot_gt_matches_3d(pts3d, kptxyz_world, matches_bin, xyz_idx, cam_center_world=C)
                # plot_gt_matches_3d(pts3d2, kptxyz_world, matches_bin2, cam_center_world=C)

        out_h5 = os.path.join(args.out_root, cat, f"{scene}.h5")
        save_scene_gt_h5_qkwise(
            out_h5_path=out_h5,
            scene=scene,
            matches_by_qk=matches_by_qk,
            pts3d_norm_by_qk=pts3d_norm_qk,
            centroid_by_qk=centroid_qk,
            m_by_qk=m_qk,
            overwrite=args.overwrite
        )
    # ------------------------------------------------------------
    # ✅ Draw distribution graph (histogram)
    # ------------------------------------------------------------
    matched_counts = np.asarray(matched_counts, dtype=np.int32)

    plot_match_rank(matched_counts, split= args.split)

    # optional: also print summary
    print("[Matched stats]",
          "N =", len(matched_counts),
          "mean =", float(matched_counts.mean()),
          "median =", float(np.median(matched_counts)),
          "min =", int(matched_counts.min()) if len(matched_counts) else None,
          "max =", int(matched_counts.max()) if len(matched_counts) else None)




    print(f"[DONE] GT saved under: {args.out_root}")

# ============================================================
# CHECK: validate & stats offline GT
# ============================================================
def plot_match_rank(matched_counts, split, show=False, save_dir=None):

    y = np.sort(np.asarray(matched_counts, dtype=np.int32))
    x = np.arange(len(y))

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=1.2)
    ax.set_xlabel("Sample rank (sorted)")
    ax.set_ylabel("Number of matched correspondences (inliers)")

    # 主标题
    ax.set_title(
        f"Matched number distribution (sorted) — split {split}",
        fontsize=12
    )

    ax.grid(True, alpha=0.25)

    if not show:
        if save_dir is None:
            save_dir = os.getcwd()   # ✅ current place
        save_path = os.path.join(
            save_dir,
            f"matched_number_distribution_split_{split}.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_gt_matches_3d(
    pts3d_world,          # (N3,3) map points (world)
    kptxyz_world,         # (M,3) keypoint 3D (world), corresponds to xyz_idx
    matches_bin,          # (N3+1, N2+1) bool, your format
    xyz_idx,              # (M,) mapping: kptxyz_world[m] <-> 2D index xyz_idx[m]
    cam_center_world=None,# (3,) camera center in world (e.g., t_c2w)
    max_map_points=6000,  # subsample map points for clarity
    show_lines=True,
    show=True,
):
    pts3d_world = np.asarray(pts3d_world, np.float32).reshape(-1, 3)
    kptxyz_world = np.asarray(kptxyz_world, np.float32).reshape(-1, 3)
    matches_bin = np.asarray(matches_bin, bool)
    xyz_idx = np.asarray(xyz_idx, np.int64).reshape(-1)

    num_3d = pts3d_world.shape[0]
    num_2d = matches_bin.shape[1] - 1  # because last col is 3D-outlier bucket

    # ---------------------------
    # 1) Extract inlier matches (3D index, 2D index)
    #    Only use the real-real block: [0:num_3d, 0:num_2d]
    # ---------------------------
    matched_3d_idx, matched_2d_idx = np.nonzero(matches_bin[:num_3d, :num_2d])

    # ---------------------------
    # 2) Build map: 2D index -> kptxyz row
    #    because kptxyz_world[m] corresponds to 2D index xyz_idx[m]
    # ---------------------------
    # use xyz_idx for those having 3d points
    kpt_row_for_2d = np.full((num_2d,), -1, dtype=np.int64)
    kpt_row_for_2d[xyz_idx] = np.arange(len(xyz_idx), dtype=np.int64)
        
    # # all pts2d has 3d point
    # kpt_row_for_2d = np.arange(num_2d, dtype=np.int64)
    
    

    matched_kpt_row = kpt_row_for_2d[matched_2d_idx]

    # keep only matches where the 2D index actually has a kptxyz
    keep = (matched_kpt_row >= 0) & (matched_kpt_row < len(kptxyz_world))
    matched_3d_idx = matched_3d_idx[keep]
    matched_kpt_row = matched_kpt_row[keep]

    matched_map_pts = pts3d_world[matched_3d_idx]         # (K,3)
    matched_kpt_pts = kptxyz_world[matched_kpt_row]       # (K,3)
    num_inliers = matched_map_pts.shape[0]

    # ---------------------------
    # 3) Subsample map points for readability
    # ---------------------------
    if len(pts3d_world) > max_map_points:
        sample_idx = np.random.choice(len(pts3d_world), max_map_points, replace=False)
        map_vis = pts3d_world[sample_idx]
    else:
        map_vis = pts3d_world

    # ---------------------------
    # 4) Plot
    # ---------------------------
    
    # ---- ggplot2 colors ----
    COLOR_KPTXYZ = "#00BFC4"   # teal
    COLOR_MATCH3D = "#F8766D"  # salmon
    COLOR_MAP = "#BDBDBD"      # light gray
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # # remove background panes
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    # # remove pane edges
    # ax.xaxis.pane.set_edgecolor('w')
    # ax.yaxis.pane.set_edgecolor('w')
    # ax.zaxis.pane.set_edgecolor('w')
    # ax.grid(False)
    # fig.patch.set_alpha(0)
    # ax.set_facecolor((0,0,0,0))

    # Map point cloud: small + transparent
    ax.scatter(map_vis[:, 0], map_vis[:, 1], map_vis[:, 2],
               s=1.5, alpha=0.5,color ='gray', label="map pts3d")

    # kptxyz: smaller points (as you requested)
    ax.scatter(kptxyz_world[:, 0], kptxyz_world[:, 1], kptxyz_world[:, 2],
               s=2, alpha=0.5, color=COLOR_KPTXYZ,label="kptxyz (3D from 2D)")

    # matched map points: slightly larger to stand out
    ax.scatter(matched_map_pts[:, 0], matched_map_pts[:, 1], matched_map_pts[:, 2],
               s=5, alpha=1.0,color=COLOR_MATCH3D, label="matched map pts3d")

    # match lines
    if show_lines:
        for a, b in zip(matched_kpt_pts, matched_map_pts):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=2, color='black',alpha=0.9)

    # camera center
    if cam_center_world is not None:
        cam_center_world = np.asarray(cam_center_world, np.float32).reshape(3,)
        ax.scatter([cam_center_world[0]], [cam_center_world[1]], [cam_center_world[2]],
                   s=90, marker="x", linewidths=2.5, label="camera center")

        # optional: also draw rays from camera to matched kptxyz (often helps intuition)
        # for p in matched_kpt_pts:
        #     ax.plot([cam_center_world[0], p[0]],
        #             [cam_center_world[1], p[1]],
        #             [cam_center_world[2], p[2]], linewidth=0.6, alpha=0.4)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"GT matches (inliers = {num_inliers})")
    ax.legend()

    _set_axes_equal(ax)
    # ax.set_axis_off()

    # plt.savefig("plot.png", dpi=300, transparent=True)
    if show:
        plt.show()
    
    return fig, ax


def check_from_saved_gt(gt_root, split, threshold=5):
    """Check GT matches from saved H5 files and print queries with very few matches."""
    with open(f'/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/cfg/split/{split}.txt', "r") as f:
        split_set = set()
        for line in f:
            s = line.strip()
            if not s:
                continue
            base = os.path.splitext(os.path.basename(s))[0]
            split_set.add(base)
    
    print(f"[INFO] Scenes in split {split}: {len(split_set)}")

    h5_files = sorted(glob.glob(os.path.join(gt_root, "**", "*.h5"), recursive=True))
    if not h5_files:
        raise RuntimeError(f"No GT files found under {gt_root}")

    few_match_queries = []

    for h5_path in tqdm(h5_files, desc="Reading GT", unit="file"):
        stem = os.path.splitext(os.path.basename(h5_path))[0]
        if stem not in split_set:
            continue        

        with h5py.File(h5_path, "r") as f:
            if "queries" not in f:
                continue
            for qk in f["queries"].keys():
                num = f["queries"][qk].attrs.get("num_matches", None)
                if num is not None and int(num) < threshold:
                    few_match_queries.append((stem, qk, int(num)))

    if few_match_queries:
        print(f"[WARNING] Queries with less than {threshold} matches:")
        for scene, qk, num in few_match_queries:
            print(f"  Scene={scene}, Query={qk}, num_matches={num}")
    else:
        print(f"[INFO] All queries have >= {threshold} matches.")

def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="scene")
    p.add_argument("--dataset", type=str, default="shapesplat", choices=["shapesplat", "scared", "stereomis"])
    p.add_argument("--sampling_rule", type=str, default="random")
    p.add_argument("--th_3d", type=float, default=0.05, help="world threshold ")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument('--inls2d_thres', type=float, default=0.004)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--check",
        action="store_true",
        help="Fast check: read saved GT files and plot match distribution"
    )
    return p.parse_args()

def plot_from_saved_gt(gt_root, split):
    matched_counts = []
    with open(f'/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/cfg/split/{split}.txt', "r") as f:
        split_set = set()
        for line in f:
            s = line.strip()
            if not s:
                continue
            base = os.path.splitext(os.path.basename(s))[0]  # 去掉 .ply，并且只保留文件名
            split_set.add(base)
    print(split_set)
    h5_files = sorted(glob.glob(os.path.join(gt_root, "**", "*.h5"), recursive=True))
    if not h5_files:
        raise RuntimeError(f"No GT files found under {gt_root}")

    for h5_path in tqdm(h5_files, desc="Reading GT", unit="file"):
        
        stem = os.path.splitext(os.path.basename(h5_path))[0]

        if stem not in split_set:
            continue        
        
        
        with h5py.File(h5_path, "r") as f:
            if "queries" not in f:
                continue
            for qk in f["queries"].keys():
                num = f["queries"][qk].attrs.get("num_matches", None)
                if num is not None:
                    matched_counts.append(int(num))

    matched_counts = np.asarray(matched_counts, dtype=np.int32)
    plot_match_rank(matched_counts, split=split,show=True)

    print("[GT stats]",
          "N =", len(matched_counts),
          "mean =", float(matched_counts.mean()),
          "median =", float(np.median(matched_counts)),
          "min =", int(matched_counts.min()) if len(matched_counts) else None,
          "max =", int(matched_counts.max()) if len(matched_counts) else None)

def main(opt,args):

    # your directory convention
    if args.dataset == 'shapesplat':
        args.gs_root = str(dataset_dir("gs_root"))
        args.xfeat_root = os.path.join(args.gs_root, "XFeat_upgrade")
        # args.xfeat_root = os.path.join(args.gs_root, "XFeat_kpt")
        th_tag = f"{int(round(args.inls2d_thres * 1000)):04d}"
        args.out_root = os.path.join(args.gs_root, f"match_new{th_tag}")
        # args.out_root = os.path.join(args.gs_root, f"match_thresh{th_tag}")

    elif args.dataset == 'scared':
        root = str(dataset_dir("gs_root_scared"))
        args.gs_root = os.path.join(root, "scared_splat")
        args.xfeat_root = os.path.join(root, "scared_XFeat")
        th_tag = f"{int(round(args.inls2d_thres * 1000)):04d}"
        args.out_root = os.path.join(root, f"scared_match_thresh{th_tag}")  


    if args.check:
        print("[FAST CHECK] Reading saved GT only")
        # plot_from_saved_gt(args.out_root, args.split)
        check_from_saved_gt(args.out_root, args.split)
        return

    ensure_dir(args.out_root)
    build_offline_gt(args, opt, debug=args.debug)



if __name__ == "__main__":
    import sys

    # 1) 解析“脚本自己的参数”（真实命令行）
    args = parse_args()

    # 2) BaseOptions：不吃任何 CLI，只用默认/配置
    old_argv = sys.argv
    sys.argv = [old_argv[0]]   # 只保留程序名
    opt = BaseOptions().parse()
    sys.argv = old_argv

    # 3) 正常用 opt（只读）+ args（控制脚本）
    main(opt, args)
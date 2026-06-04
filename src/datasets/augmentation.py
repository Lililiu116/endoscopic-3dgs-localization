import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.transform import Rotation as Rscipy

# Liwen modify
def transform_gs_np(gs, rot_type='3d', scale_thre=0.2, shift_thre=0.2, rot_perturbation=False):
    '''

    :param gs: NxC tensor: position(3)+ opacity(1) + scale (3) + rotation (4) + color (3)
    :return: gs of the same shape, detach
    '''
    assert isinstance(gs, np.ndarray), "Input gs should be a numpy array."
    device = torch.device('cpu')
    gs_tensor = torch.from_numpy(gs).to(device).float()

    # N, M = pc.size()[1], node.size()[1]

    # 1. rotate around the up axis
    if rot_type == '2d':
        x_angle, z_angle = 0, 0
        y_angle = np.random.uniform() * 2 * np.pi
    elif rot_type == '3d':
        x_angle = np.random.uniform() * 2 * np.pi
        y_angle = np.random.uniform() * 2 * np.pi
        z_angle = np.random.uniform() * 2 * np.pi
    elif rot_type is None:
        x_angle, y_angle, z_angle = 0, 0, 0
    else:
        raise Exception('Invalid rot_type.')

    if rot_perturbation == True:
        angle_sigma = 0.06
        angle_clip = 3 * angle_sigma
        x_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        y_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        z_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)

    angles = [x_angle, y_angle, z_angle]
    R = torch.from_numpy(angles2rotation_matrix(angles).astype(np.float32)).to(device)  # 3x3
    scale = np.random.uniform(low=1-scale_thre, high=1+scale_thre)
    shift = torch.from_numpy(np.random.uniform(-1*shift_thre, shift_thre, (3, 1)).astype(np.float32)).to(device)  # 3x1
    
    transformed = transform_gs_tensor(R, scale, shift, gs_tensor)
    
    # gaussian.rescale(scale)
    # gaussian.rotate_by_matrix(R)
    # gaussian.translation(tx, ty, tz)


    return transformed.cpu().numpy(), \
           R, scale, shift


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec



def transform_gs_tensor(R: torch.Tensor, scale: float, shift: torch.Tensor, gs_tensor: torch.Tensor):
    """
    Apply similarity transform (R, scale, shift) to Gaussian splatting parameters.

    Args:
        R:      torch.Tensor (3,3), rotation matrix
        scale:  float or scalar tensor, global scale factor
        shift:  torch.Tensor (3,1), translation vector
        gs_tensor: torch.Tensor (N,C), Gaussian parameters
                   [:3]   -> xyz
                   [4:7]  -> scales (linear domain)
                   [7:11] -> quaternions (wxyz)

    Returns:
        transformed_gs_tensor: torch.Tensor (N,C)
    """
    device = gs_tensor.device
    s = torch.as_tensor(scale, dtype=torch.float32, device=device)

    # --- xyz ---
    xyz = gs_tensor[:, :3].T   # (3,N)
    # xyz = s * (R @ xyz) + shift   # (3,N)
    xyz = s * torch.matmul(R, xyz) + shift 

    # --- scales (linear domain) ---
    scale_dst = gs_tensor[:, 4:7].T   # (3,N)
    scale_dst = s * scale_dst

    # --- quaternions ---
    q = gs_tensor[:, 7:11]   # (N,4)

    # convert R -> quaternion (wxyz)

    qR_np = rotmat2qvec(R.detach().cpu().numpy()).astype(np.float32)  # (4,)
    qR = torch.from_numpy(qR_np).to(device).unsqueeze(0).expand(q.shape[0], -1)  # (N,4)

    # quaternion multiplication q ⊗ qR
    w0, x0, y0, z0 = torch.split(q, 1, dim=1)
    w1, x1, y1, z1 = torch.split(qR, 1, dim=1)
    q_new = torch.cat((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=1)
    q_new = torch.nn.functional.normalize(q_new, dim=1)

    # --- assemble ---
    transformed = gs_tensor.clone()
    transformed[:, :3]   = xyz.T
    # transformed[:, 4:7]  = scale_dst.T
    # transformed[:, 7:11] = q_new

    return transformed


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def euler_zyx_R_np(rx, ry, rz):
    """input: euler angles; return: rotation matrix"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def transform_gs_numpy(gs_np: np.ndarray, angles_3d, scale, shift) -> np.ndarray:
    """
    NumPy 输入/输出；内部复用已验证的 transform_gs_tensor。
    gs_np: (N,C) with [:3]=xyz, [4:7]=scales, [7:11]=quat(wxyz)
    angles_3d: (rx, ry, rz) 弧度，顺序 z-y-x (对应 R=Rz@Ry@Rx)
    """
    rx, ry, rz = angles_3d
    R_np = euler_zyx_R_np(rx, ry, rz)                      # (3,3)
    shift_np = np.asarray(shift, dtype=np.float32).reshape(3,1)

    # numpy -> torch
    R_t   = torch.from_numpy(R_np)
    t_t   = torch.from_numpy(shift_np)
    gs_t  = torch.from_numpy(gs_np.astype(np.float32, copy=False))

    out_t = transform_gs_tensor(R_t, float(scale), t_t, gs_t)
    return out_t.detach().cpu().numpy()







def atomic_rotate(data, angles):
    '''

    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data

def rotate_point_cloud_list_3d(pc_list, angles=None):
    if angles is None:
        # uniform sampling
        angles = np.random.rand(3) * np.pi * 2

    rotated_pc_list = []
    for pc in pc_list:
        rotated_pc_list.append(atomic_rotate(pc, angles))
    return rotated_pc_list




def random_scale_point_cloud(gs_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = gs_data.shape
    scale = np.random.uniform(scale_low, scale_high)
    gs_data[:,0:3] *= scale
    gs_data[:,4:7] *= scale
    print('scale:', scale)

    return gs_data


def random_shift_point_cloud(gs_data, shift_range=0.05, return_shifts=False):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, shifted batch of point clouds
    """
    N, C = gs_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (3,))
    gs_data[:,0:3] += shifts[None,:]
    print('shifts:', shifts)
    if return_shifts:
        return gs_data, shifts
    else:
        return gs_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

if __name__ == "__main__":
    import torch, numpy as np

    N = 5
    gs_tensor = torch.zeros((N, 14), dtype=torch.float32)

    # xyz (0~1)
    gs_tensor[:, :3] = torch.rand((N, 3))

    # scales (linear domain, >0)
    gs_tensor[:, 4:7] = torch.rand((N, 3)) + 0.5

    # 随机四元数 (wxyz)
    rand_quat = torch.randn((N, 4))
    rand_quat = rand_quat / rand_quat.norm(dim=1, keepdim=True)
    gs_tensor[:, 7:11] = rand_quat

    # -------------------
    # 构造变换 (R, scale, shift)
    # -------------------
    theta = np.pi / 4  # 45 deg
    R = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]], dtype=torch.float32)

    scale = 1.5
    shift = torch.tensor([[0.1],[0.2],[0.3]], dtype=torch.float32)

    # -------------------
    # 应用变换
    # -------------------
    transformed = transform_gs_tensor(R, scale, shift, gs_tensor)

    # -------------------
    # 自动检查
    # -------------------
    ok = True

    # 1. quaternion norm 是否≈1
    if not torch.allclose(transformed[:, 7:11].norm(dim=1), torch.ones(N), atol=1e-4):
        print("Quaternion not normalized!")
        ok = False

    # 2. scales 是否≈原来的 * scale
    ratio = transformed[:, 4:7] / gs_tensor[:, 4:7]
    if not torch.allclose(ratio, torch.full_like(ratio, scale), atol=1e-4):
        print("Scales not scaled correctly!")
        ok = False

    if ok:
        print("✅ Transform is correct.")
    else:
        print("❌ Transform check failed.")
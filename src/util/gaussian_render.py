import time
import math
import torch
import numpy as np
import plyfile
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer

class GaussianModel:
    def __init__(self):
        self.means3D = None
        self.means2D = None
        self.opacities = None
        self.rotations = None
        self.scales = None
        self.colors_precomp = None

    def load(self, path):
        plydata = plyfile.PlyData.read(path)
        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),
            axis=1,
        )
        opacity = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[:, np.newaxis]
        rot = np.stack(
            (
                np.asarray(plydata.elements[0]["rot_0"]),
                np.asarray(plydata.elements[0]["rot_1"]),
                np.asarray(plydata.elements[0]["rot_2"]),
                np.asarray(plydata.elements[0]["rot_3"]),
            ),
            axis=1,
        )
        scale = np.stack(
            (
                np.asarray(plydata.elements[0]["scale_0"]),
                np.asarray(plydata.elements[0]["scale_1"]),
                np.asarray(plydata.elements[0]["scale_2"]),
            ),
            axis=1,
        )
        shs = np.stack(
            (
                np.asarray(plydata.elements[0]["f_dc_0"]),
                np.asarray(plydata.elements[0]["f_dc_1"]),
                np.asarray(plydata.elements[0]["f_dc_2"]),
            ),
            axis=1,
        )
        colors = np.clip(0.3 * shs + 0.5, 0.0, 1.0)

        self.means3D = torch.nn.Parameter(torch.from_numpy(xyz).float().cuda())
        self.means2D = torch.zeros_like(self.means3D, dtype=self.means3D.dtype, device="cuda")
        self.opacities = torch.nn.functional.sigmoid(torch.from_numpy(opacity).float().cuda())
        self.rotations = torch.nn.functional.normalize(torch.from_numpy(rot).float().cuda())
        self.scales = torch.exp(torch.from_numpy(scale).float().cuda())
        self.colors_precomp = torch.from_numpy(colors[:, :, np.newaxis]).float().cuda()

        return self
    
class Camera:
    def __init__(self):
        self.width = None
        self.height = None
        self.image_width = None
        self.image_height = None
        self.fovX = None
        self.fovY = None
        self.transformation_matrix = None
        self.projection_matrix = None
        self.full_proj_transform = None
        self.camera_center = None
        self.cam_info = None

    def load(self, cam_info):
        self.cam_info = cam_info.copy()
        position = np.array(cam_info["position"])
        rotation = np.array(cam_info["rotation"])
        fx = cam_info["fx"]
        fy = cam_info["fy"]
        cx = cam_info["cx"]
        cy = cam_info["cy"]
        self.width = cam_info["width"]
        self.height = cam_info["height"]
        self.znear = cam_info["znear"] if "znear" in cam_info else 0.01
        self.zfar  = cam_info["zfar"]  if "zfar"  in cam_info else 100.0
            
            
        self.image_width = self.width 
        self.image_height = self.height
        self.fovX = 2 * math.atan(self.width / (2 * fx))
        self.fovY = 2 * math.atan(self.height / (2 * fy))
        self.tanfovX = math.tan(self.fovX / 2)
        self.tanfovY = math.tan(self.fovY / 2)
        self.bg = torch.zeros(3).float().cuda()
        transformation_matrix = get_transformation_matrix(position, rotation)
        projection_matrix = get_projection_matrix(
            fx, fy, cx, cy, self.width, self.height, self.znear, self.zfar
        )
        self.transformation_matrix = torch.from_numpy(transformation_matrix).float().cuda()
        self.projection_matrix = torch.from_numpy(projection_matrix).float().cuda()
        self.full_proj_transform = torch.from_numpy(transformation_matrix @ projection_matrix).float().cuda()
        self.camera_center = torch.from_numpy(position).float().cuda()
        return self

    def update(self, position, rotation):
        self.cam_info["position"] = position
        self.cam_info["rotation"] = rotation
        return self.load(self.cam_info)
    

class Renderer:
    def __init__(self, gaussian_model: GaussianModel, camera: Camera, logging: bool = True):
        self.gaussian_model = gaussian_model
        self.camera = camera
        self.logging = logging
        
    def render(self):
        start_time = time.time()
        raster_settings = GaussianRasterizationSettings(
            image_height=self.camera.image_height,
            image_width=self.camera.image_width,
            tanfovx=self.camera.tanfovX,
            tanfovy=self.camera.tanfovY,
            bg=self.camera.bg,
            scale_modifier=1.0,
            viewmatrix=self.camera.transformation_matrix,
            projmatrix=self.camera.full_proj_transform,
            sh_degree=0,
            campos=self.camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        if self.logging:
            print(f"Raster settings time: {time.time() - start_time}")
        start_time = time.time()
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        if self.logging:
            print(f"Rasterizer instantiation time: {time.time() - start_time}")
        start_time = time.time()
        rendered_image, radii, depth_map, weight_map = rasterizer(
            means3D=self.gaussian_model.means3D,
            means2D=self.gaussian_model.means2D,
            scales=self.gaussian_model.scales,
            rotations=self.gaussian_model.rotations,
            colors_precomp=self.gaussian_model.colors_precomp,
            opacities=self.gaussian_model.opacities,
            shs=None,
            cov3D_precomp=None,
        )
        if self.logging:
            print(f"Render time: {time.time() - start_time}")
        start_time = time.time()
        torch.cuda.synchronize()
        if self.logging:
            print(f"Sync time: {time.time() - start_time}")
        start_time = time.time()
        return rendered_image, radii, depth_map, weight_map
        # im = rendered_image.clamp(0.0, 1.0).multiply(255).reshape(-1).type(dtype=torch.cuda.ByteTensor)
        # if self.logging:
        #     print(f"Image render time: {time.time() - start_time}")
        # return im
    
    def update(self, position, rotation):
        self.camera.update(position, rotation)    
        
def get_transformation_matrix(position, rotation):
    '''position and rotation expects c2w'''
    T = np.eye(4, 4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return np.linalg.inv(T.T)


def get_projection_matrix(fx, fy, cx, cy, width, height, znear=0.01, zfar=100.0):
    left   = -(cx) * znear / fx
    right  = (width - cx) * znear / fx
    top    = (cy) * znear / fy
    bottom = -(height - cy) * znear / fy
    P = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P.T
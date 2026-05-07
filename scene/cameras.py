#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getProjectionMatrixCenterShift,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_slerp,
)
from kornia import create_meshgrid
from copy import copy, deepcopy

class Camera:
    ray_cache = {}
    # cache of projection matrices in cuda
    # so we don't have to copy per camera
    # when we have multiple frames from the same camera
    projection_cache = {}

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", timestamp = 0.0,
                 cx=-1, cy=-1, fl_x=-1, fl_y=-1, depth=None, resolution=None, image_path=None, meta_only=False,
                 ):

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.cx = cx
        self.cy = cy
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.resolution = resolution
        self.image_path = image_path
        self.image = image
        self.gt_alpha_mask = gt_alpha_mask
        self.meta_only = meta_only
        self.trans = trans
        self.scale = scale
        self.cached_cuda = None

        self.generate_cache_key()
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        if not self.meta_only:
            if gt_alpha_mask is not None:
                self.image *= gt_alpha_mask.to(self.image.device)
            else:
                self.image *= torch.ones((1, self.image_height, self.image_width), device=self.image.device)

        self.zfar = 100.0
        self.znear = 0.01

        self.depth = depth

        self.update_projection()
        self.timestamp = timestamp
        # fix the size of the cache before compile is run
        # and then it will not moan about the size changing after compile
        Camera.ray_cache[self.cache_key] = None
        Camera.ray_cache[(*self.cache_key,"cuda")] = None

    def generate_cache_key(self):
        self.cache_key = tuple(self.R.flatten().tolist() + self.T.flatten().tolist() + [self.FoVx, self.FoVy] + [self.cx, self.cy, self.fl_x, self.fl_y])

    def update_projection(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        if self.cx > 0:
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, self.cx, self.cy, self.fl_x, self.fl_y, self.image_width, self.image_height).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.generate_cache_key()


    def lerp_transform(self,camera_source,camera_target, lerp_factor):
        #self.R
        self.T = camera_source.T * (1-lerp_factor) + camera_target.T * lerp_factor
        source_quaternion = rotation_matrix_to_quaternion(camera_source.R)
        target_quaternion = rotation_matrix_to_quaternion(camera_target.R)
        self.R = quaternion_to_rotation_matrix(
            quaternion_slerp(source_quaternion, target_quaternion, lerp_factor)
        )
        self.update_projection()
        
    def get_rays(self):
        if Camera.ray_cache[self.cache_key] is not None:
            return Camera.ray_cache[self.cache_key]
        grid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False)[0] + 0.5
        i, j = grid.unbind(-1)
        pts_view = torch.stack([(i-self.cx)/self.fl_x, (j-self.cy)/self.fl_y, torch.ones_like(i), torch.ones_like(i)], -1).to(self.data_device)
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world =  pts_view @ c2w.T
        directions = pts_world[...,:3] - self.camera_center[None,None,:]
        Camera.ray_cache[self.cache_key] = (self.camera_center[None,None], directions / torch.norm(directions, dim=-1, keepdim=True))
        return Camera.ray_cache[self.cache_key]
    
    def cuda(self):
        if self.cached_cuda is not None:
            return self.cached_cuda
        if self.cache_key in Camera.projection_cache:
             # shallow copy not deep copy so we don't copy tensors
             cuda_copy = copy(self)
             for x,y in Camera.projection_cache[self.cache_key].items():
                 cuda_copy.__dict__[x] = y
        else:
            cuda_copy = deepcopy(self)
            cache_entry = {}
            for k, v in cuda_copy.__dict__.items():
                if isinstance(v, torch.Tensor):
                    cuda_copy.__dict__[k] = v.to(cuda_copy.data_device,non_blocking = True)
                    cache_entry[k] = cuda_copy.__dict__[k]
            Camera.projection_cache[self.cache_key] = cache_entry

        self.cached_cuda=cuda_copy
        return self.cached_cuda


    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


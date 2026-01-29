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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d

from typing import List, Optional

import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import tinycudann as tcnn

from dahuffman import HuffmanCodec
import math
from tqdm import tqdm

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,
                 gaussian_dim : int = 3,
                 time_duration: list = [-0.5, 0.5],
                 rot_4d: bool = False,
                 force_sh_3d: bool = False,
                 sh_degree_t : int = 0,
                 max_hashmap: int = 0,
                 mask_prune: bool = False,
                 vq_attributes: List[str] = [],
                 ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t
        
        self.mask_prune = mask_prune
        self._mask = torch.empty(0)
        
        self.max_hashmap = max_hashmap
        if self.max_hashmap > 0:
            assert '_features_dc' not in vq_attributes and '_features_rest' not in vq_attributes, 'hash and vq is not compatible for SH'
            self.max_sh_degree_t = self.max_sh_degree = 0
            self.recolor = tcnn.Encoding(
                    n_input_dims=4,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": max_hashmap,
                        "base_resolution": 16,
                        "per_level_scale": 1.447,
                    },
            )
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 3
                },
                )
            self.mlp_head = tcnn.Network(
                    n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
                    n_output_dims=3,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    },
                )
            
            
        self.vq_attributes = vq_attributes
        self.indexed = False
        
        self.setup_functions()

        self.old_xyz = []
        self.old_t = []
        self.old_mask = []

        self.old_features_dc = []
        self.old_features_rest = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_scaling_t = []
        self.old_rotation = []
        self.old_rotation_r = []

        self.segment_times = 0

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._mask
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self._mask
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._mask) = model_args
        elif self.gaussian_dim == 4:
            if len(model_args) == 19:
                (self.active_sh_degree,
                 self._xyz,
                 self._features_dc,
                 self._features_rest,
                 self._scaling,
                 self._rotation,
                 self._opacity,
                 self.max_radii2D,
                 xyz_gradient_accum,
                 t_gradient_accum,
                 denom,
                 opt_dict,
                 self.spatial_lr_scale,
                 self._t,
                 self._scaling_t,
                 self._rotation_r,
                 self.rot_4d,
                 self.env_map,
                 self.active_sh_degree_t) = model_args

                self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")
            else: # 20
                (self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                t_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t,
                self._mask) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if not (self.indexed and '_scaling' in self.vq_attributes):
            return self.scaling_activation(self._scaling)
        else:
            return self.scaling_activation(
                self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0)
            )
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        if not (self.indexed and '_scaling' in self.vq_attributes):
            return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
        else:
            return self.scaling_activation(
                torch.cat([self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0), self._scaling_t], dim = 1)
            )
    
    @property
    def get_rotation(self):
        if not (self.indexed and '_rotation' in self.vq_attributes):
            return self.rotation_activation(self._rotation)
        else:
            return self.rotation_activation(
                self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation).sum(dim=0).squeeze(0)
            )
    
    @property
    def get_rotation_r(self):
        if not (self.indexed and '_rotation' in self.vq_attributes):
            return self.rotation_activation(self._rotation_r)
        else:
            return self.rotation_activation(
                self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation_r).sum(dim=0).squeeze(0)
            )
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_mask(self):
        return self._mask
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        if not (self.indexed and '_features_dc' in self.vq_attributes):
            features_dc = self._features_dc
        else:
            features_dc = self.vq_features_dc_model.get_codes_from_indices(self.rvq_indices_features_dc).sum(dim=0).squeeze(0).unsqueeze(1).contiguous()
        if not (self.indexed and '_features_rest' in self.vq_attributes):
            features_rest = self._features_rest
        else:
            features_rest = self.vq_features_rest_model.get_codes_from_indices(self.rvq_indices_features_rest).sum(dim=0).squeeze(0).view(self._features_rest.shape[0], -1, 3).contiguous()
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                              self._rotation, 
                                                              self._rotation_r,
                                                              dt = timestamp - self.get_t)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        
        if self.mask_prune:
            l.append(
                {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"}
            )
            
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        if self.max_hashmap > 0:
            other_params = []
            for params in self.recolor.parameters():
                other_params.append(params)
            for params in self.mlp_head.parameters():
                other_params.append(params)
            self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
            self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler([
                    torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_net, start_factor=0.01, total_iters=100
                ),
                    torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_net,
                    milestones=training_args.net_lr_step,
                    gamma=0.33,
                ),
            ])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.mask_prune:
            self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]
            
        if self.indexed:
            if '_scaling' in self.vq_attributes:
                self.rvq_indices_scaling = self.rvq_indices_scaling[:,valid_points_mask]
            if '_rotation' in self.vq_attributes:
                self.rvq_indices_rotation = self.rvq_indices_rotation[:,valid_points_mask]
                self.rvq_indices_rotation_r = self.rvq_indices_rotation_r[:,valid_points_mask]
            if '_features_dc' in self.vq_attributes:
                self.rvq_indices_features_dc = self.rvq_indices_features_dc[:,valid_points_mask]
            if '_features_rest' in self.vq_attributes:
                self.rvq_indices_features_rest = self.rvq_indices_features_rest[:,valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_mask):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.mask_prune:
            d["mask"] = new_mask
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.mask_prune:
            self._mask = optimizable_tensors["mask"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self.mask_prune:
            new_mask = self._mask[selected_pts_mask].repeat(N,1)
        else:
            new_mask = None
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        if self.mask_prune:
            new_mask = self._mask[selected_pts_mask]
        else:
            new_mask = None
        
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.mask_prune:
            prune_mask = torch.logical_or(prune_mask, (torch.sigmoid(self._mask) <= 0.05).squeeze())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def prune_by_mask(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
            
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
        
    def post_quant(self, param, prune=False):
        max_val = torch.amax(param)
        min_val = torch.amin(param)
        param = (param - min_val)/(max_val - min_val)
        quant = torch.round(param * 255.0) / 255.0
        out = (max_val - min_val)*quant + min_val
        if prune:
            quant = quant*(torch.abs(out) > 0.1)
            out = out*(torch.abs(out) > 0.1)
        return torch.nn.Parameter(out), quant
    
    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        return total_mb
    
    def vector_quantization(self, training_args, finetuning_lr_scale=0.1):
        prune_mask = (self.get_opacity <= 0.01).squeeze()
        if self.mask_prune:
            prune_mask = torch.logical_or(prune_mask, (torch.sigmoid(self._mask) <= 0.01).squeeze())
        self.prune_points(prune_mask)
        
        codebook_params = []
        
        other_params = [
            {'params': [self._xyz], 'lr': training_args.position_lr_final * self.spatial_lr_scale * finetuning_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * finetuning_lr_scale, "name": "opacity"},
            {'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale * finetuning_lr_scale, "name": "t"},
            {'params': [self._scaling_t], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "scaling_t"}
            
        ]
        
        if self.vq_attributes:
            if '_scaling' in self.vq_attributes:
                self.rvq_size_scaling = 64
                self.rvq_num_scaling = 6
                self.rvq_iter_scaling = 512
                self.rvq_bit_scaling = math.log2(self.rvq_size_scaling)
                scaling_params = self._scaling
                self.vq_scaling_model = ResidualVQ(dim = 3, codebook_size = self.rvq_size_scaling, num_quantizers = self.rvq_num_scaling, commitment_weight = 0., 
                                    kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                                    learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0004)).cuda()
                
                for _ in tqdm(range(self.rvq_iter_scaling - 1)):
                    _, _, _ = self.vq_scaling_model(scaling_params.unsqueeze(0))
                _, self.rvq_indices_scaling, _ = self.vq_scaling_model(scaling_params.unsqueeze(0))
                codebook_params.append({'params': [p for p in self.vq_scaling_model.parameters()], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "vq_scaling"})
            else:
                other_params.append({'params': [self._scaling], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "scaling"})

            if '_rotation' in self.vq_attributes:
                self.rvq_size_rotation = 64
                self.rvq_num_rotation = 6
                self.rvq_iter_rotation = 1024
                self.rvq_bit_rotation = math.log2(self.rvq_size_rotation)
                rotation_params = torch.cat([self.get_rotation, self.get_rotation_r], dim=0)
                self.vq_rotation_model = ResidualVQ(dim = 4, codebook_size=self.rvq_size_rotation, num_quantizers=self.rvq_num_rotation, 
                                commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                                learnable_codebook=True, 
                                in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0008)).cuda()
                for _ in tqdm(range(self.rvq_iter_rotation - 1)):
                    _, _, _ = self.vq_rotation_model(rotation_params.unsqueeze(0))
                _, rvq_indices_rotation, _ = self.vq_rotation_model(rotation_params.unsqueeze(0))
                self.rvq_indices_rotation, self.rvq_indices_rotation_r = rvq_indices_rotation.split(self._xyz.shape[0], dim=1)
                codebook_params.append({'params': [p for p in self.vq_rotation_model.parameters()], 'lr': training_args.rotation_lr * finetuning_lr_scale, "name": "vq_rotation"})
            else:
                other_params.append({'params': [self._rotation], 'lr': training_args.scaling_lr * finetuning_lr_scale, "name": "scaling"})
                other_params.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr * finetuning_lr_scale, "name": "rotation_r"})
                
            if '_features_dc' in self.vq_attributes:
                self.rvq_size_features_dc = 64
                self.rvq_num_features_dc = 6
                self.rvq_iter_features_dc = 1024
                self.rvq_bit_features_dc = math.log2(self.rvq_size_features_dc)
                base_features_dc_params = self._features_dc
                self.vq_features_dc_model = ResidualVQ(dim = 3, codebook_size=self.rvq_size_features_dc, num_quantizers=self.rvq_num_features_dc, 
                        commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                        learnable_codebook=True, 
                        in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
                for _ in tqdm(range(self.rvq_iter_features_dc - 1)):
                    _, _, _ = self.vq_features_dc_model(base_features_dc_params.squeeze(1).unsqueeze(0))
                _, self.rvq_indices_features_dc, _ = self.vq_features_dc_model(base_features_dc_params.squeeze(1).unsqueeze(0))
                codebook_params.append({'params': [p for p in self.vq_features_dc_model.parameters()], 'lr': training_args.feature_lr * finetuning_lr_scale, "name": "vq_features_dc"})
            else:
                other_params.append({'params': [self._features_dc], 'lr': training_args.feature_lr * finetuning_lr_scale, "name": "f_dc"})
            
            if '_features_rest' in self.vq_attributes:
                self.rvq_size_features_rest = 128
                self.rvq_num_features_rest = 16
                self.rvq_iter_features_rest = 2048
                self.rvq_bit_features_rest = math.log2(self.rvq_size_features_rest)
                rest_features_rest_params = self._features_rest.flatten(1)
                self.vq_features_rest_model = ResidualVQ(dim = rest_features_rest_params.shape[1], 
                                                         codebook_size=self.rvq_size_features_rest, num_quantizers=self.rvq_num_features_rest, 
                        commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, 
                        learnable_codebook=True, 
                        in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
                for _ in tqdm(range(self.rvq_iter_features_rest - 1)):
                    _, _, _ = self.vq_features_rest_model(rest_features_rest_params.unsqueeze(0))
                _, self.rvq_indices_features_rest, _ = self.vq_features_rest_model(rest_features_rest_params.unsqueeze(0))
                codebook_params.append({'params': [p for p in self.vq_features_rest_model.parameters()], 'lr': training_args.feature_lr / 20.0 * finetuning_lr_scale, "name": "vq_features_rest"})
            else:
                other_params.append({'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * finetuning_lr_scale, "name": "f_rest"},)
            
        self.optimizer_codebook = torch.optim.Adam(codebook_params, lr=0.0, eps=1e-15)
        self.optimizer_others = torch.optim.Adam(other_params, lr=0.0, eps=1e-15)
        self.indexed = True
            
    def final_prune(self, compress=False):
        prune_mask = (self.get_opacity <= 0.01).squeeze()
        if self.mask_prune:
            prune_mask = torch.logical_or(prune_mask, (torch.sigmoid(self._mask) <= 0.01).squeeze())
        self.prune_points(prune_mask)
        
        self._xyz = self._xyz.clone().half().float()
        position_mb = self._xyz.shape[0]*3*16/8/10**6
        
        self._t = self._t.clone().half().float()
        position_t_mb = self._xyz.shape[0]*16/8/10**6
        
        # self._opacity = self._opacity.clone().half().float()
        opacity_mb = self._xyz.shape[0]*16/8/10**6
        
        self._scaling_t = self._scaling_t.clone().half().float()
        scaling_t_mb = opacity_mb
        
        if self.indexed and '_scaling' in self.vq_attributes:
            self._scaling = self.vq_scaling_model.get_codes_from_indices(self.rvq_indices_scaling).sum(dim=0).squeeze(0)
            # scale_mb = self._xyz.shape[0]*self.rvq_bit_scaling*self.rvq_num_scaling/8/10**6 + 2**self.rvq_bit_scaling*self.rvq_num_scaling*3*32/8/10**6
            scale_mb = self.huffman_encode(self.rvq_indices_scaling) + 2**self.rvq_bit_scaling*self.rvq_num_scaling*3*32/8/10**6
        else:
            self._scaling = self._scaling.clone().half().float()
            scale_mb = position_mb
            
        if self.indexed and '_rotation' in self.vq_attributes:
            self._rotation = self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation).sum(dim=0).squeeze(0)
            self._rotation_r = self.vq_rotation_model.get_codes_from_indices(self.rvq_indices_rotation_r).sum(dim=0).squeeze(0)
            rotation_mb = 2 * self.huffman_encode(self.rvq_indices_rotation) + 2**self.rvq_bit_rotation*self.rvq_num_rotation*8*32/8/10**6
        else:
            self._rotation = self._rotation.clone().half().float()
            self._rotation_r = self._rotation_r.clone().half().float()
            rotation_mb = self._xyz.shape[0]*8*16/8/10**6
        
        if self.max_hashmap > 0:
            hash_mb = self.recolor.params.shape[0]*16/8/10**6
            mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
            color_mb = hash_mb + mlp_mb
        else:
            color_mb = 0
            if self.indexed and '_features_dc' in self.vq_attributes:
                self._features_dc = self.vq_features_dc_model.get_codes_from_indices(self.rvq_indices_features_dc).sum(dim=0).squeeze(0).unsqueeze(1).contiguous()
                color_mb += self._xyz.shape[0]*self.rvq_bit_features_dc*self.rvq_num_features_dc/8/10**6 + 2**self.rvq_bit_features_dc*self.rvq_num_features_dc*3*32/8/10**6
            else:
                self._features_dc = self._features_dc.clone().half().float()
                color_mb += self._xyz.shape[0]*3*16/8/10**6
            if self.indexed and '_features_rest' in self.vq_attributes:
                self._features_rest = self.vq_features_rest_model.get_codes_from_indices(self.rvq_indices_features_rest).sum(dim=0).squeeze(0).view(self._features_rest.shape[0], -1, 3).contiguous()
                color_mb += self._xyz.shape[0]*self.rvq_bit_features_rest*self.rvq_num_features_rest/8/10**6 + 2**self.rvq_bit_features_rest*self.rvq_num_features_rest*3*(self.get_max_sh_channels-1)*32/8/10**6
            else:
                self._features_rest = self._features_rest.clone().half().float()
                color_mb += self._xyz.shape[0]*3*(self.get_max_sh_channels-1)*16/8/10**6
        
        sum_mb = position_mb+position_t_mb+scaling_t_mb+opacity_mb+scale_mb+rotation_mb+color_mb
        
        mb_str = "Storage\nposition: "+str(position_mb)+"\nt: "+str(position_t_mb)+"\nscale: "+str(scale_mb)+"\nscale_t: "+str(scaling_t_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)
        if self.max_hashmap > 0:
            mb_str = mb_str + "\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)
        else:
            mb_str = mb_str + "\ncolor: "+str(color_mb)
        mb_str = mb_str + "\ntotal: "+str(sum_mb)+" MB"
        
        if compress:
            self._opacity, quant_opa = self.post_quant(self._opacity)
            opacity_mb = self.huffman_encode(quant_opa)
            if self.max_hashmap > 0:
                self.recolor.params, quant_hash = self.post_quant(self.recolor.params, True)
                hash_mb = self.huffman_encode(quant_hash)
                mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
                color_mb = hash_mb + mlp_mb
            sum_mb = position_mb+position_t_mb+scaling_t_mb+opacity_mb+scale_mb+rotation_mb+color_mb
            mb_str = mb_str+"\n\nAfter PP\nposition: "+str(position_mb)+"\nt: "+str(position_t_mb)+"\nscale: "+str(scale_mb)+"\nscale_t: "+str(scaling_t_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)
            if self.max_hashmap > 0:
                mb_str = mb_str + "\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)
            else:
                mb_str = mb_str + "\ncolor: "+str(color_mb)
            mb_str = mb_str + "\ntotal: "+str(sum_mb)+" MB"
        else:
            self._opacity = self._opacity.clone().half().float()
        torch.cuda.empty_cache()
        return mb_str

    @torch.no_grad()
    def segment(self, mask=None):
        assert mask is not None
        # mask = (self._mask > 0)
        mask = mask.squeeze()
        # assert mask.shape[0] == self._xyz.shape[0]
        if torch.count_nonzero(mask) == 0:
            mask = ~mask
            print("Seems like the mask is empty, segmenting the whole point cloud. Please run seg.py first.")

        self.old_xyz.append(self._xyz)
        self.old_t.append(self._t)
        self.old_mask.append(self._mask)

        self.old_features_dc.append(self._features_dc)
        self.old_features_rest.append(self._features_rest)
        self.old_opacity.append(self._opacity)
        self.old_scaling.append(self._scaling)
        self.old_scaling_t.append(self._scaling_t)
        self.old_rotation.append(self._rotation)
        self.old_rotation_r.append(self._rotation_r)

        if self.optimizer is None:
            self._xyz = self._xyz[mask]
            self._t = self._t[mask]
            # self._mask = self._mask[mask]

            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._opacity = self._opacity[mask]
            self._scaling = self._scaling[mask]
            self._scaling_t = self._scaling_t[mask]
            self._rotation = self._rotation[mask]
            self._rotation_r = self._rotation_r[mask]

        else:
            optimizable_tensors = self._prune_optimizer(mask)

            self._xyz = optimizable_tensors["xyz"]
            self._t = optimizable_tensors["t"]

            # self._mask = optimizable_tensors["mask"]

            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._scaling_t = optimizable_tensors["scaling_t"]
            self._rotation_r = optimizable_tensors["rotation_r"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]
            self.t_gradient_accum = self.t_gradient_accum[mask]

            self.denom = self.denom[mask]

        # print(self.segment_times, torch.unique(self._mask))
        self.segment_times += 1
        tmp = self._mask[self._mask == self.segment_times]
        tmp[mask] += 1
        self._mask[self._mask == self.segment_times] = tmp

        # print(self._mask[self._mask == self.segment_times][mask].shape)
        # print(self.segment_times, torch.unique(self._mask), torch.unique(mask))

    def roll_back(self):
        try:
            self._xyz = self.old_xyz.pop()
            self._t = self.old_t.pop()
            # self._mask = self.old_mask.pop()

            self._features_dc = self.old_features_dc.pop()
            self._features_rest = self.old_features_rest.pop()
            self._opacity = self.old_opacity.pop()
            self._scaling = self.old_scaling.pop()
            self._rotation = self.old_rotation.pop()
            self._scaling_t = self.old_scaling_t.pop()
            self._rotation_r = self.old_rotation_r.pop()

            self._mask[self._mask == self.segment_times + 1] -= 1
            self.segment_times -= 1
        except:
            pass

    @torch.no_grad()
    def clear_segment(self):
        try:
            self._xyz = self.old_xyz[0]
            self._t = self.old_t[0]
            # self._mask = self.old_mask[0]

            self._features_dc = self.old_features_dc[0]
            self._features_rest = self.old_features_rest[0]
            self._opacity = self.old_opacity[0]
            self._scaling = self.old_scaling[0]
            self._rotation = self.old_rotation[0]
            self._scaling_t = self.old_scaling_t[0]
            self._rotation_r = self.old_rotation_r[0]

            self.old_xyz = []
            self.old_t = []
            self.old_mask = []

            self.old_features_dc = []
            self.old_features_rest = []
            self.old_opacity = []
            self.old_scaling = []
            self.old_rotation = []
            self.old_scaling_t = []
            self.old_rotation_r = []

            self.segment_times = 0
            self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")
        except:
            # print("Roll back failed. Please run gaussians.segment() first.")
            pass
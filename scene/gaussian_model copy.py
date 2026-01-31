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
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d
from envlight.light import EnvLight

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # self.scaling_t_activation = lambda x: torch.exp(10 * x)
        # self.scaling_t_inverse_activation = lambda x: torch.log(x) / 10
        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, gaussian_dim: int = 3, time_duration: list = [-0.5, 0.5],
                 force_sh_3d: bool = False,
                 contract=False, args=None, T=0.2):
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
        self.force_sh_3d = force_sh_3d
        self._velocity = torch.empty(0)
        self.t_gradient_accum = torch.empty(0)
        if self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        self.contract = False # cyr we contract split rules here
        self.contract_split = contract
        self.t_init = args.t_init

        self.T = args.cycle # cyr， 运动周期， xyz随mu做简谐运动

        self.setup_functions()

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
                self._velocity,
                self.env_map
            )

    def restore(self, model_args, training_args=None):
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
             self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
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
             self._velocity,
             self.env_map) = model_args
        self.setup_functions()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    def clip_xyz(self, xyz): # NOTE: useless
        mask_a = xyz.norm(dim=-1) > 2
        xyz[mask_a] = xyz[mask_a] / xyz[mask_a].clone().detach().norm(dim=-1, keepdim=True) * (2 - 1e-5)
        return xyz
        # self._xyz = self._xyz.clip(-2+1e-5,2-1e-5)

    def contract_mean(self, x): # NOTE: useless
        eps = torch.finfo(x.dtype).eps
        # eps = 1e-3
        # Clamping to eps prevents non-finite gradients when x == 0.
        x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
        mask = x_mag_sq <= 1
        z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
        return z

    def linear_contract(self, x): # NOTE: useless
        scale_factor = x.norm(dim=-1)
        scale_factor = torch.where(scale_factor < 1, 1, scale_factor)
        return scale_factor

    def contract_mean_std_factor(self, x): # NOTE: useless
        eps = torch.finfo(x.dtype).eps
        # eps = 1e-3
        # Clamping to eps prevents non-finite gradients when x == 0.
        x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
        x_mag_sqrt = torch.sqrt(x_mag_sq)
        mask = x_mag_sq <= 1
        z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
        det = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2))

        scale_factor = torch.where(mask, 1, (det ** (1 / x.shape[-1])))
        return z, scale_factor

    def inv_contract_mean(self, z): # NOTE: useless
        """The inverse of contract()."""
        eps = torch.finfo(z.dtype).eps
        # eps = 1e-3
        z_clip = z + 0
        mask_a = z_clip.norm(dim=-1) >= 2
        z_clip[mask_a] = z_clip[mask_a] / z_clip[mask_a].clone().detach().norm(dim=-1, keepdim=True) * (2 - 1e-5)

        # Clamping to eps prevents non-finite gradients when z == 0.
        z_mag_sq = torch.sum(z_clip ** 2, dim=-1, keepdim=True).clamp_min(eps)
        x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq))
        return x

    def inv_contract_std_factor(self, z): # NOTE: useless
        eps = torch.finfo(z.dtype).eps
        # eps = 1e-3

        z_clip = z + 0
        mask_a = z_clip.norm(dim=-1) >= 2
        z_clip[mask_a] = z_clip[mask_a] / z_clip[mask_a].clone().detach().norm(dim=-1, keepdim=True) * (2 - 1e-5)

        # Clamping to eps prevents non-finite gradients when z == 0.
        z_mag_sq = torch.sum(z_clip ** 2, dim=-1, keepdim=True).clamp_min(eps)
        vec1 = 1 / (2 * torch.sqrt(z_mag_sq) - z_mag_sq)
        vec2 = vec1 - vec1 ** 2 * z_mag_sq * (2 / torch.sqrt(z_mag_sq) - 2)
        mask = z_mag_sq <= 1
        det = torch.where(mask, 1, (vec1 * vec2 ** 2) ** (1 / 3))
        return det


    def get_inv_contract_xyz_and_scaling(self, mask=None): # NOTE: useless
        if mask is None:
            ic_xyz = self.inv_contract_mean(self.get_xyz)
            ic_scale = ic_xyz.norm(dim=-1, keepdim=True)/self._xyz.norm(dim=-1, keepdim=True)*self.get_scaling
            return ic_xyz, ic_scale
        else:
            ic_xyz = self.inv_contract_mean(self.get_xyz[mask])
            ic_scale = ic_xyz.norm(dim=-1, keepdim=True) / self._xyz[mask].norm(dim=-1, keepdim=True) * self.get_scaling[mask]
            return ic_xyz, ic_scale


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_t_activation(self._scaling_t)

    @property
    def get_cov_t(self):
        return self.get_scaling_t ** 2

    @property
    def get_scaling_xyzt(self):
        return torch.cat([self.get_scaling, self.get_scaling_t], dim=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_velocity(self):
        return self._velocity

    def get_xyz_SHM(self, t):
        '''
        等效速度为 self._velocity*self.T * np.pi * 2
        '''
        a = 1/self.T * np.pi * 2
        return self._xyz + self._velocity*torch.sin((t-self._t)*a)/a

    def get_inst_velocity(self, t):
        """
        瞬时速度
        我们需要瞬时速度吗，不一定，振幅A应该代表他永远的速度，不确定
        """
        # return self._velocity*torch.cos((t-self._t)/self.T * np.pi * 2)
        # return self._velocity*torch.exp(-(t-self._t)**2/self.get_scaling_t**2)
        return self._velocity*torch.exp(-self.get_scaling_t/self.T/2)

        # return self._velocity


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim=1)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree + 1) ** 2
        elif self.gaussian_dim == 4:
            return sh_channels_4d[self.max_sh_degree]

    def get_marginal_t(self, timestamp, scaling_modifier=1):  # Standard
        return torch.exp(-0.5 * (self.get_t - timestamp) ** 2 / self.get_scaling_t ** 2)  # / torch.sqrt(2*torch.pi*sigma)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        ## random up and far
        z_max = fused_point_cloud[:,2].max() # xyz \in +-7, +-7, -0.2,0.5
        # import pdb; pdb.set_trace()
        r_max = 100000
        r_min = 2
        num_sph = 200000
        theta = 2*torch.pi*torch.rand(num_sph)
        phi = (torch.pi/2*0.99*torch.rand(num_sph))**1.5 # x**a decay
        s = torch.rand(num_sph)
        r_1 = s*1/r_min+(1-s)*1/r_max
        r = 1/r_1
        pts_sph = torch.stack([r*torch.cos(theta)*torch.cos(phi), r*torch.sin(theta)*torch.cos(phi), r*torch.sin(phi)],dim=-1).cuda()

        r_rec = r_min
        num_rec = 200000
        pts_rec = torch.stack([r_rec*(torch.rand(num_rec)-0.5),r_rec*(torch.rand(num_rec)-0.5),
                               r_rec*(torch.rand(num_rec))],dim=-1).cuda()

        pts_sph = torch.cat([pts_rec, pts_sph], dim=0)
        pts_sph[:,2] = -pts_sph[:,2]+1

        fused_point_cloud = torch.cat([fused_point_cloud, pts_sph], dim=0)
        features = torch.cat([features,
                              torch.zeros([pts_sph.size(0), features.size(1), features.size(2)]).float().cuda()],
                             dim=0)



        if self.gaussian_dim == 4:
            if pcd.time is None or pcd.time.shape[0] != fused_point_cloud.shape[0]:
                if self.t_init < 1:
                    # random t
                    fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                            self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
                else:
                    # fixed centered t
                    fused_times = torch.full_like(fused_point_cloud[..., :1],
                                                  0.5 * (self.time_duration[1] + self.time_duration[0]))
            else:
                fused_times = torch.from_numpy(np.asarray(pcd.time.copy())).cuda().float()
                fused_times_sh = torch.full_like(pts_sph[..., :1], 0.5 * (self.time_duration[1] + self.time_duration[0]))
                fused_times = torch.cat([fused_times, fused_times_sh], dim=0)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        # cyr, scale should calculate scale factor, still testing
        # dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.05)
        # foreground_size = 0.01
        # dist2 = torch.where(dist2>r_min, dist2-r_min+foreground_size, foreground_size)

        if self.contract:
            means, scale_factor = self.contract_mean_std_factor(fused_point_cloud)
            fused_point_cloud = means
            dist2 = dist2 * scale_factor[:, 0] ** 2

        ### not too large for
        num_pcd = pcd.points.shape[0]
        ori_norm = (self.linear_contract(fused_point_cloud)*0.1)[num_pcd:]**2 #0.1 scene extent
        dist2[num_pcd:] = torch.where(dist2[num_pcd:]>ori_norm, ori_norm, dist2[num_pcd:])

        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]

            # cyr, large time spand, for most static points
            # dist_t = torch.zeros_like(fused_times, device="cuda") + (
            #             self.time_duration[1] - self.time_duration[0]) * 0.3
            dist_t = torch.full_like(fused_times, (self.time_duration[1] - self.time_duration[0])*self.t_init)
            scales_t = self.scaling_t_inverse_activation(torch.sqrt(dist_t))
            # scales_t = torch.sqrt(dist_t)
            # velocity = torch.full((fused_point_cloud.shape[0], 3), np.log(0.01), device="cuda")
            velocity = torch.full((fused_point_cloud.shape[0], 3), 0., device="cuda")

        ## cyr, opacity ori init as 0.1, but should let point in distance converge first
        opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            self._velocity = nn.Parameter(velocity.requires_grad_(True))

    def random_init(self, min=-10, max=10, num_pts=100000, spatial_lr_scale=1.0):

        self.spatial_lr_scale = spatial_lr_scale
        min_rand_xyz = np.array([min, min, min])
        max_rand_xyz = np.array([max, max, max])
        xyz = np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.zeros_like(xyz)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                    self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(xyz).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            # dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) * 2

            scales_t = torch.log(torch.sqrt(dist_t))
            velocity = torch.full((fused_point_cloud.shape[0], 3), np.log(0.01), device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            self._velocity = nn.Parameter(velocity.requires_grad_(True))


    def init_from_depth(self, depth, rgb, extrinsic, intrinsic, num_pts=100000):
        mask = depth > 0
        pass

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4
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
        velocity = init_4d_gaussian['_velocity'].cuda()

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
        self._velocity = nn.Parameter(velocity.requires_grad_(True))

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
        if self.gaussian_dim == 4:  # TODO: tune time_lr_scale
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.t_lr_init, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling4d_lr, "name": "scaling_t"})
            l.append({'params': [self._velocity], 'lr': training_args.velocity_lr * self.spatial_lr_scale, "name": "velocity"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        final_decay = training_args.position_lr_final / training_args.position_lr_init

        self.t_scheduler_args = get_expon_lr_func(lr_init=training_args.t_lr_init,
                                                    lr_final=training_args.t_lr_init * final_decay,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.v_scheduler_args = get_expon_lr_func(lr_init=training_args.velocity_lr,
                                                    lr_final=training_args.velocity_lr * final_decay,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if param_group["name"] == "t" and self.gaussian_dim == 4:  # TODO: tune t_scheduler
                lr = self.t_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

            if param_group["name"] == "v" and self.gaussian_dim == 4:  # TODO: tune t_scheduler
                lr = self.v_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * self.get_max_sh_channels - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, self.get_max_sh_channels - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            self._velocity = optimizable_tensors['velocity']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_velocity):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            d["velocity"] = new_velocity

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            self._velocity = optimizable_tensors['velocity']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2, time_split=False,
                          joint_sample=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract_split:
            scale_factor = self._xyz.norm(dim=-1)-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent*scale_factor)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        ## cyr if we split in 4D, 2*0.8**3 = 1, 2*0.8**2 = 1.28, we should sample more points to retain energy not decay
        decay_factor = N*0.8
        if self.gaussian_dim == 4 and self.no_time_split:
            N = N+1


        if self.gaussian_dim == 4 and time_split:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_time_mask = torch.where(padded_grad_t >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense * (self.time_duration[1] - self.time_duration[0])
            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values > extend_thresh)
            # print(f"num_to_densify_time: {torch.where(padded_grad_t >= grad_t_threshold, True, False).sum()}, num_to_split_time: {selected_time_mask.sum()}")
            if joint_sample:
                print(
                    f"split joint_sample: {100 * (torch.logical_or(selected_pts_mask, selected_time_mask).sum() - selected_pts_mask.sum()) / selected_pts_mask.sum():2f}%")
                selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)
            else:
                print(f"split: {100 * selected_time_mask / selected_pts_mask.sum():2f}%")

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (decay_factor))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        xyz = self.get_xyz[selected_pts_mask]

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        # if self.contract:
        #     ori_xyz = xyz.repeat(N, 1)
        #     inv_ori_xyz = self.inv_contract_mean(ori_xyz)
        #     scale_factor = inv_ori_xyz.norm(dim=-1, keepdim=True)/ori_xyz.norm(dim=-1, keepdim=True)
        #     new_xyz_inv = inv_ori_xyz+scale_factor*(ori_xyz-new_xyz)
        #     new_xyz = self.contract_mean(new_xyz_inv)

        new_t = None
        new_scaling_t = None
        new_velocity = None
        if self.gaussian_dim == 4:
            stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
            means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
            samples_t = torch.normal(mean=means_t, std=stds_t)
            new_t = samples_t+self.get_t[selected_pts_mask].repeat(N, 1)

            new_scaling_t = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1)/ (decay_factor))
            new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
            # the rule should be : xyz grad split xyz scale, t grad determine which points to split t
            # do not split too small point in t scale or xyz scale
            # do not split too
            not_split_xyz_mask =  torch.max(self.get_scaling[selected_pts_mask], dim=1).values < \
                                  self.percent_dense * scene_extent*scale_factor[selected_pts_mask]
            new_scaling[not_split_xyz_mask.repeat(N)] = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(N, 1))[not_split_xyz_mask.repeat(N)]

            if time_split:
                not_split_t_mask = self.get_scaling_t[selected_pts_mask].squeeze() < extend_thresh
                # not_split_t_mask = torch.logical_or(not_split_t_mask, ~selected_time_mask[selected_pts_mask])
                new_scaling_t[not_split_t_mask.repeat(N)] = self.scaling_t_inverse_activation(
                    self.get_scaling_t[selected_pts_mask].repeat(N, 1))[not_split_t_mask.repeat(N)]



            if self.no_time_split:
                # no time split
                new_t = 0 + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_t_inverse_activation(
                    self.get_scaling_t[selected_pts_mask].repeat(N, 1))
            # else:
            #     # new_xyz = new_xyz + new_velocity * (-samples_t)
            #     new_xyz = new_xyz + new_velocity * (-samples_t)
            #     # new_xyz = new_xyz + self.get_inst_velocity(t) * (-samples_t)




        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_t, new_scaling_t, new_velocity)

        ### CYR!! 注释prune之后psnr反而高了。self supervise 0.5时 RGBloss减半，因此分裂的慢了，静态点确实会有很大的tscale，因此重建变得稳定
        ### 这个点需要探究，因此
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_split_joint(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2, time_split=False,
                          joint_sample=True):
        '''
        A = xyz_grad > thr, B = t_grad > thr, then we should Split xyz in A-B, Split t in B-A, joint split xyzt in A and B
        '''
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract_split:
            scale_factor = self._xyz.norm(dim=-1)-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent*scale_factor)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")

        if self.gaussian_dim == 4 and time_split:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_time_mask = torch.where(padded_grad_t >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense * (self.time_duration[1] - self.time_duration[0])
            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values > extend_thresh)
            # print(f"num_to_densify_time: {torch.where(padded_grad_t >= grad_t_threshold, True, False).sum()}, num_to_split_time: {selected_time_mask.sum()}")
            if joint_sample:
                print(
                    f"split joint_sample: {100 * (torch.logical_or(selected_pts_mask, selected_time_mask).sum() - selected_pts_mask.sum()) / selected_pts_mask.sum():2f}%")
                selected_all_mask = torch.logical_or(selected_pts_mask, selected_time_mask)
            else:
                print(f"split: {100 * selected_time_mask / selected_pts_mask.sum():2f}%")
                selected_all_mask = torch.logical_or(selected_pts_mask, selected_time_mask)
        else:
            selected_all_mask = selected_pts_mask


        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_all_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_all_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_all_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_all_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_all_mask].repeat(N, 1)

        stds = self.get_scaling[selected_all_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_all_mask]).repeat(N, 1, 1)
        xyz = self.get_xyz[selected_all_mask]

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        # if self.contract:
        #     ori_xyz = xyz.repeat(N, 1)
        #     inv_ori_xyz = self.inv_contract_mean(ori_xyz)
        #     scale_factor = inv_ori_xyz.norm(dim=-1, keepdim=True)/ori_xyz.norm(dim=-1, keepdim=True)
        #     new_xyz_inv = inv_ori_xyz+scale_factor*(ori_xyz-new_xyz)
        #     new_xyz = self.contract_mean(new_xyz_inv)

        new_t = None
        new_scaling_t = None
        new_velocity = None
        if self.gaussian_dim == 4:
            stds_t = self.get_scaling_t[selected_all_mask].repeat(N, 1)
            means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
            samples_t = torch.normal(mean=means_t, std=stds_t)
            new_t = samples_t+self.get_t[selected_all_mask].repeat(N, 1)

            new_scaling_t = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_all_mask].repeat(N, 1)/ (0.8 * N))
            new_velocity = self._velocity[selected_all_mask].repeat(N, 1)


            if self.no_time_split:
                # no time split
                new_t = 0 + self.get_t[selected_all_mask].repeat(N, 1)
                ## cyr 11/5 update , tscale larger for t smaller vibrate
                # new_t = samples_t/torch.exp(2*new_scaling_t) + self.get_t[selected_all_mask].repeat(N, 1)

                new_scaling_t = self.scaling_t_inverse_activation(
                    self.get_scaling_t[selected_all_mask].repeat(N, 1))
            else:
                new_xyz = new_xyz + new_velocity * (-samples_t)


            # the rule should be : xyz grad split xyz scale, t grad determine which points to split t
            # do not split too small point in t scale or xyz scale
            # do not split too


            only_split_xyz_mask = ~selected_time_mask[selected_all_mask]
            new_scaling_t[only_split_xyz_mask.repeat(N)] = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_all_mask].repeat(N, 1))[only_split_xyz_mask.repeat(N)]
            new_t[only_split_xyz_mask.repeat(N)] = self.get_t[selected_all_mask][only_split_xyz_mask].repeat(N, 1)

            only_split_t_mask = ~selected_pts_mask[selected_all_mask]
            new_xyz[only_split_t_mask.repeat(N)] = xyz[only_split_t_mask].repeat(N, 1)
            new_scaling[only_split_t_mask.repeat(N)] = self.scaling_inverse_activation(
                self.get_scaling[selected_all_mask][only_split_t_mask].repeat(N, 1))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_t, new_scaling_t, new_velocity)

        # prune_filter = torch.cat(
        #     (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, time_clone=False):
        # Extract points that satisfy the gradient condition
        t_scale_factor=self.get_scaling_t.clamp(0,self.T)
        t_scale_factor=torch.exp(-t_scale_factor/self.T).squeeze()

        if self.contract_split:
            scale_factor = self._xyz.norm(dim=-1)-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent*scale_factor)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        if self.gaussian_dim == 4 and time_clone:
            selected_time_mask = torch.where(torch.norm(grads_t, dim=-1) >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense * (self.time_duration[1] - self.time_duration[0])
            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values <= extend_thresh)
            print(
                f"clone: {100 * (torch.logical_or(selected_pts_mask, selected_time_mask).sum() - selected_pts_mask.sum()) / selected_pts_mask.sum():2f}%")
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)
            # print(f"num_to_densify_time: {torch.where(torch.norm(grads_t, dim=-1) >= grad_t_threshold, True, False).sum()}, num_to_clone_time: {selected_time_mask.sum()}")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_velocity = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            new_velocity = self._velocity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_t, new_scaling_t, new_velocity)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            if self.t_grad:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=True)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=True)

                ##cyr, this function is beta version, still testing !!!
                # self.densify_and_split_joint(grads, max_grad, extent, grads_t, max_grad_t, time_split=True)

            else:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=False)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=False)


        prune_mask = (self.get_opacity < min_opacity).squeeze()

        ## contract nan cyr
        prune_mask_nan = torch.isnan(self.get_opacity).squeeze()
        xyz_nan_mask = torch.isnan(self.get_scaling.sum(-1)).squeeze()
        prune_mask = torch.logical_or(torch.logical_or(prune_mask_nan, prune_mask), xyz_nan_mask)
        print(f"nan point number {prune_mask.sum()}")
        # max_screen_size = None

        if self.contract_split:
            scale_factor = self._xyz.norm(dim=-1)-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)
        else:
            scale_factor = 1

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent * scale_factor  ## ori 0.1
            # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # if prune_mask.sum() > 1e5:
        #     import pdb; pdb.set_trace()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

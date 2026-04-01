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

import gc

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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

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



    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0,
                 prefilter_var: float = -1.0,optimizer_type="default"):
        self.optimizer_type = optimizer_type
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
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.tmp_radii = None
        self.optimizer = None
        self.shoptimizer = None

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

        self.prefilter_var = prefilter_var
        
        self.setup_functions()


    def _make_save_or_restore_calls(self,fn_to_call,argument_in):
        all_vars = []
        vars_to_save = [
            "gaussian_dim",
            "active_sh_degree",
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "xyz_gradient_accum_abs",
            "denom",
            "optimizer",
            "shoptimizer",
            "spatial_lr_scale",
        ]
        if self.gaussian_dim == 4:
            vars_to_save += [
                "_t",
                "_scaling_t",
                "_rotation_r",
                "rot_4d",
                "env_map",
                "active_sh_degree_t",
                "t_gradient_accum"
            ]
        if argument_in is not None:
            assert(len(vars_to_save) == len(argument_in))        
            for var_name,in_arg in zip(vars_to_save,argument_in):
                all_vars.append(fn_to_call(self,var_name,in_arg))
        else:
            for var_name in vars_to_save:
                all_vars.append(fn_to_call(self,var_name,None))

        return all_vars            


    def restore(self, model_args, training_args):
        gaussian_dim = model_args[0]
        if gaussian_dim != self.gaussian_dim:
            raise ValueError(f"Gaussian dimension mismatch in load: expected {self.gaussian_dim}, got {gaussian_dim}")



        def restore_fn(self, var_name, in_val):
            if isinstance(in_val, torch.Tensor):
                setattr(self, var_name, in_val)
            elif isinstance(in_val, torch.optim.Optimizer):
                return (var_name, in_val)
            return None


        optimizer_params = [x for x in self._make_save_or_restore_calls(restore_fn,model_args) if x is not None]
        if training_args is not None:
            self.training_setup(training_args,reset_accumulated_gradients=False)
            for var_name, optim in optimizer_params:
                if var_name == "optimizer":
                    self.optimizer.load_state_dict(optim.state_dict())
                elif var_name == "shoptimizer" and self.shoptimizer is not None:
                    self.shoptimizer.load_state_dict(optim.state_dict())
        print("Model restored with {} points.".format(self._xyz.shape))
        total_params = 0
        for v in model_args:
            if isinstance(v, torch.Tensor):
                print(v.shape)
                total_params += np.prod(v.shape)
        print("Total number of parameters: ", total_params) 

    def capture(self):


        def save_fn(self, var_name, in_val):
            var = getattr(self, var_name)
            if isinstance(var, torch.Tensor):
                return var
            elif isinstance(var, torch.optim.Optimizer):
                print("Saving optimizer state for ", var_name)
                return var.state_dict()
            else:
                return var

        save_data = self._make_save_or_restore_calls(save_fn,None)

        return save_data



    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_sh_features_dc(self):
        return self._features_dc
    
    @property
    def get_sh_features_rest(self):
        return self._features_rest
    
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
        if self.prefilter_var > 0.0:
            sigma += self.prefilter_var
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
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path,mmap=True)
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

    def training_setup(self, training_args,reset_accumulated_gradients = True):
        self.percent_dense = training_args.percent_dense
        if reset_accumulated_gradients:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]


        if self.gaussian_dim == 4: 
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
            self.shoptimizer = None
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def optimizer_step(self, iteration,radii=None):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if self.optimizer_type == "default":
            if iteration <= 15000:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
            elif iteration <= 20000:
                if iteration % 32 ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none = True)
                    self.shoptimizer.step()
                    self.shoptimizer.zero_grad(set_to_none = True)
            else:
                if iteration % 64 ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none = True)
                    self.shoptimizer.step()
                    self.shoptimizer.zero_grad(set_to_none = True)
        elif self.optimizer_type == "sparse_adam":
            visible = radii > 0
            self.optimizer.step(visible,radii.shape[0])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

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
    
    def _prune_optimizer_single_tensor(self, mask,name):
        retval = None
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

#                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].detach().requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    retval = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].detach().requires_grad_(True))
                    retval = group["params"][0]
        return retval


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

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
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors["t"]
            self._scaling_t = optimizable_tensors["scaling_t"]
            if self.rot_4d:
                self._rotation_r = optimizable_tensors["rotation_r"]
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

        print("Tensor shapes after prune:")
        for attr in dir(self):
            var = getattr(self, attr)
            if isinstance(var, torch.Tensor):
                print(f"Tensor {attr} has shape {var.shape}")
        


    def cat_one_tensor_to_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    old_exp_avg = stored_state["exp_avg"]
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"].detach(), torch.zeros_like(tensor)), dim=0).requires_grad_(True)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"].detach(), torch.zeros_like(tensor)), dim=0).requires_grad_(True)

#                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0].detach(), tensor.detach()), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state



                    return group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0].detach(), tensor), dim=0).requires_grad_(True))
                    return group["params"][0]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        # self._xyz = self.cat_one_tensor_to_optimizer(new_xyz, "xyz")
        # self._features_dc = self.cat_one_tensor_to_optimizer(new_features_dc, "f_dc")
        # self._features_rest = self.cat_one_tensor_to_optimizer(new_features_rest, "f_rest")
        # self._opacity = self.cat_one_tensor_to_optimizer(new_opacities, "opacity")
        # self._scaling = self.cat_one_tensor_to_optimizer(new_scaling, "scaling")
        # self._rotation = self.cat_one_tensor_to_optimizer(new_rotation, "rotation")
                

        # if self.gaussian_dim == 4:
        #     self._t = self.cat_one_tensor_to_optimizer(new_t, "t")
        #     self._scaling_t = self.cat_one_tensor_to_optimizer(new_scaling_t, "scaling_t")
        #     if self.rot_4d:
        #         self._rotation_r = self.cat_one_tensor_to_optimizer(new_rotation_r, "rotation_r")
        #     self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


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
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

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
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

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
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        print("Pruned {} points. Remaining points: {}".format(prune_mask.sum(), self.get_xyz.shape[0]))

        torch.cuda.empty_cache()

    def compute_temporal_score(self, timestamps):
        """Compute per-Gaussian temporal score (Eq. 5-6 from arXiv 2503.16422).

        The temporal score measures how much each 4D Gaussian contributes across
        the set of training timestamps.  For each timestamp t_j the marginal
        temporal weight marginal_t_i(t_j) = exp(-0.5*(t_i - t_j)^2 / sigma_t_i^2)
        is evaluated, and the scores are averaged over all timestamps.

        Args:
            timestamps: 1-D float tensor (or list) of training timestamps.

        Returns:
            Tensor of shape (N,) with temporal scores in [0, 1].
            Returns a tensor of ones when gaussian_dim != 4.
        """
        if self.gaussian_dim != 4:
            return torch.ones(self.get_xyz.shape[0], device="cuda")
        ts = torch.tensor(timestamps, dtype=torch.float32, device="cuda")  # (T,)
        with torch.no_grad():
            # marginal_t: (N, T) – temporal Gaussian weight for each Gaussian at each ts
            sigma = self.get_cov_t()  # (N, 1)
            if self.prefilter_var > 0.0:
                sigma = sigma + self.prefilter_var
            dt = self.get_t - ts.unsqueeze(0)  # (N, T)
            marginal_t = torch.exp(-0.5 * dt ** 2 / sigma)  # (N, T)
            temporal_score = marginal_t.mean(dim=1)  # (N,)
        return temporal_score

    def prune_by_spatio_temporal_score(self, spatial_contribs, timestamps, score_threshold):
        """Prune 4D Gaussians by combined spatio-temporal importance score (Eq. 7).

        The spatial score (Eq. 4) is the aggregated pixel contribution obtained by
        rendering each training view with compute_contrib=True and summing the
        per-Gaussian contributions.  The temporal score (Eq. 5-6) measures temporal
        coverage.  Their product is the spatio-temporal score; Gaussians below
        score_threshold are removed.

        Args:
            spatial_contribs: 1-D float tensor of shape (N,) with accumulated pixel
                contributions across all training views.
            timestamps: List/tensor of unique training timestamps used to evaluate
                the temporal score.
            score_threshold: Gaussians with spatio_temporal_score < score_threshold
                are pruned.
        """
        if self.gaussian_dim != 4:
            return
        with torch.no_grad():
            spatial_score = spatial_contribs.to("cuda")
            temporal_score = self.compute_temporal_score(timestamps)
            st_score = spatial_score * temporal_score
            prune_mask = st_score < score_threshold
        num_pruned = prune_mask.sum().item()
        print(
            f"\n[Spatio-temporal prune] score_threshold={score_threshold:.2e}  "
            f"removing {num_pruned}/{prune_mask.shape[0]} Gaussians"
        )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def generate_prefilter_masks(self, timestamps, threshold=0.05):
        """Compute per-timestamp active Gaussian masks for prefiltering during rendering.

        For each timestamp, a boolean mask is computed indicating which Gaussians have
        a marginal temporal weight above the given threshold. These masks can be saved
        and used to skip inactive Gaussians during inference.

        Args:
            timestamps: Iterable of float timestamps to generate masks for.
            threshold: Marginal temporal weight threshold (default 0.05, matching renderer).

        Returns:
            dict mapping each timestamp (float) to a 1-D boolean numpy array of shape
            (num_gaussians,) where True marks Gaussians active at that timestamp.
        """
        if self.gaussian_dim != 4:
            return {}
        masks = {}
        with torch.no_grad():
            for ts in timestamps:
                marginal_t = self.get_marginal_t(ts)
                mask = (marginal_t[:, 0] > threshold).cpu().numpy()
                masks[float(ts)] = mask
        return masks

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # Track absolute-value gradients for FastGS split criterion (columns 2:4 of the
        # 4-component screenspace tensor, when using render_fastgs).
        if viewspace_point_tensor.grad is not None and viewspace_point_tensor.grad.shape[-1] >= 4:
            self.xyz_gradient_accum_abs[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, 2:4], dim=-1, keepdim=True
            )
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        #print("adding densification stats with grad:", viewspace_point_grad.shape, update_filter.sum(),viewspace_point_grad)

        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(
            viewspace_point_grad[update_filter], dim=-1, keepdim=True
        )

        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    # ------------------------------------------------------------------
    # FastGS densification and pruning methods
    # ------------------------------------------------------------------

    def densify_and_clone_fastgs(self, metric_mask, filter_mask):
        """Clone Gaussians that satisfy both the gradient filter and the
        multi-view metric mask (FastGS criterion)."""
        selected_pts_mask = torch.logical_and(metric_mask, filter_mask)
        print("densify and clone:",selected_pts_mask.sum(),"/",len(selected_pts_mask))
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation,
            new_t, new_scaling_t, new_rotation_r,
        )

    def densify_and_split_fastgs(self, metric_mask, filter_mask, N=2):
        """Split Gaussians that satisfy both the gradient filter and the
        multi-view metric mask (FastGS criterion)."""
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
        combined = torch.logical_and(metric_mask, filter_mask)
        selected_pts_mask[:combined.shape[0]] = combined

        print("densify and split:",selected_pts_mask.sum(),"/",len(selected_pts_mask))


        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask].repeat(N, 1)
            new_scaling_t = self.scaling_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            new_t, new_scaling_t, new_rotation_r,
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool),
        ))
        self.prune_points(prune_filter)

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii,
                                  args, importance_score=None, pruning_score=None):
        """FastGS multi-view consistent densification and pruning.

        Steps:
        1. Candidates for densification are selected by gradient thresholds.
        2. The multi-view importance score further filters which Gaussians to
           clone/split – only those visible in many high-error views are densified.
        3. Standard opacity/size pruning with budget-controlled removal guided by
           the pruning score.

        Args:
            max_screen_size (int or None): maximum screen-space radius for pruning.
            min_opacity (float): opacity threshold below which Gaussians are pruned.
            extent (float): scene extent used for size thresholding.
            radii (Tensor): per-Gaussian screen-space radii from the last render.
            args: optimisation args with ``grad_thresh``, ``grad_abs_thresh``,
                  and ``dense`` attributes.
            importance_score (Tensor or None): per-Gaussian integer counts from
                :func:`~utils.fast_utils.compute_gaussian_score_fastgs`.
            pruning_score (Tensor or None): normalised per-Gaussian pruning score.
        """
        #print(f"Importance score: {importance_score} Pruning score: {pruning_score}")
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        #self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(
            torch.norm(grad_vars, dim=-1) >= args.densify_grad_threshold, True, False
        )
        # Fall back to densify_grad_threshold for abs if not configured separately.
        # The default multiplier of 6 comes from the FastGS paper where
        # grad_abs_thresh ≈ 0.0012 ≈ 6 × grad_thresh ≈ 6 × 0.0002.
        grad_abs_thresh = getattr(args, 'densify_grad_abs_threshold', args.densify_grad_threshold * 6)
        grad_qualifiers_abs = torch.where(
            torch.norm(grads_abs, dim=-1) >= grad_abs_thresh, True, False
        )

        print("EXTENT:",extent)
        print(f"Scales{self.get_scaling.shape}:",self.get_scaling)

        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.percent_dense * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.percent_dense * extent



        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        print("GRAD VARS",grad_vars)
        print("GRAD Qualifiirs",grad_qualifiers)
        print("GRAD ABS Qualifiers",grad_qualifiers_abs)
        print("clone qualifiers",clone_qualifiers)
        print("split qualifiers",split_qualifiers)
        print("all_clones",all_clones)


        if importance_score is not None:
            # Gaussians must appear in > fastgs_importance_threshold views to
            # be considered for densification.  Default of 5 comes from the
            # FastGS paper (multi-view consistency with 10 sampled views).
            importance_threshold = getattr(args, 'fastgs_importance_threshold', 5)
            metric_mask = importance_score > importance_threshold
        else:
            # Fall back: densify all gradient-selected Gaussians.
            metric_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")


        print("Densify:",metric_mask.sum())

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        # some points won't be visible at this timestep at all
        # - don't mess with them at all
        # 
        visible_points_mask  =(importance_score>0)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
#            prune_mask = torch.logical_and(prune_mask,visible_points_mask)

        
        if pruning_score is not None:
            scores = 1.0 - pruning_score
            to_remove = torch.sum(prune_mask)
            # Only remove 50 % of eligible Gaussians per step (budget-controlled
            # pruning from FastGS) to avoid over-aggressive scene degradation.
            remove_budget = int(0.5 * to_remove)
            print("pruning, remove budget: ", remove_budget, " to_remove: ", to_remove)
            if remove_budget > 0:
                n_init_points = self.get_xyz.shape[0]
                padded_importance = torch.zeros(n_init_points, dtype=torch.float32, device="cuda")
                n = min(scores.shape[0], n_init_points)
                padded_importance[:n] = 1.0 / (1e-6 + scores[:n].squeeze())
                selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
                sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                self.prune_points(final_prune)
            else:
                self.prune_points(prune_mask)
        else:
            self.prune_points(prune_mask)

        # Clamp opacities to avoid them exploding after densification.
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        #self.tmp_radii = None

        torch.cuda.empty_cache()

    def final_prune_fastgs(self, min_opacity, pruning_score=None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view
        consistency score.

        This is called after the main training phase (e.g., every 3 000 iterations
        between iterations 15 000–30 000) to aggressively remove Gaussians that
        are no longer needed.

        Args:
            min_opacity (float): remove Gaussians with opacity below this value.
            pruning_score (Tensor or None): normalised per-Gaussian pruning score
                from :func:`~utils.fast_utils.compute_gaussian_score_fastgs`.
                Gaussians with score > 0.9 are also pruned.
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if pruning_score is not None:
            n = min(pruning_score.shape[0], self.get_xyz.shape[0])
            scores_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            # Prune Gaussians with very high reconstruction inconsistency score
            # (top 10 % of the [0, 1] range, i.e., score > 0.9).
            scores_mask[:n] = pruning_score[:n].squeeze() > 0.9
            prune_mask = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(prune_mask)
        print("FastGS final prune: removed {} points. Remaining: {}".format(
            prune_mask.sum(), self.get_xyz.shape[0]))
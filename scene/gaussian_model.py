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
                 prefilter_var: float = -1.0,optimizer_type="default",
                 densifiers=[]):
        # densifiers is a list of densifier objects
        self.training = True
        self.densifiers = densifiers
        self.optimizer_type = optimizer_type
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
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
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.prefilter_var = prefilter_var
        
        self.setup_functions()


    def _make_save_or_restore_calls(self,fn_to_call,argument_in):
        # save as a dict so we can handle missing values etc. nicely
        all_vars = {}
        vars_to_save = [
            "gaussian_dim",
            "active_sh_degree",
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
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
                "active_sh_degree_t",
            ]

        if argument_in is not None:
            for var_name in vars_to_save:
                if var_name in argument_in:
                    all_vars[var_name]=(fn_to_call(self,var_name,argument_in[var_name]))
                else:
                   print("Missing variable {} in input data for save/restore, skipping.".format(var_name)) 
        else:
            for var_name in vars_to_save:
                all_vars[var_name] =(fn_to_call(self,var_name,None))

        all_vars["densifier_vars"] = {}
        for densifier in self.densifiers:
            densifier_in_args = {}
            densifier_name = densifier.__class__.__name__
            if argument_in is not None and densifier_name in argument_in["densifier_vars"]:
                densifier_in_args = argument_in["densifier_vars"][densifier_name]

            densifier_vars = densifier.get_save_vars(self)
            densifier_out_vars = {}
            densifier_name = densifier.__class__.__name__
            for var_name in densifier_vars:
                if argument_in is not None and var_name in densifier_in_args:
                    densifier_out_vars[var_name]=(fn_to_call(densifier,var_name,densifier_in_args[var_name]))
                else:
                    densifier_out_vars[var_name] =(fn_to_call(densifier,var_name,None))
            all_vars["densifier_vars"][densifier_name] = densifier_out_vars
        return all_vars

    def restore(self, model_args, training_args):
        # print("DENSIFIER VARS:",model_args["densifier_vars"].keys())
        # for d,vals in model_args["densifier_vars"].items():
        #     print("DENSIFIER {} VARS:".format(d),vals.keys())
        if type(model_args) == tuple:
            print("LOADING OLD MODEL ARGS, display only please!")
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
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t) = model_args
            return
        

        gaussian_dim = model_args["gaussian_dim"]
        if gaussian_dim != self.gaussian_dim:
            raise ValueError(f"Gaussian dimension mismatch in load: expected {self.gaussian_dim}, got {gaussian_dim}")

        def restore_fn(self, var_name, in_val):
            if in_val is None:
                print("Missing value in restore:", var_name)
            if isinstance(in_val, torch.Tensor):
                setattr(self, var_name, in_val)
            elif isinstance(getattr(self,var_name), torch.optim.Optimizer) and isinstance(in_val, dict):
                return (var_name, in_val)
            return None

        restore_return_values = self._make_save_or_restore_calls(restore_fn,model_args)
        optimizer_params = [x for x in restore_return_values.values() if x is not None and type(x) == tuple]
        if training_args is not None:
            self.training_setup(training_args,reset_accumulated_gradients=False)
            for densifier in self.densifiers:
                densifier_name = densifier.__class__.__name__
                if densifier_name not in model_args["densifier_vars"]:
                    densifier.training_setup(self,reset_accumulated_gradients=True)
                else:
                    densifier.training_setup(self,reset_accumulated_gradients=False)
            for var_name, optim_dict in optimizer_params:
                if var_name == "optimizer":
                    self.optimizer.load_state_dict(optim_dict)
                elif var_name == "shoptimizer" and self.shoptimizer is not None:
                    self.shoptimizer.load_state_dict(optim_dict)
        else:
            self.training = False
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
        for densifier in self.densifiers:
            densifier.training_setup(self, reset_accumulated_gradients)

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
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l+ sh_l, lr=0.0, eps=1e-15)
            self.shoptimizer = None
#            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
            self.shoptimizer = None
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def optimizer_step(self, iteration,radii=None):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
        if self.shoptimizer is not None:
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none = True)

        # ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        
        # if self.optimizer_type == "default":
        #     if iteration <= 15000:
        #         self.optimizer.step()
        #         self.optimizer.zero_grad(set_to_none = True)
        #         self.shoptimizer.step()
        #         self.shoptimizer.zero_grad(set_to_none = True)
        #     elif iteration <= 20000:
        #         if iteration % 32 ==0:
        #             self.optimizer.step()
        #             self.optimizer.zero_grad(set_to_none = True)
        #             self.shoptimizer.step()
        #             self.shoptimizer.zero_grad(set_to_none = True)
        #     else:
        #         if iteration % 64 ==0:
        #             self.optimizer.step()
        #             self.optimizer.zero_grad(set_to_none = True)
        #             self.shoptimizer.step()
        #             self.shoptimizer.zero_grad(set_to_none = True)
        # elif self.optimizer_type == "sparse_adam":
        #     visible = radii > 0
        #     self.optimizer.step(visible,radii.shape[0])

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
        print("Resetting opacity of all points to max 0.01 at iteration")
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
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
        if self.training:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            
            if self.gaussian_dim == 4:
                self._t = optimizable_tensors["t"]
                self._scaling_t = optimizable_tensors["scaling_t"]
                if self.rot_4d:
                    self._rotation_r = optimizable_tensors["rotation_r"]

            for densifier in self.densifiers:
                densifier.prune_points(mask)
        else:
            # pruning without optimizer step, for inference only
            self._xyz = self._xyz[valid_points_mask]
            self._features_dc = self._features_dc[valid_points_mask]
            self._features_rest = self._features_rest[valid_points_mask]
            self._opacity = self._opacity[valid_points_mask]
            self._scaling = self._scaling[valid_points_mask]
            self._rotation = self._rotation[valid_points_mask]
            if self.gaussian_dim == 4:
                self._t = self._t[valid_points_mask]
                self._scaling_t = self._scaling_t[valid_points_mask]
                if self.rot_4d:
                    self._rotation_r = self._rotation_r[valid_points_mask]

        # print("Tensor shapes after prune:")
        # for attr in dir(self):
        #     var = getattr(self, attr)
        #     if isinstance(var, torch.Tensor):
        #         print(f"Tensor {attr} has shape {var.shape}")
        


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

        for densifier in self.densifiers:
            densifier.densification_postfix(self) 

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



    def add_densification_stats(self, *,iteration, viewspace_point_tensor, update_filter, radii,avg_t_grad=None):
        for densifier in self.densifiers:
            densifier.add_densification_stats(self, iteration=iteration, viewspace_point_tensor=viewspace_point_tensor, 
                                              update_filter=update_filter, radii=radii, avg_t_grad=avg_t_grad)
        
    def add_densification_stats_grad(self, *,iteration, viewspace_point_grad, update_filter, radii, avg_t_grad=None):
        for densifier in self.densifiers:
            densifier.add_densification_stats_grad(gaussians=self, iteration=iteration, viewspace_point_grad=viewspace_point_grad, 
                                                   update_filter=update_filter, radii=radii, avg_t_gradient=avg_t_grad)

    def run_densifiers(self, iteration, scene, radii, pipe, bg,prune_only):
        for densifier in self.densifiers:
            if densifier.needs_densification(iteration):
                densifier.densify_and_prune(iteration, scene, self, radii, pipe, bg,prune_only=prune_only)

    def call_densifier_per_iteration(self, iteration, scene, radii, pipe, bg):
        for densifier in self.densifiers:
            densifier.per_iteration(iteration, scene, self, radii, pipe, bg)

    
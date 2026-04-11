import torch
from .densifier_base import DensifierBase   


from utils.general_utils import  build_rotation, build_rotation_4d


class PlainDensifier(DensifierBase):
    def __init__(self, opt):
        self.options = opt

    def training_setup(self, gaussians, reset_accumulated_gradients=True):
        """ Any setup that needs to be done before training starts can be done here, such as initializing accumulators."""
        if reset_accumulated_gradients:
            num_gaussians = gaussians.get_xyz.shape[0]
            self.xyz_gradient_accum = torch.zeros((num_gaussians, 1), device="cuda")
            self.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((num_gaussians), device="cuda")
            if gaussians.gaussian_dim == 4:
                self.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def add_densification_stats_grad(self, gaussians, iteration, viewspace_point_grad, update_filter, radii, avg_t_gradient):
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter],
            radii[update_filter],
        )
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if gaussians.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_gradient[update_filter]


    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg):
        max_grad = self.options.densify_grad_threshold
        min_opacity = self.options.thresh_opa_prune
        max_grad_t = self.options.densify_grad_t_threshold
        extent = scene.cameras_extent
        prune_only = False
        max_screen_size = (
            20 if iteration > self.options.opacity_reset_interval else None
        )


        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if gaussians.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self._densify_and_clone(gaussians,grads, max_grad, extent, grads_t, max_grad_t)
            self._densify_and_split(gaussians, grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        gaussians.prune_points(prune_mask)
        print("Pruned {} points. Remaining points: {}".format(prune_mask.sum(), gaussians.get_xyz.shape[0]))

        torch.cuda.empty_cache()


    def _densify_and_clone(self, gaussians,grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(gaussians.get_scaling, dim=1).values <= gaussians.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = gaussians._xyz[selected_pts_mask]
        new_features_dc = gaussians._features_dc[selected_pts_mask]
        new_features_rest = gaussians._features_rest[selected_pts_mask]
        new_opacities = gaussians._opacity[selected_pts_mask]
        new_scaling = gaussians._scaling[selected_pts_mask]
        new_rotation = gaussians._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if gaussians.gaussian_dim == 4:
            new_t = gaussians._t[selected_pts_mask]
            new_scaling_t = gaussians._scaling_t[selected_pts_mask]
            if gaussians.rot_4d:
                new_rotation_r = gaussians._rotation_r[selected_pts_mask]

        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation,
            new_t, new_scaling_t, new_rotation_r,
        )

    def _densify_and_split(self, gaussians,grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = gaussians.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(gaussians.get_scaling, dim=1).values > gaussians.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = gaussians._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = gaussians._opacity[selected_pts_mask].repeat(N,1)
        
        if not gaussians.rot_4d:
            stds = gaussians.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if gaussians.gaussian_dim == 4:
                stds_t = gaussians.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + gaussians.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = gaussians.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(gaussians._rotation[selected_pts_mask], gaussians._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N,1)

        gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        gaussians.prune_points(prune_filter)

    def densification_postfix(self, gaussians):
        self.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        if self.t_gradient_accum is not None:
            self.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def prune_points(self, prune_mask):
        valid_points_mask = ~prune_mask
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.t_gradient_accum is not None:
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]



    def get_save_vars(self,gaussians):
        """Return variables that should be saved or loaded with the scene, as attribute names """
        attrs = ["xyz_gradient_accum", "denom", "max_radii2D"]
        if gaussians.gaussian_dim == 4:
            attrs.append("t_gradient_accum")
        return attrs

    @torch.no_grad()
    def apply_debug_colour(self, gaussians, scene,pipe,bg, debug_type=""):
        from utils.sh_utils import RGB2SH
        if debug_type == "scale_imbalance":
            scales = gaussians.get_scaling
            display_score = torch.max(scales, dim=1).values / torch.min(scales, dim=1).values
            print(scales.shape,display_score.shape,display_score)
#            display_score -= display_score.min()
            display_score /= 20.0
# #            display_score*=5.0
            display_score = torch.clamp(display_score, 0.0, 1.0)

            rgb_tensor = RGB2SH(torch.stack((display_score, 1.0 - display_score, torch.zeros_like(display_score)), dim=1))
            gaussians._features_dc = rgb_tensor
            gaussians._features_rest = torch.zeros_like(gaussians.get_sh_features_rest)
        elif debug_type == "scale_t":
            scales = gaussians.get_cov_t()
            print(scales.max(),scales.min())
            display_score = scales
#            display_score -= display_score.min()
#            display_score /= display_score.max()
            display_score = torch.clamp(display_score, 0.0, 1.0)

            rgb_tensor = RGB2SH(torch.stack((display_score, 1.0 - display_score, torch.zeros_like(display_score)), dim=1))
            gaussians._features_dc = rgb_tensor
            gaussians._features_rest = torch.zeros_like(gaussians.get_sh_features_rest)




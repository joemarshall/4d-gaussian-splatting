import torch
from .densifier_base import DensifierBase   


from utils.general_utils import  build_rotation, build_rotation_4d
from .split_ops import *


class PlainDensifier(DensifierBase):
    def __init__(self, opt):
        super().__init__(opt, "plain")

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


    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg, *, prune_only):
        max_grad = self.options.densify_grad_threshold
        min_opacity = self.options.thresh_opa_prune
        max_grad_t = self.options.densify_grad_t_threshold
        extent = scene.cameras_extent
        max_screen_size = (
            20 if iteration > self.options.opacity_reset_interval else None
        )

        clone_pts_mask = None
        split_pts_mask = None
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if gaussians.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None


            
            clone_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
            clone_pts_mask = torch.logical_and(clone_pts_mask,
                                        torch.max(gaussians.get_scaling, dim=1).values <= gaussians.percent_dense*extent)
            
            split_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
            split_pts_mask = torch.logical_and(split_pts_mask,
                                              torch.max(gaussians.get_scaling, dim=1).values > gaussians.percent_dense*extent)


        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = torch.zeros_like(prune_mask) #self.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            print(f"Densification iteration {iteration}: {clone_pts_mask.sum().item()} clones, {split_pts_mask.sum().item()} splits, {prune_mask.sum().item()} prunes, {big_points_vs.sum().item()} big_points_vs, {big_points_ws.sum().item()} big_points_ws, extents {extent}, max_screen_size {max_screen_size}")
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        clone_split_prune(gaussians,clone_pts_mask,split_pts_mask,prune_mask,long_axis_split=self.options.split_on_long_axis)

        #print("Pruned {} points. Remaining points: {}".format(prune_mask.sum(), gaussians.get_xyz.shape[0]))

        torch.cuda.empty_cache()

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




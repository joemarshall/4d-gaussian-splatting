import torch
from .split_ops import *
from .densifier_base import DensifierBase   

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


class FastGSDensifier(DensifierBase):
    def __init__(self, opt):
       super().__init__(opt, "fastgs")

    def training_setup(self, gaussians,reset_accumulated_gradients = True):
        if reset_accumulated_gradients:
            num_gaussians = gaussians.get_xyz.shape[0]
            self.xyz_gradient_accum = torch.zeros((num_gaussians, 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((num_gaussians, 1), device="cuda")
            self.max_radii2D = torch.zeros((num_gaussians), device="cuda")
            self.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            if gaussians.gaussian_dim == 4:
                self.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            else:
                self.t_gradient_accum = None

    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg, *, options):
        """FastGS multi-view consistent densification and pruning.

        Steps:
        1. Candidates for densification are selected by gradient thresholds.
        2. The multi-view importance score further filters which Gaussians to
           clone/split – only those visible in many high-error views are densified.
        3. Standard opacity/size pruning with budget-controlled removal guided by
           the pruning score.

        """

        num_cams = self._get_option('fastgs_num_sample_cams', 40)
        min_opacity=self._get_option('thresh_opa_prune', 0.005)
        max_screen_size = (
            20 if iteration > self._get_option('opacity_reset_interval', 3000) else None
        )

        if options['final_prune']:
            min_opacity = self._get_option('thresh_opa_final_prune', 0.01)
            num_cams = self._get_option('final_prune_num_sample_cams', 40)
        else:
            # don't prune big points in final prune
            max_screen_size = None


        my_viewpoint_stack = scene.getTrainCameras()
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)

        extent=scene.cameras_extent

        if len(camlist) < 8:
            return  # skip FastGS pruning if too few cameras to get reliable multi-view scores
        
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        densify_grad_threshold = self._get_option( 'densify_grad_threshold', 0.0002)

        grad_qualifiers = torch.where(
            torch.norm(grad_vars, dim=-1) >= densify_grad_threshold, True, False
        )
        
        # Fall back to densify_grad_threshold for abs if not configured separately.
        # The default multiplier of 6 comes from the FastGS paper where
        # grad_abs_thresh ≈ 0.0012 ≈ 6 × grad_thresh ≈ 6 × 0.0002.
        grad_abs_thresh = self._get_option( 'densify_grad_abs_threshold', densify_grad_threshold * 6)
        grad_qualifiers_abs = torch.where(
            torch.norm(grads_abs, dim=-1) >= grad_abs_thresh, True, False
        )

        clone_qualifiers = torch.max(gaussians.get_scaling, dim=1).values <= self._get_option('percent_dense', 0.01) * extent
        split_qualifiers = torch.max(gaussians.get_scaling, dim=1).values > self._get_option('percent_dense', 0.01) * extent


        # all points that will be cloned or split if the multi-view importance criterion is also satisfied
        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)


        # candidates for pruning are either too big, or too transparent
        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # all points to prune
        final_prune = None

        # points to clone or split
        final_clones = None
        final_splits = None

        # points that aren't hit by importance pruning but might be prunable based 
        # on size/opacity
        unimportant_prunes = prune_mask.clone()

        prune_budget = int(prune_mask.sum() * 0.5)
        per_frame_prune_budget = (prune_budget // len(camlist))+1

        loss_thresh = self._get_option('fastgs_loss_thresh', 0.1)

        for frame_cameras in camlist:
            importance_score, pruning_score = compute_gaussian_score_fastgs(
                frame_cameras, gaussians, pipe, bg, loss_thresh, DENSIFY= options['densify']
            )


            # Gaussians must appear in > fastgs_importance_threshold views to
            # be considered for densification.  Default of 5 comes from the
            # FastGS paper (multi-view consistency with 10 sampled views).
            importance_threshold = self._get_option('fastgs_importance_threshold', 5)

            if options['densify']:
                metric_mask = importance_score > importance_threshold

                # points to be split or cloned for this frame
                this_clones = torch.logical_and(metric_mask,all_clones)
                this_splits = torch.logical_and(metric_mask,all_splits)

                if final_clones is None:
                    final_clones = this_clones
                else:
                    final_clones = torch.logical_or(final_clones, this_clones)

                if final_splits is None:
                    final_splits = this_splits
                else:
                    final_splits = torch.logical_or(final_splits, this_splits)

            # choose for pruning based on pruning scores
            # pruning score of 0 = no error, 1 = high error
            # 
            prune_mask_frame = torch.zeros_like(prune_mask)
            if final_prune is None:
                final_prune = prune_mask_frame
            possible_prunes = (pruning_score > 0.0)
            unimportant_prunes = torch.logical_and(unimportant_prunes, ~possible_prunes)
            if possible_prunes.sum() > 0:
#                print("Poss prune:",possible_prunes.sum())
                scores = 1.0 - pruning_score
                # now score = 1 = no error, 0 = high error 
                sampling_importance = 1.0 / (1e-6 + scores.squeeze())
                # don't ever prune untested points
                sampling_importance[~possible_prunes] = 0.0
                # or points that are already marked for split / clone/ prune
                if options['densify']:
                    sampling_importance[final_clones] = 0.0
                    sampling_importance[final_splits] = 0.0
                sampling_importance[final_prune] = 0.0
                if sampling_importance.sum() != 0:
                    # importance = 1/ 1.000001 for no error
                    # 1/0.000001 = 1 million for high error
                    sampled_indices = torch.multinomial(sampling_importance, per_frame_prune_budget, replacement=False)
                    prune_mask_frame[sampled_indices] = prune_mask[sampled_indices]
                else:
                    print("No possible prunes")
            final_prune|= prune_mask_frame

            if options['densify']:
                # anything pruned already can't be cloned or split in later steps
                all_splits = torch.logical_and(all_splits, ~final_prune)
                all_clones = torch.logical_and(all_clones, ~final_prune)

        if options['densify']:
            final_clones = torch.logical_and(final_clones, ~final_prune)
            final_splits = torch.logical_and(final_splits, ~final_prune)

        important_prunes = final_prune.sum()
        prune_budget_left = min(prune_budget - final_prune.sum(), unimportant_prunes.sum())
        if prune_budget_left > 0:
            sampling_importance = torch.zeros_like(unimportant_prunes, dtype=torch.float)
            sampling_importance[unimportant_prunes] = 1.0
            sampled_indices = torch.multinomial(sampling_importance, prune_budget_left, replacement=False)
            final_prune[sampled_indices] = True

        print("\n=== FastGS Densification/Pruning Summary ===")
        print("FastGS prune budget: {} points ({} per frame)".format(prune_budget, per_frame_prune_budget))


        if options['densify']:
            print("By importance [{} frames]: selected {} points for cloning, {} for splitting, {}+{} for pruning.".format(
                len(camlist), final_clones.sum(), final_splits.sum(), important_prunes, prune_budget_left
            ))
        else:
            print("By importance [{} frames]: selected  {}+{} for pruning only.".format(
                len(camlist),important_prunes, prune_budget_left
            ))

        # now do actual densify, split etc.
        clone_split_prune(gaussians, final_clones, final_splits, final_prune, long_axis_split=self._get_option('split_on_long_axis', False), repeat_count=self._get_option('densify_repeat_count', 2))

        # Clamp opacities to avoid them exploding after densification.
        opacities_new = inverse_sigmoid(
            torch.min(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * 0.8)
        )
        optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
        gaussians._opacity = optimizable_tensors["opacity"]
        # empty cache - do we need to do this still?
        torch.cuda.empty_cache()


    def add_densification_stats_grad(self, *,gaussians, iteration, viewspace_point_grad, update_filter, radii,avg_t_gradient):
        """FastGS uses xyz_gradient and norm xyz_gradient as stats, + t gradient in 4d """
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(
            viewspace_point_grad[update_filter], dim=-1, keepdim=True
        )
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter],
            radii[update_filter],
        )


        self.denom[update_filter] += 1
        if gaussians.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_gradient[update_filter]

    def per_iteration(self, iteration, scene, gaussians, radii, pipe, bg):
        pass

    def get_save_vars(self,gaussians):
        """Return variables that should be saved or loaded with the scene, as attribute names """
        attrs = ["xyz_gradient_accum", "xyz_gradient_accum_abs", "denom", "max_radii2D"]
        if gaussians.gaussian_dim == 4:
            attrs.append("t_gradient_accum")
        return attrs
        
    def densification_postfix(self, gaussians):
        self.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        if self.t_gradient_accum is not None:
            self.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def prune_points(self, gaussians, prune_mask):
        valid_points_mask = ~prune_mask
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.t_gradient_accum is not None:
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]


    @torch.no_grad()
    def apply_debug_colour(self, gaussians, scene,pipe,bg, debug_type=""):
        from utils.sh_utils import RGB2SH
        # debug colours based on multiview-consistency score
        num_cams = 10
        my_viewpoint_stack = scene.getTrainCameras()
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)
        loss_thresh = self._get_option('fastgs_loss_thresh', 0.1)

        print(len(camlist))
        if debug_type == "flagged_errors":
            display_score, _ = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, loss_thresh, DENSIFY=True
            )
            display_score = display_score.float()
        elif debug_type == "multiview_importance":
            _, display_score = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, loss_thresh, DENSIFY=False
            )
        print(display_score.shape)
        first_mask = display_score > 0
        quantiles = torch.quantile(display_score[first_mask], torch.tensor([0, 0.25, 0.5, 0.75, .8,.9,1], device=display_score.device))
        print(quantiles)
        update_mask = display_score > quantiles[4]
        display_score-= quantiles[4]
        display_score /= (quantiles[5] - quantiles[4])

#        display_score*=5.0
        display_score = torch.clamp(display_score, 0.0, 1.0)
        print(torch.max(display_score), torch.min(display_score))
        rgb_tensor = RGB2SH(torch.stack((display_score[update_mask], 1.0 - display_score[update_mask], torch.zeros_like(display_score[update_mask])), dim=1))
        #gaussians._opacity = torch.ones_like(gaussians.get_opacity)
        num_gaussians = len(display_score)
        #gaussians._features_dc[update_mask,0] = rgb_tensor
        gaussians._opacity[update_mask] = torch.min(gaussians._opacity)
        #gaussians._features_rest = torch.zeros_like(gaussians.get_sh_features_rest)


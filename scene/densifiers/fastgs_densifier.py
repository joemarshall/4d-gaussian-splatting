import torch
from .densifier_base import DensifierBase   

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.general_utils import inverse_sigmoid, build_rotation

class FastGSDensifier(DensifierBase):
    def __init__(self, opt):
        self.options = opt

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

    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg):
        num_cams = getattr(self.options, 'fastgs_num_sample_cams', 40)
        my_viewpoint_stack = scene.getTrainCameras()
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)
        importance_score, pruning_score = compute_gaussian_score_fastgs(
            camlist, gaussians, pipe, bg, self.options, DENSIFY=True
        )

        size_threshold = (
            20 if iteration > self.options.opacity_reset_interval else None
        )

        self._densify_and_prune_fastgs(gaussians,
            max_screen_size=size_threshold,
            min_opacity=self.options.thresh_opa_prune,
            extent=scene.cameras_extent,
            radii=radii,
            args=self.options,
            importance_score=importance_score,
            pruning_score=pruning_score,
        )


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
        # FastGS final-stage pruning (runs after the main densification phase).
        fastgs_final_from = getattr(self.options, 'fastgs_final_prune_from_iter', -1)
        fastgs_final_interval = getattr(self.options, 'fastgs_final_prune_interval', 3000)
        fastgs_final_until = getattr(self.options, 'fastgs_final_prune_until_iter', 30000)
        if (
            iteration >= fastgs_final_from and
            iteration % fastgs_final_interval == 0 and
            iteration <= fastgs_final_until
        ):
            print("Running FastGS final pruning at iteration {}...".format(iteration))
            num_cams = getattr(self.options, 'fastgs_final_num_sample_cams', 40)
            my_viewpoint_stack = scene.getTrainCameras()
            camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)
            importance_score, pruning_score = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, self.options, DENSIFY=False
            )
            min_opacity = getattr(self.options, "fastgs_final_prune_min_opacity", 0.1)

            self._final_prune_fastgs(gaussians,
                min_opacity=min_opacity,
                pruning_score=pruning_score,
            )


    def get_save_vars(self,gaussians):
        """Return variables that should be saved or loaded with the scene, as attribute names """
        attrs = ["xyz_gradient_accum", "xyz_gradient_accum_abs", "denom", "max_radii2D"]
        if gaussians.gaussian_dim == 4:
            attrs.append("t_gradient_accum")
        return attrs

    def _densify_and_clone_fastgs(self, gaussians,metric_mask, filter_mask):
        """Clone Gaussians that satisfy both the gradient filter and the
        multi-view metric mask (FastGS criterion)."""
        selected_pts_mask = torch.logical_and(metric_mask, filter_mask)
        print("densify and clone:",selected_pts_mask.sum(),"/",len(selected_pts_mask))
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
    

    def _densify_and_split_fastgs(self, gaussians, metric_mask, filter_mask, N=2):
        """Split Gaussians that satisfy both the gradient filter and the
        multi-view metric mask (FastGS criterion)."""
        n_init_points = gaussians.get_xyz.shape[0]

        selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
        combined = torch.logical_and(metric_mask, filter_mask)
        selected_pts_mask[:combined.shape[0]] = combined

        print("densify and split:",selected_pts_mask.sum(),"/",len(selected_pts_mask))


        stds = gaussians.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = gaussians.scaling_inverse_activation(
            gaussians.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = gaussians._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = gaussians._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = gaussians._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = gaussians._opacity[selected_pts_mask].repeat(N, 1)
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if gaussians.gaussian_dim == 4:
            new_t = gaussians._t[selected_pts_mask].repeat(N, 1)
            new_scaling_t = gaussians.scaling_inverse_activation(
                gaussians.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
            if gaussians.rot_4d:
                new_rotation_r = gaussians._rotation_r[selected_pts_mask].repeat(N, 1)

        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            new_t, new_scaling_t, new_rotation_r,
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool),
        ))
        gaussians.prune_points(prune_filter)

    def _densify_and_prune_fastgs(self, gaussians, max_screen_size, min_opacity, extent, radii,
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
        print(f"Scales{gaussians.get_scaling.shape}:",gaussians.get_scaling)

        clone_qualifiers = torch.max(gaussians.get_scaling, dim=1).values <= args.percent_dense * extent
        split_qualifiers = torch.max(gaussians.get_scaling, dim=1).values > args.percent_dense * extent



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
            metric_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")


        print("Densify:",metric_mask.sum())

        self._densify_and_clone_fastgs(gaussians,metric_mask, all_clones)
        self._densify_and_split_fastgs(gaussians,metric_mask, all_splits)

        # some points won't be visible at this timestep at all
        # - don't mess with them at all
        # 
#        visible_points_mask  =(importance_score>0)
        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
#            prune_mask = torch.logical_and(prune_mask,visible_points_mask)

        print("**************************************")
        print("Pruning:",prune_mask.sum(),"/",len(prune_mask))
        if pruning_score is not None:
            scores = 1.0 - pruning_score
            to_remove = torch.sum(prune_mask)
            # Only remove 50 % of eligible Gaussians per step (budget-controlled
            # pruning from FastGS) to avoid over-aggressive scene degradation.
            remove_budget = int(0.5 * to_remove)
            print("pruning, remove budget: ", remove_budget, " to_remove: ", to_remove)
            if remove_budget > 0:
                n_init_points = gaussians.get_xyz.shape[0]
                padded_importance = torch.zeros(n_init_points, dtype=torch.float32, device="cuda")
                n = min(scores.shape[0], n_init_points)
                padded_importance[:n] = 1.0 / (1e-6 + scores[:n].squeeze())
                selected_pts_mask = torch.zeros(n_init_points, dtype=torch.bool, device="cuda")
                sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                gaussians.prune_points(final_prune)
            else:
                gaussians.prune_points(final_prune)

        else:
            gaussians.prune_points(final_prune)

        # Clamp opacities to avoid them exploding after densification.
        opacities_new = inverse_sigmoid(
            torch.min(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * 0.8)
        )
        optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
        gaussians._opacity = optimizable_tensors["opacity"]
        # empty cache - do we need to do this still?
        torch.cuda.empty_cache()

    def _final_prune_fastgs(self, gaussians,min_opacity, pruning_score=None):
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
        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if pruning_score is not None:
            n = min(pruning_score.shape[0], gaussians.get_xyz.shape[0])
            scores_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            # Prune Gaussians with very high reconstruction inconsistency score
            # (top 10 % of the [0, 1] range, i.e., score > 0.9).
            scores_mask[:n] = pruning_score[:n].squeeze() > 0.9
            prune_mask = torch.logical_or(prune_mask, scores_mask)
        print("FastGS final prune: removed {} points. Remaining: {}".format(
            prune_mask.sum(), gaussians.get_xyz.shape[0]))
        
    def densification_postfix(self, gaussians):
        self.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        if self.t_gradient_accum is not None:
            self.t_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def prune_points(self, prune_mask):
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
        num_cams = 100
        my_viewpoint_stack = scene.getTrainCameras()
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)
        print(len(camlist))
        if debug_type == "flagged_errors":
            display_score, _ = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, self.options, DENSIFY=True
            )
        elif debug_type == "multiview_importance":
            _, display_score = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, self.options, DENSIFY=False
            )
        print(display_score.shape)
        display_score = display_score.float()
        display_score -= display_score.min()
        display_score /= display_score.max()
        display_score*=5.0
        display_score = torch.clamp(display_score, 0.0, 1.0)
        print(torch.max(display_score), torch.min(display_score))
        rgb_tensor = RGB2SH(torch.stack((display_score, 1.0 - display_score, torch.zeros_like(display_score)), dim=1))
        #gaussians._opacity = torch.ones_like(gaussians.get_opacity)
        num_gaussians = len(display_score)
        gaussians._features_dc = rgb_tensor
        gaussians._features_rest = torch.zeros_like(gaussians.get_sh_features_rest)


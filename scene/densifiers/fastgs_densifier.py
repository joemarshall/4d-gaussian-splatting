import torch
from .split_ops import *
from .densifier_base import DensifierBase   

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras

# Then think about depth consistency pruning somehow

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

        size_threshold = (
            20 if iteration > self.options.opacity_reset_interval else None
        )

        self._densify_and_prune_fastgs(gaussians=gaussians,
            max_screen_size=size_threshold,
            min_opacity=self.options.thresh_opa_prune,
            extent=scene.cameras_extent,
            radii=radii,
            args=self.options,
            camlist = camlist,
             pipe=pipe, bg=bg
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
            for frame_cams in camlist:
                _, pruning_score = compute_gaussian_score_fastgs(
                    frame_cams, gaussians, pipe, bg, self.options, DENSIFY=False
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


    def _densify_and_prune_fastgs(self, *,camlist, pipe, bg, gaussians, max_screen_size, min_opacity, extent, radii,
                                  args):
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

        clone_qualifiers = torch.max(gaussians.get_scaling, dim=1).values <= args.percent_dense * extent
        split_qualifiers = torch.max(gaussians.get_scaling, dim=1).values > args.percent_dense * extent


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

        prune_budget = int(prune_mask.sum() * 0.5)
        per_frame_prune_budget = (prune_budget // len(camlist))+1

        for frame_cameras in camlist:
            importance_score, pruning_score = compute_gaussian_score_fastgs(
                frame_cameras, gaussians, pipe, bg, self.options, DENSIFY=True
            )


            # Gaussians must appear in > fastgs_importance_threshold views to
            # be considered for densification.  Default of 5 comes from the
            # FastGS paper (multi-view consistency with 10 sampled views).
            importance_threshold = getattr(args, 'fastgs_importance_threshold', 5)
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
            num_possible = (pruning_score > 0.0)
            print("Poss prune:",num_possible.sum())
            scores = 1.0 - pruning_score
            # now score = 1 = no error, 0 = high error 
            sampling_importance = 1.0 / (1e-6 + scores.squeeze())
            # importance = 1/ 1.000001 for no error
            # 1/0.000001 = 1 million for high error
            sampled_indices = torch.multinomial(sampling_importance, per_frame_prune_budget, replacement=False)
            prune_mask_frame = torch.zeros_like(prune_mask)
            prune_mask_frame[sampled_indices] = prune_mask[sampled_indices]
            if final_prune is None:
                final_prune = prune_mask_frame
            else:
                final_prune|= prune_mask_frame

            # anything pruned can't be cloned or split in later steps
            all_splits = torch.logical_and(all_splits, ~final_prune)
            all_clones = torch.logical_and(all_clones, ~final_prune)

        final_clones = torch.logical_and(final_clones, ~final_prune)
        final_splits = torch.logical_and(final_splits, ~final_prune)


        print("\n=== FastGS Densification/Pruning Summary ===")
        print("FastGS prune budget: {} points ({} per frame)".format(prune_budget, per_frame_prune_budget))
        print("FastGS [{} frames]: selected {} points for cloning, {} for splitting, {} for pruning.".format(
            len(camlist), final_clones.sum(), final_splits.sum(), final_prune.sum()
        ))

        # now do actual densify, split etc.
        densify_and_clone(gaussians,final_clones)

        final_splits = torch.cat(
            [final_splits, torch.zeros((gaussians.get_xyz.shape[0] - final_splits.shape[0]), device=final_splits.device, dtype=torch.bool)
                ]
        )
        densify_and_split(gaussians,final_splits)

        final_prune = torch.cat(
            [final_prune, torch.zeros((gaussians.get_xyz.shape[0] - final_prune.shape[0]), device=final_prune.device, dtype=torch.bool)
                ]
        )
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
        num_cams = 10
        my_viewpoint_stack = scene.getTrainCameras()
        camlist = sampling_cameras(my_viewpoint_stack, num_cams,dimensions=4)
        print(len(camlist))
        if debug_type == "flagged_errors":
            display_score, _ = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, self.options, DENSIFY=True
            )
            display_score = display_score.float()
        elif debug_type == "multiview_importance":
            _, display_score = compute_gaussian_score_fastgs(
                camlist, gaussians, pipe, bg, self.options, DENSIFY=False
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


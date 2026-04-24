import math

from numpy import power
from numpy import power
import torch

from .densifier_base import DensifierBase
from utils.general_utils import build_rotation, build_rotation_4d
from utils.graphics_utils import fov2focal


class LFFDensifier(DensifierBase):
    """LFF-style densification strategy adapted from EFA-GS 3DGS implementation."""

    def __init__(self, opt):
        super().__init__(opt, "lff")
        self.split_multiplier = getattr(opt, "lff_split_multiplier", 2.0)
        self.new_selected_mask = None

    def training_setup(self, gaussians, reset_accumulated_gradients=True):
        if reset_accumulated_gradients:
            num_gaussians = gaussians.get_xyz.shape[0]
            self.lff_xyz_grad_accum = torch.zeros((num_gaussians, 1), device="cuda")
            self.prev_lff_xyz_grad = torch.zeros((num_gaussians, 1), device="cuda")
            self.lff_denom = torch.zeros((num_gaussians, 1), device="cuda")
            self.max_radii2D = torch.zeros((num_gaussians), device="cuda")
            self.prev_selected_pts_mask = self._convert_bool_to_int(
                torch.ones(num_gaussians, dtype=torch.bool, device="cuda")
            )
        self.new_selected_mask = None            

    def get_save_vars(self, gaussians):
        return [
            "lff_xyz_grad_accum",
            "prev_lff_xyz_grad",
            "lff_denom",
            "max_radii2D",
            "prev_selected_pts_mask",
        ]

    def add_densification_stats_grad(
        self,
        *,
        gaussians,
        iteration,
        viewspace_point_grad,
        update_filter,
        radii,
        avg_t_gradient,
    ):
        grad_norm = torch.norm(viewspace_point_grad[update_filter], dim=-1, keepdim=True)
        self.lff_xyz_grad_accum[update_filter] += grad_norm
        self.lff_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter],
            radii[update_filter],
        )

    def densify_and_prune(self, iteration, scene, gaussians, radii, pipe, bg, *, options):
        max_grad = self.options.densify_grad_threshold
        min_opacity = self.options.thresh_opa_prune
        extent = scene.cameras_extent
        max_screen_size = (
            20 if iteration > self.options.opacity_reset_interval else None
        )

        cameras_no_images = scene.getTrainCameras()
        cameras_no_images.set_names_only(True)
        ts_zero = cameras_no_images.get_timestamps()[0]
        cameras = [cameras_no_images[c][1] for c in cameras_no_images.get_indices_for_timestamp(ts_zero)]


        def calculate_training_percent_powered(iter, min_densify_iter, max_densify_iter, power=1.0, lower_bound=0):
            #   iter:min_densify_iter -> max_densify_iter
            #     di:               1 ->                0
            # pow_di:               1 ->      lower_bound
            di = 1 - (iter - min_densify_iter) / (max_densify_iter - min_densify_iter)
            pow_di = math.exp(math.log(di)*power) * (1 - lower_bound) + lower_bound
            return pow_di

        percent_lb = 0.0
        power = 1.0
        training_percent_powered = calculate_training_percent_powered(iteration, self.options.densify_from_iter, self.options.densify_until_iter, power, percent_lb)
        print(f"Percent powered for this iteration: {iteration} = {training_percent_powered}")
                    # More training, lower splitting_lb
        splitting_lb = 1 - (iteration - self.options.densify_from_iter) / (self.options.densify_until_iter - self.options.densify_from_iter)
        self._densify_and_prune_lff(
            gaussians,
            max_grad=max_grad,
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen_size,
            cameras=cameras,
            N=getattr(self.options, "lff_num_split_samples", 1),
            scaling_multiplier_max=getattr(self.options, "lff_scaling_multiplier_max", 1.0),
            scaling_multiplier_min=getattr(self.options, "lff_scaling_multiplier_min", 1.0),
            training_percent_powered=training_percent_powered,
            splitting_ub=getattr(self.options, "lff_splitting_ub", 1.0),
            splitting_lb=splitting_lb,
            tolerance=getattr(self.options, "lff_tolerance", 1e-5),
            diffscale=getattr(self.options, "lff_diffscale", True),
        )

    @staticmethod
    def _convert_bool_to_int(bool_indices):
        return torch.nonzero(bool_indices, as_tuple=False).flatten()

    @staticmethod
    def _convert_int_to_bool(int_indices, dim):
        bool_indices = torch.zeros(dim, dtype=torch.bool, device=int_indices.device)
        if int_indices.numel() > 0:
            bool_indices[int_indices.flatten()] = True
        return bool_indices

    @staticmethod
    def _normalize_interval(interval, opt="log", norm="minmax"):
        if opt == "exp":
            interval = torch.exp(interval)
        elif opt == "log":
            interval = torch.log(interval)

        if norm == "minmax":
            min_interval = torch.min(interval)
            max_interval = torch.max(interval)
            denom = (max_interval - min_interval).clamp_min(1e-8)
            return (interval - min_interval) / denom
        if norm == "one":
            return interval / interval.sum().clamp_min(1e-8)
        return interval

    def _camera_focal_x(self, camera):
        if hasattr(camera, "focal_x") and camera.focal_x > 0:
            return float(camera.focal_x)
        if hasattr(camera, "fl_x") and camera.fl_x > 0:
            return float(camera.fl_x)
        return float(fov2focal(camera.FoVx, camera.image_width))

    @torch.no_grad()
    def _compute_3d_interval(self, gaussians, cameras):
        xyz = gaussians.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        focal_length = 0.0
        for camera in cameras:
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            xyz_cam = xyz @ R + T[None, :]

            valid_depth = xyz_cam[:, 2] > 0.2

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            focal_x = self._camera_focal_x(camera)
            focal_y = focal_x
            if hasattr(camera, "focal_y") and camera.focal_y > 0:
                focal_y = float(camera.focal_y)
            elif hasattr(camera, "fl_y") and camera.fl_y > 0:
                focal_y = float(camera.fl_y)

            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0

            in_screen = torch.logical_and(
                torch.logical_and(
                    x >= -0.15 * camera.image_width,
                    x <= camera.image_width * 1.15,
                ),
                torch.logical_and(
                    y >= -0.15 * camera.image_height,
                    y <= 1.15 * camera.image_height,
                ),
            )

            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            focal_length = max(focal_length, focal_x)

        if torch.any(valid_points):
            distance[~valid_points] = distance[valid_points].max()
        else:
            distance[:] = 1.0
        if focal_length <= 0:
            focal_length = 1.0

        interval = distance / focal_length
        return interval[..., None]

    @torch.no_grad()
    def _set_attributes(self, gaussians, attribute, mask, changes):
        assert attribute in ["xyz", "scaling", "opacity", "rotation"]
        if changes.shape[0] != torch.sum(mask).item():
            raise ValueError("The numbers of changes and mask do not match.")
        real_attribute = "_" + attribute
        if changes.numel() == 0:
            return
        getattr(gaussians, real_attribute)[mask] += changes.type(torch.float32).to(
            gaussians._scaling.device
        )

    def densification_postfix(self, gaussians):
        is_lff = self.new_selected_mask is not None

        self.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        new_num_gaussians = gaussians.get_xyz.shape[0]
        old_num_gaussians = self.prev_lff_xyz_grad.shape[0]
        new_zeros = torch.zeros((new_num_gaussians-old_num_gaussians, 1), device="cuda")

        if is_lff:
            prev_lff_xyz_grad = self.lff_xyz_grad_accum / self.lff_denom
            prev_lff_xyz_grad[prev_lff_xyz_grad.isnan()] = 0
            self.prev_lff_xyz_grad = torch.concat([prev_lff_xyz_grad, new_zeros])
            self.lff_xyz_grad_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            self.lff_denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            prev_selected_pts_mask = torch.concat(
                [
                    self.new_selected_mask,
                    new_zeros.squeeze()
                    #torch.ones(new_zeros.shape[0], dtype=torch.bool, device=self.new_selected_mask.device),
                ]
            )
        else:
            prev_selected_pts_mask = self._convert_int_to_bool(
                self.prev_selected_pts_mask,
                dim=self.prev_lff_xyz_grad.shape[0],
            )
            self.prev_lff_xyz_grad = torch.concat([self.prev_lff_xyz_grad, new_zeros])
            self.lff_xyz_grad_accum = torch.concat([self.lff_xyz_grad_accum, new_zeros])
            self.lff_denom = torch.concat([self.lff_denom, new_zeros])
            prev_selected_pts_mask = torch.concat(
                [
                    prev_selected_pts_mask,
                    new_zeros.squeeze()
#                    torch.ones(new_zeros.shape[0], dtype=torch.bool, device=prev_selected_pts_mask.device),
                ]
            )

        self.prev_selected_pts_mask = self._convert_bool_to_int(prev_selected_pts_mask)
        self.new_selected_mask = None


    def prune_points(self, gaussians, prune_mask):
        valid_points_mask = ~prune_mask
        prev_selected_pts_mask = self._convert_int_to_bool(
            self.prev_selected_pts_mask,
            dim=valid_points_mask.shape[0],
        )
        valid_prev_selected_pts_mask = prev_selected_pts_mask[valid_points_mask]
        self.prev_selected_pts_mask = self._convert_bool_to_int(valid_prev_selected_pts_mask)

        self.lff_denom = self.lff_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.lff_xyz_grad_accum = self.lff_xyz_grad_accum[valid_points_mask]
        self.prev_lff_xyz_grad = self.prev_lff_xyz_grad[valid_points_mask]

    def _densify_lff(
        self,
        gaussians,
        grads,
        grad_threshold,
        cameras,
        N=1,
        scaling_multiplier_max=1.0,
        scaling_multiplier_min=1.0,
        training_percent_powered=0.0,
        splitting_ub=1.0,
        splitting_lb=0.5,
        tolerance=1e-5,
        diffscale=True,
    ):
        assert scaling_multiplier_max >= 1.0
        assert scaling_multiplier_min >= 1.0
        assert 0.0 <= training_percent_powered <= 1.0

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        prev_selected_pts_mask = self._convert_int_to_bool(
            self.prev_selected_pts_mask,
            gaussians.get_xyz.shape[0],
        )

        intersect_selected_pts_mask = torch.logical_and(prev_selected_pts_mask, selected_pts_mask)
        enlarged_mask = torch.logical_and(torch.logical_not(prev_selected_pts_mask), selected_pts_mask)
        splitted_mask = torch.zeros_like(enlarged_mask, dtype=torch.bool)

        whether_descending = torch.where(
            torch.norm(grads[intersect_selected_pts_mask], dim=-1)
            <= torch.norm(self.prev_lff_xyz_grad[intersect_selected_pts_mask] - tolerance, dim=-1),
            True,
            False,
        )
        enlarged_mask[intersect_selected_pts_mask] = torch.logical_not(whether_descending)
        splitted_mask[intersect_selected_pts_mask] = whether_descending

        enlarged_scaling_changes = torch.zeros_like(
            gaussians._scaling[enlarged_mask],
            dtype=torch.float32,
            device=gaussians._scaling.device,
        )

        interval = self._compute_3d_interval(gaussians, cameras)
        with torch.no_grad():
            interval_coef = self._normalize_interval(interval, "log", "minmax").to(gaussians._scaling.device)
            scaling_multiplier_coef = interval_coef * scaling_multiplier_min + (1 - interval_coef) * scaling_multiplier_max
            log_scaling_multiplier = torch.log(scaling_multiplier_coef) * training_percent_powered

            if diffscale:
                coef_of_enlarge = torch.ones_like(enlarged_scaling_changes)
                if enlarged_scaling_changes.shape[0] > 0:
                    enlarged_sorted_indices = torch.sort(
                        gaussians._scaling[enlarged_mask],
                        dim=1,
                        descending=False,
                    )[1]
                    coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:, 1].unsqueeze(1), -1.0 / 3)
                    coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:, 2].unsqueeze(1), -2.0 / 3)
                enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask] * coef_of_enlarge
            else:
                enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask]

        self._set_attributes(gaussians, "scaling", enlarged_mask, enlarged_scaling_changes)
        enlarged_num = enlarged_scaling_changes.shape[0]

        splitted_scaling_changes = torch.zeros_like(
            gaussians._scaling[splitted_mask],
            dtype=torch.float32,
            device=gaussians._scaling.device,
        )

        with torch.no_grad():
            if diffscale:
                coef_of_split = torch.ones_like(splitted_scaling_changes)
                if splitted_scaling_changes.shape[0] > 0:
                    splitted_sorted_indices = torch.sort(
                        gaussians._scaling[splitted_mask],
                        dim=1,
                        descending=True,
                    )[1]
                    coef_of_split.scatter_(1, splitted_sorted_indices[:, 1].unsqueeze(1), 0.5)
                    coef_of_split.scatter_(1, splitted_sorted_indices[:, 2].unsqueeze(1), 0.0)
                split_term = 0.5 * (
                    log_scaling_multiplier[splitted_mask]
                    + math.log(self.split_multiplier)
                )
                splitted_scaling_changes -= split_term * coef_of_split
            else:
                splitted_scaling_changes -= 0.5 * (
                    log_scaling_multiplier[splitted_mask]
                    + math.log(self.split_multiplier)
                )

        self._set_attributes(gaussians, "scaling", splitted_mask, splitted_scaling_changes)

        splitting_prob_threshold = interval_coef * (splitting_ub - splitting_lb) + splitting_lb
        whether_split = torch.where(
            torch.rand_like(splitting_prob_threshold[intersect_selected_pts_mask])
            <= splitting_prob_threshold[intersect_selected_pts_mask],
            True,
            False,
        )
        if whether_split.numel() > 0:
            splitted_mask[intersect_selected_pts_mask] = torch.logical_and(
                splitted_mask[intersect_selected_pts_mask],
                whether_split.squeeze(),
            )



        if not gaussians.rot_4d:
            stds = gaussians.get_scaling[splitted_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(gaussians._rotation[splitted_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[splitted_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if gaussians.gaussian_dim == 4:
                stds_t = gaussians.get_scaling_t[splitted_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + gaussians.get_t[splitted_mask].repeat(N, 1)
                new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[splitted_mask].repeat(N,1) / (0.8*N))
        else:
            stds = gaussians.get_scaling_xyzt[splitted_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(gaussians._rotation[splitted_mask], gaussians._rotation_r[splitted_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyzt[splitted_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = gaussians.scaling_inverse_activation(gaussians.get_scaling_t[splitted_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = gaussians._rotation_r[splitted_mask].repeat(N,1)



        # stds = gaussians.get_scaling[splitted_mask].repeat(N, 1)
        # stds[stds.isnan()] = 1e-3
        # means = torch.zeros((stds.size(0), 3), device="cuda")
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(gaussians._rotation[splitted_mask]).repeat(N, 1, 1)


        new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[splitted_mask].repeat(N,1) / (0.8*N))
        new_rotation = gaussians._rotation[splitted_mask].repeat(N,1)
        new_features_dc = gaussians._features_dc[splitted_mask].repeat(N,1,1)
        new_features_rest = gaussians._features_rest[splitted_mask].repeat(N,1,1)
        new_opacity = gaussians._opacity[splitted_mask].repeat(N,1)


        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.get_xyz[splitted_mask].repeat(N, 1)
        # new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[splitted_mask].repeat(N, 1))
        # new_rotation = gaussians._rotation[splitted_mask].repeat(N, 1)
        # new_features_dc = gaussians._features_dc[splitted_mask].repeat(N, 1, 1)
        # new_features_rest = gaussians._features_rest[splitted_mask].repeat(N, 1, 1)
        # new_opacity = gaussians._opacity[splitted_mask].repeat(N, 1)

        # new_t = None
        # new_scaling_t = None
        # new_rotation_r = None
        # if gaussians.gaussian_dim == 4:
        #     new_t = gaussians._t[splitted_mask].repeat(N, 1)
        #     new_scaling_t = gaussians._scaling_t[splitted_mask].repeat(N, 1)
        #     if gaussians.rot_4d:
        #         new_rotation_r = gaussians._rotation_r[splitted_mask].repeat(N, 1)

        self.new_selected_mask = selected_pts_mask

        gaussians.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_t,
            new_scaling_t,
            new_rotation_r,
        )



        splitted_num = new_xyz.shape[0]

        print(f"LFF Densify and prune: {splitted_num} points splitted, {enlarged_num} points enlarged. Total after densification: {gaussians.get_xyz.shape[0]}.") 

        return enlarged_num, splitted_num

    def _get_nan_mask(self, gaussians):
        nan_mask = torch.logical_or(
            torch.any(gaussians.get_xyz.isnan(), dim=1),
            torch.any(gaussians.get_scaling.isnan(), dim=1),
        )
        nan_mask = torch.logical_or(nan_mask, torch.any(gaussians.get_rotation.isnan(), dim=1))
        nan_mask = torch.logical_or(nan_mask, torch.any(gaussians.get_opacity.isnan(), dim=1))
        nan_mask = torch.logical_or(
            nan_mask,
            torch.any(gaussians.get_features.isnan().reshape(gaussians.get_features.shape[0], -1), dim=1),
        )
        return nan_mask

    def reset_prev_grad(self, gaussians):
        self.prev_lff_xyz_grad = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def _densify_and_prune_lff(
        self,
        gaussians,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        cameras,
        N=1,
        scaling_multiplier_max=1.0,
        scaling_multiplier_min=1.0,
        training_percent_powered=0.0,
        splitting_ub=1.0,
        splitting_lb=0.5,
        tolerance=1e-5,
        diffscale=True,
    ):
        
        print(f"Training percent powered: {training_percent_powered}, splitting_lb: {splitting_lb} splitting_ub: {splitting_ub}, scaling_multiplier_min: {scaling_multiplier_min}, scaling_multiplier_max: {scaling_multiplier_max}")

        assert splitting_ub >= splitting_lb
        assert scaling_multiplier_max >= scaling_multiplier_min


        grads = self.lff_xyz_grad_accum / self.lff_denom
        grads[grads.isnan()] = 0.0

        self._densify_lff(
            gaussians,
            grads,
            max_grad,
            cameras,
            N,
            scaling_multiplier_max,
            scaling_multiplier_min,
            training_percent_powered,
            splitting_ub,
            splitting_lb,
            tolerance,
            diffscale,
        )

        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        nan_mask = self._get_nan_mask(gaussians)
        if nan_mask.sum() > 0:
            print(f"During densify_and_prune_lff stage, {nan_mask.sum()} points are NaN.")
        prune_mask = torch.logical_or(prune_mask, nan_mask)

        gaussians.prune_points(prune_mask)
        torch.cuda.empty_cache()
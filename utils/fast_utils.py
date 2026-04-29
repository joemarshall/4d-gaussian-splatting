"""FastGS utility functions for multi-view consistent densification and pruning.

This module implements the multi-view consistency scoring approach from FastGS
(https://github.com/fastgs/FastGS) adapted for use with 4D Gaussian Splatting.
The key idea is to render the scene from multiple sampled views, compute a
per-pixel photometric loss, and use that to score each Gaussian based on how
many high-error pixels it contributed to.  High-scoring Gaussians are good
candidates for densification, while low-scoring ones may be pruned.
"""

import random
import torch
from gaussian_renderer import calculate_gaussian_contribution, render
from .loss_utils import l1_loss, ssim
import numpy as np


def sampling_cameras(viewpoint_stack, num_cams=10, dimensions=3):
    """Randomly sample cameras from the viewpoint stack and copy.

    Args:
        viewpoint_stack (list): list of camera objects.
        num_cams (int): number of cameras to sample (default 10).

    Returns:
        list: sampled camera objects.
    """
    camlist = []
    if dimensions == 3:
        first_list = []
        num_cams = min(num_cams, len(viewpoint_stack))
        indices = np.random.permutation(len(viewpoint_stack))
        for i in indices[:num_cams]:
            first_list.append(viewpoint_stack[i])
        camlist.append(first_list)
    else:
        # 4d gaussians - sample frames from the same time point
        # to get candidates for pruning and densification etc.
        # as multi-view error doesn't make so much sense otherwise
        # n.b. if num_cams > cameras in a single frame then multiple frames will be
        # sampled
        frames = viewpoint_stack.get_timestamps()
        ts = np.array(frames)
        np.random.shuffle(ts)
        total_cams = 0
        while total_cams < num_cams and len(ts) > 0:
            frame_list = []
            timestamp = ts[0]
            ts = ts[1:]
            num_left = num_cams - total_cams
            indices = viewpoint_stack.get_indices_for_timestamp(timestamp)
            for i in indices[:num_left]:
                frame_list.append(viewpoint_stack[i])
                total_cams += 1
            camlist.append(frame_list)
    return camlist


def _get_loss_map(render_image, gt_image):
    """Compute a normalised per-pixel L1 loss map.

    Args:
        render_image: rendered image tensor (C, H, W).
        gt_image: ground-truth image tensor (C, H, W).

    Returns:
        Tensor: normalised L1 loss map of shape (H*W,) with values in [0, 1].
    """
    l1 = torch.mean(torch.abs(render_image - gt_image), 0).detach()
    l1_min = torch.min(l1)
    l1_max = torch.max(l1)
    if l1_max > l1_min:
        l1_norm = (l1 - l1_min) / (l1_max - l1_min)
    else:
        l1_norm = torch.zeros_like(l1)
    return l1_norm.view(-1)


def _compute_photometric_loss(viewpoint_cam, image, lambda_dssim=0.2):
    """Compute the photometric (L1 + optional SSIM) loss for a single view.

    Args:
        viewpoint_cam: camera with ``original_image`` attribute.
        image: rendered image tensor (C, H, W) on CUDA.
        lambda_dssim (float): weight of SSIM term; set to 0 to use pure L1.

    Returns:
        Tensor: per-pixel photometric loss of shape (H, W).
    """
    gt_image = viewpoint_cam.original_image.cuda()
    return l1_loss(image, gt_image)


def compute_visible_gaussians(camlist, gaussians, pipe, bg, args):
    """Compute a count of how many training views each Gaussians is in

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: :class:`~scene.gaussian_model.GaussianModel`
        pipe: rendering pipeline parameters (``pipe.debug``, etc.).
        bg: background colour tensor on CUDA.

    Returns:
        Tensor: per-Gaussian integer counts of how many views flagged the Gaussian as visible.
    """
    full_metric_counts = None
    fastgs_mult = getattr(args, "fastgs_mult", getattr(args, "mult", 0.5))

    for view in range(len(camlist)):
        # calculateGaussianVisibilityContribution(camlist[view][1], gaussians, pipe, bg, fastgs_mult)

        gt_image, viewpoint_cam = camlist[view]
        viewpoint_cam = viewpoint_cam.cuda()
        gt_image = gt_image.detach()

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        render_image = render_pkg["render"]

        gt_image = gt_image.cuda()

        photometric_loss = l1_loss(gt_image, render_image)

        l1_norm = _get_loss_map(render_image, gt_image)
        metric_map = (l1_norm > 0.1).to(torch.int32)

        render_pkg2 = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            metric_map=metric_map,
        )
        accum_loss_counts = render_pkg2["accum_metric_counts"]

        if full_metric_counts is None:
            full_metric_counts = accum_loss_counts.clone()
        else:
            full_metric_counts = full_metric_counts + accum_loss_counts

    return full_metric_counts


def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, loss_thresh, DENSIFY=False):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in *camlist* the function renders the scene with the FastGS
    engine, computes a per-pixel photometric loss and a binary metric map of
    high-error pixels, then accumulates per-Gaussian counts of how many views
    flagged the Gaussian and a weighted photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: :class:`~scene.gaussian_model.GaussianModel` instance.
        pipe: rendering pipeline parameters (``pipe.debug``, etc.).
        bg: background colour tensor on CUDA.
        args: runtime config.  Must provide:
            * ``args.loss_thresh`` – pixel L1-loss threshold for the metric map.
            * ``getattr(args, 'mult', getattr(args, 'fastgs_mult', 0.5))`` – AccuTile multiplier passed to FastGS rasterizer.
        DENSIFY (bool): if True also compute and return *importance_score*.

    Returns:
        tuple:
            * **importance_score** (*Tensor* or *None*): per-Gaussian integer
              counts of how many views flagged the Gaussian (floor-divided by the
              number of views).  Only returned when ``DENSIFY=True``.
            * **pruning_score** (*Tensor*): per-Gaussian score in [0, 1] used to
              prioritise pruning (higher → worse multi-view consistency).
    """
    # print(f"Computing FastGS scores with { len(camlist) } cameras:")
    # print("*************************************")
    # for x in camlist:
    #     print(x[1].image_name,",",end="")
    # print("")

    # print("*************************************")

    full_metric_counts = None
    full_metric_score = None


    for view in range(len(camlist)):
        gt_image, viewpoint_cam = camlist[view]
        viewpoint_cam = viewpoint_cam.cuda()
        gt_image = gt_image.detach()

        # First render: get the rendered image and the plain photometric loss.
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        render_image = render_pkg["render"]

        gt_image = gt_image.cuda()

        photometric_loss = l1_loss(gt_image, render_image)*0.8 + ssim(gt_image,render_image)*0.2

        # Build binary metric map: 1 where per-pixel error exceeds threshold.
        l1_norm = _get_loss_map(render_image, gt_image)
        metric_map = (l1_norm > loss_thresh).to(torch.int32)
        # Second render: accumulate per-Gaussian metric counts via get_flag.
        render_pkg2 = render(viewpoint_cam, gaussians, pipe, bg, metric_map=metric_map)
        accum_loss_counts = render_pkg2["metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts = full_metric_counts + accum_loss_counts

        weighted = photometric_loss * accum_loss_counts.float()
        if full_metric_score is None:
            full_metric_score = weighted
        else:
            full_metric_score = full_metric_score + weighted

    # Normalise the pruning score to [0, 1].
    score_min = torch.min(full_metric_score)
    score_max = torch.max(full_metric_score)
    if score_max > score_min:
        pruning_score = (full_metric_score - score_min) / (score_max - score_min)
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    if DENSIFY:
        importance_score = torch.div(
            full_metric_counts, len(camlist), rounding_mode="floor"
        )
    else:
        importance_score = None

    return importance_score, pruning_score

def calculate_per_gaussian_error_contribution(camlist, gaussians, pipe, bg):
    """
    Calculate a score for each Gaussian based on how many error pixels it contributed to across multiple views.
    - we do this by:
    1) render view vs ground truth to compute per pixel error map
    2) render again inverted, to get per gaussian contribution to each pixel and multiply the contribution by error map to
       sum the contribution of each gaussian to error pixels in the view.
       n.b. by rendering inverted, we can multply by opacity left to make contributions from front pixels more 
       important than back pixels
    """
    # render the views
    total_error = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
    total_visual = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)

    for view in range(len(camlist)):
        gt_image, viewpoint_cam = camlist[view]
        viewpoint_cam = viewpoint_cam.cuda()
        gt_image = gt_image.detach().cuda()

        # render forwards to get error per pixel
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        render_image = render_pkg["render"]

        # loss map = normalized l1 norm
        l1_norm = _get_loss_map(render_image, gt_image)

        # render backwards with ground-truth error map 
        # this will return 1) error_contribution for each gaussian
        # and 2) visual_contribution (how much it contributed weighted by opacity) 
        render_pkg2 = calculate_gaussian_contribution(viewpoint_cam,gaussians,pipe,bg, error_map = l1_norm)
        
        error_contribution = render_pkg2["error_contribution"]
        visual_contribution = render_pkg2["visual_contribution"]

        error_contribution = error_contribution / (visual_contribution.clamp(min=1e-6))

        total_error += error_contribution
        total_visual += visual_contribution

    return total_error, total_visual



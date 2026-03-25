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
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
import numpy as np


def sampling_cameras(viewpoint_stack, num_cams=10,dimensions = 3):
    """Randomly sample cameras from the viewpoint stack and copy.

    Args:
        viewpoint_stack (list): list of camera objects.  
        num_cams (int): number of cameras to sample (default 10).

    Returns:
        list: sampled camera objects.
    """
    camlist = []
    if dimensions==3:
        num_cams = min(num_cams, len(viewpoint_stack))
        indices = np.random.permutation(len(viewpoint_stack))
        for i in indices[:num_cams]:
            camlist.append(viewpoint_stack[i])
    else:
        # 4d gaussians - sample frames from the same time point
        # to get candidates for pruning and densification etc.
        # as multi-view error doesn't make so much sense otherwise
        # n.b. if num_cams > a single frame then multiple frames will be
        # sampled
        frames = viewpoint_stack.get_timestamps()
        ts = np.random.choice(frames,10)
        while len(camlist) < num_cams and len(ts)>0:
            timestamp = ts[0]
            ts = ts[1:]
            num_left = num_cams - len(camlist)
            indices = viewpoint_stack.get_indices_for_timestamp(timestamp)
            for i in indices[:num_left]:
                camlist.append( viewpoint_stack[i])
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


def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False):
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
    print("Computing FastGS scores with cameras:")
    print("*************************************")
    for x in camlist:
        print(x[1].image_name)
    full_metric_counts = None
    full_metric_score = None
    print("*************************************")

    # Read FastGS parameters with fallbacks for backward compatibility.
    fastgs_mult = getattr(args, 'fastgs_mult', getattr(args, 'mult', 0.5))
    loss_thresh = getattr(args, 'fastgs_loss_thresh', getattr(args, 'loss_thresh', 0.1))

    for view in range(len(camlist)):
        gt_image,viewpoint_cam = camlist[view]
        viewpoint_cam=viewpoint_cam.cuda()
        gt_image=gt_image.detach()


        # First render: get the rendered image and the plain photometric loss.
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, fastgs_mult)
        render_image = render_pkg["render"]

        gt_image = gt_image.cuda()

        photometric_loss =l1_loss(gt_image,render_image)

        # Build binary metric map: 1 where per-pixel error exceeds threshold.
        l1_norm = _get_loss_map(render_image, gt_image)
        metric_map = (l1_norm > loss_thresh).to(torch.int32)
        # Second render: accumulate per-Gaussian metric counts via get_flag.
        render_pkg2 = render_fastgs(
            viewpoint_cam, gaussians, pipe, bg, fastgs_mult,
            get_flag=True, metric_map=metric_map,
        )
        accum_loss_counts = render_pkg2["accum_metric_counts"]

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
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None

    return importance_score, pruning_score

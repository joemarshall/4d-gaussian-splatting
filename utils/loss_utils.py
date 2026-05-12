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

import torch
import torch.nn.functional as F
from math import exp
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

def msssim(rgb, gts):
    # assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    # assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()

def combine_losses(losses_list,render_package,gt_image,gt_depth,gaussians,args,*,iterations = 0,max_iterations=30000,divisor = 1.0):
    weight_sum = 0
    loss_sum = 0
    if divisor>0:
        multiplier = 1.0/divisor
    else:
        multiplier = 1.0
    result_dict = {}
    for name,weight,loss_fn in losses_list:
        if weight >0:
            value = loss_fn(render_package, gt_image, gt_depth, gaussians,args,iterations,max_iterations)
            if value is not None:
                result_dict[name] = multiplier * value
                loss_sum += weight *result_dict[name]
        weight_sum+= weight
    if weight_sum > 0:
        loss_sum /= weight_sum
    result_dict["loss"] = loss_sum
    return result_dict

def loss_ssim(render_package, gt_image, gt_depth, gaussians,args, iterations,max_iterations):
    return 1.0 - ssim(render_package["render"], gt_image)

def loss_bistable_opacity(render_package, gt_image, gt_depth, gaussians,args,iterations,max_iterations):
    if gaussians.get_xyz.shape[0] < 50000:
        return None
    return torch.xlogy(-gaussians.get_opacity,gaussians.get_opacity).mean()

def loss_l1(render_package, gt_image, gt_depth, gaussians,args,iterations,max_iterations):
    return l1_loss(render_package["render"], gt_image)

def loss_depth(render_package, gt_image, gt_depth, gaussians,args,iterations,max_iterations):
    if iterations > max_iterations * 5/6:    
        return None


    depth_out = render_package["depth"].squeeze()
    gt_depth_mask = (gt_depth > 0) & (gt_depth < 100)




    max_gt_depths = torch.nn.MaxPool2d(3,stride=1,padding=1)(gt_depth.unsqueeze(0).unsqueeze(0)).squeeze()
    min_gt_depths = -torch.nn.MaxPool2d(3,stride=1,padding=1)(-gt_depth.unsqueeze(0).unsqueeze(0)).squeeze()
    diff_gt_depths =torch.abs(max_gt_depths - min_gt_depths)


    gt_depth_mask = (min_gt_depths > 0) & (max_gt_depths < 100)

    

    
    out_norm = depth_out[gt_depth_mask]
    out_norm = (out_norm - out_norm.min()) / (out_norm.max() - out_norm.min() + 1e-8)
    gt_norm = gt_depth[gt_depth_mask]
    gt_norm = (gt_norm - gt_norm.min()) / (gt_norm.max() - gt_norm.min() + 1e-8)

    depth_variability = diff_gt_depths[gt_depth_mask]
    diff_gt_depths/= depth_variability.max() + 1e-8
    diff_gt_depths[~gt_depth_mask] = 0.0
    diff_gt_depths = 1.0- diff_gt_depths





    # ignore < 1% errors

    diff = torch.abs((gt_norm - out_norm)*diff_gt_depths[gt_depth_mask])

    # l1 loss ignoring <1% errors
    diff = diff - 0.01
    diff = torch.clamp(diff, min=0.0)

#    print("Depth loss:", depth_out.shape,torch.max(depth_out).item(),torch.min(depth_out).item(), gt_depth.shape,torch.max(gt_depth).item(),torch.min(gt_depth[gt_depth_mask]).item())
    return diff.mean()
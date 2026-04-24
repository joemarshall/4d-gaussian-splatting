# TODO:
# pluggable prune / densifiers:
# prune floaters (at end?)
# prune invisibles (render outside in, frame by frame and remove totally invisibles)
# prune v short time span gaussians in 4d
# 
# better optimisation strategies (e.g. train over all frames at timepoint then backprop
# - using a loss which combines multiple frame info)


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
# MAKE IT LAUNCH THE VIEWER/ RENDERER in subprocess each 100 iterations or so# MAKE IT LAUNCH THE VIEWER/ RENDERER in subprocess each 100 iterations or so
from datetime import datetime
import os
import signal
import random
import torch
import subprocess
import threading
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim,l2_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from pathlib import Path
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid

from typing import List

from scene.densifiers import *

import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

# torch.use_deterministic_algorithms(True)
# torch.utils.deterministic.fill_uninitialized_memory=True

torch.set_float32_matmul_precision('high')
torch.backends.fp32_precision = "tf32"
torch._dynamo.config.force_parameter_static_shapes = False 


TRACK_MEMORY = False
CLEAR_CACHE = False        

if TRACK_MEMORY:
    torch.cuda.memory._record_memory_history(
        max_entries=1000000
        )



#from torch.utils.viz._cycles import warn_tensor_cycles
#warn_tensor_cycles()

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def viewer_thread(args,name):
    process = subprocess.run(args,capture_output=True,text=True)
    if process.returncode ==0:
        print(f"Written mp4 {name}")
    else:
        print(f"Error rendering video {name}: {process.stdout} {process.stderr}")


def launch_viewer(args, name):
   print("Render video in subprocess with args: {}".format(args))
   threading.Thread(target=viewer_thread, args=(args,name),daemon=True).start()



@torch.compile
def run_batch(batch_data, batch_size, gaussians, pipe, background, opt):
    
    batch_point_grad = []
    batch_visibility_filter = []
    batch_radii = []


    for batch_idx in range(batch_size):
        gt_image, viewpoint_cam = batch_data[batch_idx]
#        gt_image = gt_image.cuda()
#        viewpoint_cam = viewpoint_cam.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii,depth = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"],
        )

        # Loss
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

        if opt.lambda_depth > 0:
            # depth loss - difference between rendered depth and median-filtered
            # because big jumps in depth are often artifacts
            print("Depth loss not implemented yet")
            sys.exit(1)


        # Opa mask loss removed: requires alpha, which is not returned by the new renderer

        ###### rigid loss ######
        if opt.lambda_rigid > 0:
            k = 20
            cur_time = viewpoint_cam.timestamp
            _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
            xyz_mean = gaussians.get_xyz
            xyz_cur = xyz_mean  #  + delta_mean
            idx, dist = knn(
                xyz_cur[None].contiguous().detach(),
                xyz_cur[None].contiguous().detach(),
                k,
            )
            _, velocity = gaussians.get_current_covariance_and_mean_offset(
                1.0, gaussians.get_t + 0.1
            )
            weight = torch.exp(-100 * dist)
            # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
            # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
            # weight *= marginal_weights

            # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
            # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
            # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
            vel_dist = torch.norm(
                velocity[idx] - velocity[None, :, None], p=2, dim=-1
            )
            Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
            loss = loss + opt.lambda_rigid * Lrigid
        ########################

        ###### motion loss ######
        if opt.lambda_motion > 0:
            _, velocity = gaussians.get_current_covariance_and_mean_offset(
                1.0, gaussians.get_t + 0.1
            )
            Lmotion = velocity.norm(p=2, dim=1).mean()
            loss = loss + opt.lambda_motion * Lmotion
        ########################


        loss = loss / batch_size
        loss.backward()
        batch_point_grad.append(
            torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1)
        )
        batch_radii.append(radii)
        batch_visibility_filter.append(visibility_filter)

    losses = {"loss": loss, "Ll1": Ll1, "Lssim": Lssim}
    if opt.lambda_rigid > 0:
        losses["Lrigid"] = Lrigid
    if opt.lambda_motion > 0:
        losses["Lmotion"] = Lmotion
    if opt.lambda_depth > 0:
        losses["Ldepth"] = Ldepth

    batch_viewspace_point_grad = None
    if batch_size > 1:
        visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
        visibility_filter = visibility_count > 0
        radii = torch.stack(batch_radii, 1).max(1)[0]

        batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
        batch_viewspace_point_grad[visibility_filter] = (
            batch_viewspace_point_grad[visibility_filter]
            * batch_size
            / visibility_count[visibility_filter]
        )
        batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

        if gaussians.gaussian_dim == 4:
            batch_t_grad = gaussians._t.grad.clone()[:, 0].detach()
            batch_t_grad[visibility_filter] = (
                batch_t_grad[visibility_filter]
                * batch_size
                / visibility_count[visibility_filter]
            )
            batch_t_grad = batch_t_grad.unsqueeze(1)
        return (losses, image, gt_image,visibility_filter,radii,batch_viewspace_point_grad,batch_t_grad,None)
    else:
        if gaussians.gaussian_dim == 4:
            batch_t_grad = gaussians._t.grad.clone().detach()
        return (losses, image, gt_image,visibility_filter,radii,None,batch_t_grad,viewspace_point_tensor)

def collate_fn(x):
    return x

def try_save(gaussians, iteration, scene,name):
    print("\n[ITER {}] Saving checkpoint ({} gaussians) [{}]".format(iteration,gaussians.get_xyz.shape[0], name))
    try:
        with torch.no_grad():
            torch.save(
                (gaussians.capture(), iteration),
                scene.model_path + f"/chkpnt_{name}.pth",
            )
            # empty cache after save or bad things happen
            if CLEAR_CACHE:
                torch.cuda.empty_cache()
                print("Saved and emptied cache")
            else:
                print("Saved")
        # if name.startswith("iter"): 
        #     # render a video of the output (in a subprocess)
        #     mp4_path = f"training_{iteration}.mp4"
        #     args = [ 
        #         sys.executable,
        #        "show_images.py",
        #        scene.model_path,
        #        "-f",mp4_path,
        #        "-c","2,4,8",
        #        "-r"]
        #     launch_viewer(args, mp4_path)
    except Exception as e:
        import traceback
        print("Error saving checkpoint:")
        traceback.print_exc(e)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint,
    debug_from,
    gaussian_dim,
    time_duration,
    num_pts,
    num_pts_ratio,
    rot_4d,
    force_sh_3d,
    batch_size,
    prune_short_timespan_iters=None,
    generate_prefilter_masks=False,
):

    if dataset.frame_ratio > 1:
        time_duration = [
            time_duration[0] / dataset.frame_ratio,
            time_duration[1] / dataset.frame_ratio,
        ]

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)


    use_fastgs = (
        getattr(opt, 'use_fastgs_densification', False)
    )
    use_lff = (
        getattr(opt, 'use_lff_densification', False)
    )
    densification_iterations = []
    for x in range(opt.iterations):
        if x > opt.densify_from_iter and x % opt.densification_interval == 0:
            densification_iterations.append(x)

    densifiers : List(DensifierBase) = []
    # one base densifier 
    if use_fastgs:
        densifiers = [FastGSDensifier(opt)]
    else:
        densifiers = [PlainDensifier(opt)]
    # optional lff densifier run every other densification iteration
    # which is designed to remove artifacts
    if use_lff:
        densifiers.append(LFFDensifier(opt))

    densifiers.append(RecoveryAwarePruner(opt))
    


    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
        prefilter_var=dataset.prefilter_var,
        densifiers = densifiers
    )
    scene = Scene(
        dataset,
        gaussians,
        num_pts=num_pts,
        num_pts_ratio=num_pts_ratio,
        time_duration=time_duration,
    )
    gaussians.training_setup(opt)

    if checkpoint == "auto_latest":
        all_checkpoints = [
            (x, x.stat().st_mtime)
            for x in Path(dataset.model_path).glob("chkpnt_*.pth")
        ]
        try_checkpoints = sorted(all_checkpoints, key=lambda x: x[1], reverse=True)
        loaded = False
        for checkpoint,mtime in try_checkpoints:
            try:
                (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
                gaussians.restore(model_params, opt)
                model_params = None
                print(f"Loaded checkpoint {checkpoint} modified at {mtime}")
                break
            except RuntimeError as e:
                print(f"Error loading checkpoint {checkpoint}: {e}")
                continue
    elif checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        model_params = None
        



    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    lambda_all = [
        key
        for key in opt.__dict__.keys()
        if key.startswith("lambda") and key != "lambda_dssim"
    ]
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0

    progress_bar = tqdm(
        range(0, opt.iterations), desc="Training progress", initial=first_iter
    )
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(
            torch.zeros(
                (3, pipe.env_map_res, pipe.env_map_res),
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.lowfeature_lr, eps=1e-15)
    else:
        env_map = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()



    # estimate rough number of num workers based on initial point count to avoid OOM when loading things for large scenes
    initial_size = gaussians.get_xyz.shape[0]

    # if initial_size> 100_000:
    #     print(f"Initial point count {initial_size} is large, setting num_workers to 0 to avoid OOM")
    #     num_workers = 0;
    # else:
    #     # estimate 100000 points can run 10 workers
    #     # > 100000 points < 10 workers
    #     num_workers = int((100000/ initial_size) * 10)
    #     num_workers = min(os.cpu_count(), num_workers)
    #     num_workers = min(8, num_workers)
    #     print(f"Setting num_workers to {num_workers} based on initial point count {initial_size}")
    # loading time doesn't seem to dominate at least on my PC
    # so there's little point in worker threads
    # once the stupid PIL code is cached
    num_workers = 0

    if gaussians.gaussian_dim == 4:
        training_dataloader = DataLoader(
            training_dataset,
            batch_sampler = training_dataset.get_frame_batch_sampler(),
    #        num_workers=2 if dataset.dataloader else 0,
            num_workers=num_workers,
            collate_fn=collate_fn, # don't make a lambda as isn't pickleable for num_workers>0
        )
    else:
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
    #        num_workers=2 if dataset.dataloader else 0,
            num_workers=num_workers,
            collate_fn=collate_fn, # don't make a lambda as isn't pickleable for num_workers>0
            drop_last=True,
        )


    stop_iteration = False

    def stop_handler(signum, frame):
        nonlocal stop_iteration
        print("CTRL+C received, saving checkpoint and exiting")
        stop_iteration = True

    # Set the signal handler
    signal.signal(signal.SIGINT, stop_handler)


    iteration = first_iter
    while not stop_iteration and iteration < opt.iterations + 1:

        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - 1) >= debug_from and debug_from > 0:
                pipe.debug = True
            else:
                pipe.debug = False
                
            # update batch size based on what the dataloader returns (e.g. for 4d 
            # with frame batch sampler the batch size is cameras per-frame)
            batch_size = len(batch_data)
            ts_batch = batch_data[0][1].timestamp
            batch_data = [ (data[0].cuda(), data[1].cuda()) for data in batch_data]

            #print("Training {} gaussians on batch of size {} at iteration {} (timestamp {})".format(gaussians.get_xyz.shape[0], batch_size, iteration, ts_batch))

            losses, image, gt_image,visibility_filter,radii,batch_viewspace_point_grad,batch_t_grad,viewspace_point_tensor = run_batch(
                batch_data, batch_size, gaussians, pipe, background, opt
            )
            loss_dict = losses
            loss = losses["loss"]
            Ll1 = losses["Ll1"]
            Lssim = losses["Lssim"]

            iter_end.record()

            if stop_iteration:
                break

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log

                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[
                            f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                        ]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = (
                            0.4
                            * vars()[f"L{lambda_name.replace('lambda_', '')}"].item()
                            + 0.6 * ema
                        )
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[
                            lambda_name.replace("lambda_", "L")
                        ]

                if iteration % 10 == 0:
                    postfix = {
                        "N": f"{gaussians.get_xyz.shape[0]}",
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "PSNR": f"{psnr_for_log:.{2}f}",
                        "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                        "Lssim": f"{ema_ssimloss_for_log:.{4}f}",
                    }

                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[
                                f"ema_{lambda_name.replace('lambda_', '')}_for_log"
                            ]
                            postfix[lambda_name.replace("lambda_", "L")] = (
                                f"{ema_loss:.{4}f}"
                            )
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                # Log and save
                test_psnr = training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    loss_dict,
                )
                if iteration in saving_iterations:
                    try_save(gaussians, iteration, scene,f"iter_{iteration}")
                elif iteration in testing_iterations:
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        try_save(gaussians, iteration, scene,"best")
                    else:
                        try_save(gaussians, iteration, scene,"not-best")


                # Optimizer step - n.b. densifier calls may reset gradients so do this first
                if iteration < opt.iterations:
                    gaussians.optimizer_step(iteration,radii=radii)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none=True)

                # Densification
                if iteration < opt.densify_until_iter:
                    if TRACK_MEMORY:
                        torch.cuda.memory._dump_snapshot(f"temp.pickle")

#                    print(f"[ITER {iteration}] Densifying Gaussians: {gaussians.get_xyz.shape[0]}")
                    # Keep track of max radii in image-space for pruning
                    if batch_size == 1:
                        gaussians.add_densification_stats(
                            iteration = iteration,
                            viewspace_point_tensor = viewspace_point_tensor,
                            update_filter = visibility_filter,
                            radii = radii,
                            avg_t_grad = batch_t_grad if gaussians.gaussian_dim == 4 else None,
                        )
                    else:
                        gaussians.add_densification_stats_grad(
                            iteration = iteration,
                            viewspace_point_grad = batch_viewspace_point_grad,
                            update_filter = visibility_filter,
                            radii = radii,
                            avg_t_grad = batch_t_grad if gaussians.gaussian_dim == 4 else None,
                        )

                    prune_only = opt.densify_until_num_points > 0 and  gaussians.get_xyz.shape[0] >= opt.densify_until_num_points


                    # run any densifiers configured for this step
                    gaussians.run_densifiers(iteration, scene, radii, pipe, background,prune_only = prune_only)

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()
                    
                    if CLEAR_CACHE:
                        torch.cuda.empty_cache()
                    if TRACK_MEMORY:
                        torch.cuda.memory._dump_snapshot(f"temp.pickle")

                # call densifier per iteration for any densifiers that need to run every iteration 
                # or that need to do cleanup after the main densification phase (e.g. FastGS final pruning)
                gaussians.call_densifier_per_iteration(iteration, scene, radii, pipe, background)
                

    # Generate prefilter masks at end of training (including early stop via Ctrl+C)
    if generate_prefilter_masks and gaussians.gaussian_dim == 4:
        _save_prefilter_masks(gaussians, scene, dataset)

    if stop_iteration:
        print("\n[ITER {}] Saving checkpoint before exiting".format(iteration))
        try_save(gaussians, iteration, scene,name="resume")
        sys.exit(0)


def _prune_by_spatio_temporal_score(gaussians, scene, pipe, background, opt):
    """Compute spatio-temporal scores and prune low-scoring 4D Gaussians.

    Spatial score (Eq. 4):  For each training view, render with compute_contrib=True
        to obtain per-Gaussian pixel contributions (alpha * T summed over all pixels
        in that view).  These are accumulated across all views then averaged.
    Temporal score (Eq. 5-6):  Average marginal temporal weight over all unique
        training timestamps.
    Combined score (Eq. 7):  spatial_score * temporal_score.  Gaussians with
        combined score < opt.prune_st_score_threshold are removed.
    """
    print("\n[ST-prune] Accumulating spatial contributions across all training views…")
    train_cameras = scene.getTrainCameras()
    P = gaussians.get_xyz.shape[0]
    spatial_accum = torch.zeros(P, device="cuda")
    num_views = 0

    with torch.no_grad():
        for gt_image, viewpoint_cam in train_cameras:
            viewpoint_cam = viewpoint_cam.cuda()
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, compute_contrib=True)
            contrib = render_pkg["gauss_contrib"]
            if contrib.shape[0] == P:
                spatial_accum += contrib
            elif contrib.shape[0] > 0:
                print(
                    f"[ST-prune] Warning: gauss_contrib size {contrib.shape[0]} != "
                    f"num_gaussians {P}; skipping this view's contributions."
                )
            num_views += 1

    # Average over views so score is independent of dataset size
    if num_views > 0:
        spatial_accum /= num_views

    # Unique timestamps for temporal score computation
    timestamps = sorted(set(float(cam.timestamp) for _, cam in train_cameras))
    gaussians.prune_by_spatio_temporal_score(
        spatial_accum, timestamps, opt.prune_st_score_threshold
    )


def _save_prefilter_masks(gaussians, scene, dataset):
    """Generate and save per-timestamp active Gaussian masks for prefiltering.

    For each unique timestamp in the training cameras, computes a boolean mask of
    which Gaussians have a marginal temporal weight above 0.05 (the threshold used
    in the renderer). Saves the result to ``<model_path>/prefilter_masks.pt`` as a
    dict mapping float timestamp to a boolean numpy array of shape (num_gaussians,).
    """
    print("\nGenerating prefilter masks for all training timestamps...")
    train_cameras = scene.getTrainCameras()
    # Collect unique timestamps from (gt_image, camera) pairs
    timestamps = sorted(set(float(cam.timestamp) for _, cam in train_cameras))
    masks = gaussians.generate_prefilter_masks(timestamps)
    save_path = os.path.join(dataset.model_path, "prefilter_masks.pt")
    torch.save(masks, save_path)
    total_active = sum(int(m.sum()) for m in masks.values())
    total_possible = len(timestamps) * gaussians.get_xyz.shape[0]
    print(
        f"[Prefilter masks] Saved {len(masks)} timestamp masks to {save_path} "
        f"(avg {total_active / max(len(masks), 1):.0f}/{gaussians.get_xyz.shape[0]} "
        f"active Gaussians per frame, {100.0 * total_active / max(total_possible, 1):.1f}% total)"
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    loss_dict=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ssim_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
        tb_writer.add_histogram(
            "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
        )
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/rigid_loss",
                    loss_dict["Lrigid"].item(),
                    iteration,
                )
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/depth_loss",
                    loss_dict["Ldepth"].item(),
                    iteration,
                )
            if "Ltv" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/tv_loss", loss_dict["Ltv"].item(), iteration
                )
            if "Lopa" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/opa_loss", loss_dict["Lopa"].item(), iteration
                )
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/pts_opa_loss",
                    loss_dict["Lptsopa"].item(),
                    iteration,
                )
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/smooth_loss",
                    loss_dict["Lsmooth"].item(),
                    iteration,
                )
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar(
                    "train_loss_patches/laplacian_loss",
                    loss_dict["Llaplacian"].item(),
                    iteration,
                )

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        validation_configs = (
            {
                "name": "train",
                "cameras": train_cams,
                "range": [idx % len(train_cams) for idx in range(5, 30, 5)],
            },
            {
                "name": "test",
                "cameras": test_cams,
                "range": range(min(len(test_cams), 30)),
            },
        )
        for config in validation_configs:
            if (
                config["cameras"]
                and len(config["cameras"]) > 0
                and len(config["range"]) > 0
            ):
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config["range"])):
                    gt_image, viewpoint = config["cameras"][idx]
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()

                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    # depth and alpha are not returned by the new renderer, so skip visualization
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/gt_vs_render".format(viewpoint.image_name),
                            grid[None],
                            global_step=iteration,
                        )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config["range"])
                l1_test /= len(config["range"])
                ssim_test /= len(config["range"])
                msssim_test /= len(config["range"])
                print(
                    "\n[ITER {}] Gaussians {} Evaluating {}[{}]: L1 {} PSNR {}".format(
                        iteration, len(scene.gaussians._xyz), config["name"],len(config["cameras"]), l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - msssim",
                        msssim_test,
                        iteration,
                    )
                if config["name"] == "test":
                    psnr_test_iter = psnr_test.item()

    if CLEAR_CACHE:
        torch.cuda.empty_cache()
    return psnr_test_iter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default="auto_latest")

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--logpath", type=str, default="")
    parser.add_argument(
        "--prune_short_timespan_iters",
        nargs="+",
        type=int,
        default=[],
        help="Iterations at which to prune 4D Gaussians by spatio-temporal score "
             "(set threshold via --prune_st_score_threshold in OptimizationParams).",
    )
    parser.add_argument(
        "--generate_prefilter_masks",
        action="store_true",
        default=False,
        help="After training, generate per-timestamp active Gaussian masks and save "
             "to <model_path>/prefilter_masks.pt.",
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.logpath:
        os.makedirs(args.logpath, exist_ok=True)
        log_file = os.path.join(args.logpath, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        class BothWriter:
            def __init__(self,stdout,logfile):
                self.stdout = stdout
                self.logfile = logfile
            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)
                self.logfile.flush()
            def flush(self):
                if hasattr(self.stdout, 'flush'):
                    self.stdout.flush()
                self.logfile.flush()

        sys.stdout = BothWriter(sys.stdout, open(log_file, "w"))
        print(f"Logging to {log_file}")


    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [
            i for i in range(0, op.iterations, 500)
        ]
        args.save_iterations =  [
            i for i in range(0, op.iterations, 500)
        ]

    setup_seed(args.seed)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    #torch._logging.set_logs(graph_code=True)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    try:
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.gaussian_dim,
            args.time_duration,
            args.num_pts,
            args.num_pts_ratio,
            args.rot_4d,
            args.force_sh_3d,
            args.batch_size,
            prune_short_timespan_iters=args.prune_short_timespan_iters,
            generate_prefilter_masks=args.generate_prefilter_masks,
        )
        # All done
        print("\nTraining complete.")
    except torch.OutOfMemoryError as e:
        print("CUDA Out of Memory error during training: {}".format(e))
    finally:
        if TRACK_MEMORY:
            torch.cuda.memory._dump_snapshot(f"temp.pickle")


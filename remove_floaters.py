import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import torchvision
import re
from copy import deepcopy


def remove_floaters(model_path, iteration, views, gaussians, pipeline, background):
    # load the scene first
    with torch.no_grad():

        # we have a list of gaussians - for each one we want to calculate if it is a
        # shitty floater or not. We do this by rendering all views, and looking at the number
        # of high-error pixels which each gaussian contributes to. We get this from the
        # output of the renderer
        num_gaussians = gaussians.get_xyz.shape[0]
        error_pixel_count = torch.zeros(num_gaussians, device="cuda")
        for idx, view in enumerate(
            tqdm(views, desc="Rendering progress")
        ):
            if idx > 10:
                # fast finish for testing
                break
            rendering = render(
                view[1].cuda(), gaussians, pipeline, background, compute_contrib=True
            )
            print(view[0][0:3, :, :],rendering["render"])
            error_map = (rendering["render"] - view[0][0:3, :, :].cuda()) ** 2
            print(error_map.shape)
            error_map = error_map.mean(dim=0)
            contrib_map = rendering["gauss_contrib"]
            print(error_map.shape, contrib_map.shape)
            
            mask = error_map > 0.1
            for x in contrib_map[mask]:
                error_pixel_count[x] += 1

        bad_gaussians = torch.topk(error_pixel_count, 1000, sorted=True)
        for v, i in bad_gaussians:
            print(v, i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", help="Output folder path")
    args = parser.parse_args()

    checkpoints = Path(args.output_folder).glob("*.pth")
    sorted_checkpoints = list(sorted(checkpoints, key=lambda x: x.stat().st_mtime))
    if len(sorted_checkpoints):
        latest_pth = sorted_checkpoints[-1]
        print(f"Rendering from checkpoint: {latest_pth}")
    else:
        print("No checkpoints found")
        sys.exit(-1)

    render_cmdline = [
        "--model_path",
        str(args.output_folder),
        "--loaded_pth",
        str(latest_pth),
    ]

    render_parser = argparse.ArgumentParser(description="Render params")
    model = ModelParams(render_parser, sentinel=True)
    pipeline = PipelineParams(render_parser)

    render_args = get_combined_args(render_parser, cmdlne_string=render_cmdline)

    gaussians = GaussianModel(model.sh_degree, gaussian_dim=4, rot_4d=True)

    model = model.extract(render_args)
    pipeline = pipeline.extract(render_args)

    scene = Scene(model, gaussians, shuffle=False)
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    remove_floaters(
        model.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
    )

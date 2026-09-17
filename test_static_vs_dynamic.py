import re
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import safe_open


import numpy as np


# image files live in output/9moving/images/cam<N>_<FRAME>.png_<resolution>.st
# depth files live in output/9moving/images/cam<N>_<FRAME>.depth

@dataclass
class CameraFrame:
    camera_id: int
    frame_number: int
    resolution: int
    image_path: str

# find all the image files and sort them into lists by camera, sorted by frame number
# for each camera should get a list of CameraFrame objects sorted by frame number

def find_camera_frames(images_dir: str) -> dict[int, list[CameraFrame]]:
    images_path = Path(images_dir)
    pattern = re.compile(r'^cam(\d+)_(\d+)\.png_(\d+)\.st$')
    cameras: dict[int, list[CameraFrame]] = {}
    for f in images_path.iterdir():
        m = pattern.match(f.name)
        if m:
            cam_id = int(m.group(1))
            frame_num = int(m.group(2))
            resolution = int(m.group(3))
            frame = CameraFrame(
                camera_id=cam_id,
                frame_number=frame_num,
                resolution=resolution,
                image_path=str(f),
            )
            cameras.setdefault(cam_id, []).append(frame)
    for frames in cameras.values():
        frames.sort(key=lambda x: x.frame_number)
#    cameras = {1:cameras[1]}
    return cameras

all_frames = find_camera_frames("output/9moving/images")

# now create a function to load a camera frame - for images, load the tensor with safetensors and for depth use torch.load on the .depth file

def load_camera_frame(frame: CameraFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (image, depth) tensors for a CameraFrame.
    image: float16 tensor of shape [3, H, W]
    depth: float16 tensor of shape [H, W]
    """
    with safe_open(frame.image_path, framework='pt', device='cpu') as f:
        return f.get_tensor('tensor'), f.get_tensor('depth')


# for each camera, calculate a mean colour and max depth image
def compute_camera_mean_colour_and_max_depth(
    cameras: dict[int, list[CameraFrame]]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """For each camera, returns (mean_image, max_depth) across all frames."""
    means = {}
    for cam_id, frames in cameras.items():
        images, depths = [], []
        for frame in frames:
            image, depth = load_camera_frame(frame)
            images.append(image.float())
            depths.append(depth.float())
        # do this on CPU because otherwise it will load all the images (as opposed to mmapping them)
        means[cam_id] = (
            torch.stack(images).mean(dim=0),
            torch.stack(depths).max(dim=0).values,
        )
    return means


def _weighted_mean(v,w):
    return torch.sum(v.flatten()*w.flatten()) / torch.sum(w.flatten())

def _weighted_cov(v,w):
    return torch.cov(v.to(dtype=torch.float32).flatten(),correction=0,aweights=w.flatten())

def rescale_depth(src_rgb,src_depth,tgt_rgb,tgt_depth):
    src_depth =src_depth.to(dtype=torch.float32)
    tgt_depth =tgt_depth.to(dtype=torch.float32)

    rgb_diff = (src_rgb - tgt_rgb).norm(dim=0)
    rgb_diff=torch.clamp(rgb_diff,0.0,1.0)
    weights = 1.0-rgb_diff
    #weights = torch.ones_like(rgb_diff)

    weights[src_depth < -0.5]=0.0
    weights[tgt_depth < -0.5]=0.0

    mean_src = _weighted_mean(src_depth,weights)
    mean_tgt = _weighted_mean(tgt_depth,weights)

    print("Means:",mean_src,mean_tgt)
    print(tgt_depth.flatten().shape,rgb_diff.shape) 
    #variance_src = torch.cov(tgt_depth.flatten())
    #variance_tgt = torch.cov(src_depth.flatten())
    variance_src = _weighted_cov(src_depth,weights)
    variance_tgt = _weighted_cov(tgt_depth,weights)

    print("Covs:",variance_src,variance_tgt)


    scale_src = torch.sqrt(variance_src) 
    scale_tgt = torch.sqrt(variance_tgt)
    rval =  ((tgt_depth - mean_tgt)/scale_tgt)
    rval =rval*scale_src + mean_src
    rval[weights==0.0] = -1.0
#    rval = mean_src + ((tgt_depth - mean_tgt)/scale_tgt)*scale_src
    #print(rval.shape,torch.min(mean_src),torch.min(rval),torch.max(rval))

    mean_dst = _weighted_mean(rval,weights)
    print("After Means:",mean_src,mean_dst)
    variance_dst = _weighted_cov(rval,weights)
    print("After Covs:",variance_src,variance_dst,scale_tgt)
    print("")

    return rval
    




    


def compute_midpoint_diffs(
    cameras: dict[int, list[CameraFrame]],
    camera_means: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """For each camera, loads the midpoint frame and returns (rgb_diff, depth_diff) from the mean/max."""
    diffs = {}
    for cam_id, frames in cameras.items():
        mean_image, max_depth = camera_means[cam_id]
        mid_image, mid_depth = load_camera_frame(frames[len(frames) // 2])
        mid_depth=rescale_depth(mean_image,max_depth,mid_image,mid_depth)
        diffs[cam_id] = (
            (mid_image.float() - mean_image).norm(dim=0),
            (mid_depth.float() - max_depth).abs(),
        )
    return diffs

camera_means = compute_camera_mean_colour_and_max_depth(all_frames)
camera_diffs = compute_midpoint_diffs(all_frames, camera_means)

cam_ids = sorted(camera_means.keys())
fig, axes = plt.subplots(len(cam_ids), 5)
if len(cam_ids) == 1:
    axes = [axes]

for row, cam_id in enumerate(cam_ids):
    mean_image, max_depth = camera_means[cam_id]
    rgb_diff, depth_diff = camera_diffs[cam_id]
    
    # image: [3, H, W] float32 -> [H, W, 3], clamp to [0, 1]
    rgb = mean_image.clamp(0, 1).permute(1, 2, 0).numpy()
    depth = max_depth.numpy()

    # difference of midpoint frame from mean/max
    rgb_diff = rgb_diff.numpy()
    depth_diff = depth_diff.numpy()

    depth_cutoff = depth_diff > 1.0
    rgb_cutoff = rgb_diff > 0.2

    diffmax = np.maximum(depth_cutoff, rgb_cutoff)



    axes[row][0].imshow(rgb)
    axes[row][0].set_title(f'Camera {cam_id} — mean RGB')
    axes[row][0].axis('off')

    im = axes[row][1].imshow(depth)
    axes[row][1].set_title(f'Camera {cam_id} — max depth')
    axes[row][1].axis('off')

    axes[row][2].imshow(rgb_cutoff)
    axes[row][2].set_title(f'RGB diff')
    axes[row][2].axis('off')

    im2 = axes[row][3].imshow(depth_diff)
    axes[row][3].set_title(f'depth diff')
    axes[row][3].axis('off')

    im3 = axes[row][4].imshow(diffmax)
    axes[row][4].set_title(f'both')
    axes[row][4].axis('off')


plt.tight_layout()
plt.show()


from argparse import ArgumentParser

from matplotlib import pyplot,gridspec
from omegaconf import OmegaConf, DictConfig
from arguments import ModelParams, PipelineParams, OptimizationParams
import scene

from utils.build_depth import transform_depth_to_image_size

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str)
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
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



args = parser.parse_args()

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


in_data = scene.Scene(args,gaussians=None)

dataset = in_data.getTrainCameras()
timestamps = dataset.get_timestamps()
image_ids = [dataset.get_indices_for_timestamp(t) for t in timestamps]
sampler = dataset.get_index_batch_list_sampler(image_ids,readahead=5)
for batch in sampler:
    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0, hspace=0)
    figure = pyplot.figure(figsize=(4,3))

    for i,id in enumerate(batch):
        print(id)
        image, cam, depth = dataset[id]
        if depth is None:
            continue
        print(
            "Image shape:",
            image.shape,
            "Camera ID:",
            cam.colmap_id,
            "Depth shape:",
            depth.shape,
        )
        image_and_depth = image.permute(1, 2, 0)
        depth_transformed = transform_depth_to_image_size(depth, image)
        image_and_depth[:,:,0] = depth_transformed / (depth_transformed.max() + 1e-8)
        ax = pyplot.subplot(gs1[i])
        ax.imshow(image_and_depth.cpu().numpy())
        ax.set_axis_off()

    pyplot.show()

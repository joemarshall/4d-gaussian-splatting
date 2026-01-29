import time
import os
import sys

import torch
import numpy as np
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature

# import utils.contrastive_decoder_utils
from utils.sh_utils import SH2RGB

from sklearn.preprocessing import QuantileTransformer
import hdbscan
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import imageio


def get_combined_args(parser: ArgumentParser, model_path, target_cfg_file=None):
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if target_cfg_file is None:
        if args_cmdline.target == 'seg':
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target == 'scene' or args_cmdline.target == 'xyz':
            target_cfg_file = "cfg_args"
        elif args_cmdline.target == 'feature' or args_cmdline.target == 'coarse_seg_everything' or args_cmdline.target == 'contrastive_feature':
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

# Borrowed from GARField, but modified
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()
    print(scales.max(), '?')

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        scales_shape = scales.shape

        scales = scales.reshape(-1, 1)

        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device).reshape(scales_shape)

    return quantile_transformer_func, quantile_transformer


FEATURE_DIM = 32 # fixed
MODEL_PATH = './output/N3V/sear_steak' # 30000
FEATURE_GAUSSIAN_ITERATION = 10241
SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'scale_gate{FEATURE_GAUSSIAN_ITERATION}.pt')
FEATURE_PATH = os.path.join(MODEL_PATH, f'ckpt_contrastive_feature{FEATURE_GAUSSIAN_ITERATION}.pth')
SCENE_PATH = os.path.join(MODEL_PATH, 'chkpnt30000.pth')

scale_gate = torch.nn.Sequential(
    torch.nn.Linear(1, 32, bias=True),
    torch.nn.Sigmoid()
)

scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
scale_gate = scale_gate.cuda()

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument('--target', default='scene', type=str)

args = get_combined_args(parser, MODEL_PATH)
#args = parser.parse_args(sys.argv[1:])

dataset = model.extract(args)

# If use language-driven segmentation, load clip feature and original masks
dataset.need_features = False

# To obtain mask scales
dataset.need_masks = True

gaussian_dim = 4
time_duration = [0.0, 0.5]
rot_4d = True
force_sh_3d = False
gaussians = GaussianModel(
    dataset.sh_degree,
    gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d
)
feature_gaussians = FeatureGaussianModel(
    FEATURE_DIM,
    gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d
)

(model_params, first_iter) = torch.load(SCENE_PATH)
gaussians.restore(model_params, None)

(model_params, first_iter) = torch.load(FEATURE_PATH)
feature_gaussians.restore(model_params, None)

scene = Scene(
    dataset, gaussians, feature_gaussians,
    time_duration=time_duration,
    shuffle=False, mode='eval', target='contrastive_feature'
)

all_scales = []
training_dataset = scene.getTrainCameras()
training_dataset.only_path = True
for cam in training_dataset:
    _, _, _, _, mask_scales_path = cam
    mask_scales = torch.load(mask_scales_path)
    all_scales.append(mask_scales)
all_scales = torch.cat(all_scales)
training_dataset.only_path = False

upper_bound_scale = all_scales.max().item()
# upper_bound_scale = np.percentile(all_scales.detach().cpu().numpy(), 75)

# all_scales = []
# for cam in scene.getTrainCameras():
#     cam.mask_scales = torch.clamp(cam.mask_scales, 0, upper_bound_scale).detach()
#     all_scales.append(cam.mask_scales)
# all_scales = torch.cat(all_scales)

# quantile transformer
q_trans, q_trans_ = get_quantile_func(all_scales, 'uniform')

print("There are",len(training_dataset),"views in the dataset.")
print(upper_bound_scale)

################ Video ####################
label_to_color = np.random.rand(200, 3)

#### to tune
query_points = {
    0: [830, 950],
    9: [500, 480],
    15: [650, 600],
}
seg_2d = True
seg_3d = True
for fragment_id in [0, 9, 15]:
    rgb = []
    seg = defaultdict(list)
    seg_rgb = defaultdict(list)
    seg_feature = defaultdict(list)
    seg_gaussian = defaultdict(list)

    # fragment_id = 7
    query_point = query_points[fragment_id]
    em_point = False
    ss = [0.5, 0.7, 1.0]
    sim_thresh_list = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    save_path = f'./output/segment_results/2d_video/clip{fragment_id}_3d_2'
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for ref_img_camera_id in tqdm(range(16*fragment_id, 16*fragment_id+16)):
            data = deepcopy(training_dataset[ref_img_camera_id])
            gt_image, view, features, original_masks, mask_scales = data
            view = view.cuda()

            view.feature_height, view.feature_width = view.image_height, view.image_width
            preprocessed_pth_path = view.image_path.replace(
                '.png',
                f'_{view.image_height}_{view.image_width}.pth'
            )
            img = torch.load(preprocessed_pth_path) * 255
            # img = view.original_image * 255
            img = img.permute([1, 2, 0]).detach().cpu().numpy().astype(np.uint8)

            bg_color = [0 for i in range(FEATURE_DIM)]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            rendered_feature = render_contrastive_feature(
                view, feature_gaussians, pipeline.extract(args), background,
                norm_point_features=True, smooth_type='traditional', smooth_K=16
            )['render']
            feature_h, feature_w = rendered_feature.shape[-2:]

            rgb.append(img)

            # scale-aware 2D cluster
            for s in ss:  # [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]:
                scale = torch.tensor([s]).cuda()
                gates = scale_gate(scale)

                feature_with_scale = rendered_feature
                feature_with_scale = feature_with_scale * gates.unsqueeze(-1).unsqueeze(-1)
                scale_conditioned_feature = feature_with_scale.permute([1, 2, 0])

                normed_features = torch.nn.functional.normalize(scale_conditioned_feature, dim=-1, p=2)

                if seg_2d:
                    downsampled_features = torch.nn.functional.interpolate(
                        scale_conditioned_feature.permute([2, 0, 1]).unsqueeze(0),
                        (128, 128), mode='bilinear'
                    ).squeeze()
                    cluster_normed_features = torch.nn.functional.normalize(
                        downsampled_features, dim=0, p=2
                    ).permute([1, 2, 0])

                    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01)
                    cluster_labels = clusterer.fit_predict(
                        cluster_normed_features.reshape([-1, cluster_normed_features.shape[-1]]).detach().cpu().numpy())
                    labels = cluster_labels.reshape([cluster_normed_features.shape[0], cluster_normed_features.shape[1]])
                    # print(np.unique(labels))

                    cluster_centers = torch.zeros(len(np.unique(labels)) - 1, cluster_normed_features.shape[-1])
                    for i in range(1, len(np.unique(labels))):
                        cluster_centers[i - 1] = torch.nn.functional.normalize(cluster_normed_features[labels == i - 1].mean(dim=0),
                                                                               dim=-1)

                    segmentation_res = torch.einsum('nc,hwc->hwn', cluster_centers.cuda(), normed_features)

                    segmentation_res_idx = segmentation_res.argmax(dim=-1)
                    colored_labels = label_to_color[segmentation_res_idx.cpu().numpy().astype(np.int8)]

                    colored_labels = (colored_labels * 255).astype(np.uint8)
                    # import pdb;pdb.set_trace()
                    r, g, b = colored_labels[..., 0], colored_labels[..., 1], colored_labels[..., 2]
                    target_rgb = colored_labels[query_point[0], query_point[1]]
                    mask = (r == target_rgb[0]) & (g == target_rgb[1]) & (b == target_rgb[2])
                    x = img.copy()
                    x[~mask] = 255
                    if em_point:
                        for i in [-2,-1,0,1,2]:
                            for j in [-2,-1,0,1,2]:
                                x[query_point[0]+i, query_point[1]+j, 0] = 255
                    # cv2.imwrite(
                    #     os.path.join(save_path, f'{ref_img_camera_id}_{s}.png'),
                    #     cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)
                    # )
                    seg[f'{s}'].append(colored_labels)
                    seg_rgb[f'{s}'].append(x)
                    seg_feature[f'{s}'].append(((normed_features[:,:,:3]/2.+0.5).cpu().numpy() * 255).astype(np.uint8))

                if seg_3d:
                    query_feature = normed_features[query_point[0], query_point[1]]

                    # point_features = feature_gaussians.get_point_features
                    point_features = feature_gaussians.get_smoothed_point_features()

                    scale_conditioned_point_features = point_features * gates.unsqueeze(0)

                    normed_scale_conditioned_point_features = torch.nn.functional.normalize(
                        scale_conditioned_point_features, dim=-1, p=2)

                    similarities = torch.einsum('C,NC->N', query_feature.cuda(),
                                                normed_scale_conditioned_point_features)

                    for sim_thresh in sim_thresh_list:
                        try:
                            gaussians.roll_back()
                        except:
                            pass

                        mask = torch.logical_and(
                            similarities > sim_thresh,
                            (gaussians.get_opacity > 0.2)[:, 0])
                        gaussians.segment(mask)

                        bg_color = [1 for i in range(FEATURE_DIM)]
                        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                        rendered_segmented_image = render(view, gaussians, pipeline.extract(args), background)['render']
                        # plt.imshow(rendered_segmented_image.permute([1, 2, 0]).detach().cpu())
                        gaussians.roll_back()

                        rendered_segmented_image = rendered_segmented_image.permute([1, 2, 0]).detach().cpu().numpy()
                        rendered_segmented_image = np.clip(rendered_segmented_image, 0, 1)
                        rendered_segmented_image = (rendered_segmented_image * 255).astype(np.uint8)

                        seg_gaussian[f'{s}_{sim_thresh}'].append(rendered_segmented_image)


    if seg_2d:
        imageio.mimwrite(
            os.path.join(save_path, f'rgb_clip{fragment_id}.mp4'),
            [x for x in rgb],
            fps=8
        )

        for k in seg.keys():
            imageio.mimwrite(
                os.path.join(save_path, f'seg{k}_clip{fragment_id}.mp4'),
                [x for x in seg[f'{k}']],
                fps=8)
            imageio.mimwrite(
                os.path.join(save_path, f'seg_rgb{k}_clip{fragment_id}.mp4'),
                [x for x in seg_rgb[f'{k}']],
                fps=8)
            imageio.mimwrite(
                os.path.join(save_path, f'seg_feature{k}_clip{fragment_id}.mp4'),
                [x for x in seg_feature[f'{k}']],
                fps=8)

    if seg_3d:
        for k in seg_gaussian.keys():
            imageio.mimwrite(
                os.path.join(save_path, f'seg_gaussian{k}_clip{fragment_id}.mp4'),
                [x for x in seg_gaussian[f'{k}']],
                fps=8)



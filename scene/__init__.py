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

import os
import numpy as np
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_ff import FeatureGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset

class Scene:

    gaussians : GaussianModel
    feature_gaussians : FeatureGaussianModel

    def __init__(self,
                 args : ModelParams,
                 gaussians : GaussianModel=None,
                 feature_gaussians: FeatureGaussianModel=None,
                 load_iteration=None,
                 feature_load_iteration=None,
                 shuffle=True,
                 resolution_scales=[1.0],
                 num_pts=100_000,
                 num_pts_ratio=1.0,
                 time_duration=None,
                 init_from_3dgs_pcd=False,
                 target='scene',
                 mode='train',
                 sample_rate=1.0
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.feature_loaded_iter = None
        self.gaussians = gaussians
        self.feature_gaussians = feature_gaussians
        self.white_background = args.white_background

        # if load_iteration:
        #     if load_iteration == -1:
        #         if mode == 'train':
        #             # only load feature gaussians when doing segmentation
        #             if target == 'seg' or target == 'coarse_seg_everything':
        #                 self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
        #             elif target == 'scene':
        #                 self.feature_loaded_iter = None
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
        #             elif target == 'feature' or target == 'contrastive_feature':
        #                 self.feature_loaded_iter = None
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
        #             else:
        #                 assert False and "Unknown target!"
        #         elif mode == 'eval':
        #             if target == 'seg':
        #                 self.feature_loaded_iter = None
        #                 self.feature_gaussians = None
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="seg")
        #             elif target == 'scene':
        #                 self.feature_gaussians = None
        #                 self.feature_loaded_iter = None
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
        #             elif target in ['feature', 'contrastive_feature']:
        #                 self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
        #                 self.loaded_iter = -1 if gaussians is None else searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target='scene')
        #             elif target == 'coarse_seg_everything':
        #                 self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target)
        #                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
        #             else:
        #                 assert False and "Unknown target!"
        #     else:
        #         self.loaded_iter = load_iteration
        #         if mode == 'train':
        #             if target == 'seg' or target == 'coarse_seg_everything':
        #                 self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
        #             elif target == 'scene' or 'feature' in target:
        #                 self.feature_loaded_iter = None
        #             else:
        #                 assert False and "Unknown target!"
        #         elif mode == 'eval':
        #             if target == 'seg':
        #                 self.feature_loaded_iter = None
        #                 self.feature_gaussians = None
        #             elif target == 'scene':
        #                 self.feature_gaussians = None
        #                 self.feature_loaded_iter = None
        #             elif target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
        #                 self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
        #                 self.loaded_iter = -1
        #             else:
        #                 assert False and "Unknown target!"
        #
        #     print("Loading trained model at iteration {}, {}".format(self.loaded_iter, self.feature_loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval,
                num_pts=num_pts, time_duration=time_duration, extension=args.extension,
                num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader,
                need_features=args.need_features, need_masks=args.need_masks
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.scene_info = scene_info
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load or initialize scene / seg gaussians
        # loaded in train.py using ckpt.pth

        # Load or initialize feature gaussians
        # loaded in train.py using ckpt.pth

        # if self.feature_loaded_iter and self.feature_gaussians is not None:
        #     if target == 'feature' or target == 'seg':
        #         self.feature_gaussians.load_ply(os.path.join(self.model_path,
        #                                                 "point_cloud",
        #                                                 "iteration_" + str(self.feature_loaded_iter),
        #                                                 "feature_point_cloud.ply"))
        #     elif target == 'coarse_seg_everything':
        #         if mode == 'train':
        #             self.feature_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
        #                                                     "point_cloud",
        #                                                     "iteration_" + str(self.loaded_iter),
        #                                                     "scene_point_cloud.ply"))
        #         elif mode == 'eval':
        #             self.feature_gaussians.load_ply(os.path.join(self.model_path,
        #                                                 "point_cloud",
        #                                                 "iteration_" + str(self.feature_loaded_iter),
        #                                                 "coarse_seg_everything_point_cloud.ply"))
        #     elif target == 'contrastive_feature':
        #         if mode == 'train':
        #             self.feature_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
        #                                                     "point_cloud",
        #                                                     "iteration_" + str(self.loaded_iter),
        #                                                     "scene_point_cloud.ply"))
        #         elif mode == 'eval':
        #             self.feature_gaussians.load_ply(os.path.join(self.model_path,
        #                                                 "point_cloud",
        #                                                 "iteration_" + str(self.feature_loaded_iter),
        #                                                 "contrastive_feature_point_cloud.ply"))
        #
        #
        # elif self.feature_gaussians is not None:
        #     if target=='feature' and init_from_3dgs_pcd:
        #         print("Initialize feature gaussians from 3DGS point cloud")
        #         self.feature_gaussians.create_from_pcd(fetchPly(
        #                 os.path.join(
        #                     self.model_path,
        #                     "point_cloud",
        #                     "iteration_" + str(searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene") if (self.loaded_iter is None or self.loaded_iter == -1) else self.loaded_iter),
        #                     "scene_point_cloud.ply"
        #                 ),
        #                 only_xyz=True
        #             ),
        #         self.cameras_extent
        #         )
        #     elif target == 'contrastive_feature':
        #         if mode == 'train':
        #             self.feature_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
        #                                                     "point_cloud",
        #                                                     "iteration_" + str(self.loaded_iter),
        #                                                     "scene_point_cloud.ply"))
        #         elif mode == 'eval':
        #             self.feature_gaussians.load_ply(os.path.join(self.model_path,
        #                                                 "point_cloud",
        #                                                 "iteration_" + str(self.feature_loaded_iter),
        #                                                 "contrastive_feature_point_cloud.ply"))
        #     else:
        #         print("Initialize feature gaussians from Colmap point cloud")
        #         self.feature_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, ckpt_name=None, target='scene'):
        assert target != 'feature' and "Please use save_feature() to save feature gaussians!"
        ckpt_name = str(iteration) if ckpt_name is None else ckpt_name
        torch.save((self.gaussians.capture(), iteration), self.model_path + f"/ckpt_{target}" + ckpt_name + ".pth")

    def save_mask(self, iteration, id = 0):
        path = self.model_path + f"/mask_it{iteration}_id{id}.npy"
        mask = self.gaussians._mask.detach().cpu().numpy()
        np.save(path, mask)

    def save_feature(self, iteration, ckpt_name=None, target = 'coarse_seg_everything', smooth_weights = None, smooth_type = None, smooth_K = 16):
        assert self.feature_gaussians is not None and (target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature')
        ckpt_name = str(iteration) if ckpt_name is None else ckpt_name
        torch.save((self.feature_gaussians.capture(), iteration), self.model_path + f"/ckpt_{target}" + ckpt_name + ".pth")

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, rotation_matrix_to_quaternion
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import torch
from utils.general_utils import fps
from multiprocessing.pool import ThreadPool
import imagesize


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0
    camera_pose_id: int = -1


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

    def with_copied_cameras(self, num_copies):
        """return a new sceneinfo which has train_cameras repeated num_copies time,
        with timestamps offset by the last timestamp of the previous copy plus one frame delta
        """
        if num_copies < 1:
            return self

        if not self.train_cameras:
            return self._replace(train_cameras=[])

        timestamps = sorted({cam.timestamp for cam in self.train_cameras})
        if len(timestamps) > 1:
            frame_delta = min(
                (timestamps[i + 1] - timestamps[i])
                for i in range(len(timestamps) - 1)
                if (timestamps[i + 1] - timestamps[i]) > 0
            )
        else:
            frame_delta = 1.0 / 30.0

        min_timestamp = min(cam.timestamp for cam in self.train_cameras)
        max_timestamp = max(cam.timestamp for cam in self.train_cameras)
        copy_period = (max_timestamp - min_timestamp) + frame_delta

        copied_train_cameras = []
        for copy_idx in range(num_copies + 1):
            time_offset = copy_idx * copy_period
            for cam in self.train_cameras:
                copied_train_cameras.append(
                    cam._replace(timestamp=cam.timestamp + time_offset)
                )

        return self._replace(train_cameras=copied_train_cameras)


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, dataloader):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}\n".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_name = os.path.basename(extr.name).split(".")[0]
        image_ext = os.path.basename(extr.name).split(".")[-1]
        image_basename = "_".join(image_name.split("_")[:-1])
        frame = 0
        while True:
            this_image_name = image_basename + "_%05d" % frame
            this_image_path = (
                os.path.join(images_folder, this_image_name) + "." + image_ext
            )
            if not os.path.exists(this_image_path):
                break
            if not dataloader:
                this_image = Image.open(this_image_path)
                print("Warning - loading everything into memory")
            else:
                this_image = None
            cx = width * 0.5
            cy = height * 0.5

            timestamp = frame / 30.0
            # if .depth file exists, put the name into depth
            depth_path = this_image_path.replace("." + image_ext, ".depth")
            if os.path.exists(depth_path):
                depth = depth_path
            else:
                print("Missing depth for:", this_image_path)
                depth = None
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=this_image,
                depth=depth,
                image_path=this_image_path,
                image_name=this_image_name,
                width=width,
                height=height,
                timestamp=timestamp,
                fl_x=focal_length_x,
                fl_y=focal_length_y,
                cx=cx,
                cy=cy,
                camera_pose_id=idx,
            )
            cam_infos.append(cam_info)
            frame += 1
    sys.stdout.write("\n")

    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    if "nx" in vertices:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    else:
        normals = np.zeros_like(positions)
    if "time" in vertices:
        timestamp = vertices["time"][:, None]
    else:
        timestamp = None
    return BasicPointCloud(
        points=positions, colors=colors, normals=normals, time=timestamp
    )


def makePointCloudFromImages(scene, points_per_image=1000):
    # get the training dataset from the scene
    # then for all images choose points
    # spread across the image and save to ply with the calculated point (from the depth and the camera rays)
    # and timestamp set to the image timestamp
    # and return this as a BasicPointCloud for use with gaussian initialization

    TENSOR_DEVICE = "cuda"

    dataset = scene.getTrainCameras()
    num_points_total = len(dataset) * points_per_image
    all_points = torch.zeros((num_points_total, 2), dtype=torch.int32,device=TENSOR_DEVICE)
    all_xyz = torch.zeros((num_points_total, 3), device=TENSOR_DEVICE)
    all_colors = torch.zeros((num_points_total, 3), device=TENSOR_DEVICE)
    all_normals = torch.zeros((num_points_total, 3), device=TENSOR_DEVICE)
    all_times = torch.zeros((num_points_total, 1), device=TENSOR_DEVICE)
    all_durations = torch.zeros((num_points_total, 1), device=TENSOR_DEVICE)
    all_src_cams = torch.zeros((num_points_total, 1), dtype=torch.int32, device=TENSOR_DEVICE)
    all_depths = torch.zeros((num_points_total, 1), device=TENSOR_DEVICE)

    # live points, indexed by colmap_id
    # for each colmap_id we have a list of x,y and d
    live_points = {}

    timestamps = dataset.get_timestamps()

    all_points_index = 0

    # last frame images per camera - we prioritise adding points where the image
    # changed so we capture motion
    last_frames = {}

    def add_finished_points(
        finished_points,finished_xyz, finished_depth, finished_colors, finished_times, time_now, cam_id
    ):
        nonlocal all_points_index
        num_points = finished_points.shape[0]
        all_src_cams[all_points_index : all_points_index + num_points,0] = cam_id
        all_xyz [ all_points_index : all_points_index + num_points] = finished_xyz
        all_points[all_points_index : all_points_index + num_points] = finished_points
        all_depths[all_points_index : all_points_index + num_points] = finished_depth.unsqueeze(-1)
        all_colors[all_points_index : all_points_index + num_points] = finished_colors
        all_times[all_points_index : all_points_index + num_points] = (
            finished_times.unsqueeze(-1)
        )
        all_durations[all_points_index : all_points_index + num_points] = (
            time_now - finished_times
        ).unsqueeze(-1)
        count_zero = (all_durations[all_points_index : all_points_index + num_points] == 0).sum().item()
        if count_zero > 0:
            print("Warning - finished points with zero duration:", count_zero, "out of", num_points)
            asdasd
        all_points_index += num_points


    for t in timestamps:
        frame_cams = dataset.get_indices_for_timestamp(t)
        for idx in frame_cams:
            (
                image,
                cam,
                depth,
            ) = dataset[idx]
            rays_o, rays_d = cam.cuda().get_rays()

            valid_depth = depth > 0
            point_sample_weight = valid_depth.float()

            last_frame_img = last_frames.get(cam.colmap_id, None)
            if last_frame_img is not None:
                # if the image has changed, then we want to prioritise adding points at this point
                # so we increase the weight of points that have changed in color
                color_diff = torch.linalg.vector_norm(image - last_frame_img, dim=0)
                point_sample_weight *= 0.5 + color_diff
            last_frames[cam.colmap_id] = image

            point_data = live_points.get(cam.colmap_id, None)
            if point_data is not None:
                old_points, old_xyz, old_depths, old_color, old_times = point_data
                finished_points_mask = depth[old_points[:, 0], old_points[:, 1]] < 0
                # check each old point and clear it if it is too different in depth
                # finished_points_mask |= torch.abs(depth[old_points[:,0], old_points[:,1]] - old_depths) > 0.01
                # check each old point and clear it if it is too different in color

                colors_in_this_frame = image[
                    :, old_points[:, 0], old_points[:, 1]
                ].transpose(0, 1)

                finished_points_mask |= (
                    torch.linalg.vector_norm(colors_in_this_frame - old_color, dim=-1)
                    > 0.1
                )

                still_live_points = ~finished_points_mask

                # if a point isn't finished then don't sample it again in the current frame
                point_sample_weight[
                    old_points[still_live_points][:, 0],
                    old_points[still_live_points][:, 1],
                ] = 0.0
                if still_live_points.sum() == 0:
                    live_points[cam.colmap_id] = None
                else:
                    live_points[cam.colmap_id] = (
                        old_points[still_live_points],
                        old_xyz[still_live_points],
                        old_depths[still_live_points],
                        old_color[still_live_points],
                        old_times[still_live_points],
                    )
                print(
                    "Live points:",
                    still_live_points.sum().item(),
                    "Finished points:",
                    finished_points_mask.sum().item(),
                )
                if finished_points_mask.sum() > 0:
                    # now add in the points that are finished to the all_points list with a duration
                    add_finished_points(
                        old_points[finished_points_mask],
                        old_xyz[finished_points_mask],
                        old_depths[finished_points_mask],
                        old_color[finished_points_mask],
                        old_times[finished_points_mask],
                        t,
                        cam.colmap_id
                    )

            print("Making points from image:", cam.image_path)
            # read the depth and image
            # sample points across the image
            # for each point, calculate the 3D position using the camera intrinsics and extrinsics
            # save to ply with color from the image and timestamp from the camera

            # do in frame order, keep a list of gaussians in play and
            # finish them if depth error too great

            ## distance >4 from centre gets zero weight
            ##point_dists = torch.norm(rays_d*depth.unsqueeze(-1),dim=-1)
            ##point_sample_weight[point_dists>4.0] = 0.0

            point_idxs = torch.multinomial(
                point_sample_weight.flatten(), points_per_image, replacement=False
            )

            point_xs = point_idxs % cam.image_width
            point_ys = point_idxs // cam.image_width
            new_depths = depth[point_ys, point_xs]

            new_points = torch.stack([point_ys, point_xs], dim=-1)

            projected_xyz = rays_d[new_points[:, 0], new_points[:, 1], :] * new_depths.unsqueeze(-1)
            projected_xyz += rays_o[0, 0]

            new_xyz = projected_xyz
            new_times = torch.ones(new_points.shape[0], device=TENSOR_DEVICE) * cam.timestamp
            new_colors = image[:, new_points[:, 0], new_points[:, 1]].transpose(0, 1)

            point_data = live_points.get(cam.colmap_id, None)

            if point_data is not None:
                old_points, old_xyz, old_depths, old_color, old_times = point_data
                new_points = torch.cat([new_points, old_points], dim=0)
                new_xyz = torch.cat([new_xyz, old_xyz], dim=0)
                new_depths = torch.cat([new_depths, old_depths], dim=0)
                new_colors = torch.cat([new_colors, old_color], dim=0)
                new_times = torch.cat([new_times, old_times], dim=0)
            live_points[cam.colmap_id] = (
                new_points,
                new_xyz,
                new_depths,
                new_colors,
                new_times,
            )

    last_time = timestamps[-1] + timestamps[1]-timestamps[0]

    for cam_id, point_data in live_points.items():
        if point_data is not None:
            old_points, old_xyz, old_depths, old_color, old_times = point_data
            add_finished_points(old_points,old_xyz, old_depths, old_color, old_times, last_time,cam_id)

    print("Total points made:", all_points_index, "out of", num_points_total)

    # now iterate backwards through frames (and though all_points which is sorted in order of end time), and for any points that
    # finish on the current frame plus 1, extend their duration to the current frame if the color is similar enough (as per calculation above)
    last_timestamp = None
    last_frames = {}  # image, cam, depth, t

    live_points = torch.zeros((all_points_index), dtype=torch.bool)
    for t in reversed(timestamps):
        frame_cams = dataset.get_indices_for_timestamp(t)
        for idx in frame_cams:
            (
                image,
                cam,
                depth,
            ) = dataset[idx]
            last_frame_data = last_frames.get(cam.colmap_id, None)
            if last_frame_data is not None:
                last_image, last_cam, last_depth, last_time = last_frame_data
                # check all points which end on the next frame (the last one in loop because we are going backwards)
                possible_points_to_extend = torch.logical_and(all_times[:,0] == last_time,all_src_cams[:, 0] == cam.colmap_id).nonzero()
                possible_points_to_extend = possible_points_to_extend.squeeze()

                diff_image = last_image - image
                diff_image_norm = torch.linalg.vector_norm(diff_image, dim=0)




                sample_points = all_points[possible_points_to_extend]
                print(diff_image_norm.shape,diff_image_norm.dtype,possible_points_to_extend.shape,all_points.shape,sample_points.shape)
                

                diff_image_samples = diff_image_norm[
                    sample_points[:,0], sample_points[:,1] 
                ]

                possible_points_to_extend = possible_points_to_extend[diff_image_samples< 0.1]


                all_times[possible_points_to_extend] = t
                all_durations[possible_points_to_extend] += last_time - t
                print(
                    "Extended points:",
                    possible_points_to_extend.numel(),
                    "at time:",
                    t,
                )
            last_frames[cam.colmap_id] = (image, cam, depth, t)

    all_rotations = torch.zeros((all_points_index,4), device=TENSOR_DEVICE)
    all_scales = torch.zeros((all_points_index,3), device=TENSOR_DEVICE)


    first_frame_cams = dataset.get_indices_for_timestamp(timestamps[0])
    print(len(first_frame_cams), "cameras in first frame:")

    for idx in first_frame_cams:
        (
            image,
            cam,
            depth,
        ) = dataset[idx]

        rays_o, rays_d = cam.cuda().get_rays()

        # get scale multiplier for 1 pixel move at distance 1
        ray_scale_multiplier = torch.linalg.vector_norm(
            rays_d[rays_d.shape[0] // 2, rays_d.shape[1] // 2, :] - rays_d[1+(rays_d.shape[0] // 2), rays_d.shape[1] // 2, :])
        print("Ray scale multiplier for camera", cam.colmap_id, ":", ray_scale_multiplier.item())
        this_cam_indices = cam.colmap_id == all_src_cams[:, 0]

        this_cam_points = all_points[this_cam_indices]
        this_cam_depths = all_depths[this_cam_indices]
        this_cam_times = all_times[this_cam_indices]
        this_cam_durations = all_durations[this_cam_indices]
        this_cam_end_times = this_cam_times + this_cam_durations
        this_cam_centre_times = this_cam_times+ (this_cam_durations *0.5)
        this_cam_scales = torch.zeros((this_cam_points.shape[0],3), device=TENSOR_DEVICE)
        cam_rotation = rotation_matrix_to_quaternion(torch.transpose(torch.tensor(cam.R,device=TENSOR_DEVICE,dtype=torch.float32),0,1))
        this_cam_rotations = torch.ones((this_cam_points.shape[0],4) , device=TENSOR_DEVICE)* cam_rotation.unsqueeze(0)
        last_timestamp = timestamps[0] - (timestamps[1]-timestamps[0])
        for timestamp in timestamps:            
            live_indices = torch.logical_and(this_cam_times[:,0] <= timestamp, this_cam_end_times[:,0] > timestamp)
            centre_indices = torch.logical_and(this_cam_centre_times[:,0] < timestamp, this_cam_centre_times[:,0] >= last_timestamp)
            last_timestamp = timestamp
            if live_indices.count_nonzero() == 0 or centre_indices.count_nonzero() == 0:
                continue

            live_points = this_cam_points[live_indices]
            centre_points = this_cam_points[centre_indices]
            centre_depths = this_cam_depths[centre_indices][:,0]


            from utils.pointdistance4d import find2d_nearest_points
            point_indices, centre_scales_2d = find2d_nearest_points(live_points,centre_points)
            # now project the 2d scales into 3d using the camera info
            #centre_scales_2d = torch.tensor(10.0, device=TENSOR_DEVICE).repeat(centre_points.shape[0])
#            print(centre_depths.shape,centre_scales_2d.shape,ray_scale_multiplier.shape)
            centre_scales_3d = (centre_scales_2d * ray_scale_multiplier) * centre_depths 
            # make the output be flattish in z
            centre_scales_3d = torch.stack([centre_scales_3d*3.0, centre_scales_3d*3.0, centre_scales_3d*0.5],dim=1)
#            print(centre_scales_3d.shape)
            this_cam_scales[centre_indices] = centre_scales_3d
        all_rotations[this_cam_indices] = this_cam_rotations
        all_scales[this_cam_indices] = this_cam_scales
        print("Finished calculating scales for camera:", cam.colmap_id, "with", this_cam_scales.shape[0], "points", (all_scales<=0).sum().item(), "scales <= 0")

    print("Scale distribution:",all_scales[this_cam_indices].min().item(),all_scales[this_cam_indices].max().item(),all_scales[this_cam_indices].mean().item())
    print(all_scales)

    print("Found all scales",(all_scales==0).sum().item(), "out of", all_scales.shape[0], "points")

    # test code to only use camera 2
    # only_cam_2 = all_src_cams[:,0] == 2
    # all_xyz = all_xyz[only_cam_2]
    # all_colors = all_colors[only_cam_2]
    # all_normals = all_normals[only_cam_2]
    # all_times = all_times[only_cam_2]
    # all_durations = all_durations[only_cam_2]
    # all_scales = all_scales[only_cam_2]
    # all_rotations = all_rotations[only_cam_2]
    # # now we have calculated the points then calculate the scales
    # # by going through points, keeping a track of live points
    # # and doing KNN on only live points
    # # n.b. we need to scale points somehow based on duration also
    # from utils.pointdistance4d import pointDistance4D
    # all_scales = pointDistance4D(
    #      all_xyz,all_times,all_durations,timestamps)
    # print("Found scales!",scales.shape)
    

    return BasicPointCloud(
        points=all_xyz.cpu().numpy(),
        colors=all_colors.cpu().numpy(),
        normals=all_normals,
        times=all_times.cpu().numpy(),
        durations=all_durations.cpu().numpy(),
        scales = all_scales.cpu().numpy(),
        rotations = all_rotations.cpu().numpy()
    )


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path, images, eval, llffhold=8, num_pts_ratio=1.0, dataloader=False
):
    print("Reading scene info: ", path, dataloader)
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        dataloader=dataloader,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if num_pts_ratio > 1.001:
        num_pts = int((num_pts_ratio - 1) * pcd.points.shape[0])
        mean_xyz = pcd.points.mean(axis=0)
        min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
        max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
        xyz = np.concatenate(
            [
                pcd.points,
                np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz)
                + min_rand_xyz,
            ],
            axis=0,
        )
        colors = np.concatenate(
            [pcd.colors, SH2RGB(np.random.random((num_pts, 3)) / 255.0)], axis=0
        )
        normals = np.concatenate([pcd.normals, np.zeros((num_pts, 3))], axis=0)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(
    path,
    transformsfile,
    white_background,
    extension=".png",
    time_duration=None,
    frame_ratio=1,
    dataloader=False,
):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
    if "camera_angle_x" in contents:
        fovx = contents["camera_angle_x"]

    frames = contents["frames"]
    tbar = tqdm(range(len(frames)))

    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        timestamp = frame.get("time", 0.0)
        if frame_ratio > 1:
            timestamp /= frame_ratio
        if time_duration is not None and "time" in frame:
            if timestamp < time_duration[0] or timestamp > time_duration[1]:
                return

        cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(
            path, cam_name
        )  # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
        image_name = Path(cam_name).stem

        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                print(image_path, arr.shape)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGBA")
            else:
                print(image_path, arr.shape)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = np.empty(0)
            width, height = imagesize.get(image_path)

        if "depth_path" in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
        if "fl_x" in frame and "fl_y" in frame and "cx" in frame and "cy" in frame:
            FovX = FovY = -1.0
            fl_x = frame["fl_x"]
            fl_y = frame["fl_y"]
            cx = frame["cx"]
            cy = frame["cy"]
            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                depth=depth,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                timestamp=timestamp,
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
            )

        elif (
            "fl_x" in contents
            and "fl_y" in contents
            and "cx" in contents
            and "cy" in contents
        ):
            FovX = FovY = -1.0
            fl_x = contents["fl_x"]
            fl_y = contents["fl_y"]
            cx = contents["cx"]
            cy = contents["cy"]
            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                depth=depth,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                timestamp=timestamp,
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
            )
        else:
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx
            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                depth=depth,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                timestamp=timestamp,
            )

    with ThreadPool() as pool:
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames))
        pool.close()
        pool.join()

    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]

    return cam_infos


def readNerfSyntheticInfo(
    path,
    white_background,
    eval,
    extension=".png",
    num_pts=100_000,
    time_duration=None,
    num_extra_pts=0,
    frame_ratio=1,
    dataloader=False,
):

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_train.json",
        white_background,
        extension,
        time_duration=time_duration,
        frame_ratio=frame_ratio,
        dataloader=dataloader,
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_test.json" if not path.endswith("lego") else "transforms_val.json",
        white_background,
        extension,
        time_duration=time_duration,
        frame_ratio=frame_ratio,
        dataloader=dataloader,
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if pcd.points.shape[0] > num_pts:
        mask = np.random.randint(0, pcd.points.shape[0], num_pts)
        # mask = fps(torch.from_numpy(pcd.points).cuda()[None], num_pts).cpu().numpy()
        if pcd.time is not None:
            times = pcd.time[mask]
        else:
            times = None
        xyz = pcd.points[mask]
        rgb = pcd.colors[mask]
        normals = pcd.normals[mask]
        if times is not None:
            time_mask = (times[:, 0] < time_duration[1]) & (
                times[:, 0] > time_duration[0]
            )
            xyz = xyz[time_mask]
            rgb = rgb[time_mask]
            normals = normals[time_mask]
            times = times[time_mask]
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)

    if num_extra_pts > 0:
        times = pcd.time
        xyz = pcd.points
        rgb = pcd.colors
        normals = pcd.normals
        bound_min, bound_max = xyz.min(0), xyz.max(0)
        radius = 60.0  # (bound_max - bound_min).mean() + 10
        phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
        theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        xyz_extra = np.stack([x, y, z], axis=1)
        normals_extra = np.zeros_like(xyz_extra)
        rgb_extra = np.ones((num_extra_pts, 3)) / 2

        xyz = np.concatenate([xyz, xyz_extra], axis=0)
        rgb = np.concatenate([rgb, rgb_extra], axis=0)
        normals = np.concatenate([normals, normals_extra], axis=0)

        if times is not None:
            times_extra = (
                torch.zeros(((num_extra_pts, 3)))
                + (time_duration[0] + time_duration[1]) / 2
            )
            times = np.concatenate([times, times_extra], axis=0)

        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}

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
from utils.build_depth  import transform_depth_to_image_size


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


def makePointCloudFromImages(
    scene,
    points_per_frame=1000,
    max_points=3000000,
    depth_tolerance=0.05,
    color_threshold=0.1,
):
    """Build a 4D point cloud from the depth + colour of every camera in every frame.

    Points are sampled per frame (across all cameras), merged between cameras where
    they are mutually visible, tracked through time in a single set of live points,
    and finally extended backwards in time where they still appear to exist.
    """
    TENSOR_DEVICE = "cuda"
    bg = torch.tensor((1, 1, 1) if scene.white_background else (0, 0, 0), device=TENSOR_DEVICE)

    dataset = scene.getTrainCameras()
    timestamps = list(dataset.get_timestamps())

    if len(timestamps) * points_per_frame > max_points:
        print("Skipping timestamps to limit total points to", max_points)
        num_to_keep = max(1, max_points // points_per_frame)
        timestamps = timestamps[:: max(1, len(timestamps) // num_to_keep)]
    num_frames = len(timestamps)
    frame_delta = (timestamps[1] - timestamps[0]) if num_frames > 1 else 1.0 / 30.0
    # time of each frame start, plus one extra entry for "alive until the end"
    time_values = torch.tensor(
        timestamps + [timestamps[-1] + frame_delta], dtype=torch.float32, device=TENSOR_DEVICE
    )
    print(f"Finding up to {num_frames * points_per_frame} points at {num_frames} timestamps")

    def load_frame(batch):
        frame = []
        for idx in batch:
            image, cam, depth = dataset[idx]
            depth = transform_depth_to_image_size(depth, image)
            rays_o, rays_d = cam.get_rays()
            wvt = cam.world_view_transform.to(TENSOR_DEVICE)
            if cam.cx > 0:
                fx, fy, cx, cy = cam.fl_x, cam.fl_y, cam.cx, cam.cy
            else:
                fx = cam.image_width / (2 * np.tan(cam.FoVx / 2))
                fy = cam.image_height / (2 * np.tan(cam.FoVy / 2))
                cx, cy = cam.image_width / 2, cam.image_height / 2
            frame.append(
                dict(
                    cam=cam,
                    id=cam.colmap_id,
                    image=image,
                    depth=depth,
                    width=image.shape[2],
                    height=image.shape[1],
                    origin=rays_o[0, 0],
                    pos=rays_d * depth.unsqueeze(-1) + rays_o[0, 0],
                    rot=wvt[:3, :3],
                    trans=wvt[3, :3],
                    intr=(fx, fy, cx, cy),
                )
            )
        return frame

    def project(c, xyz):
        pv = xyz @ c["rot"] + c["trans"]
        z = pv[:, 2]
        zs = z.clamp(min=1e-6)
        fx, fy, cx, cy = c["intr"]
        px = torch.floor(fx * pv[:, 0] / zs + cx).long()
        py = torch.floor(fy * pv[:, 1] / zs + cy).long()
        inside = (z > 1e-6) & (px >= 0) & (px < c["width"]) & (py >= 0) & (py < c["height"])
        return px.clamp(0, c["width"] - 1), py.clamp(0, c["height"] - 1), inside

    def observe(frame, xyz):
        """Project points into every camera of the frame.

        Returns per-camera visibility [C,N] (in view and depth agrees), colour [C,N,3],
        the 3d point the camera's own depth gives at that pixel [C,N,3] and the flat pixel index.
        """
        vis, cols, poss, pix = [], [], [], []
        for c in frame:
            px, py, inside = project(c, xyz)
            rng = torch.linalg.vector_norm(xyz - c["origin"], dim=-1)
            d = c["depth"][py, px]
            vis.append(inside & (d > 0) & ((d - rng).abs() < depth_tolerance * rng))
            cols.append(c["image"][:, py, px].T)
            poss.append(c["pos"][py, px])
            pix.append(py * c["width"] + px)
        return torch.stack(vis), torch.stack(cols), torch.stack(poss), pix

    def still_alive(vis, cols, ref_color):
        # a point has ended if its colour differs in most of the cameras that can see it
        differs = ((cols - ref_color.unsqueeze(0)).norm(dim=-1) > color_threshold) & vis
        n_vis = vis.sum(0)
        return (n_vis > 0) & (differs.sum(0) * 2 <= n_vis)

    def sample_weights(c, prev_image):
        w = (c["depth"] > 0).float()
        # strongly prefer non-background pixels
        bg_diff = torch.amax(torch.abs(c["image"] - bg.view(3, 1, 1)), dim=0)
        w *= torch.clamp((bg_diff - 0.1) * 10.0, 0.0, 1.0)
        if prev_image is not None:
            # prefer pixels that changed since the last frame so motion gets points
            w *= 0.25 + torch.clamp(torch.linalg.vector_norm(c["image"] - prev_image, dim=0), 0.0, 1.0)
        return w

    def sample_new_points(frame, weights):
        totals = torch.stack([w.sum() for w in weights])
        if totals.sum() <= 0:
            return None
        cam_counts = torch.bincount(
            torch.multinomial(totals, points_per_frame, replacement=True), minlength=len(frame)
        )
        src_cam, src_y, src_x = [], [], []
        for ci, (c, w) in enumerate(zip(frame, weights)):
            k = min(int(cam_counts[ci]), int((w > 0).sum()))
            if k == 0:
                continue
            idx = torch.multinomial(w.flatten(), k, replacement=False)
            src_cam.append(torch.full((k,), ci, device=TENSOR_DEVICE))
            src_y.append(idx // c["width"])
            src_x.append(idx % c["width"])
        if not src_cam:
            return None
        src_cam, src_y, src_x = torch.cat(src_cam), torch.cat(src_y), torch.cat(src_x)
        n = src_cam.shape[0]

        own_xyz = torch.zeros((n, 3), device=TENSOR_DEVICE)
        own_col = torch.zeros((n, 3), device=TENSOR_DEVICE)
        own_depth = torch.zeros(n, device=TENSOR_DEVICE)
        own_pix = torch.zeros(n, dtype=torch.long, device=TENSOR_DEVICE)
        for ci, c in enumerate(frame):
            m = src_cam == ci
            own_xyz[m] = c["pos"][src_y[m], src_x[m]]
            own_col[m] = c["image"][:, src_y[m], src_x[m]].T
            own_depth[m] = c["depth"][src_y[m], src_x[m]]
            own_pix[m] = src_y[m] * c["width"] + src_x[m]

        vis, cols, poss, pix = observe(frame, own_xyz)
        sample_ids = torch.arange(n, device=TENSOR_DEVICE)
        keep = torch.ones(n, dtype=torch.bool, device=TENSOR_DEVICE)
        for ci, c in enumerate(frame):
            m = src_cam == ci
            # a sample always sees itself, whatever the rounding says
            vis[ci, m] = True
            cols[ci, m] = own_col[m]
            poss[ci, m] = own_xyz[m]
            pix[ci] = torch.where(m, own_pix, pix[ci])
            # drop samples that land on the pixel of an earlier sample: same point seen again
            claim = torch.full((c["width"] * c["height"],), n, dtype=torch.long, device=TENSOR_DEVICE)
            claim.scatter_reduce_(0, pix[ci][vis[ci]], sample_ids[vis[ci]], reduce="amin")
            keep[m] &= claim[own_pix[m]] == sample_ids[m]

        v = vis.float().unsqueeze(-1)
        count = v.sum(0).clamp(min=1)
        xyz = ((poss * v).sum(0) / count)[keep]
        col = ((cols * v).sum(0) / count)[keep]
        src_ids = torch.tensor([c["id"] for c in frame], device=TENSOR_DEVICE)[src_cam][keep]
        src_pixel = torch.stack([src_y, src_x], dim=-1)[keep]
        return xyz, col, src_ids, src_pixel, own_depth[keep]

    def empty_points():
        return dict(
            xyz=torch.zeros((0, 3), device=TENSOR_DEVICE),
            col=torch.zeros((0, 3), device=TENSOR_DEVICE),
            start=torch.zeros(0, dtype=torch.long, device=TENSOR_DEVICE),
            src_cam=torch.zeros(0, dtype=torch.long, device=TENSOR_DEVICE),
            src_pix=torch.zeros((0, 2), dtype=torch.long, device=TENSOR_DEVICE),
            depth=torch.zeros(0, device=TENSOR_DEVICE),
        )

    live = empty_points()
    finished = []  # (points dict, end frame index)

    def finish(mask, end_idx):
        if mask.any():
            finished.append(({k: v[mask] for k, v in live.items()}, end_idx))

    # forward pass
    batches = [dataset.get_indices_for_timestamp(t) for t in timestamps]
    sampler = dataset.get_index_batch_list_sampler(batches, readahead=5, output_device="cuda")
    progress = tqdm(total=num_frames * 2, desc="Making point cloud from images")
    prev_images = {}
    for f, batch in enumerate(sampler):
        progress.update(1)
        frame = load_frame(batch)
        weights = [sample_weights(c, prev_images.get(c["id"])) for c in frame]

        if live["xyz"].shape[0] > 0:
            vis, cols, _, pix = observe(frame, live["xyz"])
            alive = still_alive(vis, cols, live["col"])
            # don't resample pixels that already belong to a live point
            for ci in range(len(frame)):
                weights[ci].view(-1)[pix[ci][vis[ci] & alive]] = 0.0
            finish(~alive, f)
            live = {k: v[alive] for k, v in live.items()}

        new = sample_new_points(frame, weights)
        if new is not None:
            xyz, col, src_ids, src_pixel, depth = new
            live = {
                k: torch.cat([live[k], v])
                for k, v in zip(
                    ["xyz", "col", "start", "src_cam", "src_pix", "depth"],
                    [xyz, col, torch.full((xyz.shape[0],), f, dtype=torch.long, device=TENSOR_DEVICE), src_ids, src_pixel, depth],
                )
            }
        for c in frame:
            prev_images[c["id"]] = c["image"]

    finish(torch.ones(live["xyz"].shape[0], dtype=torch.bool, device=TENSOR_DEVICE), num_frames)
    pts = {k: torch.cat([p[k] for p, _ in finished]) for k in finished[0][0]}
    end = torch.cat([torch.full((p["xyz"].shape[0],), e, device=TENSOR_DEVICE) for p, e in finished]).long()
    print("Total points made:", pts["xyz"].shape[0])

    # backward pass: points starting at frame k+1 are extended to frame k if they still look the same there
    batches = [dataset.get_indices_for_timestamp(t) for t in reversed(timestamps)]
    sampler = dataset.get_index_batch_list_sampler(batches, readahead=5, output_device="cuda")
    for k, batch in zip(reversed(range(num_frames)), sampler):
        progress.update(1)
        candidates = (pts["start"] == k + 1).nonzero().squeeze(1)
        if candidates.numel() == 0:
            continue
        vis, cols, _, _ = observe(load_frame(batch), pts["xyz"][candidates])
        extend = still_alive(vis, cols, pts["col"][candidates])
        pts["start"][candidates[extend]] = k
    progress.close()

    all_xyz = pts["xyz"]
    all_colors = pts["col"]
    all_times = time_values[pts["start"]].unsqueeze(-1)
    all_durations = (time_values[end] - time_values[pts["start"]]).unsqueeze(-1)
    all_src_cams = pts["src_cam"]
    all_points = pts["src_pix"]
    all_depths = pts["depth"].unsqueeze(-1)
    all_rotations = torch.zeros((all_xyz.shape[0], 4), device=TENSOR_DEVICE)
    all_scales = torch.zeros((all_xyz.shape[0], 3), device=TENSOR_DEVICE)

    # scales come from the 2d spacing of points in their source camera
    from utils.pointdistance4d import find2d_nearest_points

    for idx in dataset.get_indices_for_timestamp(timestamps[0]):
        _, cam, _ = dataset[idx]
        _, rays_d = cam.get_rays()
        # scale multiplier for a 1 pixel move at distance 1
        ray_scale_multiplier = torch.linalg.vector_norm(
            rays_d[rays_d.shape[0] // 2, rays_d.shape[1] // 2, :]
            - rays_d[1 + (rays_d.shape[0] // 2), rays_d.shape[1] // 2, :]
        )
        this_cam = all_src_cams == cam.colmap_id
        if not this_cam.any():
            continue
        cam_points = all_points[this_cam]
        cam_depths = all_depths[this_cam]
        cam_times = all_times[this_cam]
        cam_end_times = cam_times + all_durations[this_cam]
        cam_centre_times = cam_times + all_durations[this_cam] * 0.5
        cam_scales = torch.zeros((cam_points.shape[0], 3), device=TENSOR_DEVICE)
        cam_rotation = rotation_matrix_to_quaternion(
            torch.tensor(cam.R, device=TENSOR_DEVICE, dtype=torch.float32).T
        )
        all_rotations[this_cam] = cam_rotation.unsqueeze(0)
        last_timestamp = timestamps[0] - frame_delta
        for timestamp in timestamps:
            live_idx = (cam_times[:, 0] <= timestamp) & (cam_end_times[:, 0] > timestamp)
            centre_idx = (cam_centre_times[:, 0] < timestamp) & (cam_centre_times[:, 0] >= last_timestamp)
            last_timestamp = timestamp
            if live_idx.sum() == 0 or centre_idx.sum() == 0:
                continue
            _, centre_scales_2d = find2d_nearest_points(cam_points[live_idx], cam_points[centre_idx])
            scale_3d = centre_scales_2d * ray_scale_multiplier * cam_depths[centre_idx][:, 0]
            # make the output flattish in z
            cam_scales[centre_idx] = torch.stack([scale_3d * 3.0, scale_3d * 3.0, scale_3d * 0.5], dim=1)
        all_scales[this_cam] = cam_scales

    all_scales=all_scales.clamp(min=1e-6,max=0.05)
    print("Found all scales", (all_scales == 0).sum().item(), "zero out of", all_scales.numel())

    return BasicPointCloud(
        points=all_xyz.cpu().numpy(),
        colors=all_colors.cpu().numpy(),
        normals=np.zeros(all_xyz.shape, dtype=np.float32),
        times=all_times.cpu().numpy(),
        durations=all_durations.cpu().numpy(),
        scales=all_scales.cpu().numpy(),
        rotations=all_rotations.cpu().numpy(),
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

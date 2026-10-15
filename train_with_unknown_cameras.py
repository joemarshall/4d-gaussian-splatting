def undistort_image_and_save(x,tgt):
    if tgt.exists():
        print("Skipping undistortion for", x.name, "because", tgt, "already exists.")
        return
    print("Undistorting image", x.name,tgt.name)
    dst_image, _dst_cam = pycolmap.undistort_image(
        options=undistort_options,
        distorted_camera=distorted_camera,
        distorted_image=pycolmap.Bitmap.read(x, as_rgb=True),
    )
    dst_image.write(tgt)


if __name__ == "__main__":
    

    from fileinput import filename
    import json
    import subprocess
    import argparse
    import shutil
    import os
    import re
    import sys
    import numpy as np
    import copy
    from pathlib import Path
    import pycolmap
    from tqdm import tqdm



    from pyparsing import line

    from utils.graphics_utils import (
        getWorld2View2,
        getProjectionMatrix,
        getProjectionMatrixCenterShift,
        rotation_matrix_to_quaternion,
        quaternion_to_rotation_matrix,
        quaternion_slerp,
    )

    USE_AUTO_RECONSTRUCTION = False


    def parse_colmap_text(file_path: Path):
        in_lines = file_path.read_text().splitlines()
        lines = [line for line in in_lines if not line.strip().startswith("#")]
        lines = [line.strip().split(" ") for line in lines]
        header_lines = [line for line in in_lines if line.strip().startswith("#")]
        return lines, header_lines


    class ColmapFile:
        def __init__(self, path, lines_per_entry):
            self.path = path
            self.lines = {}
            self.headers = []
            if path != None:
                in_lines, self.headers = parse_colmap_text(path)
                for x in range(0, len(in_lines), lines_per_entry):
                    new_lines = []
                    for i in range(lines_per_entry):
                        if x + i < len(in_lines):
                            new_lines.append(in_lines[x + i])
                        else:
                            new_lines.append([])
                    id = int(new_lines[0][0])
                    self.lines[id] = new_lines

        def write(self, path):
            out_data = ""
            for header_line in self.headers:
                out_data += header_line + "\n"
            for id in sorted(self.lines.keys()):
                for subline in self.lines[id]:
                    out_data += " ".join(subline) + "\n"
            print("Writing:", sorted(self.lines.keys()))
            path.write_text(out_data)

        def items(self):
            return self.lines.items()

        def keys(self):
            return self.lines.keys()

        def __setitem__(self, key, value):
            self.lines[key] = value

        def __getitem__(self, key):
            return self.lines[key]

        def __contains__(self, key):
            return key in self.lines

        def __len__(self):
            return len(self.lines)


    def split_frame_name(frame_name):
        match = re.match(r"(\D)*(\d+)_(\d+).png", frame_name)
        if not match:
            raise ValueError(
                f"Frame name {frame_name} does not match expected pattern txt[cam_num]_[frame_num].png"
            )
        base_name, camera_id, frame_idx = match.groups()
        return base_name, int(camera_id), int(frame_idx)


    # Quaternion to rotation matrix conversion
    def quat_to_rot(qw, qx, qy, qz):
        """
        Convert quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
        Quaternion should be in the form (w, x, y, z).
        """
        # Normalize quaternion
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        R = np.array(
            [
                [
                    1 - 2 * qy * qy - 2 * qz * qz,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx * qx - 2 * qz * qz,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx * qx - 2 * qy * qy,
                ],
            ]
        )
        return R


    def colmap_image_line_to_world_to_camera(image_line):
        rotation = np.array(list(map(float, image_line[1:5])))
        translation = np.array(list(map(float, image_line[5:8])))

        transform = np.eye(4)
        transform[:3, :3] = quaternion_to_rotation_matrix(rotation)
        transform[:3, 3] = translation
        return transform


    def colmap_image_line_to_camera_to_world(image_line):
        return np.linalg.inv(colmap_image_line_to_world_to_camera(image_line))


    def camera_to_world_to_colmap_pose(camera_to_world):
        world_to_camera = np.linalg.inv(camera_to_world)
        rotation = rotation_matrix_to_quaternion(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        return rotation, translation


    def average_rigid_transforms(transforms):
        averaged_transform = np.eye(4)
        averaged_transform[:3, 3] = np.mean(
            [transform[:3, 3] for transform in transforms], axis=0
        )

        summed_rotations = np.sum([transform[:3, :3] for transform in transforms], axis=0)
        U, _, Vt = np.linalg.svd(summed_rotations)
        averaged_rotation = U @ Vt
        if np.linalg.det(averaged_rotation) < 0:
            U[:, -1] *= -1
            averaged_rotation = U @ Vt
        averaged_transform[:3, :3] = averaged_rotation
        return averaged_transform


    def rotation_angle_degrees(rotation_matrix):
        trace = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))


    def remap_colmap_ids_to_filename_camera_ids(images_data, frames_data):
        image_id_map = {}
        remapped_images = {}
        for old_image_id, image_lines in images_data.items():
            image_line = image_lines[0].copy()
            points_line = image_lines[1].copy()
            filename_camera_id = split_frame_name(image_line[-1])[1]

            if filename_camera_id in remapped_images:
                raise ValueError(
                    f"Duplicate filename camera id {filename_camera_id} in images.txt; cannot remap IDs uniquely."
                )

            image_id_map[old_image_id] = filename_camera_id
            image_line[0] = str(filename_camera_id)
            remapped_images[filename_camera_id] = [image_line, points_line]

        images_data.lines = remapped_images

        print("Images after map", list(sorted(remapped_images.keys())))
        print("Image ID remapping:", image_id_map)
        print("Frames: before:", list(frames_data.keys()))
        remapped_frames = {}
        for cur_id, frame_lines in frames_data.items():
            print(f"Handling frame id {cur_id} with lines {frame_lines}")
            frame_line = frame_lines[0].copy()
            num_data_ids = int(frame_line[9])
            data_start = 10

            remapped_data_ids = []
            for data_idx in range(num_data_ids):
                base_idx = data_start + (data_idx * 3)
                old_data_id = int(frame_line[base_idx + 2])
                new_data_id = image_id_map[old_data_id]
                frame_line[base_idx + 2] = str(new_data_id)
                remapped_data_ids.append(new_data_id)
                print(
                    f"Remapping frame {frame_line[0]} data id {old_data_id} to {new_data_id}"
                )

            if len(remapped_data_ids) == 0:
                continue

            new_frame_id = remapped_data_ids[0]

            frame_line[0] = str(new_frame_id)
            print(
                f"Remapping frame {frame_line[0]} with data ids {remapped_data_ids} to new frame id {new_frame_id}"
            )
            print(frame_line[0])

            if new_frame_id in remapped_frames:
                raise ValueError(
                    f"Duplicate remapped frame id {new_frame_id} in frames.txt; cannot remap IDs uniquely."
                )
            remapped_frames[new_frame_id] = [frame_line]
        print(remapped_frames)
        frames_data.lines = remapped_frames


    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=Path)
    parser.add_argument(
        "--fix-camera-resolution",
        "-f",
        action="store_true",
        help="Whether to fix the video resolution to be the same as the colmap camera resolution. This is useful if colmap fails to reconstruct any cameras due to unexpected video resolution, e.g. 1088p.",
    )
    parser.add_argument("-o", "--output_folder", type=Path)
    parser.add_argument("-nf", "--num_frames", type=int, default=-1)
    parser.add_argument(
        "-i",
        "--import_missing_cameras_from_scene",
        type=Path,
        default=None,
        help="Path to scene folder to import missing cameras from."
        "This is useful if colmap fails to reconstruct some cameras,"
        "e.g. if we are using masked source video, so we can import "
        "the camera poses from the original scene.",
    )
    parser.add_argument(
        "-r",
        "--replace_cameras_from_scene",
        type=Path,
        default=None,
        help="Path to scene to replace cameras from",
    )

    parser.add_argument(
        "--use_existing_scene_poses",
        "-u",
        type=Path,
        default=None,
        help="Path to scene to copy all camera poses from"
        "This is useful if the source video is e.g. a masked version of"
        "another scene which has poses already",
    )


    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = Path("output") / (args.video_folder.stem)
    args.output_folder = args.output_folder.resolve()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.num_frames > 0:
        ffmpeg_time_limit = ["-t", str(args.num_frames / 30)]
    else:
        ffmpeg_time_limit = []


    def camera_id_from_path(video_path, index):
        match = re.match(r"(\D)*(\d+)*", video_path.stem)
        if match:
            return int(match.groups()[1])
        return str(index)


    video_resolution = None
    # 1) for each mp4 in video folder, ffmpeg to extract frames into subfolder called images if the frames don't exist
    for i, video_path in enumerate(sorted(args.video_folder.glob("*.mp4"))):
        if video_resolution is None:
            # probe the video to get its resolution
            result = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    str(video_path),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in result.splitlines():
                split_line = line.split("x")
                if (
                    len(split_line) == 2
                    and split_line[0].isdigit()
                    and split_line[1].isdigit()
                ):
                    video_resolution = (int(split_line[0]), int(split_line[1]))
                    print(f"Detected video resolution: {video_resolution}")
                    break
        distorted_image_folder = args.output_folder / "images_distorted"
        if (
            not distorted_image_folder.exists()
            or not (
                distorted_image_folder / f"cam{camera_id_from_path(video_path,i)}_00000.png"
            ).exists()
        ):
            print(f"Extracting frames from {video_path} to {distorted_image_folder}")
            distorted_image_folder.mkdir(exist_ok=True)
            print(f"Getting frames from {video_path}")
            result = subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-start_number",
                    "0",
                    "-pix_fmt","rgb24",
                ]
                + ffmpeg_time_limit
                + [
                    str(
                        distorted_image_folder
                        / f"cam{camera_id_from_path(video_path,i)}_%05d.png"
                    ),
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            print(f"Frames already extracted for {video_path}, skipping.")


    # 2) make a colmap processing folder in output/colmap
    colmap_path = args.output_folder / "colmap"
    colmap_path.mkdir(exist_ok=True)

    # 3) copy *_00000.png into colmap_path/images, except for camera 0, saved for testing purposes
    distorted_image_folder = args.output_folder / "images_distorted"
    colmap_images_folder = colmap_path / "images"
    colmap_images_folder.mkdir(exist_ok=True)
    for image in distorted_image_folder.glob("*_00000.png"):
        if split_frame_name(image.name)[1] != 0:
            shutil.copy(image, colmap_images_folder)

    colmap_database = colmap_path / "database.db"

    sparse_output = colmap_path / "sparse"
    sparse_output.mkdir(exist_ok=True)

    dense_output = colmap_path / "dense" / "0"
    dense_output.mkdir(parents=True, exist_ok=True)

    sparse_final = args.output_folder / "sparse" / "0"

    if (sparse_final / "images.txt").exists():
        print(
            f"Colmap output already exists at {sparse_final}, skipping colmap reconstruction."
        )
    else:
        print("Running colmap reconstruction.")
        pycolmap.extract_features(database_path=str(colmap_database), image_path=str(colmap_images_folder))
        pycolmap.match_exhaustive(database_path=str(colmap_database))

        with pycolmap.Database.open(str(colmap_database)) as database:
            num_images = database.num_images()

        # def image_pair_callback(*args):
        #     print("Initial image pair found.",args)
        # def next_image_callback(*args):
        #     print("Next image found.",args)

        # for x in tqdm(range(num_images), desc="Running colmap mapper"):
        #     reconstructions = pycolmap.incremental_mapping(
        #             database_path=colmap_database,
        #             image_path=colmap_images_folder,
        #             output_path=colmap_path / "sparse",
        #             initial_image_pair_callback=image_pair_callback,
        #             next_image_callback=next_image_callback,
        #         )




        reconstruction = pycolmap.Reconstruction(colmap_path / "sparse" / "0")
        pycolmap.bundle_adjustment(
            reconstruction = reconstruction,
        )
        bundle_adjusted = colmap_path / "bundle_adjusted"
        bundle_adjusted.mkdir(exist_ok=True)
        reconstruction.write_text(bundle_adjusted)


        pycolmap.undistort_images(
            input_path=bundle_adjusted,
            image_path=colmap_images_folder,
            output_path=colmap_path / "undistorted",
        )

        reconstruction = pycolmap.Reconstruction(colmap_path / "undistorted"/"sparse")

        sparse_final.mkdir(exist_ok=True, parents=True)
        reconstruction.write_text(sparse_final)



    # now undistort all the other images based on
    # the reconstructed cameras

    distorted_reconstruction = pycolmap.Reconstruction(colmap_path / "bundle_adjusted")
    distorted_camera = distorted_reconstruction.camera(1)

    image_folder = args.output_folder / "images"
    if not image_folder.exists():
        image_folder.mkdir(exist_ok=True)

    distorted_image_list = list(distorted_image_folder.glob(f"cam*_*.png"))
    image_folder_list = list(image_folder.glob(f"cam*_*.png"))

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    if len(image_folder_list) < len(distorted_image_list):
        print("Found more distorted images than undistorted images.")
        print("Undistorting the remaining images.")
        undistort_options = pycolmap.UndistortCameraOptions()



        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for x in distorted_image_list:
                if x.name in [img.name for img in image_folder_list]:
                    continue
                else:
                    print(x.name,(image_folder / x.name).exists())
                    futures.append(executor.submit(undistort_image_and_save, x,image_folder/x.name))
            executor.shutdown(wait=True)

        # for x in distorted_image_list:
        #     if x.name in [img.name for img in image_folder_list]:
        #         continue
        #     else:
        #         print("Undistorting image", x.name)
        #         inp_image = pycolmap.Bitmap.read(x,as_rgb=True)
        #         (dst_image,_dst_cam) = pycolmap.undistort_image(options = undistort_options,distorted_camera = distorted_camera,distorted_image=inp_image)
        #         dst_image.write(image_folder / x.name)


        
    # find max frame number
    all_frames = (args.output_folder / "images").glob("*.png")
    max_frame_idx = 0
    for x in all_frames:
        _base, cam_idx, frame_idx = split_frame_name(x.name)
        max_frame_idx = max(frame_idx, max_frame_idx)

    print("Max frame index:", max_frame_idx)

    duration_seconds = max_frame_idx / 30


    # make depth image for each frame for training
    from utils.build_depth import calculate_depths

    calculate_depths(args.output_folder, ignore_existing=True)


    if args.fix_camera_resolution:
        cameras_file = ColmapFile(
            sparse_output / "cameras.txt", 1
        )  # verify we can read the existing poses
        for c in cameras_file.keys():
            print(c, cameras_file[c])
            resolution = cameras_file[c][0][2:4]
            resolution = (int(resolution[0]), int(resolution[1]))
            if resolution != video_resolution:
                print(
                    "Fixing resolution for camera",
                    c,
                    "from",
                    resolution,
                    "to",
                    video_resolution,
                )
                multiplier = (
                    video_resolution[0] / resolution[0],
                    video_resolution[1] / resolution[1],
                )
                if multiplier[0] != multiplier[1]:
                    print(
                        "Warning: non-uniform resolution multiplier",
                        multiplier,
                        "for camera",
                        c,
                    )

                def fix_multiplier(value, multiplier):
                    return str(float(value) * multiplier)

                cameras_file[c][0][2] = video_resolution[0]
                cameras_file[c][0][3] = video_resolution[1]
                for i in range(4, len(cameras_file[c][0]), 2):
                    cameras_file[c][0][i] = fix_multiplier(
                        cameras_file[c][0][i], multiplier[0]
                    )
                    cameras_file[c][0][i + 1] = fix_multiplier(
                        cameras_file[c][0][i + 1], multiplier[1]
                    )

                print(cameras_file[c])

        cameras_file.write(sparse_output / "cameras.txt")


    # write config file with default settings for training
    train_file_data = f"""
    gaussian_dim: 4
    time_duration: [0.0, {duration_seconds}]
    num_pts: {max_frame_idx}_000
    num_pts_ratio: 1.0
    rot_4d: True
    force_sh_3d: False
    batch_size: {num_cameras}
    exhaust_test: True

    ModelParams:
    sh_degree: 3
    source_path: "{args.output_folder}"
    model_path: "{args.output_folder}/model_output"
    images: "images"
    resolution: 1
    white_background: False
    data_device: "cuda"
    eval: False
    extension: ".png"
    num_extra_pts: 0
    loaded_pth: ""
    frame_ratio: 1
    dataloader: True

    PipelineParams:
    convert_SHs_python: False
    compute_cov3D_python: False
    debug: False
    env_map_res: 0
    env_optimize_until: 5000
    env_optimize_from: 0
    eval_shfs_4d: True

    OptimizationParams:
    iterations: 30_000
    position_lr_init: 0.00016
    position_t_lr_init: -1.0
    position_lr_final: 0.0000016
    position_lr_delay_mult: 0.01
    position_lr_max_steps: 30_000
    lowfeature_lr: 0.0025
    highfeature_lr: 0.005
    opacity_lr: 0.05
    scaling_lr: 0.005
    rotation_lr: 0.001
    percent_dense: 0.01
    thresh_opa_prune: 0.005
    densification_interval: 100
    opacity_reset_interval: 3000
    densify_from_iter: 500
    densify_until_iter: 15_000
    densify_grad_threshold: 0.0002
    densify_grad_t_threshold: 0.0002 / 40
    densify_grad_abs_threshold: 0.0012
    densify_until_num_points: 3000000
    final_prune_from_iter: -1
    sh_increase_start: 5000
    sh_increase_interval: 1000
    lambda_opa_mask: 0.0
    lambda_rigid: 0.0
    lambda_motion: 0.0
    lambda_depth: 0.2
    lambda_dssim: 0.2
    lambda_opa_bistable: 0.2
    prune_st_score_threshold: 0.1
    fastgs_num_sample_cams: 90
    fastgs_final_prune_num_sample_cams: 90
    fastgs_final_prune_from_iter: 20_000
    fastgs_final_prune_min_opacity: 0.05
    fastgs_final_prune_interval: 1000
    fastgs_densify_repeat_count: 3
    fastgs_loss_thresh: 0.01
    split_on_long_axis: True
    lff_diffscale: True
    lff_densify_grad_threshold: 0.0002
    lff_densify_from_iter: 550
    time_space_pruning_iterations: 25000

    """.replace("\\", "\\\\")
    (args.output_folder / "config.yaml").write_text(train_file_data)

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

from pyparsing import line

from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getProjectionMatrixCenterShift,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_slerp,
)


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


class ColmapRunner:
    def __init__(self, root_path):
        if root_path.endswith("/"):
            root_path = root_path[:-1]
        self.root_path = root_path

    def run_cmd(self, args):
        try:
            output_path_index = args.index("--output_path")
            output_folder = Path(args[output_path_index + 1])
            if output_folder.is_file() or output_folder.suffix != "":
                output_folder = output_folder.parent
            output_folder.mkdir(parents=True, exist_ok=True)
        except ValueError:
            pass
        args = [str(x) if type(x) != str else x for x in args]
        args = [
            (
                x.replace(self.root_path, "/working").replace("\\", "/")
                if x.startswith(self.root_path)
                else x
            )
            for x in args
        ]
        args = [
            "wsl",
            "docker",
            "run",
            "--runtime=nvidia",
            f"-v",
            f".:/working",
            "colmap/colmap:latest",
        ] + args
        print("Running colmap command:", " ".join(args))
        result = subprocess.check_output(args, text=True)
        print(f"Command output: {result}")
        args = [
            "docker",
            "run",
            "--runtime=nvidia",
            "-it",
            f"-v",
            f".:/working",
            "colmap/colmap:latest",
            "chown",
            "-R",
            # f"{os.getuid()}:{os.getgid()}",
            "/working",
        ]
        # result = subprocess.run(args, text=True)
        print("Chown done")


args = parser.parse_args()
if args.output_folder is None:
    args.output_folder = Path("output") / (args.video_folder.stem)
args.output_folder = args.output_folder.resolve()
colmap = ColmapRunner(os.getcwd())

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
    image_folder = args.output_folder / "images"
    if (
        not image_folder.exists()
        or not (
            image_folder / f"cam{camera_id_from_path(video_path,i)}_00000.png"
        ).exists()
    ):
        print(f"Extracting frames from {video_path} to {image_folder}")
        image_folder.mkdir(exist_ok=True)
        print(f"Getting frames from {video_path}")
        result = subprocess.check_output(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-start_number",
                "0",
            ]
            + ffmpeg_time_limit
            + [
                str(image_folder / f"cam{camera_id_from_path(video_path,i)}_%05d.png"),
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
image_folder = args.output_folder / "images"
colmap_images_folder = colmap_path / "images"
colmap_images_folder.mkdir(exist_ok=True)
for image in image_folder.glob("*_00000.png"):
    if split_frame_name(image.name)[1] != 0:
        shutil.copy(image, colmap_images_folder)

colmap_database = colmap_path / "database.db"


if args.use_existing_scene_poses is not None:
    print("Using existing scene poses from", args.use_existing_scene_poses)
    existing_scene_sparse = args.use_existing_scene_poses / "colmap" / "text_model"
    images_file = ColmapFile(
        existing_scene_sparse / "images.txt", 2
    )  # verify we can read the existing poses
    cameras_file = ColmapFile(
        existing_scene_sparse / "cameras.txt", 1
    )  # verify we can read the existing poses
    points_file = ColmapFile(None, 1)
    for x in images_file.keys():
        images_file[x][-1] = []

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
                sys.exit(-1)
            multiplier = multiplier[0]

            def fix_multiplier(value, multiplier):
                return str(float(value) * multiplier)

            cameras_file[c][0][2:7] = [
                fix_multiplier(val, multiplier) for val in cameras_file[c][0][2:7]
            ]
            print(cameras_file[c])

    copied_data_folder = colmap_path / "copied_data"
    copied_data_folder.mkdir(exist_ok=True)

    sparse_output = colmap_path / "sparse"
    sparse_output.mkdir(exist_ok=True)

    dense_output = colmap_path / "dense" / "0"
    dense_output.mkdir(parents=True, exist_ok=True)

    images_file.write(copied_data_folder / "images.txt")
    points_file.write(copied_data_folder / "points3D.txt")
    cameras_file.write(copied_data_folder / "cameras.txt")

    colmap.run_cmd(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            colmap_database,
            "--image_path",
            colmap_images_folder,
            "--ImageReader.single_camera",
            "1",
        ]
    )

    colmap.run_cmd(["colmap", "exhaustive_matcher", "--database_path", colmap_database])
    colmap.run_cmd(
        [
            "colmap",
            "point_triangulator",
            "--database_path",
            colmap_database,
            "--image_path",
            colmap_images_folder,
            "--input_path",
            copied_data_folder,
            "--output_path",
            sparse_output,
        ]
    )
    colmap.run_cmd(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            colmap_images_folder,
            "--input_path",
            sparse_output,
            "--output_path",
            dense_output,
        ]
    )

    colmap.run_cmd(["colmap", "patch_match_stereo", "--workspace_path", dense_output])
    colmap.run_cmd(
        [
            "colmap",
            "stereo_fusion",
            "--workspace_path",
            dense_output,
            "--output_path",
            dense_output / "fused.ply",
            "--StereoFusion.min_num_pixels",
            "1",
        ]
    )

    colmap_dense_sparse = colmap_path / "dense" / "0" / "sparse"

    colmap.run_cmd(
        [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            colmap_dense_sparse,
            "--output_path",
            colmap_dense_sparse,
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "1",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustment.max_num_iterations",
            "10000",
        ]
    )
    # now we should have everything ready to go
else:

    # use colmap auto reconstruction
    colmap.run_cmd(
        [
            "colmap",
            "automatic_reconstructor",
            "--workspace_path",
            colmap_path,
            "--image_path",
            colmap_images_folder,
            "--single_camera",
            "1",
        ]
    )

    # bundle adjustment
    colmap_dense_sparse = colmap_path / "dense" / "0" / "sparse"
    colmap.run_cmd(
        [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            colmap_dense_sparse,
            "--output_path",
            colmap_dense_sparse,
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "1",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustment.max_num_iterations",
            "10000",
        ]
    )    

# copy the output into output_folder/sparse/0
colmap_dense_sparse = colmap_path / "dense" / "0" / "sparse"
sparse_output = args.output_folder / "sparse" / "0"


# convert the final model to text format so we can read the focal length etc.
(colmap_path / "text_model").mkdir(exist_ok=True)
colmap.run_cmd(
    [
        "colmap",
        "model_converter",
        "--input_path",
        colmap_dense_sparse,
        "--output_type",
        "TXT",
        "--output_path",
        sparse_output,
    ]
)


# add cameras that colmap failed to find or
# copy cameras and frames from other scene entirely
# n.b. whichever way the cameras will need rotating / translating to match
# our coordinate frame
if (
    args.import_missing_cameras_from_scene is not None
    or args.replace_cameras_from_scene is not None
):
    import_scene_path = args.replace_cameras_from_scene
    if import_scene_path is None:
        import_scene_path = args.import_missing_cameras_from_scene
    our_camera_list = ColmapFile(sparse_output / "images.txt", 2)
    import_camera_list = ColmapFile(
        import_scene_path / "colmap" / "text_model" / "images.txt", 2
    )

    our_camera_frames = ColmapFile(sparse_output / "frames.txt", 1)
    import_camera_frames = ColmapFile(
        import_scene_path / "colmap" / "text_model" / "frames.txt", 1
    )

    import_to_our_camera_id_map = {}

    for other_cam_id, other_cam_lines in import_camera_list.items():
        other_cam_name = other_cam_lines[0][-1]
        import_to_our_camera_id_map[other_cam_id] = None
        for cam_id, cam_lines in our_camera_list.items():
            frame_name = cam_lines[0][-1]
            if other_cam_name == frame_name:
                import_to_our_camera_id_map[other_cam_id] = cam_id
                print("Old frame exists in our colmap reconstruction", frame_name)
                break

    print(import_to_our_camera_id_map)
    transform_estimates = []
    for other_cam_id, our_cam_id in import_to_our_camera_id_map.items():
        if our_cam_id is not None:
            our_camera_to_world = colmap_image_line_to_camera_to_world(
                our_camera_list[our_cam_id][0]
            )
            other_camera_to_world = colmap_image_line_to_camera_to_world(
                import_camera_list[other_cam_id][0]
            )

            # COLMAP stores images.txt poses as world-to-camera. Convert both to
            # camera-to-world and solve:
            #   C2W_ours = T_theirs_to_ours @ C2W_theirs
            from_import_to_ours = our_camera_to_world @ np.linalg.inv(
                other_camera_to_world
            )
            transform_estimates.append(from_import_to_ours)
            print(
                f"CamID ours:{our_cam_id} old {other_cam_id} transform: {from_import_to_ours}"
            )

    if len(transform_estimates) == 0:
        raise ValueError(
            "Could not estimate transform from imported reconstruction: no overlapping cameras were found."
        )

    theirs_to_ours_matrix = average_rigid_transforms(transform_estimates)

    translation_errors = []
    rotation_errors = []
    for other_cam_id, our_cam_id in import_to_our_camera_id_map.items():
        if our_cam_id is None:
            continue

        our_camera_to_world = colmap_image_line_to_camera_to_world(
            our_camera_list[our_cam_id][0]
        )
        other_camera_to_world = colmap_image_line_to_camera_to_world(
            import_camera_list[other_cam_id][0]
        )
        estimated_camera_to_world = theirs_to_ours_matrix @ other_camera_to_world

        translation_error = np.linalg.norm(
            estimated_camera_to_world[:3, 3] - our_camera_to_world[:3, 3]
        )
        rotation_error = rotation_angle_degrees(
            estimated_camera_to_world[:3, :3] @ our_camera_to_world[:3, :3].T
        )
        translation_errors.append(float(translation_error))
        rotation_errors.append(float(rotation_error))
        print(
            f"CamID ours:{our_cam_id} old {other_cam_id} errors: "
            f"translation={translation_error:.6f}, rotation_deg={rotation_error:.6f}"
        )

    print("Transform from import to our reconstruction:", theirs_to_ours_matrix)
    print(
        "Transform residual summary: "
        f"translation_mean={np.mean(translation_errors):.6f}, "
        f"translation_max={np.max(translation_errors):.6f}, "
        f"rotation_mean_deg={np.mean(rotation_errors):.6f}, "
        f"rotation_max_deg={np.max(rotation_errors):.6f}"
    )

    if args.replace_cameras_from_scene:
        new_camera_list = ColmapFile(None, 2)
        new_camera_frames = ColmapFile(None, 1)
        new_camera_list.headers = our_camera_list.headers + ["# replaced"]
        new_camera_frames.headers = our_camera_frames.headers + ["# replaced"]
    else:
        new_camera_list = our_camera_list
        new_camera_frames = our_camera_frames

    for other_cam_id, our_cam_id in import_to_our_camera_id_map.items():
        if our_cam_id is not None and args.replace_cameras_from_scene is None:
            continue

        other_camera_to_world = colmap_image_line_to_camera_to_world(
            import_camera_list[other_cam_id][0]
        )
        estimated_camera_to_world = theirs_to_ours_matrix @ other_camera_to_world
        estimated_world_to_camera = np.linalg.inv(estimated_camera_to_world)

        rotation, translation = camera_to_world_to_colmap_pose(
            estimated_camera_to_world
        )
        new_cam_lines = import_camera_list[other_cam_id]
        new_cam_lines[1] = []
        new_cam_lines[0][1:5] = map(str, rotation)
        new_cam_lines[0][5:8] = map(str, translation)
        new_cam_id = 1 if len(new_camera_list) == 0 else max(new_camera_list.keys()) + 1
        new_camera_list[new_cam_id] = new_cam_lines
        print(f"Added camera {new_cam_id} for missing camera {other_cam_id}")
        # add this camera to frames.txt
        new_frame_id = (
            1 if len(new_camera_frames) == 0 else max(new_camera_frames.keys()) + 1
        )
        for frame_id, frame_lines in import_camera_frames.items():
            import_camera_id = int(frame_lines[0][-1])
            if import_camera_id == other_cam_id:
                new_frame_lines = copy.deepcopy(frame_lines)
                new_frame_lines[0][0] = str(new_frame_id)
                new_frame_lines[0][-1] = str(new_cam_id)
                new_frame_lines[0][2:6] = map(str, rotation)
                new_frame_lines[0][6:9] = map(str, translation)
                new_camera_frames[new_frame_id] = new_frame_lines
                break

    print("Frames after map", list(new_camera_frames.keys()))

    remap_colmap_ids_to_filename_camera_ids(new_camera_list, new_camera_frames)

    new_camera_list.write(sparse_output / "images.txt")
    new_camera_frames.write(sparse_output / "frames.txt")

# find max frame number
all_frames = (args.output_folder / "images").glob("*.png")
max_frame_idx = 0
for x in all_frames:
    _base, cam_idx, frame_idx = split_frame_name(x.name)
    max_frame_idx = max(frame_idx, max_frame_idx)

print("Max frame index:", max_frame_idx)

duration_seconds = max_frame_idx / 30

# write config file with default settings for training
train_file_data = f"""
gaussian_dim: 4
time_duration: [0.0, {duration_seconds}]
num_pts: {max_frame_idx}_000
num_pts_ratio: 1.0
rot_4d: True
force_sh_3d: False
batch_size: 4
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
  env_map_res: 500
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
  lambda_dssim: 0.2
  thresh_opa_prune: 0.005
  densification_interval: 100
  opacity_reset_interval: 3000
  densify_from_iter: 500
  densify_until_iter: 15_000
  densify_grad_threshold: 0.0002
  densify_grad_t_threshold: 0.0002 / 40
  densify_until_num_points: -1
  final_prune_from_iter: -1
  sh_increase_interval: 1000
  lambda_opa_mask: 0.0
  lambda_rigid: 0.0
  lambda_motion: 0.0
""".replace("\\", "\\\\")
(args.output_folder / "config.yaml").write_text(train_file_data)

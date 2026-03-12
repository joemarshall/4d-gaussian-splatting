import json
import subprocess
import argparse
import shutil
import os
import re
import numpy as np
from pathlib import Path


def parse_colmap_text(file_path: Path):
    lines = file_path.read_text().splitlines()
    lines = [line for line in lines if not line.strip().startswith("#")]
    lines = [line.strip().split(" ") for line in lines]
    return lines


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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--colmap_path", type=str, default="../colmap")
parser.add_argument("video_folder", type=Path)
parser.add_argument("-o", "--output_folder", type=Path)
parser.add_argument("-nf", "--num_frames", type=int, default=-1)


class ColmapRunner:
    def __init__(self, root_path):
        if root_path.endswith("/"):
            root_path = root_path[:-1]
        self.root_path = root_path

    def run_cmd(self, args):
        try:
            output_path_index = args.index("--output_path")
            output_folder = Path(args[output_path_index + 1])
            if output_folder.is_file():
                output_folder = output_folder.parent
            output_folder.mkdir(parents=True, exist_ok=True)
        except ValueError:
            pass
        args = [
            x.replace(self.root_path, "/working").replace("\\", "/") if x.startswith(self.root_path) else x
            for x in args
        ]
        args = ["wsl",
            "docker",
            "run",
            "--runtime=nvidia",
            f"-v",
            f".:/working",
            "colmap/colmap:latest",
        ] + args
        print("Running colmap command:", args)
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
            #f"{os.getuid()}:{os.getgid()}",
            "/working",
        ]
        #result = subprocess.run(args, text=True)
        print("Chown done")


args = parser.parse_args()
if args.output_folder is None:
    args.output_folder = Path("output") / (args.video_folder.stem)
args.output_folder = args.output_folder.resolve()
colmap = ColmapRunner(os.getcwd())

args.output_folder.mkdir(parents=True, exist_ok=True)

if args.num_frames > 0:
    ffmpeg_time_limit = ["-t",str(args.num_frames / 30)]
else:
    ffmpeg_time_limit = []    

def camera_id_from_path(video_path,index):
    match = re.match(r"(\D)*(\d+)*", video_path.stem)
    if match:
        return int(match.groups()[1])
    return str(index)

# 1) for each mp4 in video folder, ffmpeg to extract frames into subfolder called images if the frames don't exist
for i,video_path in enumerate(sorted(args.video_folder.glob("*.mp4"))):
    image_folder = args.output_folder / "images"
    if (
        not image_folder.exists()
        or not (image_folder / f"cam{camera_id_from_path(video_path,i)}_00000.png").exists()
    ):
        print(f"Extracting frames from {video_path} to {image_folder}")
        image_folder.mkdir(exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-start_number",
                "0",
            ] + ffmpeg_time_limit + [
                str(image_folder / f"cam{camera_id_from_path(video_path,i)}_%05d.png"),
            ]
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


# use colmap auto reconstruction
colmap.run_cmd(["colmap", "automatic_reconstructor", "--workspace_path", str(colmap_path),"--image_path",str(colmap_images_folder),"--single_camera","1"])

#convert the final model to text format so we can read the focal length etc.
(colmap_path / "text_model").mkdir(exist_ok=True)
colmap.run_cmd(["colmap", "model_converter", "--input_path", str(colmap_path / "sparse/0"), "--output_type", "TXT","--output_path",str(colmap_path / "text_model")])

# make json file for train and test split
# read the data from the colmap cameras.txt file
camera_data = parse_colmap_text(colmap_path / "text_model/cameras.txt")
frames_data = parse_colmap_text(colmap_path / "text_model/frames.txt")
images_data = parse_colmap_text(colmap_path / "text_model/images.txt")

combined_images_data = []
for x in range(0, len(images_data) // 2):
    combined_images_data.append((images_data[x * 2], images_data[x * 2 + 1]))

camera_id_to_pose_index = {
    split_frame_name(x[0][-1])[1]: int(x[0][0]) for x in combined_images_data
}
print(list(camera_id_to_pose_index.keys()))


poses = {}

# get poses for each camera
for x in frames_data:
    if len(x) < 10:
        continue
    pose_index = int(x[-1])
    qw, qx, qy, qz = map(float, x[2:6])
    tx, ty, tz = map(float, x[6:9])
    print(tx, ty, tz)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = quat_to_rot(qw, qx, qy, qz)
    transform_matrix[:3, 3] = [tx, ty, tz]
    poses[pose_index] = transform_matrix.tolist()


camera_data = camera_data[0]
W = int(camera_data[2])
H = int(camera_data[3])
fx = float(camera_data[4])
fy = float(camera_data[5])
cx = float(camera_data[6])
cy = float(camera_data[7])

all_camera_ids = sorted(list(camera_id_to_pose_index.keys()))
print("All camera ids:", all_camera_ids)
first_camera_id = all_camera_ids[0]

train_frames = []
test_frames = []
all_frames = {split_frame_name(x.name)[1:3]: x for x in image_folder.glob("*.png")}
max_frame_idx = 0
min_frame_idx = 0

while True:
    missing_frame = False
    for cam_id in all_camera_ids:
        if (cam_id, max_frame_idx) not in all_frames:
            missing_frame = True
            break
    if missing_frame:
        break
    max_frame_idx += 1

print("Max frame index:", max_frame_idx)
frame_valid = all_frames.keys()
frame_valid = sorted([x[0] for x in frame_valid if x[1] == 0])
print(frame_valid)
print(all_camera_ids)


for frame_idx in range(max_frame_idx):
    for camera_id in all_camera_ids:
        if (camera_id, frame_idx) not in all_frames:
            print(
                f"Missing frame for camera {camera_id} at index {frame_idx}, skipping this frame."
            )
            continue

        frame_data = {
            "file_path": str(
                all_frames[(camera_id, frame_idx)].relative_to(args.output_folder)
            ),
            "transform_matrix": poses[camera_id_to_pose_index[camera_id]],
            "time": frame_idx / 30,
        }
        if camera_id == first_camera_id:
            test_frames.append(frame_data)
        else:
            train_frames.append(frame_data)

duration_seconds = max_frame_idx / 30

# copy colmap/dense/0/sparse to outputfolder/sparse
colmap_dense_sparse = colmap_path / "dense" / "0" / "sparse"
sparse_output = args.output_folder / "sparse" / "0"
sparse_output.mkdir(parents=True, exist_ok=True)
for file in colmap_dense_sparse.glob("*"):
    if file.is_file():
        shutil.copy(file, sparse_output)

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
  feature_lr: 0.0025
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
""".replace("\\","\\\\")
(args.output_folder / "config.yaml").write_text(train_file_data)

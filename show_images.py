import argparse
from pathlib import Path
import subprocess
import sys
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

parser = argparse.ArgumentParser()
parser.add_argument("output_folder", help="Output folder path")
parser.add_argument("-r", "--render", action="store_true", help="Render mode")
parser.add_argument("--test", action="store_true", help="Test mode")
parser.add_argument("--train", action="store_true", help="Training mode")
parser.add_argument("--limit", type=int, default=-1, help="Limit the number of images to show")
parser.add_argument(
    "--render_camera",
    "-c",
    type=str,
    default="",
    help="Render a specific camera index (1-based) instead of all cameras. If multiple cameras are" \
    "given in a comma separated list it will lerp between them over time.",
)
args = parser.parse_args()

if len(args.render_camera)>0:
    args.render_camera = [int(x) for x in args.render_camera.split(",")]

if not args.test and not args.train:
    args.test = True
    args.train = True


def camera_id_and_frame_from_path(video_path):
    match = re.match(r"(?:\D)*(\d+)_(\d+)", video_path)
    if match:
        return int(match.groups()[0]), int(match.groups()[1])
    return ""


def render_set(model_path, iteration, views, gaussians, pipeline, background):
    # remove old PNG files
    for x in list(Path(model_path).glob("t*/**/*.png")):
        x.unlink()

    for idx, (name,view) in enumerate(tqdm(views, desc="Rendering progress")):
        render_path = Path(model_path) / name / f"ours_{iteration}" / "renders"
        gts_path = Path(model_path) / name / f"ours_{iteration}" / "gt"
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        @torch.compile
        def render_wrapper(args):
            return render(*args)
        rendering = render_wrapper((view[1].cuda(), gaussians, pipeline, background))["render"]
        gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, render_path / f"{idx:05d}.png")
        torchvision.utils.save_image(gt, gts_path / f"{idx:05d}.png")

try:
    if args.render:
        checkpoints = Path(args.output_folder).glob("*.pth")
        sorted_checkpoints = list(sorted(checkpoints, key=lambda x: x.stat().st_mtime))
        if len(sorted_checkpoints):
            latest_pth = sorted_checkpoints[-1]
            print(f"Rendering from checkpoint: {latest_pth}")
        else:
            print("No checkpoints found")
            sys.exit(-1)

        with torch.no_grad():
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

            model = model.extract(render_args)
            pipeline = pipeline.extract(render_args)

            gaussians = GaussianModel(model.sh_degree, gaussian_dim=4, rot_4d=True)
            scene = Scene(model, gaussians, shuffle=False)

            bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            print(scene.getTestCameras())
            print(scene.getTrainCameras())

            all_cameras = []
            camera_sets = []
            if args.test:
                camera_sets.append(("test", scene.getTestCameras()))
            if args.train:
                camera_sets.append(("train", scene.getTrainCameras()))

            # if multiple cameras are given, render frames from the first camera,
            # but lerp the position
            master_camera = None
            other_cameras =[]
            if len(args.render_camera) > 0:
                master_camera = args.render_camera[0]
                if len(args.render_camera)>1:
                    other_cameras = args.render_camera
            other_camera_info = {}
            max_frame = 0
            filtered_cameras = []
            for name, cameras in camera_sets:
                if master_camera is not None:
                    for x in cameras:
                        camera_id, frame_id = camera_id_and_frame_from_path(x[1].image_name)
                        if camera_id == master_camera:
                            filtered_cameras.append((name,x))
                        if camera_id in other_cameras and frame_id ==0:
                            # this camera is just used as a reference for lerping, so we only need
                            # the first frame
                            other_camera_info[camera_id] = x
                        max_frame = max(max_frame, frame_id)
                else:
                    filtered_cameras = all_cameras
            if master_camera is not None and len(other_cameras)>0:
                print(f"Rendering with master camera {master_camera} and lerping to other cameras {other_cameras}")
                lerped_cameras = []
                for x in range(len(filtered_cameras)):
                    name, cam = filtered_cameras[x]
                    camera_id, frame_id = camera_id_and_frame_from_path(cam[1].image_name)
                    frame_fraction = (frame_id / max_frame)*(len(other_cameras)-1)
                    other_cam_index = int(frame_fraction)
                    lerp_factor = frame_fraction - other_cam_index
                    cam_before = other_camera_info[other_cameras[other_cam_index]]
                    cam_after = other_camera_info[other_cameras[min(other_cam_index+1, len(other_cameras)-1)]]
                    new_cam = deepcopy(cam[1])
                    print(f"Lerping transform! Camera: {camera_id}, Frame: {frame_id}, Frame fraction: {frame_fraction}, Between: {other_cameras[other_cam_index], other_cameras[min(other_cam_index+1, len(other_cameras)-1)]}")
                    new_cam.lerp_transform(cam_before[1], cam_after[1], lerp_factor)
                    lerped_cameras.append((name, (cam[0], new_cam)))
                render_set(
                    model.model_path,
                    scene.loaded_iter,
                    lerped_cameras,
                    gaussians,
                    pipeline,
                    background,
                )

            else:
                render_set(
                    model.model_path,
                    scene.loaded_iter,
                    filtered_cameras,
                    gaussians,
                    pipeline,
                    background,
                )



            if len(filtered_cameras) == 0:
                print(f"No cameras found for {name} with the specified index.")
            else:
                print(f"Rendering {len(filtered_cameras)} cameras for {name} set.")
                render_set(
                    model.model_path,
                    name,
                    scene.loaded_iter,
                    filtered_cameras,
                    gaussians,
                    pipeline,
                    background,
                )
except KeyboardInterrupt:
    print("Rendering interrupted by user, showing output so far")

output_folder = Path(args.output_folder)

files_in_order = []

if args.test:
    test_folders = output_folder.glob("test/**/renders/")
    for test_folder in test_folders:
        if test_folder.exists() and test_folder.is_dir():
            all_files = sorted(list(test_folder.glob("*.png")))
            if args.limit > 0 and len(all_files) > args.limit:
                all_files = all_files[: args.limit]
            if len(all_files) > 0:
                #subprocess.run(["timg", "-p", "s"] + all_files)
                files_in_order.extend(all_files)

if args.train:
    training_folders = output_folder.glob("train/**/renders/")
    for training_folder in training_folders:
        if training_folder.exists() and training_folder.is_dir():
            all_files = sorted(list(training_folder.glob("*.png")))
            if args.limit > 0 and len(all_files) > args.limit:
                all_files = all_files[: args.limit]
            if len(all_files) > 0:
               # subprocess.run(["timg", "-p", "s"] + all_files)
                files_in_order.extend(all_files)

subprocess.run(["mpv","--no-correct-pts","--merge-files=yes","-mf-fps=30",*files_in_order,"--loop=10"])
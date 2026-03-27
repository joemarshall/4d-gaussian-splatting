import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

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

import sys

OUR_PATH = str(Path(__file__).parent.resolve())
print(OUR_PATH)

def trace_fn(frame, event, arg):
    global OUR_PATH
    fname = frame.f_code.co_filename
    if fname.find(OUR_PATH) <0:
        return None
    print(f"Trace: {event}:{arg} in {fname} at line {frame.f_lineno}")
    #if frame.f_code.co_name.find("show_images.py") >= 0:
    #    return trace_fn
    return None

#sys.settrace(trace_fn)

parser = argparse.ArgumentParser()
parser.add_argument("output_folder", help="Output folder path")
parser.add_argument( "--render","-r", action="store_true", help="Render mode")
parser.add_argument("--test", action="store_true", help="Test mode")
parser.add_argument("--train", action="store_true", help="Training mode")
parser.add_argument("--limit", type=int, default=-1, help="Limit the number of images to show")
parser.add_argument("--save_video_name","-f", type=str, help="Render a video")
parser.add_argument("--play_video","-p", action="store_true", help="Play the rendered video")
parser.add_argument(
    "--render_camera",
    "-c",
    type=str,
    default="",
    help="Render a specific camera index (1-based) instead of all cameras. If multiple cameras are" \
    "given in a comma separated list it will lerp between them over time.",
)
parser.add_argument(
    "--graph_camera_positions",
    "-g",
    action="store_true",
    help="Show a matplotlib 3D scatter of frame-0 camera positions color-coded by camera ID."
    " If camera lerping is enabled, also overlays the generated interpolation track.",
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


def graph_camera_positions_3d(camera_sets, lerp_track_positions=None, lerp_track_cameras=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping camera position plot")
        return

    frame_zero_cameras = {}
    for _, cameras in camera_sets:
        for cam_data in cameras:
            camera_id, frame_id = camera_id_and_frame_from_path(cam_data[1].image_name)
            if frame_id == 0 and camera_id not in frame_zero_cameras:
                frame_zero_cameras[camera_id] = cam_data[1]

    if len(frame_zero_cameras) == 0:
        print("No frame-0 camera positions found to plot")
        return

    sorted_camera_ids = sorted(frame_zero_cameras.keys())
    cmap = plt.get_cmap("tab20", max(len(sorted_camera_ids), 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_positions = [
        frame_zero_cameras[camera_id].camera_center.detach().cpu() for camera_id in sorted_camera_ids
    ]
    stacked_positions = torch.stack(all_positions, dim=0)
    position_extent = (stacked_positions.max(dim=0).values - stacked_positions.min(dim=0).values).max()
    arrow_length = max(float(position_extent) * 0.08, 0.05)
    label_offset = arrow_length * 0.15

    for idx, camera_id in enumerate(sorted_camera_ids):
        camera = frame_zero_cameras[camera_id]
        pos = camera.camera_center.detach().cpu()
        c2w = torch.linalg.inv(camera.world_view_transform.transpose(0, 1)).detach().cpu()

        look_direction = c2w[:3, 2]
        look_direction = look_direction / (torch.linalg.norm(look_direction) + 1e-12)

        up_direction = c2w[:3, 1]
        up_direction = up_direction / (torch.linalg.norm(up_direction) + 1e-12)

        ax.scatter(
            float(pos[0]),
            float(pos[2]),
            float(-pos[1]),
            color=cmap(idx),
            s=40,
            label=f"Cam {camera_id}",
        )
        ax.text(
            float(pos[0]) + label_offset,
            float(pos[2]) + label_offset,
            float(-pos[1]) + label_offset,
            str(camera_id),
            color=cmap(idx),
            fontsize=9,
        )
        ax.quiver(
            float(pos[0]),
            float(pos[2]),
            float(-pos[1]),
            float(look_direction[0]),
            float(look_direction[2]),
            float(-look_direction[1]),
            length=arrow_length,
            normalize=True,
            color=cmap(idx),
            linewidth=1.5,
            arrow_length_ratio=0.25,
        )
        ax.quiver(
            float(pos[0]),
            float(pos[2]),
            float(-pos[1]),
            float(up_direction[0]),
            float(up_direction[2]),
            float(up_direction[1]),
            length=arrow_length,
            normalize=True,
            color="gray",
            linewidth=1.2,
            arrow_length_ratio=0.25,
        )

    if lerp_track_positions is not None and len(lerp_track_positions) > 0:
        xs = [float(pos[0]) for pos in lerp_track_positions]
        ys = [float(pos[2]) for pos in lerp_track_positions]
        zs = [float(-pos[1]) for pos in lerp_track_positions]
        ax.plot(xs, ys, zs, color="black", linewidth=2.0, label="Lerp track")

        if len(lerp_track_positions) > 1:
            direction_stride = 5
            direction_arrow_length = arrow_length * 0.7
            for index in range(0, len(lerp_track_positions) - 1, direction_stride):
                current_pos = lerp_track_positions[index]
                next_pos = lerp_track_positions[index + 1]
                direction = (next_pos - current_pos).detach().cpu()
                direction = direction / (torch.linalg.norm(direction) + 1e-12)

                ax.quiver(
                    float(current_pos[0]),
                    float(current_pos[2]),
                    float(-current_pos[1]),
                    float(direction[0]),
                    float(direction[2]),
                    float(-direction[1]),
                    length=direction_arrow_length,
                    normalize=True,
                    color="black",
                    linewidth=1.3,
                    arrow_length_ratio=0.4,
                )

    if lerp_track_cameras is not None and len(lerp_track_cameras) > 0:
        sample_stride = 5
        sampled_indices = list(range(0, len(lerp_track_cameras), sample_stride))
        if sampled_indices[-1] != len(lerp_track_cameras) - 1:
            sampled_indices.append(len(lerp_track_cameras) - 1)

        track_arrow_length = arrow_length * 0.9
        for index in sampled_indices:
            camera = lerp_track_cameras[index]
            pos = camera.camera_center.detach().cpu()
            c2w = torch.linalg.inv(camera.world_view_transform.transpose(0, 1)).detach().cpu()

            look_direction = c2w[:3, 2]
            look_direction = look_direction / (torch.linalg.norm(look_direction) + 1e-12)

            up_direction = c2w[:3, 1]
            up_direction = up_direction / (torch.linalg.norm(up_direction) + 1e-12)

            ax.quiver(
                float(pos[0]),
                float(pos[2]),
                float(-pos[1]),
                float(look_direction[0]),
                float(look_direction[2]),
                float(-look_direction[1]),
                length=track_arrow_length,
                normalize=True,
                color="black",
                linewidth=1.4,
                arrow_length_ratio=0.25,
            )
            ax.quiver(
                float(pos[0]),
                float(pos[2]),
                float(-pos[1]),
                float(up_direction[0]),
                float(up_direction[2]),
                float(up_direction[1]),
                length=track_arrow_length,
                normalize=True,
                color="dimgray",
                linewidth=1.1,
                arrow_length_ratio=0.25,
            )

    ax.set_title("Frame-0 Camera Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.tight_layout()
    plt.show()


@torch.compile
def render_wrapper(args):
    return render(*args)


def render_set(model_path, iteration, views, gaussians, pipeline, background):
    # remove old PNG files
    for x in list(Path(model_path).glob("t*/**/*.png")):
        x.unlink()

    for idx, (name,view) in enumerate(tqdm(views, desc="Rendering progress")):
        render_path = Path(model_path) / name / f"ours_{iteration}" / "renders"
        gts_path = Path(model_path) / name / f"ours_{iteration}" / "gt"
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        rendering = render_wrapper((view[1].cuda(), gaussians, pipeline, background))["render"]
        #gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, render_path / f"{idx:05d}.png")
        #torchvision.utils.save_image(gt, gts_path / f"{idx:05d}.png")

try:
    if args.render or args.graph_camera_positions:
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

            print(f"Loaded model, {len(gaussians.get_xyz)} gaussians")

            bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
            lerp_track_positions = []
            print("BOO")
            for name, cameras in camera_sets:
                if master_camera is not None:
                    cameras.set_names_only(True)
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
                    for x in cameras:
                        filtered_cameras.append((name,x))
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

                if args.graph_camera_positions:
                    sorted_lerped_cameras = sorted(
                        lerped_cameras,
                        key=lambda item: camera_id_and_frame_from_path(item[1][1].image_name)[1],
                    )
                    lerp_track_positions = [
                        entry[1][1].camera_center.detach().cpu() for entry in sorted_lerped_cameras
                    ]
                    lerp_track_pose_cameras = [entry[1][1] for entry in sorted_lerped_cameras]
                    graph_camera_positions_3d(
                        camera_sets,
                        lerp_track_positions=lerp_track_positions,
                        lerp_track_cameras=lerp_track_pose_cameras,
                    )
                if args.render:
                    render_set(
                        model.model_path,
                        scene.loaded_iter,
                        lerped_cameras,
                        gaussians,
                        pipeline,
                        background,
                    )

            else:
                if args.graph_camera_positions:
                    graph_camera_positions_3d(camera_sets)
                if args.render:
                    render_set(
                        model.model_path,
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

extra_args = []
if os.name == "nt":
    extra_args = ["-vo=gpu"]

if not args.save_video_name and not args.graph_camera_positions:
    args.play_video= True

if args.play_video and len(files_in_order)>0:
    subprocess.run(["mpv","--no-correct-pts","--merge-files=yes","-mf-fps=30",*files_in_order,"--loop=10"]+extra_args )

if args.save_video_name and len(files_in_order)>0:
    with tempfile.TemporaryDirectory() as temp_dir:
        files_text_path = Path(temp_dir) / "files.txt"
        with open(files_text_path, "wb") as f:
            for file in files_in_order:                
                f.write(f"file '{str(Path(file).absolute())}'\n".encode())
                f.write(f"duration 0.033\n".encode())
            f.close()
        subprocess.run(["ffmpeg", "-y", "-safe","0","-f","concat", "-i", files_text_path, "-framerate", "30","-vf","crop=floor(in_w/4)*4:floor(in_h/4)*4:0:0", "-pix_fmt", "yuv420p", args.save_video_name])
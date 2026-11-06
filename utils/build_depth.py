import itertools
import os
from pathlib import Path
import re
import time
from depth_anything_3 import model
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype

torch._dynamo.config.capture_scalar_outputs = True

from concurrent.futures import ThreadPoolExecutor, wait


def frame_and_camid_from_path(video_path):
    match = re.match(r"(?:\D)*(\d+)_(\d+)", video_path)
    if match:
        return (
            int(match.groups()[1]),
            int(match.groups()[0]),
        )
    return ""


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


class ModelWrapper(torch.nn.Module):
    """
    Thin wrapper around DepthAnything3 that accepts N image tensors, intrinsic and extrinsic matrices
    of shape (N, 3, H, W) and outputs depth (N, 1, H, W)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # underlying nn.Module

    def forward(self, images, extrinsics, intrinsics):
        out = self.model(
            images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_feat_layers=[],
            infer_gs=False,
        )
        return out["depth"]  # (B, N, H, W)


def make_onnx_depth_model(model_name, output_path):
    device = torch.device("cpu")
    print("Loading PyTorch model: {}".format(model_name))
    pt_model = DepthAnything3.from_pretrained(model_name)

    wrapper = OnnxWrapper(pt_model)
    wrapper.eval()
    wrapper = wrapper.to(device=device)

    torch.onnx.export(
        wrapper,
        (
            torch.zeros((2, 10, 3, 518, 518), device=device),
            torch.zeros((2, 10, 4, 4), device=device),
            torch.zeros((2, 10, 3, 3), device=device),
        ),
        input_names=["image", "extrinsics", "intrinsics"],
        output_names=["depth"],
        dynamic_axes={
            "image": {0: "Batch", 1: "N", 2: "H", 3: "W"},
            "depth": {0: "Batch", 1: "N", 2: "H", 3: "W"},
            "extrinsics": {0: "Batch", 1: "N"},
            "intrinsics": {0: "Batch", 1: "N"},
        },
        dynamo=False,
        opset_version=17,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print("Exported onnx model to {}".format(output_path))


def get_depth_rescale_and_clipping_size(src_size):
    scale_size = src_size
    if max(scale_size) > 2000:
        scale_size = (scale_size[0] // 2, scale_size[1] // 2)
        # pad image to make it multiple of 14 on both axes (patch size)
    clipped_size = (
        (scale_size[0] // 14) * 14,
        (scale_size[1] // 14) * 14,
    )
    total_pad = (scale_size[0] - clipped_size[0], scale_size[1] - clipped_size[1])
    pad_start = (total_pad[0] // 2, total_pad[1] //2)
    pad_end = (total_pad[0] - pad_start[0], total_pad[1] - pad_start[1])
    #print(f"Source size: {src_size}, Clipped size: {clipped_size}, Scale size: {scale_size}, Pad start: {pad_start}, Pad end: {pad_end}")
    return (scale_size, clipped_size,pad_start,pad_end)

def transform_image_to_depth_size(image):
    """
    Transform the input image to match the depth model's expected input size.
    This includes resizing and clipping the image to make it a multiple of 14 on both axes.
    """
    scale_size, clip_size, pad_start, pad_end = get_depth_rescale_and_clipping_size((image.width, image.height))
    if scale_size != (image.width, image.height):
        image = image.resize(scale_size, Image.Resampling.LANCZOS)
    if clip_size != scale_size:
        clip_rect = (pad_start[0], pad_start[1], pad_start[0] + clip_size[0], pad_start[1] + clip_size[1])
        image = image.crop(clip_rect)
    return image

def interpolate_2d(tensor,size,mode="nearest"):
    orig_shape = tensor.shape
    while len(tensor.shape) < 4:
        tensor = tensor.unsqueeze(0)
    new_shape = orig_shape[:-2] + size
    retval =torch.nn.functional.interpolate(tensor, size=size, mode=mode)
    assert retval.shape[-2:] == size, f"Interpolated tensor shape {retval.shape[-2:]} does not match expected size {size}"
    retval = retval.reshape(new_shape)
    return retval

    

def transform_depth_to_image_size(depth_tensor,image_tensor):
    """
    Transform the input tensor to match the original image size.
    This includes resizing and padding the tensor with -1
    """
    h, w = image_tensor.shape[-2:]
    scale_size, clip_size, pad_start, pad_end = get_depth_rescale_and_clipping_size((w, h))
    assert clip_size==(depth_tensor.shape[-1],depth_tensor.shape[-2]),f"Depth tensor shape {depth_tensor.shape[-2:]} does not match expected clip size {clip_size}"

    if clip_size == (w,h):
        return depth_tensor

    orig_shape = depth_tensor.shape
    while len(depth_tensor.shape) < 4:
        depth_tensor = depth_tensor.unsqueeze(0)


    if clip_size != scale_size:
        # expand by scale_size filled with -1 values
        depth_tensor = torch.nn.functional.pad(depth_tensor, (pad_start[0], pad_end[0], pad_start[1], pad_end[1]), mode='constant', value=-1)
        #print(depth_tensor.shape,scale_size,pad_start,pad_end)
        assert(depth_tensor.shape[-2:] == (scale_size[1],scale_size[0]))
    #print(depth_tensor.shape,scale_size,w,h)
    if scale_size != (w,h):
        depth_tensor = interpolate_2d(depth_tensor, size=(h,w))


    return depth_tensor.reshape(orig_shape[:-2] + (h,w))

def transform_tensor_to_depth_size(tensor):
    """
    Transform the input (image sized) tensor to match the depth model's expected input size.
    This includes resizing and clipping the tensor to make it a multiple of 14 on both axes.
    """
    h, w = tensor.shape[-2:]
    scale_size, clip_size, pad_start, pad_end = get_depth_rescale_and_clipping_size((w, h))
    if scale_size != (w, h):
        tensor = interpolate_2d(tensor, (scale_size[1],scale_size[0]))
    if clip_size != scale_size:
        tensor = tensor[..., pad_start[1]: scale_size[1]-pad_end[1], pad_start[0]:scale_size[0]-pad_end[0]]
    return tensor


def calculate_depths(output_path: Path, ignore_existing: bool = False):
    """
    Calculate depth maps for all images in the output path.
    Uses colmap camera intrinsics and extrinsics to improve depth estimation.
"""
    print("Depth for:", output_path)
    image_path = output_path / "images"
    images = image_path.glob("*.png")
    image_names = [str(image.name) for image in images]
    if ignore_existing:
        ids = [
            (frame_and_camid_from_path(image_name), image_name)
            for image_name in image_names
        ]
        sorted_ids = sorted(ids, key=lambda x: x[0])
        ids_by_frame = itertools.groupby(sorted_ids, key=lambda x: x[0][0])
        image_names = []
        frame_expected_count = None
        for frame, images in ids_by_frame:
            missing_depth = False
            this_images = []
            for x in images:
                image_name = x[1]
                this_images.append(image_name)
                depth_path = image_path / image_name.replace(".png", ".depth")
                if not depth_path.exists():
                    missing_depth = True
            if frame_expected_count is None:
                frame_expected_count = len(this_images)
            if len(this_images) != frame_expected_count:
                print(
                    "Warning: frame {} has {} images, expected {}. Dropping".format(
                        frame, len(this_images), frame_expected_count
                    )
                )
                missing_depth = False
            if missing_depth:
                image_names.extend(this_images)

    if len(image_names) == 0:
        print("No images to process for depth calculation, skipping.")
        return
    print("Calculating depth for {} images".format(len(image_names)))

    print("Loading depth model")
    device = torch.device("cuda")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")

    from PIL import Image

    test_img = Image.open(image_path / image_names[0])
    w = test_img.width
    h = test_img.height
    # import logging
    # torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)
    tensor_dtype = torch.float32
    wrapper = ModelWrapper(model)
    wrapper = wrapper.eval().to(device=device, dtype=tensor_dtype)

    colmap_path = output_path / "sparse" / "0"
    one_intrinsic, cam_w, cam_h = intrinsics_from_colmap_camera_file(colmap_path, w, h)
    extrinsics = extrinsics_from_colmap(colmap_path)
    intrinsics = [one_intrinsic for _ in range(len(extrinsics))]
    print("Writing depth outputs")

    extrinsics = model._normalize_extrinsics(
        torch.tensor(np.stack(extrinsics), device=device, dtype=tensor_dtype).unsqueeze(
            0
        )
    )
    intrinsics = torch.tensor(
        np.stack(intrinsics), dtype=tensor_dtype, device=device
    ).unsqueeze(0)

    num_cameras = len(extrinsics[0])

    image_ids = [
        (frame_and_camid_from_path(image_name), str(image_path / image_name))
        for image_name in image_names
    ]

    image_ids.sort(key=lambda x: (x[0]))
    ids_by_frame = [
        list(y) for x, y in itertools.groupby(image_ids, key=lambda x: x[0][0])
    ]


    # load a frame worth of images, and return them as a GPU tensor
    def load_frame_images(imgs):
        print(".", end="", flush=True)
        loaded_images = []
        img_names = []
        for _id, img_path in imgs:
            try:
                img_names.append(str(img_path))
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                raise e
            resized_image = transform_image_to_depth_size(img)
            img_tensor = pil_to_tensor(resized_image)
            img_tensor = convert_image_dtype(img_tensor, dtype=tensor_dtype)
            loaded_images.append(img_tensor)
        return img_names, torch.stack(loaded_images, dim=0)  # (N, 3, H, W)

    def save_depth_task(depth_tensor, img_names):
        print(">", end="", flush=True)
        for i, x in enumerate(img_names):
            depth_path = x.replace(".png", f".depth")
            #print(torch.max(depth_tensor[i]), torch.min(depth_tensor[i]))
            depth_tensor_i = depth_tensor[i]
            depth_tensor_i = depth_tensor_i.to(dtype=torch.float16)
            #print(f"Saving depth {x}:{depth_tensor_i.shape}")
            torch.save(depth_tensor_i, depth_path)
        return img_names[0]

    scale_size, clip_size, pad_start, pad_end = get_depth_rescale_and_clipping_size((w, h))

    compiled_name = f"compiled_d_a_{clip_size[0]}x{clip_size[1]}.pt2"
    try:
        print("AOT ")
        compiled_fn = torch._inductor.aoti_load_package(compiled_name)
        print("Loaded compiled model")

    except:
        print("Couldn't load compiled model")
        compiled_fn = None
    if not compiled_fn:
        print("Compiling and saving model, resolution:", clip_size)
        example_images = torch.randn(
            1, num_cameras, 3, clip_size[1], clip_size[0], device=extrinsics.device
        )
        example_inputs = (
            (example_images,),
            {
                "extrinsics": torch.randn_like(extrinsics),
                "intrinsics": torch.randn_like(intrinsics),
            },
        )

        try:
            with open("compiler_cache.tmp", "rb") as f:
                artifact_bytes = f.read()
                torch.compiler.load_cache_artifacts(artifact_bytes)
                print("Loaded compiler cache")
        except Exception as e:
            print(f"Error loading compiler cache: {e}")

        print("Exporting model", flush=True)
        aot_compiled = None
        compiled_fn = None

        torch._dynamo.config.reorderable_logging_functions.add(print)

        if aot_compiled is None and compiled_fn is None:
            print("Running eagerly to create cache")
            with torch.no_grad():
                wrapper(example_inputs[0][0], **example_inputs[1])
            print("Done running eagerly to create cache ")
            compiled_fn = torch.compile(wrapper.forward,fullgraph=True)
            print("Running eagerly to compile")
            with torch.no_grad():
                compiled_fn(example_inputs[0][0], **example_inputs[1])
            print("Done running eagerly to compile ")

            artifacts = torch.compiler.save_cache_artifacts()
            if  artifacts is not None:
                # we've compiled something, save cache
                artifact_bytes, cache_info = artifacts
                with open("compiler_cache.tmp", "wb") as f:
                    f.write(artifact_bytes)

        ## Torch aot_compile version
        # with torch.no_grad():
        #     print("A")
        #     compiled = torch.compile(wrapper.forward,fullgraph=True,dynamic=False)
        #     print("B")
        #     aot_compiled = compiled.aot_compile(example_inputs= example_inputs)
        #     print("C")

        ## AOT_Inductor export version
        #     # ready_to_compile = torch.export.export(
        #     #     wrapper,strict=True,args=example_inputs[0],kwargs=example_inputs[1]
        #     # )
        #     # inference_only_export = ready_to_compile
        #     # #print("Running decompositions",flush=True)
        #     # #inference_only_export = ready_to_compile.run_decompositions(decomp_table={})
        #     # print("Compile and package",flush=True)
        #     # torch._inductor.aoti_compile_and_package(inference_only_export, package_path=compiled_name)

        # # eager run to populate frequency cache
        # print("Running eagerly to populate frequency cache")
        # wrapper.forward(*example_inputs[0], **example_inputs[1])

        # compiled_fn = ready_to_compile.forward.aot_compile(
        #     example_inputs= example_inputs
        # )
        # print("Frequency cache after one run eager",model.model.backbone.pretrained.rope.frequency_cache)
        # compiled_fn.save_compiled_function(compiled_name)
        # print("Compiled and saved model")
    #   else:
    # print("Pre-populate frequency cache with one run of the model")
    # example_images = torch.randn(1,10, 3, scale_size[1], scale_size[0],device=extrinsics.device)
    # example_inputs = (
    #         (
    #             example_images,
    #         ),
    #         {"extrinsics":torch.randn_like(extrinsics),
    #         "intrinsics":torch.randn_like(intrinsics),
    #         },

    # )
    # wrapper.forward(*example_inputs[0], **example_inputs[1])

    #    print(model.model.backbone.pretrained.rope.frequency_cache)

    first_launch = False
    def launch_model_task(img_names, processed_imgs, extrinsics, intrinsics):
        nonlocal first_launch
        #print("#", end="", flush=True)
        print(f"Launching model, device:{device}")
        processed_imgs = processed_imgs.unsqueeze(0).to(device=device,non_blocking=True)
        #print(processed_imgs.shape)
        start_time = time.time()
        #torch.cuda.synchronize(device)
        try:
            if first_launch:
                first_launch = False
                print("Profiling first run") 
                from torch.profiler import profile, ProfilerActivity
                activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]
                with profile(activities=activities) as prof:
                    raw_output = compiled_fn(
                        processed_imgs, extrinsics=extrinsics, intrinsics=intrinsics
                    )
                prof.export_chrome_trace("first_run.json")

            with torch.no_grad():
                if aot_compiled is not None:
                    raw_output = aot_compiled(
                        wrapper,
                        processed_imgs,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                    )
                elif compiled_fn is not None:
                    raw_output = compiled_fn(
                        processed_imgs, extrinsics=extrinsics, intrinsics=intrinsics
                    )
                else:
                    raw_output = wrapper(
                        processed_imgs, extrinsics=extrinsics, intrinsics=intrinsics
                    )
            #print("Model run: syncing")
            #torch.cuda.synchronize(device)
            # TODO: this copy forces a cuda synchronisation.
            # we could try instead 
            prediction_depth = raw_output[0].to(device="cpu")
            time_taken = time.time() - start_time
            print(f"Model run complete, device: {device}, time: {time_taken}")
            del raw_output
        except Exception as e:
            print(f"Error in model execution: {e}")
            raise e
        del processed_imgs
        torch.cuda.memory.empty_cache()
        return prediction_depth, img_names

    depth_pool = ThreadPoolExecutor(max_workers=1)
    load_pool = ThreadPoolExecutor(max_workers=5)
    save_pool = ThreadPoolExecutor(max_workers=2)
    futures = []

    completed = len(ids_by_frame)
    load_pos = 0
    save_pos = 0
    load_futures = []
    process_futures = []
    save_futures = []
    with tqdm(total=len(ids_by_frame), desc="Depth Calculation") as pbar:
        while save_pos < len(ids_by_frame):
            while len(load_futures) < 5 and load_pos < len(ids_by_frame):
                load_futures.append(
                    load_pool.submit(load_frame_images, ids_by_frame[load_pos])
                )
                load_pos += 1
            load_ready = [future for future in load_futures if future.done()]
            # only process so many frames at once, otherwise
            # we fill memory
            while len(process_futures) < 4 and len(load_ready) > 0:
                f = load_ready.pop(0)
                load_futures.remove(f)
                img_names, processed_imgs = f.result()
                process_futures.append(
                    depth_pool.submit(
                        launch_model_task,
                        img_names,
                        processed_imgs,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                    )
                )
            process_ready = [future for future in process_futures if future.done()]
            for f in process_ready:
                process_futures.remove(f)
                prediction_depth, img_names = f.result()
                save_futures.append(
                    save_pool.submit(save_depth_task, prediction_depth, img_names)
                )
            save_ready = [future for future in save_futures if future.done()]
            for f in save_ready:
                save_pos += 1
                save_futures.remove(f)
                #print("Done save:", f.result())
                pbar.update(1)
            wait(
                load_futures + save_futures + process_futures,
                return_when="FIRST_COMPLETED",
            )


def intrinsics_from_colmap_camera_file(colmap_path, dest_w, dest_h):
    colmap_cam_line = None
    with open(colmap_path / "cameras.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            colmap_cam_line = line.split()
            if len(colmap_cam_line) >= 8:
                break

    w = int(colmap_cam_line[2])
    h = int(colmap_cam_line[3])

    w_scale = dest_w / w
    h_scale = dest_h / h

    intrinsic = np.eye(3)
    intrinsic[0, 0] = float(colmap_cam_line[4]) * w_scale
    intrinsic[1, 1] = float(colmap_cam_line[5]) * h_scale
    intrinsic[0, 2] = float(colmap_cam_line[6]) * w_scale
    intrinsic[1, 2] = float(colmap_cam_line[7]) * h_scale
    return intrinsic, w, h


def extrinsics_from_colmap(colmap_path):
    all_extrinsics = {}
    odd_line = True
    with open(colmap_path / "images.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            colmap_image_line = line.split()

            if len(colmap_image_line) == 10:
                # print(colmap_image_line)
                R = quat_to_rot(
                    float(colmap_image_line[1]),
                    float(colmap_image_line[2]),
                    float(colmap_image_line[3]),
                    float(colmap_image_line[4]),
                )
                T = np.array(
                    [
                        float(colmap_image_line[5]),
                        float(colmap_image_line[6]),
                        float(colmap_image_line[7]),
                    ]
                )
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T

                # R=quat_to_rot(
                #     float(colmap_image_line[1]),
                #     float(colmap_image_line[2]),
                #     float(colmap_image_line[3]),
                #     float(colmap_image_line[4]),
                # ).transpose()
                # T=np.array(
                #     [
                #         float(colmap_image_line[5]),
                #         float(colmap_image_line[6]),
                #         float(colmap_image_line[7]),
                #     ]
                # )

                # extrinsic = np.eye(4)
                # extrinsic[:3, :3] = R
                # extrinsic[:3, 3] = -R.dot(T)

                frame_id, cam_id = frame_and_camid_from_path(colmap_image_line[9])
                # print("EXTRINSIC:", frame_id, cam_id, extrinsic)
                all_extrinsics[cam_id] = extrinsic
    return [all_extrinsics[x] for x in sorted(all_extrinsics.keys())]


# Quaternion to rotation matrix conversion
def quat_to_rot(qw, qx, qy, qz):
    """
    Convert quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    Quaternion should be in the form (w, x, y, z).
    """
    # # Normalize quaternion
    # norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    # qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
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


def qvec2rotmat(qvec):  #!wxyz
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# get minimum depth across frames for each camera
# this is used when we make an initial point cloud
# i.e. if a point is depth consistent with the min depth in a frame in all visible cameras then
# we can make a static gaussian. Otherwise it needs to be dynamic
def get_min_depth_maps(output_path):
    scene_info = readColmapSceneInfo(output_path, "images", False, dataloader=True)

    class TempArgs:
        def __init__(self):
            self.dataloader = True
            self.data_device = "cuda"
            self.resolution_scale = 1.0
            self.resolution = 1

    cam_list = cameraList_from_camInfos(scene_info.train_cameras, 1.0, TempArgs())
    dataset = CameraDataset(cam_list, False)

    timestamps = dataset.get_timestamps()
    num_cameras = dataset.get_num_different_cameras()

    depth_maximums = {}

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3)
    for viewpoint_image, viewpoint_cam, depth in dataset:
        pose_id = viewpoint_cam.camera_pose_id
        if pose_id not in depth_maximums:
            depth_maximums[pose_id] = depth.clone()
        else:
            depth_maximums[pose_id] = torch.max(depth_maximums[pose_id], depth)

    depth_maximums_list = list(depth_maximums.values())
    for i, x in enumerate(depth_maximums_list):
        axs[i % 3, i // 3].imshow(x.cpu().numpy())
    plt.show()

    return depth_maximums


if __name__ == "__main__":
    #    make_onnx_depth_model("depth-anything/DA3-SMALL", Path("output/depth_model.onnx"))

    output_path = Path("output/9moving")
    calculate_depths(output_path, ignore_existing=True)

    tensor = torch.rand( 2074,2104)
#    tensor = torch.rand(1, 1, 2073, 2005)
    transformed_tensor = transform_tensor_to_depth_size(tensor)
    print(transformed_tensor.shape, tensor.shape)

    image_size_tensor = transform_depth_to_image_size(transformed_tensor, tensor)
    print(image_size_tensor.shape, tensor.shape,"ImgSize tensor:\n",image_size_tensor[...,0:10,0:10])
    retransformed = transform_tensor_to_depth_size(image_size_tensor)
    print(torch.count_nonzero(retransformed < 0.0))
    print(retransformed.shape, tensor.shape,torch.all(retransformed==transformed_tensor))
    if torch.any(retransformed != transformed_tensor):
        print("Error: retransformed tensor does not match original transformed tensor")
        print("Original transformed tensor:\n", transformed_tensor)
        print("Retransformed tensor:\n", retransformed)


import itertools
from pathlib import Path
import re
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype

from concurrent.futures import ThreadPoolExecutor,wait


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



class OnnxWrapper(torch.nn.Module):
    """
    Thin wrapper around DepthAnything3 that accepts N image tensors, intrinsic and extrinsic matrices
    of shape (N, 3, H, W) and outputs depth (N, 1, H, W) 
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model  # underlying nn.Module

    def forward(self, images,extrinsics,intrinsics):
        # extrinsics (N,4,4) intrinsics (N,4,3) I think (TODO)
        out = self.model(images, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False)
#        out = self.model(images, extrinsics, intrinsics, export_feat_layers=[], infer_gs=False)
        return out["depth"]  # (B, N, H, W)

def make_onnx_depth_model(model_name, output_path):
    device = torch.device("cpu")
    print("Loading PyTorch model: {}".format(model_name))
    pt_model = DepthAnything3.from_pretrained(model_name)
    pt_model = pt_model.to(device)
    pt_model.eval()

    wrapper = OnnxWrapper(pt_model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (torch.zeros((2,10, 3, 518, 518), device=device), torch.zeros((2,10, 4, 4), device=device), torch.zeros((2,10, 3, 3), device=device)),
        input_names=["image","extrinsics","intrinsics"],
        output_names=["depth"],
        dynamic_axes={
            "image": {0:"Batch",1: "N", 2: "H", 3:"W"},
            "depth": {0:"Batch",1: "N", 2: "H", 3:"W"},
            "extrinsics": {0:"Batch",1: "N"},
            "intrinsics": {0:"Batch",1: "N"},
        },
        dynamo=False,
        opset_version=17,
        training=torch.onnx.TrainingMode.EVAL
    )
    print("Exported onnx model to {}".format(output_path))

    

# get camera intrinsics from colmap file
# and use that for depth-anything to generate depthmaps for image
# ( the estimate is better than the DA estimate)
def calculate_depths(output_path: Path, ignore_existing: bool = False):

    image_path = output_path / "images"
    images = image_path.glob("*.png")
    image_names = [str(image.name) for image in images]
    if ignore_existing:
        ids = [
            (frame_and_camid_from_path(image_name), image_name)
            for image_name in image_names
        ]
        sorted_ids = sorted(ids, key=lambda x: x[0])
        ids_by_frame= itertools.groupby(sorted_ids, key=lambda x: x[0][0])
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
                print("Warning: frame {} has {} images, expected {}. Dropping".format(frame,len(this_images),frame_expected_count))
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
    #model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")

    

    from PIL import Image

    test_img = Image.open(image_path / image_names[0])
    w = test_img.width
    h = test_img.height
    # import logging
    # torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)
    tensor_dtype= torch.float32
    model = model.eval().to(device=device,dtype=tensor_dtype)


    colmap_path = output_path / "sparse" / "0"
    one_intrinsic, cam_w, cam_h = intrinsics_from_colmap_camera_file(colmap_path, w, h)
    extrinsics = extrinsics_from_colmap(colmap_path)
    intrinsics = [one_intrinsic for _ in range(len(extrinsics))]
    print("Writing depth outputs")

    extrinsics = model._normalize_extrinsics(torch.tensor(extrinsics,device=device,dtype=tensor_dtype).unsqueeze(0))
    intrinsics = torch.tensor(intrinsics,dtype=tensor_dtype,device=device).unsqueeze(0)



    

    image_ids = [
        (frame_and_camid_from_path(image_name), image_name)
        for image_name in image_names
    ]
    image_ids.sort(key=lambda x: x[0])

    cur_frame = None
    frame_images = []

    # load a frame worth of images, and return them as a GPU tensor
    def load_frame_images(imgs):
        loaded_images = []
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            if img.width > 2000 or image.height > 2000:
                img = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)
            # pad image to make it multiple of 14 on both axes (patch size)
            padded_size = (((img.width + 13) // 14) * 14,((img.height + 13) // 14) * 14)
            img_padded = Image.new("RGB",padded_size,(0,0,0))
            img_padded.paste(img, (0, 0))
            img_tensor = pil_to_tensor(img_padded)
            img_tensor= convert_image_dtype(img_tensor, dtype=tensor_dtype) 
            loaded_images.append(img_tensor)
        return torch.stack(loaded_images, dim=0)  # (N, 3, H, W)



    def launch_model_task(model, img_names, processed_imgs,extrinsics,intrinsics):
        processed_imgs = processed_imgs.to(device=device)
        @torch.compile
        def compile_fn(model, processed_imgs,extrinsics,intrinsics):
            return model._run_model_forward(processed_imgs.unsqueeze(0), extrinsics, intrinsics,export_feat_layers=[])
        raw_output = compile_fn(model, processed_imgs,extrinsics,intrinsics)
        prediction_depth = raw_output["depth"][0].to(device="cpu") 
        del raw_output
        del processed_imgs
        torch.cuda.memory.empty_cache()
        return prediction_depth, img_names

    pool = ThreadPoolExecutor(max_workers=1)
    futures = []

    def process_images(imgs):
        nonlocal futures
        process_depth_results()
        processed_imgs = load_frame_images(imgs)
        print("Submitting:",processed_imgs.shape,processed_imgs.device,processed_imgs.dtype,extrinsics.dtype,intrinsics.dtype)
        futures.append(pool.submit(launch_model_task, model, imgs, processed_imgs,extrinsics,intrinsics))
        print("Futures length:",len(futures))

    def process_depth_results(max_left_in_queue=2):
        processed_futures = []
        running_count = [f.done() for f in futures].count(False)
                
        for f in futures:
            if running_count > max_left_in_queue:
                print("Waiting for completion of futures, running_count:",running_count)
                wait([f for f in futures if not f.done()], return_when="FIRST_COMPLETED")
                running_count = [f.done() for f in futures].count(False)
            if f.done():
                print("Future done, processing results")
                prediction_depth, img_names = f.result()
                for i, x in enumerate(img_names):
                    depth_path = x.replace(".png", f".depth")
                    print(
                        torch.max(prediction_depth[i]), torch.min(prediction_depth[i])
                    )
                    depth_tensor = prediction_depth[i]
                    depth_tensor=depth_tensor.to(dtype=torch.float16)
                    print(f"Saving depth {x}:{depth_tensor.shape}")
                    torch.save(depth_tensor, depth_path)
                processed_futures.append(f)
        futures[:] = [f for f in futures if not f in processed_futures]


    for (frame, camid), name in tqdm(image_ids):
        if cur_frame != frame:
            print(frame_images)
            if len(frame_images) > 0:
                process_images(frame_images)
            frame_images = []
        cur_frame = frame
        frame_images.append(str(image_path / name))

    if len(frame_images) > 0:
        process_images(frame_images)


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
                print(colmap_image_line)
                R=quat_to_rot(
                    float(colmap_image_line[1]),
                    float(colmap_image_line[2]),
                    float(colmap_image_line[3]),
                    float(colmap_image_line[4]),
                )
                T=np.array(
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

                frame_id,cam_id = frame_and_camid_from_path(colmap_image_line[9])
                print("EXTRINSIC:",frame_id,cam_id,extrinsic)
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
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])



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

    output_path = Path("output/esper_ball")
    calculate_depths(output_path,ignore_existing=True)

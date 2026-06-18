from pathlib import Path
import re
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth
import numpy as np
from tqdm import tqdm

from PIL import Image


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



# get camera intrinsics from colmap file
# and use that for depth-anything to generate depthmaps for image
# ( the estimate is better than the DA estimate)
def calculate_depths(output_path: Path, ignore_existing: bool = False):

    image_path = output_path / "images"
    images = image_path.glob("*.png")
    image_names = [str(image.name) for image in images]
    if ignore_existing:
        image_names = [
            name
            for name in image_names
            if not (image_path / name.replace(".png", ".depth")).exists()
        ]

    if len(image_names) == 0:
        print("No images to process for depth calculation, skipping.")
        return

    print("Loading depth model")
    device = torch.device("cuda")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    #model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")
    
    model = model.to(device=device)

    images = image_path.glob("*.png")

    image_names = [str(image.name) for image in images]
    from PIL import Image

    test_img = Image.open(image_path / image_names[0])
    w = test_img.width
    h = test_img.height

    colmap_path = output_path / "sparse" / "0"
    one_intrinsic, cam_w, cam_h = intrinsics_from_colmap_camera_file(colmap_path, w, h)
    extrinsics = extrinsics_from_colmap(colmap_path)
    intrinsics = [one_intrinsic for _ in range(len(extrinsics))]
    print("Writing depth outputs")

    image_ids = [
        (frame_and_camid_from_path(image_name), image_name)
        for image_name in image_names
    ]
    image_ids.sort(key=lambda x: x[0])

    cur_frame = None
    frame_images = []

    def process_images(imgs):
        prediction = model.inference(
            imgs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            process_res=max(w, h),
            process_res_method="upper_bound_crop",
        )
        h_out = prediction.depth.shape[1]
        w_out = prediction.depth.shape[2]
        pad_top = int(round((h - h_out) / 2.0))
        pad_bottom = h - (h_out + pad_top)
        pad_left = int(round((w - w_out) / 2.0))
        pad_right = w - (w_out + pad_left)


#        zero_centred_extrinsics = torch.tile(torch.eye(4),len(imgs))
        print(torch.tensor(extrinsics).shape)
        for x in range(len(extrinsics)):
            print(x,"---\n",extrinsics[x],"\n------------------------")

#        CAM_NUM=len(extrinsics)-1
        CAM_START=0
        CAM_END=len(extrinsics)

        from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
        #prediction.extrinsics[-1] = np.eye(4)


        conf_thresh = np.percentile(prediction.conf, 40.0)
        all_points,all_colors = _depths_to_world_points_with_colors(prediction.depth[CAM_START:CAM_END],
        prediction.intrinsics[CAM_START:CAM_END],
        prediction.extrinsics[CAM_START:CAM_END],  # w2c
        prediction.processed_images[CAM_START:CAM_END],
        prediction.conf[CAM_START:CAM_END],
        conf_thresh)

        camera_indices = torch.zeros(all_points[:,0].shape,dtype=torch.int32,device="cuda")
        print(prediction.depth.shape)
        indices_length = camera_indices.shape[0]//prediction.depth.shape[0]
        for x in range(prediction.depth.shape[0]):
            print("CAMERA INDICES:",x,x*indices_length,(x+1)*indices_length)
            camera_indices[x*indices_length:(x+1)*indices_length] = x
        print("Before filter:",torch.min(camera_indices),torch.max(camera_indices))


        all_colors = torch.tensor(all_colors, dtype=torch.float32) * (1.0 / 255.0)
        all_points = torch.tensor(all_points, dtype=torch.float32)

        # all_points=torch.cat((all_points,torch.tensor(prediction.extrinsics[CAM_NUM,0:3,3],dtype=torch.float32).unsqueeze(0)),dim=0)
        # all_colors = torch.cat((all_colors,torch.tensor([1.0,0.0,0.0],dtype=torch.float32).unsqueeze(0)),dim=0)
        print(all_points[-1],all_colors[-1])

#        all_points = depth_to_point_cloud_vectorized(prediction.depth[0:1],prediction.intrinsics[0:1],prediction.extrinsics[0:1])
#        all_points = unproject_depth(torch.tensor(prediction.depth[0:1],dtype=torch.float32).unsqueeze(0).unsqueeze(-1),torch.tensor(prediction.intrinsics,dtype=torch.float32).unsqueeze(0),c2w= torch.tensor(prediction.extrinsics,dtype=torch.float32).unsqueeze(0))[0]
#        all_colors = torch.tensor(prediction.processed_images[0:1],dtype=torch.float32)*(1.0/255.0)
#        print(all_points.shape,all_colors.shape,prediction.processed_images.shape,prediction.depth.shape)
#        all_points= all_points.reshape(-1,3)
#        all_colors = all_colors.reshape(-1,3)
        print(torch.max(all_colors),torch.min(all_colors))
        print(all_points.shape,all_colors.shape,prediction.processed_images.shape,prediction.depth.shape)
        
        pts_cuda = torch.tensor(all_points).to("cuda")
        colors_cuda = torch.tensor(all_colors).to("cuda")
        print(prediction.extrinsics[CAM_START])
        fwd_direction = torch.tensor(prediction.extrinsics[CAM_START][0:3,2], device="cuda")
        camera_position = torch.tensor(prediction.extrinsics[CAM_START][0:3,3], device="cuda")
        camera_position = torch.tensor(prediction.extrinsics[CAM_START][0:3,0:3],device="cuda")@camera_position
        print("********",camera_position)
        look_at = all_points[all_points.shape[0]//2 + prediction.depth.shape[-1]//2]
        up = torch.tensor([0.0, -1.0, 0], device="cuda")
        print("RENDER:",camera_position,look_at,up)
        from pointcloud_renderer import show_pointcloud_glfw_pytorch3d

#        show_pointcloud_glfw_pytorch3d(pts_cuda,colors_cuda,title="Depth Anything 3D point cloud",look_at=look_at,up=up,camera_position=camera_position,fov_degrees=70.0,camera_indices=camera_indices)

        padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        prediction_depth_padded = np.pad(
            prediction.depth, padding, mode="constant", constant_values=-1.0
        )
        for i, x in enumerate(frame_images):
            depth_path = x.replace(".png", f".depth")
            print(
                np.max(prediction_depth_padded[i]), np.min(prediction_depth_padded[i])
            )
            depth_tensor = torch.tensor(prediction_depth_padded[i])
            print(depth_tensor.shape)
            torch.save(depth_tensor, depth_path)

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
    output_path = Path("output/9moving")
    calculate_depths(output_path)

from pathlib import Path
import re
from matplotlib.pylab import rint
import torch
from depth_anything_3.api import DepthAnything3
import numpy as np
from tqdm import tqdm

from PIL import Image

# get camera intrinsics from colmap file 
# and use that for depth-anything to generate depthmaps for image
# ( the estimate is better than the DA estimate)
def calculate_depths(output_path: Path, ignore_existing: bool = False):
    def frame_and_camid_from_path(video_path):
        match = re.match(r"(?:\D)*(\d+)_(\d+)", video_path)
        if match:
            return (
                int(match.groups()[1]),
                int(match.groups()[0]),
            )
        return ""

    image_path = output_path / "images"
    images = image_path.glob("*.png")
    image_names = [str(image.name) for image in images]
    if ignore_existing:
        image_names = [name for name in image_names if not (image_path / name.replace(".png", ".depth")).exists()]
    
    if len(image_names) == 0:
        print("No images to process for depth calculation, skipping.")
        return
    

    print("Loading depth model")
    device = torch.device("cuda")
    #model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
#    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
    model = model.to(device=device)

    images = image_path.glob("*.png")

    image_names = [str(image.name) for image in images]
    from PIL import Image
    test_img = Image.open(image_path / image_names[0])
    w = test_img.width
    h = test_img.height


    colmap_path = output_path / "sparse" / "0"
    one_intrinsic,cam_w,cam_h = intrinsics_from_colmap_camera_file(colmap_path,w,h)
    extrinsics = extrinsics_from_colmap(colmap_path)
    intrinsics = [one_intrinsic for _ in range(len(extrinsics))]
    print("Writing depth outputs")

    image_ids = [
        (frame_and_camid_from_path(image_name), image_name) for image_name in image_names
    ]
    image_ids.sort(key=lambda x: x[0])

    cur_frame = None
    frame_images = []


    def process_images( imgs):
        prediction = model.inference(
            imgs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            process_res = max(w,h),
            process_res_method = "upper_bound_crop"
        )
        h_out = prediction.depth.shape[1]
        w_out = prediction.depth.shape[2]                
        pad_top = int(round((h - h_out) / 2.0))
        pad_bottom = h - (h_out + pad_top)
        pad_left = int(round((w - w_out) / 2.0))
        pad_right = w - (w_out + pad_left)

        padding = ((0,0),(pad_top,pad_bottom) , (pad_left,pad_right))
        prediction_depth_padded = np.pad(prediction.depth,padding, mode="constant", constant_values=-1.0)
#                print("Depth out:",prediction_depth_padded.shape)
        for i, x in enumerate(frame_images):
            depth_path = x.replace(".png", f".depth")
            print(np.max(prediction_depth_padded[i]), np.min(prediction_depth_padded[i]))
            depth_tensor = torch.tensor(prediction_depth_padded[i])
            print(depth_tensor.shape)
            torch.save(depth_tensor, depth_path)


    for (frame, camid), name in tqdm(image_ids):
        if cur_frame != frame:
            if len(frame_images) > 0:
                process_images( frame_images)
            frame_images = []
        cur_frame = frame
        frame_images.append(str(image_path / name))

    if len(frame_images) > 0:
        process_images(frame_images)



def intrinsics_from_colmap_camera_file(colmap_path,dest_w,dest_h):
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
    intrinsic[0, 0] = float(colmap_cam_line[4])*w_scale
    intrinsic[1, 1] = float(colmap_cam_line[5])*h_scale
    intrinsic[0, 2] = float(colmap_cam_line[6])*w_scale
    intrinsic[1, 2] = float(colmap_cam_line[7])*h_scale
    return intrinsic, w, h

def extrinsics_from_colmap(colmap_path):
    all_extrinsics = []
    odd_line = True
    with open(colmap_path / "images.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            colmap_image_line = line.split()
            if len(colmap_image_line) == 10:
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = quat_to_rot(
                    float(colmap_image_line[1]),
                    float(colmap_image_line[2]),
                    float(colmap_image_line[3]),
                    float(colmap_image_line[4]))
                extrinsic[:3, 3] = np.array(
                    [float(colmap_image_line[5]), float(colmap_image_line[6]), float(colmap_image_line[7])]
                )
                all_extrinsics.append(extrinsic)
    return all_extrinsics



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


if __name__ == "__main__":
    output_path = Path("output/9moving")
    calculate_depths(output_path)
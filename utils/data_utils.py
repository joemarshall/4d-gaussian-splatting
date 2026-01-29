import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        self.only_path = False

    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if self.only_path:
            viewpoint_image = viewpoint_cam.image_path
            features = viewpoint_cam.original_features
            masks = viewpoint_cam.original_masks
            mask_scales = viewpoint_cam.mask_scales
            return viewpoint_image, viewpoint_cam, features, masks, mask_scales

        if viewpoint_cam.meta_only:
            preprocessed_pth_path = viewpoint_cam.image_path.replace(
                '.png',
                f'_{viewpoint_cam.image_height}_{viewpoint_cam.image_width}.pth'
            )
            if os.path.exists(preprocessed_pth_path):
                viewpoint_image = torch.load(preprocessed_pth_path)
            else:
                with Image.open(viewpoint_cam.image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
                image_load = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
                resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
                viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                if resized_image_rgb.shape[1] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    viewpoint_image *= gt_alpha_mask
                else:
                    viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
                torch.save(viewpoint_image, preprocessed_pth_path)
        else:
            viewpoint_image = viewpoint_cam.image

        features = torch.load(viewpoint_cam.original_features) if viewpoint_cam.original_features is not None else None
        masks = torch.load(viewpoint_cam.original_masks) if viewpoint_cam.original_masks is not None else None
        mask_scales = torch.load(viewpoint_cam.mask_scales) if viewpoint_cam.mask_scales is not None else None
        if masks is not None:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(),
                mode='bilinear',
                size=(viewpoint_cam.image_height, viewpoint_cam.image_width),
                align_corners=False
            ).squeeze(1)

        return viewpoint_image, viewpoint_cam, features, masks, mask_scales
    
    def __len__(self):
        return len(self.viewpoint_stack)
    

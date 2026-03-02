from genericpath import exists
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from pathlib import Path

# TODO: write permanent cache files into subfolder of image path based on the width
# and return memorymapped tensor based on it 
# or perhaps .to_cuda on that tensor

class ImageCache:

    @staticmethod
    def get_image_for_file(path,width):
        cache_path = f"{path}_{width}.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path,mmap=True)
        else:
            return None
        
    @staticmethod
    def set_image_for_file(path,width,tensor):
        cache_path = f"{path}_{width}.pt"
        torch.save(tensor, cache_path)

class CameraDataset(Dataset):
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            cached = ImageCache.get_image_for_file(viewpoint_cam.image_path, viewpoint_cam.image_width)
            if cached is not None:
#                print("Using cached image:", viewpoint_cam.image_path)
                return cached, viewpoint_cam
                
            # load to memory mapped tensor (stored in tempfile)
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
            ImageCache.set_image_for_file(viewpoint_cam.image_path, viewpoint_cam.image_width, viewpoint_image)
        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    

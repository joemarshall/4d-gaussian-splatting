from genericpath import exists
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, Sampler
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Iterator

# TODO: write permanent cache files into subfolder of image path based on the width
# and return memorymapped tensor based on it 
# or perhaps .to_cuda on that tensor

class ImageCache:

    @staticmethod
    def get_image_for_file(path,width):
        cache_path = f"{path}_{width}.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path,map_location="cuda")
        else:
            return None
        
    @staticmethod
    def set_image_for_file(path,width,tensor):
        cache_path = f"{path}_{width}.pt"
        torch.save(tensor, cache_path)

class CameraDataset(Dataset):

    class FrameSampler(Sampler[List[int]]):
        def __init__(self, dataset):
            self.dataset = dataset
            self.timestamps = self.dataset.get_timestamps()
            self.num_cameras = self.dataset.get_num_different_cameras()

        def __len__(self):
            return len(self.timestamps)

        def __iter__(self) -> Iterator[List[int]]:
            timestamps= self.timestamps.copy()
            np.random.shuffle(timestamps)
            for t in timestamps:
                indices = self.dataset.get_indices_for_timestamp(t)
                indices = np.random.shuffle(indices)
                yield indices
                
    class CameraSampler(Sampler[List[int]]):
        def __init__(self, dataset,batch_size = 10):
            self.dataset = dataset
            self.timestamps = self.dataset.get_timestamps()
            self.num_cameras = self.dataset.get_num_different_cameras()
            self.length = (self.num_cameras * len(self.timestamps)) // batch_size
            self.batch_size = batch_size

        def __len__(self):
            return self.length

        def __iter__(self) -> Iterator[List[int]]:
            # each sample = random sample from timestamps
            # batches are cameras + random frames from those cameras
            timestamps= self.timestamps.copy()
            np.random.shuffle(timestamps)
            camera_indices = np.arange(self.num_cameras)
            np.random.shuffle(camera_indices)
            for _ in range(self.length):
                these_timestamps = timestamps[0:self.batch_size]
                timestamps = timestamps[self.batch_size:]
                camera_id = camera_indices[0]
                camera_indices = camera_indices[1:]
                if len(camera_indices) == 0:
                    camera_indices = np.arange(self.num_cameras)
                    np.random.shuffle(camera_indices)
                if len(timestamps) < self.batch_size:
                    timestamps= self.timestamps.copy()
                    np.random.shuffle(timestamps)
                indices = self.dataset.get_indices_for_timestamp(these_timestamps,camera_id)
                yield indices

    class CameraAndFrameSampler(Sampler[List[int]]):
        def __init__(self, dataset,batch_size = 10):
            self.sampler1 = CameraDataset.CameraSampler(dataset,batch_size)
            self.sampler2 = CameraDataset.FrameSampler(dataset)
            self.length = len(self.sampler1)+len(self.sampler2)
            self.next_sampler=0

        def __len__(self):
            return self.length

        def __iter__(self) -> Iterator[List[int]]:
            # each sample = random sample from timestamps
            # batches are cameras + random frames from those cameras
            iter1 = iter(self.sampler1)
            iter2 = iter(self.sampler2)
            if self.next_sampler ==0:
                yield from iter1
            else:
                yield from iter2
            

    def __init__(self, copy_dataset):
        self.viewpoint_stack = copy_dataset.viewpoint_stack.copy()
        self.bg = copy_dataset.bg
        self.names_only = copy_dataset.names_only

    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        self.names_only = False
        self.timestamps = None
        self.num_cameras = None

    def set_names_only(self, names_only):
        self.names_only = names_only

    def _get_camera_cache_key(self, camera):
        return (*camera.R.flatten(),*camera.T.flatten())

    def _calc_cameras_and_timestamps(self):
        timestamp_set = set()
        viewpoint_set = set()

        for x in self.viewpoint_stack:
            viewpoint_set.add(self._get_camera_cache_key(x))
            timestamp_set.add(x.timestamp)
        self.num_cameras= len(viewpoint_set)
        self.timestamps = sorted(list(timestamp_set))
        self.different_cam_list = list(viewpoint_set)
        print(self.num_cameras)
        print(len(self.timestamps))
        print(len(self.viewpoint_stack))


    def get_num_different_cameras(self):
        # how many different camera viewpoints are in this
        if self.num_cameras is None:
            self._calc_cameras_and_timestamps()
        return self.num_cameras


    def get_timestamps(self):
        if self.timestamps is None:
            self._calc_cameras_and_timestamps()
        return self.timestamps

    def get_indices_for_timestamp(self,t,camera_id = None):
        timestamp_list = t if hasattr(t, '__iter__') else [t]
        filter = None
        if camera_id is not None:
            filter = self.different_cam_list[camera_id]
        camlist = []
        for idx,x in enumerate(self.viewpoint_stack):
            if x.timestamp in  timestamp_list:
                if filter is None or self._get_camera_cache_key(x) == filter:
                    camlist.append(idx)
        return camlist

    def get_frame_batch_sampler(self):
        return CameraDataset.CameraAndFrameSampler(self)
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only and not self.names_only:
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
            resized_image_rgb.requires_grad = False
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
            ImageCache.set_image_for_file(viewpoint_cam.image_path, viewpoint_cam.image_width, viewpoint_image)
            
            return viewpoint_image, viewpoint_cam
        if self.names_only:
            return None, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)

    def copy(self):
        return CameraDataset(self)
    

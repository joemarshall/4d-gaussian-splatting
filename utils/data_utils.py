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
from concurrent.futures import ThreadPoolExecutor

import safetensors.torch


class ImageCache:

    preload_cache = {}
    executor = ThreadPoolExecutor(max_workers=10)


    def loader_task(path,width,has_depth):
        cache_path = f"{path}_{width}.st"
        if os.path.exists(cache_path):
            rval = safetensors.torch.load_file(cache_path,device="cpu")
            if has_depth and "depth" not in rval:
                print("Warning: expected depth but not found in cache for", path)
            elif not has_depth and "depth" in rval:
                print("Warning: found unexpected depth in cache for", path)
            r_img = rval["tensor"].to(device="cuda", dtype=torch.float32, non_blocking = (has_depth == False))
            r_depth = None
            if "depth" in rval:
                r_depth = rval["depth"].to(device="cuda", dtype=torch.float32, non_blocking = True)
            return (r_img,r_depth)
        else:
            return (None,None)


    @staticmethod
    def preload_image(path,width,has_depth):
        # load into preload cache in a thread
        # so that we can return it next call
        if (path,width) not in ImageCache.preload_cache:
            #print("Preloading image for", path,width)
            ImageCache.preload_cache[(path,width)] = {"future": ImageCache.executor.submit(ImageCache.loader_task, path, width, has_depth),"refcount":1}
        else:
            ImageCache.preload_cache[(path,width)]["refcount"] += 1
            #print("Already preloaded", path,width,ImageCache.preload_cache[(path,width)]["refcount"])
    
        return ImageCache.preload_cache[(path,width)]["future"]

    @staticmethod
    def get_image_for_file(path, width, has_depth):
        cache_path = f"{path}_{width}.st"
        if (path,width) not in ImageCache.preload_cache:
            if os.path.exists(cache_path):
                # load and wait
                #print("Not preloaded but cache file exists, loading for", path)
                ImageCache.preload_image(path,width,has_depth).result()
            else:
            # doesn't exist, return None
                return None,None
        cache_entry = ImageCache.preload_cache[(path,width)]
        refcount = cache_entry["refcount"]
        if refcount <= 1:
            return ImageCache.preload_cache.pop((path,width))["future"].result()    
        cache_entry["refcount"] -= 1
        return ImageCache.preload_cache[(path,width)]["future"].result()

        # if rval[0] is not None:
        #     return rval
        #     # r_img = rval["tensor"].to(device="cuda", dtype=torch.float32, non_blocking = True)
        #     # r_depth = None
        #     # if "depth" in rval and has_depth:
        #     #     r_depth = rval["depth"].to(device="cuda", dtype=torch.float32, non_blocking = True)
        #     # return r_img, r_depth
        # else:
        #     return None, None

    @staticmethod
    def set_image_for_file(path, width, tensor,depth):  
        print("Caching image for", path)
        cache_path = f"{path}_{width}.st"
        all_tensors = {"tensor": tensor.to(device="cpu", dtype=torch.float16).contiguous()}
        if depth is not None:
            all_tensors["depth"] = depth.to(device="cpu", dtype=torch.float16).contiguous()
        safetensors.torch.save_file(all_tensors, cache_path)


class CameraDataset(Dataset):

    class ReadAheadSampler(Sampler[List[int]]):
        def __init__(self, dataset, child_sampler):
            self.dataset = dataset
            self.child_sampler = child_sampler

        def __len__(self):
            return len(self.child_sampler)
        
        def __iter__(self) -> Iterator[List[int]]:
            # read ahead one batch 
            child_iter = iter(self.child_sampler)
            next_batch = None
            done = False
            readahead_buffer = []
            readahead_batches = 1
            while not done:
                # count_in_progress = len([x for x in ImageCache.preload_cache.values() if x["future"].running() ])
                # count_waiting = sum(x["refcount"] for x in ImageCache.preload_cache.values() if x["future"].done() )
                # print("ReadAheadSampler buffer size:", len(readahead_buffer), "in progress:", count_in_progress,"ready:",count_waiting)
                if len(readahead_buffer)<= readahead_batches and not done:
                    try:
                        next_batch =next(child_iter)
                        for x in next_batch:
                            ImageCache.preload_image(self.dataset.viewpoint_stack[x].image_path, self.dataset.viewpoint_stack[x].image_width, self.dataset.viewpoint_stack[x].depth is not None)
                        readahead_buffer.append(next_batch)
                    except StopIteration:
                        print("ReadAheadSampler child sampler exhausted")
                        done = True
                if len(readahead_buffer) > readahead_batches:
                    yield readahead_buffer.pop(0)
                if done:
                    while len(readahead_buffer) > 0:
                        yield readahead_buffer.pop(0)


    class FrameSampler(Sampler[List[int]]):
        def __init__(self, dataset):
            self.dataset = dataset
            self.timestamps = self.dataset.get_timestamps()
            self.num_cameras = self.dataset.get_num_different_cameras()

        def __len__(self):
            return len(self.timestamps)

        def __iter__(self) -> Iterator[List[int]]:
            timestamps = self.timestamps.copy()
            np.random.shuffle(timestamps)
            for t in timestamps:
                indices = self.dataset.get_indices_for_timestamp(t)
                if indices is not None:
                    np.random.shuffle(indices)
                    yield indices

    class CameraSampler(Sampler[List[int]]):
        def __init__(self, dataset, batch_size=10):
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
            timestamps = self.timestamps.copy()
            np.random.shuffle(timestamps)
            camera_indices = np.arange(self.num_cameras)
            np.random.shuffle(camera_indices)
            camera_id = camera_indices[0]
            camera_indices = camera_indices[1:]
            for _ in range(self.length):
                these_timestamps = timestamps[0 : self.batch_size]
                timestamps = timestamps[self.batch_size :]
                if len(timestamps) < self.batch_size:
                    timestamps = self.timestamps.copy()
                    np.random.shuffle(timestamps)
                    camera_id = camera_indices[0]
                    camera_indices = camera_indices[1:]
                    if len(camera_indices) == 0:
                        camera_indices = np.arange(self.num_cameras)
                        np.random.shuffle(camera_indices)
                indices = self.dataset.get_indices_for_timestamp(
                    these_timestamps, camera_id
                )
                yield indices

    class RandomSampler(Sampler[List[int]]):
        def __init__(self, dataset, batch_size=10):
            self.dataset = dataset
            self.length = len(self.dataset) // batch_size
            self.batch_size = batch_size

        def __len__(self):
            return self.length

        def __iter__(self) -> Iterator[List[int]]:
            # each sample = random sample from dataet
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            for i in range(0, self.length * self.batch_size, self.batch_size):
                yield indices[i : i + self.batch_size]

    class TimeCoherentCameraAndFrameSampler(Sampler[List[int]]):
        """
        This sampler returns batches of indices that are close together in time, i.e. are within the
        same timestep range for multiple batches. This means training only has to reload timesteps 
        infrequently.

        It does this by selecting a timestamp range, then sampling:
        step 1) either random cameras in that timestamp range
        step 2) all cameras for a frame in that timestamp range

        until it has done all frames (and the same number of camera samples)


        """
        def __init__(self,dataset, time_range=0.5):
            self.dataset = dataset
            self.time_range = time_range
            self.timestamps = self.dataset.get_timestamps()
            self.num_cameras = self.dataset.get_num_different_cameras()
            self.length = (len(self.timestamps) * self.num_cameras)*2
            self.sample_frame = True

        def __len__(self):
            return self.length

        def __iter__(self) -> Iterator[List[int]]:
            timestamps = sorted(self.timestamps)
            timestamp_ranges = []
            range_start = timestamps[0]
            this_range = []
            for x in timestamps:
                this_range.append(x)
                if x - range_start > self.time_range:
                    timestamp_ranges.append(this_range)
                    range_start = x
            if len(this_range) > 0:
                timestamp_ranges.append(this_range)
            # now timestamp_ranges contains a set of timestamp ranges which are
            # all within a single time_range
            np.random.shuffle(timestamp_ranges)
            for r in timestamp_ranges:
                cameras = np.arange(self.num_cameras)
                np.random.shuffle(cameras)
                cam_times = np.array(r)
                frame_times = np.array(r)            
                np.random.shuffle(frame_times)

                while len(frame_times) > 0:
                    indices = self.dataset.get_indices_for_timestamp(frame_times[0])
                    if indices is not None:
                        np.random.shuffle(indices)
                        yield indices
                    frame_times = frame_times[1:]

                    camera = cameras[0]
                    cameras = cameras[1:]
                    if len(cameras) == 0:
                        cameras = np.arange(self.num_cameras)
                        np.random.shuffle(cameras)
                    
                    camera_times = np.random.choice(cam_times , size=self.num_cameras,replace = False)

                    indices = self.dataset.get_indices_for_timestamp(camera_times, camera)
                    if indices is not None:
                        np.random.shuffle(indices)
                        yield indices



    class CameraAndFrameSampler(Sampler[List[int]]):
        def __init__(self, dataset, batch_size=10):
            self.sampler1 = CameraDataset.CameraSampler(dataset, batch_size)
            self.sampler2 = CameraDataset.FrameSampler(dataset)
            self.length = len(self.sampler1) + len(self.sampler2)
            self.next_sampler = 0

        def __len__(self):
            return self.length

        def __iter__(self) -> Iterator[List[int]]:
            # each sample = random sample from timestamps
            # batches are cameras + random frames from those cameras
            iter1 = iter(self.sampler1)
            iter2 = iter(self.sampler2)
            while True:
                if self.next_sampler == 0:
                    it = iter1
                else:
                    it = iter2
                try:
                    rv = next(it)
                    yield rv
                    self.next_sampler = 1 - self.next_sampler
                except StopIteration:
                    if it== iter1:
                        iter1 = iter(self.sampler1)
                    else:
                        iter2 = iter(self.sampler2)

    def __init__(self, copy_dataset):
        self.viewpoint_stack = copy_dataset.viewpoint_stack.copy()
        self.bg = copy_dataset.bg
        self.names_only = copy_dataset.names_only
        self.timestamps = copy_dataset.timestamps
        self.num_cameras = copy_dataset.num_cameras

    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        self.names_only = False
        self.timestamps = None
        self.num_cameras = None
        self._calc_cameras_and_timestamps()

    def set_names_only(self, names_only):
        self.names_only = names_only

    def _get_camera_cache_key(self, camera):
        return (*camera.R.flatten(), *camera.T.flatten())

    def _calc_cameras_and_timestamps(self):
        timestamp_set = set()
        viewpoint_set = set()

        for x in self.viewpoint_stack:
            viewpoint_set.add(self._get_camera_cache_key(x))
            timestamp_set.add(x.timestamp)
        self.num_cameras = len(viewpoint_set)
        self.timestamps = sorted(list(timestamp_set))
        self.different_cam_list = list(viewpoint_set)

        for x in self.viewpoint_stack:
            x.camera_pose_id = self.different_cam_list.index(
                self._get_camera_cache_key(x)
            )

    def get_num_different_cameras(self):
        # how many different camera viewpoints are in this
        if self.num_cameras is None:
            self._calc_cameras_and_timestamps()
        return self.num_cameras

    def get_timestamps(self):
        if self.timestamps is None:
            self._calc_cameras_and_timestamps()
        return self.timestamps

    def get_indices_for_timestamp(self, t, camera_id=None):
        timestamp_list = t if hasattr(t, "__iter__") else [t]
        filter = None
        if camera_id is not None:
            filter = self.different_cam_list[camera_id]
        camlist = []
        for idx, x in enumerate(self.viewpoint_stack):
            if x.timestamp in timestamp_list:
                if filter is None or self._get_camera_cache_key(x) == filter:
                    camlist.append(idx)
        return camlist

    def get_frame_batch_sampler(self, suggested_batch_size=4):
        return CameraDataset.ReadAheadSampler(self,CameraDataset.TimeCoherentCameraAndFrameSampler(self))

        # return CameraDataset.RandomSampler(self,batch_size=suggested_batch_size)
        #        return CameraDataset.ReadAheadSampler(self,CameraDataset.CameraAndFrameSampler(self))

    def metadata(self):
        return self.viewpoint_stack

    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only and not self.names_only:
            cached_image,cached_depth = ImageCache.get_image_for_file(
                viewpoint_cam.image_path, viewpoint_cam.image_width, viewpoint_cam.depth is not None
            )
            if cached_image is not None:
                #                print("Using cached image:", viewpoint_cam.image_path)
                return cached_image, viewpoint_cam, cached_depth

            # load to memory mapped tensor (stored in tempfile)
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (
                1 - norm_data[:, :, 3:4]
            )
            if viewpoint_cam.resolution == 1:
                viewpoint_image = arr
            else:
                image_load = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
                resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
                resized_image_rgb.requires_grad = False
                viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                if resized_image_rgb.shape[1] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    viewpoint_image *= gt_alpha_mask
                else:
                    viewpoint_image *= torch.ones(
                        (1, viewpoint_cam.image_height, viewpoint_cam.image_width)
                    )
            depth = None
            viewpoint_image= viewpoint_image.contiguous()
            if viewpoint_cam.depth is not None:
                depth = torch.load(viewpoint_cam.depth, map_location="cpu")
                if depth.shape[-2:] != viewpoint_image.shape[-2:]:
                    print("Resizing depth image from: ",depth.shape,viewpoint_image.shape)
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),
                        size=viewpoint_image.shape,
                        mode="nearest",
                    ).squeeze(0)
            ImageCache.set_image_for_file(
                viewpoint_cam.image_path, viewpoint_cam.image_width, viewpoint_image,depth
            )
            #print("Caching",viewpoint_image, depth)
            return viewpoint_image, viewpoint_cam, depth
        if self.names_only:
            return None, viewpoint_cam, None

    def __len__(self):
        return len(self.viewpoint_stack)

    def copy(self):
        return CameraDataset(self)

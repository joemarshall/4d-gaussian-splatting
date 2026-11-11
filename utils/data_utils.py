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
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

import safetensors.torch

from pywuffs import ImageDecoderType, PixelFormat
from pywuffs.aux import (
    ImageDecoder,
    ImageDecoderConfig,
    ImageDecoderFlags
)


class FastPNGToTorch:
    def __init__(self):
        self.config = ImageDecoderConfig()
        # All decoders are enabled by default
        self.config.enabled_decoders = [ImageDecoderType.PNG]
        # No metadata is reported by default
        self.config.flags = [ImageDecoderFlags.REPORT_METADATA_EXIF]
        # Pixel format is PixelFormat.BGRA_PREMUL by default
        self.config.pixel_format = PixelFormat.RGB
        self.decoder = ImageDecoder(self.config)

    def decode(self, path,device,non_blocking):
        decoding_result = self.decoder.decode(str(path))
        img_u8 = torch.from_numpy(decoding_result.pixbuf).to(device,non_blocking=non_blocking)
        img = img_u8.permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
        return img

class PILPNGToTorch:

    def decode(self, path,device,non_blocking):
        decoding_result = Image.open(path).convert("RGB")
        decoding_result = np.array(decoding_result)
        img_u8 = torch.from_numpy(decoding_result).to(device,non_blocking=non_blocking)
        img = img_u8.permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
        return img




class ImageCache:


    preload_cache = {}
    # this looks like a massive number of threads,
    # but wuffs is a) very fast and b) drops GIL 
    # during disk IO etc. so it is worth doing this
    executor = ThreadPoolExecutor(max_workers=50)

    @staticmethod 
    def loader_task(path,width,has_depth,output_device):
        decoder = FastPNGToTorch()
        non_blocking = True if output_device=="cuda" else False
        r_img = decoder.decode(path,output_device,non_blocking)

        if has_depth:
            depth_path = Path(path).with_suffix(".depth")
            if os.path.exists(depth_path):
                r_depth = torch.load(depth_path).to(dtype=torch.float32,device=output_device,non_blocking=non_blocking)
        else:
            r_depth = None
        return (r_img,r_depth)


    @staticmethod
    def preload_image(path,width,has_depth, output_device = "cpu"):
        # load into preload cache in a thread
        # so that we can return it next call
        if (path,width) not in ImageCache.preload_cache:
            #print("Preloading image for", path,width)
            ImageCache.preload_cache[(path,width)] = {"future": ImageCache.executor.submit(ImageCache.loader_task, path, width, has_depth, output_device),"refcount":1}
        else:
            ImageCache.preload_cache[(path,width)]["refcount"] += 1
            #print("Already preloaded", path,width,ImageCache.preload_cache[(path,width)]["refcount"])
        #print(sum(x["future"].done() for x in ImageCache.preload_cache.values())," ",end="",flush=True)
        return ImageCache.preload_cache[(path,width)]["future"]

    @staticmethod
    def get_image_for_file(path, width, has_depth):
        if (path,width) not in ImageCache.preload_cache:
            # load and wait
            print("Not preloaded but cache file exists, loading for", path)
            ImageCache.preload_image(path,width,has_depth).result()
        cache_entry = ImageCache.preload_cache[(path,width)]
        refcount = cache_entry["refcount"]
        if refcount <= 1:
            return ImageCache.preload_cache.pop((path,width))["future"].result()
        cache_entry["refcount"] -= 1
        return ImageCache.preload_cache[(path,width)]["future"].result()



class CameraDataset(Dataset):

    class IndexListSampler(Sampler[List[int]]):
        def __init__(self, dataset, index_list_batches):
            self.dataset = dataset
            self.index_list = index_list_batches

        def __len__(self):
            return len(self.index_list)

        def __iter__(self) -> Iterator[List[int]]:
            for batch in self.index_list:
                yield batch

    class ReadAheadSampler(Sampler[List[int]]):
        def __init__(self, dataset, child_sampler,num_batches_readahead=1,output_device = "cpu"):
            self.dataset = dataset
            self.child_sampler = child_sampler
            self.num_batches_readahead = num_batches_readahead
            self.output_device = output_device

        def __len__(self):
            return len(self.child_sampler)
        
        def __iter__(self) -> Iterator[List[int]]:
            # read ahead one batch 
            child_iter = iter(self.child_sampler)
            next_batch = None
            done = False
            readahead_buffer = []
            readahead_batches = self.num_batches_readahead
            while not done:
                # count_in_progress = len([x for x in ImageCache.preload_cache.values() if x["future"].running() ])
                # count_waiting = sum(x["refcount"] for x in ImageCache.preload_cache.values() if x["future"].done() )
                # print("ReadAheadSampler buffer size:", len(readahead_buffer), "in progress:", count_in_progress,"ready:",count_waiting)
                if len(readahead_buffer)<= readahead_batches and not done:
                    try:
                        next_batch =next(child_iter)
                        for x in next_batch:
                            ImageCache.preload_image(self.dataset.viewpoint_stack[x].image_path, self.dataset.viewpoint_stack[x].image_width, self.dataset.viewpoint_stack[x].depth is not None, output_device = self.output_device)
                        readahead_buffer.append(next_batch)
                    except StopIteration:
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
        return camera.camera_pose_id

    def _calc_cameras_and_timestamps(self):
        timestamp_set = set()
        viewpoint_set = set()

        for x in self.viewpoint_stack:
            viewpoint_set.add(self._get_camera_cache_key(x))
            timestamp_set.add(x.timestamp)
        self.num_cameras = len(viewpoint_set)
        self.timestamps = sorted(list(timestamp_set))
        self.different_cam_list = list(viewpoint_set)


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

    def get_frame_batch_sampler(self, suggested_batch_size=4,output_device = "cpu"):
        return CameraDataset.ReadAheadSampler(self,CameraDataset.TimeCoherentCameraAndFrameSampler(self),output_device=output_device)

    def get_index_batch_list_sampler(self, index_list,readahead=5,output_device = "cpu"):
        return CameraDataset.ReadAheadSampler(self,CameraDataset.IndexListSampler(self, index_list),num_batches_readahead=readahead,output_device=output_device)

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
            else:
                raise RuntimeError(
                    "ImageCache get_image_for_file returned None for image: {}".format(
                        viewpoint_cam.image_path
                    )
                )

        if self.names_only:
            return None, viewpoint_cam, None

    def __len__(self):
        return len(self.viewpoint_stack)

    def copy(self):
        return CameraDataset(self)

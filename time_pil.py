from PIL import Image

from pathlib import Path
import timeit
import tqdm

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.data_utils import ImageCache


from pywuffs import ImageDecoderType, PixelFormat
from pywuffs.aux import (
    ImageDecoder,
    ImageDecoderConfig,
    ImageDecoderFlags,
)

from pywuffs import ImageDecoderQuirks

def load_one_pil(path):
    im = Image.open(path)
    im.load()
    return np.mean(np.array(im))

def load_one_wuffs(path):
    config = ImageDecoderConfig()
    config.pixel_format = PixelFormat.BGR
    config.enabled_decoders = [ImageDecoderType.PNG]
    decoder = ImageDecoder(config)
    decoding_result = decoder.decode(str(path))
    return np.mean(np.array(decoding_result.pixbuf))

def load_all_pil():
    print("PIL:")
    NUM_THREADS=64
    pool=ThreadPoolExecutor(max_workers=NUM_THREADS)

    im_means = []
    all_images = list(Path("output/esper_ball/images").glob("*.png"))
    futures = []
    for x in tqdm.tqdm(all_images):
        futures.append(pool.submit(load_one_pil, x))
        if len(futures)>=NUM_THREADS:
            for future in as_completed(futures):
                im_means.append(future.result())
            futures = []
    for future in as_completed(futures):
        im_means.append(future.result())
    return im_means

def load_all_wuffs():
    print("WUFFS:")
    im_means = []
    config = ImageDecoderConfig()
    config.pixel_format = PixelFormat.BGRA_PREMUL
    config.enabled_decoders = [ImageDecoderType.PNG]
    all_images = list(Path("output/esper_ball/images").glob("*.png"))

    for x in tqdm.tqdm(all_images):
        decoder = ImageDecoder(config)
        decoding_result = decoder.decode(str(x))
        #np_im = np.array(decoding_result.pixbuf)
        #im_means.append(np.mean(np_im))
    return im_means

def load_all_imagecache(has_depth = False):
    print(f"ImageCache (depth:{has_depth}):")
    im_means = []
    all_images = list(Path("output/esper_ball/images").glob("*.png"))
    delay_list = []
    with tqdm.tqdm(total=len(all_images)) as pbar:
        for x in all_images:
            im = ImageCache.preload_image(str(x), 3927, has_depth=has_depth)
            delay_list.append(str(x))
            if len(delay_list)>=24:
                img = ImageCache.get_image_for_file(delay_list[0],3927)
                delay_list = delay_list[1:]
                pbar.update(1)
        while len(delay_list) > 0:
            im = ImageCache.get_image_for_file(delay_list[0],3927)
            delay_list = delay_list[1:]
            pbar.update(1)
    return im_means

class ReadAheadSampler:
    def __init__(self,dataset,num_batches_readahead=1):
        self.dataset = dataset 
        self.num_batches_readahead = num_batches_readahead

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self) -> Iterator[List[int]]:
        # read ahead one batch 
        child_iter = iter(self.dataset)
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
                    ImageCache.preload_image(next_batch, 3927, True)
                    readahead_buffer.append(next_batch)
                except StopIteration:
                    print("ReadAheadSampler child sampler exhausted")
                    done = True
            if len(readahead_buffer) > readahead_batches:
                yield readahead_buffer.pop(0)
            if done:
                while len(readahead_buffer) > 0:
                    yield readahead_buffer.pop(0)


def load_imagecache_sampler():
    print(f"ImageCache sampler")
    im_means = []
    all_images = list(Path("output/esper_ball/images").glob("*.png"))
    for x in tqdm.tqdm(ReadAheadSampler(all_images, num_batches_readahead=25)):
        img = ImageCache.get_image_for_file(x,3927,has_depth=True)

    
    

print (timeit.timeit(load_imagecache_sampler, number=1, globals=globals()))


print (timeit.timeit("load_all_imagecache(True)", number=1, globals=globals()))
print (timeit.timeit(load_all_imagecache, number=1, globals=globals()))

print (timeit.timeit(load_all_pil, number=1, globals=globals()))
print (timeit.timeit(load_all_wuffs, number=1, globals=globals()))



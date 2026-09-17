import numpy as np
import psutil
from pathlib import Path

from scene.streaming_tensor_cache import StreamingTensorCache


def dump_memory(label):
    process = psutil.Process()
    print(process.memory_full_info())
    all_rss = process.memory_full_info().uss
    print(f"{label}: {all_rss} bytes")
    mmap_total = 0
    for x in process.memory_maps(grouped=True):
        if x.path.find("model_output") >= 0:
            #print(x.path, x.rss)
            mmap_total+= x.rss
    print(f"{label}: {mmap_total} bytes in memory maps")
    print(f"{label}: {all_rss - mmap_total } bytes")


dump_memory("before torch")


import torch


def copy_to_numpy():
    input = torch.load("output/9moving/model_output/chkpnt_iter_500.pth", mmap = True,map_location="cpu",weights_only=False)
    dump_memory("On memory load")

    print(input[0].keys())

    initial_data = input[0]

    # for k, v in initial_data.items():
    #     if type(v) == torch.Tensor or type(v) == torch.nn.Parameter:
    #         print(k, v.shape, v.dtype, v.device)

    # make blob from each gaussian and insert into rtree table
    # first calculate the start and end times:
        
    # start_times = initial_data["_t"][0].detach().numpy().tolist()
    # end_times= (initial_data["_t"][0].detach()+torch.tensor(0.1)).numpy().tolist()

    maps = {}

    for col, val in initial_data.items():
        if type(val) == torch.Tensor or type(val) == torch.nn.Parameter:
            co_file = f"output/9moving/model_output/{col}.npmap"
            maps[col] = np.memmap(co_file, dtype=np.float32, mode="w+",shape=val.shape)
            maps[col][:] = val.detach().numpy()[:]

    return maps

def load_from_numpy():
    column_files = list(Path("output/9moving/model_output").glob("*.npmap"))
    maps = {}
    for col in column_files:
        maps[col] = np.memmap(str(col), dtype=np.float32, mode="r+")
    return maps

dump_memory("startup")
maps = load_from_numpy()
dump_memory("After mmaps")

# with torch.no_grad():
#     maps = copy_to_numpy()
#     dump_memory("After mmaps")

import gc
gc.collect()

del maps

import gc
gc.collect()

dump_memory("After deleting maps")








from ast import arg

import torch
import numpy as np

# make resizeable mmapped tensor


class StreamingTensorCache:
    def __init__(
        self, cache_folder, time_column="_t", duration_column="_visible_range"
    ):
        self.cache_folder = cache_folder
        self.cache = {}  # dict of memorymapped tensors for each variable
        self.loaded_indices = None
        self.time_column = time_column
        self.duration_column = duration_column

    def update_cache_tensor(self, var_name, tensor):
        # Update the memory-mapped tensor for the given variable name
        if var_name not in self.cache or self.loaded_indices is None:
            # Create a new memory-mapped tensor if it doesn't exist or we haven't loaded indices yet
            np_tensor = tensor.detach().cpu().numpy()
            fname = f"{self.cache_folder}/{var_name}.dat"
            np_tensor.tofile(fname)
            self.cache[var_name] = torch.from_file(fname, dtype=tensor.dtype, size=tensor.numel()).reshape(tensor.shape)
        else:
            # TODO: make this work if the shape of the tensor has changed, e.g. if number of gaussians has changed
            self.cache[var_name][
                self.loaded_indices
            ] = tensor.detach().cpu()  # Copy changed data from gpu to memory-mapped tensor

    def load_gaussians_for_time_range(self, time_range):
        # Load the memory-mapped tensors for the given time range
        if self.time_column not in self.cache or self.duration_column not in self.cache:
            raise ValueError(
                "Time and duration columns must be present in the cache to load time range."
            )

        in1 = self.cache[self.time_column].flatten() - self.cache[self.duration_column].flatten() <= time_range[1]
        in2 = self.cache[self.time_column].flatten() + self.cache[self.duration_column].flatten() >= time_range[0]
        self.loaded_indices = torch.logical_and(
            in1,
            in2,
        )
        loaded_tensors = {}
        for var_name, memmap_tensor in self.cache.items():
            loaded_tensors[var_name] = memmap_tensor[self.loaded_indices].to(
                "cuda"
            )  # Load to GPU
        return loaded_tensors

if __name__ == "__main__":
    from .gaussian_model import GaussianModel

    gaussians = GaussianModel(
        3,
        gaussian_dim=4,
        time_duration=10.0,
        rot_4d=True,
        force_sh_3d=False,
        sh_degree_t=0,
        prefilter_var=-1,
        densifiers=[],
    )

    checkpoint = "output/9moving/model_output/chkpnt_iter_500.pth"
    model_params, first_iter, total_training_points = torch.load(
        checkpoint, weights_only=False
    )
    gaussians.restore(model_params, None)
    gaussians.update_t_visible_range()

    histogram = torch.histc(gaussians._visible_range, bins=100, min=0.0, max=5.0)
    print("<0.5", (gaussians._visible_range<0.5).sum().item())
    print(">=0.5", (gaussians._visible_range>=0.5).sum().item())
    print(">=1.0", (gaussians._visible_range>=1.0).sum().item())
    print(">=2.0", (gaussians._visible_range>=2.0).sum().item())
    print(">=3.0", (gaussians._visible_range>=3.0).sum().item())
    print(">=4.0", (gaussians._visible_range>=4.0).sum().item())
    print(">=5.0", (gaussians._visible_range>=5.0).sum().item())
    cache = StreamingTensorCache("output/9moving/model_output")

    def add_to_cache(gaussians,var_name,update_tensor):
        tensor = getattr(gaussians, var_name, None)            
        if tensor is not None and type(tensor) == torch.Tensor or type(tensor) == torch.nn.Parameter:
            if update_tensor is not None:
                cache.update_cache_tensor(var_name, update_tensor)
                print("Updated cache for:", var_name, "shape:", update_tensor.shape)
            else:
                print(var_name)
                print("Adding to cache:", var_name, "shape:", tensor.shape)
                cache.update_cache_tensor(var_name, tensor)

    cache.update_cache_tensor("_visible_range", gaussians._visible_range)
    gaussians._make_save_or_restore_calls(add_to_cache, None)

    gpu_tensors = cache.load_gaussians_for_time_range((4.0, 5.0))
    # now update with the same values
    gaussians._make_save_or_restore_calls(add_to_cache, gpu_tensors)
    

import torch
from tqdm import tqdm
from torch.utils.cpp_extension import load
import os

_parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff-gaussian-rasterization-fastgs")
_glm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff-gaussian-rasterization-fastgs", "third_party", "glm")
torch_include_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "include", "torch", "csrc", "api", "include")
torch_lib_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "lib")
_C = load(
    name='diff_gaussian_rasterization_fastgs',
    extra_cflags=["-I " + _glm_dir, "-I" + torch_include_dir],
    extra_cuda_cflags=["-I " + _glm_dir, "-I" + torch_include_dir,"-G","-dopt=on"],
    extra_ldflags=["-LIBDIR:" + torch_lib_dir],
    sources=[
        os.path.join(_parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(_parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(_parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(_parent_dir, "cuda_rasterizer/adam.cu"),
        os.path.join(_parent_dir, "rasterize_points.cu"),
        os.path.join(_parent_dir, "ext.cpp"),
    ],
    verbose=True,
)


args=torch.load("crash_quick.dump")
args = [x.cuda() if isinstance(x, torch.Tensor) else x for x in args]
print(args)

(bg,
means3D,
colors_precomp,
opacities,
scales,
rotations,
scale_modifier,
cov3Ds_precomp,
metric_map,
viewmatrix,
projmatrix,
tanfovx,
tanfovy,
image_height,
image_width,
dc,
sh,
sh_degree,
campos,
mult,
prefiltered,
debug,
get_flag ) = args


while True:
    for x in tqdm(range(1000)):
        num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount = _C.rasterize_gaussians(*args)
        grad_out_color = torch.zeros((3, image_height, image_width)).cuda()
        print(grad_out_color.shape)

        bw_args = (
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            tanfovx,
            tanfovy,
            grad_out_color,
            dc,
            sh,
            sh_degree,
            campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            num_buckets,
            sampleBuffer,
            debug,
        )
        (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
            grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales,
            grad_rotations) = _C.rasterize_gaussians_backward(*bw_args)


        

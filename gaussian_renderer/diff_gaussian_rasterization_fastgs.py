#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# FastGS integration: fast rendering engine with AccuTile optimization
# and multi-view consistency scoring via metric_map accumulation.
#

from typing import NamedTuple, Tuple
import torch.nn as nn
import torch
import os
from torch.utils.cpp_extension import load

_parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diff-gaussian-rasterization-fastgs",
)
_glm_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diff-gaussian-rasterization-fastgs",
    "third_party",
    "glm",
)
torch_include_dir = os.path.join(
    os.getenv("CONDA_PREFIX"), "Library", "include", "torch", "csrc", "api", "include"
)
torch_lib_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "lib")
_C = load(
    name="diff_gaussian_rasterization_fastgs",
    extra_cflags=["-I " + _glm_dir, "-I" + torch_include_dir],
    extra_cuda_cflags=["-I " + _glm_dir, "-I" + torch_include_dir],
    # extra_cflags=["-I " + _glm_dir, "-I" + torch_include_dir, "-G"],
    # extra_cuda_cflags=["-I " + _glm_dir, "-I" + torch_include_dir,"-G"],
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


# @torch.library.custom_op("fastgs::rasterize_fwd", mutates_args=())
def C_RasterizeGaussiansCUDA(
    background: torch.Tensor,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    metric_map: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    dc: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    mult: float,
    prefiltered: bool,
    debug: bool,
    get_flag: bool,
    ts: torch.Tensor,
    scales_t: torch.Tensor,
    rotations_r: torch.Tensor,
    prefilter_var: float,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    sh_degree_t: int,
) -> Tuple[
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _C.rasterize_gaussians(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        metric_map,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        image_height,
        image_width,
        dc,
        sh,
        degree,
        campos,
        mult,
        prefiltered,
        debug,
        get_flag,
        ts,
        scales_t,
        rotations_r,
        prefilter_var,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        sh_degree_t,
    )


# @C_RasterizeGaussiansCUDA.register_fake
def C_RasterizeGaussiansCUDA_FAKE(
    background: torch.Tensor,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    metric_map: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    dc: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    mult: float,
    prefiltered: bool,
    debug: bool,
    get_flag: bool,
) -> Tuple[
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Fake implementation for compiler checks in torch dynamo
    ctx = torch.library.get_ctx()
    P = means3D.shape[0]
    out_color = torch.zeros((3, image_height, image_width), device=background.device)
    out_depth = torch.zeros((image_height, image_width), device=background.device)
    radii = torch.zeros((P,), device=background.device)
    metricCount = torch.zeros((P,), device=background.device, dtype=torch.int32)

    rendered = 0
    num_buckets = 0
    num_geom = ctx.new_dynamic_size()
    num_binning = ctx.new_dynamic_size()
    num_imgBuffer = ctx.new_dynamic_size()
    num_sampleBuffer = ctx.new_dynamic_size()
    geomBuffer = torch.tensor(shape=(num_geom,), dtype=torch.uint8, device=bg.device)
    binningBuffer = torch.tensor(
        shape=(num_binning,), dtype=torch.uint8, device=bg.device
    )
    imgBuffer = torch.tensor(
        shape=(num_imgBuffer,), dtype=torch.uint8, device=bg.device
    )
    sampleBuffer = torch.tensor(
        shape=(num_sampleBuffer,), dtype=torch.uint8, device=bg.device
    )

    return (
        rendered,
        num_buckets,
        out_color,
        out_depth,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        sampleBuffer,
        metricCount,
    )


# @torch.library.custom_op("fastgs::rasterize_bwd", mutates_args=())
def C_RasterizeGaussiansBackwardCUDA(
    background: torch.Tensor,
    means3D: torch.Tensor,
    out_means3D: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    dL_dout_color: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    geomBuffer: torch.Tensor,
    R: int,
    binningBuffer: torch.Tensor,
    imageBuffer: torch.Tensor,
    B: int,
    sampleBuffer: torch.Tensor,
    debug: bool,
    ts: torch.Tensor,
    opacity: torch.Tensor,
    scales_t: torch.Tensor,
    rotations_r: torch.Tensor,
    prefilter_var: float,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    sh_degree_t: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _C.rasterize_gaussians_backward(
        background,
        means3D,
        out_means3D,
        radii,
        colors,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        dL_dout_color,
        dc,
        sh,
        degree,
        campos,
        geomBuffer,
        R,
        binningBuffer,
        imageBuffer,
        B,
        sampleBuffer,
        debug,
        ts,
        opacity,
        scales_t,
        rotations_r,
        prefilter_var,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        sh_degree_t,
    )


# @C_RasterizeGaussiansBackwardCUDA.register_fake
def C_RasterizeGaussiansBackwardCUDA_FAKE(
    background: torch.Tensor,
    means3D: torch.Tensor,
    out_means3D: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    dL_dout_color: torch.Tensor,
    dc: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    campos: torch.Tensor,
    geomBuffer: torch.Tensor,
    R: int,
    binningBuffer: torch.Tensor,
    imageBuffer: torch.Tensor,
    B: int,
    sampleBuffer: torch.Tensor,
    debug: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Fake implementation for compiler checks in torch dynamo
    P = means3D.shape[0]

    dL_dmeans3D = torch.zeros_like(means3D)
    dL_dcolors = torch.zeros_like(colors)
    dL_dcov3D_precomp = torch.zeros_like(cov3D_precomp)
    dL_dsh = torch.zeros_like(sh)
    dL_dscales = torch.zeros_like(scales)
    dL_drotations = torch.zeros_like(rotations)
    dL_ddc = torch.zeros_like(dc)
    dL_dbackground = torch.zeros_like(background)
    dL_dviewmatrix = torch.zeros_like(viewmatrix)

    return (
        dL_dmeans3D,
        dL_dcolors,
        dL_dcov3D_precomp,
        dL_dsh,
        dL_dscales,
        dL_drotations,
        dL_ddc,
        dL_dbackground,
        dL_dviewmatrix,
    )


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    mult: float
    prefiltered: bool
    debug: bool
    get_flag: bool
    metric_map: torch.Tensor
    time_duration: float
    rot_4d: bool
    gaussian_dim: int
    force_sh_3d: bool
    sh_degree_t: int
    timestamp: float
    prefilter_var: float


def calculateGaussianVisibilityContribution(
    means3D,
    opacities,
    scales,
    rotations,
    # 4D Gaussian params
    ts: torch.Tensor,
    scales_t: torch.Tensor,
    rotations_r: torch.Tensor,
    time_duration: float,
    raster_settings,
    opacity_cutoff=0.01,
):
    """
    Compute per-gaussian pixel-contribution sums for visibility analysis.

    Gaussians are sorted back-to-front (reversed depth) within each tile.
    For each pixel the running product of alphas is tracked; the contribution
    weight assigned to gaussian n (in back-to-front order) is
        alpha_1 * alpha_2 * ... * alpha_n
    Processing stops for a pixel once this product drops below *opacity_cutoff*.

    Returns a float32 tensor of shape [P] containing the summed weighted pixel
    counts for each Gaussian across the whole image.
    """
    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    dc = torch.Tensor([])
    sh = torch.Tensor([])

    args = (
        means3D,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        dc,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.mult,
        raster_settings.prefiltered,
        raster_settings.debug,
        opacity_cutoff,
    )

    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args)
        try:
            gaussian_contrib = _C.calculate_gaussian_visibility_contribution(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_visibility.dump")
            print(
                "\nAn error occurred in calculateGaussianVisibilityContribution. "
                "Please forward snapshot_visibility.dump for debugging."
            )
            raise ex
    else:
        gaussian_contrib = _C.calculate_gaussian_visibility_contribution(*args)

    return gaussian_contrib


def rasterize_gaussians_fastgs(*,
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3D_precomp,
    raster_settings,
    # 4D Gaussian params
    ts: torch.Tensor,
    scales_t: torch.Tensor,
    rotations_r: torch.Tensor,
):
    return _RasterizeGaussiansFastGS.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings,
        ts,
        scales_t,
        rotations_r,
    )


class _RasterizeGaussiansFastGS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        raster_settings,
        ts,
        scales_t,
        rotations_r,
    ):

        inputs = (
            means3D,
            means2D,
            dc,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            ts,
            scales_t,
            rotations_r,
        )

        for i, x in enumerate(inputs):
             if x is not None and (isinstance(x, torch.Tensor) or isinstance(x,torch.nn.Parameter)) and x.shape[0]>0:
                 print(i, x.shape, x.dtype, x.device,torch.min(x),torch.max(x))
             else:
                 print(i)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            (
                raster_settings.metric_map
                if raster_settings.metric_map is not None
                else torch.zeros(
                    raster_settings.image_height * raster_settings.image_width,
                    dtype=torch.int32,
                    device="cuda",
                )
            ),
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.mult,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.get_flag if raster_settings.get_flag is not None else False,
            ts,
            scales_t,
            rotations_r,
            raster_settings.prefilter_var,
            raster_settings.timestamp if raster_settings.timestamp is not None else 0.0,
            (
                raster_settings.time_duration
                if raster_settings.time_duration is not None
                else 1.0
            ),
            raster_settings.rot_4d if raster_settings.rot_4d is not None else False,
            (
                raster_settings.gaussian_dim
                if raster_settings.gaussian_dim is not None
                else 3
            ),
            (
                raster_settings.force_sh_3d
                if raster_settings.force_sh_3d is not None
                else False
            ),
            raster_settings.sh_degree_t if raster_settings.sh_degree_t is not None else 0,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:

                (
                    num_rendered,
                    num_buckets,
                    color,
                    depth,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    sampleBuffer,
                    metricCount,
                    out_means3D,
                ) = C_RasterizeGaussiansCUDA(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw_fastgs.dump")
                print(
                    "\nAn error occurred in forward (fastgs). Please forward snapshot_fw_fastgs.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                num_buckets,
                color,
                depth,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                sampleBuffer,
                metricCount,
                out_means3D,
            ) = C_RasterizeGaussiansCUDA(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            out_means3D,
            scales,
            rotations,
            cov3D_precomp,
            radii,
            dc,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            ts,
            scales_t,
            rotations_r,
        )
        return color, depth, radii, metricCount, out_means3D

    @staticmethod
    def backward(
        ctx, grad_out_color, grad_depth, grad_radii, grad_metricCount, grad_out_means3D
    ):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            out_means3D,
            scales,
            rotations,
            cov3D_precomp,
            radii,
            dc,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            ts,
            scales_t,
            rotations_r,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            out_means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            num_buckets,
            sampleBuffer,
            raster_settings.debug,
            ts,
            opacities,
            scales_t,
            rotations_r,
            (
                raster_settings.prefilter_var
                if raster_settings.prefilter_var is not None
                else 0.0
            ),
            raster_settings.timestamp if raster_settings.timestamp is not None else 0.0,
            (
                raster_settings.time_duration
                if raster_settings.time_duration is not None
                else 1.0
            ),
            raster_settings.rot_4d if raster_settings.rot_4d is not None else False,
            (
                raster_settings.gaussian_dim
                if raster_settings.gaussian_dim is not None
                else 3
            ),
            (
                raster_settings.force_sh_3d
                if raster_settings.force_sh_3d is not None
                else False
            ),
            raster_settings.sh_degree_t if raster_settings.sh_degree_t is not None else 0,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_dc,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_ts,
                    grad_scales_t,
                    grad_rotations_r,
                    grad_opacity_out,
                ) = C_RasterizeGaussiansBackwardCUDA(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw_fastgs.dump")
                print(
                    "\nAn error occurred in backward (fastgs). Writing snapshot_bw_fastgs.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_dc,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_ts,
                grad_scales_t,
                grad_rotations_r,
                grad_opacity_out,
            ) = C_RasterizeGaussiansBackwardCUDA(*args)

        grads = (
            grad_means3D,
            grad_means2D,  # shape (P, 4): [:, :2] = regular grads, [:, 2:] = abs grads
            grad_dc,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster settings (no gradients)
            grad_ts,
            grad_scales_t,
            grad_rotations_r,
        )

        grad_names = [
            "grad_means3D",
            "grad_means2D",
            "grad_dc",
            "grad_sh",
            "grad_colors_precomp",
            "grad_opacities",
            "grad_scales",
            "grad_rotations",
            "grad_cov3Ds_precomp",
            "None (raster_settings)",
            "grad_ts",
            "grad_scales_t",
            "grad_rotations_r",
        ]
        print("\n---------------------------------")
        print("\nBackward pass gradients (fastgs):")
        print("\n---------------------------------")
        for x in sorted(zip(grad_names, grads)):
            mean_val = torch.mean(x[1]) if type(x[1]) is torch.Tensor else None
            max_val = torch.max(x[1]) if type(x[1]) is torch.Tensor else None
            min_val = torch.min(x[1]) if type(x[1]) is torch.Tensor else None
            print(f"{x[0]}: {min_val}, {mean_val}, {max_val}")
        print("\n---------------------------------")

        return grads


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        dc=None,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        ts=None,
        scales_t=None,
        rotations_r=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if ts is None:
            if raster_settings.gaussian_dim == 4:
                print(
                    "Warning: Rasterization expects 4D Gaussian parameters but no timestamps provided. Defaulting to zeros."
                )
            ts = torch.Tensor([])
        if scales_t is None:
            scales_t = torch.Tensor([])
        if rotations_r is None:
            rotations_r = torch.Tensor([])

        return rasterize_gaussians_fastgs(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            sh = shs,
            colors_precomp = colors_precomp,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            raster_settings = raster_settings,
            ts = ts,
            scales_t = scales_t,
            rotations_r = rotations_r,
        )

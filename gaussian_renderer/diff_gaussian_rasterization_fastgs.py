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

from typing import NamedTuple
import torch.nn as nn
import torch
import os
from torch.utils.cpp_extension import load

_parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization-fastgs")
_glm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization-fastgs", "third_party", "glm")
torch_include_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "include", "torch", "csrc", "api", "include")
torch_lib_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "lib")
_C = load(
    name='diff_gaussian_rasterization_fastgs',
    extra_cflags=["-I " + _glm_dir, "-I" + torch_include_dir, "-g"],
    extra_cuda_cflags=["-I " + _glm_dir, "-I" + torch_include_dir, "-g"],
    extra_ldflags=["-L " + torch_lib_dir],
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


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
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
    get_flag: bool = False
    metric_map: torch.Tensor = None


def rasterize_gaussians_fastgs(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
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
        cov3Ds_precomp,
        raster_settings,
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
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.metric_map if raster_settings.metric_map is not None
                else torch.zeros(raster_settings.image_height * raster_settings.image_width,
                                 dtype=torch.int32, device="cuda"),
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
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                torch.save(cpu_args, "snapshot_fw_fastgs.dump")
                num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw_fastgs.dump")
                print("\nAn error occurred in forward (fastgs). Please forward snapshot_fw_fastgs.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(
            colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
            radii, dc, sh, opacities,
            geomBuffer, binningBuffer, imgBuffer, sampleBuffer,
        )
        return color, radii, metricCount

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_metricCount):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        (
            colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
            radii, dc, sh, opacities,
            geomBuffer, binningBuffer, imgBuffer, sampleBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
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
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
                 grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales,
                 grad_rotations) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw_fastgs.dump")
                print("\nAn error occurred in backward (fastgs). Writing snapshot_bw_fastgs.dump for debugging.\n")
                raise ex
        else:
            (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
             grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales,
             grad_rotations) = _C.rasterize_gaussians_backward(*args)

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
            None,  # raster_settings
        )

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
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

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

        return rasterize_gaussians_fastgs(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

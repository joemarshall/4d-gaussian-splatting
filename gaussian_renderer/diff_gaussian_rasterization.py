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

from typing import NamedTuple
import torch.nn as nn
import torch

# from . import _C
import os
from torch.utils.cpp_extension import load

from typing import Tuple

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diff-gaussian-rasterization",
)
_C = load(
    name="diff_gaussian_rasterization",
    extra_cuda_cflags=["-I " + os.path.join(parent_dir, "third_party/glm/"), "-O3"],#,"-g"],
    extra_cflags=["/FS","/GL","/O2"],# without this windows builds are flaky because of pdb files being locked
    sources=[
        os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/calc_contribution.cu"),
        os.path.join(parent_dir, "rasterize_points.cu"),
        os.path.join(parent_dir, "ext.cpp"),
    ],
    verbose=True,
)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    flow_2d,
    opacities,
    ts,
    scales,
    scales_t,
    rotations,
    rotations_r,
    cov3D_precomp,
    prefilter_var,
    raster_settings,
):
    all_results = _forward_op(
        means3D,
        means2D,
        sh,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3D_precomp,
        prefilter_var,
        raster_settings.bg,
        raster_settings.scale_modifier,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        raster_settings.sh_degree,
        raster_settings.sh_degree_t,
        raster_settings.campos,
        raster_settings.timestamp,
        raster_settings.time_duration,
        raster_settings.rot_4d,
        raster_settings.gaussian_dim,
        raster_settings.force_sh_3d,
        raster_settings.prefiltered,
        raster_settings.debug,
        raster_settings.metric_map,
    )
    return all_results[
        0:7
    ]  # return the tensors we actually want, and not those needed for backward pass


# this needs to have raster_settings unpacked so
# everything is tensors
@torch.library.custom_op("gaussian_4d::fwd_op", mutates_args=())
def _forward_op(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    flow_2d: torch.Tensor,
    opacities: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tanfovx: float,
    tanfovy: float,
    image_height: int,
    image_width: int,
    sh_degree: int,
    sh_degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    prefiltered: bool,
    debug: bool,
    metric_map: torch.Tensor,
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
    int,
]:
    # need to
    # 1) call forward with correct args
    # 2) return values we actually want, as well as
    #    those that are needed in the backward pass
    # n.b. in backward pass return None gradients for the non-optimised args

    # Restructure arguments the way that the C++ lib expects them
    args = (
        bg,
        means3D,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        scale_modifier,
        cov3D_precomp,
        prefilter_var,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
        sh,
        sh_degree,
        sh_degree_t,
        campos,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        prefiltered,
        debug,
        metric_map,
    )

    # print("-----------------------------------  ")
    # for x in args:
    #     print(type(x), x.shape if isinstance(x, torch.Tensor) else None,x.dtype if isinstance(x, torch.Tensor) else None)

    # Invoke C++/CUDA rasterizer
    if debug:
        cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
        try:
            (
                num_rendered,
                color,
                flow,
                depth,
                T,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                covs_com,
                out_means3D,
                out_metric_counts,
            ) = C_rasterize_gaussians(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw.dump")
            print(
                "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
            )
            raise ex
    else:
        (
            num_rendered,
            color,
            flow,
            depth,
            T,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            covs_com,
            out_means3D,
            out_metric_counts,
        ) = C_rasterize_gaussians(*args)


    # print(
    #     "Real outs:",
    #      (
    #     color.shape,
    #     radii.shape,
    #     depth.shape,
    #     (1 - T).shape,
    #     flow.shape,
    #     covs_com.shape,
    #     out_metric_counts,
    #     out_means3D.shape,
    #     geomBuffer.shape,
    #     binningBuffer.shape,
    #     imgBuffer.shape,
    #     num_rendered,         ),
    #  )


    return (
        color,
        radii,
        depth,
        1 - T,
        flow,
        covs_com,
        out_metric_counts,
        out_means3D,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        num_rendered,
    )


@_forward_op.register_fake
def _(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: torch.Tensor,
    flow_2d: torch.Tensor,
    opacities: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tanfovx: float,
    tanfovy: float,
    image_height: int,
    image_width: int,
    sh_degree: int,
    sh_degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    prefiltered: bool,
    debug: bool,
    metric_map: torch.Tensor,
):
    ctx = torch.library.get_ctx()
    P = means3D.shape[0]

    color = means3D.new_empty(3, image_height,image_width)
    radii = means3D.new_empty(P, dtype=torch.int32)
    depth = means3D.new_empty(1, image_height,image_width)
    T = means3D.new_empty(1, image_height,image_width)
    flow = means3D.new_empty( 2, image_height, image_width)
    if metric_map.ndim >0:
        out_metric_counts = means3D.new_empty(P, dtype=torch.int32)
    else:
        out_metric_counts = means3D.new_empty(0, dtype=torch.int32)
    out_means3D = means3D.clone()

    num_geom = ctx.new_dynamic_size()
    num_binning = ctx.new_dynamic_size()
    num_imgBuffer = ctx.new_dynamic_size()
    geomBuffer = means3D.new_empty(num_geom, dtype=torch.uint8)
    binningBuffer = means3D.new_empty(num_binning, dtype=torch.uint8)
    imgBuffer = means3D.new_empty(num_imgBuffer, dtype=torch.uint8)

    covs_com = means3D.new_empty((P, 6))

    num_rendered = 0

    # print(
    #     "FAKE outs:",
    #      (
    #     color.shape,
    #     radii.shape,
    #     depth.shape,
    #     T.shape,
    #     flow.shape,
    #     covs_com.shape,
    #     out_metric_counts,
    #     out_means3D.shape,
    #     geomBuffer.shape,
    #     binningBuffer.shape,
    #     imgBuffer.shape,
    #     num_rendered,         ),
    #  )


    return (
        color,
        radii,
        depth,
        T,
        flow,
        covs_com,
        out_metric_counts,
        out_means3D,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        num_rendered,
    ) 




def _setup_ctx(ctx, inputs, output):
    (
        means3D,
        means2D,
        sh,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3D_precomp,
        prefilter_var,
        bg,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
        sh_degree,
        sh_degree_t,
        campos,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        prefiltered,
        debug,
        metric_map,
    ) = inputs

    (
        color,
        radii,
        depth,
        minusT,
        flow,
        covs_com,
        out_metric_counts,
        out_means3D,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        num_rendered,
    ) = output

    ctx.num_rendered = num_rendered
    ctx.prefilter_var = prefilter_var
    ctx.save_for_backward(
        colors_precomp,
        means3D,
        out_means3D,
        scales,
        rotations,
        cov3D_precomp,
        radii,
        sh,
        flow_2d,
        opacities,
        ts,
        scales_t,
        rotations_r,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        bg,
        viewmatrix,
        projmatrix,
        campos,
    )

    ctx.scale_modifier = scale_modifier
    ctx.tanfovx = tanfovx
    ctx.tanfovy = tanfovy
    ctx.image_height = image_height
    ctx.image_width = image_width
    ctx.sh_degree = sh_degree
    ctx.sh_degree_t = sh_degree_t
    ctx.timestamp = timestamp
    ctx.time_duration = time_duration
    ctx.rot_4d = rot_4d
    ctx.gaussian_dim = gaussian_dim
    ctx.force_sh_3d = force_sh_3d
    ctx.prefiltered = prefiltered
    ctx.debug = debug


def _backward_op(
    ctx,
    grad_out_color,
    grad_radii,
    grad_depth,
    grad_alpha,
    grad_flow,
    grad_covs_com,
    grad_metric_counts,
    grad_out_means3D,
    grad_geomBuffer,
    grad_binningBuffer,
    grad_imgBuffer,
    grad_num_rendered,    
):
#    print("BACKWARD OP!!!!")
    # Restore necessary values from context
    num_rendered = ctx.num_rendered
    prefilter_var = ctx.prefilter_var
    (
        colors_precomp,
        means3D,
        out_means3D,
        scales,
        rotations,
        cov3D_precomp,
        radii,
        sh,
        flow_2d,
        opacities,
        ts,
        scales_t,
        rotations_r,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        bg,
        viewmatrix,
        projmatrix,
        campos,
    ) = ctx.saved_tensors

    scale_modifier = ctx.scale_modifier
    tanfovx = ctx.tanfovx
    tanfovy = ctx.tanfovy
    image_height = ctx.image_height
    image_width = ctx.image_width
    sh_degree = ctx.sh_degree
    sh_degree_t = ctx.sh_degree_t
    timestamp = ctx.timestamp
    time_duration = ctx.time_duration
    rot_4d = ctx.rot_4d
    gaussian_dim = ctx.gaussian_dim
    force_sh_3d = ctx.force_sh_3d
    prefiltered = ctx.prefiltered
    debug = ctx.debug
    rot_4d = ctx.rot_4d

    # Restructure args as C++ method expects them
    args = (
        bg,
        means3D,
        out_means3D,
        radii,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        scale_modifier,
        cov3D_precomp,
        prefilter_var,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        grad_out_color,
        grad_depth,
        grad_alpha,
        grad_flow,
        sh,
        sh_degree,
        sh_degree_t,
        campos,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        debug,
    )


    # Compute gradients for relevant tensors by invoking backward method
    if debug:
        cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
        try:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3D_precomp,
                grad_sh,
                grad_flows,
                grad_ts,
                grad_scales,
                grad_scales_t,
                grad_rotations,
                grad_rotations_r,
            ) = C_rasterize_gaussians_backward(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_bw.dump")
            print(
                "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
            )
            raise ex
    else:
        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3D_precomp,
            grad_sh,
            grad_flows,
            grad_ts,
            grad_scales,
            grad_scales_t,
            grad_rotations,
            grad_rotations_r,
        ) = C_rasterize_gaussians_backward(*args)

    grads = (
        grad_means3D,
        grad_means2D,
        grad_sh,
        grad_colors_precomp,
        grad_flows,
        grad_opacities,
        grad_ts,
        grad_scales,
        grad_scales_t,
        grad_rotations,
        grad_rotations_r,
        grad_cov3D_precomp,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    # print("Backward pass - return types:")
    # for x in grads:
    #     print(type(x),x.shape if isinstance(x, torch.Tensor) else None, x.dtype if isinstance(x, torch.Tensor) else None)


    grad_names = [
        "grad_means3D",
        "grad_means2D",
        "grad_sh",
        "grad_colors_precomp",
        "grad_flows",
        "grad_opacities",
        "grad_ts",
        "grad_scales",
        "grad_scales_t",
        "grad_rotations",
        "grad_rotations_r",
        "grad_cov3D_precomp",
    ]

    # print("\n---------------------------------")
    # print("\nBackward pass gradients (old):")
    # print("\n---------------------------------")
    # for x in sorted(zip(grad_names, grads)):
    #     mean_val = torch.mean(x[1]) if type(x[1]) is torch.Tensor else None
    #     max_val = torch.max(x[1]) if type(x[1]) is torch.Tensor else None
    #     min_val = torch.min(x[1]) if type(x[1]) is torch.Tensor else None
    #     print(f"{x[0]}: {min_val}, {mean_val}, {max_val}")
    # print("\n---------------------------------")

    return grads


_forward_op.register_autograd(_backward_op, setup_context=_setup_ctx)


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
    sh_degree_t: int
    campos: torch.Tensor
    timestamp: float
    time_duration: float
    rot_4d: bool
    gaussian_dim: int
    force_sh_3d: bool
    prefiltered: bool
    debug: bool
    metric_map: torch.Tensor


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    # not differentiable, for pruning / densification only
    @torch.no_grad()
    def calculate_gaussian_contributions(
        self,
        means3D,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3D_precomp,
        prefilter_var,
    ):
        raster_settings = self.raster_settings

        if ts is None:
            ts = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if scales_t is None:
            scales_t = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if rotations_r is None:
            rotations_r = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Restructure arguments the way that the C++ lib expects them
        # n.b. no colour arguments needed because we're only rendering opacity
        args = (
            means3D,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            raster_settings.scale_modifier,
            cov3D_precomp,
            prefilter_var,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.timestamp,
            raster_settings.time_duration,
            raster_settings.rot_4d,
            raster_settings.gaussian_dim,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.metric_map,  # metric map is tensor::float of error per pixel, which is multiplied by contribution if provided
        )

        # print("sending args:")
        # for x in args:
        #     print(type(x), x.shape if isinstance(x, torch.Tensor) else None,x.dtype if isinstance(x, torch.Tensor) else None)
        # print("---------------")

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                num_rendered, view_contribution_tensor, error_contribution_tensor = (
                    _C.calculate_gaussian_contributions(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, view_contribution_tensor, error_contribution_tensor = (
                _C.calculate_gaussian_contributions(*args)
            )

        return num_rendered, view_contribution_tensor, error_contribution_tensor

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        flow_2d=None,
        ts=None,
        scales=None,
        scales_t=None,
        rotations=None,
        rotations_r=None,
        cov3D_precomp=None,
        prefilter_var=-1.0,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please `provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if (
            self.raster_settings.rot_4d
            and cov3D_precomp is None
            and (rotations_r is None or scales_t is None or ts is None)
        ):
            raise Exception(
                "Please provide exactly rotations_r and scales_t and ts if rot_4d and cov3D_precomp is None!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if flow_2d is None:
            flow_2d = torch.Tensor([])

        if ts is None:
            ts = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if scales_t is None:
            scales_t = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if rotations_r is None:
            rotations_r = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            cov3D_precomp,
            prefilter_var,
            raster_settings,
        )


## Custom Operator so it can be inside torch.compile
# @torch.library.custom_op("gaussian_4d::rasterize_fwd", mutates_args=())
def C_rasterize_gaussians(
    background: torch.Tensor,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    flows: torch.Tensor,
    opacity: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    sh: torch.Tensor,
    degree: int,
    degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    prefiltered: bool,
    debug: bool,
    metric_map: torch.Tensor,
) -> Tuple[
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
    torch.Tensor,
    torch.Tensor,
]:
    args = (
        background,
        means3D,
        colors,
        flows,
        opacity,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        scale_modifier,
        cov3D_precomp,
        prefilter_var,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        image_height,
        image_width,
        sh,
        degree,
        degree_t,
        campos,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        prefiltered,
        debug,
        metric_map,
    )
    # print("C++ rasterizer called with args:")
    # for x in args:
    #      print(type(x),x.device if hasattr(x,"device") else "N/A")
    return _C.rasterize_gaussians(*args)


# fake version of above so torch compile can use the op
def C_rasterize_gaussians_FAKE(
    background: torch.Tensor,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    flows: torch.Tensor,
    opacity: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    sh: torch.Tensor,
    degree: int,
    degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    prefiltered: bool,
    debug: bool,
    metric_map: torch.Tensor,
) -> Tuple[
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
    torch.Tensor,
    torch.Tensor,
]:

    ctx = torch.library.get_ctx()

    P = means3D.shape[0]
    rendered = 0

    if ctx == None:
        num_geom = 72
        num_binning = 128
        num_imgBuffer = 12345
    else:
        num_geom = ctx.new_dynamic_size()
        num_binning = ctx.new_dynamic_size()
        num_imgBuffer = ctx.new_dynamic_size()

    geom_shape = [num_geom]
    binning_shape = [num_binning]
    imgBuffer_shape = [num_imgBuffer]

    out_color = means3D.new_empty((3, image_height, image_width))
    out_flow = means3D.new_empty((2, image_height, image_width))
    out_depth = means3D.new_empty((1, image_height, image_width))
    out_T = means3D.new_empty((1, image_height, image_width))
    radii = means3D.new_empty((P,), dtype=torch.int32)

    geomBuffer = means3D.new_empty(geom_shape, dtype=torch.uint8)
    binningBuffer = means3D.new_empty(binning_shape, dtype=torch.uint8)
    imgBuffer = means3D.new_empty(imgBuffer_shape, dtype=torch.uint8)
    covs3D_com = means3D.new_empty((P, 6))
    out_means3D = means3D.new_empty((P, 3))
    out_metric_count = means3D.new_empty((P,), dtype=torch.int32)

    # print(
    #     "FAKE outs:",
    #     (
    #         rendered,
    #         out_color,
    #         out_flow,
    #         out_depth,
    #         out_T,
    #         radii,
    #         geomBuffer,
    #         binningBuffer,
    #         imgBuffer,
    #         covs3D_com,
    #         out_means3D,
    #         out_metric_count,
    #     ),
    # )

    return (
        rendered,
        out_color,
        out_flow,
        out_depth,
        out_T,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
        covs3D_com,
        out_means3D,
        out_metric_count,
    )


@torch.library.custom_op("gaussian_4d::rasterize_bwd", mutates_args=())
def C_rasterize_gaussians_backward(
    background: torch.Tensor,
    means3D: torch.Tensor,
    out_means3D: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    flows_2d: torch.Tensor,
    opacities: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    dL_dout_color: torch.Tensor,
    dL_dout_depth: torch.Tensor,
    dL_dout_mask: torch.Tensor,
    dL_dout_flow: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    geomBuffer: torch.Tensor,
    R: int,
    binningBuffer: torch.Tensor,
    imageBuffer: torch.Tensor,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    retval =  _C.rasterize_gaussians_backward(
        background,
        means3D,
        out_means3D,
        radii,
        colors,
        flows_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        scale_modifier,
        cov3D_precomp,
        prefilter_var,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        dL_dout_color,
        dL_dout_depth,
        dL_dout_mask,
        dL_dout_flow,
        sh,
        degree,
        degree_t,
        campos,
        timestamp,
        time_duration,
        rot_4d,
        gaussian_dim,
        force_sh_3d,
        geomBuffer,
        R,
        binningBuffer,
        imageBuffer,
        debug,
    )

    (
        dL_dmeans2D,
        dL_dcolors,
        dL_dopacity,
        dL_dmeans3D,
        dL_dcov3D,
        dL_dsh,
        dL_dflows,
        dL_dts,
        dL_dscales,
        dL_dscales_t,
        dL_drotations,
        dL_drotations_r,
    )=retval

    # print(
    #     "Real BW:",
    #      (
    #     dL_dmeans2D.shape,
    #     dL_dcolors.shape,
    #     dL_dopacity.shape,
    #     dL_dmeans3D.shape,
    #     dL_dcov3D.shape,
    #     dL_dsh.shape,
    #     dL_dflows.shape,
    #     dL_dts.shape,
    #     dL_dscales.shape,
    #     dL_dscales_t.shape,
    #     dL_drotations.shape,
    #     dL_drotations_r.shape,
    #  ))


    return retval




@C_rasterize_gaussians_backward.register_fake
def C_rasterize_gaussians_backward_FAKE(
    background: torch.Tensor,
    means3D: torch.Tensor,
    out_means3D: torch.Tensor,
    radii: torch.Tensor,
    colors: torch.Tensor,
    flows_2d: torch.Tensor,
    opacities: torch.Tensor,
    ts: torch.Tensor,
    scales: torch.Tensor,
    scales_t: torch.Tensor,
    rotations: torch.Tensor,
    rotations_r: torch.Tensor,
    scale_modifier: float,
    cov3D_precomp: torch.Tensor,
    prefilter_var: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    tan_fovx: float,
    tan_fovy: float,
    dL_dout_color: torch.Tensor,
    dL_dout_depth: torch.Tensor,
    dL_dout_mask: torch.Tensor,
    dL_dout_flow: torch.Tensor,
    sh: torch.Tensor,
    degree: int,
    degree_t: int,
    campos: torch.Tensor,
    timestamp: float,
    time_duration: float,
    rot_4d: bool,
    gaussian_dim: int,
    force_sh_3d: bool,
    geomBuffer: torch.Tensor,
    R: int,
    binningBuffer: torch.Tensor,
    imageBuffer: torch.Tensor,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    P = means3D.shape[0]
    M = sh.shape[1]
    dL_dmeans3D = means3D.new_empty((P, 3))
    dL_dmeans2D = means3D.new_empty((P, 3))
    dL_dcolors = means3D.new_empty((P, 3))
    dL_dflows = means3D.new_empty((P, 2))
    dL_dopacity = means3D.new_empty((P, 1))
    dL_dts = means3D.new_empty((P, 1))
    dL_dcov3D = means3D.new_empty((P, 6))
    dL_dsh = means3D.new_empty((P, M, 3))
    dL_dscales = means3D.new_empty((P, 3))
    dL_dscales_t = means3D.new_empty((P, 1))
    dL_drotations = means3D.new_empty((P, 4))
    dL_drotations_r = means3D.new_empty((P, 4))

    # print(
    #     "Fake BW:",
    #      (
    #     dL_dmeans2D.shape,
    #     dL_dcolors.shape,
    #     dL_dopacity.shape,
    #     dL_dmeans3D.shape,
    #     dL_dcov3D.shape,
    #     dL_dsh.shape,
    #     dL_dflows.shape,
    #     dL_dts.shape,
    #     dL_dscales.shape,
    #     dL_dscales_t.shape,
    #     dL_drotations.shape,
    #     dL_drotations_r.shape,
    #  ))


    return (
        dL_dmeans2D,
        dL_dcolors,
        dL_dopacity,
        dL_dmeans3D,
        dL_dcov3D,
        dL_dsh,
        dL_dflows,
        dL_dts,
        dL_dscales,
        dL_dscales_t,
        dL_drotations,
        dL_drotations_r,
    )

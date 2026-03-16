"""Tests for StochasticGaussianRasterizer.

Verifies:
1. Output shapes match the CUDA renderer interface.
2. Deterministic mode (num_samples=-1) produces consistent results across
   two calls (no randomness).
3. Stochastic mode (num_samples=K) produces an unbiased estimate:
   the mean over many stochastic renders converges to the deterministic render.
4. Gradients flow correctly to Gaussian parameters.
5. The renderer handles the edge case of zero Gaussians gracefully.
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

# Import the stochastic module directly to avoid triggering the CUDA extension
# load that happens when `gaussian_renderer` package __init__ is imported.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "stochastic_rasterization",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "gaussian_renderer", "stochastic_rasterization.py"),
)
_mod = _ilu.module_from_spec(_spec)

# Stub the heavy utils imports so we can test without CUDA / full environment
import types
# Provide minimal stubs for utils that the module needs
_utils_sh = types.ModuleType("utils.sh_utils")
_utils_sh.eval_sh = None
_utils_sh.eval_shfs_4d = None
sys.modules.setdefault("utils", types.ModuleType("utils"))
sys.modules.setdefault("utils.sh_utils", _utils_sh)

# Stub utils.general_utils with the real PyTorch implementations
_utils_gen = types.ModuleType("utils.general_utils")


def _build_rotation(r):
    norm = torch.sqrt((r * r).sum(-1))
    q = r / norm[:, None]
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(q.shape[0], 3, 3)
    R[:, 0, 0] = 1 - 2 * (c * c + d * d)
    R[:, 0, 1] = 2 * (b * c - a * d)
    R[:, 0, 2] = 2 * (b * d + a * c)
    R[:, 1, 0] = 2 * (b * c + a * d)
    R[:, 1, 1] = 1 - 2 * (b * b + d * d)
    R[:, 1, 2] = 2 * (c * d - a * b)
    R[:, 2, 0] = 2 * (b * d - a * c)
    R[:, 2, 1] = 2 * (c * d + a * b)
    R[:, 2, 2] = 1 - 2 * (b * b + c * c)
    return R


def _build_scaling_rotation(s, r):
    L = torch.zeros(s.shape[0], 3, 3)
    R = _build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return L @ R


def _build_rotation_4d(l, r):
    l_n = l / l.norm(dim=-1, keepdim=True)
    r_n = r / r.norm(dim=-1, keepdim=True)
    a, b, c, d = l_n.unbind(-1)
    p, q, rv, s = r_n.unbind(-1)
    M_l = torch.stack([a, -b, -c, -d,
                        b, a, -d, c,
                        c, d, a, -b,
                        d, -c, b, a]).view(4, 4, -1).permute(2, 0, 1)
    M_r = torch.stack([p, q, rv, s,
                        -q, p, -s, rv,
                        -rv, s, p, -q,
                        -s, -rv, q, p]).view(4, 4, -1).permute(2, 0, 1)
    A = M_l @ M_r
    return A.flip(1, 2)


def _build_scaling_rotation_4d(s, l, r):
    L = torch.zeros(s.shape[0], 4, 4)
    R = _build_rotation_4d(l, r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L[:, 3, 3] = s[:, 3]
    return R @ L


_utils_gen.build_rotation = _build_rotation
_utils_gen.build_scaling_rotation = _build_scaling_rotation
_utils_gen.build_scaling_rotation_4d = _build_scaling_rotation_4d
sys.modules.setdefault("utils.general_utils", _utils_gen)

_spec.loader.exec_module(_mod)

GaussianRasterizationSettings = _mod.GaussianRasterizationSettings
StochasticGaussianRasterizer = _mod.StochasticGaussianRasterizer
_build_cov3D = _mod._build_cov3D
_project_gaussians = _mod._project_gaussians


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster_settings(H=32, W=32, device="cpu"):
    """Construct minimal raster settings for unit tests."""
    fov = math.pi / 4.0
    tan_half = math.tan(fov / 2.0)

    # world_view_transform = getWorld2View2(R, T).T
    # For identity camera looking down +Z, shifted by 5 along Z:
    view = torch.eye(4, device=device)
    view[3, 2] = 5.0   # translation: camera at z=-5 sees world z>0 at positive depth

    # projection_matrix = getProjectionMatrix(...).T
    # Original P (before transpose): P[3,2]=1, P[2,2]=far/(far-near), P[2,3]=-far*near/(far-near)
    near, far = 0.01, 100.0
    f = 1.0 / tan_half
    proj_T = torch.zeros(4, 4, device=device)   # already-transposed form
    proj_T[0, 0] = f
    proj_T[1, 1] = f
    proj_T[2, 2] = far / (far - near)
    proj_T[2, 3] = -(far * near) / (far - near)
    proj_T[3, 2] = 1.0                          # w-component divisor from z

    full_proj = view @ proj_T

    campos = view.inverse()[3, :3]  # camera position in world space

    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tan_half,
        tanfovy=tan_half,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full_proj,
        sh_degree=0,
        sh_degree_t=0,
        campos=campos,
        timestamp=0.0,
        time_duration=1.0,
        rot_4d=False,
        gaussian_dim=3,
        force_sh_3d=False,
        prefiltered=False,
        debug=False,
    )


def _make_gaussians(N=20, device="cpu", seed=42):
    """Create N random 3-D Gaussians at origin with small spread."""
    torch.manual_seed(seed)
    means3D = torch.randn(N, 3, device=device) * 0.5  # near origin
    means3D[:, 2] += 2.0                               # push to z=2 (in front of cam)
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)

    scales    = torch.ones(N, 3, device=device) * 0.3
    rotations = torch.zeros(N, 4, device=device)
    rotations[:, 0] = 1.0                              # identity quaternion

    opacities = torch.ones(N, 1, device=device) * 0.5
    colors    = torch.rand(N, 3, device=device)

    return means3D, means2D, scales, rotations, opacities, colors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    """The renderer must return tensors with the correct shapes."""

    def test_output_shapes(self):
        H, W, N = 32, 32, 20
        rs  = _make_raster_settings(H, W)
        rz  = StochasticGaussianRasterizer(rs, num_samples=4)

        means3D, means2D, scales, rots, opacities, colors = _make_gaussians(N)

        out = rz(
            means3D=means3D, means2D=means2D,
            opacities=opacities,
            colors_precomp=colors,
            scales=scales, rotations=rots,
        )
        color, radii, depth, alpha, flow, covs = out

        assert color.shape == (3, H, W),   f"color shape {color.shape}"
        assert radii.shape == (N,),        f"radii shape {radii.shape}"
        assert depth.shape == (1, H, W),   f"depth shape {depth.shape}"
        assert alpha.shape == (1, H, W),   f"alpha shape {alpha.shape}"
        assert flow.shape  == (2, H, W),   f"flow shape {flow.shape}"
        assert covs.shape  == (N, 6),      f"covs shape {covs.shape}"

    def test_zero_gaussians(self):
        H, W = 16, 16
        rs = _make_raster_settings(H, W)
        rz = StochasticGaussianRasterizer(rs, num_samples=4)

        means3D   = torch.zeros(0, 3)
        means2D   = torch.zeros(0, 3, requires_grad=True)
        scales    = torch.zeros(0, 3)
        rotations = torch.zeros(0, 4)
        opacities = torch.zeros(0, 1)
        colors    = torch.zeros(0, 3)

        out = rz(means3D=means3D, means2D=means2D,
                 opacities=opacities, colors_precomp=colors,
                 scales=scales, rotations=rotations)
        color, radii, depth, alpha, flow, covs = out

        assert color.shape == (3, H, W)
        assert radii.shape == (0,)


class TestDeterministicConsistency:
    """num_samples <= 0 -> deterministic -> two calls give identical results."""

    def test_deterministic_reproducibility(self):
        rs  = _make_raster_settings()
        rz  = StochasticGaussianRasterizer(rs, num_samples=-1)
        means3D, means2D, scales, rots, opacities, colors = _make_gaussians()

        kwargs = dict(means3D=means3D,
                      means2D=means2D.detach().clone().requires_grad_(True),
                      opacities=opacities, colors_precomp=colors,
                      scales=scales, rotations=rots)

        color1 = rz(**kwargs)[0].detach()

        kwargs2 = dict(means3D=means3D,
                       means2D=means2D.detach().clone().requires_grad_(True),
                       opacities=opacities, colors_precomp=colors,
                       scales=scales, rotations=rots)
        color2 = rz(**kwargs2)[0].detach()
        assert torch.allclose(color1, color2, atol=0.0), \
            "Deterministic render not reproducible"


class TestStochasticUnbiasedness:
    """Mean of many stochastic renders should converge to deterministic."""

    def test_stochastic_converges_to_deterministic(self):
        torch.manual_seed(0)
        rs       = _make_raster_settings(H=16, W=16)
        rz_det   = StochasticGaussianRasterizer(rs, num_samples=-1)
        rz_stoch = StochasticGaussianRasterizer(rs, num_samples=8)

        means3D, _, scales, rots, opacities, colors = _make_gaussians(N=10, seed=7)

        kwargs = dict(
            means3D=means3D,
            means2D=torch.zeros(10, 3, requires_grad=False),
            opacities=opacities,
            colors_precomp=colors,
            scales=scales,
            rotations=rots,
        )

        # Deterministic baseline
        det_color = rz_det(**kwargs)[0].detach()

        # Average 200 stochastic renders
        n_trials = 200
        stoch_sum = torch.zeros_like(det_color)
        for _ in range(n_trials):
            stoch_sum = stoch_sum + rz_stoch(**kwargs)[0].detach()
        stoch_mean = stoch_sum / n_trials

        # Mean absolute error should be small
        mae = (det_color - stoch_mean).abs().mean().item()
        assert mae < 0.05, (
            f"Stochastic mean (MAE={mae:.4f}) diverges from deterministic")


class TestGradientFlow:
    """Loss gradients must reach Gaussian parameters."""

    def test_color_gradients(self):
        rs  = _make_raster_settings(H=16, W=16)
        rz  = StochasticGaussianRasterizer(rs, num_samples=-1)

        means3D   = torch.randn(10, 3) * 0.5
        means3D[:, 2] += 2.0
        means2D   = torch.zeros(10, 3, requires_grad=True)
        scales    = torch.ones(10, 3) * 0.3
        rotations = torch.zeros(10, 4); rotations[:, 0] = 1.0
        opacities = torch.ones(10, 1) * 0.5
        colors    = torch.rand(10, 3, requires_grad=True)

        color, *_ = rz(means3D=means3D, means2D=means2D,
                       opacities=opacities, colors_precomp=colors,
                       scales=scales, rotations=rotations)
        loss = color.sum()
        loss.backward()

        assert colors.grad is not None, "No gradient reached colors"
        assert colors.grad.abs().sum() > 0, "Color gradient is all-zeros"

    def test_opacity_gradients(self):
        rs  = _make_raster_settings(H=16, W=16)
        rz  = StochasticGaussianRasterizer(rs, num_samples=-1)

        means3D   = torch.randn(10, 3) * 0.5
        means3D[:, 2] += 2.0
        means2D   = torch.zeros(10, 3, requires_grad=True)
        scales    = torch.ones(10, 3) * 0.3
        rotations = torch.zeros(10, 4); rotations[:, 0] = 1.0
        opacities = torch.ones(10, 1, requires_grad=True) * 0.5
        colors    = torch.rand(10, 3)

        color, *_ = rz(means3D=means3D, means2D=means2D,
                       opacities=opacities, colors_precomp=colors,
                       scales=scales, rotations=rotations)
        loss = color.sum()
        loss.backward()

        assert opacities.grad is not None, "No gradient reached opacities"

    def test_screenspace_grad(self):
        """screenspace_points.grad must be set after backward."""
        rs  = _make_raster_settings(H=16, W=16)
        rz  = StochasticGaussianRasterizer(rs, num_samples=-1)

        means3D   = torch.randn(10, 3) * 0.5
        means3D[:, 2] += 2.0
        means2D   = torch.zeros(10, 3, requires_grad=True)
        means2D.retain_grad()

        scales    = torch.ones(10, 3) * 0.3
        rotations = torch.zeros(10, 4); rotations[:, 0] = 1.0
        opacities = torch.ones(10, 1) * 0.5
        colors    = torch.rand(10, 3)

        color, *_ = rz(means3D=means3D, means2D=means2D,
                       opacities=opacities, colors_precomp=colors,
                       scales=scales, rotations=rotations)
        loss = color.sum()
        loss.backward()

        assert means2D.grad is not None, "screenspace_points.grad is None"


class TestProjectionHelper:
    """Unit-test the internal projection helper independently."""

    def test_projection_depths_positive(self):
        """Gaussians placed in front of the camera should have positive depth."""
        rs = _make_raster_settings(H=64, W=64)
        N = 5
        means3D = torch.zeros(N, 3)
        means3D[:, 2] = torch.linspace(1.0, 5.0, N)

        cov3D = torch.zeros(N, 6)
        cov3D[:, 0] = 0.1; cov3D[:, 3] = 0.1; cov3D[:, 5] = 0.1

        _, _, depths, valid, _ = _project_gaussians(
            means3D, cov3D,
            rs.viewmatrix, rs.projmatrix,
            rs.image_height, rs.image_width,
            rs.tanfovx, rs.tanfovy,
        )
        assert valid.all(), "All front-facing Gaussians should be valid"
        assert (depths > 0).all(), "All depths should be positive"

    def test_cov3d_builder(self):
        """_build_cov3D should produce a symmetric matrix with positive diagonal."""
        N = 4
        scales    = torch.ones(N, 3) * 0.5
        rotations = torch.zeros(N, 4); rotations[:, 0] = 1.0
        cov = _build_cov3D(scales, 1.0, rotations)
        assert cov.shape == (N, 6)
        # Diagonal elements must be positive
        assert (cov[:, 0] > 0).all() and (cov[:, 3] > 0).all() \
               and (cov[:, 5] > 0).all()


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test without pytest
    t = TestOutputShapes()
    t.test_output_shapes()
    print("OK output shapes")

    t.test_zero_gaussians()
    print("OK zero gaussians")

    d = TestDeterministicConsistency()
    d.test_deterministic_reproducibility()
    print("OK deterministic reproducibility")

    s = TestStochasticUnbiasedness()
    s.test_stochastic_converges_to_deterministic()
    print("OK stochastic unbiasedness")

    g = TestGradientFlow()
    g.test_color_gradients()
    print("OK colour gradients")

    g.test_opacity_gradients()
    print("OK opacity gradients")

    g.test_screenspace_grad()
    print("OK screenspace grad")

    p = TestProjectionHelper()
    p.test_projection_depths_positive()
    print("OK projection depths")

    p.test_cov3d_builder()
    print("OK cov3D builder")

    print("\nAll tests passed!")

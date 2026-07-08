import math
import time
from typing import Any, List, Set, Tuple

import glfw
import numpy as np
import torch
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPixelStorei,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glViewport,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds

from scene.cameras import Camera as _GaussianViewerCamera


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)


def _camera_rt(
    eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    eye_b = eye.view(1, 3)
    at_b = at.view(1, 3)
    up_b = up.view(1, 3)
    return look_at_view_transform(eye=eye_b, at=at_b, up=up_b)


def _normalize_input_views(
    input_views: List[Tuple[torch.Tensor, torch.Tensor, float, float]] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor, float, float]]:
    normalized_input_views: List[Tuple[torch.Tensor, torch.Tensor, float, float]] = []
    if input_views is None:
        return normalized_input_views

    for i, (R_raw, T_raw, fov_x_raw, fov_y_raw) in enumerate(input_views):
        R_view = torch.as_tensor(R_raw, device=device, dtype=dtype)
        T_view = torch.as_tensor(T_raw, device=device, dtype=dtype)
        if R_view.shape == (1, 3, 3):
            R_view = R_view[0]
        if T_view.shape == (1, 3):
            T_view = T_view[0]
        if R_view.shape != (3, 3):
            raise ValueError(
                f"input_views[{i}][0] must have shape [3, 3] or [1, 3, 3]."
            )
        if T_view.shape != (3,):
            raise ValueError(f"input_views[{i}][1] must have shape [3] or [1, 3].")
        fov_x_view = float(fov_x_raw)
        fov_y_view = float(fov_y_raw)
        if fov_x_view <= 0.0 or fov_x_view >= float(torch.pi):
            raise ValueError(f"input_views[{i}][2] fov_x must be in (0, pi) radians.")
        if fov_y_view <= 0.0 or fov_y_view >= float(torch.pi):
            raise ValueError(f"input_views[{i}][3] fov_y must be in (0, pi) radians.")
        normalized_input_views.append(
            (
                R_view.contiguous().view(1, 3, 3),
                T_view.contiguous().view(1, 3),
                fov_x_view,
                fov_y_view,
            )
        )
    return normalized_input_views


class TemporalPointCloudStore:
    def __init__(
        self,
        xyz: torch.Tensor,
        times: torch.Tensor,
        duration: torch.Tensor,
        color: torch.Tensor,
        camera_indices: torch.Tensor,
        max_render_points: int = 1_000_000,
    ) -> None:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3].")
        if color.ndim != 2 or color.shape[-1] != 3:
            raise ValueError("color must have shape [N, 3].")
        if times.ndim != 1:
            raise ValueError("times must have shape [N].")
        if duration.ndim != 1:
            raise ValueError("duration must have shape [N].")
        if camera_indices.ndim != 1:
            raise ValueError("camera_indices must have shape [N].")

        if xyz.shape[0] != color.shape[0]:
            raise ValueError("xyz and color must have the same number of points.")
        if times.shape[0] != xyz.shape[0] or duration.shape[0] != xyz.shape[0]:
            raise ValueError(
                "times and duration must have the same number of entries as xyz."
            )
        if camera_indices.shape[0] != xyz.shape[0]:
            raise ValueError(
                "camera_indices must have the same number of entries as xyz."
            )

        self.device = xyz.device
        self.dtype = xyz.dtype

        self.xyz = xyz.to(device=self.device, dtype=self.dtype).contiguous()
        self.color = color.to(device=self.device, dtype=self.dtype).contiguous()
        self.times = times.to(device=self.device, dtype=self.dtype).contiguous()
        self.duration = duration.to(device=self.device, dtype=self.dtype).contiguous()
        self.camera_indices = camera_indices.to(
            device=self.device, dtype=torch.int64
        ).contiguous()

        if float(self.color.max().detach().cpu().item()) > 1.0:
            self.color = self.color / 255.0
        self.color = self.color.clamp(0.0, 1.0)

        if max_render_points > 0 and self.xyz.shape[0] > max_render_points:
            keep = torch.randperm(self.xyz.shape[0], device=self.device)[
                :max_render_points
            ]
            self.xyz = self.xyz.index_select(0, keep)
            self.color = self.color.index_select(0, keep)
            self.times = self.times.index_select(0, keep)
            self.duration = self.duration.index_select(0, keep)
            self.camera_indices = self.camera_indices.index_select(0, keep)

        self.enabled_camera_indices = set(range(10))
        self.indexed_point_masks = {
            idx: self.camera_indices == idx for idx in range(10)
        }

    @property
    def cloud_center(self) -> torch.Tensor:
        return self.xyz.mean(dim=0)

    def enabled_text(self) -> str:
        return "".join(
            str(idx) if idx in self.enabled_camera_indices else "-" for idx in range(10)
        )

    def toggle_camera_index(self, idx: int) -> None:
        if idx in self.enabled_camera_indices:
            self.enabled_camera_indices.remove(idx)
        else:
            self.enabled_camera_indices.add(idx)

    def build_visible_mask(self, current_time: torch.Tensor) -> torch.Tensor:
        visible_mask = torch.ones(
            self.camera_indices.shape[0], dtype=torch.bool, device=self.device
        )
        for idx, idx_mask in self.indexed_point_masks.items():
            if idx not in self.enabled_camera_indices:
                visible_mask = visible_mask & (~idx_mask)

        t = current_time
        temporal_mask = (self.times <= t) & (t <= (self.times + self.duration))
        return visible_mask & temporal_mask

    def build_visible_point_cloud(
        self, current_time: torch.Tensor
    ) -> Pointclouds | None:
        visible_mask = self.build_visible_mask(current_time)
        if not bool(visible_mask.any().item()):
            return None
        print(
            f"Rendering {visible_mask.sum().item()} points out of {self.xyz.shape[0]} "
            f"total points at t={float(current_time.item()):.3f}."
        )
        return Pointclouds(
            points=[self.xyz[visible_mask]],
            features=[self.color[visible_mask]],
        )


class PyTorch3DPointCloudRenderer:
    def __init__(
        self,
        store: TemporalPointCloudStore,
        point_radius: float = 0.001,
        points_per_pixel: int = 8,
    ) -> None:
        self.store = store
        self.device = store.device
        self.point_radius = point_radius
        self.points_per_pixel = points_per_pixel

    def render(
        self,
        fb_w: int,
        fb_h: int,
        R: torch.Tensor,
        T: torch.Tensor,
        fov_x: float,
        fov_y: float,
        current_time: torch.Tensor,
    ) -> Tuple[np.ndarray, int, int]:
        point_cloud = self.store.build_visible_point_cloud(current_time)
        if point_cloud is None:
            return np.zeros((fb_h, fb_w, 3), dtype=np.uint8), fb_w, fb_h

        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=float(math.degrees(fov_y)),
            znear=0.5,
        )
        raster = PointsRasterizationSettings(
            image_size=(fb_h, fb_w),
            radius=self.point_radius,
            points_per_pixel=self.points_per_pixel,
            max_points_per_bin=1000,
        )
        renderer = PulsarPointsRenderer(
            rasterizer=PointsRasterizer(raster_settings=raster, cameras=cameras),
        ).to(self.device)

        with torch.no_grad():
            image = renderer(point_cloud, gamma=(1e-4,))[0, ..., :3]
        image_u8 = (
            (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy()
        )
        return np.ascontiguousarray(image_u8), fb_w, fb_h



class GaussianSplatRenderer:
    def __init__(
        self,
        gaussians: Any,
        pipe: Any,
        bg_color: torch.Tensor,
        tensor_gradient_2d_buffer: torch.Tensor | None = None,
        render_width: int | None = None,
        render_height: int | None = None,
    ) -> None:
        from gaussian_renderer import render as gaussian_render

        self.gaussians = gaussians
        self.pipe = pipe
        self.gaussian_render = gaussian_render

        self.device = gaussians.get_xyz.device
        self.dtype = gaussians.get_xyz.dtype
        self.bg_color = bg_color.to(device=self.device, dtype=self.dtype)

        if tensor_gradient_2d_buffer is None:
            tensor_gradient_2d_buffer = torch.zeros_like(
                gaussians.get_xyz,
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
        self.tensor_gradient_2d_buffer = tensor_gradient_2d_buffer.to(
            device=self.device, dtype=self.dtype
        )

        if render_width is not None and render_width <= 0:
            raise ValueError("render_width must be > 0 when provided.")
        if render_height is not None and render_height <= 0:
            raise ValueError("render_height must be > 0 when provided.")
        self.fixed_render_width = (
            int(render_width) if render_width is not None else None
        )
        self.fixed_render_height = (
            int(render_height) if render_height is not None else None
        )

    def render(
        self,
        fb_w: int,
        fb_h: int,
        R: torch.Tensor,
        T: torch.Tensor,
        fov_x: float,
        fov_y: float,
        current_time: torch.Tensor,
    ) -> Tuple[np.ndarray, int, int]:
        if getattr(self.pipe, "env_map_res", False):
            raise ValueError(
                "GaussianSplatRenderer viewer does not support env_map_res yet."
            )

        target_w = (
            self.fixed_render_width
            if self.fixed_render_width is not None
            else int(fb_w)
        )
        target_h = (
            self.fixed_render_height
            if self.fixed_render_height is not None
            else int(fb_h)
        )

        camera = _GaussianViewerCamera(
            -1,
            R=R.detach().cpu().squeeze().transpose(0, 1).numpy(),
            T=T.detach().cpu().squeeze().numpy(),
            FoVx=float(fov_x),
            FoVy=float(fov_y),
            resolution=(target_w, target_h),
            timestamp=float(current_time.item()),
            image=torch.zeros((3, target_h, target_w), device=self.device, dtype=self.dtype),
            gt_alpha_mask= None,
            image_name="",
            uid = -1
        )

        camera= camera.cuda()

        with torch.no_grad():
            render_output = self.gaussian_render(
                camera,
                self.gaussians,
                self.pipe,
                self.bg_color,
                self.tensor_gradient_2d_buffer,
            )
        image = render_output["render"]
        if image.ndim != 3:
            raise ValueError(
                "gaussian_renderer.render output['render'] must have shape [C, H, W]."
            )
        image_hwc = image[:3].permute(1, 2, 0)
        image_u8 = (
            (image_hwc.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy()
        )
        return np.ascontiguousarray(image_u8), target_w, target_h


def _show_glfw_with_renderer(
    renderer_backend: Any,
    device: torch.device,
    dtype: torch.dtype,
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor,
    fov_x: float,
    fov_y: float,
    title: str,
    window_size: Tuple[int, int],
    time_step: float,
    input_views: List[Tuple[torch.Tensor, torch.Tensor, float, float]] | None,
    playback_fps: float,
    point_store: TemporalPointCloudStore | None = None,
) -> None:
    if camera_position.numel() != 3 or look_at.numel() != 3 or up.numel() != 3:
        raise ValueError(
            "camera_position, look_at, and up must each contain exactly 3 values."
        )
    if fov_x <= 0.0 or fov_x >= float(torch.pi):
        raise ValueError("fov_x must be in (0, pi) radians.")
    if fov_y <= 0.0 or fov_y >= float(torch.pi):
        raise ValueError("fov_y must be in (0, pi) radians.")
    if time_step <= 0.0:
        raise ValueError("time_step must be > 0.")
    if playback_fps <= 0.0:
        raise ValueError("playback_fps must be > 0.")

    cam_pos = camera_position.to(device=device, dtype=dtype).view(3).clone()
    at = look_at.to(device=device, dtype=dtype).view(3).clone()
    base_world_up = _normalize(up.to(device=device, dtype=dtype).view(3).clone())
    current_time = torch.tensor(0.0, device=device, dtype=dtype)

    if float(torch.linalg.norm(at - cam_pos).detach().cpu().item()) < 1e-8:
        raise ValueError("camera_position and look_at cannot be the same.")

    def _build_c2w_rotation(
        forward_hint: torch.Tensor, up_hint: torch.Tensor
    ) -> torch.Tensor:
        forward_dir = _normalize(forward_hint)
        # Build a right-handed camera frame: right = up x forward.
        right_dir = torch.cross(up_hint, forward_dir, dim=0)
        if float(torch.linalg.norm(right_dir).item()) < 1e-8:
            fallback_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            if float(torch.abs(torch.dot(forward_dir, fallback_up)).item()) > 0.99:
                fallback_up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            right_dir = torch.cross(fallback_up, forward_dir, dim=0)
        right_dir = _normalize(right_dir)
        up_dir = _normalize(torch.cross(forward_dir, right_dir, dim=0))
        return torch.stack([right_dir, up_dir, forward_dir], dim=1)

    def _rotation_matrix(axis: torch.Tensor, angle: float) -> torch.Tensor:
        axis_n = _normalize(axis)
        kx, ky, kz = axis_n[0], axis_n[1], axis_n[2]
        K = torch.tensor(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        I = torch.eye(3, device=device, dtype=dtype)
        s = float(np.sin(angle))
        c = float(np.cos(angle))
        return I + s * K + (1.0 - c) * (K @ K)

    cam_rot_c2w = _build_c2w_rotation(at - cam_pos, base_world_up)

    normalized_input_views = _normalize_input_views(
        input_views, device=device, dtype=dtype
    )
    has_input_views = len(normalized_input_views) > 0
    is_playing_input_views = False
    has_started_input_views = False
    input_view_index = 0
    playback_start_time = 0.0

    startup_window_w = int(window_size[0])
    startup_window_h = int(window_size[1])
    fixed_w = getattr(renderer_backend, "fixed_render_width", None)
    fixed_h = getattr(renderer_backend, "fixed_render_height", None)
    if fixed_w is not None and fixed_h is not None:
        startup_window_w = int(fixed_w)
        startup_window_h = int(fixed_h)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    window = glfw.create_window(startup_window_w, startup_window_h, title, None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    def _sync_manual_pose_from_input_view(view_index: int) -> None:
        nonlocal cam_pos, cam_rot_c2w, fov_x, fov_y
        if not has_input_views:
            return
        R_view, T_view, fov_x_view, fov_y_view = normalized_input_views[view_index]
        R3 = R_view[0]
        T3 = T_view[0]
        cam_rot_c2w = R3.transpose(0, 1).contiguous()
        cam_pos = -(cam_rot_c2w @ T3)
        fov_x = fov_x_view
        fov_y = fov_y_view

    def _rotate_camera_local(yaw_delta: float, pitch_delta: float) -> None:
        nonlocal cam_rot_c2w
        changed = False
        if abs(yaw_delta) > 0.0:
            up_axis = cam_rot_c2w[:, 1]
            cam_rot_c2w = _rotation_matrix(up_axis, yaw_delta) @ cam_rot_c2w
            changed = True
        if abs(pitch_delta) > 0.0:
            right_axis = cam_rot_c2w[:, 0]
            cam_rot_c2w = _rotation_matrix(right_axis, pitch_delta) @ cam_rot_c2w
            changed = True
        if changed:
            cam_rot_c2w = _build_c2w_rotation(cam_rot_c2w[:, 2], cam_rot_c2w[:, 1])

    def _update_window_title() -> None:
        parts = [f"{title} | t={float(current_time.item()):.3f}"]
        if point_store is not None:
            parts.append(f"cameras: {point_store.enabled_text()}")
        if has_input_views:
            state = "playing" if is_playing_input_views else "paused"
            if not has_started_input_views:
                state = "ready"
            parts.append(
                f"views: {state} {input_view_index + 1}/{len(normalized_input_views)}"
            )
        glfw.set_window_title(window, " | ".join(parts))

    _update_window_title()

    tex_id = int(glGenTextures(1))
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, 0)
    glClearColor(0.02, 0.02, 0.02, 1.0)

    keys_down: Set[int] = set()
    dragging = False
    last_mouse = (0.0, 0.0)
    last_time = time.perf_counter()
    scene_dirty = True
    image_u8 = np.zeros(
        (max(1, startup_window_h), max(1, startup_window_w), 3), dtype=np.uint8
    )
    last_fb_size = (0, 0)

    move_speed = 5.0
    slow_move_factor = 0.25
    mouse_sensitivity = 0.003
    key_rotate_speed = 1.6
    scroll_speed = 0.5

    def _mouse_button_cb(_window: Any, button: int, action: int, _mods: int) -> None:
        nonlocal dragging, last_mouse, scene_dirty
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging = True
                last_mouse = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                dragging = False
            scene_dirty = True

    def _cursor_pos_cb(_window: Any, xpos: float, ypos: float) -> None:
        nonlocal last_mouse, scene_dirty
        if not dragging:
            return
        last_x, last_y = last_mouse
        dx = float(xpos - last_x)
        dy = float(ypos - last_y)
        last_mouse = (xpos, ypos)

        _rotate_camera_local(dx * mouse_sensitivity, -dy * mouse_sensitivity)
        scene_dirty = True

    def _scroll_cb(_window: Any, _xoffset: float, yoffset: float) -> None:
        nonlocal cam_pos, scene_dirty
        if yoffset == 0.0:
            return
        direction = cam_rot_c2w[:, 2]
        cam_pos = cam_pos + direction * scroll_speed * float(np.sign(yoffset))
        scene_dirty = True

    def _key_cb(
        _window: Any, key: int, _scancode: int, action: int, _mods: int
    ) -> None:
        nonlocal scene_dirty, cam_pos, cam_rot_c2w, current_time
        nonlocal is_playing_input_views, has_started_input_views, playback_start_time
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if key == glfw.KEY_P and action == glfw.PRESS and has_input_views:
            if is_playing_input_views:
                is_playing_input_views = False
                _sync_manual_pose_from_input_view(input_view_index)
            else:
                is_playing_input_views = True
                has_started_input_views = True
                playback_start_time = time.perf_counter() - (
                    input_view_index / playback_fps
                )
            _update_window_title()
            scene_dirty = True
        if key == glfw.KEY_H and action == glfw.PRESS:
            cloud_center = (
                point_store.cloud_center
                if point_store is not None
                else torch.zeros(3, device=device, dtype=dtype)
            )
            cam_pos = cloud_center + torch.tensor(
                [0.0, 0.0, 5.0], device=device, dtype=dtype
            )
            new_forward = _normalize(cloud_center - cam_pos)
            cam_rot_c2w = _build_c2w_rotation(new_forward, base_world_up)
            scene_dirty = True
        if key == glfw.KEY_RIGHT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
            current_time = current_time + time_step
            _update_window_title()
            scene_dirty = True
        if key == glfw.KEY_LEFT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
            current_time = torch.clamp(current_time - time_step, min=0.0)
            _update_window_title()
            scene_dirty = True
        if (
            point_store is not None
            and action == glfw.PRESS
            and glfw.KEY_0 <= key <= glfw.KEY_9
        ):
            point_store.toggle_camera_index(key - glfw.KEY_0)
            _update_window_title()
            scene_dirty = True
        if action in (glfw.PRESS, glfw.RELEASE):
            scene_dirty = True

    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window, _cursor_pos_cb)
    glfw.set_scroll_callback(window, _scroll_cb)
    glfw.set_key_callback(window, _key_cb)

    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            dt = max(1e-4, now - last_time)
            last_time = now

            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)

            for key in (
                glfw.KEY_W,
                glfw.KEY_S,
                glfw.KEY_A,
                glfw.KEY_D,
                glfw.KEY_Q,
                glfw.KEY_E,
                glfw.KEY_LEFT,
                glfw.KEY_RIGHT,
                glfw.KEY_UP,
                glfw.KEY_DOWN,
            ):
                if glfw.get_key(window, key) == glfw.PRESS:
                    keys_down.add(key)
                else:
                    keys_down.discard(key)

            yaw_delta = 0.0
            pitch_delta = 0.0
            if glfw.KEY_LEFT in keys_down:
                yaw_delta -= key_rotate_speed * dt
            if glfw.KEY_RIGHT in keys_down:
                yaw_delta += key_rotate_speed * dt
            if glfw.KEY_UP in keys_down:
                pitch_delta += key_rotate_speed * dt
            if glfw.KEY_DOWN in keys_down:
                pitch_delta -= key_rotate_speed * dt
            if yaw_delta != 0.0 or pitch_delta != 0.0:
                _rotate_camera_local(yaw_delta, pitch_delta)
                scene_dirty = True

            right_dir = cam_rot_c2w[:, 0]
            up_dir = cam_rot_c2w[:, 1]
            forward_dir = cam_rot_c2w[:, 2]

            move = torch.zeros(3, device=device, dtype=dtype)
            if glfw.KEY_W in keys_down:
                move = move + forward_dir
            if glfw.KEY_S in keys_down:
                move = move - forward_dir
            if glfw.KEY_A in keys_down:
                move = move - right_dir
            if glfw.KEY_D in keys_down:
                move = move + right_dir
            if glfw.KEY_Q in keys_down:
                move = move - up_dir
            if glfw.KEY_E in keys_down:
                move = move + up_dir

            mag = torch.linalg.norm(move)
            if float(mag.item()) > 0.0:
                move = move / mag
                is_shift_down = (
                    glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                    or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
                )
                speed_scale = slow_move_factor if is_shift_down else 1.0
                cam_pos = cam_pos + move * move_speed * speed_scale * dt
                scene_dirty = True

            fb_w, fb_h = glfw.get_framebuffer_size(window)
            fb_w = max(1, int(fb_w))
            fb_h = max(1, int(fb_h))
            if (fb_w, fb_h) != last_fb_size:
                last_fb_size = (fb_w, fb_h)
                scene_dirty = True

            if has_input_views and is_playing_input_views:
                if is_playing_input_views:
                    next_index = int((now - playback_start_time) * playback_fps) % len(
                        normalized_input_views
                    )
                    if next_index != input_view_index:
                        input_view_index = next_index
                        _update_window_title()
                        scene_dirty = True
                R, T, current_fov_x, current_fov_y = normalized_input_views[input_view_index]
            else:
                R = cam_rot_c2w.transpose(0, 1).contiguous().view(1, 3, 3)
                T = (-(R[0] @ cam_pos)).contiguous().view(1, 3)
                current_fov_x = float(fov_x)
                current_fov_y = float(fov_y)

            if scene_dirty:
                image_u8, tex_w, tex_h = renderer_backend.render(
                    fb_w=fb_w,
                    fb_h=fb_h,
                    R=R,
                    T=T,
                    fov_x=current_fov_x,
                    fov_y=current_fov_y,
                    current_time=current_time,
                )

            glViewport(0, 0, fb_w, fb_h)
            glClear(GL_COLOR_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            if scene_dirty:
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    tex_w,
                    tex_h,
                    0,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    image_u8,
                )
                scene_dirty = False

            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            # Flip vertically because OpenGL texture origin is bottom-left.
            glTexCoord2f(0.0, 1.0)
            glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 1.0)
            glVertex2f(1.0, -1.0)
            glTexCoord2f(1.0, 0.0)
            glVertex2f(1.0, 1.0)
            glTexCoord2f(0.0, 0.0)
            glVertex2f(-1.0, 1.0)
            glEnd()

            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)
            glfw.swap_buffers(window)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def show_pointcloud_glfw_pytorch3d(
    xyz: torch.Tensor,
    times: torch.Tensor,
    duration: torch.Tensor,
    color: torch.Tensor,
    camera_indices: torch.Tensor,
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor,
    fov_x: float,
    fov_y: float,
    window_size: Tuple[int, int] = (1280, 720),
    title: str = "PyTorch3D Point Cloud Viewer",
    point_radius: float = 0.001,
    points_per_pixel: int = 8,
    max_render_points: int = 1_000_000,
    time_step: float = 0.05,
    input_views: List[Tuple[torch.Tensor, torch.Tensor, float, float]] | None = None,
    playback_fps: float = 24.0,
) -> None:
    """
    Open a blocking GLFW + OpenGL window and interactively render a point cloud
    using PyTorch3D.

    Controls:
    - 0-9: toggle visibility for points with camera index 0-9
    - ] / [: move forward/backward in time
    - P: play/pause optional input views sequence
    - H: look at cloud center from +Z, 5 units away
    - W/S: move forward/back
    - A/D: strafe left/right
    - Q/E: move down/up
    - Hold Shift: move slower
    - Left mouse drag: rotate view (yaw/pitch)
    - Arrow keys: rotate view (yaw/pitch)
    - Mouse wheel: move forward/backward
    - Esc: close window
    """
    store = TemporalPointCloudStore(
        xyz=xyz,
        times=times,
        duration=duration,
        color=color,
        camera_indices=camera_indices,
        max_render_points=max_render_points,
    )
    backend = PyTorch3DPointCloudRenderer(
        store=store,
        point_radius=point_radius,
        points_per_pixel=points_per_pixel,
    )
    _show_glfw_with_renderer(
        renderer_backend=backend,
        device=store.device,
        dtype=store.dtype,
        camera_position=camera_position,
        look_at=look_at,
        up=up,
        fov_x=fov_x,
        fov_y=fov_y,
        title=title,
        window_size=window_size,
        time_step=time_step,
        input_views=input_views,
        playback_fps=playback_fps,
        point_store=store,
    )


def show_gaussians_glfw(
    gaussians: Any,
    pipe: Any,
    bg_color: torch.Tensor,
    tensor_gradient_2d_buffer: torch.Tensor | None,
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor,
    fov_x: float,
    fov_y: float,
    window_size: Tuple[int, int] = (1280, 720),
    title: str = "Gaussian Splat Viewer",
    time_step: float = 0.05,
    input_views: List[Tuple[torch.Tensor, torch.Tensor, float, float]] | None = None,
    playback_fps: float = 24.0,
    render_width: int | None = None,
    render_height: int | None = None,
) -> None:
    """
    Open a blocking GLFW + OpenGL window and interactively render gaussians
    using gaussian_renderer.render.

    Controls:
    - ] / [: move forward/backward in time (timestamp)
    - P: play/pause optional input views sequence
    - H: reset orbit camera
    - W/S/A/D/Q/E + mouse: move camera
    - Esc: close window
    """
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype

    backend = GaussianSplatRenderer(
        gaussians=gaussians,
        pipe=pipe,
        bg_color=bg_color,
        tensor_gradient_2d_buffer=tensor_gradient_2d_buffer,
        render_width=render_width,
        render_height=render_height,
    )
    _show_glfw_with_renderer(
        renderer_backend=backend,
        device=device,
        dtype=dtype,
        camera_position=camera_position,
        look_at=look_at,
        up=up,
        fov_x=fov_x,
        fov_y=fov_y,
        title=title,
        window_size=window_size,
        time_step=time_step,
        input_views=input_views,
        playback_fps=playback_fps,
        point_store=None,
    )


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_theta = 1000
    n_phi = 1000
    theta = torch.linspace(0.0, 2.0 * np.pi, n_theta, device=test_device)
    phi = torch.linspace(0.0, np.pi, n_phi, device=test_device)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    r = 1.0 + 0.15 * torch.sin(4.0 * theta_grid) * torch.sin(3.0 * phi_grid)
    x = r * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = r * torch.cos(phi_grid)
    z = r * torch.sin(phi_grid) * torch.sin(theta_grid)
    xyz_test = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(torch.float32)

    xyz_min = xyz_test.min(dim=0).values
    xyz_max = xyz_test.max(dim=0).values
    color_test = (xyz_test - xyz_min) / (xyz_max - xyz_min + 1e-8)

    camera_position_test = torch.tensor(
        [0.0, 0.0, 3.5], dtype=torch.float32, device=test_device
    )
    look_at_test = torch.tensor(
        [0.0, 0.0, 0.0], dtype=torch.float32, device=test_device
    )
    up_test = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=test_device)

    show_pointcloud_glfw_pytorch3d(
        xyz=xyz_test,
        color=color_test,
        times=torch.zeros(xyz_test.shape[0], device=test_device),
        duration=torch.ones(xyz_test.shape[0], device=test_device),
        camera_indices=torch.arange(xyz_test.shape[0], device=test_device) % 10,
        camera_position=camera_position_test,
        look_at=look_at_test,
        up=up_test,
        fov_x=float(torch.deg2rad(torch.tensor(60.0)).item()),
        fov_y=float(torch.deg2rad(torch.tensor(60.0)).item()),
        window_size=(1280, 720),
        title="PyTorch3D GLFW Point Cloud Viewer - Smoke Test",
    )

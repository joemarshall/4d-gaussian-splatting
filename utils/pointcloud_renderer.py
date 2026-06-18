import time
from typing import Set, Tuple

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
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)


def _camera_rt(eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eye_b = eye.view(1, 3)
    at_b = at.view(1, 3)
    up_b = up.view(1, 3)
    return look_at_view_transform(eye=eye_b, at=at_b, up=up_b)


def show_pointcloud_glfw_pytorch3d(
    xyz: torch.Tensor,
    color: torch.Tensor,
    camera_indices: torch.Tensor,
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor,
    fov_degrees: float,
    window_size: Tuple[int, int] = (1280, 720),
    title: str = "PyTorch3D Point Cloud Viewer",
    point_radius: float = 0.001,
#    point_radius: float = 0.006,
    points_per_pixel: int = 8,
    max_render_points: int = 1_000_000,
) -> None:
    """
    Open a blocking GLFW + OpenGL window and interactively render a point cloud
    using PyTorch3D.

    Controls:
    - 0-9: toggle visibility for points with camera index 0-9
    - W/S: move forward/back
    - A/D: strafe left/right
    - Q/E: move down/up
    - Hold Shift: move slower
    - Left mouse drag: rotate view (yaw/pitch)
    - Arrow keys: rotate view (yaw/pitch)
    - Mouse wheel: move forward/backward
    - Esc: close window

    The function blocks until the window is closed, then returns.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N, 3].")
    if color.ndim != 2 or color.shape[-1] != 3:
        raise ValueError("color must have shape [N, 3].")
    if xyz.shape[0] != color.shape[0]:
        raise ValueError("xyz and color must have the same number of points.")
    if camera_indices.ndim != 1:
        raise ValueError("camera_indices must have shape [N].")
    if camera_indices.shape[0] != xyz.shape[0]:
        raise ValueError("camera_indices must have the same number of entries as xyz.")
    if camera_position.numel() != 3 or look_at.numel() != 3 or up.numel() != 3:
        raise ValueError("camera_position, look_at, and up must each contain exactly 3 values.")
    if fov_degrees <= 0.0 or fov_degrees >= 179.0:
        raise ValueError("fov_degrees must be in (0, 179).")

    device = xyz.device
    dtype = xyz.dtype

    xyz = xyz.to(device=device, dtype=dtype).contiguous()
    color = color.to(device=device, dtype=dtype).contiguous()
    camera_indices = camera_indices.to(device=device, dtype=torch.int64).contiguous()

    if float(color.max().detach().cpu().item()) > 1.0:
        color = color / 255.0
    color = color.clamp(0.0, 1.0)

    eye = camera_position.to(device=device, dtype=dtype).view(3).clone()
    at = look_at.to(device=device, dtype=dtype).view(3).clone()
    world_up = _normalize(up.to(device=device, dtype=dtype).view(3).clone())

    if float(torch.linalg.norm(at - eye).detach().cpu().item()) < 1e-8:
        raise ValueError("camera_position and look_at cannot be the same.")

    forward = _normalize(at - eye)
    yaw = float(torch.atan2(forward[0], forward[2]).item())
    pitch = float(torch.asin(torch.clamp(forward[1], -0.999, 0.999)).item())



    if max_render_points > 0 and xyz.shape[0] > max_render_points:
        keep = torch.randperm(xyz.shape[0], device=device)[:max_render_points]
        xyz = xyz.index_select(0, keep)
        color = color.index_select(0, keep)
        camera_indices = camera_indices.index_select(0, keep)
        print(keep[0:20])

    enabled_camera_indices = set(range(10))
    indexed_point_masks = {idx: camera_indices == idx for idx in range(10)}

    def _update_window_title() -> None:
        enabled_text = "".join(str(idx) if idx in enabled_camera_indices else "-" for idx in range(10))
        glfw.set_window_title(window, f"{title} | cameras: {enabled_text}")

    def _build_visible_point_cloud() -> Pointclouds | None:
        visible_mask = torch.ones(camera_indices.shape[0], dtype=torch.bool, device=device)
        for idx, idx_mask in indexed_point_masks.items():
            if idx not in enabled_camera_indices:
                visible_mask = visible_mask & (~idx_mask)

        if not bool(visible_mask.any().item()):
            return None
        print(f"Rendering {visible_mask.sum().item()} points out of {xyz.shape[0]} total points.")
        print(torch.min(camera_indices),torch.max(camera_indices))
        return Pointclouds(
            points=[xyz[visible_mask]],
            features=[color[visible_mask]],
        )

    point_cloud = _build_visible_point_cloud()

    if not glfw.init():
        raise RuntimeError("glfw.init() failed.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    window = glfw.create_window(int(window_size[0]), int(window_size[1]), title, None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
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
    cached_size = (0, 0)
    renderer = None
    scene_dirty = True
    visibility_dirty = False
    texture_w = 0
    texture_h = 0

    move_speed = 5.0
    slow_move_factor = 0.25
    mouse_sensitivity = 0.003
    key_rotate_speed = 1.6
    scroll_speed = 0.5

    def _view_dir() -> torch.Tensor:
        d = torch.tensor(
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            device=device,
            dtype=dtype,
        )
        return _normalize(d)

    def _mouse_button_cb(_window: glfw._GLFWwindow, button: int, action: int, _mods: int) -> None:
        nonlocal dragging, last_mouse, scene_dirty
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging = True
                last_mouse = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                dragging = False
            scene_dirty = True

    def _cursor_pos_cb(_window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        nonlocal yaw, pitch, last_mouse, scene_dirty
        if not dragging:
            return
        last_x, last_y = last_mouse
        dx = float(xpos - last_x)
        dy = float(ypos - last_y)
        last_mouse = (xpos, ypos)

        yaw += dx * mouse_sensitivity
        pitch -= dy * mouse_sensitivity
        pitch = float(np.clip(pitch, -1.55, 1.55))
        scene_dirty = True

    def _scroll_cb(_window: glfw._GLFWwindow, _xoffset: float, yoffset: float) -> None:
        nonlocal eye, scene_dirty
        if yoffset == 0.0:
            return
        direction = _view_dir()
        eye = eye + direction * scroll_speed * float(np.sign(yoffset))
        scene_dirty = True

    def _key_cb(_window: glfw._GLFWwindow, key: int, _scancode: int, action: int, _mods: int) -> None:
        nonlocal scene_dirty, visibility_dirty, point_cloud
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        if action == glfw.PRESS and glfw.KEY_0 <= key <= glfw.KEY_9:
            toggled_index = key - glfw.KEY_0
            if toggled_index in enabled_camera_indices:
                enabled_camera_indices.remove(toggled_index)
            else:
                enabled_camera_indices.add(toggled_index)
            point_cloud = _build_visible_point_cloud()
            _update_window_title()
            visibility_dirty = True
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

            if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
                keys_down.add(glfw.KEY_W)
            else:
                keys_down.discard(glfw.KEY_W)
            if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
                keys_down.add(glfw.KEY_S)
            else:
                keys_down.discard(glfw.KEY_S)
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
                keys_down.add(glfw.KEY_A)
            else:
                keys_down.discard(glfw.KEY_A)
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
                keys_down.add(glfw.KEY_D)
            else:
                keys_down.discard(glfw.KEY_D)
            if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                keys_down.add(glfw.KEY_Q)
            else:
                keys_down.discard(glfw.KEY_Q)
            if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
                keys_down.add(glfw.KEY_E)
            else:
                keys_down.discard(glfw.KEY_E)
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                keys_down.add(glfw.KEY_LEFT)
            else:
                keys_down.discard(glfw.KEY_LEFT)
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                keys_down.add(glfw.KEY_RIGHT)
            else:
                keys_down.discard(glfw.KEY_RIGHT)
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                keys_down.add(glfw.KEY_UP)
            else:
                keys_down.discard(glfw.KEY_UP)
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                keys_down.add(glfw.KEY_DOWN)
            else:
                keys_down.discard(glfw.KEY_DOWN)

            rotated = False
            if glfw.KEY_LEFT in keys_down:
                yaw -= key_rotate_speed * dt
                rotated = True
            if glfw.KEY_RIGHT in keys_down:
                yaw += key_rotate_speed * dt
                rotated = True
            if glfw.KEY_UP in keys_down:
                pitch += key_rotate_speed * dt
                rotated = True
            if glfw.KEY_DOWN in keys_down:
                pitch -= key_rotate_speed * dt
                rotated = True
            if rotated:
                pitch = float(np.clip(pitch, -1.55, 1.55))
                scene_dirty = True

            forward_dir = _view_dir()
            right_dir = _normalize(torch.cross(forward_dir, world_up, dim=0))
            up_dir = _normalize(torch.cross(right_dir, forward_dir, dim=0))

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
                eye = eye + move * move_speed * speed_scale * dt
                scene_dirty = True

            fb_w, fb_h = glfw.get_framebuffer_size(window)
            fb_w = max(1, int(fb_w))
            fb_h = max(1, int(fb_h))
            at_now = eye + forward_dir
            R, T = _camera_rt(eye, at_now, world_up)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov_degrees),znear=0.5)
            raster = PointsRasterizationSettings(
                image_size=(fb_h, fb_w),
                radius=point_radius,
                points_per_pixel=points_per_pixel,
                max_points_per_bin=1000
            )
            renderer = PulsarPointsRenderer(
                rasterizer=PointsRasterizer(raster_settings=raster,cameras=cameras),                     
            ).to(device)
            cached_size = (fb_h, fb_w)
            scene_dirty = True

            if visibility_dirty:
                scene_dirty = True
                visibility_dirty = False

            if scene_dirty:
                if point_cloud is None:
                    image_u8 = np.zeros((fb_h, fb_w, 3), dtype=np.uint8)
                else:
                    with torch.no_grad():
                        image = renderer(point_cloud, gamma=(1e-4,))[0, ..., :3]
                    image_u8 = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu().numpy()
                    image_u8 = np.ascontiguousarray(image_u8)

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
                    fb_w,
                    fb_h,
                    0,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    image_u8,
                )
                texture_w = fb_w
                texture_h = fb_h
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

    camera_position_test = torch.tensor([0.0, 0.0, 3.5], dtype=torch.float32, device=test_device)
    look_at_test = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=test_device)
    up_test = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=test_device)

    show_pointcloud_glfw_pytorch3d(
        xyz=xyz_test,
        color=color_test,
        camera_indices=torch.arange(xyz_test.shape[0], device=test_device) % 10,
        camera_position=camera_position_test,
        look_at=look_at_test,
        up=up_test,
        fov_degrees=60.0,
        window_size=(1280, 720),
        title="PyTorch3D GLFW Point Cloud Viewer - Smoke Test",
    )

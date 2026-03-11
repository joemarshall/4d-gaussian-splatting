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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    time : np.array = None

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fl_x, fl_y, w, h):
    top = cy / fl_y * znear
    bottom = -(h-cy) / fl_y * znear
    
    left = -(w-cx) / fl_x * znear
    right = cx / fl_x * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def _to_rotation_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _match_input_type(value, reference):
    if isinstance(reference, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    if isinstance(reference, np.ndarray):
        return value.cpu().numpy().astype(reference.dtype, copy=False)
    return value.cpu().numpy()


def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert a 3x3 rotation matrix to a quaternion in [w, x, y, z] order."""
    matrix = _to_rotation_tensor(rotation_matrix)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {tuple(matrix.shape)}")

    trace = matrix.trace()
    if trace > 0.0:
        s = 2.0 * torch.sqrt(trace + 1.0)
        quaternion = torch.tensor([
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ], dtype=matrix.dtype, device=matrix.device)
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion = torch.tensor([
            (matrix[2, 1] - matrix[1, 2]) / s,
            0.25 * s,
            (matrix[0, 1] + matrix[1, 0]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
        ], dtype=matrix.dtype, device=matrix.device)
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion = torch.tensor([
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[0, 1] + matrix[1, 0]) / s,
            0.25 * s,
            (matrix[1, 2] + matrix[2, 1]) / s,
        ], dtype=matrix.dtype, device=matrix.device)
    else:
        s = 2.0 * torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion = torch.tensor([
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ], dtype=matrix.dtype, device=matrix.device)

    quaternion = quaternion / torch.linalg.norm(quaternion)
    return _match_input_type(quaternion, rotation_matrix)


def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion in [w, x, y, z] order to a 3x3 rotation matrix."""
    quat = _to_rotation_tensor(quaternion).flatten()
    if quat.numel() != 4:
        raise ValueError(f"Expected a quaternion with 4 values, got shape {tuple(quat.shape)}")

    quat = quat / torch.linalg.norm(quat)
    w, x, y, z = quat

    rotation_matrix = torch.tensor([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=quat.dtype, device=quat.device)
    return _match_input_type(rotation_matrix, quaternion)


def quaternion_slerp(quaternion_start, quaternion_end, t):
    """Spherically interpolate between two quaternions in [w, x, y, z] order."""
    q0 = _to_rotation_tensor(quaternion_start).flatten()
    q1 = _to_rotation_tensor(quaternion_end).flatten()
    if q0.numel() != 4 or q1.numel() != 4:
        raise ValueError("Expected both quaternions to contain 4 values")

    interpolation = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    q0 = q0 / torch.linalg.norm(q0)
    q1 = q1 / torch.linalg.norm(q1)

    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + interpolation * (q1 - q0)
        result = result / torch.linalg.norm(result)
        return _match_input_type(result, quaternion_start)

    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    theta = theta_0 * interpolation
    q2 = q1 - dot * q0
    q2 = q2 / torch.linalg.norm(q2)
    result = q0 * torch.cos(theta) + q2 * torch.sin(theta)
    result = result / torch.linalg.norm(result)
    return _match_input_type(result, quaternion_start)
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Camera transformation helper code. Borrowed from NerfStudio
https://github.com/nerfstudio-project/nerfstudio/blob/59d6acf9da718d1d513d260b8808a9bed1bbbd39/nerfstudio/cameras/camera_utils.py#L163
"""

import math
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor
from einops import rearrange, repeat

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data: NDArray, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix: NDArray, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
        K = np.array(K)
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: NDArray, quat1: NDArray, fraction: float, spin: int = 0, shortestpath: bool = True
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion: NDArray) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(pose_a: NDArray, pose_b: NDArray, steps: int = 10) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        pose_a: first pose
        pose_b: second pose
        steps: number of steps the interpolated pose path should contain
    """

    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        # poses_ab.append(pose[:3])
        poses_ab.append(pose)
    return poses_ab


def get_interpolated_k(
    k_a: Float[Tensor, "3 3"], k_b: Float[Tensor, "3 3"], steps: int = 10
) -> List[Float[Tensor, "3 4"]]:
    """
    Returns interpolated path between two camera poses with specified number of steps.

    Args:
        k_a: camera matrix 1
        k_b: camera matrix 2
        steps: number of steps the interpolated pose path should contain

    Returns:
        List of interpolated camera poses
    """
    Ks: List[Float[Tensor, "3 3"]] = []
    ts = np.linspace(0, 1, steps)
    for t in ts:
        new_k = k_a * (1.0 - t) + k_b * t
        Ks.append(new_k)
    return Ks


def get_ordered_poses_and_k(
    poses: Float[Tensor, "num_poses 3 4"],
    Ks: Float[Tensor, "num_poses 3 3"],
) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
    """
    Returns ordered poses and intrinsics by euclidian distance between poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics

    Returns:
        tuple of ordered poses and intrinsics

    """

    poses_num = len(poses)

    ordered_poses = torch.unsqueeze(poses[0], 0)
    ordered_ks = torch.unsqueeze(Ks[0], 0)

    # remove the first pose from poses
    poses = poses[1:]
    Ks = Ks[1:]

    for _ in range(poses_num - 1):
        distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=1)
        idx = torch.argmin(distances)
        ordered_poses = torch.cat((ordered_poses, torch.unsqueeze(poses[idx], 0)), dim=0)
        ordered_ks = torch.cat((ordered_ks, torch.unsqueeze(Ks[idx], 0)), dim=0)
        poses = torch.cat((poses[0:idx], poses[idx + 1 :]), dim=0)
        Ks = torch.cat((Ks[0:idx], Ks[idx + 1 :]), dim=0)

    return ordered_poses, ordered_ks


def get_interpolated_poses_many(
    poses: Float[Tensor, "num_poses 3 4"],
    Ks: Float[Tensor, "num_poses 3 3"],
    steps_per_transition: int = 10,
    order_poses: bool = False,
) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
    """Return interpolated poses for many camera poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics
        steps_per_transition: number of steps per transition
        order_poses: whether to order poses by euclidian distance

    Returns:
        tuple of new poses and intrinsics
    """
    traj = []
    k_interp = []

    if order_poses:
        poses, Ks = get_ordered_poses_and_k(poses, Ks)

    for idx in range(poses.shape[0] - 1):
        pose_a = poses[idx].cpu().numpy()
        pose_b = poses[idx + 1].cpu().numpy()
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
        traj += poses_ab
        k_interp += get_interpolated_k(Ks[idx], Ks[idx + 1], steps=steps_per_transition)

    traj = np.stack(traj, axis=0)
    k_interp = torch.stack(k_interp, dim=0)

    return torch.tensor(traj, dtype=torch.float32), torch.tensor(k_interp, dtype=torch.float32)


def normalize(x: torch.Tensor) -> Float[Tensor, "*batch"]:
    """Returns a normalized vector."""
    return x / torch.linalg.norm(x)


def normalize_with_norm(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """

    norm = torch.maximum(torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([_EPS]).to(x))
    return x / norm, norm


def viewmatrix(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor) -> Float[Tensor, "*batch"]:
    """Returns a camera transformation matrix.

    Args:
        lookat: The direction the camera is looking.
        up: The upward direction of the camera.
        pos: The position of the camera.

    Returns:
        A camera transformation matrix.
    """
    vec2 = normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_relative_pose_batch(poses1: torch.Tensor, poses2: torch.Tensor) -> torch.Tensor:
    """Returns the relative poses between two sets of camera poses.
        Both poses are c2w matrices.

    Args:
        poses1: The first set of camera poses of shape (B, N1, 4, 4).
        poses2: The second set of camera poses of shape (B, N2, 4, 4).

    Returns:
        The relative poses between the two sets of camera poses of shape (B, N1, N2, 4, 4).
        This is T2->T1: if T1 is the world center, then the new pose is still c2w
    """ 
    num_pose1 = poses1.shape[-3]
    num_pose2 = poses2.shape[-3]
    poses1_inv = torch.inverse(poses1)
    poses1_inv = poses1_inv.unsqueeze(2).repeat_interleave(num_pose2, dim=2)  # shape becomes (B, N2, N1, 4, 4)
    poses2 = poses2.unsqueeze(1).repeat_interleave(num_pose1, dim=1)  # shape becomes (B, N2, N1, 4, 4)
    return torch.matmul(poses1_inv, poses2)  # shape becomes (B, N1, N2, 4, 4) # B,v,f,4,4


def find_mid_point_from_circle(P1, P2, r=None, degree=None):
    '''
    P1: torch.tensor, shape (3,)
    P2: torch.tensor, shape (3,)
    r: float, radius of the circle, default is None. if None, calculate the distance between P1 and P2 and set r = distance / 2
    degree: float, degree of the circle, default is None. if None, generate a random degree between 0 and 360
    
    return: torch.tensor, shape (3,)
    '''
    # Convert points to tensors
    # P1 = torch.tensor(P1, dtype=torch.float32)
    # P2 = torch.tensor(P2, dtype=torch.float32)
    
    # Calculate the unit direction vector of the line
    d = P2 - P1
    u = d / torch.norm(d)
    
    if r is None:
        # Calculate the distance between the two points
        r = torch.norm(d) / 2
    
    # Calculate the midpoint
    Pc = (P1 + P2) / 2
    
    # Generate two random vectors not parallel to u
    v1 = torch.randn(3)
    v2 = torch.randn(3)
    while torch.dot(u, v1) == 0:
        v1 = torch.randn(3)
    while torch.dot(u, v2) == 0 or torch.dot(v1, v2) == 0:
        v2 = torch.randn(3)
    
    # Calculate two orthogonal vectors in the plane
    n1 = torch.cross(u, v1)
    n1 = n1 / torch.norm(n1)
    n2 = torch.cross(u, n1)
    n2 = n2 / torch.norm(n2)
    
    # Use these to define a circle in the plane at distance r
    # P = Pc + cos(theta) * r * n1 + sin(theta) * r * n2
    if degree is not None:
        # convert degree to rad
        theta = degree * torch.pi / 180
    else:
        theta = torch.rand(1) * 2 * torch.pi
        
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    circle_points = Pc + r * (cos_t * n1 + sin_t * n2)
    
    ## inform str
    inform_str = f"R{r.item():.2f}_D{theta.item():.2f}"
    return circle_points, inform_str

def sample_mid_point_poses(
    pose_a: Float[Tensor, "3 4"], 
    pose_b: Float[Tensor, "3 4"],
    steps: int,
    radius: Float | None = None,
    degree: Float | None = None,):
    '''
    pose_a: torch.tensor, shape (3, 4), the first pose
    pose_b: torch.tensor, shape (3, 4), the second pose
    steps: int, number of steps to generate
    radius: float, radius of the circle
    degree: float, degree of the circle
    '''
    # import ipdb
    # ipdb.set_trace()
    mid_point, inform_str = find_mid_point_from_circle(pose_a[:3, 3], pose_b[:3, 3], r=radius, degree=degree)
    interpolated_poses = get_interpolated_poses(pose_a.cpu().numpy(), pose_b.cpu().numpy(), steps=steps)
    assert steps % 2 == 0, "steps must be even"
    
    ts = np.linspace(0, 1, steps//2+1)
    trans_a = [(1 - t) * pose_a[:3, 3] + t * mid_point for t in ts[:-1]]
    trans_b = [(1 - t) * mid_point + t * pose_b[:3, 3] for t in ts[1:]]
    trans = trans_a + trans_b
    poses = []
    for pose, tran in zip(interpolated_poses, trans):
        pose[:3, 3] = tran.cpu().numpy()
        poses.append(pose)
    
    return poses, inform_str
    


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def vlength(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / vlength(x, eps)


def look_at(campos, target, cam_type="blender"):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix

    if cam_type == "opengl":
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    
    elif cam_type == "blender":
        # camera forward aligns with -z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 0, 1], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))

    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, cam_type="blender"):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    
    if cam_type == "opengl":
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)
    elif cam_type == "blender":
        ## NOTE blender should be same with opengl, quite strange
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.cos(elevation) * np.cos(azimuth)
        z = radius * np.sin(elevation)
    else:
        raise ValueError(f"Unknown cam_type {cam_type}")
    # x = radius * np.cos(elevation) * np.sin(azimuth)
    # y = radius * np.cos(elevation) * np.cos(azimuth)
    # z = radius * np.sin(elevation)
        
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, cam_type)
    T[:3, 3] = campos
    return T

def normalize(v):
    """Normalize a 3D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


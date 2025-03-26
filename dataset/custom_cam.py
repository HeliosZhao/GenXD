
import os

import numpy as np
from PIL import Image, ImageDraw
import torch

from einops import rearrange
from torch.utils.data import Dataset

import json

from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8, Int
from torch import Tensor
from glob import glob
from jaxtyping import install_import_hook
from typing import Tuple, List
from PIL import Image
import ipdb
from utils.cam_utils import orbit_camera
import imageio.v2 as iio
import math
import torch.nn.functional as F

# Configure beartype and jaxtyping.
from .crop_shim import apply_crop_shim

def pad_pose(pose):
    '''
    pose: N,3,4
    '''
    return torch.cat([pose, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(pose.shape[0], 1, 1)], dim=1)

# Function to find the nearest condition frames to a given group centroid
def find_nearest_condition_frames(condition_positions, group_positions, K):
    '''
    group_positions: N,num_targets,3
    condition_positions: C,3
    '''
    centroid_position = torch.mean(group_positions, dim=1, keepdim=True) # N,1,3
    distances = torch.norm(condition_positions[None] - centroid_position, dim=-1) # N,C
    nearest_condition_indices = torch.argsort(distances, dim=1)[:, :K] # N,K
    return nearest_condition_indices

def opengl_c2w_to_opencv_w2c(c2w):
    ## objaverse use opengl w2c, we need to convert it to opencv w2c
    opengl_to_colmap = np.array([[  1,  0,  0,  0],
                                            [  0, -1,  0,  0],
                                            [  0,  0, -1,  0],
                                            [  0,  0,  0,  1]]).astype(np.float32)
    gl_transform = np.linalg.inv(c2w)
    w2c = gl_transform
    w2c_opencv = opengl_to_colmap @ w2c
    return w2c_opencv[:3] # 3x4

def generate_orbit_cameras(elevation, azimuth_range, radius=2, is_degree=True, target=None, cam_type="blender", N=12, consistent4d=False):
    """
    Generate N cameras along an orbit trajectory with azimuth from [a1, a2).
    
    :param elevation: Scalar, elevation angle.
    :param azimuth_range: Tuple (a1, a2), azimuth range.
    :param radius: Scalar, camera distance from the target.
    :param is_degree: Whether angles are in degrees.
    :param target: 3D point to look at, default is the origin.
    :param cam_type: Camera type, 'blender' or 'opengl'.
    :param N: Number of cameras to generate.
    :return: List of N camera pose matrices, each of shape [4, 4].
    """
    # if consistent4d:
    #     azimuth_range = (15, 345)
    #     is_degree = True
    print(f"Generating orbit cameras, elevation: {elevation}, azimuth_range: {azimuth_range}, radius: {radius}, is_degree: {is_degree}, target: {target}, cam_type: {cam_type}, N: {N}")
    a1, a2 = azimuth_range
    if is_degree:
        a1, a2 = np.deg2rad([a1, a2])  # Convert to radians if in degrees
        elevation = np.deg2rad(elevation)

    # Generate N evenly spaced azimuth angles in the range [a1, a2)
    azimuths = np.linspace(a1, a2, N, endpoint=False)
    if consistent4d:
        azimuths = [0] + np.linspace(a1, a2, N-1, endpoint=False).tolist()
        azimuths = 2 *np.pi - np.array(azimuths)
        # azimuths = azimuths.tolist()

    # Generate camera poses for each azimuth angle
    camera_poses = []
    for azimuth in azimuths:
        pose = orbit_camera(elevation, azimuth, radius, is_degree=False, target=target, cam_type=cam_type) # 4x4, blender c2w
        pose = opengl_c2w_to_opencv_w2c(pose) # 3x4, colmap w2c
        camera_poses.append(pose)

    return np.stack(camera_poses, axis=0) # N,3,4


def generate_forwardfacing_cameras(elevation, azimuth_range, radius=2, is_degree=True, target=None, cam_type="blender", N=12):
    """
    Generate N cameras along an orbit trajectory with azimuth from [a1, a2).
    
    :param elevation: Scalar, elevation angle.
    :param azimuth_range: Tuple (a1, a2), azimuth range.
    :param radius: Scalar, camera distance from the target.
    :param is_degree: Whether angles are in degrees.
    :param target: 3D point to look at, default is the origin.
    :param cam_type: Camera type, 'blender' or 'opengl'.
    :param N: Number of cameras to generate.
    :return: List of N camera pose matrices, each of shape [4, 4].
    """
    num_points_in_set = N//4
    
    a1, a2 = azimuth_range
    if is_degree:
        a1, a2 = np.deg2rad([a1, a2])  # Convert to radians if in degrees
        elevation = np.deg2rad(elevation)

    # Generate N evenly spaced azimuth angles in the range [a1, a2)
    azimuths = np.concatenate([np.linspace(a1, a2, N//2, endpoint=False), np.linspace(a2, a1, N//2, endpoint=False)])
    elevations = np.concatenate([np.linspace(0, elevation, N//4, endpoint=False)] + [np.linspace(elevation, 0, N//4, endpoint=False)] + [np.linspace(0, -elevation, N//4, endpoint=False)] + [np.linspace(-elevation, 0, N//4, endpoint=False)])
    # elevations = elevations + elevations
    # ipdb.set_trace()
    assert len(elevations) == len(azimuths) == N
    azimuths[0] = 0
    elevations[0] = 0

    # Generate camera poses for each azimuth angle
    camera_poses = []
    for azi, ele in zip(azimuths, elevations):
        pose = orbit_camera(ele, azi, radius, is_degree=False, target=target, cam_type=cam_type) # 4x4, blender c2w
        pose = opengl_c2w_to_opencv_w2c(pose) # 3x4, colmap w2c
        camera_poses.append(pose)
    # ipdb.set_trace()
    return np.stack(camera_poses, axis=0) # N,3,4


## evaluation dataset
class StaticCamSingleDataset(Dataset):
    def __init__(self,
                 data_root: str = "datasets/MR6",
                 num_frames=12,
                 size=256,
                 mode="first",
                 indicator=0,
                 camera_info=None,
                 scale=1.0,
                 pad_to_square=False,
                 focal=1.414,
                **kwargs) -> None:
        super().__init__()
        print(f"Using static camera dataset, focal: {focal}")
        self.data_root = data_root
        self.pad_to_square = pad_to_square
        self.focal = focal
        if os.path.isdir(data_root):
            data_paths = sorted(glob(os.path.join(data_root, "*.png"))) + sorted(glob(os.path.join(data_root, "*.jpg")))
        else:
            data_paths = [data_root]
        
        data_list = []
        for dp in data_paths:
            ## load image
            img = Image.open(dp)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, (0, 0), img)
                img = background
                
            data = tf.ToTensor()(img)
            scene = os.path.basename(dp).split(".")[0]
            if mode == "all":
                for mm in ["first", "mid", "last"]:
                    data_list.append({
                        "scene": scene,
                        "mode": mm,
                        "data": data
                    })
            else:
                data_list.append({
                    "scene": scene,
                    "mode": mode,
                    "data": data
                })
            
        # self.data = Image.open(data_root).convert('RGB')
        # self.data = tf.ToTensor()(self.data)
        # self.data = data_list
        # ipdb.set_trace()
        target_trajectory = []
        if camera_info.mode == "orbit":
            # random choose one 
            w2c = generate_orbit_cameras(camera_info.elevation, camera_info.azimuth_range, radius=camera_info.radius, is_degree=True, target=None, cam_type="blender", N=num_frames, consistent4d="consistent4d" in data_root)
            traj_name = f"static" if "evaluation-object" in data_root else f"orbit-{camera_info.elevation}-{camera_info.azimuth_range[0]}-{camera_info.azimuth_range[1]}"
            target_trajectory.append((traj_name, w2c.reshape(-1, 3, 4))) # N,3,4
        elif camera_info.mode == "forward":
            w2c = generate_forwardfacing_cameras(camera_info.elevation, camera_info.azimuth_range, radius=camera_info.radius, is_degree=True, target=None, cam_type="blender", N=num_frames)
            traj_name = f"forward-{camera_info.elevation}-{camera_info.azimuth_range[0]}-{camera_info.azimuth_range[1]}"
            target_trajectory.append((traj_name, w2c.reshape(-1, 3, 4)))
            
        for traj in target_trajectory:
            for data in data_list:
                data.update({
                    "trajectory": traj
                })
                
        self.near = 1
        self.far = 100
        self.num_frames = num_frames
        self.size = size
        self.mode = mode
        self.indicator = indicator
        self.scale = scale
        self.data = data_list

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self):
        return len(self.data)
    

    def padding_images_to_square(self, image, intrinsics):
        '''
        image, F,C,H,W
        '''
        _,C,H,W = image.shape
        # Calculate padding
        diff = abs(H - W)
        if H < W:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            pad_left, pad_right = 0, 0
        else:
            pad_left = diff // 2
            pad_right = diff - pad_left
            pad_top, pad_bottom = 0, 0

        # Apply padding
        padded_tensor = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom))
        size = padded_tensor.shape[-1]
        # get the padding size after resizing to self.size
        pad_info = torch.tensor([pad_top, pad_bottom, pad_left, pad_right])
        pad_info = pad_info * self.size // size
        # Adjust the intrinsics to account for the cropping.
        intrinsics = intrinsics.clone()
        intrinsics[..., 0, 0] *= W / size  # fx
        intrinsics[..., 1, 1] *= H / size  # fy
    
        return padded_tensor, intrinsics, pad_info
    
    
    def __getitem__(self, index):
        # ipdb.set_trace()
        sample = self.data[index]
        scene = sample["scene"]
        mode = sample["mode"]
        data = sample["data"]
        name, w2c = sample["trajectory"]
        w2c = torch.from_numpy(w2c)[:self.num_frames]
        key = os.path.join(scene, str(self.indicator), f"{name}")
        # images = data['images']
        if mode == "mid":
            context_indices = torch.tensor([self.num_frames//2], dtype=torch.int64)
            context_target_correspondence = torch.tensor([self.num_frames//2], dtype=torch.int64)
        elif mode == "first":
            context_indices = torch.tensor([0], dtype=torch.int64)
            context_target_correspondence = torch.tensor([0], dtype=torch.int64)
        elif mode == "last":
            context_indices = torch.tensor([self.num_frames-1], dtype=torch.int64)
            context_target_correspondence = torch.tensor([self.num_frames-1], dtype=torch.int64)
        target_indices = torch.arange(self.num_frames)

        intrinsics = torch.eye(3, dtype=torch.float32)[None].repeat(self.num_frames, 1, 1) # N,3,3
        
        intrinsics[:, 0, 0] = self.focal
        intrinsics[:, 1, 1] = self.focal
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5
        
        extrinsics = torch.eye(4, dtype=torch.float32)[None].repeat(self.num_frames, 1, 1) # N,4,4
        extrinsics[:, :3] = w2c[:, :3] # 4,4
        extrinsics = extrinsics.inverse() # c2w # 4,4
        # multiply the translation by scale
        extrinsics[:, :3, 3] *= self.scale
        # Load the images.
        context_images = data[None]
        cond_intrinsics = intrinsics[context_indices]
        if self.pad_to_square:
            context_images, cond_intrinsics, pad_info = self.padding_images_to_square(context_images, cond_intrinsics)
            
            
        target_images = data[None].repeat(self.num_frames, 1, 1, 1)
        target_intrinsics = intrinsics[target_indices]
        if self.pad_to_square:
            target_images, target_intrinsics, _ = self.padding_images_to_square(target_images, target_intrinsics)
            
            
        scale = 1

        nf_scale = 1.0
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": cond_intrinsics,
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
                "scale": scale,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
                "scale": scale,
            },
            "scene": key,
            "indicator": self.indicator,
        }
        if self.pad_to_square:
            example["pad_info"] = pad_info
            
            
        example["context_target_correspondence"] = context_target_correspondence
        return apply_crop_shim(example, (self.size, self.size))



## evaluation dataset
class StaticCamMultiDataset(Dataset):
    def __init__(self,
                 data_root: str = "datasets/MR6",
                 num_frames=12,
                 size=256,
                 mode="first",
                 indicator=0,
                 camera_info=None,
                 scale=1.0,
                 video_dir=None,
                **kwargs) -> None:
        super().__init__()
        self.data_root = data_root
        if os.path.isdir(data_root):
            data_paths = sorted(glob(os.path.join(data_root, "*.png"))) + sorted(glob(os.path.join(data_root, "*.jpg")))
        else:
            data_paths = [data_root]
        
        num_context_views = 3
        num_targets = num_frames - num_context_views
        
        data_list = []
        data_names = [os.path.basename(dp).split(".")[0] for dp in data_paths]
        
        
        
        self.data = []
        self.videos = {}
        cam_mode = camera_info.mode
        if cam_mode == "orbit":
            # random choose one 
            w2c = generate_orbit_cameras(camera_info.elevation, camera_info.azimuth_range, radius=camera_info.radius, is_degree=True, target=None, cam_type="blender", N=81)
            traj_name = f"static" if "evaluation-object" in data_root else f"orbit-{camera_info.elevation}-{camera_info.azimuth_range[0]}-{camera_info.azimuth_range[1]}"
        elif cam_mode == "forward":
            w2c = generate_forwardfacing_cameras(camera_info.elevation, camera_info.azimuth_range, radius=camera_info.radius, is_degree=True, target=None, cam_type="blender", N=72)
            traj_name = f"forward-{camera_info.elevation}-{camera_info.azimuth_range[0]}-{camera_info.azimuth_range[1]}"
        
        self.cam_info_ext = f"-{camera_info.elevation}-{camera_info.azimuth_range[0]}-{camera_info.azimuth_range[1]}"
        
        gen_poses = torch.from_numpy(w2c) # G,3,4
        
        for dp in data_names:
            video_path = os.path.join(video_dir, f"{dp}/0/{traj_name}.mp4")
            video = np.stack(iio.mimread(video_path), axis=0) # F,H,W,3
            
            video_cams_info = json.load(
                open(video_path.replace(".mp4", ".json"))
            )
            video_cams = torch.tensor(video_cams_info['cameras']) # N,3,4, w2c
            self.videos[dp] = {
                "video": video,
                "cameras": video_cams,
                "fx": video_cams_info['fx'] / video_cams_info['width'],
                "fy": video_cams_info['fy'] / video_cams_info['height'],
                "cx": video_cams_info['cx'] / video_cams_info['width'],
                "cy": video_cams_info['cy'] / video_cams_info['height'],
            }
            
            
            if not os.path.exists(video_path):
                raise ValueError(f"Video path {video_path} does not exist")
            
            cond_c2w_t = pad_pose(video_cams).inverse()[:,:3, -1] # C,3
            gen_c2w_t = pad_pose(gen_poses).inverse()[:,:3, -1] # G,3

            ## sequential grouping
            nearest_indices = torch.arange(num_targets).reshape(1, -1) # 1, num_targets
            num_groups = gen_poses.size(0) // num_targets
            offset = torch.arange(num_groups).reshape(-1, 1) * num_targets # num_groups, 1
            nearest_indices = nearest_indices.repeat(num_groups, 1) + offset # num_groups, num_targets

            group_positions = gen_c2w_t[nearest_indices] # N, num_targets, 3
            nearest_condition_indices = find_nearest_condition_frames(cond_c2w_t, group_positions, num_context_views) # N, num_context_views
            target_poses = gen_poses[nearest_indices] # N, num_targets, 3, 4
            for i in range(target_poses.size(0)):
                self.data.append({
                    "key": dp,
                    "target_poses": target_poses[i], # num_targets, 3, 4
                    "condition_indices": nearest_condition_indices[i].tolist(),
                    "group": i
                })
                
        self.near = 1
        self.far = 100
        self.num_frames = num_frames
        self.size = size
        self.mode = mode
        self.indicator = indicator
        self.scale = scale
        self.pad_to_square = False
        self.num_context_views = num_context_views

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self):
        return len(self.data)

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
        scene: str = None,
    ) -> Tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        c2w = w2c.inverse()
        
        return c2w, intrinsics
    
    def __getitem__(self, index):
        # ipdb.set_trace()
        
        data = self.data[index]
        key = data["key"]
        scene = os.path.join(key, str(self.indicator), f"{data['group']:03d}-{self.cam_info_ext}")
        
        condition_indices = data["condition_indices"]
        torch_data = self.videos[key]

        condition_images = torch.from_numpy(torch_data["video"][condition_indices]) # F,H,W,3
        condition_poses = torch_data["cameras"][condition_indices] # F,3,4
        fx, fy, cx, cy = torch_data["fx"], torch_data["fy"], torch_data["cx"], torch_data["cy"]
        cond_intrinsics = torch.tensor([fx, fy, cx, cy, 0, 0]).reshape(1,-1).repeat(condition_poses.shape[0], 1) # F,6
        condition_poses = torch.cat([cond_intrinsics, condition_poses.reshape(-1, 12)], dim=1) # F,18
        
        target_poses = torch.tensor(data["target_poses"]).reshape(-1, 12) # N,3,4
        target_poses_intrinsics = torch.tensor([fx, fy, cx, cy, 0, 0]).reshape(1,-1).repeat(target_poses.shape[0], 1)
        target_poses = torch.cat([target_poses_intrinsics, target_poses], dim=1) # N,18

        cond_extrinsics, cond_intrinsics = self.convert_poses(condition_poses, key)
        tgt_extrinsics, tgt_intrinsics = self.convert_poses(target_poses, key)
        context_images = condition_images.permute(0, 3, 1, 2).float() / 255.0
        if self.pad_to_square:
            context_images, cond_intrinsics, pad_info = self.padding_images_to_square(context_images, cond_intrinsics)
        target_images = context_images[:1].repeat_interleave(self.num_frames, dim=0) # placeholder
        
        if self.num_context_views == 1:
            if self.mode == "mid":
                context_target_correspondence = torch.tensor([self.num_frames//2], dtype=torch.int64)
            elif self.mode == "first":
                context_target_correspondence = torch.tensor([0], dtype=torch.int64)
            elif self.mode == "last":
                context_target_correspondence = torch.tensor([self.num_frames-1], dtype=torch.int64)
        elif self.num_context_views == 2:
            context_target_correspondence = torch.tensor([0, self.num_frames-1], dtype=torch.int64)
            # concatenate the first and last frame
            tgt_extrinsics = torch.cat([cond_extrinsics[:1], tgt_extrinsics, cond_extrinsics[1:]], dim=0)
            tgt_intrinsics = torch.cat([cond_intrinsics[:1], tgt_intrinsics, cond_intrinsics[1:]], dim=0)
            
        elif self.num_context_views == 3:
            target_length = self.num_frames - self.num_context_views
            target_extrinsics = torch.cat([
                cond_extrinsics[:1], # first frame
                tgt_extrinsics[:target_length//2],  # half of the frames
                cond_extrinsics[1:2], # mid frame
                tgt_extrinsics[target_length//2:], # half of the frames
                cond_extrinsics[2:]], dim=0)
            tgt_extrinsics = target_extrinsics
            tgt_intrinsics = cond_intrinsics[:1].repeat_interleave(self.num_frames, dim=0)
            
            context_target_correspondence = torch.tensor([0, target_length//2+1, self.num_frames-1], dtype=torch.int64)

        nf_scale = scale = 1 # scale is controlled by radius

        assert tgt_extrinsics.size(0) == self.num_frames
        assert tgt_intrinsics.size(0) == self.num_frames
        assert target_images.size(0) == self.num_frames
        assert cond_extrinsics.size(0) == self.num_context_views
        assert cond_intrinsics.size(0) == self.num_context_views
        assert context_images.size(0) == self.num_context_views
        
        example = {
            "context": {
                "extrinsics": cond_extrinsics,
                "intrinsics": cond_intrinsics,
                "image": context_images,
                "near": self.get_bound("near", len(cond_extrinsics)) / nf_scale,
                "far": self.get_bound("far", len(cond_extrinsics)) / nf_scale,
                "index": torch.tensor(condition_indices),
                "scale": scale,
            },
            "target": {
                "extrinsics": tgt_extrinsics,
                "intrinsics": tgt_intrinsics,
                "image": target_images,
                "near": self.get_bound("near", self.num_frames) / nf_scale,
                "far": self.get_bound("far", self.num_frames) / nf_scale,
                "index": torch.arange(self.num_frames),
                "scale": scale,
            },
            "scene": scene,
            "indicator": 0,
            
        }
        
        
        example["context_target_correspondence"] = context_target_correspondence
        if self.pad_to_square:
            example["pad_info"] = pad_info
        
        return apply_crop_shim(example, (self.size, self.size))



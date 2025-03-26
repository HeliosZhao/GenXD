
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
import random
import ipdb
from typing import Tuple, List
import torch.nn.functional as F
# Configure beartype and jaxtyping.
from .crop_shim import apply_crop_shim

def sample_evenly(data_list, num_samples):
    if num_samples <= 0 or num_samples > len(data_list):
        raise ValueError("num_samples must be between 1 and the length of the data_list.")
    
    step = len(data_list) / num_samples
    return [data_list[int(i * step)] for i in range(num_samples)]


def farthest_point_sample_tensor(point, npoint, start_idx=None):
    """
    A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.
    
    Input:
        point: point data for sampling, [N, D] 
        npoint: number of samples
    Return:
        centroids: sampled point index, [npoint, D]
    """
    device = point.device
    N, D = point.shape
    xyz = point
    centroids = torch.zeros((npoint,), device=device)
    distance = torch.ones((N,), device=device) * 1e10
    if start_idx is None:
        farthest = np.random.randint(0, N)
    else:
        farthest = start_idx
        
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    point = point[centroids.long()]
    return point, centroids.long()


    
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

def pad_pose(pose):
    '''
    pose: N,3,4
    '''
    return torch.cat([pose, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(pose.shape[0], 1, 1)], dim=1)

## evaluation dataset
class ReconFusionDirectGenDataset(Dataset):
    '''
    ReconFusion Dataset directly generating target samples
    '''
    near_far = {
        "re10k": (1, 100),
        "co3d": (0.5, 40),
        "dtu": (1.125, 4.25),
        "mipnerf": (0.1, 1e5),
        "llff": (10, 100),
    }
    def __init__(self,
                 data_root: str = "datasets/reconfusion-torch",
                 data_name: str = "re10k",
                 pose_dir: str = "outputs-svd/poses", # dir to the pose data
                 num_context_views=2,
                 n_views=3,
                 num_frames=12,
                 size=256,
                 mode="first",
                 scene_scale=False,
                 group_strategy="group", # group or sequential
                 pad_to_square=False,
                 generate_all=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        data_name = os.path.basename(data_root)
        index = json.load(open(os.path.join(data_root, "index.json")))
        data_list = {}
        pose_dict = {}
        self.data = []
        num_targets = num_frames - num_context_views
        self.pad_to_square = pad_to_square
        # ipdb.set_trace()
        assert num_context_views == 3, "only support 3 context views"
        
        print(f"Generating dataset for {data_name}, group strategy: {group_strategy}")
        for key, torchpath in index.items():
            torch_data = torch.load(os.path.join(data_root, torchpath))[0]
            data_list[key] = torch_data
            few_view_train_idx = torch_data[f"train_test_split_{n_views}"]['train_ids']
            few_view_test_idx = torch_data[f"train_test_split_{n_views}"]['test_ids']
            if generate_all:
                few_view_test_idx = [i for i in range(len(torch_data["images"]))]
            # ipdb.set_trace()
            if len(few_view_test_idx) % num_targets != 0:
                ## pad the test indices to make it divisible by num_targets
                num_pad = num_targets - len(few_view_test_idx) % num_targets
                few_view_test_idx = few_view_test_idx + few_view_test_idx[:num_pad]
                if len(few_view_test_idx) < num_targets:
                    # hard code to pad to num_targets, for llff
                    few_view_test_idx = few_view_test_idx + few_view_test_idx[:num_targets-len(few_view_test_idx)]
                
                
            condition_poses = torch_data["cameras"][few_view_train_idx] # N,18
            condition_extrinsics = condition_poses[:, 6:].reshape(-1, 3, 4) # N,3,4

            target_poses = torch_data["cameras"][few_view_test_idx] # N,18
            target_extrinsics = target_poses[:, 6:].reshape(-1, 3, 4) # N,3,4
            target_poses = target_extrinsics.reshape(-1, num_targets, 3, 4) # N,num_targets,3,4
            few_view_test_idx = torch.tensor(few_view_test_idx).reshape(-1, num_targets)
    
            for i in range(target_poses.size(0)):
                self.data.append({
                    "key": key,
                    "target_poses": target_poses[i], # num_targets, 3, 4
                    "condition_indices": [i for i in range(num_context_views)],
                    "group": i + 1000*int(pad_to_square),
                    "target_indices": few_view_test_idx[i].tolist(),
                })
                
        
        self.data_list = data_list
        self.pose_dict = pose_dict
            
        self.num_context_views = num_context_views
        self.near, self.far = 1, 100
        self.baseline_epsilon = 1e-3
        self.n_views = n_views
        self.num_frames = num_frames
        self.make_baseline_1 = False
        self.size = size
        self.data_name = data_name
        self.scale = scene_scale

        # ipdb.set_trace()
        self.mode = mode

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
        if self.scale:
            scale_factor, translation_vector = self.scene_scale_dict[scene]
            translation_vector = torch.FloatTensor(translation_vector)
            translation_vector = repeat(translation_vector, "h -> b h", b=b) # b,3
            ## normalize c2w translation
            c2w[:, :3, 3] = (c2w[:, :3, 3] + translation_vector) / scale_factor
            
        return c2w, intrinsics

    def convert_images(
        self,
        images: List[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(tf.ToTensor()(image))
        return torch.stack(torch_images)

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
        
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ipdb.set_trace()
        
        data = self.data[index]
        key = data["key"]
        
        condition_indices = data["condition_indices"]
        torch_data = self.data_list[key]
        few_view_train_idx = torch_data[f"train_test_split_{self.n_views}"]['train_ids']
        condition_indices = [few_view_train_idx[c] for c in condition_indices]
        # condition_images = [torch_data["images"][c] for c in condition_indices]
        condition_poses = torch_data["cameras"][condition_indices]
        fx, fy, cx, cy = condition_poses[0, :4]
        # ipdb.set_trace()
        target_poses = torch.tensor(data["target_poses"]).reshape(-1, 12) # N,3,4
        target_poses_intrinsics = torch.tensor([fx, fy, cx, cy, 0, 0]).reshape(1,-1).repeat(target_poses.shape[0], 1)
        target_poses = torch.cat([target_poses_intrinsics, target_poses], dim=1) # N,18

        cond_extrinsics, cond_intrinsics = self.convert_poses(condition_poses, key)
        tgt_extrinsics, tgt_intrinsics = self.convert_poses(target_poses, key)
        images = self.convert_images(torch_data["images"]) # F,3,H,W
        context_images = images[condition_indices] # C,3,H,W
        target_images = images[data["target_indices"]] # N,3,H,W
        if self.pad_to_square:
            context_images, cond_intrinsics, pad_info = self.padding_images_to_square(context_images, cond_intrinsics)
            target_images, tgt_intrinsics, _ = self.padding_images_to_square(target_images, tgt_intrinsics)
            # context_images, pad_info = self.padding_images_to_square(context_images)
            # target_images, _ = self.padding_images_to_square(target_images)
        # target_images = context_images[:1].repeat_interleave(self.num_frames, dim=0) # placeholder
        
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
            
            target_images = torch.cat([
                context_images[:1], # first frame
                target_images[:target_length//2], # half of the frames
                context_images[1:2], # mid frame
                target_images[target_length//2:], # half of the frames
                context_images[2:]], dim=0)
            
            context_target_correspondence = torch.tensor([0, target_length//2+1, self.num_frames-1], dtype=torch.int64)

        nf_scale = scale = 1

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
                "index": torch.tensor(data["target_indices"]),
                "scale": scale,
            },
            "scene": os.path.join(self.data_name, f"{self.n_views}views", data['key']+f"_group{data['group']}"),
            "indicator": 0,
            # "pad_info": pad_info,
            "target_indices": torch.tensor(data["target_indices"]),
        }
        
        example["context_target_correspondence"] = context_target_correspondence
        if self.pad_to_square:
            example["pad_info"] = pad_info
        return apply_crop_shim(example, (self.size, self.size))

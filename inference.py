import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from einops import rearrange

import diffusers
# from diffusers import StableVideoDiffusionPipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler

from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image

from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader, Dataset, IterableDataset

import imageio

import json

from dataset import create_dataset
from models.unet_4d import UNetMVTemporalConditionModel
from genxd_pipeline import GenXDPipeline, get_relative_pose_batch, get_rays
import ipdb

from utils.config import load_typed_root_config
from utils.global_cfg import set_cfg
    
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")



def save_image_video(context, color, path, fps=8, **kwargs):
    '''
    context: 2,3,H,W
    color: N,3,H,W
    '''
    # ipdb.set_trace()
    num_frames = color.size(0)
    ctx1 = []
    if context is not None:
        for i in range(context.size(0)):
            ctx1.append( context[i:i+1].repeat_interleave(num_frames, dim=0) ) # F,3,H,W
        
    ctx1.append(color)
    # ctx1 = torch.cat(ctx1, dim=-1) # N,3,H,W
    # ctx1 = context.repeat_interleave(num_frames, dim=0) # N,3,H,W
    # ctx2 = context[1:2].repeat_interleave(num_frames, dim=0) # N,3,H,W
    vid = torch.cat(ctx1, dim=-1).clamp(0,1) # N,3,H,3W
    vid = rearrange(vid, "f c h w -> f h w c").numpy()
    vidnp = (vid * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, list(vidnp), fps=fps, **kwargs)
    return vidnp

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
@torch.no_grad()
def main(cfg_dict: DictConfig):
    # ipdb.set_trace()
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    args = cfg.diffusion

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        dispatch_batches=False,
        # kwargs_handlers=[ddp_kwargs]
    )
    
    device = accelerator.device


    # If passed along, set the training seed now.
    if args.seed is None:
        args.seed = 3407

    set_seed(args.seed)
    generator = torch.Generator(
        device=device).manual_seed(args.seed)

    
    print(f"Using seed {args.seed} for evaluation, generator seed {generator.initial_seed()}")
        
    unet = UNetMVTemporalConditionModel.from_pretrained(
        args.pretrain_unet,
        subfolder="unet",
        # low_cpu_mem_usage=True,
    )
    

    dataset_kwargs = {
        "data_root": cfg.evaluator.data_root,
        "data_name": cfg.evaluator.data_name,
        "pose_dir": cfg.evaluator.pose_dir,
        "num_context_views": cfg.evaluator.num_context_views,
        "n_views": cfg.evaluator.n_views,
        "num_frames": args.num_frames,
        "size": args.height,
        "indicator": cfg.evaluator.indicator,
        "mode": cfg.evaluator.mode,
        "camera_trajectory": cfg.evaluator.camera_trajectory,
        "scale": cfg.evaluator.scale,
        "scene_scale": cfg.evaluator.scene_scale,
        "group_strategy": cfg.evaluator.group_strategy,
        "pad_to_square": cfg.evaluator.pad_to_square,
        "num_pose_per_traj": cfg.evaluator.num_pose_per_traj,
        "time_interval": cfg.evaluator.time_interval,
        "generate_all": cfg.evaluator.generate_all,
        "camera_info": cfg.evaluator.camera_info,
        "video_dir": cfg.evaluator.video_dir,
        "data_stage": cfg.evaluator.data_stage,
        "focal": cfg.evaluator.focal, # objverse3d 1.0938, obj4d 1.3889
        "subset": cfg.evaluator.gso_subset, # gso subset, 25 for nvs, 36 for neus
    }

    evaluation_dataset = create_dataset(cfg.evaluator.data_name, **dataset_kwargs)
    
    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        sampler=None,
        batch_size=1, #cfg.data_loader.train.batch_size,
        shuffle=False, # randomsampler or IterableDataset do not need shuffle
        num_workers=0, #cfg.data_loader.train.num_workers,
    )
    
    weight_type = torch.float32
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrain_unet,
        subfolder="vae",
        # low_cpu_mem_usage=True,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrain_unet,
        subfolder="scheduler",
        # low_cpu_mem_usage=True,
    )

    pipeline_kwargs = {
        "unet": unet,
        "vae": vae,
        "scheduler": scheduler,
    }
    pipeline = GenXDPipeline(**pipeline_kwargs)

    # Prepare everything with our `accelerator`.
    evaluation_dataloader = accelerator.prepare(
        evaluation_dataloader
    )
    
    pipeline = pipeline.to(device=device, dtype=weight_type)
    pipeline.set_progress_bar_config(disable=False)

    # run inference
    val_save_dir = os.path.join(
        args.output_dir, f"motion-{cfg.evaluator.motion_strength}-{cfg.evaluator.mode}-cfg{cfg.evaluator.guidance_scale}")

    # if not os.path.exists(val_save_dir):
    os.makedirs(val_save_dir, exist_ok=True)

    # import copy
    if cfg.evaluator.share_latent:
        shape = (
            1,
            args.num_frames,
            4,
            args.height // 8,
            args.width // 8,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=unet.dtype)
    else:
        latents = None
        
    for batch_idx, eval_batch in enumerate(evaluation_dataloader):
        # ipdb.set_trace()
        if "valid" in eval_batch:
            continue
        num_frames = args.num_frames
        indicator = eval_batch['indicator'][0] if "indicator" in eval_batch else 0
        if "flow" in eval_batch:
            motion = F.sigmoid(eval_batch['flow']) - 0.5 ## consider change to *20
            motion_strength = motion.item() * 100
        else:
            motion_strength = float(cfg.evaluator.motion_strength)
            
        pipeline_call_kwargs = {
            'batch': eval_batch,
            'height': args.height,
            'width': args.width,
            'num_frames': num_frames,
            'generator': generator,
            'output_type': 'pt',
            # 'use_image_embedding': use_image_embedding,
            'indicator': indicator,
            'motion_strength': indicator if args.motion_mode == "indicator" else motion_strength,
            "guidance_scale": cfg.evaluator.guidance_scale,
            "latents": latents,
            
        }
        
        video_output = pipeline(
            **pipeline_call_kwargs
        )
        video_frames = video_output.frames[0] # f,c,h,w
        # renderings = video_output.renderings[0]
        context_image = eval_batch['context']['image'][0].data.cpu() # 0-1
        target_video = eval_batch['target']['image'][0] # 0-1 f,c,h,w
        bool_index = torch.ones(target_video.size(0))
        bool_index[eval_batch['context_target_correspondence']] = 0
        bool_index = bool_index.bool()
        

        
        if cfg.evaluator.save_target_only:
            # only save target frames
            target_video = target_video[bool_index]
            video_frames = video_frames[bool_index]
            eval_batch['target']['extrinsics'] = eval_batch['target']['extrinsics'][0][bool_index][None]
            
        if cfg.evaluator.pad_to_square:
            # ipdb.set_trace()
            pad_info = eval_batch['pad_info'][0]
            pad_top, pad_bottom, pad_left, pad_right = pad_info
            pad_bottom = target_video.size(2) - pad_bottom
            pad_right = target_video.size(3) - pad_right
            # unpad target video and video frames
            target_video = target_video[:, :, pad_top:pad_bottom, pad_left:pad_right]
            video_frames = video_frames[:, :, pad_top:pad_bottom, pad_left:pad_right]
            # unpad context image
            context_image = context_image[:, :, pad_top:pad_bottom, pad_left:pad_right]
            
        save_video = torch.cat([target_video, video_frames], dim=-1) # f,c,h,3w
        (scene,) = eval_batch["scene"]
        if "flow" in eval_batch:
            scene = scene + f"motion{motion_strength:.2f}"
        out_file = os.path.join(
            val_save_dir, "compare",
            f"{scene}.mp4",
        )
        
        save_image_video(context_image, save_video.data.cpu(), out_file, fps=num_frames//2)
        
        if cfg.evaluator.pad_to_square:
            out_vid_file = os.path.join(
                val_save_dir, "vid_pad",
                f"{scene}.mp4",
            )
        else:
            out_vid_file = os.path.join(
                val_save_dir, "vid",
                f"{scene}.mp4",
            )
        # print(video_frames.shape)
        save_image_video(None, video_frames.data.cpu(), out_vid_file, num_frames//2, **{'macro_block_size': 16})
        
        ## draw relative camera trajectory and save camera pose
        relative_pose_4x4 = get_relative_pose_batch(eval_batch['context']['extrinsics'], eval_batch['target']['extrinsics']) # 1,S,f,4,4
        relative_pose_4x4 = relative_pose_4x4[0,0].cpu().numpy() # f,4,4 to the first frame
        ## save camera pose
        w2c = eval_batch['target']['extrinsics'][0].inverse().cpu().numpy() # f,4,4
        w2c = w2c[:, :3]
        height, width = args.height, args.width
        new_height, new_width = video_frames.size(2), video_frames.size(3)
        K = eval_batch['target']['intrinsics'][0,0].cpu()
        fx = K[0,0].item() * width
        fy = K[1,1].item() * height
        cx = K[0,2].item() * new_width
        cy = K[1,2].item() * new_height
        
        # ipdb.set_trace()
        with open(out_vid_file[:-4] + ".json", "w") as f:
            json.dump(
                {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": new_height, "width": new_width, "cameras": w2c.tolist()}, 
                f, indent=4)

        ## save camera pose
        w2c_rel = np.linalg.inv(relative_pose_4x4) # f,4,4
        w2c_rel = w2c_rel[:, :3]
        with open(out_vid_file[:-4] + "_rel.json", "w") as f:
            json.dump(w2c_rel.tolist(), f, indent=4)
        # plot_trajectory_with_orientations(w2c[1:-1,:,3], w2c[1:-1,:,:3], w2c[[0,-1]][:,:,3], w2c[[0,-1]][:,:,:3], save_path=out_file[:-4] + "_w2c.png")
        
        # save target indices for direct generation
        if "target_indices" in eval_batch:
            target_indices = eval_batch["target_indices"][0].cpu().numpy()
            with open(out_vid_file[:-4] + "_target_indices.json", "w") as f:
                json.dump(target_indices.tolist(), f, indent=4)




if __name__ == "__main__":
    main()
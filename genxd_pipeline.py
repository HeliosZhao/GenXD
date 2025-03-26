import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing, EXAMPLE_DOC_STRING
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents, retrieve_timesteps

from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64
from torch import Tensor
from typing import Tuple, List

import ipdb


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> Tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> Tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


class Ray:
    def __init__(self, 
                 origins: Float[Tensor, "*#batch h w 3"], 
                 directions: Float[Tensor, "*#batch h w 3"]) -> None:
        self.origins = origins
        self.directions = directions
        self.height, self.width = origins.size(-3), origins.size(-2)
    
    def to_plucker(self,):
        ray_origins = self.origins # B,V,H,W,3
        ray_directions = self.directions # B,V,H,W,3
        # Normalize ray directions to unit vectors
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1) # B,V,H,W,3
        plucker_rays = torch.cat([ray_directions, plucker_normal], dim=-1) # B,V,H,W,6
        plucker_rays = rearrange(plucker_rays, "b v h w c -> b v c h w")
        return plucker_rays
    
            
def get_rays(
    # coordinates: Float[Tensor, "*#batch dim"],
    height: int,
    width: int,
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"], # B,V,4,4
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"], # B,V,3,3 #batch=B,V,R # here is the normalized intrinsics
) -> Tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]: 
    # import ipdb; ipdb.set_trace() # 
    batch_size, num_views = extrinsics.shape[:2]
    device = extrinsics.device
    # Convert the features and depths into Gaussians.
    xy_ray, _ = sample_image_grid((height, width), device)
    coordinates = repeat(xy_ray, "h w xy -> b v (h w) xy", b=batch_size, v=num_views) # b,v,r,2
    extrinsics = rearrange(extrinsics, "b v i j -> b v  () i j") # b,v,1,4,4
    intrinsics = rearrange(intrinsics, "b v i j -> b v  () i j") # b,v,1,4,4
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape) # b,v,r,3
    origins = origins.reshape(batch_size, num_views, height, width, -1)
    directions = directions.reshape(batch_size, num_views, height, width, -1)

    return Ray(origins, directions)




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

# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


@dataclass
class GenXDPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.FloatTensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor
            of shape `(batch_size, num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.FloatTensor]
    renderings: Union[List[List[PIL.Image.Image]], np.ndarray, torch.FloatTensor]


class GenXDPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):

        video_length = image.size(1)
        t = rearrange(image, "b f c h w -> (b f) c h w").to(device=device)
        latents = self.vae.encode(t).latent_dist.mode()
        image_latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length) 
        # no need to multiple scaling_factor for condition
        
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames


    def prepare_latents_strength(self, video, timestep, dtype, device, generator=None):
        if not isinstance(video, (torch.Tensor)):
            raise ValueError(
                f"`video` has to be of type `torch.Tensor` but is {type(video)}"
            )
        # ipdb.set_trace()
        video = video.to(device=device, dtype=self.vae.dtype) # b,f,c,h,w
        ## inputs to vae should be -1 to 1 
        video = video * 2 - 1 # b,f,c,h,w float32; range -1 to 1
        batch_size, num_frames = video.size(0), video.size(1)
        video = rearrange(video, "b f c h w -> (b f) c h w")
        init_latents = retrieve_latents(self.vae.encode(video), generator=generator)
        init_latents = self.vae.config.scaling_factor * init_latents
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents.to(dtype), noise, timestep)
        latents = init_latents
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num_frames)

        return latents
    
    def prepare_latents_normal(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_latents(
        self, 
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
        video: Optional[torch.FloatTensor] = None,
        timestep: Optional[int] = None,
    ):
        if video is None:
            return self.prepare_latents_normal(
                batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator, latents
            )
        else:
            return self.prepare_latents_strength(
                video, timestep, dtype, device, generator
            )


    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def batch_to_device(self, batch, device):
        for mode in ["context", "target"]:
            for k,v in batch[mode].items():
                batch[mode][k] = v.to(self.unet.dtype).to(device)
        return batch

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
    

    @torch.no_grad()
    def __call__(
        self,
        batch: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 256,
        width: int = 256,
        num_frames: Optional[int] = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        strength: float = 0.5,
        video: torch.FloatTensor | None = None, # b,f,c,h,w
        indicator: int = 2, # 0 for mv, 1 for temporal, 2 for both
        motion_strength: float = 0.5,
        use_motion_embedding: bool = True,
        single_view_inference: bool = False,
        **kwargs,
    ):
        
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0, 1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames`
                (14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the expense of more memory usage. By default, the decoder decodes all frames at once for maximal
                quality. For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            strength (`float`, *optional*, defaults to 0.5): noise strength added to latents, if set to 1, pure noise.
            video (`torch.FloatTensor`, *optional*): video used for SDEdit, if set, the video will be used as the initial, used together with strength.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.FloatTensor`) is returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        ## model condition setting
        enable_camera_add_time = self.unet.config.get("camera_add_time", False)
        enable_ray_condition = self.unet.config.get("ray_condition", False)
        enable_scale_add_time = self.unet.config.get("scale_add_time", False)

        batch_size = batch["target"]["image"].size(0)
        assert batch_size == 1
        assert num_videos_per_prompt == 1
        # ipdb.set_trace()
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale
        batch = self.batch_to_device(batch, device)

        # ipdb.set_trace()
        # Get the context image embeddings for conditioning.
        context_pixel_values = batch["context"]["image"].to(self.vae.dtype) #[:, :1] # use two context frames
        context_pixel_values = context_pixel_values * 2 - 1 # -1 to 1, B,2,C,H,W
        # ipdb.set_trace()
        # 1. Encode input image, first frame
        ## NOTE no image embedding
        image_embeddings = torch.zeros((batch_size*num_videos_per_prompt*2, 256), dtype=self.unet.dtype, device=device)
        
        motion_ids = None
        # ipdb.set_trace()
        added_time_dim = self.unet.config.projection_class_embeddings_input_dim // 256
        added_time_ids = torch.zeros((batch_size*num_videos_per_prompt, num_frames, added_time_dim), dtype=image_embeddings.dtype, device=device)
        if use_motion_embedding:
            motion_ids = torch.ones((batch_size*num_videos_per_prompt), dtype=image_embeddings.dtype, device=device) * motion_strength
        else:
            added_time_ids[:,:,0] = motion_strength
            
        added_time_ids[:,:,1] = 0 if indicator == 1 else 1 # indicator 1 is video, 0 is mv and 2 is both
        cur_indicator = indicator
        indicator = torch.tensor([indicator]).to(device).repeat(batch_size*num_videos_per_prompt) # b,f
        indicator = torch.cat([indicator, indicator], dim=0) # 2b,f
        
        if enable_scale_add_time:
            ## batch['target']['near'] b,f
            added_time_ids = torch.stack([batch['target']['near'], batch['target']['far']], dim=-1).to(device, image_embeddings.dtype) # b,f,2
        if self.do_classifier_free_guidance:
            added_time_ids = torch.cat([torch.zeros_like(added_time_ids), added_time_ids], dim=0) # 2bn,f,14
            if use_motion_embedding:
                motion_ids = torch.cat([torch.zeros_like(motion_ids), motion_ids], dim=0) # 2bn
            ## NOTE add time ids is not dropped during training
            # added_time_ids = torch.cat([added_time_ids, added_time_ids], dim=0)
        # ipdb.set_trace()
        relative_pose_4x4 = get_relative_pose_batch(batch['context']['extrinsics'].to(torch.float32), batch['target']['extrinsics'].to(torch.float32)).to(self.dtype) # 1,S,f,4,4
        relative_pose = relative_pose_4x4[:, :, :, :3].reshape(*relative_pose_4x4.shape[:3], -1) # 1,S,f,3,4 -> 1,S,f,12
        intrinsics = batch['target']['intrinsics'] # b,v,3,3
        fx = intrinsics[:, :, 0, 0] # b,v
        fy = intrinsics[:, :, 1, 1] # b,v
        focal = torch.stack([fx, fy], dim=-1).unsqueeze(1).repeat_interleave(relative_pose.size(1), dim=1)
        relative_pose = torch.cat([relative_pose, focal], dim=-1) # b,v,f,14
        
        if enable_camera_add_time: # Add to time embedding
            added_time_ids = relative_pose[:, 0].repeat_interleave(batch_size*num_videos_per_prompt, dim=0) # bn,f,14 # relative pose to the first frame
            added_time_ids = added_time_ids.to(device, image_embeddings.dtype)
            if self.do_classifier_free_guidance:
                ## NOTE TODO CHECK HERE WE SHOULD USE ZERO OR VALUES
                negative_add_time_ids = torch.zeros_like(added_time_ids) # bn,f,14
                added_time_ids = torch.cat([negative_add_time_ids, added_time_ids], dim=0) # 2bn,f,14
                

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        ## concate context latents with splat latents
        image_latents = torch.zeros((2, num_frames, 4, height // self.vae_scale_factor, width // self.vae_scale_factor), dtype=image_embeddings.dtype, device=device)
    

        if enable_ray_condition:
            # relative_pose_4x4 = relative_pose_4x4[:,0] # b,f,4,4
            rays = get_rays(height // self.vae_scale_factor, width // self.vae_scale_factor, relative_pose_4x4[:,0].to(torch.float32), batch['target']['intrinsics'].to(torch.float32)) 
            # relative pose to the first frame is the ray direction
            ray_conditions = rays.to_plucker().to(image_latents.dtype) # 1,f,6,h,w
            if cur_indicator == 1: # video should not use ray condition
                ray_conditions = torch.zeros_like(ray_conditions)
            if self.do_classifier_free_guidance:
                negative_ray_conditions = torch.zeros_like(ray_conditions).to(image_latents.dtype)
                ray_conditions = torch.cat([negative_ray_conditions, ray_conditions], dim=0) # 2,f,6,h,w
            image_latents = torch.cat([image_latents, ray_conditions], dim=2) # b,f,vc+c,h,w
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        latent_timestep = None
        if video is not None:
            ## use SDEdit
            # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(batch_size)
        
        # ipdb.set_trace()
        num_channels_latents = 8 
        # NOTE this is channels for latent, set to self.unet.config.in_channels initailly.
        ## NOTE in self.prepare_latents, num_channels will be // 2
        
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
            video,
            timestep=latent_timestep,
        )

        self._guidance_scale = guidance_scale

        latent_repeat_factor = 2 if self.do_classifier_free_guidance else 1

        matches = batch['context_target_correspondence'].long().to(device)
        if single_view_inference:
            # ipdb.set_trace()
            matches = matches[:, :1] # inference with single view
            context_pixel_values = context_pixel_values[:, :1] # inference with single view

        context_mask = torch.zeros(1, num_frames).to(device) # b,f
        context_mask.scatter_(1, matches, 1) # one hot
        ## expand context mask to the same size of inp_noisy_latents
        context_mask_map = repeat(context_mask, "b f -> b f c h w", c=1, h=height//8, w=width//8) # b,f,1,h,w
        if self.do_classifier_free_guidance:
            ## context mask should be kept same ,indicating which is condition
            # but the condition values are not the same, it will be zero for cfg
            ## here is different with cfg noise, cfg noise use zero context map to make the latent as pure noise
            context_mask_map = torch.cat([context_mask_map, context_mask_map], dim=0) # 2,f,1,h,w
        context_mask_map = context_mask_map.to(image_embeddings.dtype)

        ## encode context values and replace latents with it
        context_latents = self._encode_vae_image(
            context_pixel_values.to(self.vae.dtype),
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=False,
        ) # 1,1,4,h,w no cfg here
        context_latents = context_latents.to(image_embeddings.dtype) * self.vae.config.scaling_factor # need to multiply with this, as training
        # ipdb.set_trace()
        if self.do_classifier_free_guidance:
            context_latents = torch.cat([torch.zeros_like(context_latents), context_latents], dim=0) # 2,1,4,h,w
        scatter_map = matches[..., None, None, None].repeat(2, 1, context_latents.size(2), context_latents.size(3), context_latents.size(4)) # 2,matches,4,h,w
        
        context_latents_full = torch.zeros_like(latents) # b,f,c,h,w
        if self.do_classifier_free_guidance:
            context_latents_full = torch.cat([context_latents_full, context_latents_full], dim=0) # 2,f,c,h,w
        context_latents_full.scatter_(1, scatter_map.long(), context_latents) # b,f,c,h,w

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
            
        # selected_features = A[batch_indices, index_tensor]
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ipdb.set_trace()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * latent_repeat_factor) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # ipdb.set_trace()
                # replace the matched values with the context values
                latent_model_input = latent_model_input * (1 - context_mask_map) + context_latents_full * context_mask_map
                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents, context_mask_map], dim=2)
                noise_pred = self.unet(
                    latent_model_input.to(self.unet.dtype),
                    t.to(self.unet.dtype),
                    encoder_hidden_states=image_embeddings.to(self.unet.dtype),
                    added_time_ids=added_time_ids.to(self.unet.dtype),
                    return_dict=False,
                    indicator=indicator,
                    motion_ids=motion_ids,
                )[0]

                # perform guidance
                ## NOTE here the condition view also predict noise, and I donot change it here.
                # During training, such noise is not predicted, but inference, cfg requires the noise, so I keep it here.
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        # ipdb.set_trace()
        latents = latents * (1 - context_mask_map[1:]) + context_latents_full[1:] * context_mask_map[1:]
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents.to(self.vae.dtype), num_frames, decode_chunk_size) # -1,1 # b,c,f,h,w
            frames = tensor2vid(frames, self.image_processor, output_type=output_type) # b f c h w
            
            ## output is 0-1 if output type is pt
        else:
            frames = latents
            renderings = None

        self.maybe_free_model_hooks()

        
        if not return_dict:
            return frames

        return GenXDPipelineOutput(frames=frames, renderings=None)


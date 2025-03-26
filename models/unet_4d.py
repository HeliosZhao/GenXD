from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from .unet_3d_blocks import get_down_block, get_up_block
from .unet_4d_blocks import UNetMidBlockMVCascadeTemporal

import ipdb
from glob import glob
import os

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNetMVTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`], [`~models.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockMVCascadeTemporal",
            "CrossAttnDownBlockMVCascadeTemporal",
            "CrossAttnDownBlockMVCascadeTemporal",
            "DownBlockMVCascadeTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockMVCascadeTemporal",
            "CrossAttnUpBlockMVCascadeTemporal",
            "CrossAttnUpBlockMVCascadeTemporal",
            "CrossAttnUpBlockMVCascadeTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
        camera_embedding_dim: int = -1,
        uncertainty_condition: bool = False,
        camera_add_time: bool = False,
        ray_condition: bool = False,
        scale_add_time: bool = False,
        motion_condition: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        ## NOTE projection_class_embeddings_input_dim is the input feature dim of time embedding linear layer
        ## it should be addition_time_embed_dim * num_additional_values
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        self.motion_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim) if motion_condition else None

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        else:
            cross_attention_dim = [None] * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockMVCascadeTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )
        
        # construct linear projection layer for concatenating image CLIP embedding and RT
        # (RT 3x4 + K 4)*2 means context cam and target cam
        # NOTE this can also be the relative pose to each context frame
        ## NOTE using relative pose do not need K anymore
        if camera_embedding_dim > 0:
            self.cc_projection = nn.Linear(cross_attention_dim[0]+camera_embedding_dim, cross_attention_dim[0])
            nn.init.zeros_(list(self.cc_projection.parameters())[0])
            nn.init.eye_(list(self.cc_projection.parameters())[0][:cross_attention_dim[0], :cross_attention_dim[0]])
            nn.init.zeros_(list(self.cc_projection.parameters())[1])
        else:
            self.cc_projection = nn.Identity()

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        indicator: torch.Tensor,
        return_dict: bool = True,
        motion_ids: torch.Tensor = None,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            indicator: (`torch.LongTensor`): batch size
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)
        # if not self.training:
        #     ipdb.set_trace()
        # ipdb.set_trace()
        
        time_embeds = self.add_time_proj(added_time_ids.flatten()) # B*F,D
        if added_time_ids.ndim == 3: # B,F,D
            time_embeds = time_embeds.reshape((added_time_ids.size(0) * added_time_ids.size(1), -1)) # B*F,D
        else:
            time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        if aug_emb.size(0) == batch_size:
            ## if aug emb is a single, repeat it for each frame
            ## else keep it as per frame
            aug_emb = aug_emb.repeat_interleave(num_frames, dim=0)

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        emb = emb + aug_emb 
        if motion_ids is not None:
            # ipdb.set_trace()
            motion_emb = self.time_proj(motion_ids.flatten()) # motion emb is a single value
            motion_emb = motion_emb.reshape((batch_size, -1))
            motion_emb = motion_emb.to(emb.dtype) # B,C
            motion_emb = self.motion_embedding(motion_emb) # B,C
            motion_emb = motion_emb.repeat_interleave(num_frames, dim=0)
            emb = torch.cat([emb, motion_emb], dim=-1)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        # ipdb.set_trace()
        if encoder_hidden_states.size(0) == batch_size:
            ## repeat encoder_hidden_states for each frame
            ## cc_projection should be identity here
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0) 
            encoder_hidden_states = self.cc_projection(encoder_hidden_states)   # (b f),v,c
        else:
            assert encoder_hidden_states.size(0) == batch_size * num_frames
            encoder_hidden_states = self.cc_projection(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)
        # ipdb.set_trace()
        ## NOTE: indicator is used to indicate whether the input is 3d or video or 4d
        image_only_indicator = indicator.unsqueeze(1).repeat(1, num_frames) # B,F
        # image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # import ipdb; ipdb.set_trace()
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)

    def from_pretrained_model(self, pretrained_model: nn.Module, verbose: bool = True):
        pretrained_model_state_dict = pretrained_model.state_dict()
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        ## load multiview from temporal layer
        # ipdb.set_trace()
        non_matched_keys = []
        for k, cur_v in cur_state_dict.items():
            new_state_dict[k] = cur_v
            if k not in pretrained_model_state_dict:
                if "mv_res_block" in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("mv_res_block", "temporal_res_block")]
                elif 'multiview_transformer_blocks' in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("multiview_transformer_blocks", "temporal_transformer_blocks")]
                elif 'spatial_temporal_factor' in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("spatial_temporal_factor", "mix_factor")]
                else:
                    non_matched_keys.append(k)
                    
                continue
            # new_state_dict[k] = torch.zeros_like(cur_v)
            svd_v = pretrained_model_state_dict[k]
            if svd_v.shape == cur_v.shape:
                new_state_dict[k] = svd_v
            else:
                print(f"Shape for {k} param is different, current shape: {cur_v.shape}, pretrained shape : {svd_v.shape}")
                # Shape for conv_in.weight param is different, current shape: torch.Size([320, 16, 3, 3]), pretrained shape : torch.Size([320, 8, 3, 3])
                # ipdb.set_trace()
                new_param = torch.zeros_like(cur_v)
                if new_param.size(1) < svd_v.size(1):
                    new_param = svd_v[:, :new_param.size(1)]
                else:
                    new_param[:, :svd_v.size(1)] = svd_v
                new_state_dict[k] = new_param
                
        if verbose:
            for nmk in non_matched_keys:
                if 'spatial_mv_factor' in nmk or 'gamma' in nmk:
                    continue
                else:
                    raise ValueError(f"Non matched key: {nmk}")
        
        self.load_state_dict(new_state_dict)


    def from_pretrained_4d(self, pretrained_model: nn.Module, verbose: bool = True):
        print("==> Loading pretrained 4d model")
        pretrained_model_state_dict = pretrained_model.state_dict()
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        ## load multiview from temporal layer
        # ipdb.set_trace()
        non_matched_keys = []
        for k, cur_v in cur_state_dict.items():
            new_state_dict[k] = cur_v
            if k not in pretrained_model_state_dict:
                # ipdb.set_trace()
                non_matched_keys.append(k)
                continue
            # new_state_dict[k] = torch.zeros_like(cur_v)
            svd_v = pretrained_model_state_dict[k]
            if svd_v.shape == cur_v.shape:
                new_state_dict[k] = svd_v
            else:
                print(f"Shape for {k} param is different, current shape: {cur_v.shape}, pretrained shape : {svd_v.shape}")
                # Shape for conv_in.weight param is different, current shape: torch.Size([320, 16, 3, 3]), pretrained shape : torch.Size([320, 8, 3, 3])
                # ipdb.set_trace()
                new_param = torch.zeros_like(cur_v)
                if new_param.size(1) < svd_v.size(1):
                    new_param = svd_v[:, :new_param.size(1)]
                else:
                    new_param[:, :svd_v.size(1)] = svd_v
                new_state_dict[k] = new_param
        # ipdb.set_trace()
        if verbose:
            for nmk in non_matched_keys:
                if 'time_mixer' in nmk:
                    print(f"Skipping {nmk}")
                    continue
                else:
                    raise ValueError(f"Non matched key: {nmk}")
        
        self.load_state_dict(new_state_dict)


    def from_pretrained_model(self, pretrained_model: nn.Module, verbose: bool = True):
        pretrained_model_state_dict = pretrained_model.state_dict()
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        ## load multiview from temporal layer
        # ipdb.set_trace()
        non_matched_keys = []
        for k, cur_v in cur_state_dict.items():
            new_state_dict[k] = cur_v
            if k not in pretrained_model_state_dict:
                if "mv_res_block" in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("mv_res_block", "temporal_res_block")]
                elif 'multiview_transformer_blocks' in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("multiview_transformer_blocks", "temporal_transformer_blocks")]
                elif 'spatial_temporal_factor' in k:
                    new_state_dict[k] = pretrained_model_state_dict[k.replace("spatial_temporal_factor", "mix_factor")]
                else:
                    non_matched_keys.append(k)
                    
                continue
            # new_state_dict[k] = torch.zeros_like(cur_v)
            svd_v = pretrained_model_state_dict[k]
            if svd_v.shape == cur_v.shape:
                new_state_dict[k] = svd_v
            else:
                print(f"Shape for {k} param is different, current shape: {cur_v.shape}, pretrained shape : {svd_v.shape}")
                # Shape for conv_in.weight param is different, current shape: torch.Size([320, 16, 3, 3]), pretrained shape : torch.Size([320, 8, 3, 3])
                # ipdb.set_trace()
                new_param = torch.zeros_like(cur_v)
                if new_param.size(1) < svd_v.size(1):
                    new_param = svd_v[:, :new_param.size(1)]
                else:
                    new_param[:, :svd_v.size(1)] = svd_v
                new_state_dict[k] = new_param
                
        if verbose:
            for nmk in non_matched_keys:
                if 'spatial_mv_factor' in nmk or 'gamma' in nmk:
                    continue
                else:
                    raise ValueError(f"Non matched key: {nmk}")
        
        self.load_state_dict(new_state_dict)


    def from_pretrained_3d(self, pretrained_model: nn.Module, verbose: bool = True, skip_gamma=True):
        print("==> Loading pretrained 4d model")
        pretrained_model_state_dict = pretrained_model.state_dict()
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        ## load multiview from temporal layer
        # ipdb.set_trace()
        non_matched_keys = []
        for k, cur_v in cur_state_dict.items():
            new_state_dict[k] = cur_v
            if k not in pretrained_model_state_dict:
                # ipdb.set_trace()
                non_matched_keys.append(k)
                continue
            elif skip_gamma and 'gamma' in k:
                print(f"Skipping {k}")
                continue
            else:
                new_state_dict[k] = pretrained_model_state_dict[k]
        # ipdb.set_trace()
        if verbose:
            for nmk in non_matched_keys:
                print(f"Non matched key: {nmk}")

        self.load_state_dict(new_state_dict)


    def from_pretrained_skip_keys(self, pretrained_model_path: str, verbose: bool = True):
        print(f"==> Loading pretrained model {pretrained_model_path} with missing keys")
        model_path = glob(os.path.join(pretrained_model_path, "*.bin")) + glob(os.path.join(pretrained_model_path, "*.safetensors"))
        if not len(model_path) == 1:
            raise ValueError(f"Only one model should be in {pretrained_model_path}")
        if model_path[0].endswith(".safetensors"):
            import safetensors
            pretrained_model_state_dict = safetensors.torch.load_file(model_path[0], device="cpu")
        else:
            pretrained_model_state_dict = torch.load(model_path[0])
        # pretrained_model_state_dict = pretrained_model.state_dict()
        cur_state_dict = self.state_dict()
        new_state_dict = {}
        ## load multiview from temporal layer
        # ipdb.set_trace()
        non_matched_keys = []
        for k, cur_v in cur_state_dict.items():
            new_state_dict[k] = cur_v
            if k not in pretrained_model_state_dict:
                # ipdb.set_trace()
                non_matched_keys.append(k)
                continue
            else:
                new_state_dict[k] = pretrained_model_state_dict[k]
        # ipdb.set_trace()
        if verbose:
            for nmk in non_matched_keys:
                print(f"Non matched key: {nmk}")

        self.load_state_dict(new_state_dict)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from diffusers.models.resnet import ResnetBlock2D, TemporalResnetBlock



class GammaBlender(nn.Module):
    r"""
    A module to blend spatial, multiview and temporal features

    Parameters:
        alpha_spatial_mv (`float`): The initial value of the blending factor to mix spatial self-attn and multiview.
        alpha_spatial_temporal (`float`): The initial value of the blending factor to mix spatial self-attn and temporal.
        gamma (`float`): The initial value of the blending factor to mix the above two to form 4D.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha_spatial_mv: float,
        alpha_spatial_temporal: float,
        gamma: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_buffer("spatial_mv_factor", torch.Tensor([alpha_spatial_mv]))
            self.register_buffer("spatial_temporal_factor", torch.Tensor([alpha_spatial_temporal]))
            self.register_buffer("gamma", torch.Tensor([gamma]))
            
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("spatial_mv_factor", torch.nn.Parameter(torch.Tensor([alpha_spatial_mv])))
            self.register_parameter("spatial_temporal_factor", torch.nn.Parameter(torch.Tensor([alpha_spatial_temporal])))
            self.register_parameter("gamma", torch.nn.Parameter(torch.Tensor([gamma])))

        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_mix_value(self, mix_value: torch.nn.Parameter) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = mix_value

        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            alpha = torch.sigmoid(mix_value)
        else:
            raise NotImplementedError

        return alpha

    def get_gamma(self, image_only_indicator: torch.Tensor, ndims: int, mix_value: torch.nn.Parameter) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = mix_value

        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(mix_value)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator == 2,
                torch.sigmoid(mix_value)[..., None],
                torch.zeros(1, 1, device=image_only_indicator.device),
            ) # if indicator match the value, then use the learned value, otherwise use 0

            alpha[image_only_indicator == 0] = 1.0
            alpha[image_only_indicator == 1] = 0.0
            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha
    
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        x_multiview: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        image only indicator: Literal[0,1,2] 
        if 0, then only multiview images
        if 1, then only temporal video
        if 2, then both multiview images and temporal video
        '''
        # import ipdb; ipdb.set_trace()
        alpha_spatial_mv = self.get_mix_value(self.spatial_mv_factor)
        alpha_spatial_temporal = self.get_mix_value(self.spatial_temporal_factor)
        gamma = self.get_gamma(image_only_indicator, x_temporal.dim(), self.gamma)

        x_spatial_mv = alpha_spatial_mv * x_spatial + (1.0 - alpha_spatial_mv) * x_multiview
        x_spatial_temporal = alpha_spatial_temporal * x_spatial + (1.0 - alpha_spatial_temporal) * x_temporal

        x = gamma * x_spatial_mv + (1.0 - gamma) * x_spatial_temporal
        
        return x



class MVTemporalBlender(nn.Module):
    r"""
    A module to blend spatial, multiview and temporal features

    Parameters:
        alpha_spatial_mv (`float`): The initial value of the blending factor to mix spatial self-attn and multiview.
        alpha_spatial_temporal (`float`): The initial value of the blending factor to mix spatial self-attn and temporal.
        gamma (`float`): The initial value of the blending factor to mix the above two to form 4D.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha_spatial_mv: float,
        gamma: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_parameter("spatial_mv_factor", torch.Tensor([alpha_spatial_mv]))
            self.register_buffer("gamma", torch.Tensor([gamma]))
            
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("spatial_mv_factor", torch.nn.Parameter(torch.Tensor([alpha_spatial_mv])))
            self.register_parameter("gamma", torch.nn.Parameter(torch.Tensor([gamma])))

        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_mix_value(self, mix_value: torch.nn.Parameter) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = mix_value

        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            alpha = torch.sigmoid(mix_value)
        else:
            raise NotImplementedError

        return alpha

    def get_gamma(self, image_only_indicator: torch.Tensor, ndims: int, mix_value: torch.nn.Parameter) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = mix_value

        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(mix_value)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator == 2,
                torch.sigmoid(mix_value)[..., None],
                torch.zeros(1, 1, device=image_only_indicator.device),
            ) # if indicator match the value, then use the learned value, otherwise use 0

            alpha[image_only_indicator == 0] = 1.0
            alpha[image_only_indicator == 1] = 0.0
            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha
    
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        x_multiview: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        image only indicator: Literal[0,1,2] 
        if 0, then only multiview images
        if 1, then only temporal video
        if 2, then both multiview images and temporal video
        '''
        alpha_spatial_mv = self.get_mix_value(self.spatial_mv_factor)
        x_multiview = alpha_spatial_mv * x_spatial + (1.0 - alpha_spatial_mv) * x_multiview
        # if (image_only_indicator == 2).any():
        #     import ipdb; ipdb.set_trace()
        gamma = self.get_gamma(image_only_indicator, x_temporal.dim(), self.gamma)
        x = gamma * x_multiview + (1.0 - gamma) * x_temporal
        
        return x


class MVSpatioTemporalResBlock(nn.Module):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.temb_channels = temb_channels

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.mv_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )
        
        self.time_mixer = GammaBlender(
            alpha_spatial_mv=merge_factor,
            alpha_spatial_temporal=merge_factor,
            gamma=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        # import ipdb; ipdb.set_trace()
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states_temporal = self.temporal_res_block(hidden_states, temb)
        hidden_states_mv = self.mv_res_block(hidden_states_mix, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states_temporal,
            x_multiview=hidden_states_mv,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states


class MVCascadeTemporalResBlock(MVSpatioTemporalResBlock):
    def __init__(self, in_channels: int, out_channels: int | None = None, temb_channels: int = 512, eps: float = 0.000001, temporal_eps: float | None = None, merge_factor: float = 0.5, merge_strategy="learned_with_images", switch_spatial_to_temporal_mix: bool = False):
        super().__init__(in_channels, out_channels, temb_channels, eps, temporal_eps, merge_factor, merge_strategy, switch_spatial_to_temporal_mix)
        self.time_mixer = MVTemporalBlender(
            alpha_spatial_mv=merge_factor,
            gamma=5, ## sigmoid(10) is close to 1, sigmoid(5) is close to 0.99
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        
        num_frames = image_only_indicator.shape[-1]
        batch_size = hidden_states.size(0) // num_frames
        
        use_motion_emb = temb is not None and temb.size(-1) > self.temb_channels
        if use_motion_emb:
            # import ipdb; ipdb.set_trace()
            temb, motion_emb = temb.chunk(2, dim=-1)
            motion_emb = motion_emb.reshape(batch_size, num_frames, -1)
        hidden_states = self.spatial_res_block(hidden_states, temb)
        
        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )


        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)
            
        hidden_states_mv = self.mv_res_block(hidden_states_mix, temb)
        
        if use_motion_emb:
            hidden_states_temporal = self.temporal_res_block(hidden_states_mv, motion_emb)
        else:
            hidden_states_temporal = self.temporal_res_block(hidden_states_mv, temb)
        
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states_temporal,
            x_multiview=hidden_states_mv,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states
    
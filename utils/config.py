from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

# from .dataset.data_module import DataLoaderCfg, DatasetCfg, DiffusionDatasetCfg
# from .loss import LossCfgWrapper
# from .model.decoder import DecoderCfg
# from .model.encoder import EncoderCfg
# from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg
from typing import Tuple, List



@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int

@dataclass
class CameraCfg:
    azimuth_range: list = field(default_factory=lambda: [0., 360.])
    elevation: float | int = 0.
    radius: float | int = 2.
    mode: str = 'orbit'
    
@dataclass
class DiffusionCfg:
    pretrained_model_name_or_path: str | None = "stabilityai/stable-video-diffusion-img2vid-xt"
    unet_config: str = "diffusion/unet_configs/unet_config.json"
    pixelsplat_path: str | None = None
    revision: str | None = None
    num_frames: int = 16
    width: int = 256
    height: int = 256
    num_validation_images: int = 1
    validation_steps: int = 500
    output_dir: str = "outputs-svd"
    seed: int | None = None # random seed, this will help to auto restart
    per_gpu_batch_size: int = 1
    max_train_steps: int = 100000
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    conditioning_dropout_prob: float = 0.1
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    use_ema: bool = False
    non_ema_revision: str | None = None
    num_workers: int = 8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2 # set to 1e-4?
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    push_to_hub: bool = False
    hub_token: str | None = None
    hub_model_id: str | None = None
    logging_dir: str = "logs"
    mixed_precision: str = "fp16" # ["no", "fp16", "bf16"]
    report_to: str = "wandb" # Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.
    local_rank: int = -1
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: str | None = "latest"
    enable_xformers_memory_efficient_attention: bool = False
    pretrain_unet: str | None = None # this is set to unet path
    rank: int = 128
    splat_dropout_prob: float = 0.1 # drop out splat condition, this will lead the model predict from camera parameters
    use_image_embedding: bool = False
    static_motion_vector: bool = False
    motion_mode: Literal['latent', 'indicator', 'flow'] = 'flow'
    unet4d_pretrained: str | None = None # pretrained 4D model in previous experiments
    unet3d_pretrained: str | None = None ## pretrained 3D model for 4D model
    cfg_zero: bool = True
    use_motion_embedding: bool = False
    single_view_ratio: float = 0.0
    
    ## diffusion config for transformers
    weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap"] = "logit_normal"  
    logit_mean: float = 0.0 # mean to use when using the `'logit_normal'` weighting scheme.
    logit_std: float = 1.0 # std to use when using the `'logit_normal'` weighting scheme.
    mode_scale: float = 1.29 # Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.
    precondition_outputs: bool = True # Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how model `target` is calculated.
    pretrain_transformer: str | None = None # pretrain transformer model
    no_camera: str | None = None # no camera condition, can be set to None | all | ray | pose, denoting use camera, no camera, no ray or no pose
    
    

@dataclass
class EvaluatorLossCfg:
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.1


@dataclass
class EvaluatorOptimCfg:
    percent_dense: float = 0.01
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 1000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

@dataclass
class EvaluatorCfg:
    loss: EvaluatorLossCfg
    optim: EvaluatorOptimCfg
    max_iterations: int = 1000
    densify_until_iter: int = 1000
    densify_from_iter: int = 0
    densification_interval: int = 50
    densify_grad_threshold: float = 0.0002
    prune_threshold: float = 0.005
    opacity_reset_interval: int = 3000
    size_threshold: bool = True
    sh_degree: int = 3
    aug_mode: Literal['reverse', 'reverse_interpolate', None] = 'reverse_interpolate'
    ply_path: str | None = None
    ## render mode: gs_condition: using gs output as condition, splat_condition: using mvsplat as condition, gs_latent: using mvsplat as condition and gs output as latent
    render_mode: Literal['gs_condition', 'splat_condition', 'gs_latent'] = 'splat_condition'
    test_freq: int = 100000
    num_views: int = 4
    diff_weight: float = 1.0
    data_root: str = "datasets/reconfusion-torch"
    data_name: str = "re10k"
    num_context_views: int = 1
    n_views: int = 3
    pipeline: str = "latent_cfg_zero"
    motion_strength: float | int = 5.0
    mode: str = "first"
    indicator: int = 0
    camera_trajectory: str | None = None
    scale: float = 1.0 # the camera trajectoies are normalized, this can be used to scale the camera trajectory
    model_name: str = "cascade4d"
    scene_scale: bool = False
    guidance_scale: float | int = 3.0 # classifier free guidance scale
    pose_dir: str | None = None # the pose directory for the camera trajectory
    group_strategy: str = "group" # group or sequential
    save_target_only: bool = False # will only save the target view
    pad_to_square: bool = False # pad the image to square
    num_pose_per_traj: Optional[int] = None # number of poses per trajectory
    time_interval: int = 0 # time interval between two frames, 0 denotes static scene
    generate_all: bool = False # generate all views
    camera_info: CameraCfg = CameraCfg()
    video_dir: str | None = None
    data_stage: str = "test" # data stage for flow4d dataset
    focal: float = 1.414
    share_latent: bool = False # share latent for video to 4d
    gso_subset: Literal['nvs', 'recon'] = 'nvs' # subset of gso, 25 for nvs, 36 for neus
    
    
@dataclass
class RootCfg:
    wandb: dict
    seed: int
    diffusion: DiffusionCfg
    evaluator: EvaluatorCfg


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg, resolve=True),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


# def separate_loss_cfg_wrappers(joined: dict) -> List[LossCfgWrapper]:
#     # The dummy allows the union to be converted.
#     @dataclass
#     class Dummy:
#         dummy: LossCfgWrapper

#     return [
#         load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
#         for k, v in joined.items()
#     ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
    )

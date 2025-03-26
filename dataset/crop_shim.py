import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from typing import Tuple, List, Dict

def rescale(
    image: Float[Tensor, "c h_in w_in"],
    shape: Tuple[int, int],
) -> Float[Tensor, "c h_out w_out"]:
    h, w = shape
    is_mask = False
    if image.size(0) == 1:
        is_mask = True
        image = repeat(image, "1 h w -> c h w", c=3)
        
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w") if not is_mask else rearrange(image_new, "h w c -> c h w")[:1]


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: Tuple[int, int],
) -> Tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: Tuple[int, int],
) -> Tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: Dict, shape: Tuple[int, int]) -> Dict:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    if views.get("mask", None) is not None:
        mask, _ = rescale_and_crop(views["mask"], views["intrinsics"], shape)
        return {
            **views,
            "image": images,
            "mask": mask,
            "intrinsics": intrinsics,
        }
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: Dict, shape: Tuple[int, int]) -> Dict:
    """Crop images in the example."""
    if "static" in example:
        return {
            **example,
            "context": apply_crop_shim_to_views(example["context"], shape),
            "target": apply_crop_shim_to_views(example["target"], shape),
            "static": apply_crop_shim_to_views(example["static"], shape),
        }
        
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }

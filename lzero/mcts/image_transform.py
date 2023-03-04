from typing import Tuple
import random
import torch
import torch.nn as nn


class Intensity(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class RandomCrop(nn.Module):
    def __init__(self, image_shape: Tuple[int]) -> None:
        super().__init__()
        self.image_shape = image_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        H_, W_ = self.image_shape
        dh, dw = H - H_, W - W_
        h, w = random.randint(0, dh), random.randint(0, dw)
        return x[..., h: h + H_, w: w + W_]


class Transforms(object):
    def __init__(self, augmentation, shift_delta=4, image_shape=(96, 96)):
        self.augmentation = augmentation

        self.transforms = []
        for aug in self.augmentation:
            if aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(shift_delta), RandomCrop(image_shape))
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
            else:
                raise NotImplementedError("not support augmentation type: {}".format(aug))
            self.transforms.append(transformation)

    @torch.no_grad()
    def transform(self, images):
        images = images.float() / 255. if images.dtype == torch.uint8 else images
        processed_images = images.reshape(-1, *images.shape[-3:])
        for transform in self.transforms:
            processed_images = transform(processed_images)

        processed_images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images

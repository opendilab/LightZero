from typing import Tuple, List
import random
import torch
import torch.nn as nn


class Intensity(nn.Module):
    """
    Overview:
        Intensity transformation for data augmentation. Scale the image intensity by a random factor.
    """
    def __init__(self, scale: float) -> None:
        """
        Arguments:
            - scale (:obj:`float`): The scale factor for intensity transformation.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): The input image tensor with shape (B, C, H, W).
            - output (:obj:`torch.Tensor`): The output image tensor with shape (B, C, H, W).
        """
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class RandomCrop(nn.Module):
    """
    Overview:
        Random crop the image to the given size.
    """
    def __init__(self, image_shape: Tuple[int]) -> None:
        """
        Arguments:
            - image_shape (:obj:`Tuple[int]`): The target shape of the image to be cropped.
        """
        super().__init__()
        self.image_shape = image_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): The input image tensor with shape (B, C, H, W), where H and W are \
                the original image shape.
            - output (:obj:`torch.Tensor`): The output image tensor with shape (B, C, H_, W_), where H_ and W_ are \
                the target image shape indicated by `image_shape`.
        """
        H, W = x.shape[2:]
        H_, W_ = self.image_shape
        dh, dw = H - H_, W - W_
        h, w = random.randint(0, dh), random.randint(0, dw)
        return x[..., h: h + H_, w: w + W_]


class ImageTransforms(object):
    """
    Overview:
        Image transformation for data augmentation. Including image normalization (divide 255), random crop and
        intensity transformation.
    """
    def __init__(self, augmentation: List[str], shift_delta: int = 4, image_shape: Tuple[int] = (96, 96)) -> None:
        """
        Arguments:
            - augmentation (:obj:`List[str]`): The list of augmentation types. Now support "shift" and "intensity".
            - shift_delta (:obj:`int`): The delta value for random shift padding before crop. Use ReplicationPad2d \
                to pad the image without the loss of information.
            - image_shape (:obj:`Tuple[int]`): The target shape of the image to be cropped.
        """
        self.augmentation = augmentation

        self.image_transforms = []
        for aug in self.augmentation:
            if aug == "shift":
                # TODO validate the effectiveness of ReflectionPad2d
                transformation = nn.Sequential(nn.ReplicationPad2d(shift_delta), RandomCrop(image_shape))
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
            else:
                raise NotImplementedError("not support augmentation type: {}".format(aug))
            self.image_transforms.append(transformation)

    @torch.no_grad()
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): The input image tensor with shape (B, C, H, W), where H and W are \
                the original image shape.
            - output (:obj:`torch.Tensor`): The output image tensor with shape (B, C, H_, W_), where H_ and W_ are \
                the target image shape indicated by `image_shape`.

        .. note::
            Use torch.no_grad() to save cuda memory. Transformations are not trainable.
        """
        images = images.float() / 255. if images.dtype == torch.uint8 else images
        processed_images = images.reshape(-1, *images.shape[-3:])
        for transform in self.image_transforms:
            processed_images = transform(processed_images)

        processed_images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images

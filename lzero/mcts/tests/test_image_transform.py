import random

import pytest
import torch
from lzero.model import ImageTransforms, RandomCrop


@pytest.mark.unittest
def test_image_transform():
    img = torch.rand((4, 3, 96, 96))
    transform = ImageTransforms(['shift', 'intensity'])
    processed_img = transform.transform(img)
    assert img.shape == (4, 3, 96, 96)
    assert not (img == processed_img).all()


@pytest.mark.unittest
def test_random_crop_samples_offsets_per_batch_item():
    torch.manual_seed(0)
    image = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    images = image.expand(32, -1, -1, -1).clone()

    cropped = RandomCrop((2, 2))(images)

    assert cropped.shape == (32, 1, 2, 2)
    # Identical inputs should not all receive the same minibatch-global crop.
    assert torch.unique(cropped[:, 0, 0, 0]).numel() > 1


@pytest.mark.unittest
def test_random_crop_uses_torch_rng_only():
    images = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 1, 4, 4)
    transform = RandomCrop((2, 2))

    torch.manual_seed(7)
    first = transform(images)
    torch.manual_seed(7)
    second = transform(images)

    assert torch.equal(first, second)

    random.seed(123)
    expected_next_python_random = random.random()
    random.seed(123)
    transform(images)
    assert random.random() == expected_next_python_random

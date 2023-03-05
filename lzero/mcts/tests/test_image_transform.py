import pytest
import torch
from lzero.mcts.image_transform import Transforms


@pytest.mark.unittest
def test_image_transform():
    img = torch.rand((4, 3, 96, 96))
    transform = Transforms(['shift', 'intensity'])
    processed_img = transform.transform(img)
    assert img.shape == (4, 3, 96, 96)
    assert not (img == processed_img).all()


test_image_transform()
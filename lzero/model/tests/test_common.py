import pytest
import torch
from ding.torch_utils import is_differentiable

from lzero.model.common import RepresentationNetwork


@pytest.mark.unittest
class TestCommon:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('batch_size', [10])
    def test_representation_network(self, batch_size):
        batch = batch_size
        obs = torch.rand(batch, 1, 3, 3)
        representation_network = RepresentationNetwork(
            observation_shape=[1, 3, 3], num_res_blocks=1, num_channels=16, downsample=False
        )
        state = representation_network(obs)
        assert state.shape == torch.Size([10, 16, 3, 3])

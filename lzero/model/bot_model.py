from typing import Optional, Tuple
import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType


@MODEL_REGISTRY.register('bot_model')
class BotModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            categorical_distribution: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            representation_network: nn.Module = None,
            last_linear_layer_init_zero: bool = True,
            downsample: bool = False,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            value_head_channels: int = 16,
            policy_head_channels: int = 16,
            fc_value_layers: SequenceType = [32],
            fc_policy_layers: SequenceType = [32],
            value_support_size: int = 601,
    ):
        """
        Overview:
            The fake model for bot, which is used to test the league.
        """
        super(BotModel, self).__init__()

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

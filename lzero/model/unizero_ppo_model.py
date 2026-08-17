from typing import Optional

import torch
from ding.utils import MODEL_REGISTRY

from .common import MZNetworkOutput
from .unizero_model import UniZeroModel
from .unizero_world_models.ppo_world_model import PPOWorldModel


@MODEL_REGISTRY.register('UniZeroPPOModel')
class UniZeroPPOModel(UniZeroModel):
    """UniZero model variant that owns the PPO-only world-model extension."""

    world_model_cls = PPOWorldModel

    def initial_inference(self, obs_batch: torch.Tensor, action_batch: Optional[torch.Tensor] = None,
                          current_obs_batch: Optional[torch.Tensor] = None, start_pos: int = 0,
                          ready_env_id: Optional[list] = None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of the UniZero model, which is the first step of the UniZero model.
            This method uses the representation network to obtain the ``latent_state`` and the prediction network
            to predict the ``value`` and ``policy_logits`` of the ``latent_state``.

        Arguments:
            - obs_batch (:obj:`torch.Tensor`): The 3D image observation data.
            - action_batch (:obj:`Optional[torch.Tensor]`): The actions taken, defaults to None.
            - current_obs_batch (:obj:`Optional[torch.Tensor]`): The current observations, defaults to None.
            - start_pos (:obj:`int`): The starting position for inference, defaults to 0.

        Returns:
            - MZNetworkOutput: Contains the predicted value, reward, policy logits, and latent state.

        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
        """
        batch_size = obs_batch.size(0)
        obs_act_dict = {
            'obs': obs_batch,
            'action': action_batch,
            'current_obs': current_obs_batch,
            'ready_env_id': ready_env_id,
        }

        # Perform initial inference using the world model
        output_sequence, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict, start_pos)

        # Extract and squeeze the outputs for clarity
        latent_state = obs_token
        reward = logits_rewards
        policy_logits = logits_policy.squeeze(1)
        value = logits_value.squeeze(1)

        return MZNetworkOutput(
            value=value,
            reward=[0. for _ in range(batch_size)],  # Initialize reward to zero vector
            policy_logits=policy_logits,
            latent_state=latent_state,
            policy_features=output_sequence[:, -1],
        )

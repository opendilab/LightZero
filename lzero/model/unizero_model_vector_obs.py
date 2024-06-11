from typing import Optional

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY

from .common import MZNetworkOutput, RepresentationNetworkMLP
from .unizero_world_models.tokenizer.tokenizer import Tokenizer
from .unizero_world_models.world_model import WorldModel


@MODEL_REGISTRY.register('UniZeroModel')
class UniZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: int = 2,
            action_space_size: int = 6,
            latent_state_dim: int = 256,
            reward_support_size: int = 601,
            value_support_size: int = 601,
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = False,
            categorical_distribution: bool = True,
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            discrete_action_encoding_type: str = 'one_hot',
            norm_type: Optional[str] = 'BN',
            res_connection_in_dynamics: bool = False,
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of the network model of UniZero, which is a generalization version for 1D vector obs.
            The networks are mainly built on fully connected layers.
            The representation network is an MLP network which maps the raw observation to a latent state.
            The dynamics network is an MLP network which predicts the next latent state, and reward given the current latent state and action.
            The prediction network is an MLP network which predicts the value and policy given the current latent state.
        Arguments:
            - observation_shape (:obj:`int`): Observation space shape, e.g. 8 for Lunarlander.
            - action_space_size: (:obj:`int`): Action space size, e.g. 4 for Lunarlander.
            - latent_state_dim (:obj:`int`): The dimension of latent state, such as 256.
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks in UniZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of value/policy mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for latent states, default sets it to True.
            - discrete_action_encoding_type (:obj:`str`): The encoding type of discrete action, which can be 'one_hot' or 'not_one_hot'.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - res_connection_in_dynamics (:obj:`bool`): Whether to use residual connection for dynamics network, default set it to False.
        """
        super(UniZeroModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        if not self.categorical_distribution:
            self.reward_support_size = 1
            self.value_support_size = 1
        else:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size

        self.action_space_size = action_space_size
        self.continuous_action_space = False
        # The dim of action space. For discrete action space, it is 1.
        # For continuous action space, it is the dimension of continuous action.
        self.action_space_dim = action_space_size if self.continuous_action_space else 1
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.continuous_action_space:
            self.action_encoding_dim = action_space_size
        else:
            if self.discrete_action_encoding_type == 'one_hot':
                self.action_encoding_dim = action_space_size
            elif self.discrete_action_encoding_type == 'not_one_hot':
                self.action_encoding_dim = 1

        self.latent_state_dim = latent_state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.res_connection_in_dynamics = res_connection_in_dynamics

        self.representation_network = RepresentationNetworkMLP(
            observation_shape=observation_shape, hidden_channels=self.latent_state_dim, norm_type=norm_type
        )
        world_model_cfg = kwargs['world_model']
        self.tokenizer = Tokenizer(with_lpips=True, encoder=self.representation_network)
        self.world_model = WorldModel(act_vocab_size=self.action_space_size,
                                      config=world_model_cfg, tokenizer=self.tokenizer)
        print(f'{sum(p.numel() for p in self.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')

    def initial_inference(self, obs: torch.Tensor, action_batch=None, current_obs_batch=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of UniZero model, which is the first step of the UniZero model.
            To perform the initial inference, we first use the representation network to obtain the "latent_state" of the observation.
            Then we use the prediction network to predict the "value" and "policy_logits" of the "latent_state", and
            also prepare the zeros-like ``reward`` for the next step of the UniZero model.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 1D vector observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - value_prefix (:obj:`torch.Tensor`): The predicted prefix sum of value for input state. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The hidden state of LSTM about reward. In initial inference, \
                we set it to the zeros-like hidden state (H and C).
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, obs_shape)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        batch_size = obs.size(0)
        obs_act_dict = {'obs': obs, 'action': action_batch, 'current_obs': current_obs_batch}
        x, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict)
        latent_state, reward, policy_logits, value = obs_token, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def recurrent_inference(self, state_action_history: torch.Tensor, simulation_index=0,
                            latent_state_index_in_search_path=[]) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of UniZero model, which is the rollout step of the UniZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward`` by the given current ``latent_state`` and ``action``.
             We then use the prediction network to predict the ``value`` and ``policy_logits``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input obs.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
        """
        x, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, latent_state_index_in_search_path)
        next_latent_state, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        reward = reward.squeeze(1)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

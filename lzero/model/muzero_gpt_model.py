"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType
from numpy import ndarray

from .common import MZNetworkOutput, RepresentationNetworkGPT, PredictionNetwork, LatentDecoder
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('MuZeroModelGPT')
class MuZeroModelGPT(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        fc_reward_layers: SequenceType = [32],
        fc_value_layers: SequenceType = [32],
        fc_policy_layers: SequenceType = [32],
        reward_support_size: int = 601,
        value_support_size: int = 601,
        proj_hid: int = 1024,
        proj_out: int = 1024,
        pred_hid: int = 512,
        pred_out: int = 1024,
        self_supervised_learning_loss: bool = False,
        categorical_distribution: bool = True,
        activation: nn.Module = nn.ReLU(inplace=True),
        last_linear_layer_init_zero: bool = True,
        state_norm: bool = False,
        downsample: bool = False,
        norm_type: Optional[str] = 'BN',
        discrete_action_encoding_type: str = 'one_hot',
        *args,
        **kwargs
    ):
        """
        Overview:
            The definition of the neural network model used in MuZero.
            MuZero model which consists of a representation network, a dynamics network and a prediction network.
            The networks are built on convolution residual blocks and fully connected layers.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - fc_value_layers (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - fc_policy_layers (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
                in MuZero model, default set it to False.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value and reward.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to False.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. options = {'one_hot', 'not_one_hot'}
        """
        super(MuZeroModelGPT, self).__init__()
        if isinstance(observation_shape, int) or len(observation_shape) == 1:
            # for vector obs input, e.g. classical control and box2d environments
            # to be compatible with LightZero model/policy, transform to shape: [C, W, H]
            observation_shape = [1, observation_shape, 1]

        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size

        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = action_space_size
        elif self.discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample

        flatten_output_size_for_reward_head = (
            (reward_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (reward_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_output_size_for_value_head = (
            (value_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (value_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_output_size_for_policy_head = (
            (policy_head_channels * math.ceil(observation_shape[1] / 16) *
             math.ceil(observation_shape[2] / 16)) if downsample else
            (policy_head_channels * observation_shape[1] * observation_shape[2])
        )


        # self.dynamics_network = DynamicsNetwork(
        #     observation_shape,
        #     self.action_encoding_dim,
        #     num_res_blocks,
        #     num_channels + self.action_encoding_dim,
        #     reward_head_channels,
        #     fc_reward_layers,
        #     self.reward_support_size,
        #     flatten_output_size_for_reward_head,
        #     downsample,
        #     last_linear_layer_init_zero=self.last_linear_layer_init_zero,
        #     activation=activation,
        #     norm_type=norm_type
        # )

        from .gpt_models.world_model import WorldModel
        from .gpt_models.tokenizer.tokenizer import Tokenizer
        from .gpt_models.tokenizer.nets import Encoder, Decoder
        # from .gpt_models.cfg_cartpole import cfg
        from .gpt_models.cfg_atari import cfg

        self.representation_network = RepresentationNetworkGPT(
            observation_shape,
            num_res_blocks,
            num_channels,
            downsample,
            activation=activation,
            norm_type=norm_type,
            # embedding_dim=cfg.embedding_dim,
            embedding_dim=cfg.world_model.embed_dim,
        )
        # Instantiate the decoder
        # decoder_network = LatentDecoder(embedding_dim=cfg.world_model.embed_dim, output_shape=(4, 64, 64)) # TODO: For K=4
        decoder_network = LatentDecoder(embedding_dim=cfg.world_model.embed_dim, output_shape=(3, 64, 64)) # TODO: For K=1



        Encoder = Encoder(cfg.tokenizer.encoder)
        Decoder = Decoder(cfg.tokenizer.decoder)
        self.tokenizer = Tokenizer(cfg.tokenizer.vocab_size, cfg.tokenizer.embed_dim, Encoder, Decoder, with_lpips=True, representation_network=self.representation_network,
                            decoder_network=decoder_network)
        self.world_model = WorldModel(obs_vocab_size=self.tokenizer.vocab_size, act_vocab_size=self.action_space_size,
                                      config=cfg.world_model, tokenizer=self.tokenizer)
        print(f'{sum(p.numel() for p in self.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.tokenizer.representation_network.parameters())} parameters in agent.tokenizer.representation_network')
        print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')

        print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')


    def initial_inference(self, obs: torch.Tensor, action_batch=None, current_obs_batch=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of MuZero model, which is the first step of the MuZero model.
            To perform the initial inference, we first use the representation network to obtain the ``latent_state``.
            Then we use the prediction network to predict ``value`` and ``policy_logits`` of the ``latent_state``.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 2D image observation data.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        batch_size = obs.size(0)
        # latent_state = self._representation(obs)
        # policy_logits, value = self._prediction(latent_state)
        # return MZNetworkOutput(
        #     value,
        #     [0. for _ in range(batch_size)],
        #     policy_logits,
        #     latent_state,
        # )

        obs_act_dict = {'obs':obs, 'action':action_batch, 'current_obs':current_obs_batch}
        x, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(
            obs_act_dict)
        # x, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(
        #     obs, action_batch)
        reward, policy_logits, value = logits_rewards, logits_policy, logits_value

        # obs discrete distribution to one_hot latent state?
        # torch.Size([8, 16, 512]) -> torch.Size([8, 16])
        # latent_state = torch.argmax(logits_observations, dim=2, keepdim=False)
        latent_state = obs_token

        # torch.Size([8, 1, 2]) - > torch.Size([8, 2])
        policy_logits = policy_logits.squeeze(1)
        # torch.Size([8, 1, 601]) - > torch.Size([8, 601])
        value = value.squeeze(1)

        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    # def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
    def recurrent_inference(self, state_action_history: torch.Tensor) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of MuZero model, which is the rollout step of the MuZero model.
            To perform the recurrent inference, we first use the dynamics network to predict ``next_latent_state``,
            ``reward``, by the given current ``latent_state`` and ``action``.
            We then use the prediction network to predict the ``value`` and ``policy_logits`` of the current
            ``latent_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - next_latent_state (:obj:`torch.Tensor`): The predicted next latent state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
         """
        # next_latent_state, reward = self._dynamics(latent_state, action)
        #
        # policy_logits, value = self._prediction(next_latent_state)
        # return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

        x, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history)
        logits_observations, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value

        # obs discrete distribution to one_hot latent state?
        # torch.Size([8, 16])
        next_latent_state = logits_observations

        # torch.Size([8, 1, 2]) - > torch.Size([8, 2])
        policy_logits = policy_logits.squeeze(1)
        # torch.Size([8, 1, 601]) - > torch.Size([8, 601])
        value = value.squeeze(1)

        # torch.Size([8,]) - > torch.Size([8, 1])
        # reward = torch.tensor(reward).unsqueeze(-1) # TODO(pu)
        reward = reward.squeeze(1)

        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Use the representation network to encode the observations into latent state.
        Arguments:
            - obs (:obj:`torch.Tensor`): The 2D image observation data.
        Returns:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
        """
        latent_state = self.representation_network(observation)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        return latent_state

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the prediction network to predict ``policy_logits`` and ``value``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        return self.prediction_network(latent_state)

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            and ``reward``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - reward (:obj:`torch.Tensor`): The predicted reward of the current latent state and selected action.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        if self.discrete_action_encoding_type == 'one_hot':
            # Stack latent_state with the one hot encoded action.
            # The final action_encoding shape is (batch_size, action_space_size, latent_state[2], latent_state[3]), e.g. (8, 2, 4, 1).
            if len(action.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action = action.unsqueeze(-1)

            # transform action to one-hot encoding.
            # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
            action_one_hot = torch.zeros(action.shape[0], self.action_space_size, device=action.device)
            # transform action to torch.int64
            action = action.long()
            action_one_hot.scatter_(1, action, 1)

            action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
            action_encoding = action_encoding_tmp.expand(
                latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
            )

        elif self.discrete_action_encoding_type == 'not_one_hot':
            # Stack latent_state with the normalized encoded action.
            # The final action_encoding shape is (batch_size, 1, latent_state[2], latent_state[3]), e.g. (8, 1, 4, 1).
            if len(action.shape) == 2:
                # (batch_size, action_dim=1) -> (batch_size, 1, 1, 1)
                # e.g.,  torch.Size([8, 1]) ->  torch.Size([8, 1, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1)
            elif len(action.shape) == 1:
                # (batch_size,) -> (batch_size, 1, 1, 1)
                # e.g.,  -> torch.Size([8, 1, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            action_encoding = action.expand(
                latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]
            ) / self.action_space_size

        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim, latent_state[2], latent_state[3]) or
        # (batch_size, latent_state[1] + action_space_size, latent_state[2], latent_state[3]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)
        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, reward

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Project the latent state to a lower dimension to calculate the self-supervised loss, which is involved in
            MuZero algorithm in EfficientZero.
            For more details, please refer to the paper ``Exploring Simple Siamese Representation Learning``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - with_grad (:obj:`bool`): Whether to calculate gradient for the projection result.
        Returns:
            - proj (:obj:`torch.Tensor`): The result embedding vector of projection operation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - proj (:obj:`torch.Tensor`): :math:`(B, projection_output_dim)`, where B is batch_size.

        Examples:
            >>> latent_state = torch.randn(256, 64, 6, 6)
            >>> output = self.project(latent_state)
            >>> output.shape # (256, 1024)

        .. note::
            for Atari:
            observation_shape = (12, 96, 96),  # original shape is (3,96,96), frame_stack_num=4
            if downsample is True, latent_state.shape: (batch_size, num_channel, obs_shape[1] / 16, obs_shape[2] / 16)
            i.e., (256, 64, 96 / 16, 96 / 16) = (256, 64, 6, 6)
            latent_state reshape: (256, 64, 6, 6) -> (256,64*6*6) = (256, 2304)
            # self.projection_input_dim = 64*6*6 = 2304
            # self.projection_output_dim = 1024
        """
        latent_state = latent_state.reshape(latent_state.shape[0], -1)
        proj = self.projection(latent_state)

        if with_grad:
            # with grad, use prediction_head
            return self.prediction_head(proj)
        else:
            return proj.detach()

    def get_params_mean(self) -> float:
        return get_params_mean(self)


class DynamicsNetwork(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType,
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        fc_reward_layers: SequenceType = [32],
        output_support_size: int = 601,
        flatten_output_size_for_reward_head: int = 64,
        downsample: bool = False,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
    ):
        """
        Overview:
            The definition of dynamics network in MuZero algorithm, which is used to predict next latent state and
            reward given current latent state and action.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of input observation, e.g., (12, 96, 96).
            - action_encoding_dim (:obj:`int`): The dimension of action encoding.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of input, including obs and action encoding.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - fc_reward_layers (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - flatten_output_size_for_reward_head (:obj:`int`): The flatten size of output for reward head, i.e., \
                the input size of reward head.
            - downsample (:obj:`bool`): Whether to downsample the input observation, default set it to False.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of \
                reward mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        assert num_channels > action_encoding_dim, f'num_channels:{num_channels} <= action_encoding_dim:{action_encoding_dim}'

        self.num_channels = num_channels
        self.flatten_output_size_for_reward_head = flatten_output_size_for_reward_head
        self.action_encoding_dim = action_encoding_dim

        self.conv = nn.Conv2d(num_channels, num_channels - self.action_encoding_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        if norm_type == 'BN':
            self.norm_common = nn.BatchNorm2d(num_channels - self.action_encoding_dim)
        elif norm_type == 'LN':
            if downsample:
                self.norm_common = nn.LayerNorm([num_channels - self.action_encoding_dim, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_common = nn.LayerNorm([num_channels - self.action_encoding_dim, observation_shape[-2], observation_shape[-1]])
            
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels - self.action_encoding_dim, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - self.action_encoding_dim, reward_head_channels, 1)

        if norm_type == 'BN':
            self.norm_reward = nn.BatchNorm2d(reward_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_reward = nn.LayerNorm([reward_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)])
            else:
                self.norm_reward = nn.LayerNorm([reward_head_channels, observation_shape[-2], observation_shape[-1]])

        self.fc_reward_head = MLP(
            self.flatten_output_size_for_reward_head,
            hidden_channels=fc_reward_layers[0],
            layer_num=len(fc_reward_layers) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         Overview:
            Forward computation of the dynamics network. Predict the next latent state given current latent state and action.
         Arguments:
             - state_action_encoding (:obj:`torch.Tensor`): The state-action encoding, which is the concatenation of \
                    latent state and action encoding, with shape (batch_size, num_channels, height, width).
         Returns:
             - next_latent_state (:obj:`torch.Tensor`): The next latent state, with shape (batch_size, num_channels, \
                    height, width).
            - reward (:obj:`torch.Tensor`): The predicted reward, with shape (batch_size, output_support_size).
         """
        # take the state encoding, state_action_encoding[:, -self.action_encoding_dim:, :, :] is action encoding
        state_encoding = state_action_encoding[:, :-self.action_encoding_dim:, :, :]
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)

        # the residual link: add state encoding to the state_action encoding
        x += state_encoding
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        next_latent_state = x

        x = self.conv1x1_reward(next_latent_state)
        x = self.norm_reward(x)
        x = self.activation(x)
        x = x.view(-1, self.flatten_output_size_for_reward_head)

        # use the fully connected layer to predict reward
        reward = self.fc_reward_head(x)

        return next_latent_state, reward

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        return get_reward_mean(self)

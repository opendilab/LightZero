from typing import Optional, List

import torch
import torch.nn as nn
from ding.torch_utils import MLP
from ding.utils import MODEL_REGISTRY, SequenceType
from easydict import EasyDict

from .common import MZNetworkOutput, RepresentationNetworkUniZero, LatentDecoder, \
    FeatureAndGradientHook, SimNorm
from .unizero_world_models.tokenizer import Tokenizer
from .unizero_world_models.world_model_multitask import WorldModelMT


class RepresentationNetworkMLPMT(nn.Module):
    def __init__(
            self,
            observation_shapes: List[int],  # List of observation shapes for each task
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: Optional[str] = 'BN',
            group_size: int = 8,
    ) -> torch.Tensor:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the vector obs into latent state \
            with Multi-Layer Perceptron (MLP).
        Arguments:
            - observation_shapes (:obj:`List[int]`): The list of observation shape for each task.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - layer_num (:obj:`int`): The number of layers in the MLP.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh').
            - norm_type (:obj:`str`): The type of normalization in networks, defaults to 'BN'.
            - group_size (:obj:`int`): The group size used in SimNorm.
        """
        super().__init__()
        self.env_num = len(observation_shapes)
        self.fc_representation = nn.ModuleList([
            MLP(
                in_channels=obs_shape,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                layer_num=layer_num,
                activation=activation,
                norm_type=norm_type,
                # don't use activation and norm in the last layer of representation network is important for convergence.
                output_activation=False,
                output_norm=False,
                # last_linear_layer_init_zero=True is beneficial for convergence speed.
                last_linear_layer_init_zero=True,
            )
            for obs_shape in observation_shapes
        ])
        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - task_id (:obj:`int`): The ID of the current task.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        x = self.fc_representation[task_id](x)
        x = self.sim_norm(x)
        return x


@MODEL_REGISTRY.register('SampledUniZeroMTModel')
class SampledUniZeroMTModel(nn.Module):
    def __init__(
            self,
            observation_shapes: List[SequenceType],  # List of observation shapes for each task
            action_space_sizes: List[int],  # List of action space sizes for each task
            num_res_blocks: int = 1,
            num_channels: int = 64,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            downsample: bool = True,
            norm_type: Optional[str] = 'LN',
            # world_model_cfgs: List[EasyDict] = None,  # List of world model configs for each task
            world_model_cfg: List[EasyDict] = None,  # List of world model configs for each task
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of data procession in the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667), including two main parts:
                - initial_inference, which is used to predict the value, policy, and latent state based on the current observation.
                - recurrent_inference, which is used to predict the value, policy, reward, and next latent state based on the current latent state and action.
            The world model consists of three main components:
                - a tokenizer, which encodes observations into embeddings,
                - a transformer, which processes the input sequences,
                - and heads, which generate the logits for observations, rewards, policy, and value.
        Arguments:
            - observation_shapes (:obj:`List[SequenceType]`): List of observation space shapes for each task, e.g. [C, W, H]=[3, 64, 64] for Atari.
            - action_space_sizes (:obj:`List[int]`): List of action space sizes for each task.
            - num_res_blocks (:obj:`int`): The number of res blocks in UniZero model.
            - num_channels (:obj:`int`): The channels of hidden states in representation network.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - norm_type (:obj=`str`): The type of normalization in networks. Defaults to 'LN'.
            - world_model_cfgs (:obj=`List[EasyDict]`): The list of world model configurations for each task.
        """
        super(SampledUniZeroMTModel, self).__init__()
        self.task_num = len(observation_shapes)
        self.activation = activation
        self.downsample = downsample

        # Initialize environment-specific networks and models
        self.representation_networks = nn.ModuleList()
        # self.decoder_networks = nn.ModuleList()
        # self.world_models = nn.ModuleList()

        for task_id in range(self.task_num):
            # world_model_cfg = world_model_cfgs[task_id]
            world_model_cfg.norm_type = norm_type
            assert world_model_cfg.max_tokens == 2 * world_model_cfg.max_blocks, 'max_tokens should be 2 * max_blocks, because each timestep has 2 tokens: obs and action'

            if world_model_cfg.obs_type == 'vector':
                self.representation_network = RepresentationNetworkMLPMT(
                    observation_shapes=observation_shapes,
                    hidden_channels=world_model_cfg.embed_dim,
                    layer_num=2,
                    activation=self.activation,
                    norm_type=norm_type,
                    group_size=world_model_cfg.group_size,
                )
                self.tokenizer = Tokenizer(encoder=self.representation_network,
                                      decoder_network=None, with_lpips=False)
                self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=self.tokenizer)
            elif world_model_cfg.obs_type == 'image':
                self.representation_network = RepresentationNetworkUniZero(
                    observation_shapes[task_id],
                    num_res_blocks,
                    num_channels,
                    self.downsample,
                    activation=self.activation,
                    norm_type=norm_type,
                    embedding_dim=world_model_cfg.embed_dim,
                    group_size=world_model_cfg.group_size,
                )

                # ====== for analysis ======
                if world_model_cfg.analysis_sim_norm:
                    self.encoder_hook = FeatureAndGradientHook()
                    self.encoder_hook.setup_hooks(self.representation_network)

                tokenizer = Tokenizer(encoder=self.representation_network,
                                      decoder_network=None, with_lpips=False)
                self.world_model = WorldModelMT(config=world_model_cfg, tokenizer=tokenizer)

            # Print model parameters for debugging
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)

    def initial_inference(self, obs_batch: torch.Tensor, action_batch=None, current_obs_batch=None, task_id=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of UniZero model, which is the first step of the UniZero model.
            To perform the initial inference, we first use the representation network to obtain the ``latent_state``.
            Then we use the prediction network to predict ``value`` and ``policy_logits`` of the ``latent_state``.
        Arguments:
            - obs_batch (:obj:`torch.Tensor`): The 3D image observation data.
            - task_id (:obj:`int`): The ID of the current task.
        Returns (MZNetworkOutput):
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj:`torch.Tensor`): The predicted reward of input state and selected action. \
                In initial inference, we set it to zero vector.
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj=`torch.Tensor`): The encoding latent state of input state.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - value (:obj=`torch.Tensor`): :math=`(B, value_support_size)`, where B is batch_size.
            - reward (:obj=`torch.Tensor`): :math=`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj=`torch.Tensor`): :math=`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj=`torch.Tensor`): :math=`(B, H_, W_)`, where B is batch_size, H_ is the height of latent state, W_ is the width of latent state.
        """
        batch_size = obs_batch.size(0)
        obs_act_dict = {'obs': obs_batch, 'action': action_batch, 'current_obs': current_obs_batch}
        _, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict, task_id=task_id)
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
                            latent_state_index_in_search_path=[], task_id=0) -> MZNetworkOutput:
        """
        Overview:
            Recurrent inference of UniZero model. To perform the recurrent inference, we concurrently predict the latent dynamics (reward/next_latent_state)
            and decision-oriented quantities (value/policy) conditioned on the learned latent history in the world_model.
        Arguments:
            - state_action_history (:obj:`torch.Tensor`): The history of states and actions.
            - task_id (:obj:`int`): The ID of the current task.
            - simulation_index (:obj=`int`): The index of the current simulation.
            - latent_state_index_in_search_path (:obj=`List[int]`): The indices of latent states in the search path.
        Returns (MZNetworkOutput):
            - value (:obj=`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
            - reward (:obj=`torch.Tensor`): The predicted reward of input state and selected action.
            - policy_logits (:obj=`torch.Tensor`): The output logit to select discrete action.
            - latent_state (:obj=`torch.Tensor`): The encoding latent state of input state.
            - next_latent_state (:obj=`torch.Tensor`): The predicted next latent state.
        Shapes:
            - obs (:obj=`torch.Tensor`): :math=`(B, num_channel, obs_shape[1], obs_shape[2])`, where B is batch_size.
            - action (:obj=`torch.Tensor`): :math=`(B, )`, where B is batch_size.
            - value (:obj=`torch.Tensor`): :math=`(B, value_support_size)`, where B is batch_size.
            - reward (:obj=`torch.Tensor`): :math=`(B, reward_support_size)`, where B is batch_size.
            - policy_logits (:obj=`torch.Tensor`): :math=`(B, action_dim)`, where B is batch_size.
            - latent_state (:obj=`torch.Tensor`): :math=`(B, H_, W_)`, where B is batch_size, H_ is the height of latent state, W_ is the width of latent state.
            - next_latent_state (:obj=`torch.Tensor`): :math=`(B, H_, W_)`, where B is batch_size, H_ is the height of latent state, W_ is the width of latent state.
         """
        _, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, latent_state_index_in_search_path, task_id=task_id)
        next_latent_state, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        reward = reward.squeeze(1)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
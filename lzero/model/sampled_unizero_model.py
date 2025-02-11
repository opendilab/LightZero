from typing import Optional

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from easydict import EasyDict

from .common import MZNetworkOutput, RepresentationNetworkUniZero, RepresentationNetworkMLP, LatentDecoder, \
    VectorDecoderForMemoryEnv, LatentEncoderForMemoryEnv, LatentDecoderForMemoryEnv, FeatureAndGradientHook
from .unizero_world_models.tokenizer import Tokenizer
from .unizero_world_models.world_model import WorldModel


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('SampledUniZeroModel')
class SampledUniZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (4, 64, 64),
            action_space_size: int = 6,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            downsample: bool = True,
            norm_type: Optional[str] = 'LN',
            world_model_cfg: EasyDict = None,
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
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[3, 64, 64] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in UniZero model.
            - num_channels (:obj:`int`): The channels of hidden states in representation network.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - world_model_cfg (:obj:`EasyDict`): The configuration of the world model, including the following keys:
                - obs_type (:obj:`str`): The type of observation, which can be 'image', 'vector', or 'image_memory'.
                - embed_dim (:obj:`int`): The dimension of the embedding.
                - group_size (:obj:`int`): The group size of the transformer.
                - max_blocks (:obj:`int`): The maximum number of blocks in the transformer.
                - max_tokens (:obj:`int`): The maximum number of tokens in the transformer.
                - context_length (:obj:`int`): The context length of the transformer.
                - device (:obj:`str`): The device of the model, which can be 'cuda' or 'cpu'.
                - action_space_size (:obj:`int`): The shape of the action.
                - num_layers (:obj:`int`): The number of layers in the transformer.
                - num_heads (:obj:`int`): The number of heads in the transformer.
                - policy_entropy_weight (:obj:`float`): The weight of the policy entropy.
                - analysis_sim_norm (:obj:`bool`): Whether to analyze the similarity of the norm.
        """
        super(SampledUniZeroModel, self).__init__()
        self.action_space_size = action_space_size
        self.activation = activation
        self.downsample = downsample
        world_model_cfg.norm_type = norm_type
        assert world_model_cfg.max_tokens == 2 * world_model_cfg.max_blocks, 'max_tokens should be 2 * max_blocks, because each timestep has 2 tokens: obs and action'

        if world_model_cfg.obs_type == 'vector':
            self.representation_network = RepresentationNetworkMLP(
                observation_shape,
                hidden_channels=world_model_cfg.embed_dim,
                layer_num=2,
                activation=self.activation,
                norm_type=norm_type,
                group_size=world_model_cfg.group_size,
            )
            # TODO: only for MemoryEnv now
            self.decoder_network = VectorDecoderForMemoryEnv(embedding_dim=world_model_cfg.embed_dim, output_shape=25, norm_type=norm_type)
            self.tokenizer = Tokenizer(encoder=self.representation_network,
                                       decoder_network=self.decoder_network, with_lpips=False)
            self.world_model = WorldModel(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image':
            self.representation_network = RepresentationNetworkUniZero(
                observation_shape,
                num_res_blocks,
                num_channels,
                self.downsample,
                activation=self.activation,
                norm_type=norm_type,
                embedding_dim=world_model_cfg.embed_dim,
                group_size=world_model_cfg.group_size,
            )
            # TODO: we should change the output_shape to the real observation shape
            self.decoder_network = LatentDecoder(embedding_dim=world_model_cfg.embed_dim, output_shape=(3, 64, 64))

            # ====== for analysis ======
            if world_model_cfg.analysis_sim_norm:
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.tokenizer = Tokenizer(encoder=self.representation_network,
                                       decoder_network=self.decoder_network, with_lpips=True,)
            self.world_model = WorldModel(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print(f'{sum(p.numel() for p in self.world_model.parameters()) - sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.world_model - (decoder_network and lpips)')

            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image_memory':
            self.representation_network = LatentEncoderForMemoryEnv(
                image_shape=(3, 5, 5),
                embedding_size=world_model_cfg.embed_dim,
                channels=[16, 32, 64],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
                activation=self.activation,
                group_size=world_model_cfg.group_size,
            )
            self.decoder_network = LatentDecoderForMemoryEnv(
                image_shape=(3, 5, 5),
                embedding_size=world_model_cfg.embed_dim,
                channels=[64, 32, 16],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
                activation=self.activation,
            )

            if world_model_cfg.analysis_sim_norm:
                # ====== for analysis ======
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.tokenizer = Tokenizer(with_lpips=True, encoder=self.representation_network,
                                       decoder_network=self.decoder_network)
            self.world_model = WorldModel(config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print(f'{sum(p.numel() for p in self.world_model.parameters()) - sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.world_model - (decoder_network and lpips)')

            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')
            print('==' * 20)

    def initial_inference(self, obs_batch: torch.Tensor, action_batch=None, current_obs_batch=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of UniZero model, which is the first step of the UniZero model.
            To perform the initial inference, we first use the representation network to obtain the ``latent_state``.
            Then we use the prediction network to predict ``value`` and ``policy_logits`` of the ``latent_state``.
        Arguments:
            - obs_batch (:obj:`torch.Tensor`): The 3D image observation data.
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
        batch_size = obs_batch.size(0)
        obs_act_dict = {'obs': obs_batch, 'action': action_batch, 'current_obs': current_obs_batch}
        _, obs_token, logits_rewards, logits_policy, logits_value = self.world_model.forward_initial_inference(obs_act_dict)
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
            Recurrent inference of UniZero model.To perform the recurrent inference, we concurrently predict the latent dynamics (reward/next_latent_state)
            and decision-oriented quantities (value/policy) conditioned on the learned latent history in the world_model.
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
        _, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, latent_state_index_in_search_path)
        next_latent_state, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        reward = reward.squeeze(1)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
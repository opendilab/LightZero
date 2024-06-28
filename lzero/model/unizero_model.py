from typing import Optional

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType

from .common import MZNetworkOutput, RepresentationNetworkGPT, RepresentationNetworkMLP, LatentDecoder, \
    VectorDecoderMemory, LatentEncoderMemory, LatentDecoderMemory, FeatureAndGradientHook
from .unizero_world_models.tokenizer.tokenizer import Tokenizer
from .unizero_world_models.world_model import WorldModel


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('UniZeroModel')
class UniZeroModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (12, 96, 96),
            action_space_size: int = 6,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            reward_support_size: int = 601,
            value_support_size: int = 601,
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = False,
            categorical_distribution: bool = True,
            activation: nn.Module = nn.GELU(),
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            downsample: bool = False,
            norm_type: Optional[str] = 'BN',
            discrete_action_encoding_type: str = 'one_hot',
            env_name='atari',
            *args,
            **kwargs
    ):
        """
        Overview:
            The definition of the neural network model used in UniZero.
            UniZero model which consists of a representation network, a dynamics network and a prediction network.
            The networks are built on convolution residual blocks and fully connected layers.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - reward_support_size (:obj:`int`): The size of categorical reward output
            - value_support_size (:obj:`int`): The size of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
                in UniZero model, default set it to False.
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
        super(UniZeroModel, self).__init__()

        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = reward_support_size
            self.value_support_size = value_support_size
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size
        self.activation = activation
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
        self.env_name = env_name

        world_model_cfg = kwargs['world_model']
        if world_model_cfg.obs_type == 'vector':
            self.representation_network = RepresentationNetworkMLP(
                observation_shape,
                hidden_channels=world_model_cfg.embed_dim,
                layer_num=2,
                activation=self.activation,
                group_size=world_model_cfg.group_size,
            )
            # TODO: only for visualmatch now
            decoder_network = VectorDecoderMemory(embedding_dim=world_model_cfg.embed_dim, output_shape=25)
            self.tokenizer = Tokenizer(encoder=self.representation_network,
                                       decoder_network=decoder_network, with_lpips=False)
            self.world_model = WorldModel(act_vocab_size=self.action_space_size,
                                          config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image':
            self.representation_network = RepresentationNetworkGPT(
                observation_shape,
                num_res_blocks,
                num_channels,
                downsample,
                # activation=self.activation,
                activation=nn.LeakyReLU(negative_slope=0.01),  # TODO: LN+LeakyReLU ========
                norm_type=norm_type,
                embedding_dim=world_model_cfg.embed_dim,
                group_size=world_model_cfg.group_size,
            )
            # Instantiate the decoder
            # TODO: we should change the output_shape to the real observation shape
            decoder_network = LatentDecoder(embedding_dim=world_model_cfg.embed_dim, output_shape=(3, 64, 64))

            # ====== for analysis ======
            if world_model_cfg.analysis_sim_norm:
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.tokenizer = Tokenizer(encoder=self.representation_network,
                                       decoder_network=decoder_network, with_lpips=True,)
            self.world_model = WorldModel(act_vocab_size=self.action_space_size,
                                          config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print(f'{sum(p.numel() for p in self.world_model.parameters()) - sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.world_model - (decoder_network and lpips)')

            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')
            print(f'{sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.tokenizer.lpips')
            print('==' * 20)
        elif world_model_cfg.obs_type == 'image_memory':
            # bigger encoder/decoder
            self.representation_network = LatentEncoderMemory(
                image_shape=(3, 5, 5),
                embedding_size=world_model_cfg.embed_dim,
                channels=[16, 32, 64],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
                activation=self.activation,
                group_size=world_model_cfg.group_size,
            )
            # Instantiate the decoder
            decoder_network = LatentDecoderMemory(
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
                                       decoder_network=decoder_network)
            self.world_model = WorldModel(act_vocab_size=self.action_space_size,
                                          config=world_model_cfg, tokenizer=self.tokenizer)
            print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in agent.world_model')
            print(f'{sum(p.numel() for p in self.world_model.parameters()) - sum(p.numel() for p in self.tokenizer.decoder_network.parameters()) - sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.world_model - (decoder_network and lpips)')

            print('==' * 20)
            print(f'{sum(p.numel() for p in self.world_model.transformer.parameters())} parameters in agent.world_model.transformer')
            print(f'{sum(p.numel() for p in self.tokenizer.encoder.parameters())} parameters in agent.tokenizer.encoder')
            print(f'{sum(p.numel() for p in self.tokenizer.decoder_network.parameters())} parameters in agent.tokenizer.decoder_network')
            print(f'{sum(p.numel() for p in self.tokenizer.lpips.parameters())} parameters in agent.tokenizer.lpips')
            print('==' * 20)

    def initial_inference(self, obs: torch.Tensor, action_batch=None, current_obs_batch=None) -> MZNetworkOutput:
        """
        Overview:
            Initial inference of UniZero model, which is the first step of the UniZero model.
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
        x, logits_observations, logits_rewards, logits_policy, logits_value = self.world_model.forward_recurrent_inference(
            state_action_history, simulation_index, latent_state_index_in_search_path)
        next_latent_state, reward, policy_logits, value = logits_observations, logits_rewards, logits_policy, logits_value
        policy_logits = policy_logits.squeeze(1)
        value = value.squeeze(1)
        reward = reward.squeeze(1)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)
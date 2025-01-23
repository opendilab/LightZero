from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType
from numpy import ndarray

from .common import MZNetworkOutput, RepresentationNetwork, PredictionNetwork, FeatureAndGradientHook
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean


@MODEL_REGISTRY.register('MuZeroMTModel')
class MuZeroMTModel(nn.Module):

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
        analysis_sim_norm: bool = False,
        task_num: int = 1,  # 任务数量
        *args,
        **kwargs
    ):
        """
        多任务MuZero模型的定义，继承自MuZeroModel。
        增加了多任务相关的处理，如任务数量和动作空间大小调整。
        """
        super(MuZeroMTModel, self).__init__()
        
        print(f'==========MuZeroMTModel, num_res_blocks:{num_res_blocks}, num_channels:{num_channels}, task_num:{task_num}===========')

        if discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = action_space_size
        elif discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1

        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type

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

        self.task_num = task_num
        self.action_space_size = 18  # 假设每个任务的动作空间相同

        self.categorical_distribution = categorical_distribution

        self.discrete_action_encoding_type = 'one_hot'

        # 共享表示网络
        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_res_blocks,
            num_channels,
            downsample,
            activation=activation,
            norm_type=norm_type
        )

        # ====== for analysis ======
        if analysis_sim_norm:
            self.encoder_hook = FeatureAndGradientHook()
            self.encoder_hook.setup_hooks(self.representation_network)

        # 共享动态网络
        self.dynamics_network = DynamicsNetwork(
            observation_shape,
            action_encoding_dim=self.action_encoding_dim,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels + self.action_encoding_dim,
            reward_head_channels=reward_head_channels,
            fc_reward_layers=fc_reward_layers,
            output_support_size=reward_support_size,
            flatten_output_size_for_reward_head=reward_head_channels * self._get_latent_size(observation_shape, downsample),
            downsample=downsample,
            last_linear_layer_init_zero=last_linear_layer_init_zero,
            activation=activation,
            norm_type=norm_type
        )

        # 独立的预测网络，每个任务一个
        # 计算flatten_output_size
        value_flatten_size = int(value_head_channels * self._get_latent_size(observation_shape, downsample))
        policy_flatten_size = int(policy_head_channels * self._get_latent_size(observation_shape, downsample))
        
        self.prediction_networks = nn.ModuleList([
            PredictionNetwork(
                observation_shape,
                action_space_size,
                num_res_blocks,
                num_channels,
                value_head_channels,
                policy_head_channels,
                fc_value_layers,
                fc_policy_layers,
                self.value_support_size,
                flatten_output_size_for_value_head=value_flatten_size,
                flatten_output_size_for_policy_head=policy_flatten_size,
                downsample=downsample,
                last_linear_layer_init_zero=last_linear_layer_init_zero,
                activation=activation,
                norm_type=norm_type
            ) for _ in range(task_num)
        ])

        # 共享投影和预测头（如果使用自监督学习损失）
        if self_supervised_learning_loss:
            self.projection_network = nn.Sequential(
                nn.Linear(num_channels * self._get_latent_size(observation_shape, downsample), proj_hid),
                nn.BatchNorm1d(proj_hid),
                activation,
                nn.Linear(proj_hid, proj_hid),
                nn.BatchNorm1d(proj_hid),
                activation,
                nn.Linear(proj_hid, proj_out),
                nn.BatchNorm1d(proj_out)
            )

            self.prediction_head = nn.Sequential(
                nn.Linear(proj_out, pred_hid),
                nn.BatchNorm1d(pred_hid),
                activation,
                nn.Linear(pred_hid, pred_out),
            )

        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.state_norm = state_norm
        self.downsample = downsample

    def _get_latent_size(self, observation_shape: SequenceType, downsample: bool) -> int:
        """
        辅助函数，根据观测形状和下采样选项计算潜在状态的大小。
        """
        if downsample:
            return math.ceil(observation_shape[-2] / 16) * math.ceil(observation_shape[-1] / 16)
        else:
            return observation_shape[-2] * observation_shape[-1]

    def initial_inference(self, obs: torch.Tensor, task_id: int = 0) -> MZNetworkOutput:
        """
        多任务初始推理，基于任务ID选择对应的预测网络。
        """
        batch_size = obs.size(0)
        latent_state = self.representation_network(obs)
        if self.state_norm:
            latent_state = renormalize(latent_state)
        prediction_net = self.prediction_networks[task_id]
        policy_logits, value = prediction_net(latent_state)

        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor, task_id: int = 0) -> MZNetworkOutput:
        """
        多任务递归推理，根据任务ID选择对应的预测网络。
        """
        next_latent_state, reward = self._dynamics(latent_state, action)

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        prediction_net = self.prediction_networks[task_id]
        policy_logits, value = prediction_net(next_latent_state)

        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)


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
        多任务投影方法，当前实现为共享投影网络。
        """
        if not self.self_supervised_learning_loss:
            raise NotImplementedError("Self-supervised learning loss is not enabled for this model.")
        
        latent_state = latent_state.reshape(latent_state.shape[0], -1)
        proj = self.projection_network(latent_state)
        if with_grad:
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
        DynamicsNetwork定义，适用于多任务共享。
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must be in ['BN', 'LN']"
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
        DynamicsNetwork的前向传播，预测下一个潜在状态和奖励。
        """
        # 提取状态编码（去除动作编码部分）
        state_encoding = state_action_encoding[:, :-self.action_encoding_dim, :, :]
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)

        # 残差连接
        x += state_encoding
        x = self.activation(x)

        for block in self.resblocks:
            x = block(x)
        next_latent_state = x

        x = self.conv1x1_reward(next_latent_state)
        x = self.norm_reward(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)

        # 使用全连接层预测奖励
        reward = self.fc_reward_head(x)

        return next_latent_state, reward

    def get_dynamic_mean(self) -> float:
        return get_dynamic_mean(self)

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        return get_reward_mean(self)
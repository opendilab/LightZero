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
import math
from typing import Sequence, Tuple, List
import torch.nn.init as init

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MZNetworkOutput, PredictionNetwork, FeatureAndGradientHook, MLP_V2, DownSample, SimNorm
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean
from lzero.model.muzero_model import MuZeroModel
from lzero.model.muzero_model_mlp import DynamicsNetworkVector, PredictionNetworkMLP


class RepresentationNetworkMemoryEnv(nn.Module):
    def __init__(
        self,
        image_shape: Sequence = (3, 5, 5),  # 单步输入 shape，每一步的 channel 为 image_shape[0]
        embedding_size: int = 100,
        channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [3, 3, 3],
        strides: List[int] = [1, 1, 1],
        activation: nn.Module = nn.GELU(approximate='tanh'),
        normalize_pixel: bool = False,
        group_size: int = 8,
        fusion_mode: str = 'mean',  # 当前仅支持均值融合，后续可扩展为其它融合方式
        **kwargs,
    ):
        """
        表征网络，用于 MemoryEnv，将2D图像 obs 编码为 latent state，并支持对多历史步进行融合。
        除了对单步图像进行编码（如 image_shape 为 (3, 5, 5)），本网络扩展为：
          1. 根据输入通道数（total_channels）与单步输入通道数（image_shape[0]）的比值，划分为多个历史步，
             即输入 x 的 shape 为 [B, total_channels, W, H]，其中 total_channels 应为 (history_length * image_shape[0])。
          2. 分别编码每一步，输出 latent series，形状为 [B, history_length, embedding_size]。
          3. 根据 fusion_mode 对 history_length 个 latent 进行融合，得到最终 latent state，形状为 [B, embedding_size]。
        """
        super(RepresentationNetworkMemoryEnv, self).__init__()
        self.image_shape = image_shape
        self.single_step_in_channels = image_shape[0]
        self.embedding_size = embedding_size
        self.normalize_pixel = normalize_pixel
        self.fusion_mode = fusion_mode

        # 构建单步 CNN encoder 网络（和 LatentEncoderForMemoryEnv 保持一致的基本结构）
        self.channels = [image_shape[0]] + list(channels)
        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=kernel_sizes[i] // 2,  # 保持 feature map 大小不变
                )
            )
            layers.append(nn.BatchNorm2d(self.channels[i + 1]))
            layers.append(activation)
        # 自适应池化，输出形状固定为 1x1
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*layers)

        # 全连接层将 CNN 输出转为 embedding 表征
        self.linear = nn.Linear(self.channels[-1], embedding_size, bias=False)
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')

        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        对单步输入进行编码：
          x: [B, single_step_in_channels, W, H]
          返回: [B, embedding_size]
        """
        if self.normalize_pixel:
            x = x / 255.0
        x = self.cnn(x.float())  # 输出形状 (B, C, 1, 1)
        # import ipdb;ipdb.set_trace()

        x = torch.flatten(x, start_dim=1)  # 转换为形状 (B, C)
        x = self.linear(x)  # (B, embedding_size)
        x = self.sim_norm(x)  # 归一化处理
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 tensor，形状 [B, total_channels, W, H]，其中 total_channels 应为 (history_length * single_step_in_channels)
               例如输入 shape 可为: [8, 12, 5, 5]，其中 12 = 4 * 3，表示 4 个历史步，每步 3 个 channel。
        Returns:
            latent_series: 各步的 latent 表征，形状为 [B, history_length, embedding_size]
            latent_fused: 融合后的 latent 表征，形状为 [B, embedding_size]
        """
        B, total_channels, W, H = x.shape
        if total_channels % self.single_step_in_channels != 0:
            raise ValueError(
                f"总通道数 {total_channels} 不能整除单步通道数 {self.single_step_in_channels}"
            )
        history_length = total_channels // self.single_step_in_channels

        latent_series = []
        for t in range(history_length):
            # 取第 t 个历史步的数据
            x_t = x[:, t * self.single_step_in_channels:(t + 1) * self.single_step_in_channels, :, :]
            latent_t = self.forward_single(x_t)  # [B, embedding_size]
            latent_series.append(latent_t.unsqueeze(1))  # 在时间维度上扩展

        latent_series = torch.cat(latent_series, dim=1)  # [B, history_length, embedding_size]


        # 根据 fusion_mode 对所有历史步进行融合
        if self.fusion_mode == 'mean':
            latent_fused = latent_series.mean(dim=1)
            # import ipdb;ipdb.set_trace()
        else:
            # 其它融合方式：例如先拼接后通过全连接层融合
            B, T, E = latent_series.shape
            latent_concat = latent_series.view(B, -1)  # [B, T * E]
            fusion_fc = nn.Linear(T * E, E).to(x.device)
            latent_fused = fusion_fc(latent_concat)

        # return latent_series, latent_fused
        return latent_fused


# 修改后的扩展版本的 RepresentationNetwork
class RepresentationNetwork(nn.Module):
    def __init__(
            self,
            observation_shape: Sequence = (3, 64, 64),  # 单步输入 shape, 每一步3个channel
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            use_sim_norm: bool = False,
            fusion_mode: str = 'mean',  # 可以扩展为其它融合方式
    ) -> None:
        """
        表征网络，将2D图像 obs 编码为 latent state。

        除了本来的单步编码（例如 obs_with_history[:,:3,:,:]），该网络扩展为：
         1. 根据输入的第二维（通道维度）划分为多个历史步（每步3个 channel）。
         2. 分别计算每一步的 latent state，输出 shape 为 [B, T, num_channels, H_out, W_out]。
         3. 将 T 步的信息融合（例如均值融合）得到最终 latent state，其 shape 为 [B, num_channels, H_out, W_out]。
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        # 这里单步输入channels为 observation_shape[0]，一般设置为 3
        self.single_step_in_channels = observation_shape[0]
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv2d(self.single_step_in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]], eps=1e-5)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation
        self.use_sim_norm = use_sim_norm

        if self.use_sim_norm:
            self.embedding_dim = embedding_dim
            self.sim_norm = SimNorm(simnorm_dim=group_size)

        # 融合模式，当前仅支持均值融合；可以扩展为其它方式，例如使用 1D 卷积融合时间步信息
        self.fusion_mode = fusion_mode

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理单步输入：
          x: [B, single_step_in_channels, W, H]
          返回: [B, num_channels, W_out, H_out]
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)

        for block in self.resblocks:
            x = block(x)

        if self.use_sim_norm:
            x = self.sim_norm(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 tensor，shape 为 [B, total_channels, W, H]，其中 total_channels 应为 (history_length * single_step_in_channels)。
               例如 collect/eval 阶段的输入 shape: [8, 12, 64, 64]，其中 12 = 4 * 3，表示 4 个历史步，每步 3 个 channel。

        Returns:
            latent_series: 各步的 latent state，shape 为 [B, history_length, num_channels, W_out, H_out]
            latent_fused: 融合后的 latent state，shape 为 [B, num_channels, W_out, H_out]
        """
        B, total_channels, W, H = x.shape
        assert total_channels % self.single_step_in_channels == 0, (
            f"Total channels {total_channels} 不能整除单步通道数 {self.single_step_in_channels}"
        )
        history_length = total_channels // self.single_step_in_channels

        latent_series = []
        for t in range(history_length):
            # 对应第 t 步的数据：取第 t*channel 到 (t+1)*channel
            x_t = x[:, t * self.single_step_in_channels:(t + 1) * self.single_step_in_channels, :, :]
            latent_t = self.forward_single(x_t)  # [B, num_channels, W_out, H_out]
            latent_series.append(latent_t.unsqueeze(1))  # 在时间维度上扩展

        latent_series = torch.cat(latent_series, dim=1)  # [B, history_length, num_channels, W_out, W_out]

        # import ipdb;ipdb.set_trace()

        # 根据 fusion_mode 融合历史步信息
        if self.fusion_mode == 'mean':
            latent_fused = latent_series.mean(dim=1)  # 均值融合, [B, num_channels, W_out, H_out]
        else:
            # 可增加其它融合方式，比如拼接后通过1x1卷积
            B, T, C, H_out, W_out = latent_series.shape
            latent_concat = latent_series.view(B, -1, H_out, W_out)  # [B, T * C, H_out, W_out]
            fusion_conv = nn.Conv2d(T * C, C, kernel_size=1)
            latent_fused = fusion_conv(latent_concat)

        # return latent_series, latent_fused
        return latent_fused



# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('MuZeroHistoryModel')
class MuZeroHistoryModel(MuZeroModel):

    def __init__(
        self,
        observation_shape: SequenceType = (12, 96, 96),
        action_space_size: int = 6,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 16,
        value_head_channels: int = 16,
        policy_head_channels: int = 16,
        reward_head_hidden_channels: SequenceType = [32],
        value_head_hidden_channels: SequenceType = [32],
        policy_head_hidden_channels: SequenceType = [32],
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
        history_length: int = 5,
        num_unroll_steps: int = 5,
        use_sim_norm: bool = False,
        analysis_sim_norm: bool = False,
        *args,
        **kwargs
    ):
        """
        Overview:
            The definition of the model for MuZero w/ Context, a variant of MuZero.
            This variant retains the same training settings as MuZero but diverges during inference
            by employing a k-step recursively predicted latent representation at the root node,
            proposed in the UniZero paper https://arxiv.org/abs/2406.10667.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
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
        super(MuZeroHistoryModel, self).__init__()

        self.timestep = 0
        self.history_length = history_length  # NOTE
        self.num_unroll_steps = num_unroll_steps

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
        print('action_space_size:', action_space_size)
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
        self.analysis_sim_norm = analysis_sim_norm

        if observation_shape[1] == 96:
            latent_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
        elif observation_shape[1] == 64:
            latent_size = math.ceil(observation_shape[1] / 8) * math.ceil(observation_shape[2] / 8)
        elif observation_shape[1] == 5:
            latent_size = 64


        flatten_input_size_for_reward_head = (
            (reward_head_channels * latent_size) if downsample else
            (reward_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_input_size_for_value_head = (
            (value_head_channels * latent_size) if downsample else
            (value_head_channels * observation_shape[1] * observation_shape[2])
        )
        flatten_input_size_for_policy_head = (
            (policy_head_channels * latent_size) if downsample else
            (policy_head_channels * observation_shape[1] * observation_shape[2])
        )

        if observation_shape[1] == 5:
            # MemoryEnv
            embedding_size = 768
            self.representation_network = RepresentationNetworkMemoryEnv(
                observation_shape,
                embedding_size=embedding_size,
                channels= [16, 32, 64],
                group_size= 8,
            )
            self.num_channels = num_channels
            self.latent_state_dim = self.num_channels - self.action_encoding_dim

            self.dynamics_network = DynamicsNetworkVector(
                action_encoding_dim=self.action_encoding_dim,
                num_channels=embedding_size + self.action_encoding_dim,
                common_layer_num=2,
                reward_head_hidden_channels=reward_head_hidden_channels,
                output_support_size=self.reward_support_size,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                norm_type=norm_type,
                res_connection_in_dynamics=True,
            )
            self.vector_ynamics_network = True
            self.prediction_network = PredictionNetworkMLP(
                action_space_size=action_space_size,
                num_channels=embedding_size,
                value_head_hidden_channels=value_head_hidden_channels,
                policy_head_hidden_channels=policy_head_hidden_channels,
                output_support_size=self.value_support_size,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                norm_type=norm_type
            )
        else:
            # atari
            self.representation_network = RepresentationNetwork(
                observation_shape,
                num_res_blocks,
                num_channels,
                downsample,
                activation=activation,
                norm_type=norm_type,
                embedding_dim=768,
                group_size=8,
                use_sim_norm=use_sim_norm,  # NOTE
            )
                    # ====== for analysis ======
            if self.analysis_sim_norm:
                self.encoder_hook = FeatureAndGradientHook()
                self.encoder_hook.setup_hooks(self.representation_network)

            self.dynamics_network = DynamicsNetwork(
                observation_shape,
                self.action_encoding_dim,
                num_res_blocks,
                num_channels + self.action_encoding_dim,
                reward_head_channels,
                reward_head_hidden_channels,
                self.reward_support_size,
                flatten_input_size_for_reward_head,
                downsample,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                activation=activation,
                norm_type=norm_type,
                embedding_dim=768,
                group_size=8,
                use_sim_norm=use_sim_norm,  # NOTE
            )
            self.vector_ynamics_network = False

            self.prediction_network = PredictionNetwork(
                observation_shape,
                action_space_size,
                num_res_blocks,
                num_channels,
                value_head_channels,
                policy_head_channels,
                value_head_hidden_channels,
                policy_head_hidden_channels,
                self.value_support_size,
                flatten_input_size_for_value_head,
                flatten_input_size_for_policy_head,
                downsample,
                last_linear_layer_init_zero=self.last_linear_layer_init_zero,
                activation=activation,
                norm_type=norm_type
            )

        if self.self_supervised_learning_loss:
            # projection used in EfficientZero
            if self.downsample:
                # In Atari, if the observation_shape is set to (12, 96, 96), which indicates the original shape of
                # (3,96,96), and frame_stack_num is 4. Due to downsample, the encoding of observation (latent_state) is
                # (64, 96/16, 96/16), where 64 is the number of channels, 96/16 is the size of the latent state. Thus,
                # self.projection_input_dim = 64 * 96/16 * 96/16 = 64*6*6 = 2304
                self.projection_input_dim = num_channels * latent_size
            else:
                self.projection_input_dim = num_channels * observation_shape[1] * observation_shape[2]

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def stack_history_torch(self, obs, history_length):
        """
        对输入观测值 `obs`（形状 [T, 3, 64, 64]）进行转换，
        将每个时间步变为堆叠了前 history_length 帧的观测数据，
        如果前面的历史不足 history_length 则使用零填充。
        
        最终返回的 obs_with_history 形状为 [T, 3*history_length, 64, 64].
        
        参数:
        obs: PyTorch tensor，形状为 [T, 3, H, W] (例如 H = W = 64)
        history_length: 要堆叠的历史步数
        
        返回:
        obs_with_history: PyTorch tensor，形状为 [T, 3*history_length, H, W]
        """
        T, C, H, W = obs.shape
        # seq_batch_size = 3
        seq_batch_size = int(T/(self.num_unroll_steps + self.history_length + 1))

        # import ipdb;ipdb.set_trace()

        # Step 1: 重构 shape 为 [seq_batch_size, (num_unroll_steps+history_length), 3, 64, 64]
        obs = obs.view(seq_batch_size, self.num_unroll_steps + self.history_length + 1, 3, H, W)
        # 此时 obs.shape = [3, 7, 3, 64, 64]

        # Step 2: 对时间维度应用 sliding window 操作（unfold）；
        # unfolding 参数: 在 dim=1 上，窗口大小为 history_length，步长为 1.
        # unfolding 后形状：[seq_batch_size, (7 - history_length + 1), history_length, 3, 64, 64]
        windows = obs.unfold(dimension=1, size=self.history_length, step=1)  # 形状：[3, 6, 3, 64, 64, 2]
        # print("Step 2 windows.shape:", windows.shape)

        # windows.shape torch.Size([3, 7, 3, 64, 64, 2]) -> [3, 7+self.history_length-1, 3, 64, 64, 2] 请在前面补零，补齐为后者的维度
        # 计算需要补零的数量（在前面补上 history_length - 1 个零）
        pad_len = self.history_length - 1
        # 构造与 windows 除待补维度外其他维度相同的补零张量，其形状为 [3, pad_len, 3, 64, 64, 2]
        padding_tensor = torch.zeros(
            (windows.size(0), pad_len, windows.size(2), windows.size(3), windows.size(4), windows.size(5)),
            dtype=windows.dtype,
            device=windows.device
        )
        # 在维度 1 上拼接补零张量
        windows_padded = torch.cat([padding_tensor, windows], dim=1)

        # Step 4: 将窗口中的观测在通道维度上进行拼接
        # 原本每个窗口形状为 [2, 3, 64, 64]，将 2 (history_length) 个通道拼接后变为 [6, 64, 64]
        # 整体结果 shape 最终为 [seq_batch_size, num_unroll_steps, history_length*3, 64, 64] = [3, 5, 6, 64, 64]
        windows_padded = windows_padded.reshape(seq_batch_size, self.num_unroll_steps+self.history_length + 1, history_length * 3, H, W)

        obs_with_history = windows_padded.view(-1, self.history_length * 3, H, W)


        return obs_with_history

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

        # import ipdb;ipdb.set_trace()

        if self.training or action_batch is None:
            # train phase
            # import ipdb;ipdb.set_trace()
            if action_batch is None and obs.shape[1] != 3*self.history_length:
                # ======train phase: compute target value =======
                # 已知seq_batch_size=3， self.num_unroll_steps=5， self.history_length=2，
                # 已知目前obs.shape == [seq_batch_size,(self.num_unroll_steps+self.history_length),3,64,64]  请先变换为 -> [seq_batch_size,(self.num_unroll_steps+self.history_length),3,64,64]  
                # 例如[21, 3, 64, 64] ->  [3, 7, 3, 64, 64]
                # 对于其中每个序列，[i,7, 3,64,64], 
                # 取self.history_length之后的时间步，每个时间步都保留前面self.history_length的ob，
                # 即变为[i,7-self.history_length, 3*self.history_length,64,64] = [i,5,6,64,64]
                # 总的数据变换过程为 [21, 3, 64, 64] -> [3*7, 3, 64, 64] -> [3, 7, 3, 64, 64]-> [3, 6, 6, 64, 64]
                obs_with_history = self.stack_history_torch(obs, self.history_length)
                # print(f"train phase (compute target value) obs_with_history.shape:{obs_with_history.shape}")

            else:
                # ======= train phase: init_infer =======
                obs_with_history = obs
                # print(f"train phase (init inference) obs_with_history.shape:{obs_with_history.shape}")

            assert obs_with_history.shape[1] == 3*self.history_length
            # TODO(pu)
            self.latent_state = self.representation_network(obs_with_history)

            self.timestep = 0
        else:
            # print(f"collect/eval phase obs_with_history.shape:{obs.shape}")
            # ======== collect/eval phase ========
            obs_with_history = obs

            # ===== obs: torch.Tensor, action_batch=None, current_obs_batch=None
            assert obs_with_history.shape[1] == 3*self.history_length
            # TODO(pu)
            self.latent_state = self.representation_network(obs_with_history)
            # print(f"collect/eval phase latent_state.shape:{self.latent_state.shape}")


        # import ipdb;ipdb.set_trace()

        policy_logits, value = self.prediction_network(self.latent_state)
        self.timestep += 1
        return MZNetworkOutput(
            value,
            [0. for _ in range(batch_size)],
            policy_logits,
            self.latent_state,
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
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
        if self.vector_ynamics_network:
            next_latent_state, reward = self._dynamics_vector(latent_state, action)
        else:
            next_latent_state, reward = self._dynamics(latent_state, action)

        policy_logits, value = self.prediction_network(next_latent_state)
        self.latent_state = next_latent_state  # NOTE: update latent_state
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def _dynamics_vector(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            ``reward`` and ``next_reward_hidden_state``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_reward_hidden_state (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted reward for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H)`, where B is batch_size, H is the dimension of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        if self.discrete_action_encoding_type == 'one_hot':
            # Stack latent_state with the one hot encoded action
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
            action_encoding = action_one_hot
        elif self.discrete_action_encoding_type == 'not_one_hot':
            action_encoding = action / self.action_space_size
            if len(action_encoding.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action_encoding = action_encoding.unsqueeze(-1)

        action_encoding = action_encoding.to(latent_state.device).float()
        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim]) or
        # (batch_size, latent_state[1] + action_space_size]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        next_latent_state, reward = self.dynamics_network(state_action_encoding)

        if not self.state_norm:
            return next_latent_state, reward
        else:
            next_latent_state_normalized = renormalize(next_latent_state)
            return next_latent_state_normalized, reward


class DynamicsNetwork(nn.Module):

    def __init__(
        self,
        observation_shape: SequenceType,
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        reward_head_hidden_channels: SequenceType = [32],
        output_support_size: int = 601,
        flatten_input_size_for_reward_head: int = 64,
        downsample: bool = False,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        embedding_dim: int = 256,
        group_size: int = 8,
        use_sim_norm: bool = False,
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
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical reward output.
            - flatten_input_size_for_reward_head (:obj:`int`): The flatten size of output for reward head, i.e., \
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
        self.flatten_input_size_for_reward_head = flatten_input_size_for_reward_head

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
            self.flatten_input_size_for_reward_head,
            hidden_channels=reward_head_hidden_channels[0],
            layer_num=len(reward_head_hidden_channels) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.activation = activation
        self.use_sim_norm = use_sim_norm
        if self.use_sim_norm:
            self.sim_norm = SimNorm(simnorm_dim=group_size)

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
        x = x.view(-1, self.flatten_input_size_for_reward_head)

        # use the fully connected layer to predict reward
        reward = self.fc_reward_head(x)

        if self.use_sim_norm:
            next_latent_state = self.sim_norm(next_latent_state)

        return next_latent_state, reward

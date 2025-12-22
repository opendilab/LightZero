import logging
import copy
import random
from collections import defaultdict, deque
from typing import Union, Tuple, List, Dict, Optional
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.model import FCEncoder, ConvEncoder
from ding.reward_model.base_reward_model import BaseRewardModel
from ding.torch_utils.data_helper import to_tensor
from ding.utils import RunningMeanStd
from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from easydict import EasyDict
from ding.utils import get_rank, get_world_size, build_logger, allreduce, synchronize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy


class RewardForwardFilter:
    """
    Forward discounted return filter for intrinsic reward normalization.
    Time must advance EXACTLY once per real env step.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.rewems = None  # running discounted return

    def update(self, reward):
        """
        reward: scalar or np.ndarray / torch.Tensor
                shape: [num_env] or []
        """
        if self.rewems is None:
            self.rewems = reward
        else:
            self.rewems = self.rewems * self.gamma + reward
        return self.rewems


class RNDNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], frame_stack_num: int, hidden_size_list: SequenceType, output_dim: int = 512, activation_type: str = "ReLU", kernel_size_list=[8,4,3], stride_size_list=[4,2,1]) -> None:
        super(RNDNetwork, self).__init__()
        assert len(hidden_size_list) >= 1, "hidden_size_list must contain at least one element."
        if activation_type == "ReLU":
            self.activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            raise KeyError("not support activation_type for RND model: {}, please customize your own RND model".format(activation_type))
        
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            target_backbone = []
            predictor_backbone = []
            input_size = obs_shape
            for i in range(len(hidden_size_list)):
                target_backbone.append(nn.Linear(input_size , hidden_size_list[i]))
                target_backbone.append(self.activation)
                
                predictor_backbone.append(nn.Linear(input_size , hidden_size_list[i]))
                predictor_backbone.append(self.activation)
                input_size = hidden_size_list[i]
            self.target = nn.Sequential(
                            *target_backbone,
                            nn.Linear(input_size, output_dim)
                        )
            self.predictor = nn.Sequential(
                            *predictor_backbone,
                            nn.Linear(input_size, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, output_dim)
                        )
            
        elif len(obs_shape) == 3:
            target_backbone = []
            predictor_backbone = []
            input_size = obs_shape[0] * frame_stack_num
            for i in range(len(hidden_size_list)):
                target_backbone.append(nn.Conv2d(input_size , hidden_size_list[i], kernel_size_list[i], stride_size_list[i]))
                target_backbone.append(self.activation)
                
                predictor_backbone.append(nn.Conv2d(input_size , hidden_size_list[i], kernel_size_list[i], stride_size_list[i]))
                predictor_backbone.append(self.activation)
                input_size = hidden_size_list[i]
            target_backbone.append(nn.Flatten())
            predictor_backbone.append(nn.Flatten())
            
            target_backbone_tmp = nn.Sequential(*target_backbone)
            predictor_backbone_tmp = nn.Sequential(*predictor_backbone)
            with torch.no_grad():
                new_obs_shape = (obs_shape[0] * frame_stack_num, *obs_shape[1:])
                dummy = torch.zeros(1, *new_obs_shape)
                feat = target_backbone_tmp(dummy)
                last_hidden_dim = feat.shape[1]
            self.target = nn.Sequential(
                            target_backbone_tmp,
                            nn.Linear(last_hidden_dim, output_dim)
                        )
            self.predictor = nn.Sequential(
                            predictor_backbone_tmp,
                            nn.Linear(last_hidden_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, output_dim)
                        )
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )        
        
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        
        for param in self.target.parameters():
            param.requires_grad = False
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(obs)
        with torch.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature


class RNDNetworkRepr(nn.Module):
    """
    Overview:
        The RND reward model class (https://arxiv.org/abs/1810.12894v1) with representation network.
    """

    def __init__(self, obs_shape: Union[int, SequenceType], latent_shape: Union[int, SequenceType],  hidden_size_list: SequenceType,
                 representation_network) -> None:
        super(RNDNetworkRepr, self).__init__()
        self.representation_network = representation_network
        assert len(hidden_size_list) >= 1, "hidden_size_list must contain at least one element."
        feature_dim = hidden_size_list[-1]
        activation = nn.ReLU()

        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            target_backbone = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            target_backbone = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )

        if isinstance(latent_shape, int) or (isinstance(latent_shape, SequenceType) and len(latent_shape) == 1):
            predictor_backbone = FCEncoder(latent_shape, hidden_size_list)
        elif isinstance(latent_shape, SequenceType) and len(latent_shape) == 3:
            predictor_backbone = ConvEncoder(latent_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support latent_shape for pre-defined encoder: {}, please customize your own RND model".
                format(latent_shape)
            )

        self.target = nn.Sequential(target_backbone, activation)
        self.predictor = nn.Sequential(
            predictor_backbone,
            activation,
            nn.Linear(feature_dim, feature_dim),
            activation,
            nn.Linear(feature_dim, feature_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predict_feature = self.predictor(self.representation_network(obs))
        with torch.no_grad():
            target_feature = self.target(obs)

        return predict_feature, target_feature


@REWARD_MODEL_REGISTRY.register('rnd_muzero')
class RNDRewardModel(BaseRewardModel):
    """
    Overview:
        The RND reward model class (https://arxiv.org/abs/1810.12894v1) modified for MuZero.
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_data``, \
            ``__init__``, ``_train``, ``load_state_dict``, ``state_dict``
    Config:
        == ====================  =====  =============  =======================================  =======================
        ID Symbol                Type   Default Value  Description                              Other(Shape)
        == ====================  =====  =============  =======================================  =======================
        1   ``type``              str     rnd          | Reward model register name, refer      |
                                                       | to registry ``REWARD_MODEL_REGISTRY``  |
        2  | ``intrinsic_``      str      add          | the intrinsic reward type              | including add, new
           | ``reward_type``                           |                                        | , or assign
        4  | ``batch_size``      int      64           | Training batch size                    |
        5  | ``hidden``          list     [64, 64,     | the MLP layer shape                    |
           | ``_size_list``      (int)    128]         |                                        |
        6  | ``update_per_``     int      100          | Number of updates per collect          |
           | ``collect``                               |                                        |
        7  | ``input_norm``        bool     True         | Observation normalization              |
        8  | ``input_norm_``       int      0            | min clip value for obs normalization   |
           | ``clamp_min``
        9  | ``input_norm_``       int      1            | max clip value for obs normalization   |
           | ``clamp_max``
        10 | ``intrinsic_``      float    0.01         | the weight of intrinsic reward         | r = w*r_i + r_e
             ``reward_weight``
        11 | ``extrinsic_``      bool     True         | Whether to normlize extrinsic reward
             ``reward_norm``
        12 | ``extrinsic_``      int      1            | the upper bound of the reward
            ``reward_norm_max``                        | normalization
        == ====================  =====  =============  =======================================  =======================
    """
    rnd_config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='rnd',
        # (str) The intrinsic reward type, including add, new, or assign.
        intrinsic_reward_type='add',
        # (list(int)) Sequence of ``hidden_size`` of reward network.
        # If obs.shape == 1,  use MLP layers.
        # If obs.shape == 3,  use conv layer and final dense layer.
        hidden_size_list=[64, 64, 128],
        # (int) How many updates(iterations) to train after collector's one collection.
        # (bool) Observation normalization: transform obs to mean 0, std 1.
        input_norm=True,
        # (int) Min clip value for observation normalization.
        input_norm_clamp_min=-5,
        # (int) Max clip value for observation normalization.
        input_norm_clamp_max=5,
        # Means the relative weight of RND intrinsic_reward.
        # (float) The weight of intrinsic reward
        # r = intrinsic_reward_weight * r_i + r_e.
        # (bool) Whether to normalize extrinsic reward using running statistics.
        # (float) Discount factor used when adjusting target value.
        discount_factor=1.0,
        # 新增：图片日志总开关与可视化参数
        enable_image_logging=False,         # ← 总开关：是否在TB输出图片（时间线+关键帧等）
        peaks_topk=12,                      # 关键帧个数
        
        # —— 新增：自适应权重调度 —— #
        use_intrinsic_weight_schedule=True,     # 打开自适应权重
        intrinsic_weight_mode='cosine',         # 'cosine' | 'linear' | 'constant'
        intrinsic_weight_warmup=1000,           # 前多少次 estimate 权重=0
        intrinsic_weight_ramp=5000,            # 从0升到max所需的 estimate 数
        intrinsic_weight_min=0.0,               
        intrinsic_weight_max=0.02,        
        intrinsic_norm=True,
        intrinsic_norm_type='return', # 'reward | 'return'
        instrinsic_gamma=0.99,      
        frame_stack_num=1,
    )

    def __init__(self, config: EasyDict, device: str = 'cpu', representation_network: nn.Module = None, 
                target_representation_network: nn.Module = None, use_momentum_representation_network: bool = True, 
                bp_update_sync: bool = True, multi_gpu: bool = False) -> None:  # noqa
        super(RNDRewardModel, self).__init__()
        self.cfg = EasyDict(deepcopy(RNDRewardModel.rnd_config))
        self.cfg.update(config)
        self.representation_network = representation_network
        self.target_representation_network = target_representation_network
        self.use_momentum_representation_network = use_momentum_representation_network
        self.input_type = self.cfg.input_type
        assert self.input_type in ['obs', 'latent_state', 'obs_latent_state'], self.input_type
        self.intrinsic_reward_type = self.cfg.intrinsic_reward_type
        self.discount_factor = getattr(self.cfg, 'discount_factor', 1.0)
        self.update_proportion = self.cfg.update_proportion
        self.frame_stack_num = self.cfg.frame_stack_num
        
        self._rank = get_rank()
        self._world_size = get_world_size()
        self.multi_gpu = multi_gpu
        self._bp_update_sync = bp_update_sync
        self._device = device
        self.activation_type = getattr(self.cfg, 'activation_type', 'ReLU')
        self.enable_image_logging = bool(getattr(self.cfg, 'enable_image_logging', False))
        self.use_intrinsic_weight_schedule = bool(getattr(self.cfg, 'use_intrinsic_weight_schedule', False))
        
        if self.multi_gpu:
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if 'cuda' in device else 'cpu'
        else:
            self._device = device
        
        if self.input_type == 'obs':
            self.input_shape = self.cfg.obs_shape
            self.reward_model = RNDNetwork(self.input_shape, self.frame_stack_num, self.cfg.hidden_size_list, activation_type=self.activation_type).to(self._device)
        elif self.input_type == 'latent_state':
            self.input_shape = self.cfg.latent_state_dim
            self.reward_model = RNDNetwork(self.input_shape, self.frame_stack_num, self.cfg.hidden_size_list, activation_type=self.activation_type).to(self._device)
        elif self.input_type == 'obs_latent_state':
            if self.use_momentum_representation_network:
                self.reward_model = RNDNetworkRepr(self.cfg.obs_shape, self.cfg.latent_state_dim, self.cfg.hidden_size_list[0:-1],
                                                  self.target_representation_network, activation_type=self.activation_type).to(self._device)
            else:
                self.reward_model = RNDNetworkRepr(self.cfg.obs_shape, self.cfg.latent_state_dim, self.cfg.hidden_size_list[0:-1],
                                                  self.representation_network, activation_type=self.activation_type).to(self._device)

        assert self.intrinsic_reward_type in ['add', 'new', 'assign']

        self.rnd_return_rms = RunningMeanStd(epsilon=1e-4) 
        self._running_mean_std_rnd_reward = RunningMeanStd(epsilon=1e-4)
        self._running_mean_std_rnd_obs = RunningMeanStd(epsilon=1e-4)
        self._running_mean_std_ext_reward = RunningMeanStd(epsilon=1e-4)

        self.estimate_cnt_rnd = 0
        self.train_cnt_rnd = 0
        self._state_visit_counts = defaultdict(int)
        self._initial_reward_samples: List[np.ndarray] = []
        self._initial_consistency_logged = False

    def _init_log(self, tb_logger, _exp_name, _instance_name: str = 'RNDModel'):
        self._logger, _ = build_logger(
            path='./{}/log/{}'.format(_exp_name, _instance_name),
            name=_instance_name,
            need_tb=False
        )
        self._tb_logger = tb_logger
        
        self._logger.info(
            "[RND] device=%s | input_type=%s | hidden=%s",
            self._device, self.input_type, str(self.cfg.hidden_size_list)
        )
        if self.use_intrinsic_weight_schedule:
            self._logger.info(
                "[RND] intrinsic weight schedule: ENABLED | mode=%s | warmup=%d | ramp=%d | min=%.3f | max=%.3f",
                self.cfg.intrinsic_weight_mode, self.cfg.intrinsic_weight_warmup, self.cfg.intrinsic_weight_ramp, 
                self.cfg.intrinsic_weight_min, self.cfg.intrinsic_weight_max
            )
        else:
            self._logger.info(
                "[RND] intrinsic weight schedule: disabled | fixed_weight=%.3f", self.cfg.intrinsic_weight_max
            )
        self._logger.info(f"[RND] predictor: {self.reward_model.predictor}")
        self._logger.info(f"[RND] predictor: {self.reward_model.target}")

    def reset_discounted_reward(self, _env_num):
        self.discount_reward_env_ids = {env_id:  RewardForwardFilter(gamma=self.cfg.instrinsic_gamma) for env_id in range(_env_num)}
    
    def _flatten_obs_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Overview:
            Flatten time/batch dimensions while keeping the per-observation shape intact.
        """
        if not isinstance(obs_batch, np.ndarray):
            obs_batch = np.asarray(obs_batch)
        feature_size = int(np.prod(self.cfg.obs_shape))
        total = obs_batch.size // feature_size
        if isinstance(self.cfg.obs_shape, int):
            target_shape = (total, self.cfg.obs_shape)
        elif isinstance(self.cfg.obs_shape, tuple) or isinstance(self.cfg.obs_shape, list):
            target_shape = (total,) + tuple(self.cfg.obs_shape)
        else:
            raise ValueError(f'self.input_shape={type(self.input_shape)}')
        return obs_batch.reshape(target_shape)

    def _get_latent_state_from_obs(self, obs_tensor, batch_size=128):
        _pad_token = self.representation_network.tokenizer.pad_token_id
        latent_state_list = []
        num_transitions = obs_tensor.shape[0]
        for start in range(0, num_transitions, batch_size):
            end = start + batch_size
            batch = obs_tensor[start:end]     
            x = batch.long()
            batch_attention_mask = x != _pad_token
            with torch.no_grad():       
                batch_latent = self.representation_network.pretrained_model(x, attention_mask=batch_attention_mask).last_hidden_state[:, 0, :]
            latent_state_list.append(batch_latent.detach().cpu())
        latent_state_tensor = torch.cat(latent_state_list, dim=0)
        return latent_state_tensor
    
    def _prepare_inputs_from_obs(self, obs_array: np.ndarray) -> torch.Tensor:
        """
        Overview:
            Convert raw observations into tensors that can be consumed by the RND networks
            according to the configured input type.
        """
        obs_tensor = to_tensor(obs_array).to(self._device)
        if self.input_type == 'latent_state':
            inputs = self._get_latent_state_from_obs(obs_tensor=obs_tensor).to(self._device)
        else:
            inputs = obs_tensor
        return inputs

    def _update_input_running_stats(self, tensor: torch.Tensor) -> None:
        """
        Overview:
            Update running mean/std for input normalization using the provided tensor.
        """
        if not self.cfg.input_norm or tensor.numel() == 0:
            return
        self._running_mean_std_rnd_obs.update(tensor.detach().cpu().numpy())
        

    def _stack_frames_after_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: (L, 1, H, W)  # 已 normalize
            return: (L, k, H, W)  # 用首帧 padding，逐步形成 k 帧堆叠
        """
        if x.dim() != 4 or self.frame_stack_num <= 1:
            return x
        frame_stack_num = self.frame_stack_num

        L, C, H, W = x.shape
        out = torch.zeros((L, C * frame_stack_num, H, W), device=x.device, dtype=x.dtype)
        buf = deque([x[0]] * frame_stack_num, maxlen=frame_stack_num)  
        for t in range(L):
            buf.append(x[t])
            out[t] = torch.cat(list(buf), dim=0)  

        return out
    
    def _update_rnd_return_rms(self, rnd_next_obs_seq: Dict[int, List[np.ndarray]]):
        returns = []
        
        for env_id, seq in rnd_next_obs_seq.items():
            if seq is None or len(seq) == 0:
                continue
            
            obs_batch = np.stack(seq, axis=0)
            flat_obs = self._flatten_obs_batch(obs_batch)
            inputs = self._prepare_inputs_from_obs(flat_obs) 
            inputs = self._normalize_inputs(inputs)   
            
            inputs = self._stack_frames_after_norm(inputs)
            
            with torch.no_grad():
                pred_f, tgt_f = self.reward_model(inputs)
                raw_int = F.mse_loss(pred_f, tgt_f, reduction='none').sum(dim=-1) / 2.0  
            raw_int = raw_int.detach().cpu().numpy().astype(np.float32)
        
            forward_filter = self.discount_reward_env_ids[env_id]
            for r in raw_int:
                ret = forward_filter.update(float(r))   
                returns.append(float(ret))

        self.rnd_return_rms.update(np.asarray(returns, dtype=np.float32))

        
    def _normalize_intrinsic_rewards(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Normalize intrinsic rewards with the running std statistics.
        """
        if self.cfg.intrinsic_norm:
            if self.cfg.intrinsic_norm_type == 'reward': 
                std = to_tensor(self._running_mean_std_rnd_reward.std).to(self._device)
                std = torch.clamp(std, min=1e-6)
                normalized = tensor / std
                
                return torch.clamp(
                    normalized,
                    min=0,
                    max=getattr(self.cfg, 'intrinsic_norm_reward_clamp_max', 10)
                )
            elif self.cfg.intrinsic_norm_type == 'return':
                std = to_tensor(self.rnd_return_rms.std).to(self._device)
                std = torch.clamp(std, min=1e-6)
                normalized = tensor / std
                return normalized   
            else:
                return tensor
        else:
            return tensor
    
    def _normalize_inputs(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Normalize inputs with the running mean/std statistics (intrinsic input normalization).
        """
        if not self.cfg.input_norm or tensor.numel() == 0:
            return tensor
        if self.cfg.input_norm:
            mean = to_tensor(self._running_mean_std_rnd_obs.mean).to(self._device)
            std = to_tensor(self._running_mean_std_rnd_obs.std).to(self._device)
            std = torch.clamp(std, min=1e-6)
            normalized = (tensor - mean) / std
            return torch.clamp(normalized, min=self.cfg.input_norm_clamp_min, max=self.cfg.input_norm_clamp_max)
        return tensor

    def _normalize_ext_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize extrinsic rewards using running statistics when enabled.
        """
        if rewards.size == 0:
            return rewards
        normalized = np.asarray(rewards, dtype=np.float32)
        if getattr(self.cfg, 'extrinsic_norm', False):
            self._running_mean_std_ext_reward.update(normalized)
            mean = np.asarray(self._running_mean_std_ext_reward.mean, dtype=np.float32)
            std = np.asarray(self._running_mean_std_ext_reward.std, dtype=np.float32) + 1e-6
            normalized = (normalized - mean) / std
            normalized = np.clip(
                normalized,
                a_min=getattr(self.cfg, 'extrinsic_norm_clamp_min', -5),
                a_max=getattr(self.cfg, 'extrinsic_norm_clamp_max', 5)
            )
        elif getattr(self.cfg, 'extrinsic_sign', False):
            normalized = np.sign(normalized)
        return normalized

    def _hash_obs(self, obs: np.ndarray) -> int:
        return hash(obs.tobytes())

    def _update_visit_counts(self, obs_array: np.ndarray) -> None:
        if obs_array.size == 0:
            return
        flat = obs_array.reshape(obs_array.shape[0], -1)
        for obs in flat:
            self._state_visit_counts[self._hash_obs(obs)] += 1

    def _spearmanr(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        x_rank = x_rank.astype(np.float32)
        y_rank = y_rank.astype(np.float32)
        x_rank -= x_rank.mean()
        y_rank -= y_rank.mean()
        denom = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
        if denom == 0:
            return 0.0
        return float(np.dot(x_rank, y_rank) / denom)

    def _log_initial_bonus_consistency(self) -> None:
        if self._initial_consistency_logged or not self._initial_reward_samples:
            return
        rewards = np.concatenate(self._initial_reward_samples, axis=0)
        if rewards.size == 0:
            return
        rewards = rewards - rewards.min()
        rewards = rewards + 1e-8
        p = rewards / rewards.sum()
        kl = float(np.sum(p * np.log(p * len(p))))
        if self._tb_logger:
            self._tb_logger.add_scalar('rnd_reward_model/bcs_initial_kl', kl, 0)
        self._initial_consistency_logged = True
        self._initial_reward_samples = []

    def _log_final_metrics(self, intrinsic_rewards: np.ndarray, obs_array: np.ndarray, step: int) -> None:
        if intrinsic_rewards.size == 0 or obs_array.size == 0:
            return
        intrinsic_flat = intrinsic_rewards.reshape(-1)
        flat_obs = obs_array.reshape(obs_array.shape[0], -1)
        hashes = [self._hash_obs(obs) for obs in flat_obs]
        counts = np.array([max(self._state_visit_counts.get(h, 1), 1) for h in hashes], dtype=np.float32)
        inv_counts = 1.0 / (counts + 1e-6)
        bcs_final = self._spearmanr(intrinsic_flat, inv_counts)
        pca_spearman = bcs_final
        if self._tb_logger:
            self._tb_logger.add_scalar('rnd_reward_model/bcs_final_spearman', bcs_final, step)
            self._tb_logger.add_scalar('rnd_reward_model/pca_spearman', pca_spearman, step)

    def _discount_cumsum(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        if rewards.ndim != 2:
            rewards = rewards.reshape(rewards.shape[0], -1)
        discounted = np.zeros_like(rewards, dtype=np.float32)
        if rewards.shape[1] == 0:
            return discounted
        discounted[:, -1] = rewards[:, -1]
        for t in range(rewards.shape[1] - 2, -1, -1):
            discounted[:, t] = rewards[:, t] + gamma * discounted[:, t + 1]
        return discounted

    def warmup_with_random_segments(self, data: list) -> None:
        """
        Overview:
            Use randomly collected segments to bootstrap the input normalization statistics
            before the main training loop starts.
        """
        if data is None or len(data) == 0:
            return
        self._logger.info(f"[RND] for input_obs_norm, random_collect_data={len(data)}")

        concatenated = np.stack(data, axis=0)
        flattened = self._flatten_obs_batch(concatenated)
        inputs = self._prepare_inputs_from_obs(flattened)
        self._update_input_running_stats(inputs)
            
    def sync_gradients(self, model: torch.nn.Module) -> None:
        """
        Overview:
            Synchronize (allreduce) gradients of model parameters in data-parallel multi-gpu training.
        Arguments:
            - model (:obj:`torch.nn.Module`): The model to synchronize gradients.

        .. note::
            This method is only used in multi-gpu training, and it should be called after ``backward`` method and \
            before ``step`` method. The user can also use ``bp_update_sync`` config to control whether to synchronize \
            gradients allreduce and optimizer updates.
        """

        if self._bp_update_sync:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        allreduce(param.grad.data)
                    else:
                        zero_grad = torch.zeros_like(param.data)
                        allreduce(zero_grad)
        else:
            synchronize()


    def compute_loss(self, obs_batch) -> None:
        batch_size = obs_batch.shape[0]
        T = obs_batch.shape[1] if self.frame_stack_num == 1 else obs_batch.shape[1] - self.frame_stack_num + 1
        flat_obs = self._flatten_obs_batch(obs_batch)
        prepared_inputs = self._prepare_inputs_from_obs(flat_obs)
        if prepared_inputs.numel() == 0:
            return
        
        normalized_input = self._normalize_inputs(prepared_inputs)
        if self.frame_stack_num > 1:
            inputs = torch.zeros((batch_size, T, self.frame_stack_num, *self.input_shape[1:]), device=normalized_input.device, dtype=normalized_input.dtype)
            normalized_input = normalized_input.reshape(batch_size, -1, normalized_input.shape[-2], normalized_input.shape[-1])
            for j in range(T):
                inputs[:,j] = normalized_input[:, j:j+self.frame_stack_num]
            inputs = inputs.reshape(batch_size*T, *inputs.shape[2:])
        else:
            inputs = normalized_input
        
        predict_feature, target_feature = self.reward_model(inputs)
        forward_mse = nn.MSELoss(reduction='none')
        forward_loss = forward_mse(predict_feature, target_feature).mean(-1)
        
        # Proportion of exp used for predictor update
        mask = torch.rand(len(forward_loss)).to(self._device)
        mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self._device)
        loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self._device))

        if self._tb_logger:
            self._tb_logger.add_scalar('rnd_reward_model/rnd_mse_loss', loss.item(), self.train_cnt_rnd)
        self.train_cnt_rnd += 1
        self._update_input_running_stats(prepared_inputs)
        return loss

    def _intrinsic_weight(self, step: int) -> float:
        """
        根据当前 estimate 步数返回 RND 权重：
        - step < warmup  → 0
        - warmup 之后    → 线性/余弦从 min 升到 max
        """
        if not self.cfg.use_intrinsic_weight_schedule:
            return float(self.cfg.intrinsic_weight_max)

        wmin = float(self.cfg.intrinsic_weight_min)
        wmax = float(self.cfg.intrinsic_weight_max)
        warmup = int(self.cfg.intrinsic_weight_warmup)
        ramp   = max(1, int(self.cfg.intrinsic_weight_ramp))
        mode   = str(self.cfg.intrinsic_weight_mode).lower()

        if step <= warmup:
            return 0.0

        # 归一化进度 p ∈ [0,1]
        t = min(max(step - warmup, 0), ramp)
        p = t / float(ramp)

        if mode == 'linear':
            w = wmin + (wmax - wmin) * p
        elif mode == 'cosine':
            w = wmin + 0.5 * (wmax - wmin) * (1.0 - np.cos(np.pi * p))
        else:
            w = float(self.cfg.intrinsic_weight_max) 

        return float(w)

    def estimate(self, obs_batch_ori, target_reward) -> List[Dict]:
        """
        Rewrite the reward key in each row of the data.
        """
        batch_size = obs_batch_ori.shape[0]
        T = target_reward.shape[1]  

        # NOTE: deepcopy reward part of data is very important,
        original_reward = np.reshape(np.array(target_reward, dtype=np.float32), (batch_size * T, 1))
        obs_batch_tmp = self._flatten_obs_batch(obs_batch_ori)
        input_data = copy.deepcopy(self._prepare_inputs_from_obs(obs_batch_tmp))
        input_data = self._normalize_inputs(input_data)
        extrinsic_normalized = self._normalize_ext_rewards(original_reward)
        if self.frame_stack_num > 1:
            inputs = torch.zeros((batch_size, T, self.frame_stack_num, *self.input_shape[1:]), device=input_data.device, dtype=input_data.dtype)
            input_data = input_data.reshape(batch_size, -1, input_data.shape[-2], input_data.shape[-1])
            for j in range(T):
                inputs[:,j] = input_data[:, j:j+self.frame_stack_num]
            inputs = inputs.reshape(batch_size*T, *inputs.shape[2:])
        else:
            inputs = input_data
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(inputs)
            # mse = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=-1)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').sum(dim=-1) / 2
        
        rnd_reward_tensor = self._normalize_intrinsic_rewards(mse)
        self._running_mean_std_rnd_reward.update(mse.cpu().numpy())
        
        rnd_reward_np = rnd_reward_tensor.cpu().numpy()
        
        self.estimate_cnt_rnd += 1
        if self._tb_logger:
            self._tb_logger.add_scalar('rnd_reward_model/intrinsic_reward_max', rnd_reward_np.max(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/intrinsic_reward_mean', rnd_reward_np.mean(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/intrinsic_reward_min', rnd_reward_np.min(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/intrinsic_reward_std', rnd_reward_np.std(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/extrinsic_reward_max', extrinsic_normalized.max(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/extrinsic_reward_mean', extrinsic_normalized.mean(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/extrinsic_reward_min', extrinsic_normalized.min(), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/extrinsic_reward_std', extrinsic_normalized.std(), self.estimate_cnt_rnd)
                
        cur_w = self._intrinsic_weight(self.estimate_cnt_rnd)
        if self._tb_logger is not None:
            self._tb_logger.add_scalar('rnd_reward_model/intrinsic_weight', cur_w, self.estimate_cnt_rnd)

        rnd_reward_flat = rnd_reward_np.reshape(batch_size * T, 1)
        if self.intrinsic_reward_type == 'add':
            target_reward_augmented = extrinsic_normalized + rnd_reward_flat * cur_w
        elif self.intrinsic_reward_type == 'new':
            target_reward_augmented = rnd_reward_flat * cur_w
        elif self.intrinsic_reward_type == 'assign':
            target_reward_augmented = rnd_reward_flat
        else:
            target_reward_augmented = extrinsic_normalized

        if self._tb_logger is not None:
            self._tb_logger.add_scalar('rnd_reward_model/augmented_reward_max', np.max(target_reward_augmented), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/augmented_reward_mean', np.mean(target_reward_augmented),self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/augmented_reward_min', np.min(target_reward_augmented), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('rnd_reward_model/augmented_reward_std', np.std(target_reward_augmented), self.estimate_cnt_rnd)

        target_reward_augmented = np.reshape(target_reward_augmented, (batch_size, T, 1))
        return target_reward_augmented.reshape(batch_size, T)
    
    def state_dict(self) -> Dict:
        return self.reward_model.state_dict()

    def load_state_dict(self, _state_dict: Dict) -> None:
        self.reward_model.load_state_dict(_state_dict)

    def clear_data(self):
        pass

    def train(self):
        pass

    def collect_data(self, data) -> None:
        pass
        
    # ---------------------- 可视化辅助（新增） ---------------------- #
    def _select_peaks(self, y: np.ndarray, k: int) -> List[int]:
        order = np.argsort(-y)
        picked: List[int] = []
        for i in order:
            if len(picked) >= k:
                break
            picked.append(int(i))
        picked.sort()
        return picked

    def _obs_to_rgb(self, obs_any: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_any)
        img = x.squeeze()

        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img
    
    def UpdateFuncAnimation(self, all_obs_per_episode: List[np.ndarray]) -> None:
        """
        Overview:
            给定一条 episode 的完整 obs 序列（list，每个元素为 (C,H,W) 或 (H,W,C) 的 numpy 数组），
            使用当前 RND 模型重新计算这一条轨迹上每一步的 intrinsic reward，
            并画出：
                - 上方：若干关键帧（intrinsic reward 较大的若干步的观测）
                - 下方：整条时间线上的 intrinsic reward 曲线

            图像会被写入 TensorBoard（若 _tb_logger 不为 None）：
                tag = "rnd_visual/episode_intrinsic_timeline"
                step = self.estimate_cnt_rnd
        """
        if not all_obs_per_episode or len(self.input_shape)!= 3:
            return

        if not getattr(self, 'enable_image_logging', False) or self._tb_logger is None:
            return      

        obs_array = np.stack(all_obs_per_episode, axis=0)  # (T, C, H, W) 或 (T, H, W, C)
        flat_obs = self._flatten_obs_batch(obs_array)  # (T, *obs_shape)
        if flat_obs.size == 0:
            return
        # 3) 准备输入 + 归一化（与 estimate 中逻辑一致）
        inputs = self._prepare_inputs_from_obs(flat_obs)

        # 更新输入 running mean/std，再做标准化
        norm_inputs = self._normalize_inputs(inputs.clone())
        inputs = norm_inputs.reshape(*obs_array.shape)
        # 4) 通过 RND 模型得到每一步 intrinsic reward（MSE）
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(inputs)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').sum(dim=-1) / 2  # (T,)

        rnd_reward_tensor = self._normalize_intrinsic_rewards(mse)
        rnd_rewards = rnd_reward_tensor.cpu().numpy().reshape(-1)  # (T,)

        T = rnd_rewards.shape[0]
        steps = np.arange(T, dtype=np.int32)
        # 5) 选出若干“峰值位置”，对应关键帧
        k_cfg = int(getattr(self.cfg, 'peaks_topk', 10))
        k = max(1, min(k_cfg, T)) 
        peak_indices = self._select_peaks(rnd_rewards, k=k)  
        # 6) 把对应 obs 转成 RGB / Gray 图像
        frames: List[np.ndarray] = []
        for idx in peak_indices:
            if self.frame_stack_num > 1:
                frames.append(self._obs_to_rgb(obs_array[idx][-1]))
            else:
                frames.append(self._obs_to_rgb(flat_obs[idx]))
            
        # 7) 画图：上方一行 key frames，下方一行 reward 曲线
        ncols = k
        fig = plt.figure(figsize=(ncols * 1.5, 3.8 + 1.5), dpi=120)
        gs = fig.add_gridspec(2, ncols, height_ratios=[1, 2])

        # 上排：关键帧
        for i in range(k):
            ax = fig.add_subplot(gs[0, i])
            img = frames[i]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = img.transpose(1, 2, 0)
                ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(f"t={peak_indices[i]}", fontsize=8, pad=2)

        # 下排：时间线曲线
        ax_line = fig.add_subplot(gs[1, :])
        ax_line.plot(steps, rnd_rewards, linewidth=1.0)
        ax_line.set_xlabel("Episode step")
        ax_line.set_ylabel("Intrinsic reward")

        for i, idx in enumerate(peak_indices):
            ax_line.scatter([steps[idx]], [rnd_rewards[idx]], s=14)
            ax_line.annotate(
                str(i + 1),
                (steps[idx], rnd_rewards[idx]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

        fig.tight_layout()

        # 8) 写进 TensorBoard
        global_step = int(self.estimate_cnt_rnd)
        self._tb_logger.add_figure("rnd_visual/episode_intrinsic_timeline", fig, global_step)
        plt.close(fig)

        logging.info(
            "[RND] UpdateFuncAnimation: logged episode intrinsic timeline | T=%d | peaks=%d | step=%d",
            T, k, global_step,
        )

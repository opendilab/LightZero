import copy
import random
from typing import Union, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.model import FCEncoder, ConvEncoder
from ding.reward_model.base_reward_model import BaseRewardModel
from ding.torch_utils.data_helper import to_tensor
from ding.utils import RunningMeanStd
from ding.utils import SequenceType, REWARD_MODEL_REGISTRY, allreduce, synchronize, get_rank, get_world_size, build_logger, broadcast
from easydict import EasyDict


class RNDNetwork(nn.Module):

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType) -> None:
        super(RNDNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.target = FCEncoder(obs_shape, hidden_size_list)
            self.predictor = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.target = ConvEncoder(obs_shape, hidden_size_list)
            self.predictor = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
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
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.target = FCEncoder(obs_shape, hidden_size_list)
            self.predictor = FCEncoder(latent_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.target = ConvEncoder(obs_shape, hidden_size_list)
            self.predictor = ConvEncoder(latent_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own RND model".
                format(obs_shape)
            )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.representation_network(obs)
        predict_feature = self.predictor(z)
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
        3  | ``learning_rate``   float    0.001        | The step size of gradient descent      |
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
    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='rnd',
        # (str) The intrinsic reward type, including add, new, or assign.
        intrinsic_reward_type='add',
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (float) Batch size.
        batch_size=64,
        # (list(int)) Sequence of ``hidden_size`` of reward network.
        # If obs.shape == 1,  use MLP layers.
        # If obs.shape == 3,  use conv layer and final dense layer.
        hidden_size_list=[64, 64, 128],
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (bool) Observation normalization: transform obs to mean 0, std 1.
        input_norm=True,
        # (int) Min clip value for observation normalization.
        input_norm_clamp_min=-1,
        # (int) Max clip value for observation normalization.
        input_norm_clamp_max=1,
        # Means the relative weight of RND intrinsic_reward.
        # (float) The weight of intrinsic reward
        # r = intrinsic_reward_weight * r_i + r_e.
        intrinsic_reward_weight=0.01,
        # (bool) Whether to normalize extrinsic reward.
        # Normalize the reward to [0, extrinsic_reward_norm_max].
        extrinsic_reward_norm=True,
        # (int) The upper bound of the reward normalization.
        extrinsic_reward_norm_max=1,
    )

    def __init__(self, config: EasyDict, device: str = 'cpu', tb_logger: 'SummaryWriter' = None, exp_name: str = "default_experiment",
                instance_name: str = 'RNDModel', representation_network: nn.Module = None, 
                target_representation_network: nn.Module = None, use_momentum_representation_network: bool = True, 
                bp_update_sync: bool = True, multi_gpu: bool = False) -> None:  # noqa
        super(RNDRewardModel, self).__init__()
        self.cfg = config
        self.representation_network = representation_network
        self.target_representation_network = target_representation_network
        self.use_momentum_representation_network = use_momentum_representation_network
        self.input_type = self.cfg.input_type
        self._bp_update_sync = bp_update_sync
        self.multi_gpu = multi_gpu
        assert self.input_type in ['obs', 'latent_state', 'obs_latent_state'], self.input_type
        self.rnd_buffer_size = config.rnd_buffer_size
        self.intrinsic_reward_type = self.cfg.intrinsic_reward_type
        
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._rank = get_rank()
        self._world_size = get_world_size()
        if self.multi_gpu:
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if 'cuda' in device else 'cpu'
        else:
            self._device = device
            
        print('rnd.device:' + self._device)
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name),
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = None
        
        if self.input_type == 'obs':
            self.input_shape = self.cfg.obs_shape
            self.reward_model = RNDNetwork(self.input_shape, self.cfg.hidden_size_list).to(self._device)
        elif self.input_type == 'latent_state':
            self.input_shape = self.cfg.latent_state_dim
            self.reward_model = RNDNetwork(self.input_shape, self.cfg.hidden_size_list).to(self._device)
        elif self.input_type == 'obs_latent_state':
            if self.use_momentum_representation_network:
                self.reward_model = RNDNetworkRepr(self.cfg.obs_shape, self.cfg.latent_state_dim, self.cfg.hidden_size_list[0:-1],
                                                  self.target_representation_network).to(self._device)
            else:
                self.reward_model = RNDNetworkRepr(self.cfg.obs_shape, self.cfg.latent_state_dim, self.cfg.hidden_size_list[0:-1],
                                                  self.representation_network).to(self._device)

        if self.multi_gpu and self._world_size > 1:
            with torch.no_grad():
                for _, p in self.reward_model.target.state_dict().items():
                    broadcast(p.data, 0)
                for _, p in self.reward_model.predictor.state_dict().items():
                    broadcast(p.data, 0)
        
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        if self.input_type in ['obs', 'obs_latent_state']:
            self.train_obs = []
        if self.input_type == 'latent_state':
            self.train_latent_state = []

        self._optimizer_rnd = torch.optim.Adam(
            self.reward_model.predictor.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )

        self._running_mean_std_rnd_reward = RunningMeanStd(epsilon=1e-4)
        self._running_mean_std_rnd_obs = RunningMeanStd(epsilon=1e-4)
        self.estimate_cnt_rnd = 0
        self.train_cnt_rnd = 0
        
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
            
    def _train_with_data_one_step(self) -> None:
        if self.input_type in ['obs', 'obs_latent_state']:
            train_data = random.sample(self.train_obs, self.cfg.batch_size)
        elif self.input_type == 'latent_state':
            train_data = random.sample(self.train_latent_state, self.cfg.batch_size)

        train_data = torch.stack(train_data).to(self._device)

        if self.cfg.input_norm:
            # Note: observation normalization: transform obs to mean 0, std 1
            self._running_mean_std_rnd_obs.update(train_data.detach().cpu().numpy())
            normalized_train_data = (train_data - to_tensor(self._running_mean_std_rnd_obs.mean).to(
                self._device)) / to_tensor(
                self._running_mean_std_rnd_obs.std
            ).to(self._device)
            train_data = torch.clamp(normalized_train_data, min=self.cfg.input_norm_clamp_min,
                                     max=self.cfg.input_norm_clamp_max)

        predict_feature, target_feature = self.reward_model(train_data)
        loss = F.mse_loss(predict_feature, target_feature)
        
        if self._tb_logger is not None:
            self._tb_logger.add_scalar('rnd_reward_model/rnd_mse_loss', loss, self.train_cnt_rnd)
        self._optimizer_rnd.zero_grad()
        loss.backward()
        
        if self.multi_gpu:
            self.sync_gradients(self.reward_model.predictor)
        self._optimizer_rnd.step()

    def train_with_data(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            # for name, param in self.reward_model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {torch.isnan(param.grad).any()}, {torch.isinf(param.grad).any()}")
            #         print(f"{name}: grad min: {param.grad.min()}, grad max: {param.grad.max()}")
            # # enable the following line to check whether there is nan or inf in the gradient.
            # torch.autograd.set_detect_anomaly(True)
            self._train_with_data_one_step()
            self.train_cnt_rnd += 1

    def estimate(self, data: list) -> List[Dict]:
        """
        Rewrite the reward key in each row of the data.
        """
        # current_batch, target_batch = data
        # obs_batch_orig, action_batch, mask_batch, indices, weights, make_time = current_batch
        # target_reward, target_value, target_policy = target_batch
        obs_batch_orig = data[0][0]
        target_reward = data[1][0]
        batch_size = obs_batch_orig.shape[0]
        T = target_reward.shape[1]  

        obs_batch_tmp = np.reshape(obs_batch_orig, (batch_size * T, self.cfg.obs_shape))

        if self.input_type == 'latent_state':
            with torch.no_grad():
                latent_state = self.representation_network(torch.from_numpy(obs_batch_tmp).to(self._device))
            input_data = latent_state
        elif self.input_type in ['obs', 'obs_latent_state']:
            input_data = to_tensor(obs_batch_tmp).to(self._device)

        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        target_reward_augmented = copy.deepcopy(target_reward)
        target_reward_augmented = np.reshape(target_reward_augmented, (batch_size * T, 1))

        if self.cfg.input_norm:
            # add this line to avoid inplace operation on the original tensor.
            input_data = input_data.clone()
            # Note: observation normalization: transform obs to mean 0, std 1
            input_data = (input_data - to_tensor(self._running_mean_std_rnd_obs.mean
                                                 ).to(self._device)) / to_tensor(self._running_mean_std_rnd_obs.std).to(
                self._device)
            input_data = torch.clamp(input_data, min=self.cfg.input_norm_clamp_min, max=self.cfg.input_norm_clamp_max)
        else:
            input_data = input_data
        with torch.no_grad():
            predict_feature, target_feature = self.reward_model(input_data)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)
            self._running_mean_std_rnd_reward.update(mse.detach().cpu().numpy())

            # Note: according to the min-max normalization, transform rnd reward to [0,1]
            rnd_reward = (mse - mse.min()) / (mse.max() - mse.min() + 1e-6)

            # save the rnd_reward statistics into tb_logger
            self.estimate_cnt_rnd += 1
            if self._tb_logger is not None:
                self._tb_logger.add_scalar('rnd_reward_model/rnd_reward_max', rnd_reward.max(), self.estimate_cnt_rnd)
                self._tb_logger.add_scalar('rnd_reward_model/rnd_reward_mean', rnd_reward.mean(), self.estimate_cnt_rnd)
                self._tb_logger.add_scalar('rnd_reward_model/rnd_reward_min', rnd_reward.min(), self.estimate_cnt_rnd)
                self._tb_logger.add_scalar('rnd_reward_model/rnd_reward_std', rnd_reward.std(), self.estimate_cnt_rnd)

        rnd_reward = rnd_reward.to(self._device).unsqueeze(1).cpu().numpy()
        if self.intrinsic_reward_type == 'add':
            if self.cfg.extrinsic_reward_norm:
                target_reward_augmented = target_reward_augmented / self.cfg.extrinsic_reward_norm_max + rnd_reward * self.cfg.intrinsic_reward_weight
            else:
                target_reward_augmented = target_reward_augmented + rnd_reward * self.cfg.intrinsic_reward_weight
        elif self.intrinsic_reward_type == 'new':
            if self.cfg.extrinsic_reward_norm:
                target_reward_augmented = target_reward_augmented / self.cfg.extrinsic_reward_norm_max
        elif self.intrinsic_reward_type == 'assign':
            target_reward_augmented = rnd_reward

        if self._tb_logger is not None:
            self._tb_logger.add_scalar('augmented_reward/reward_max', np.max(target_reward_augmented), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('augmented_reward/reward_mean', np.mean(target_reward_augmented),
                                    self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('augmented_reward/reward_min', np.min(target_reward_augmented), self.estimate_cnt_rnd)
            self._tb_logger.add_scalar('augmented_reward/reward_std', np.std(target_reward_augmented), self.estimate_cnt_rnd)

        # reshape to (target_reward_augmented.shape[0], T, 1)
        target_reward_augmented = np.reshape(target_reward_augmented, (batch_size, T, 1))
        # TODO why? batchsizw * T -> batchsize * T * 1
        data[1][0] = target_reward_augmented
        train_data_augmented = data

        return train_data_augmented

    def collect_data(self, data: list) -> None:
        segments = [game_segment.obs_segment for game_segment in data[0] if len(game_segment.obs_segment)]
        if not segments:
            return
        collected_transitions = np.concatenate(segments, axis=0)
        if self.input_type == 'latent_state':
            with torch.no_grad():
                self.train_latent_state.extend(
                    self.representation_network(torch.from_numpy(collected_transitions).to(self._device)))
        elif self.input_type == 'obs':
            self.train_obs.extend(to_tensor(collected_transitions).to(self._device))
        elif self.input_type == 'obs_latent_state':
            self.train_obs.extend(to_tensor(collected_transitions).to(self._device))

    def clear_old_data(self) -> None:
        if self.input_type == 'latent_state':
            if len(self.train_latent_state) >= self.cfg.rnd_buffer_size:
                self.train_latent_state = self.train_latent_state[-self.cfg.rnd_buffer_size:]
        elif self.input_type == 'obs':
            if len(self.train_obs) >= self.cfg.rnd_buffer_size:
                self.train_obs = self.train_obs[-self.cfg.rnd_buffer_size:]
        elif self.input_type == 'obs_latent_state':
            if len(self.train_obs) >= self.cfg.rnd_buffer_size:
                self.train_obs = self.train_obs[-self.cfg.rnd_buffer_size:]

    def state_dict(self) -> Dict:
        return self.reward_model.state_dict()

    def load_state_dict(self, _state_dict: Dict) -> None:
        self.reward_model.load_state_dict(_state_dict)

    def clear_data(self):
        pass

    def train(self):
        pass

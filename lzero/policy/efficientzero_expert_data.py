import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import treetensor.torch as ttorch
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.rl_utils import get_nstep_return_data, get_train_sample
from ding.torch_utils import to_tensor, to_device, to_ndarray
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss

# python mcts
import lzero.rl_utils.mcts.ptree_efficientzero as ptree
from lzero.rl_utils import EfficientZeroMCTSPtree as MCTSPtree
from lzero.rl_utils import MCTSCtree
# cpp mcts
from lzero.rl_utils import cytree as ctree
from lzero.rl_utils import scalar_transform, inverse_scalar_transform
from lzero.rl_utils import select_action
from lzero.rl_utils import Transforms, visit_count_temperature, modified_cross_entropy_loss, value_phi, reward_phi, DiscreteSupport


@POLICY_REGISTRY.register('efficientzero_expert_data')
class EfficientZeroExertDataPolicy(Policy):
    """
    Overview:
        The policy class for EfficientZero
    """
    config = dict(
        type='efficientzero_expert_data',
        # (bool) Whether use cuda in policy
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy)
        on_policy=False,
        # (bool) Whether enable priority experience sample
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns
        discount_factor=0.97,
        # (int) The number of step for calculating target q_value
        nstep=1,
        model=dict(
            observation_shape=(12, 96, 96),
            action_space_size=6,
            num_blocks=1,
            num_channels=64,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            downsample=True,
            lstm_hidden_size=512,
            bn_mt=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            last_linear_layer_init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=10,
            # (int) How many samples in a training batch
            batch_size=256,
            # (float) The step size of gradient descent
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=200,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            weight_decay=1e-4,
            momentum=0.9,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=8,
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, type='game')
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For DQN, ``ding.model.template.q_learning.DQN``
        """
        return 'EfficientZeroNet', ['lzero.model.efficientzero.efficientzero_model']

    def _init_learn(self) -> None:
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            momentum=self._cfg.learn.momentum,
            weight_decay=self._cfg.learn.weight_decay,
            # grad_clip_type=self._cfg.learn.grad_clip_type,
            # clip_value=self._cfg.learn.grad_clip_value,
        )
        # self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        # self._learn_model = self._model
        self._learn_model.reset()
        self._target_model.reset()
        if self._cfg.use_augmentation:
            self.transforms = Transforms(
                self._cfg.augmentation, image_shape=(self._cfg.obs_shape[1], self._cfg.obs_shape[2])
            )
        self.value_support = DiscreteSupport(-self._cfg.support_size, self._cfg.support_size, delta=1)
        self.reward_support = DiscreteSupport(-self._cfg.support_size, self._cfg.support_size, delta=1)

    # @profile
    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        self._target_model.train()

        # TODO(pu): priority
        inputs_batch, targets_batch, replay_buffer = data

        obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
        target_value_prefix, target_value, target_policy = targets_batch

        # [:, 0: config.frame_stack_num * 3,:,:]
        # obs_batch_ori is the original observations in a batch
        # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
        # obs_target_batch is the observations for s_t (hidden states from representation function)

        # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames

        if self._cfg.image_based:
            obs_batch_ori = torch.from_numpy(obs_batch_ori / 255.0).to(self._cfg.device).float()
        else:
            obs_batch_ori = torch.from_numpy(obs_batch_ori).to(self._cfg.device).float()

        # collector data process:
        # (batch_size, stack_num+num_unroll_steps, W, H, C) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H )

        # e.g. in pong: stack_num=4, num_unroll_steps=5
        # (4, 9, 96, 96, 3) -> (4, 9*3, 96, 96) = 4,27,96,96

        # in the second dim:
        # timestep t: 1,2,3,4,5,6,7,8,9
        # channel_num:    3    3     3   3     3    3    3   3       3
        #                ---, ---, ---, ---,  ---, ---, ---, ---,   ---

        # (4, 4*3, 96, 96) = (4, 12, 96, 96)
        # take the first stacked obs at timestep t: o_t_stack
        # used in initial_inference
        obs_batch = obs_batch_ori[:, 0:self._cfg.frame_stack_num * self._cfg.image_channel, :, :]

        # take the all obs other than timestep t1:
        # obs_target_batch is used for calculate consistency loss, which is only performed in the last 8 timesteps
        # for i in rnage(num_unroll_steeps):
        #   beg_index = self._cfg.image_channel * step_i
        #   end_index = self._cfg.image_channel * (step_i + self._cfg.frame_stack_num)
        obs_target_batch = obs_batch_ori[:, self._cfg.image_channel:, :, :]

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.transforms.transform(obs_batch)
            obs_target_batch = self.transforms.transform(obs_target_batch)

        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        mask_batch = torch.from_numpy(mask_batch).to(self._cfg.device).float()
        target_value_prefix = torch.from_numpy(target_value_prefix.astype('float64')).to(self._cfg.device).float()
        target_value = torch.from_numpy(target_value.astype('float64')).to(self._cfg.device).float()
        target_policy = torch.from_numpy(target_policy).to(self._cfg.device).float()
        weights = torch.from_numpy(weights_lst).to(self._cfg.device).float()

        # TODO
        target_value_prefix = target_value_prefix.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        batch_size = obs_batch.size(0)
        assert batch_size == self._cfg.batch_size == target_value_prefix.size(0)
        metric_loss = torch.nn.L1Loss()

        # some logs preparation
        other_log = {}
        other_dist = {}

        other_loss = {
            'l1': -1,
            'l1_1': -1,
            'l1_-1': -1,
            'l1_0': -1,
        }
        for i in range(self._cfg.num_unroll_steps):
            key = 'unroll_' + str(i + 1) + '_l1'
            other_loss[key] = -1
            other_loss[key + '_1'] = -1
            other_loss[key + '_-1'] = -1
            other_loss[key + '_0'] = -1

        # transform targets to categorical representation
        transformed_target_value_prefix = scalar_transform(target_value_prefix, self._cfg.support_size)
        target_value_prefix_phi = reward_phi(self.reward_support, transformed_target_value_prefix)

        transformed_target_value = scalar_transform(target_value, self._cfg.support_size)
        target_value_phi = value_phi(self.value_support, transformed_target_value)

        network_output = self._learn_model.initial_inference(obs_batch)

        value = network_output.value
        value_prefix = network_output.value_prefix
        hidden_state = network_output.hidden_state  # （2, 64, 6, 6）
        reward_hidden_state = network_output.reward_hidden_state  # {tuple:2} (1,2,512)
        policy_logits = network_output.policy_logits  # {list: 2} {list:6}

        reward_hidden_state = to_device(reward_hidden_state, self._cfg.device)
        scaled_value = inverse_scalar_transform(value, self._cfg.support_size)

        # TODO(pu)
        if not self._learn_model.training:
            # if not in training, obtain the scalars of the value/reward
            scaled_value = scaled_value.detach().cpu().numpy()
            scaled_value_prefix = inverse_scalar_transform(value_prefix, self._cfg.support_size).detach().cpu().numpy()
            hidden_state = hidden_state.detach().cpu().numpy()
            reward_hidden_state = (
                reward_hidden_state[0].detach().cpu().numpy(), reward_hidden_state[1].detach().cpu().numpy()
            )
            policy_logits = policy_logits.detach().cpu().numpy()

        if self._cfg.monitor_statistics:
            state_lst = hidden_state.detach().cpu().numpy()

        predicted_value_prefixs = []
        # Note: Following line is just for logging.
        if self._cfg.monitor_statistics:
            predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition
        value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + self._cfg.prioritized_replay_eps

        # calculate loss for the first step
        value_loss = modified_cross_entropy_loss(value, target_value_phi[:, 0])
        policy_loss = modified_cross_entropy_loss(policy_logits, target_policy[:, 0])
        value_prefix_loss = torch.zeros(batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(batch_size, device=self._cfg.device)

        target_value_prefix_cpu = target_value_prefix.detach().cpu()
        gradient_scale = 1 / self._cfg.num_unroll_steps

        # loss of the unrolled steps
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function
            network_output = self._learn_model.recurrent_inference(
                hidden_state, reward_hidden_state, action_batch[:, step_i]
            )
            value = network_output.value
            value_prefix = network_output.value_prefix
            policy_logits = network_output.policy_logits  # {list: 2} {list:6}
            hidden_state = network_output.hidden_state  # （2, 64, 6, 6）
            reward_hidden_state = network_output.reward_hidden_state  # {tuple:2} (1,2,512)

            # TODO(pu)
            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                value = inverse_scalar_transform(value, self._cfg.support_size).detach().cpu().numpy()
                value_prefix = inverse_scalar_transform(value_prefix, self._cfg.support_size).detach().cpu().numpy()
                hidden_state = hidden_state.detach().cpu().numpy()
                reward_hidden_state = (
                    reward_hidden_state[0].detach().cpu().numpy(), reward_hidden_state[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy()

            beg_index = self._cfg.image_channel * step_i
            end_index = self._cfg.image_channel * (step_i + self._cfg.frame_stack_num)

            # consistency loss
            if self._cfg.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                presentation_state = network_output.hidden_state

                hidden_state = to_tensor(hidden_state)
                presentation_state = to_tensor(presentation_state)

                # no grad for the presentation_state branch
                dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                observation_proj = self._learn_model.project(presentation_state, with_grad=False)
                temp_loss = self._consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            # the target policy, target_value_phi, target_value_prefix_phi is calculated in game buffer now
            policy_loss += modified_cross_entropy_loss(policy_logits, target_policy[:, step_i + 1])
            value_loss += modified_cross_entropy_loss(value, target_value_phi[:, step_i + 1])
            value_prefix_loss += modified_cross_entropy_loss(value_prefix, target_value_prefix_phi[:, step_i])

            # Follow MuZero, set half gradient
            # hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % self._cfg.lstm_horizon_len == 0:
                reward_hidden_state = (
                    torch.zeros(1, self._cfg.batch_size, self._cfg.lstm_hidden_size).to(self._cfg.device),
                    torch.zeros(1, self._cfg.batch_size, self._cfg.lstm_hidden_size).to(self._cfg.device)
                )

            if self._cfg.monitor_statistics:
                scaled_value_prefixs = inverse_scalar_transform(value_prefix.detach(), self._cfg.support_size)
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, inverse_scalar_transform(value, self._cfg.support_size).detach().cpu())
                )
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_1],
                        target_value_prefix_base[value_prefix_indices_1]
                    )
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_n1],
                        target_value_prefix_base[value_prefix_indices_n1]
                    )
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_0],
                        target_value_prefix_base[value_prefix_indices_0]
                    )
        # ----------------------------------------------------------------------------------
        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
            self._cfg.consistency_coeff * consistency_loss + self._cfg.policy_loss_coeff * policy_loss +
            self._cfg.value_loss_coeff * value_loss + self._cfg.reward_loss_coeff * value_prefix_loss
        )
        weighted_loss = (weights * loss).mean()

        # backward
        parameters = self._learn_model.parameters()

        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, self._cfg.max_grad_norm)
        self._optimizer.step()

        # =============
        # target model update
        # =============
        self._target_model.update(self._learn_model.state_dict())

        # ----------------------------------------------------------------------------------
        # update priority
        # priority_info = {'indices':indices, 'make_time':make_time, 'batch_priorities':value_priority}
        replay_buffer.batch_update(indices=indices, metas={'make_time': make_time, 'batch_priorities': value_priority})

        # packing data for logging
        loss_data = (
            total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
            value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean()
        )
        if self._cfg.monitor_statistics:

            # reward l1 loss
            value_prefix_indices_0 = (
                target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0
            )
            value_prefix_indices_n1 = (
                target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1
            )
            value_prefix_indices_1 = (
                target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1
            )

            target_value_prefix_base = target_value_prefix_cpu[:, :self._cfg.num_unroll_steps].reshape(-1).unsqueeze(-1)

            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
            other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
            if value_prefix_indices_1.any():
                other_loss['l1_1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1]
                )
            if value_prefix_indices_n1.any():
                other_loss['l1_-1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1]
                )
            if value_prefix_indices_0.any():
                other_loss['l1_0'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0]
                )

            td_data = (
                value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst, other_loss,
                other_log, other_dist
            )
            priority_data = (weights, indices)
        else:
            td_data, priority_data = None, None

        return {
            # 'priority':priority_info,
            'total_loss': loss_data[0],
            'weighted_loss': loss_data[1],
            'loss_mean': loss_data[2],
            'policy_loss': loss_data[4],
            'value_prefix_loss': loss_data[5],
            'value_loss': loss_data[6],
            'consistency_loss': loss_data[7],
            'value_priority': td_data[0].flatten().mean().item(),
            'target_value_prefix': td_data[1].flatten().mean().item(),
            'target_value': td_data[2].flatten().mean().item(),
            'predicted_value_prefixs': td_data[7].flatten().mean().item(),
            'predicted_values': td_data[8].flatten().mean().item(),
            # 'target_policy':td_data[9],
            # 'predicted_policies':td_data[10]
            # 'td_data': td_data,
            # 'priority_data_weights': priority_data[0],
            # 'priority_data_indices': priority_data[1]
        }

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        # self._collect_model = model_wrap(self._model, 'base')
        self._collect_model = self._learn_model
        self._collect_model.reset()
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)

        # set temperature for distributions
        self.collect_temperature = np.array(
            [
                visit_count_temperature(
                    self._cfg.auto_temperature,
                    self._cfg.fixed_temperature_value,
                    self._cfg.max_training_steps,
                    trained_steps=0
                ) for _ in range(self._cfg.collector_env_num)
            ]
        )

    # @profile
    def _forward_collect(
        self, data: ttorch.Tensor, action_mask: list = None, temperature: list = None, to_play=None, ready_env_id=None
    ):
        """
        Shapes:
            obs: (B, S, C, H, W), where S is the stack num
            temperature: (N1, ), where N1 is the number of collect_env.
        """
        #######################
        # expert_data related
        #######################
        self._collect_model.eval()
        stack_obs = data
        active_collect_env_num = stack_obs.shape[0]
        data_id = [i for i in range(active_collect_env_num)]
        output = {i: None for i in data_id}

        eps = 0
        # dqn only use the current obs
        data = data[-1][-4:].unsqueeze(0)

        if self._cuda:
            data = to_device(data, self._cfg.device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        action_num = int(action_mask[0].sum())
        distributions = np.ones(action_num, dtype=np.float32)
        visit_entropy = np.ones(1, dtype=np.float32)
        value = np.ones(1, dtype=np.float32)
        pred_values_pool = np.ones(1, dtype=np.float32)

        # TODO
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)

        for i, env_id in enumerate(ready_env_id):
            output[env_id] = {
                'action': int(to_ndarray(output['action'])),
                'distributions': distributions,
                'visit_count_distribution_entropy': visit_entropy,
                'value': value,
                'pred_value': pred_values_pool,
                'policy_logits': output['logit'],
            }

        return output

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        # self._eval_model = self._learn_model
        # TODO(pu)
        self._eval_model = model_wrap(self._model, wrapper_name='base')

        self._eval_model.reset()
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)

    # @profile
    def _forward_eval(self, data: ttorch.Tensor, action_mask: list, to_play: None, ready_env_id=None):
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        """
        self._eval_model.eval()
        stack_obs = data
        active_eval_env_num = stack_obs.shape[0]
        with torch.no_grad():
            # stack_obs shape [B, S x C, W, H] e.g. {Tensor:(B,12,96,96)}
            network_output = self._eval_model.initial_inference(stack_obs)
            hidden_state_roots = network_output.hidden_state  # for atari, shape（B, 64, 6, 6）
            pred_values_pool = network_output.value  # for atari, shape（B, 601）
            reward_hidden_roots = network_output.reward_hidden_state  # {tuple:2} each element (1,2,512)
            value_prefix_pool = network_output.value_prefix  # shape（B, 1）
            policy_logits_pool = network_output.policy_logits  # shape（B, A）

            # TODO(pu)
            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values_pool = inverse_scalar_transform(pred_values_pool, self._cfg.support_size
                                                            ).detach().cpu().numpy()  # shape（B, 1）
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()  # list shape（B, A）

            if self._cfg.mcts_ctree:
                # cpp mcts
                # TODO(pu): for board games, when action_num is a list, adapt the Roots method
                action_num = int(action_mask[0].sum())
                roots = ctree.Roots(active_eval_env_num, action_num, self._cfg.num_simulations)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                # do MCTS for a policy (argmax in testing)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots)
            else:
                # python mcts
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                ]
                roots = ptree_efficientzero.Roots(active_eval_env_num, legal_actions, self._cfg.num_simulations)

                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                # do MCTS for a policy (argmax in testing)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)

            # root visit count
            roots_distributions = roots.get_distributions()  # {list: 1} each element {list:6}
            roots_values = roots.get_values()  # {list: 1}
            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_distributions[i], roots_values[i]
                # select the argmax, not sampling
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # TODO(pu): transform to the real action index in legal action set
                action = np.where(action_mask[i] == 1.0)[0][action]
                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values_pool[i],
                    'policy_logits': policy_logits_pool[i],
                }

        return output

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'total_loss',
            'weighted_loss',
            'loss_mean',
            'policy_loss',
            'value_prefix_loss',
            'value_loss',
            'consistency_loss',
            #
            'value_priority',
            'target_value_prefix',
            'target_value',
            'predicted_value_prefixs',
            'predicted_values',

            # 'visit_count_distribution_entropy',
            # 'target_policy',
            # 'predicted_policies'
            # 'td_data',
            # 'priority_data_weights',
            # 'priority_data_indices'
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _process_transition(
            self, obs: ttorch.Tensor, policy_output: ttorch.Tensor, timestep: ttorch.Tensor
    ) -> ttorch.Tensor:
        return ttorch.as_tensor(
            {
                'obs': obs,
                'action': policy_output.action,
                'distribution': policy_output.distribution,
                'value': policy_output.value,
                'next_obs': timestep.obs,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        )

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _data_preprocess_learn(self, data: ttorch.Tensor):
        # TODO data augmentation before learning
        data = data.cuda(self._cfg.device)
        data = ttorch.stack(data)
        return data

    @staticmethod
    def _consist_loss_func(f1, f2):
        """
        Consistency loss function: similarity loss
        """
        f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
        f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
        return -(f1 * f2).sum(dim=1)

    @staticmethod
    def _get_max_entropy(action_shape: int) -> None:
        p = 1.0 / action_shape
        return -action_shape * p * np.log2(p)
import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from ditk import logging
from torch.distributions import Categorical, Independent, Normal
from torch.nn import L1Loss

from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, ez_network_output_unpack, select_action
from .utils import negative_cosine_similarity


@POLICY_REGISTRY.register('sampled_efficientzero')
class SampledEfficientZeroPolicy(Policy):
    """
    Overview:
        The policy class for Sampled EfficientZero.
    """

    # The default_config for Sampled fEficientZero policy.
    config = dict(
        # (bool) ``sampled_algo=True`` means the policy is sampled-based algorithm (e.g. Sampled EfficientZero), which is used in ``collector``.
        sampled_algo=True,
        model=dict(
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) the stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            action_space_size=6,
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            # (int) the image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The scale of supports used in categorical distribution. Only effective when ``categorical_distribution=True``.
            support_scale=300,
            # (int) The hidden size in LSTM.
            lstm_hidden_size=512,
            # the sampled specific config
            sigma_type='conditioned',  # options={'conditioned', 'fixed'}
            fixed_sigma_value=0.3,
        ),
        ## common
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda in policy.
        cuda=True,
        collector_env_num=8,
        evaluator_env_num=3,
        # (str) The type of environment. options is ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of battle mode. options is ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment`` block.
        game_segment_length=200,

        ## observation
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) style of augmentation.
        augmentation=['shift', 'intensity'],

        ## learn
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
        update_per_collect=100,
        # (int) How many samples in a training batch.
        batch_size=256,
        cos_lr_scheduler=False,
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,  # init lr for manually decay schedule
        # optim_type='Adam',
        # lr_piecewise_constant_decay=False,
        # learning_rate=0.003,  # lr for Adam optimizer
        # (float) Weight uniform initialization range in the last output layer
        init_w=3e-3,
        normalize_prob_of_sampled_actions=False,
        policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        # (int) Frequency of target network update.
        target_update_freq=100,
        weight_decay=1e-4,
        momentum=0.9,
        grad_clip_value=10,
        # You can use either "n_sample" or "n_episode" in collector.collect.
        # Get "n_episode" episodes per collect.
        n_episode=8,
        # (float) the number of simulations in MCTS.
        num_simulations=50,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.997,
        # (int) The number of step for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        policy_entropy_loss_weight=0,

        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=2,  # NOTE: default is 2.
        # ``threshold_training_steps_for_final_lr`` is used for adjusting lr manually.
        # lr_piecewise_constant_decay: lr: 0.2 -> 0.02 -> 0.002
        threshold_training_steps_for_final_lr=int(1e5),
        # ``threshold_training_steps_for_final_temperature`` is used for adjusting temperature manually.
        # manual_temperature_decay: temperature: 1 -> 0.5 -> 0.25
        threshold_training_steps_for_final_temperature=int(1e5),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (float) ``fixed_temperature_value`` is effective only when manual_temperature_decay=False
        fixed_temperature_value=0.25,

        ## Priority
        # (bool) Whether to use priority when sampling from the buffer.
        use_priority=True,
        # (bool) Whether to use the maximum priority for new data.
        use_max_priority_for_new_data=True,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization, while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction, while a value of 1 means full correction.
        priority_prob_beta=0.4,

        ## UCB
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For Sampled EfficientZero, ``lzero.model.sampled_efficientzero_model.SampledEfficientZeroModel``
        """
        return 'SampledEfficientZeroModel', ['lzero.model.sampled_efficientzero_model']

    def _init_learn(self) -> None:
        if self._cfg.model.continuous_action_space:
            # Weight Init for the last output layer of gaussian policy head in prediction network.
            init_w = self._cfg.init_w
            self._model.prediction_network.sampled_fc_policy.mu.weight.data.uniform_(-init_w, init_w)
            self._model.prediction_network.sampled_fc_policy.mu.bias.data.uniform_(-init_w, init_w)
            self._model.prediction_network.sampled_fc_policy.log_sigma_layer.weight.data.uniform_(-init_w, init_w)
            try:
                self._model.prediction_network.sampled_fc_policy.log_sigma_layer.bias.data.uniform_(-init_w, init_w)
            except Exception as exception:
                logging.warning(exception)

        assert self._cfg.optim_type in ['SGD', 'Adam']
        if self._cfg.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay,
            )

        elif self._cfg.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._cfg.learning_rate, weight_decay=self._cfg.weight_decay
            )

        if self._cfg.cos_lr_scheduler is True:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.lr_scheduler = CosineAnnealingLR(self._optimizer, 1e6, eta_min=0, last_epoch=-1)

        if self._cfg.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.target_update_freq}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        self.value_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.reward_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    def _forward_learn(self, data: torch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        self._target_model.train()

        current_batch, targets_batch = data
        # ==============================================================
        # sampled related core code
        # ==============================================================
        obs_batch_ori, action_batch, child_sampled_actions_batch, mask_batch, indices, weights, make_time = current_batch
        target_value_prefix, target_value, target_policy = targets_batch

        """
        ``obs_batch_ori`` is the original observations in a batch style, shape is:
        (batch_size, stack_num+num_unroll_steps, W, H, C) -> (batch_size, (stack_num+num_unroll_steps)*C, W, H )

        e.g. in pong: stack_num=4, num_unroll_steps=5
        (4, 9, 96, 96, 3) -> (4, 9*3, 96, 96) = (4, 27, 96, 96)

        the second dim of ``obs_batch_ori``:
        timestep t:     1,   2,   3,  4,    5,   6,   7,   8,     9
        channel_num:    3    3    3   3     3    3    3    3      3
                       ---, ---, ---, ---,  ---, ---, ---, ---,   ---
        """
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(self._cfg.device).float()
        # ``obs_batch`` is used in ``initial_inference()``, which is the first stacked obs at timestep t in
        # ``obs_batch_ori``. shape is (4, 4*3, 96, 96) = (4, 12, 96, 96)
        obs_batch = obs_batch_ori[:, 0:self._cfg.model.frame_stack_num * self._cfg.model.image_channel, :, :]

        if self._cfg.model.self_supervised_learning_loss:
            # ``obs_target_batch`` is only used for calculate consistency loss, which take the all obs other than
            # timestep t1, and is only performed in the last 8 timesteps in the second dim in ``obs_batch_ori``.
            obs_target_batch = obs_batch_ori[:, self._cfg.model.image_channel:, :, :]

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if self._cfg.model.self_supervised_learning_loss:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .float(), in continuous action space.
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).float().unsqueeze(-1)
        data_list = [
            mask_batch,
            target_value_prefix.astype('float64'),
            target_value.astype('float64'), target_policy, weights
        ]
        [mask_batch, target_value_prefix, target_value, target_policy,
         weights] = to_torch_float_tensor(data_list, self._cfg.device)
        # ==============================================================
        # sampled related core code
        # ==============================================================
        # shape: (batch_size, num_unroll_steps+1, num_of_sampled_actions, action_dim, 1), e.g. (4, 6, 5, 1, 1)
        child_sampled_actions_batch = torch.from_numpy(child_sampled_actions_batch).to(self._cfg.device).unsqueeze(-1)

        target_value_prefix = target_value_prefix.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        assert obs_batch.size(0) == self._cfg.batch_size == target_value_prefix.size(0)

        # ``scalar_transform`` to transform the original value to the scaled value,
        # i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        transformed_target_value_prefix = scalar_transform(target_value_prefix)
        transformed_target_value = scalar_transform(target_value)
        # transform a scalar to its categorical_distribution. After this transformation, each scalar is
        # represented as the linear combination of its two adjacent supports.
        target_value_prefix_categorical = phi_transform(self.reward_support, transformed_target_value_prefix)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        # ==============================================================
        # the core initial_inference in SampledEfficientZero policy.
        # ==============================================================
        network_output = self._learn_model.initial_inference(obs_batch)
        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        hidden_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(network_output)

        # transform the scaled value or its categorical representation to its original value,
        # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        original_value = self.inverse_scalar_transform_handle(value)

        # Note: The following lines are just for logging.
        predicted_value_prefixs = []
        if self._cfg.monitor_extra_statistics:
            hidden_state_list = hidden_state.detach().cpu().numpy()
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition.
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + 1e-6

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])

        policy_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        # ==============================================================
        # sampled related core code: calculate policy loss, typically cross_entropy_loss
        # ==============================================================
        if self._cfg.model.continuous_action_space:
            """continuous action space"""
            policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(
                policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch, unroll_step=0)
        else:
            """discrete action space"""
            policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions = self._calculate_policy_loss_disc(
                policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch,
                unroll_step=0)

        value_prefix_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        target_value_prefix_cpu = target_value_prefix.detach().cpu()
        gradient_scale = 1 / self._cfg.num_unroll_steps

        # ==============================================================
        # the core recurrent_inference in SampledEfficientZero policy.
        # ==============================================================
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``hidden_state``, ``reward_hidden``,
            # `` value_prefix`` given current ``hidden_state`` ``reward_hidden`` and ``action``.
            # And then predict policy_logits and value  with the prediction function.
            network_output = self._learn_model.recurrent_inference(
                hidden_state, reward_hidden_state, action_batch[:, step_i]
            )
            hidden_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(
                network_output)

            # transform the scaled value or its categorical representation to its original value,
            # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
            original_value = self.inverse_scalar_transform_handle(value)
            original_value_prefix = self.inverse_scalar_transform_handle(value_prefix)

            beg_index = self._cfg.model.image_channel * step_i
            end_index = self._cfg.model.image_channel * (step_i + self._cfg.model.frame_stack_num)

            if self._cfg.model.self_supervised_learning_loss:
                # ==============================================================
                # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
                # ==============================================================
                if self._cfg.ssl_loss_weight > 0:
                    # obtain the oracle hidden states from representation function.
                    network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                    presentation_state = network_output.hidden_state

                    hidden_state = to_tensor(hidden_state)
                    presentation_state = to_tensor(presentation_state)

                    # NOTE: no grad for the presentation_state branch.
                    dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                    observation_proj = self._learn_model.project(presentation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                    consistency_loss += temp_loss

            # NOTE: the target policy, target_value_categorical, target_value_prefix_categorical is calculated in
            # game buffer now.
            # ==============================================================
            # sampled related core code:
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the += in policy loss.
            # ==============================================================
            if self._cfg.model.continuous_action_space:
                """continuous action space"""
                policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(
                    policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch,
                    unroll_step=step_i + 1)
            else:
                """discrete action space"""
                policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions = self._calculate_policy_loss_disc(
                    policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch,
                    unroll_step=step_i + 1)

            value_loss += cross_entropy_loss(value, target_value_categorical[:, step_i + 1])
            value_prefix_loss += cross_entropy_loss(value_prefix,
                                                    target_value_prefix_categorical[:, step_i])

            # reset hidden states every ``lstm_horizon_len`` unroll steps.
            if (step_i + 1) % self._cfg.lstm_horizon_len == 0:
                reward_hidden_state = (
                    torch.zeros(1, self._cfg.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device),
                    torch.zeros(1, self._cfg.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device)
                )

            if self._cfg.monitor_extra_statistics:
                original_value_prefixs = self.inverse_scalar_transform_handle(value_prefix)
                original_value_prefixs_cpu = original_value_prefixs.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, self.inverse_scalar_transform_handle(value).detach().cpu())
                )
                predicted_value_prefixs.append(original_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                hidden_state_list = np.concatenate((hidden_state_list, hidden_state.detach().cpu().numpy()))

        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
                self._cfg.ssl_loss_weight * consistency_loss + self._cfg.policy_loss_weight * policy_loss +
                self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * value_prefix_loss +
                self._cfg.policy_entropy_loss_weight * policy_entropy_loss
        )
        weighted_total_loss = (weights * loss).mean()
        # TODO(pu): test the effect
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(),
                                                                     self._cfg.grad_clip_value)
        self._optimizer.step()
        if self._cfg.cos_lr_scheduler is True or self._cfg.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())

        loss_data = (
            weighted_total_loss.item(), loss.mean().item(), policy_loss.mean().item(),
            value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean()
        )
        if self._cfg.monitor_extra_statistics:
            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)

            td_data = (
                value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                transformed_target_value_prefix.detach().cpu().numpy(),
                transformed_target_value.detach().cpu().numpy(),
                target_value_prefix_categorical.detach().cpu().numpy(),
                target_value_categorical.detach().cpu().numpy(), predicted_value_prefixs.detach().cpu().numpy(),
                predicted_values.detach().cpu().numpy(), target_policy.detach().cpu().numpy(),
                predicted_policies.detach().cpu().numpy(), hidden_state_list
            )

        if self._cfg.model.continuous_action_space:
            return {
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'collect_mcts_temperature': self.collect_mcts_temperature,
                'weighted_total_loss': loss_data[0],
                'total_loss': loss_data[1],
                'policy_loss': loss_data[2],
                'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'value_prefix_loss': loss_data[3],
                'value_loss': loss_data[4],
                'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                # ==============================================================
                # priority related
                # ==============================================================
                'value_priority': td_data[0].flatten().mean().item(),
                'value_priority_orig': value_priority,

                'target_value_prefix': td_data[1].flatten().mean().item(),
                'target_value': td_data[2].flatten().mean().item(),
                'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                'transformed_target_value': td_data[4].flatten().mean().item(),
                'predicted_value_prefixs': td_data[7].flatten().mean().item(),
                'predicted_values': td_data[8].flatten().mean().item(),

                # ==============================================================
                # sampled related core code
                # ==============================================================
                'policy_mu_max': mu[:, 0].max().item(),
                'policy_mu_min': mu[:, 0].min().item(),
                'policy_mu_mean': mu[:, 0].mean().item(),
                'policy_sigma_max': sigma.max().item(),
                'policy_sigma_min': sigma.min().item(),
                'policy_sigma_mean': sigma.mean().item(),
                # take the fist dim in action space
                'target_sampled_actions_max': target_sampled_actions[:, :, 0].max().item(),
                'target_sampled_actions_min': target_sampled_actions[:, :, 0].min().item(),
                'target_sampled_actions_mean': target_sampled_actions[:, :, 0].mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip
            }
        else:
            return {
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'collect_mcts_temperature': self.collect_mcts_temperature,
                'weighted_total_loss': loss_data[0],
                'total_loss': loss_data[1],
                'policy_loss': loss_data[2],
                'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'value_prefix_loss': loss_data[3],
                'value_loss': loss_data[4],
                'consistency_loss': loss_data[5] / self._cfg.num_unroll_steps,

                # ==============================================================
                # priority related
                # ==============================================================
                'value_priority': td_data[0].flatten().mean().item(),
                'value_priority_orig': value_priority,

                'target_value_prefix': td_data[1].flatten().mean().item(),
                'target_value': td_data[2].flatten().mean().item(),
                'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                'transformed_target_value': td_data[4].flatten().mean().item(),
                'predicted_value_prefixs': td_data[7].flatten().mean().item(),
                'predicted_values': td_data[8].flatten().mean().item(),

                # ==============================================================
                # sampled related core code
                # ==============================================================
                # take the fist dim in action space
                'target_sampled_actions_max': target_sampled_actions[:, :].float().max().item(),
                'target_sampled_actions_min': target_sampled_actions[:, :].float().min().item(),
                'target_sampled_actions_mean': target_sampled_actions[:, :].float().mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip
            }

    def _calculate_policy_loss_cont(self, policy_loss, policy_logits, target_policy, mask_batch,
                                    child_sampled_actions_batch,
                                    unroll_step):
        (mu, sigma) = policy_logits[:, :self._cfg.model.
            action_space_size], policy_logits[:, -self._cfg.model.action_space_size:]

        dist = Independent(Normal(mu, sigma), 1)

        # take the init hypothetical step k=unroll_step
        target_normalized_visit_count = target_policy[:, unroll_step]

        # Note: The target_policy_entropy is just for debugging.
        target_normalized_visit_count_masked = torch.index_select(
            target_normalized_visit_count, 0,
            torch.nonzero(mask_batch[:, unroll_step]).squeeze(-1)
        )
        target_dist = Categorical(target_normalized_visit_count_masked)
        target_policy_entropy = target_dist.entropy().mean()

        # shape: (batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1) -> (batch_size,
        # num_of_sampled_actions, action_dim) e.g. (4, 6, 20, 2, 1) ->  (4, 20, 2)
        target_sampled_actions = child_sampled_actions_batch[:, unroll_step].squeeze(-1)

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -dist.entropy()

        # Project the sampled-based improved policy back onto the space of representable policies. calculate KL
        # loss (batch_size, num_of_sampled_actions) -> (4,20) target_normalized_visit_count is
        # categorical distribution, the range of target_log_prob_sampled_actions is (-inf, 0), add 1e-6 for
        # numerical stability.
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        log_prob_sampled_actions = []
        for k in range(self._cfg.model.num_of_sampled_actions):
            # target_sampled_actions[:,i,:].shape: batch_size, action_dim -> 4,2
            # dist.log_prob(target_sampled_actions[:,i,:]).shape: batch_size -> 4
            # dist is normal distribution, the range of log_prob_sampled_actions is (-inf, inf)

            # way 1:
            # log_prob = dist.log_prob(target_sampled_actions[:, k, :])

            # way 2: SAC-like
            y = 1 - target_sampled_actions[:, k, :].pow(2)

            # NOTE: for numerical stability.
            target_sampled_actions_clamped = torch.clamp(target_sampled_actions[:, k, :], torch.tensor(-1 + 1e-6),
                                                         torch.tensor(1 - 1e-6))
            target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)

            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(target_sampled_actions_before_tanh).unsqueeze(-1)
            log_prob = log_prob - torch.log(y + 1e-6).sum(-1, keepdim=True)
            log_prob = log_prob.squeeze(-1)

            log_prob_sampled_actions.append(log_prob)

        # shape: (batch_size, num_of_sampled_actions) e.g. (4,20)
        log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

        if self._cfg.normalize_prob_of_sampled_actions:
            # normalize the prob of sampled actions
            prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                log_prob_sampled_actions
            ).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach()
            # the above line is equal to the following line.
            # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1, eps=1e-6)
            log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

        # NOTE: the +=.
        if self._cfg.policy_loss_type == 'KL':
            # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
            policy_loss += (
                                   torch.exp(target_log_prob_sampled_actions.detach()) *
                                   (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                           ).sum(-1) * mask_batch[:, unroll_step]
        elif self._cfg.policy_loss_type == 'cross_entropy':
            # cross_entropy loss: - sum(p * log (q) )
            policy_loss += -torch.sum(
                torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
            ) * mask_batch[:, unroll_step]

        return policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def _calculate_policy_loss_disc(self, policy_loss, policy_logits, target_policy, mask_batch,
                                    child_sampled_actions_batch,
                                    unroll_step):

        prob = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(prob)

        # take the init hypothetical step k=unroll_step
        target_normalized_visit_count = target_policy[:, unroll_step]

        # Note: The target_policy_entropy is just for debugging.
        target_normalized_visit_count_masked = torch.index_select(
            target_normalized_visit_count, 0,
            torch.nonzero(mask_batch[:, unroll_step]).squeeze(-1)
        )
        target_dist = Categorical(target_normalized_visit_count_masked)
        target_policy_entropy = target_dist.entropy().mean()

        # shape: (batch_size, num_unroll_steps, num_of_sampled_actions, action_dim, 1) -> (batch_size,
        # num_of_sampled_actions, action_dim) e.g. (4, 6, 20, 2, 1) ->  (4, 20, 2)
        target_sampled_actions = child_sampled_actions_batch[:, unroll_step].squeeze(-1)

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -dist.entropy()

        # Project the sampled-based improved policy back onto the space of representable policies. calculate KL
        # loss (batch_size, num_of_sampled_actions) -> (4,20) target_normalized_visit_count is
        # categorical distribution, the range of target_log_prob_sampled_actions is (-inf, 0), add 1e-6 for
        # numerical stability.
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)

        log_prob_sampled_actions = []
        for k in range(self._cfg.model.num_of_sampled_actions):
            # target_sampled_actions[:,i,:] shape: (batch_size, action_dim) e.g. (4,2)
            # dist.log_prob(target_sampled_actions[:,i,:]) shape: batch_size e.g. 4
            # dist is normal distribution, the range of log_prob_sampled_actions is (-inf, inf)

            if len(target_sampled_actions.shape) == 2:
                target_sampled_actions = target_sampled_actions.unsqueeze(-1)

            log_prob = torch.log(prob.gather(-1, target_sampled_actions[:, k].long()).squeeze(-1) + 1e-6)
            log_prob_sampled_actions.append(log_prob)

        # (batch_size, num_of_sampled_actions) e.g. (4,20)
        log_prob_sampled_actions = torch.stack(log_prob_sampled_actions, dim=-1)

        if self._cfg.normalize_prob_of_sampled_actions:
            # normalize the prob of sampled actions
            prob_sampled_actions_norm = torch.exp(log_prob_sampled_actions) / torch.exp(
                log_prob_sampled_actions
            ).sum(-1).unsqueeze(-1).repeat(1, log_prob_sampled_actions.shape[-1]).detach()
            # the above line is equal to the following line.
            # prob_sampled_actions_norm = F.normalize(torch.exp(log_prob_sampled_actions), p=1., dim=-1, eps=1e-6)
            log_prob_sampled_actions = torch.log(prob_sampled_actions_norm + 1e-6)

        # NOTE: the +=.
        if self._cfg.policy_loss_type == 'KL':
            # KL divergence loss: sum( p* log(p/q) ) = sum( p*log(p) - p*log(q) )= sum( p*log(p)) - sum( p*log(q) )
            policy_loss += (
                                   torch.exp(target_log_prob_sampled_actions.detach()) *
                                   (target_log_prob_sampled_actions.detach() - log_prob_sampled_actions)
                           ).sum(-1) * mask_batch[:, unroll_step]
        elif self._cfg.policy_loss_type == 'cross_entropy':
            # cross_entropy loss: - sum(p * log (q) )
            policy_loss += -torch.sum(
                torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
            ) * mask_batch[:, unroll_step]

        return policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions

    def _init_collect(self) -> None:
        self._collect_model = self._model

        if self._cfg.mcts_ctree:
            from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSTree
            self._mcts_collect = MCTSTree(self._cfg)
        else:
            from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSTree
            self._mcts_collect = MCTSTree(self._cfg)

        self.collect_mcts_temperature = 1

    def _forward_collect(
            self, data: torch.Tensor, action_mask: list = None, temperature: np.ndarray = 1, to_play=-1,
            ready_env_id=None
    ):
        """
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
                shape: (N, *obs_shape), i.e. (N, C*S, H, W), where N is the number of collect_env.
            - action_mask: shape: ``{list: N} -> (action_space_size, ) or None``.
            - temperature: shape: (N, ), where N is the number of collect_env.
            - to_play: shape: ``{list: N} -> (2, ) or None``, where N is the number of collect_env.
            - ready_env_id: None.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        self._collect_model.eval()
        self.collect_mcts_temperature = temperature
        active_collect_env_num = data.shape[0]
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            hidden_state_roots, value_prefix_roots, reward_hidden_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output)

            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy().tolist()

            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)]
                    for _ in range(active_collect_env_num)
                ]
            else:
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                ]

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(
                    active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(
                    active_collect_env_num,
                    legal_actions,
                    self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions,
                    self._cfg.model.continuous_action_space
                )

            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet(
                    [self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]

            roots.prepare(
                self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play
            )
            self._mcts_collect.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions()  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}
            # TODO
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception as error:
                    # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])
                # TODO(pu):
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self.collect_mcts_temperature, deterministic=False
                )
                if action_mask[0] is not None and not self._cfg.model.continuous_action_space:
                    # only discrete action space have action mask
                    try:
                        action = roots_sampled_actions[i][action].value
                        # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])
                else:
                    try:
                        action = roots_sampled_actions[i][action].value
                        # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values[i],
                    'policy_logits': policy_logits[i],
                }

        return output

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: -1, ready_env_id=None):
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
        active_eval_env_num = data.shape[0]
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._eval_model.initial_inference(data)
            hidden_state_roots, value_prefix_roots, reward_hidden_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy(
                )  # shape（B, 1）
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)]
                    for _ in range(active_eval_env_num)
                ]
            else:
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                ]

            # cpp mcts_tree
            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(
                    active_eval_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(
                    active_eval_env_num,
                    legal_actions,
                    self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions,
                    self._cfg.model.continuous_action_space
                )

            roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions()  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            # ==============================================================
            # sampled related core code
            # ==============================================================
            roots_sampled_actions = roots.get_sampled_actions()  # shape: ``{list: batch_size} ->{list: action_space_size}``

            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception as error:
                    # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                # Setting deterministic=True implies choosing the action with the highest value (argmax) rather than sampling during the evaluation phase.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # ==============================================================
                # sampled related core code
                # ==============================================================
                if action_mask[0] is not None and not self._cfg.model.continuous_action_space:
                    # only discrete action space have action mask
                    try:
                        action = roots_sampled_actions[i][action].value
                        # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])
                else:
                    try:
                        action = roots_sampled_actions[i][action].value
                        # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception as error:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'root_sampled_actions': root_sampled_actions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values[i],
                    'policy_logits': policy_logits[i],
                }

        return output

    def _monitor_vars_learn(self) -> List[str]:
        if self._cfg.model.continuous_action_space:
            return [
                'collect_mcts_temperature',
                'cur_lr',
                'total_loss',
                'weighted_total_loss',
                'policy_loss',
                'value_prefix_loss',
                'value_loss',
                'consistency_loss',
                'value_priority',
                'target_value_prefix',
                'target_value',
                'predicted_value_prefixs',
                'predicted_values',
                'transformed_target_value_prefix',
                'transformed_target_value',

                # ==============================================================
                # sampled related core code
                # ==============================================================
                'policy_entropy',
                'target_policy_entropy',
                'policy_mu_max',
                'policy_mu_min',
                'policy_mu_mean',
                'policy_sigma_max',
                'policy_sigma_min',
                'policy_sigma_mean',
                # take the fist dim in action space
                'target_sampled_actions_max',
                'target_sampled_actions_min',
                'target_sampled_actions_mean',

                'total_grad_norm_before_clip',
            ]
        else:
            return [
                'collect_mcts_temperature',
                'cur_lr',
                'total_loss',
                'weighted_total_loss',
                'loss_mean',
                'policy_loss',
                'value_prefix_loss',
                'value_loss',
                'consistency_loss',
                'value_priority',
                'target_value_prefix',
                'target_value',
                'predicted_value_prefixs',
                'predicted_values',
                'transformed_target_value_prefix',
                'transformed_target_value',

                # ==============================================================
                # sampled related core code
                # ==============================================================
                'policy_entropy',
                'target_policy_entropy',

                # take the fist dim in action space
                'target_sampled_actions_max',
                'target_sampled_actions_min',
                'target_sampled_actions_mean',

                'total_grad_norm_before_clip',
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
            self, obs: torch.Tensor, policy_output: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        # be compatible with DI-engine base_policy
        pass

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # be compatible with DI-engine base_policy
        pass

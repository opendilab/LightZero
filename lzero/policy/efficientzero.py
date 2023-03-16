import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.distributions import Categorical
from torch.nn import L1Loss

# python mcts_tree
import lzero.mcts.ptree.ptree_ez as ptree
from lzero.mcts import EfficientZeroMCTSCtree as MCTS_ctree
from lzero.mcts import EfficientZeroMCTSPtree as MCTS_ptree
from lzero.mcts import ImageTransforms, modified_cross_entropy_loss, phi_transform, DiscreteSupport
from lzero.mcts import scalar_transform, InverseScalarTransform
from lzero.mcts import select_action
# cpp mcts_tree
from lzero.mcts.ctree.ctree_efficientzero import ez_tree as ctree
from lzero.mcts.utils import to_torch_float_tensor, ez_network_output_unpack
from .utils import negative_cosine_similarity


@POLICY_REGISTRY.register('efficientzero')
class EfficientZeroPolicy(Policy):
    """
    Overview:
        The policy class for EfficientZero.
    """

    # The default_config for EfficientZero policy.
    config = dict(
        # ``sampled_algo=True`` means the policy is sampled-based algorithm, which is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy)
        on_policy=False,
        model=dict(
            # the stacked obs shape -> the transformed obs shape:
            # [S, W, H, C] -> [S x C, W, H]
            # e.g. [4, 96, 96, 3] -> [4*3, 96, 96]
            # observation_shape=(12, 96, 96),  # if frame_stack_num=4, gray_scale=False
            # observation_shape=(3, 96, 96),  # if frame_stack_num=1, gray_scale=False
            observation_shape=(4, 96, 96),  # if frame_stack_num=4, gray_scale=True
            # whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            # the key difference setting between image-input and vector input.
            image_channel=1,
            lstm_hidden_size=512,
            frame_stack_num=4,
            support_scale=300,
        ),
        # learn_mode config
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            # For different env, we have different episode_length,
            # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor
            update_per_collect=100,
            # (int) How many samples in a training batch
            batch_size=256,
            # (bool) Whether to use piecewise constant decayed learning rate
            lr_piecewise_constant_decay=True,
            optim_type='SGD',
            learning_rate=0.2,  # init lr for manually decay schedule
            # lr_piecewise_constant_decay=False,
            # optim_type='Adam',
            # learning_rate=0.003,  # lr for Adam optimizer
            # (int) Frequency of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            weight_decay=1e-4,
            momentum=0.9,
            grad_clip_value=10,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_episode" episodes per collect.
            n_episode=8,
            unroll_len=1,
        ),
        # ==============================================================
        # begin of additional game_config
        # ==============================================================
        ## common
        mcts_ctree=True,
        device='cuda',
        collector_env_num=8,
        evaluator_env_num=3,
        env_type='not_board_games',
        battle_mode='play_with_bot_mode',
        game_wrapper=True,
        monitor_statistics=True,
        game_block_length=200,

        ## observation
        cvt_string=False,
        use_augmentation=True,
        # style of augmentation
        augmentation=['shift', 'intensity'],

        ## learn
        num_simulations=50,
        discount_factor=0.997,
        td_steps=5,
        num_unroll_steps=5,
        lstm_horizon_len=5,
        # the weight of different loss
        reward_loss_weight=1,
        value_loss_weight=0.25,
        policy_loss_weight=1,
        ssl_loss_weight=2,

        # ``threshold_training_steps_for_final_lr`` is only used for adjusting lr manually.
        # threshold_training_steps_for_final_lr=int(
        #     threshold_env_steps_for_final_lr / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # lr_piecewise_constant_decay: lr: 0.2 -> 0.02 -> 0.002
        threshold_training_steps_for_final_lr=int(2e5),

        # ``threshold_training_steps_for_final_temperature`` is only used for adjusting temperature manually.
        # threshold_training_steps_for_final_temperature=int(
        #     threshold_env_steps_for_final_temperature / collector_env_num / average_episode_length_when_converge * update_per_collect),
        # manual_temperature_decay: temperature: 1 -> 0.5 -> 0.25
        threshold_training_steps_for_final_temperature=int(2e5),

        # (bool) Whether to use manually decayed temperature
        manual_temperature_decay=False,
        # ``fixed_temperature_value`` is effective only when manual_temperature_decay=False
        fixed_temperature_value=0.25,

        ## reanalyze
        reanalyze_ratio=0.3,
        reanalyze_outdated=True,
        # whether to use root value in reanalyzing part
        use_root_value=False,
        mini_infer_size=256,

        ## priority
        use_priority=True,
        use_max_priority_for_new_data=True,
        # how much prioritization is used: 0 means no prioritization while 1 means full prioritization
        priority_prob_alpha=0.6,
        # how much correction is used: 0 means no correction while 1 means full correction
        priority_prob_beta=0.4,
        prioritized_replay_eps=1e-6,

        # UCB related config
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        # ==============================================================
        # end of additional game_config
        # ==============================================================
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For EfficientZero, ``lzero.model.efficientzero_model.EfficientZeroModel``
        """
        return 'EfficientZeroModel', ['lzero.model.efficientzero_model']

    def _init_learn(self) -> None:
        if 'optim_type' not in self._cfg.learn.keys() or self._cfg.learn.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                momentum=self._cfg.learn.momentum,
                weight_decay=self._cfg.learn.weight_decay,
            )

        elif self._cfg.learn.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                weight_decay=self._cfg.learn.weight_decay,
            )

        if self._cfg.learn.lr_piecewise_constant_decay:
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
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
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
        obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
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
        # ``obs_target_batch`` is only used for calculate consistency loss, which take the all obs other than
        # timestep t1, and is only performed in the last 8 timesteps in the second dim in ``obs_batch_ori``.
        obs_target_batch = obs_batch_ori[:, self._cfg.model.image_channel:, :, :]

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .long(), in discrete action space.
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        data_list = [
            mask_batch,
            target_value_prefix.astype('float64'),
            target_value.astype('float64'), target_policy, weights
        ]
        [mask_batch, target_value_prefix, target_value, target_policy,
         weights] = to_torch_float_tensor(data_list, self._cfg.device)

        target_value_prefix = target_value_prefix.view(self._cfg.learn.batch_size, -1)
        target_value = target_value.view(self._cfg.learn.batch_size, -1)
        assert obs_batch.size(0) == self._cfg.learn.batch_size == target_value_prefix.size(0)

        # ``scalar_transform`` to transform the original value to the scaled value,
        # i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        transformed_target_value_prefix = scalar_transform(target_value_prefix)
        transformed_target_value = scalar_transform(target_value)
        if self._cfg.model.categorical_distribution:
            # transform a scalar to its categorical_distribution. After this transformation, each scalar is
            # represented as the linear combination of its two adjacent supports.
            target_value_prefix_categorical = phi_transform(self.reward_support, transformed_target_value_prefix)
            target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        # ==============================================================
        # the core initial_inference in EfficientZero policy.
        # ==============================================================
        network_output = self._learn_model.initial_inference(obs_batch)
        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        hidden_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(network_output)

        # transform the scaled value or its categorical representation to its original value,
        # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        original_value = self.inverse_scalar_transform_handle(value)

        # Note: The following lines are just for debugging.
        predicted_value_prefixs = []
        if self._cfg.monitor_statistics:
            hidden_state_list = hidden_state.detach().cpu().numpy()
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition.
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + self._cfg.prioritized_replay_eps

        prob = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(prob)
        policy_entropy = dist.entropy().mean()

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        policy_loss = modified_cross_entropy_loss(policy_logits, target_policy[:, 0])

        # only for debug. take the init hypothetical step k=0.
        target_normalized_visit_count_init_step = target_policy[:, 0]
        try:
            # if there is zero in target_normalized_visit_count_init_step
            target_dist = Categorical(target_normalized_visit_count_init_step)
            target_policy_entropy = target_dist.entropy().mean()
        except Exception as error:
            target_policy_entropy = 0

        if self._cfg.model.categorical_distribution:
            value_loss = modified_cross_entropy_loss(value, target_value_categorical[:, 0])
        else:
            value_loss = torch.nn.MSELoss(reduction='none')(value.squeeze(-1), transformed_target_value[:, 0])

        value_prefix_loss = torch.zeros(self._cfg.learn.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.learn.batch_size, device=self._cfg.device)

        # ==============================================================
        # the core recurrent_inference in EfficientZero policy.
        # ==============================================================
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``hidden_state``, ``reward_hidden``,
            # `` value_prefix`` given current ``hidden_state`` ``reward_hidden`` and ``action``.
            # And then predict policy_logits and value  with the prediction function.
            network_output = self._learn_model.recurrent_inference(
                hidden_state, reward_hidden_state, action_batch[:, step_i]
            )
            hidden_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(network_output)

            # transform the scaled value or its categorical representation to its original value,
            # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
            original_value = self.inverse_scalar_transform_handle(value)
            original_value_prefix = self.inverse_scalar_transform_handle(value_prefix)

            beg_index = self._cfg.model.image_channel * step_i
            end_index = self._cfg.model.image_channel * (step_i + self._cfg.model.frame_stack_num)

            # ==============================================================
            # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
            # ==============================================================
            if self._cfg.ssl_loss_weight > 0:
                # obtain the oracle hidden states from representation function.
                network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index, :, :])
                hidden_state = to_tensor(hidden_state)
                presentation_state = to_tensor(network_output.hidden_state)

                # NOTE: no grad for the presentation_state branch.
                dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                observation_proj = self._learn_model.project(presentation_state, with_grad=False)
                temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                consistency_loss += temp_loss

            # NOTE: the target policy, target_value_categorical, target_value_prefix_categorical is calculated in
            # game buffer now.
            # ==============================================================
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += modified_cross_entropy_loss(policy_logits, target_policy[:, step_i + 1])

            # only for debug. take th hypothetical step k = step_i + 1
            prob = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(prob)
            policy_entropy += dist.entropy().mean()
            target_normalized_visit_count = target_policy[:, step_i + 1]
            try:
                target_dist = Categorical(target_normalized_visit_count)
                target_policy_entropy += target_dist.entropy().mean()
            except Exception as error:
                # if there is zero in target_normalized_visit_count
                target_policy_entropy += 0

            if self._cfg.model.categorical_distribution:
                value_loss += modified_cross_entropy_loss(value, target_value_categorical[:, step_i + 1])
                value_prefix_loss += modified_cross_entropy_loss(
                    value_prefix, target_value_prefix_categorical[:, step_i]
                )
            else:
                value_loss += torch.nn.MSELoss(reduction='none'
                                               )(value.squeeze(-1), transformed_target_value[:, step_i + 1])
                value_prefix_loss += torch.nn.MSELoss(
                    reduction='none'
                )(value_prefix.squeeze(-1), transformed_target_value_prefix[:, step_i])

            # reset hidden states every ``lstm_horizon_len`` unroll steps.
            if (step_i + 1) % self._cfg.lstm_horizon_len == 0:
                reward_hidden_state = (
                    torch.zeros(1, self._cfg.learn.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device),
                    torch.zeros(1, self._cfg.learn.batch_size, self._cfg.model.lstm_hidden_size).to(self._cfg.device)
                )

            if self._cfg.monitor_statistics:
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
            self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * value_prefix_loss
        )
        weighted_total_loss = (weights * loss).mean()
        # TODO(pu): test the effect
        gradient_scale = 1 / self._cfg.num_unroll_steps
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(), self._cfg.learn.grad_clip_value
        )
        self._optimizer.step()
        if self._cfg.learn.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())
        
        # packing loss info for tensorboard logging
        loss_info = (
            weighted_total_loss.item(), loss.mean().item(), policy_loss.mean().item(), value_prefix_loss.mean().item(),
            value_loss.mean().item(), consistency_loss.mean()
        )

        if self._cfg.monitor_statistics:
            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)

            if self._cfg.model.categorical_distribution:
                td_data = (
                    value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                    transformed_target_value_prefix.detach().cpu().numpy(),
                    transformed_target_value.detach().cpu().numpy(),
                    target_value_prefix_categorical.detach().cpu().numpy(),
                    target_value_categorical.detach().cpu().numpy(), predicted_value_prefixs.detach().cpu().numpy(),
                    predicted_values.detach().cpu().numpy(), target_policy.detach().cpu().numpy(),
                    predicted_policies.detach().cpu().numpy(), hidden_state_list
                )
            else:
                td_data = (
                    value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                    transformed_target_value_prefix.detach().cpu().numpy(),
                    transformed_target_value.detach().cpu().numpy(), predicted_value_prefixs.detach().cpu().numpy(),
                    predicted_values.detach().cpu().numpy(), target_policy.detach().cpu().numpy(),
                    predicted_policies.detach().cpu().numpy(), hidden_state_list
                )

        if self._cfg.model.categorical_distribution:
            return {
                'collect_mcts_temperature': self.collect_mcts_temperature,
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'weighted_total_loss': loss_info[0],
                'total_loss': loss_info[1],
                'policy_loss': loss_info[2],
                'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'value_prefix_loss': loss_info[3],
                'value_loss': loss_info[4],
                'consistency_loss': loss_info[5] / self._cfg.num_unroll_steps,

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
                'total_grad_norm_before_clip': total_grad_norm_before_clip
            }
        else:
            return {
                'collect_mcts_temperature': self.collect_mcts_temperature,
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'weighted_total_loss': loss_info[0],
                'total_loss': loss_info[1],
                'policy_loss': loss_info[2],
                'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'value_prefix_loss': loss_info[3],
                'value_loss': loss_info[4],
                'consistency_loss': loss_info[5] / self._cfg.num_unroll_steps,
                'value_priority': td_data[0].flatten().mean().item(),
                # ==============================================================
                # priority related
                'value_priority_orig': value_priority,
                # ==============================================================
                'target_value_prefix': td_data[1].flatten().mean().item(),
                'target_value': td_data[2].flatten().mean().item(),
                'transformed_target_value_prefix': td_data[3].flatten().mean().item(),
                'transformed_target_value': td_data[4].flatten().mean().item(),
                'predicted_value_prefixs': td_data[5].flatten().mean().item(),
                'predicted_values': td_data[6].flatten().mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip
            }

    def _init_collect(self) -> None:
        self._collect_model = self._model
        self._unroll_len = self._cfg.collect.unroll_len
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTS_ctree(self._cfg)
        else:
            self._mcts_collect = MCTS_ptree(self._cfg)
        self.collect_mcts_temperature = 1

    def _forward_collect(
        self, data: torch.Tensor, action_mask: list = None, temperature: list = None, to_play=-1, ready_env_id=None
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
        self.collect_mcts_temperature = temperature[0]
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

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                if to_play[0] in [None, -1]:
                    # we use to_play=0 means play_with_bot_mode game in mcts_ctree
                    to_play = [0 for _ in range(active_collect_env_num)]
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                ]
                roots = ctree.Roots(active_collect_env_num, legal_actions)
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_roots, policy_logits, to_play
                )
                self._mcts_collect.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots, to_play)
            else:
                # python mcts_tree
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                ]
                roots = ptree.Roots(active_collect_env_num, self._cfg.num_simulations, legal_actions)
                # the only difference between collect and eval is the dirichlet noise.
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                roots.prepare(
                    self._cfg.root_exploration_fraction, noises, value_prefix_roots, policy_logits, to_play
                )
                self._mcts_collect.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions()  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}
            # TODO
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # TODO(pu):
                # only legal actions have visit counts.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=temperature[i], deterministic=False
                )
                # NOTE: transform to the real action index in legal action set.
                action = np.where(action_mask[i] == 1.0)[0][action]
                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
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
            self._mcts_eval = MCTS_ctree(self._cfg)
        else:
            self._mcts_eval = MCTS_ptree(self._cfg)

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: None, ready_env_id=None):
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
                shape: (N, *obs_shape), i.e. (N, C*S, H, W), where N is the number of collect_env.
            - action_mask: shape: ``{list: N} -> (action_space_size, ) or None``.
            - temperature: shape: (N, ), where N is the number of collect_env.
            - to_play: shape: ``{list: N} -> (2, ) or None``, where N is the number of collect_env.
            - ready_env_id: None.
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

            # TODO(pu)
            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy(
                )  # shape（B, 1）
                hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
                reward_hidden_roots = (
                    reward_hidden_roots[0].detach().cpu().numpy(), reward_hidden_roots[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                if to_play[0] in [None, -1]:
                    # we use to_play=0 means play_with_bot_mode game in mcts_ctree
                    to_play = [-1 for i in range(active_eval_env_num)]
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                ]
                roots = ctree.Roots(active_eval_env_num, legal_actions)
                roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)

            else:
                # python mcts_tree
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)
                ]
                roots = ptree.Roots(active_eval_env_num, self._cfg.num_simulations, legal_actions)

                roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
                self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions()  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # ``deterministic=True`` means selecting the argmax action, not sampling in eval phase.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: transform to the real action index in legal action set.
                action = np.where(action_mask[i] == 1.0)[0][action]
                output[env_id] = {
                    'action': action,
                    'distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values[i],
                    'policy_logits': policy_logits[i],
                }

        return output

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'collect_mcts_temperature',
            'cur_lr',
            'weighted_total_loss',
            'total_loss',
            'policy_loss',
            'policy_entropy',
            'target_policy_entropy',
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


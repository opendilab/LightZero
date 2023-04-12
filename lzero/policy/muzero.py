import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss

from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, prepare_obs


@POLICY_REGISTRY.register('muzero')
class MuZeroPolicy(Policy):
    """
    Overview:
        The policy class for MuZero.
    """

    # The default_config for MuZero policy.
    config = dict(
        # (bool) ``sampled_algo=True`` means the policy is sampled-based algorithm (e.g. Sampled EfficientZero), which is used in ``collector``.
        sampled_algo=False,
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For 3-dimensional image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) the stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            # (bool) Whether to use discrete support to represent categorical distribution for value, reward/value_prefix.
            categorical_distribution=True,
            # (int) the image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution. Only effective when ``categorical_distribution=True``.
            support_scale=300,
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
        optim_type='SGD',
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,  # init lr for manually decay schedule
        # optim_type='Adam',
        # lr_piecewise_constant_decay=False,
        # learning_rate=0.003,  # lr for Adam optimizer
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
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
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
            by import_names path. For MuZero, ``lzero.model.muzero_model.MuZeroModel``
        """
        if self._cfg.model.model_type == "conv":
            return 'MuZeroModel', ['lzero.model.muzero_model']
        elif self._cfg.model.model_type == "mlp":
            return 'MuZeroModelMLP', ['lzero.model.muzero_model_mlp']

    def _init_learn(self) -> None:
        assert self._cfg.optim_type in ['SGD', 'Adam']
        # NOTE: in board_gmaes, for fixed lr 0.003, 'Adam' is better than 'SGD'.
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
        obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
        target_reward, target_value, target_policy = targets_batch

        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if self._cfg.model.self_supervised_learning_loss:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .long(), in discrete action space.
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        data_list = [
            mask_batch,
            target_reward.astype('float64'),
            target_value.astype('float64'), target_policy, weights
        ]
        [mask_batch, target_reward, target_value, target_policy,
         weights] = to_torch_float_tensor(data_list, self._cfg.device)

        target_reward = target_reward.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        assert obs_batch.size(0) == self._cfg.batch_size == target_reward.size(0)

        # ``scalar_transform`` to transform the original value to the scaled value,
        # i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # transform a scalar to its categorical_distribution. After this transformation, each scalar is
        # represented as the linear combination of its two adjacent supports.
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)

        # ==============================================================
        # the core initial_inference in MuZero policy.
        # ==============================================================
        network_output = self._learn_model.initial_inference(obs_batch)

        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        hidden_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

        # transform the scaled value or its categorical representation to its original value,
        # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        original_value = self.inverse_scalar_transform_handle(value)

        # Note: The following lines are just for debugging.
        predicted_rewards = []
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
        policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])
        value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])

        reward_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        target_reward_cpu = target_reward.detach().cpu()
        gradient_scale = 1 / self._cfg.num_unroll_steps

        # ==============================================================
        # the core recurrent_inference in MuZero policy.
        # ==============================================================
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``hidden_state``, ``reward``,
            # given current ``hidden_state`` and ``action``.
            # And then predict policy_logits and value with the prediction function.
            network_output = self._learn_model.recurrent_inference(hidden_state, action_batch[:, step_i])
            hidden_state, reward, value, policy_logits = mz_network_output_unpack(network_output)

            # transform the scaled value or its categorical representation to its original value,
            # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
            original_value = self.inverse_scalar_transform_handle(value)
            original_reward = self.inverse_scalar_transform_handle(reward)

            if self._cfg.model.self_supervised_learning_loss:
                # ==============================================================
                # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
                # ==============================================================
                if self._cfg.ssl_loss_weight > 0:
                    # obtain the oracle hidden states from representation function.
                    if self._cfg.model.model_type == 'conv':
                        beg_index = self._cfg.model.image_channel * step_i
                        end_index = self._cfg.model.image_channel * (step_i + self._cfg.model.frame_stack_num)
                        network_output = self._learn_model.initial_inference(
                            obs_target_batch[:, beg_index:end_index, :, :]
                        )
                    elif self._cfg.model.model_type == 'mlp':
                        beg_index = self._cfg.model.observation_shape * step_i
                        end_index = self._cfg.model.observation_shape * (step_i + self._cfg.model.frame_stack_num)
                        network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index])

                    hidden_state = to_tensor(hidden_state)
                    representation_state = to_tensor(network_output.latent_state)

                    # NOTE: no grad for the representation_state branch
                    dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_i]
                    consistency_loss += temp_loss

            # NOTE: the target policy, target_value_categorical, target_reward_categorical is calculated in
            # game buffer now.
            # ==============================================================
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += cross_entropy_loss(policy_logits, target_policy[:, step_i + 1])

            value_loss += cross_entropy_loss(value, target_value_categorical[:, step_i + 1])
            reward_loss += cross_entropy_loss(reward, target_reward_categorical[:, step_i])

            # Follow MuZero, set half gradient
            # hidden_state.register_hook(lambda grad: grad * 0.5)

            if self._cfg.monitor_extra_statistics:
                original_rewards = self.inverse_scalar_transform_handle(reward)
                original_rewards_cpu = original_rewards.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, self.inverse_scalar_transform_handle(value).detach().cpu())
                )
                predicted_rewards.append(original_rewards_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                hidden_state_list = np.concatenate((hidden_state_list, hidden_state.detach().cpu().numpy()))

        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
            self._cfg.ssl_loss_weight * consistency_loss + self._cfg.policy_loss_weight * policy_loss +
            self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * reward_loss
        )
        weighted_total_loss = (weights * loss).mean()

        gradient_scale = 1 / self._cfg.num_unroll_steps
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(), self._cfg.grad_clip_value
        )
        self._optimizer.step()
        if self._cfg.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())

        # packing loss info for tensorboard logging
        loss_info = (
            weighted_total_loss.item(), loss.mean().item(), policy_loss.mean().item(), reward_loss.mean().item(),
            value_loss.mean().item(), consistency_loss.mean()
        )
        if self._cfg.monitor_extra_statistics:
            predicted_rewards = torch.stack(predicted_rewards).transpose(1, 0).squeeze(-1)
            predicted_rewards = predicted_rewards.reshape(-1).unsqueeze(-1)

            td_data = (
                value_priority,
                target_reward.detach().cpu().numpy(),
                target_value.detach().cpu().numpy(),
                transformed_target_reward.detach().cpu().numpy(),
                transformed_target_value.detach().cpu().numpy(),
                target_reward_categorical.detach().cpu().numpy(),
                target_value_categorical.detach().cpu().numpy(),
                predicted_rewards.detach().cpu().numpy(),
                predicted_values.detach().cpu().numpy(),
                target_policy.detach().cpu().numpy(),
                predicted_policies.detach().cpu().numpy(),
                hidden_state_list,
            )

        return {
            'collect_mcts_temperature': self.collect_mcts_temperature,
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': loss_info[0],
            'total_loss': loss_info[1],
            'policy_loss': loss_info[2],
            'reward_loss': loss_info[3],
            'value_loss': loss_info[4],
            'consistency_loss': loss_info[5] / self._cfg.num_unroll_steps,

            # ==============================================================
            # priority related
            # ==============================================================
            'value_priority_orig': value_priority,
            'value_priority': td_data[0].flatten().mean().item(),
            'target_reward': td_data[1].flatten().mean().item(),
            'target_value': td_data[2].flatten().mean().item(),
            'transformed_target_reward': td_data[3].flatten().mean().item(),
            'transformed_target_value': td_data[4].flatten().mean().item(),
            'predicted_rewards': td_data[7].flatten().mean().item(),
            'predicted_values': td_data[8].flatten().mean().item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip
        }

    def _init_collect(self) -> None:
        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)
        self.collect_mcts_temperature = 1

    def _forward_collect(
        self, data: torch.Tensor, action_mask: list = None, temperature: np.ndarray = 1, to_play=-1, ready_env_id=None
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
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}

            # TODO
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self.collect_mcts_temperature, deterministic=False
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
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
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

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
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'predicted_rewards',
            'predicted_values',
            'transformed_target_reward',
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

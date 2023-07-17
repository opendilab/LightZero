from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from .sampled_efficientzero import SampledEfficientZeroPolicy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from ditk import logging
from torch.distributions import Categorical, Independent, Normal
from torch.nn import L1Loss

from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, ez_network_output_unpack, select_action, negative_cosine_similarity, prepare_obs, \
    configure_optimizers
from collections import defaultdict
from ding.torch_utils import to_device


@POLICY_REGISTRY.register('gobigger_sampled_efficientzero')
class GoBiggerSampledEfficientZeroPolicy(SampledEfficientZeroPolicy):
    """
    Overview:
        The policy class for GoBigger Sampled EfficientZero.
    """

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and model import_names.
                - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
                - import_names (:obj:`List[str]`): The model class path list used in this algorithm.

        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For Sampled EfficientZero, ``lzero.model.sampled_efficientzero_model.SampledEfficientZeroModel``
        """
        return 'GoBiggerSampledEfficientZeroModel', ['lzero.model.gobigger.gobigger_sampled_efficientzero_model']

    def _forward_learn(self, data: torch.Tensor) -> Dict[str, Union[float, int]]:
        """
         Overview:
             The forward function for learning policy in learn mode, which is the core of the learning process.
             The data is sampled from replay buffer.
             The loss is calculated by the loss function and the loss is backpropagated to update the model.
         Arguments:
             - data (:obj:`Tuple[torch.Tensor]`): The data sampled from replay buffer, which is a tuple of tensors.
                 The first tensor is the current_batch, the second tensor is the target_batch.
         Returns:
             - info_dict (:obj:`Dict[str, Union[float, int]]`): The information dict to be logged, which contains \
                 current learning loss and learning statistics.
         """
        self._learn_model.train()
        self._target_model.train()

        current_batch, target_batch = data
        # ==============================================================
        # sampled related core code
        # ==============================================================
        obs_batch_ori, action_batch, child_sampled_actions_batch, mask_batch, indices, weights, make_time = current_batch
        target_value_prefix, target_value, target_policy = target_batch

        obs_batch_ori = obs_batch_ori.tolist()
        obs_batch_ori = np.array(obs_batch_ori)
        obs_batch = obs_batch_ori[:, 0:self._cfg.model.frame_stack_num]
        if self._cfg.model.self_supervised_learning_loss:
            obs_target_batch = obs_batch_ori[:, self._cfg.model.frame_stack_num:]

        # do augmentations
        # if self._cfg.use_augmentation:
        #     obs_batch = self.image_transforms.transform(obs_batch)
        #     if self._cfg.model.self_supervised_learning_loss:
        #         obs_target_batch = self.image_transforms.transform(obs_target_batch)

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

        assert obs_batch.size == self._cfg.batch_size == target_value_prefix.size(0)

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
        obs_batch = obs_batch.tolist()
        obs_batch = sum(obs_batch, [])
        obs_batch = to_tensor(obs_batch)
        obs_batch = to_device(obs_batch, self._cfg.device)
        network_output = self._learn_model.initial_inference(obs_batch)
        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        latent_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(network_output)

        # transform the scaled value or its categorical representation to its original value,
        # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        original_value = self.inverse_scalar_transform_handle(value)

        # Note: The following lines are just for logging.
        predicted_value_prefixs = []
        if self._cfg.monitor_extra_statistics:
            latent_state_list = latent_state.detach().cpu().numpy()
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
                policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch, unroll_step=0
            )
        else:
            """discrete action space"""
            policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions = self._calculate_policy_loss_disc(
                policy_loss, policy_logits, target_policy, mask_batch, child_sampled_actions_batch, unroll_step=0
            )

        value_prefix_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        gradient_scale = 1 / self._cfg.num_unroll_steps

        # ==============================================================
        # the core recurrent_inference in SampledEfficientZero policy.
        # ==============================================================
        for step_i in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``latent_state``, ``reward_hidden_state``,
            # `` value_prefix`` given current ``latent_state`` ``reward_hidden_state`` and ``action``.
            # And then predict policy_logits and value  with the prediction function.
            network_output = self._learn_model.recurrent_inference(
                latent_state, reward_hidden_state, action_batch[:, step_i]
            )
            latent_state, value_prefix, reward_hidden_state, value, policy_logits = ez_network_output_unpack(
                network_output
            )

            # transform the scaled value or its categorical representation to its original value,
            # i.e. h^(-1)(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
            original_value = self.inverse_scalar_transform_handle(value)

            if self._cfg.model.self_supervised_learning_loss:
                # ==============================================================
                # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
                # ==============================================================
                if self._cfg.ssl_loss_weight > 0:
                    beg_index = step_i
                    end_index = step_i + self._cfg.model.frame_stack_num
                    obs_target_batch_tmp = obs_target_batch[:, beg_index:end_index].tolist()
                    obs_target_batch_tmp = sum(obs_target_batch_tmp, [])
                    obs_target_batch_tmp = to_tensor(obs_target_batch_tmp)
                    obs_target_batch_tmp = to_device(obs_target_batch_tmp, self._cfg.device)
                    network_output = self._learn_model.initial_inference(obs_target_batch_tmp)

                    latent_state = to_tensor(latent_state)
                    representation_state = to_tensor(network_output.latent_state)

                    # NOTE: no grad for the representation_state branch.
                    dynamic_proj = self._learn_model.project(latent_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
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
                    policy_loss,
                    policy_logits,
                    target_policy,
                    mask_batch,
                    child_sampled_actions_batch,
                    unroll_step=step_i + 1
                )
            else:
                """discrete action space"""
                policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions = self._calculate_policy_loss_disc(
                    policy_loss,
                    policy_logits,
                    target_policy,
                    mask_batch,
                    child_sampled_actions_batch,
                    unroll_step=step_i + 1
                )

            value_loss += cross_entropy_loss(value, target_value_categorical[:, step_i + 1])
            value_prefix_loss += cross_entropy_loss(value_prefix, target_value_prefix_categorical[:, step_i])

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
                latent_state_list = np.concatenate((latent_state_list, latent_state.detach().cpu().numpy()))

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
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(), self._cfg.grad_clip_value
        )
        self._optimizer.step()
        if self._cfg.cos_lr_scheduler or self._cfg.lr_piecewise_constant_decay:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())

        if self._cfg.monitor_extra_statistics:
            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)

        return_data = {
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'collect_epsilon': self.collect_epsilon,
                'collect_mcts_temperature': self.collect_mcts_temperature,
                'weighted_total_loss': weighted_total_loss.item(),
                'total_loss': loss.mean().item(),
                'policy_loss': policy_loss.mean().item(),
                'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
                'value_prefix_loss': value_prefix_loss.mean().item(),
                'value_loss': value_loss.mean().item(),
                'consistency_loss': consistency_loss.mean() / self._cfg.num_unroll_steps,

                # ==============================================================
                # priority related
                # ==============================================================
                'value_priority': value_priority.flatten().mean().item(),
                'value_priority_orig': value_priority,
                'target_value_prefix': target_value_prefix.detach().cpu().numpy().mean().item(),
                'target_value': target_value.detach().cpu().numpy().mean().item(),
                'transformed_target_value_prefix': transformed_target_value_prefix.detach().cpu().numpy().mean().item(),
                'transformed_target_value': transformed_target_value.detach().cpu().numpy().mean().item(),
                'predicted_value_prefixs': predicted_value_prefixs.detach().cpu().numpy().mean().item(),
                'predicted_values': predicted_values.detach().cpu().numpy().mean().item()
        }

        if self._cfg.model.continuous_action_space:
            return_data.update({
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
            })
        else:
            return_data.update({
                # ==============================================================
                # sampled related core code
                # ==============================================================
                # take the fist dim in action space
                'target_sampled_actions_max': target_sampled_actions[:, :].float().max().item(),
                'target_sampled_actions_min': target_sampled_actions[:, :].float().min().item(),
                'target_sampled_actions_mean': target_sampled_actions[:, :].float().mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip
            })
        
        return return_data

    def _forward_collect(
        self, data: torch.Tensor, action_mask: list = None, temperature: np.ndarray = 1, to_play=-1, epsilon: float = 0.25, ready_env_id=None
    ):
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Choosing the action through sampling during the collect mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()
        self.collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon

        active_collect_env_num = len(data)
        data = to_tensor(data)
        data = sum(sum(data, []), [])
        batch_size = len(data)
        data = to_device(data, self._cfg.device)
        agent_num = batch_size // active_collect_env_num
        to_play = np.array(to_play).reshape(-1).tolist()

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output
            )

            if not self._learn_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy().tolist()

            action_mask = sum(action_mask, [])
            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [[-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(batch_size)]
            else:
                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(
                    batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(
                    batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )

            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                                    ).astype(np.float32).tolist() for j in range(batch_size)
            ]

            roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
            self._mcts_collect.search(
                roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
            )

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: defaultdict(list) for i in data_id}
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i in range(batch_size):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception:
                    # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self.collect_mcts_temperature, deterministic=False
                )
                try:
                    action = roots_sampled_actions[i][action].value
                    # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                except Exception:
                    # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])

                output[i // agent_num]['action'].append(action)
                output[i // agent_num]['distributions'].append(distributions)
                output[i // agent_num]['root_sampled_actions'].append(root_sampled_actions)
                output[i // agent_num]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                output[i // agent_num]['value'].append(value)
                output[i // agent_num]['pred_value'].append(pred_values[i])
                output[i // agent_num]['policy_logits'].append(policy_logits[i])

        return output

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: -1, ready_env_id=None):
        """
         Overview:
             The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
             Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
         Arguments:
             - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
             - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
             - to_play (:obj:`int`): The player to play.
             - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
         Shape:
             - data (:obj:`torch.Tensor`):
                 - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                     S is the number of stacked frames, H is the height of the image, W is the width of the image.
                 - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
             - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
             - to_play: :math:`(N, 1)`, where N is the number of collect_env.
             - ready_env_id: None
         Returns:
             - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                 ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
         """
        self._eval_model.eval()
        active_eval_env_num = len(data)
        data = to_tensor(data)
        data = sum(sum(data, []), [])
        batch_size = len(data)
        data = to_device(data, self._cfg.device)
        agent_num = batch_size // active_eval_env_num
        to_play = np.array(to_play).reshape(-1).tolist()

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._eval_model.initial_inference(data)
            latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output
            )

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            action_mask = sum(action_mask, [])
            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [[-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(batch_size)]
            else:
                legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]

            # cpp mcts_tree
            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(
                    batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(
                    batch_size, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )

            roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, reward_hidden_state_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            # ==============================================================
            # sampled related core code
            # ==============================================================
            roots_sampled_actions = roots.get_sampled_actions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``

            data_id = [i for i in range(active_eval_env_num)]
            output = {i: defaultdict(list) for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i in range(batch_size):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                try:
                    root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])
                except Exception:
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

                try:
                    action = roots_sampled_actions[i][action].value
                    # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                except Exception:
                    # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                    action = np.array(roots_sampled_actions[i][action])

                if not self._cfg.model.continuous_action_space:
                    if len(action.shape) == 0:
                        action = int(action)
                    elif len(action.shape) == 1:
                        action = int(action[0])
                output[i // agent_num]['action'].append(action)
                output[i // agent_num]['distributions'].append(distributions)
                output[i // agent_num]['root_sampled_actions'].append(root_sampled_actions)
                output[i // agent_num]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                output[i // agent_num]['value'].append(value)
                output[i // agent_num]['pred_value'].append(pred_values[i])
                output[i // agent_num]['policy_logits'].append(policy_logits[i])

        return output

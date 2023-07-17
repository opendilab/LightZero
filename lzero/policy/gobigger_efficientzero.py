from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from .efficientzero import EfficientZeroPolicy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.distributions import Categorical
from torch.nn import L1Loss

from lzero.mcts import EfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import EfficientZeroMCTSPtree as MCTSPtree
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, select_action, to_torch_float_tensor, ez_network_output_unpack, negative_cosine_similarity, prepare_obs, \
    configure_optimizers
from collections import defaultdict
from ding.torch_utils import to_device


@POLICY_REGISTRY.register('gobigger_efficientzero')
class GoBiggerEfficientZeroPolicy(EfficientZeroPolicy):
    """
    Overview:
        The policy class for GoBiggerEfficientZero.
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
            by import_names path. For EfficientZero, ``lzero.model.efficientzero_model.EfficientZeroModel``
        """
        return 'GoBiggerEfficientZeroModel', ['lzero.model.gobigger.gobigger_efficientzero_model']

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
        obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
        target_value_prefix, target_value, target_policy = target_batch

        obs_batch_ori = obs_batch_ori.tolist()
        obs_batch_ori = np.array(obs_batch_ori)
        obs_batch = obs_batch_ori[:, 0:self._cfg.model.frame_stack_num]
        if self._cfg.model.self_supervised_learning_loss:
            obs_target_batch = obs_batch_ori[:, self._cfg.model.frame_stack_num:]
        # obs_batch, obs_target_batch = obs_batch_ori.tolist()

        # # do augmentations
        # if self._cfg.use_augmentation:
        #     obs_batch = self.image_transforms.transform(obs_batch)
        #     obs_target_batch = self.image_transforms.transform(obs_target_batch)

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
        # the core initial_inference in EfficientZero policy.
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

        # Note: The following lines are just for debugging.
        predicted_value_prefixs = []
        if self._cfg.monitor_extra_statistics:
            latent_state_list = latent_state.detach().cpu().numpy()
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition.
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + 1e-6

        prob = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(prob)
        policy_entropy = dist.entropy().mean()

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])

        # Here we take the init hypothetical step k=0.
        target_normalized_visit_count_init_step = target_policy[:, 0]

        # ******* NOTE: target_policy_entropy is only for debug.  ******
        non_masked_indices = torch.nonzero(mask_batch[:, 0]).squeeze(-1)
        # Check if there are any unmasked rows
        if len(non_masked_indices) > 0:
            target_normalized_visit_count_masked = torch.index_select(
                target_normalized_visit_count_init_step, 0, non_masked_indices
            )
            target_dist = Categorical(target_normalized_visit_count_masked)
            target_policy_entropy = target_dist.entropy().mean()
        else:
            # Set target_policy_entropy to 0 if all rows are masked
            target_policy_entropy = 0

        value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])

        value_prefix_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        # ==============================================================
        # the core recurrent_inference in EfficientZero policy.
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
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += cross_entropy_loss(policy_logits, target_policy[:, step_i + 1])

            # Here we take the hypothetical step k = step_i + 1
            prob = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(prob)
            policy_entropy += dist.entropy().mean()
            target_normalized_visit_count = target_policy[:, step_i + 1]

            # ******* NOTE: target_policy_entropy is only for debug.  ******
            non_masked_indices = torch.nonzero(mask_batch[:, step_i + 1]).squeeze(-1)
            # Check if there are any unmasked rows
            if len(non_masked_indices) > 0:
                target_normalized_visit_count_masked = torch.index_select(
                    target_normalized_visit_count, 0, non_masked_indices
                )
                target_dist = Categorical(target_normalized_visit_count_masked)
                target_policy_entropy += target_dist.entropy().mean()
            else:
                # Set target_policy_entropy to 0 if all rows are masked
                target_policy_entropy += 0

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
            self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * value_prefix_loss
        )
        weighted_total_loss = (weights * loss).mean()
        # TODO(pu): test the effect of gradient scale.
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
            weighted_total_loss.item(), loss.mean().item(), policy_loss.mean().item(), value_prefix_loss.mean().item(),
            value_loss.mean().item(), consistency_loss.mean()
        )

        if self._cfg.monitor_extra_statistics:
            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)

            td_data = (
                value_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                target_value_prefix_categorical.detach().cpu().numpy(), target_value_categorical.detach().cpu().numpy(),
                predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), latent_state_list
            )

        return {
            'collect_mcts_temperature': self.collect_mcts_temperature,
            'collect_epsilon': self.collect_epsilon,
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

    def _forward_collect(
        self,
        data: torch.Tensor,
        action_mask: list = None,
        temperature: float = 1,
        to_play: List = [-1],
        random_collect_episode_num: int = 0,
        epsilon: float = 0.25,
        ready_env_id=None
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
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
            # the only difference between collect and eval is the dirichlet noise.
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(batch_size)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(batch_size, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(batch_size, legal_actions)
            roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
            self._mcts_collect.search(
                roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
            )

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: defaultdict(list) for i in data_id}
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i in range(batch_size):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                if random_collect_episode_num>0:  # random collect
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self.collect_mcts_temperature, deterministic=False
                    )
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                else: 
                    if self._cfg.eps.eps_greedy_exploration_in_collect:  # eps greedy collect
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self.collect_mcts_temperature, deterministic=True
                        )
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else: # collect
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self.collect_mcts_temperature, deterministic=False
                        )
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                output[i // agent_num]['action'].append(action)
                output[i // agent_num]['distributions'].append(distributions)
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
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(batch_size, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(batch_size, legal_actions)
            roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, reward_hidden_state_roots, to_play)

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}
            data_id = [i for i in range(active_eval_env_num)]
            output = {i: defaultdict(list) for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i in range(batch_size):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                output[i // agent_num]['action'].append(action)
                output[i // agent_num]['distributions'].append(distributions)
                output[i // agent_num]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                output[i // agent_num]['value'].append(value)
                output[i // agent_num]['pred_value'].append(pred_values[i])
                output[i // agent_num]['policy_logits'].append(policy_logits[i])

        return output


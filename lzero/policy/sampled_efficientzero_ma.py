import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.torch_utils import to_tensor, to_device, to_dtype, to_ndarray
from ding.utils.data import default_collate, default_decollate
from ding.utils import POLICY_REGISTRY
from ditk import logging
from torch.distributions import Categorical, Independent, Normal
from torch.nn import L1Loss

from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, ez_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, \
    configure_optimizers
from lzero.policy.muzero import MuZeroPolicy
from lzero.policy.sampled_efficientzero import SampledEfficientZeroPolicy


@POLICY_REGISTRY.register('sampled_efficientzero_ma')
class SampledEfficientZeroMAPolicy(SampledEfficientZeroPolicy):
    """
    Overview:
        The policy class for Sampled EfficientZero proposed in the paper https://arxiv.org/abs/2104.06303.
    """

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for multi-agent.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and model import_names.
            - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
            - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        """
        # if self._cfg.model.model_type == "conv":
        #     return 'SampledEfficientZeroModel', ['lzero.model.sampled_efficientzero_model']
        if self._cfg.model.model_type == "mlp":
            return 'SampledEfficientZeroModelMLPMaIndependent', ['lzero.model.sampled_efficientzero_model_mlp_ma_independent']
        else:
            raise ValueError("model type {} is not supported".format(self._cfg.model.model_type))

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        super()._init_learn()
        self._multi_agent = self._cfg.model.multi_agent

    def _prepocess_data(self, data_list):
        def get_depth(lst):
            if not isinstance(lst, list):
                return 0
            return 1 + get_depth(lst[0])
        for i in range(len(data_list)):
            depth = get_depth(data_list[i])
            if depth != 0:
                for _ in range(depth):
                    data_list[i] = default_collate(data_list[i])
                data_list[i] = to_dtype(to_device(data_list[i], self._cfg.device), torch.float)
                data_list[i] = data_list[i].permute(*range(depth-1, -1, -1), *range(depth, data_list[i].dim()))
            else:
                data_list[i] = to_dtype(to_device(to_tensor(data_list[i]), self._cfg.device), torch.float)
        return data_list

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

        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if self._cfg.model.self_supervised_learning_loss:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .float(), in continuous action space.
        if self._cfg.model.multi_agent:
            data_list = [action_batch, mask_batch, target_value_prefix, target_value, target_policy, weights, child_sampled_actions_batch]
            [action_batch, mask_batch, target_value_prefix, target_value,
                target_policy, weights, child_sampled_actions_batch] = self._prepocess_data(data_list)
        else:
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device).float().unsqueeze(-1)
            data_list = [
                mask_batch,
                target_value_prefix.astype('float32'),
                target_value.astype('float32'), target_policy, weights
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
        if self._cfg.use_priority:
            value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
            value_priority = value_priority.data.cpu().numpy() + 1e-6
        else:
            value_priority = np.ones(self._cfg.model.agent_num*self._cfg.batch_size)

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        if self._multi_agent:
            # (B, unroll_step, agent_num, 601) -> (B*agent_num, unroll_step, 601)
            target_value_categorical = target_value_categorical.transpose(1, 2)
            target_value_categorical = target_value_categorical.reshape((-1, *target_value_categorical.shape[2:]))
            # (B, unroll_step, agent_num, action_dim) -> (B*agent_num, unroll_step, action_dim)
            action_batch = action_batch.transpose(1,2)
            action_batch = action_batch.reshape((-1, *action_batch.shape[2:]))

            target_value_prefix_categorical = torch.repeat_interleave(target_value_prefix_categorical, repeats=self._cfg.model.agent_num, dim=0)

            weights = torch.repeat_interleave(weights, repeats=self._cfg.model.agent_num)
            # value shape (B*agent_num, 601)
            value_loss = cross_entropy_loss(value, target_value_categorical[:, 0])
            policy_loss = torch.zeros(self._cfg.batch_size*self._cfg.model.agent_num, device=self._cfg.device)
        else:
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

        if self._multi_agent:
            value_prefix_loss = torch.zeros(self._cfg.batch_size*self._cfg.model.agent_num, device=self._cfg.device)
            consistency_loss = torch.zeros(self._cfg.batch_size*self._cfg.model.agent_num, device=self._cfg.device)
        else:
            value_prefix_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
            consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        # ==============================================================
        # the core recurrent_inference in SampledEfficientZero policy.
        # ==============================================================
        for step_k in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``latent_state``, ``reward_hidden_state``,
            # `` value_prefix`` given current ``latent_state`` ``reward_hidden_state`` and ``action``.
            # And then predict policy_logits and value  with the prediction function.
            network_output = self._learn_model.recurrent_inference(
                latent_state, reward_hidden_state, action_batch[:, step_k]
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
                    # obtain the oracle latent states from representation function.
                    beg_index, end_index = self._get_target_obs_index_in_step_k(step_k)
                    network_output = self._learn_model.initial_inference(obs_target_batch[:, beg_index:end_index])

                    latent_state = to_tensor(latent_state)
                    representation_state = to_tensor(network_output.latent_state)

                    # NOTE: no grad for the representation_state branch.
                    dynamic_proj = self._learn_model.project(latent_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_k]

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
                    unroll_step=step_k + 1
                )
            else:
                """discrete action space"""
                policy_loss, policy_entropy, policy_entropy_loss, target_policy_entropy, target_sampled_actions = self._calculate_policy_loss_disc(
                    policy_loss,
                    policy_logits,
                    target_policy,
                    mask_batch,
                    child_sampled_actions_batch,
                    unroll_step=step_k + 1
                )

            value_loss += cross_entropy_loss(value, target_value_categorical[:, step_k + 1])
            value_prefix_loss += cross_entropy_loss(value_prefix, target_value_prefix_categorical[:, step_k])

            # reset hidden states every ``lstm_horizon_len`` unroll steps.
            if (step_k + 1) % self._cfg.lstm_horizon_len == 0:
                if self._multi_agent:
                    reward_hidden_state = (
                    torch.zeros(1, self._cfg.batch_size*self._cfg.model.agent_num, self._cfg.model.lstm_hidden_size).to(self._cfg.device),
                    torch.zeros(1, self._cfg.batch_size*self._cfg.model.agent_num, self._cfg.model.lstm_hidden_size).to(self._cfg.device)
                )
                else:
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

        gradient_scale = 1 / self._cfg.num_unroll_steps
        weighted_total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
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
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': loss.mean().item(),
            'policy_loss': policy_loss.mean().item(),
            'policy_entropy': policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
            'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
            'value_prefix_loss': value_prefix_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'consistency_loss': consistency_loss.mean().item() / self._cfg.num_unroll_steps,

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
                # take the first dim in action space
                'target_sampled_actions_max': target_sampled_actions[:, :, 0].max().item(),
                'target_sampled_actions_min': target_sampled_actions[:, :, 0].min().item(),
                'target_sampled_actions_mean': target_sampled_actions[:, :, 0].mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip.item()
            })
        else:
            return_data.update({
                # ==============================================================
                # sampled related core code
                # ==============================================================
                # take the first dim in action space
                'target_sampled_actions_max': target_sampled_actions[:, :].float().max().item(),
                'target_sampled_actions_min': target_sampled_actions[:, :].float().min().item(),
                'target_sampled_actions_mean': target_sampled_actions[:, :].float().mean().item(),
                'total_grad_norm_before_clip': total_grad_norm_before_clip.item()
            })

        return return_data

    def _init_collect(self) -> None:
        """
          Overview:
              Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
          """
        super()._init_collect()
        self._multi_agent = self._cfg.model.multi_agent

    def _forward_collect(
            self, data: torch.Tensor, action_mask: list = None, temperature: np.ndarray = 1, to_play=-1,
            epsilon: float = 0.25, ready_env_id: np.array = None,
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
        self._collect_mcts_temperature = temperature
        if isinstance(data, dict):
            # If data is a dictionary, find the first non-dictionary element and get its shape[0]
            # TODO(rjy): written in recursive form
            for k, v in data.items():
                if not isinstance(v, dict):
                    active_collect_env_num = v.shape[0] * v.shape[1]
                    agent_num = v.shape[1]  # multi-agent
        elif isinstance(data, torch.Tensor):
            # If data is a torch.tensor, directly return its shape[0]
            active_collect_env_num = data.shape[0]
            agent_num = 1  # single-agent

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output
            )

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            reward_hidden_state_roots = (
                reward_hidden_state_roots[0].detach().cpu().numpy(),
                reward_hidden_state_roots[1].detach().cpu().numpy()
            )
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(active_collect_env_num)
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
                    active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )

            # the only difference between collect and eval is the dirichlet noise
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]

            roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
            self._mcts_collect.search(
                roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
            )

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}
            roots_sampled_actions = roots.get_sampled_actions()  # {list: 1}->{list:6}

            if self._multi_agent:
                active_collect_env_num = active_collect_env_num // agent_num
            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                output[env_id] = {
                    'action': [],
                    'visit_count_distributions': [],
                    'root_sampled_actions': [],
                    'visit_count_distribution_entropy': [],
                    'searched_value': [],
                    'predicted_value': [],
                    'predicted_policy_logits': [],
                }
                for j in range(agent_num):
                    index = i * agent_num + j
                    distributions, value = roots_visit_count_distributions[index], roots_values[index]
                    if self._cfg.mcts_ctree:
                        # In ctree, the method roots.get_sampled_actions() returns a list object.
                        root_sampled_actions = np.array([action for action in roots_sampled_actions[index]])
                    else:
                        # In ptree, the same method roots.get_sampled_actions() returns an Action object.
                        root_sampled_actions = np.array([action.value for action in roots_sampled_actions[index]])

                    # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                    # the index within the legal action set, rather than the index in the entire action set.
                    action, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self._collect_mcts_temperature, deterministic=False
                    )

                    if self._cfg.mcts_ctree:
                        # In ctree, the method roots.get_sampled_actions() returns a list object.
                        action = np.array(roots_sampled_actions[index][action])
                    else:
                        # In ptree, the same method roots.get_sampled_actions() returns an Action object.
                        action = roots_sampled_actions[index][action].value

                    if not self._cfg.model.continuous_action_space:
                        if len(action.shape) == 0:
                            action = int(action)
                        elif len(action.shape) == 1:
                            action = int(action[0])

                    output[env_id]['action'].append(action)
                    output[env_id]['visit_count_distributions'].append(distributions)
                    output[env_id]['root_sampled_actions'].append(root_sampled_actions)
                    output[env_id]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                    output[env_id]['searched_value'].append(value)
                    output[env_id]['predicted_value'].append(pred_values[index])
                    output[env_id]['predicted_policy_logits'].append(policy_logits[index])
                
                for k, v in output[env_id].items():
                    output[env_id][k] = np.array(v)

        return output

    def _init_eval(self) -> None:
        """
         Overview:
             Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
         """
        super()._init_eval()
        self._multi_agent = self._cfg.model.multi_agent

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: -1, ready_env_id: np.array = None,):
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
        if isinstance(data, dict):
            # If data is a dictionary, find the first non-dictionary element and get its shape[0]
            for k, v in data.items():
                if not isinstance(v, dict):
                    active_eval_env_num = v.shape[0] * v.shape[1]
                    agent_num = v.shape[1]  # multi-agent
        elif isinstance(data, torch.Tensor):
            # If data is a torch.tensor, directly return its shape[0]
            active_eval_env_num = data.shape[0]
            agent_num = 1  # single-agent
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

            if self._cfg.model.continuous_action_space is True:
                # when the action space of the environment is continuous, action_mask[:] is None.
                # NOTE: in continuous action space env: we set all legal_actions as -1
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(active_eval_env_num)
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
                    active_eval_env_num, legal_actions, self._cfg.model.action_space_size,
                    self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                )

            roots.prepare_no_noise(value_prefix_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, reward_hidden_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}
            # ==============================================================
            # sampled related core code
            # ==============================================================
            roots_sampled_actions = roots.get_sampled_actions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``

            if self._multi_agent:
                active_eval_env_num = active_eval_env_num // agent_num
            data_id = [i for i in range(active_eval_env_num)]
            output = {i: None for i in data_id}

            if ready_env_id is None:
                ready_env_id = np.arange(active_eval_env_num)

            for i, env_id in enumerate(ready_env_id):
                output[env_id] = {
                    'action': [],
                    'visit_count_distributions': [],
                    'root_sampled_actions': [],
                    'visit_count_distribution_entropy': [],
                    'searched_value': [],
                    'predicted_value': [],
                    'predicted_policy_logits': [],
                }
                for j in range(agent_num):
                    index = i * agent_num + j
                    distributions, value = roots_visit_count_distributions[index], roots_values[index]
                    try:
                        root_sampled_actions = np.array([action.value for action in roots_sampled_actions[index]])
                    except Exception:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        root_sampled_actions = np.array([action for action in roots_sampled_actions[index]])
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
                        action = roots_sampled_actions[index][action].value
                        # logging.warning('ptree_sampled_efficientzero roots.get_sampled_actions() return array')
                    except Exception:
                        # logging.warning('ctree_sampled_efficientzero roots.get_sampled_actions() return list')
                        action = np.array(roots_sampled_actions[index][action])

                    if not self._cfg.model.continuous_action_space:
                        if len(action.shape) == 0:
                            action = int(action)
                        elif len(action.shape) == 1:
                            action = int(action[0])

                    output[env_id]['action'].append(action)
                    output[env_id]['visit_count_distributions'].append(distributions)
                    output[env_id]['root_sampled_actions'].append(root_sampled_actions)
                    output[env_id]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                    output[env_id]['searched_value'].append(value)
                    output[env_id]['predicted_value'].append(pred_values[index])
                    output[env_id]['predicted_policy_logits'].append(policy_logits[index])
                
                for k, v in output[env_id].items():
                    output[env_id][k] = np.array(v)

        return output
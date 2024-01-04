from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from .efficientzero import EfficientZeroPolicy
from ding.utils import POLICY_REGISTRY

from lzero.mcts import EfficientZeroMCTSCtree as MCTSCtree
from lzero.mcts import EfficientZeroMCTSPtree as MCTSPtree
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, select_action, to_torch_float_tensor, ez_network_output_unpack, negative_cosine_similarity, prepare_obs, \
    configure_optimizers
from collections import defaultdict
from ding.utils.data import default_collate
from ding.torch_utils import to_device, to_tensor


@POLICY_REGISTRY.register('multi_agent_efficientzero')
class MultiAgentEfficientZeroPolicy(EfficientZeroPolicy):
    """
    Overview:
        The policy class for Multi Agent EfficientZero. 
        Independent Learning mode is a method in which each agent learns and adapts to the environment independently \
        without directly considering the learning and strategies of other agents.
    """

    def _forward_collect(
        self,
        data: torch.Tensor,
        action_mask: list = None,
        temperature: float = 1,
        to_play: List = [-1],
        epsilon: float = 0.25,
        ready_env_id = None
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
        data = sum(sum(data, []), [])
        batch_size = len(data)
        data = default_collate(data)
        data = to_device(data, self._device)
        agent_num = self._cfg['model']['agent_num']
        action_mask = sum(action_mask, [])
        to_play = np.array(to_play).reshape(-1).tolist()

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
                if self._cfg.eps.eps_greedy_exploration_in_collect:
                    # eps-greedy collect
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self.collect_mcts_temperature, deterministic=True
                    )
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    if np.random.rand() < self.collect_epsilon:
                        action = np.random.choice(legal_actions[i])
                else:
                    # normal collect
                    # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                    # the index within the legal action set, rather than the index in the entire action set.
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self.collect_mcts_temperature, deterministic=False
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
        data = sum(sum(data, []), [])
        batch_size = len(data)
        data = default_collate(data)
        data = to_device(data, self._device)
        agent_num = self._cfg['model']['agent_num']
        action_mask = sum(action_mask, [])
        to_play = np.array(to_play).reshape(-1).tolist()

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._eval_model.initial_inference(data)
            latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                network_output
            )

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            reward_hidden_state_roots = (
                reward_hidden_state_roots[0].detach().cpu().numpy(),
                reward_hidden_state_roots[1].detach().cpu().numpy()
            )
            policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

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
                # Save according to agent dimension
                output[i // agent_num]['action'].append(action)
                output[i // agent_num]['distributions'].append(distributions)
                output[i // agent_num]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                output[i // agent_num]['value'].append(value)
                output[i // agent_num]['pred_value'].append(pred_values[i])
                output[i // agent_num]['policy_logits'].append(policy_logits[i])

        return output
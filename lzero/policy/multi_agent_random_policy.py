from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY

from lzero.policy import InverseScalarTransform, select_action, ez_network_output_unpack, mz_network_output_unpack
from .random_policy import LightZeroRandomPolicy
from collections import defaultdict
from ding.torch_utils import to_device, to_tensor
from ding.utils.data import default_collate


@POLICY_REGISTRY.register('multi_agent_lightzero_random_policy')
class MultiAgentLightZeroRandomPolicy(LightZeroRandomPolicy):
    """
    Overview:
        The policy class for Multi Agent LightZero Random Policy.
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
        data = to_tensor(data)
        data = sum(sum(data, []), [])
        batch_size = len(data)
        data = default_collate(data)
        agent_num = batch_size // active_collect_env_num
        to_play = np.array(to_play).reshape(-1).tolist()

        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            if 'efficientzero' in self._cfg.type: # efficientzero or multi_agent_efficientzero
                latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                    network_output
                )
            elif 'muzero' in self._cfg.type: # muzero or multi_agent_muzero
                latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            if 'efficientzero' in self._cfg.type:
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
                roots = self.MCTSCtree.roots(batch_size, legal_actions)
            else:
                # python mcts_tree
                roots = self.MCTSPtree.roots(batch_size, legal_actions)
            if 'efficientzero' in self._cfg.type: # efficientzero or multi_agent_efficientzero
                roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
                self._mcts_collect.search(
                    roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
                )
            elif 'muzero' in self._cfg.type: # muzero or multi_agent_muzero
                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: defaultdict(list) for i in data_id}
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i in range(batch_size):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self.collect_mcts_temperature, deterministic=False
                )

                # ****** sample a random action from the legal action set ********
                # all items except action are formally obtained from MCTS 
                random_action = int(np.random.choice(legal_actions[i], 1))
                # ****************************************************************

                output[i // agent_num]['action'].append(random_action)
                output[i // agent_num]['distributions'].append(distributions)
                output[i // agent_num]['visit_count_distribution_entropy'].append(visit_count_distribution_entropy)
                output[i // agent_num]['value'].append(value)
                output[i // agent_num]['pred_value'].append(pred_values[i])
                output[i // agent_num]['policy_logits'].append(policy_logits[i])

        return output

from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY

from lzero.policy import InverseScalarTransform, select_action, ez_network_output_unpack, mz_network_output_unpack


@POLICY_REGISTRY.register('lightzero_random_policy')
class LightZeroRandomPolicy(Policy):
    """
    Overview:
        The policy class for LightZero RandomPolicy.
    """

    def __init__(
        self,
        cfg: dict,
        model: Optional[Union[type, torch.nn.Module]] = None,
        enable_field: Optional[List[str]] = None
    ):
        if 'muzero' in cfg.type:
            from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
            from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
        elif 'efficientzero' in cfg.type:
            from lzero.mcts import EfficientZeroMCTSCtree as MCTSCtree
            from lzero.mcts import EfficientZeroMCTSPtree as MCTSPtree
        else:
            raise NotImplementedError("need to implement pipeline: {}".format(cfg.type))
        self.MCTSCtree = MCTSCtree
        self.MCTSPtree = MCTSPtree
        super().__init__(cfg, model, enable_field)

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
        if self._cfg.model.model_type == "conv":
            if self._cfg.type == 'efficientzero':
                return 'EfficientZeroModel', ['lzero.model.efficientzero_model']
            elif self._cfg.type == 'muzero':
                return 'MuZeroModel', ['lzero.model.muzero_model']
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))
        elif self._cfg.model.model_type == "mlp":
            if self._cfg.type == 'efficientzero':
                return 'EfficientZeroModelMLP', ['lzero.model.efficientzero_model_mlp']
            elif self._cfg.type == 'muzero':
                return 'MuZeroModelMLP', ['lzero.model.muzero_model_mlp']
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))
        elif self._cfg.model.model_type == 'structure':
            if 'efficientzero' in self._cfg.type: # efficientzero or multi_agent_efficientzero
                return 'EfficientZeroModelStructure', ['lzero.model.efficientzero_model_structure']
            elif 'muzero' in self._cfg.type:  # muzero or multi_agent_muzero
                return 'MuZeroModelStructure', ['lzero.model.muzero_model_structure']
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

    def _init_collect(self) -> None:
        """
         Overview:
             Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
         """
        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_collect = self.MCTSCtree(self._cfg)
        else:
            self._mcts_collect = self.MCTSPtree(self._cfg)
        self.collect_mcts_temperature = 1
        self.collect_epsilon = 0.0
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

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
        active_collect_env_num = data.shape[0]
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            if self._cfg.type == 'efficientzero':
                latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                    network_output
                )
            elif self._cfg.type == 'muzero':
                latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            if self._cfg.type == 'efficientzero':
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            # the only difference between collect and eval is the dirichlet noise.
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = self.MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = self.MCTSPtree.roots(active_collect_env_num, legal_actions)
            if self._cfg.type == 'efficientzero':
                roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
                self._mcts_collect.search(
                    roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
                )
            elif self._cfg.type == 'muzero':
                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            roots_visit_count_distributions = roots.get_distributions(
            )  # shape: ``{list: batch_size} ->{list: action_space_size}``
            roots_values = roots.get_values()  # shape: {list: batch_size}

            data_id = [i for i in range(active_collect_env_num)]
            output = {i: None for i in data_id}
            if ready_env_id is None:
                ready_env_id = np.arange(active_collect_env_num)

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self.collect_mcts_temperature, deterministic=False
                )

                # ****** sample a random action from the legal action set ********
                # all items except action are formally obtained from MCTS 
                random_action = int(np.random.choice(legal_actions[env_id], 1))
                # ****************************************************************

                # NOTE: The action is randomly selected from the legal action set, the distribution is the real visit count distribution from the MCTS seraech.
                output[env_id] = {
                    'action': random_action,
                    'distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'value': value,
                    'pred_value': pred_values[i],
                    'policy_logits': policy_logits[i],
                }

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = self.MCTSCtree(self._cfg)
        else:
            self._mcts_eval = self.MCTSPtree(self._cfg)

    # be compatible with DI-engine Policy class
    def _init_learn(self) -> None:
        pass

    def _forward_learn(self, data: torch.Tensor) -> Dict[str, Union[float, int]]:
        pass

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: -1, ready_env_id=None):
        pass

    def _monitor_vars_learn(self) -> List[str]:
        pass

    def _state_dict_learn(self) -> Dict[str, Any]:
        pass

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        pass

    def _process_transition(self, obs, policy_output, timestep):
        pass

    def _get_train_sample(self, data):
        pass

from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY

from lzero.entry.utils import initialize_zeros_batch, initialize_pad_batch
from lzero.policy import DiscreteSupport, InverseScalarTransform, select_action, ez_network_output_unpack, mz_network_output_unpack


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
        enable_field: Optional[List[str]] = None,
        action_space: Any = None,
    ):
        if cfg.type == 'muzero':
            from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
            from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
        elif cfg.type == 'efficientzero':
            from lzero.mcts import EfficientZeroMCTSCtree as MCTSCtree
            from lzero.mcts import EfficientZeroMCTSPtree as MCTSPtree
        elif cfg.type == 'sampled_efficientzero':
            from lzero.mcts import SampledEfficientZeroMCTSCtree as MCTSCtree
            from lzero.mcts import SampledEfficientZeroMCTSPtree as MCTSPtree
        elif cfg.type == 'unizero':
            from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
        else:
            raise NotImplementedError("need to implement pipeline: {}".format(cfg.type))
        if cfg.mcts_ctree:
            self.MCTSCtree = MCTSCtree
        else:
            self.MCTSPtree = MCTSPtree
        self.action_space = action_space
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
            elif self._cfg.type == 'sampled_efficientzero':
                return 'SampledEfficientZeroModel', ['lzero.model.sampled_efficientzero_model']
            elif self._cfg.type == 'unizero':
                return 'UniZeroModel', ['lzero.model.unizero_model']
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))
        elif self._cfg.model.model_type == "mlp":
            if self._cfg.type == 'efficientzero':
                return 'EfficientZeroModelMLP', ['lzero.model.efficientzero_model_mlp']
            elif self._cfg.type == 'muzero':
                return 'MuZeroModelMLP', ['lzero.model.muzero_model_mlp']
            elif self._cfg.type == 'sampled_efficientzero':
                return 'SampledEfficientZeroModelMLP', ['lzero.model.sampled_efficientzero_model_mlp']
            elif self._cfg.type == 'unizero':
                return 'UniZeroModel', ['lzero.model.unizero_model']
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
        self._collect_mcts_temperature = 1
        self.collect_epsilon = 0.0
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        self.value_inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)
        self.reward_inverse_scalar_transform_handle = InverseScalarTransform(self.reward_support, self._cfg.model.categorical_distribution)
        if self._cfg.type == 'unizero':
            self.collector_env_num = self._cfg.collector_env_num
            if self._cfg.model.model_type == 'conv':
                self.last_batch_obs = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
                self.last_batch_action = [-1 for i in range(self.collector_env_num)]
                
            elif self._cfg.model.model_type == 'mlp':
                self.last_batch_obs = torch.full(
                    [self.collector_env_num, self._cfg.model.observation_shape], fill_value=self.pad_token_id,
                ).to(self._cfg.device)
                self.last_batch_action = [-1 for i in range(self.collector_env_num)]
    def _forward_collect(
        self,
        data: torch.Tensor,
        action_mask: list = None,
        temperature: float = 1,
        to_play: List = [-1],
        epsilon: float = 0.25,
        ready_env_id: np.array = None,
        timestep: List = [0]
    ) -> Dict:
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
            - timestep (:obj:`list`): The step index of the env in one episode.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]

        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            if self._cfg.type in ['efficientzero', 'sampled_efficientzero']:
                network_output = self._collect_model.initial_inference(data)
                latent_state_roots, value_prefix_roots, reward_hidden_state_roots, pred_values, policy_logits = ez_network_output_unpack(
                    network_output
                )
            elif self._cfg.type == 'muzero':
                network_output = self._collect_model.initial_inference(data)
                latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            elif self._cfg.type == 'unizero':
                network_output = self._collect_model.initial_inference(
                    self.last_batch_obs, self.last_batch_action, data, timestep
                )
                latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            if self._cfg.type in ['efficientzero', 'sampled_efficientzero']:
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            if self._cfg.model.continuous_action_space:
                legal_actions = [
                    [-1 for _ in range(self._cfg.model.num_of_sampled_actions)] for _ in range(active_collect_env_num)
                ]
            else:
                legal_actions = [
                    [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)
                ]

            if self._cfg.type in ['sampled_efficientzero']:
                noises = [
                    np.random.dirichlet(
                        [self._cfg.root_dirichlet_alpha] * int(self._cfg.model.num_of_sampled_actions)
                    ).astype(np.float32).tolist()
                    for _ in range(active_collect_env_num)
                ]
            else:
                noises = [
                    np.random.dirichlet(
                        [self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                    ).astype(np.float32).tolist()
                    for j in range(active_collect_env_num)
                ]

            if self._cfg.mcts_ctree:
                if self._cfg.type in ['sampled_efficientzero']:
                    roots = self.MCTSCtree.roots(
                        active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                else:
                    roots = self.MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                if self._cfg.type in ['sampled_efficientzero']:
                    roots = self.MCTSPtree.roots(
                        active_collect_env_num, legal_actions, self._cfg.model.action_space_size,
                        self._cfg.model.num_of_sampled_actions, self._cfg.model.continuous_action_space
                    )
                else:
                    roots = self.MCTSPtree.roots(active_collect_env_num, legal_actions)

            if self._cfg.type in ['efficientzero', 'sampled_efficientzero']:
                roots.prepare(self._cfg.root_noise_weight, noises, value_prefix_roots, policy_logits, to_play)
                self._mcts_collect.search(
                    roots, self._collect_model, latent_state_roots, reward_hidden_state_roots, to_play
                )
            elif self._cfg.type == 'muzero':
                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)
            elif self._cfg.type == 'unizero':
                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play, timestep)
            else:
                raise NotImplementedError("need to implement pipeline: {}".format(self._cfg.type))

            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}
            if self._cfg.type in ['sampled_efficientzero']:
                roots_sampled_actions = roots.get_sampled_actions()

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]

                if self._cfg.type in ['sampled_efficientzero']:
                    if self._cfg.mcts_ctree:
                        # In ctree, the method roots.get_sampled_actions() returns a list object.
                        root_sampled_actions = np.array([action for action in roots_sampled_actions[i]])
                    else:
                        # In ptree, the same method roots.get_sampled_actions() returns an Action object.
                        root_sampled_actions = np.array([action.value for action in roots_sampled_actions[i]])

                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=self._collect_mcts_temperature, deterministic=False
                )

                # ****************************************************************
                # NOTE: The action is randomly selected from the legal action set, 
                # the distribution is the real visit count distribution from the MCTS search.
                if self._cfg.type in ['sampled_efficientzero']:
                    # ****** sample a random action from the legal action set ********
                    random_action = self.action_space.sample()
                    output[env_id] = {
                        'action': random_action,
                        'visit_count_distributions': distributions,
                        'root_sampled_actions': root_sampled_actions,
                        'visit_count_distribution_entropy': visit_count_distribution_entropy,
                        'searched_value': value,
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }
                elif self._cfg.type == 'unizero':
                    # ****** sample a random action from the legal action set ********
                    random_action = int(np.random.choice(legal_actions[i], 1))
                    # all items except action are formally obtained from MCTS
                    output[env_id] = {
                        'action': random_action,
                        'visit_count_distributions': distributions,
                        'visit_count_distribution_entropy': visit_count_distribution_entropy,
                        'searched_value': value,
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }

            if self._cfg.type == 'unizero':
                batch_action = [output[env_id]['action'] for env_id in ready_env_id]
                self.last_batch_obs = data
                self.last_batch_action = batch_action

        return output


    def _reset_collect(self, env_id: int = None, current_steps: int = None, reset_init_data: bool = True) -> None:
        """
        Overview:
            This method resets the collection process for a specific environment. It clears caches and memory
            when certain conditions are met, ensuring optimal performance. If reset_init_data is True, the initial data
            will be reset.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. If None or list, the function returns immediately.
            - current_steps (:obj:`int`, optional): The current step count in the environment. Used to determine
              whether to clear caches.
            - reset_init_data (:obj:`bool`, optional): Whether to reset the initial data. If True, the initial data will be reset.
        """
        if self._cfg.type != 'unizero':
            return
        if reset_init_data:
            if self._cfg.model.model_type == 'conv':
                pad_token_id = -1
            else:
                encoder_tokenizer = getattr(self._model.tokenizer.encoder, 'tokenizer', None)
                spad_token_id = encoder_tokenizer.pad_token_id if encoder_tokenizer is not None else 0
            self.last_batch_obs = initialize_pad_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device,
                pad_token_id=pad_token_id
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

        # Return immediately if env_id is None or a list
        if env_id is None or isinstance(env_id, list):
            return

        # Determine the clear interval based on the environment's sample type
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else 200

        # Clear caches if the current steps are a multiple of the clear interval
        if current_steps % clear_interval == 0:
            print(f'clear_interval: {clear_interval}')

            # Clear various caches in the collect model's world model
            world_model = self._collect_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up GPU memory
            torch.cuda.empty_cache()

    
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

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: -1, ready_env_id: np.array = None,):
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

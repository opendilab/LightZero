import copy
from collections import namedtuple
from typing import List, Dict, Tuple

import numpy as np
import torch.distributions
import torch.nn.functional as F
import torch.optim as optim
from ding.policy.base_policy import Policy
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate
from easydict import EasyDict

from lzero.policy import configure_optimizers
from lzero.policy.utils import pad_and_get_lengths, compute_entropy


@POLICY_REGISTRY.register('sampled_alphazero')
class SampledAlphaZeroPolicy(Policy):
    """
    Overview:
        The policy class for Sampled AlphaZero.
    """

    # The default_config for AlphaZero policy.
    config = dict(
        # (str) The type of policy, as the key of the policy registry.
        type='alphazero',
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled AlphaZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        normalize_prob_of_sampled_actions=False,
        policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        # (bool) Whether to use torch.compile method to speed up our model, which required torch>=2.0.
        torch_compile=False,
        # (bool) Whether to use TF32 for our model.
        tensor_float_32=False,
        model=dict(
            # (tuple) The stacked obs shape.
            observation_shape=(3, 6, 6),
            # (int) The number of res blocks in AlphaZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in AlphaZero model.
            num_channels=32,
        ),
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.model_update_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        model_update_ratio=0.1,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam', 'AdamW']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (float) The weight of value loss.
        value_weight=1.0,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e5),
        # (bool) Whether to use manually temperature decay.
        # i.e. temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        mcts=dict(
            # (int) The number of simulations to perform at each move.
            num_simulations=50,
            # (int) The maximum number of moves to make in a game.
            max_moves=512,  # for chess and shogi, 722 for Go.
            # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
            root_dirichlet_alpha=0.3,
            # (float) The noise weight at the root node of the search tree.
            root_noise_weight=0.25,
            # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_base=19652,
            # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_init=1.25,
            #
            legal_actions=None,
            # (int) The action space size.
            action_space_size=9,
            # (int) The number of sampled actions for each state.
            num_of_sampled_actions=2,
            #
            continuous_action_space=False,
        ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=int(1e6),
            save_episode=False,
        )),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
            - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        """
        return 'AlphaZeroModel', ['lzero.model.alphazero_model']

    def _init_learn(self) -> None:
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
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
        elif self._cfg.optim_type == 'AdamW':
            self._optimizer = configure_optimizers(
                model=self._model,
                weight_decay=self._cfg.weight_decay,
                learning_rate=self._cfg.learning_rate,
                device_type=self._cfg.device
            )

        if self._cfg.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            # lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            lr_lambda = lambda step: 1 if step < max_step * 0.33 else (0.1 if step < max_step * 0.66 else 0.01)  # noqa

            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # Algorithm config
        self._value_weight = self._cfg.value_weight
        self._entropy_weight = self._cfg.entropy_weight
        # Main and target models
        self._learn_model = self._model

        # TODO(pu): test the effect of torch 2.0
        if self._cfg.torch_compile:
            self._learn_model = torch.compile(self._learn_model)

    def _forward_learn(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        for input_dict in inputs:
            # Check and remove 'katago_game_state' from 'obs' if it exists
            if 'katago_game_state' in input_dict['obs']:
                del input_dict['obs']['katago_game_state']

            # Check and remove 'katago_game_state' from 'next_obs' if it exists
            if 'katago_game_state' in input_dict['next_obs']:
                del input_dict['next_obs']['katago_game_state']

        # list of dict -> dict of list
        # inputs_deepcopy = copy.deepcopy(inputs)
        # only for env with variable legal actions
        inputs = pad_and_get_lengths(inputs, self._cfg.mcts.num_of_sampled_actions)
        inputs = default_collate(inputs)
        valid_action_length = inputs['action_length']

        if self._cuda:
            inputs = to_device(inputs, self._device)
        self._learn_model.train()

        state_batch = inputs['obs']['observation']
        mcts_visit_count_probs = inputs['probs']
        reward = inputs['reward']
        root_sampled_actions = inputs['root_sampled_actions']

        if len(root_sampled_actions.shape) == 1:
            print(f"root_sampled_actions.shape: {root_sampled_actions.shape}")
        state_batch = state_batch.to(device=self._device, dtype=torch.float)
        mcts_visit_count_probs = mcts_visit_count_probs.to(device=self._device, dtype=torch.float)
        reward = reward.to(device=self._device, dtype=torch.float)

        policy_probs, values = self._learn_model.compute_policy_value(state_batch)
        policy_log_probs = torch.log(policy_probs)

        # calculate policy entropy, for monitoring only
        entropy = compute_entropy(policy_probs)
        entropy_loss = -entropy

        # ==============================================================
        # policy loss
        # ==============================================================
        # mcts_visit_count_probs = mcts_visit_count_probs / (mcts_visit_count_probs.sum(dim=1, keepdim=True) + 1e-6)
        # policy_loss = torch.nn.functional.kl_div(
        #     policy_log_probs, mcts_visit_count_probs, reduction='batchmean'
        # )
        # orig implementation
        # policy_loss = -torch.mean(torch.sum(mcts_visit_count_probs * policy_log_probs, 1))

        policy_loss = self._calculate_policy_loss_disc(policy_probs, mcts_visit_count_probs, root_sampled_actions, valid_action_length)

        # ==============================================================
        # value loss
        # ==============================================================
        value_loss = F.mse_loss(values.view(-1), reward)

        total_loss = self._value_weight * value_loss + policy_loss + self._entropy_weight * entropy_loss
        self._optimizer.zero_grad()
        total_loss.backward()

        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=self._cfg.grad_clip_value,
        )
        self._optimizer.step()
        if self._cfg.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            'collect_mcts_temperature': self.collect_mcts_temperature,
        }

    def _calculate_policy_loss_disc(
            self, policy_probs: torch.Tensor, target_policy: torch.Tensor,
            target_sampled_actions: torch.Tensor, valid_action_lengths: torch.Tensor
    ) -> torch.Tensor:

        # For each batch and each sampled action, get the corresponding probability
        # from policy_probs and target_policy, and put it into sampled_policy_probs and
        # sampled_target_policy at the same position.
        sampled_policy_probs = policy_probs.gather(1, target_sampled_actions)
        sampled_target_policy = target_policy.gather(1, target_sampled_actions)

        # Create a mask for valid actions
        max_length = target_sampled_actions.size(1)
        mask = torch.arange(max_length).expand(len(valid_action_lengths), max_length) < valid_action_lengths.unsqueeze(
            1)
        mask = mask.to(device=self._device)

        # Apply the mask to sampled_policy_probs and sampled_target_policy
        sampled_policy_probs = sampled_policy_probs * mask.float()
        sampled_target_policy = sampled_target_policy * mask.float()

        # Normalize sampled_policy_probs and sampled_target_policy
        sampled_policy_probs = sampled_policy_probs / (sampled_policy_probs.sum(dim=1, keepdim=True) + 1e-6)
        sampled_target_policy = sampled_target_policy / (sampled_target_policy.sum(dim=1, keepdim=True) + 1e-6)

        # after normalization, the sum of each row should be 1, but the prob corresponding to valid action becomes a small non-zero value
        # Use torch.where to prevent gradients for invalid actions
        sampled_policy_probs = torch.where(mask, sampled_policy_probs, torch.zeros_like(sampled_policy_probs))
        sampled_target_policy = torch.where(mask, sampled_target_policy, torch.zeros_like(sampled_target_policy))

        if self._cfg.policy_loss_type == 'KL':
            # Calculate the KL divergence between sampled_policy_probs and sampled_target_policy
            # The KL divergence between 2 probability distributions P and Q is defined as:
            # KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
            # We use the PyTorch function kl_div to calculate it.
            loss = torch.nn.functional.kl_div(
                sampled_policy_probs.log(), sampled_target_policy, reduction='none'
            )

            # TODO(pu)
            # 使用 nan_to_num 将 loss 中的 nan 值设置为0
            loss = torch.nan_to_num(loss)

            # Apply the mask to the loss
            loss = loss * mask.float()
            # Calculate the mean loss over the batch
            loss = loss.sum() / mask.sum()

        elif self._cfg.policy_loss_type == 'cross_entropy':
            # Calculate the cross entropy loss between sampled_policy_probs and sampled_target_policy
            # The cross entropy between 2 probability distributions P and Q is defined as:
            # H(P, Q) = -sum(P(i) * log(Q(i)))
            # We use the PyTorch function cross_entropy to calculate it.
            loss = torch.nn.functional.cross_entropy(
                sampled_policy_probs, torch.argmax(sampled_target_policy, dim=1), reduction='none'
            )

            # 使用 nan_to_num 将 loss 中的 nan 值设置为0
            loss = torch.nan_to_num(loss)

            # Apply the mask to the loss
            loss = loss * mask.float()
            # Calculate the mean loss over the batch
            loss = loss.sum() / mask.sum()

        else:
            raise ValueError(f"Invalid policy_loss_type: {self._cfg.policy_loss_type}")


        return loss

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._get_simulation_env()

        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            self._collect_mcts = mcts_alphazero.MCTS(self._cfg.mcts.max_moves, self._cfg.mcts.num_simulations,
                                                     self._cfg.mcts.pb_c_base,
                                                     self._cfg.mcts.pb_c_init, self._cfg.mcts.root_dirichlet_alpha,
                                                     self._cfg.mcts.root_noise_weight, self.simulate_env)
        else:
            if self._cfg.sampled_algo:
                from lzero.mcts.ptree.ptree_az_sampled import MCTS
            else:
                from lzero.mcts.ptree.ptree_az import MCTS
            self._collect_mcts = MCTS(self._cfg.mcts, self.simulate_env)

        self.collect_mcts_temperature = 1

    @torch.no_grad()
    def _forward_collect(self, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:

        """
        Overview:
            The forward function for collecting data in collect mode. Use real env to execute MCTS search.
        Arguments:
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
            - temperature (:obj:`float`): The temperature for MCTS search.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
        """
        self.collect_mcts_temperature = temperature
        ready_env_id = list(obs.keys())
        init_state = {env_id: obs[env_id]['board'] for env_id in ready_env_id}
        try:
            katago_game_state = {env_id: obs[env_id]['katago_game_state'] for env_id in ready_env_id}
        except Exception as e:
            katago_game_state = {env_id: None for env_id in ready_env_id}

        start_player_index = {env_id: obs[env_id]['current_player_index'] for env_id in ready_env_id}
        output = {}
        self._policy_model = self._collect_model
        for env_id in ready_env_id:
            # print('[collect] start_player_index={}'.format(start_player_index[env_id]))
            # print('[collect] init_state=\n{}'.format(init_state[env_id]))

            state_config_for_env_reset = EasyDict(dict(start_player_index=start_player_index[env_id],
                                                       init_state=init_state[env_id],
                                                       katago_policy_init=True,
                                                       katago_game_state=katago_game_state[env_id]))

            action, mcts_visit_count_probs = self._collect_mcts.get_next_action(
                state_config_for_env_reset,
                self._policy_value_func,
                self.collect_mcts_temperature,
                True,
            )

            # if np.array_equal(self._collect_mcts.get_sampled_actions(), np.array([2, 2, 3])):
            #     print('debug')
            output[env_id] = {
                'action': action,
                'probs': mcts_visit_count_probs,
                'root_sampled_actions': self._collect_mcts.get_sampled_actions(),
            }

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._get_simulation_env()
        # TODO(pu): use double num_simulations for evaluation
        if self._cfg.mcts_ctree:
            self._eval_mcts = mcts_alphazero.MCTS(self._cfg.mcts.max_moves, 2 * self._cfg.mcts.num_simulations,
                                                  self._cfg.mcts.pb_c_base,
                                                  self._cfg.mcts.pb_c_init, self._cfg.mcts.root_dirichlet_alpha,
                                                  self._cfg.mcts.root_noise_weight, self.simulate_env)
        else:
            if self._cfg.sampled_algo:
                from lzero.mcts.ptree.ptree_az_sampled import MCTS
            else:
                from lzero.mcts.ptree.ptree_az import MCTS
            mcts_eval_config = copy.deepcopy(self._cfg.mcts)
            # TODO
            mcts_eval_config.num_simulations = mcts_eval_config.num_simulations
            # mcts_eval_config.num_simulations = min(800, mcts_eval_config.num_simulations * 4)

            self._eval_mcts = MCTS(mcts_eval_config, self.simulate_env)

        self._eval_model = self._model

    def _forward_eval(self, obs: Dict) -> Dict[str, torch.Tensor]:

        """
        Overview:
            The forward function for evaluating the current policy in eval mode, similar to ``self._forward_collect``.
        Arguments:
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
        """
        ready_env_id = list(obs.keys())
        init_state = {env_id: obs[env_id]['board'] for env_id in ready_env_id}
        try:
            katago_game_state = {env_id: obs[env_id]['katago_game_state'] for env_id in ready_env_id}
        except Exception as e:
            katago_game_state = {env_id: None for env_id in ready_env_id}

        start_player_index = {env_id: obs[env_id]['current_player_index'] for env_id in ready_env_id}
        output = {}
        self._policy_model = self._eval_model
        for env_id in ready_env_id:
            # print('[eval] start_player_index={}'.format(start_player_index[env_id]))
            # print('[eval] init_state=\n {}'.format(init_state[env_id]))

            state_config_for_env_reset = EasyDict(dict(start_player_index=start_player_index[env_id],
                                                       init_state=init_state[env_id],
                                                       katago_policy_init=False,
                                                       katago_game_state=katago_game_state[env_id]))

            # try:
            action, mcts_visit_count_probs = self._eval_mcts.get_next_action(state_config_for_env_reset, self._policy_value_func,
                                                                 1.0, False)
            # except Exception as e:
            #     print(f"Exception occurred: {e}")
            #     print(f"Is self._policy_value_func callable? {callable(self._policy_value_func)}")
            #     raise  # re-raise the exception
            # print("="*20)
            # print(action, mcts_visit_count_probs)
            # print("="*20)
            output[env_id] = {
                'action': action,
                'probs': mcts_visit_count_probs,
            }
        return output

    def _get_simulation_env(self):
        assert self._cfg.simulation_env_name in ['tictactoe', 'gomoku', 'go'], self._cfg.simulation_env_name
        assert self._cfg.simulation_env_config_type in ['play_with_bot', 'self_play', 'league', 'sampled_play_with_bot'], self._cfg.simulation_env_config_type
        if self._cfg.simulation_env_name == 'tictactoe':
            from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
            if self._cfg.simulation_env_config_type == 'play_with_bot':
                from zoo.board_games.tictactoe.config.tictactoe_alphazero_bot_mode_config import \
                    tictactoe_alphazero_config
            elif self._cfg.simulation_env_config_type == 'self_play':
                from zoo.board_games.tictactoe.config.tictactoe_alphazero_sp_mode_config import \
                    tictactoe_alphazero_config
            elif self._cfg.simulation_env_config_type == 'league':
                from zoo.board_games.tictactoe.config.tictactoe_alphazero_league_config import \
                    tictactoe_alphazero_config
            elif self._cfg.simulation_env_config_type == 'sampled_play_with_bot':
                from zoo.board_games.tictactoe.config.tictactoe_sampled_alphazero_bot_mode_config import \
                    tictactoe_sampled_alphazero_config as tictactoe_alphazero_config

            self.simulate_env = TicTacToeEnv(tictactoe_alphazero_config.env)

        elif self._cfg.simulation_env_name == 'gomoku':
            from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv
            if self._cfg.simulation_env_config_type == 'play_with_bot':
                from zoo.board_games.gomoku.config.gomoku_alphazero_bot_mode_config import gomoku_alphazero_config
            elif self._cfg.simulation_env_config_type == 'self_play':
                from zoo.board_games.gomoku.config.gomoku_alphazero_sp_mode_config import gomoku_alphazero_config
            elif self._cfg.simulation_env_config_type == 'league':
                from zoo.board_games.gomoku.config.gomoku_alphazero_league_config import gomoku_alphazero_config
            elif self._cfg.simulation_env_config_type == 'sampled_play_with_bot':
                from zoo.board_games.gomoku.config.gomoku_sampled_alphazero_bot_mode_config import gomoku_sampled_alphazero_config as gomoku_alphazero_config

            self.simulate_env = GomokuEnv(gomoku_alphazero_config.env)
        elif self._cfg.simulation_env_name == 'go':
            from zoo.board_games.go.envs.go_env import GoEnv
            if self._cfg.simulation_env_config_type == 'play_with_bot':
                from zoo.board_games.go.config.go_alphazero_bot_mode_config import go_alphazero_config
            elif self._cfg.simulation_env_config_type == 'self_play':
                from zoo.board_games.go.config.go_alphazero_sp_mode_config import go_alphazero_config
            elif self._cfg.simulation_env_config_type == 'league':
                from zoo.board_games.go.config.go_alphazero_league_config import go_alphazero_config
            elif self._cfg.simulation_env_config_type == 'sampled_play_with_bot':
                from zoo.board_games.go.config.go_sampled_alphazero_bot_mode_config import \
                    go_sampled_alphazero_config as go_alphazero_config

            self.simulate_env = GoEnv(go_alphazero_config.env)

    @torch.no_grad()
    def _policy_value_func(self, environment: 'Environment') -> Tuple[Dict[int, np.ndarray], float]:
        # Retrieve the legal actions in the current environment
        legal_actions = environment.legal_actions

        # Retrieve the current state and its scale from the environment
        current_state, state_scale = environment.current_state()

        # Convert the state scale to a PyTorch FloatTensor, adding a dimension to match the model's input requirements
        state_scale_tensor = torch.from_numpy(state_scale).to(
            device=self._device, dtype=torch.float
        ).unsqueeze(0)

        # Compute action probabilities and state value for the current state using the policy model, without gradient computation
        with torch.no_grad():
            action_probabilities, state_value = self._policy_model.compute_policy_value(state_scale_tensor)

        # Extract the probabilities of the legal actions from the action probabilities, and convert the result to a numpy array
        legal_action_probabilities = dict(
            zip(legal_actions, action_probabilities.squeeze(0)[legal_actions].detach().cpu().numpy()))

        # Return probabilities of the legal actions and the state value
        return legal_action_probabilities, state_value.item()

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return super()._monitor_vars_learn() + [
            'cur_lr', 'total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'total_grad_norm_before_clip',
            'collect_mcts_temperature'
        ]

    def _process_transition(self, obs: Dict, model_output: Dict[str, torch.Tensor], timestep: namedtuple) -> Dict:
        """
        Overview:
            Generate the dict type transition (one timestep) data from policy learning.
        """
        if 'katago_game_state' in obs.keys():
            del obs['katago_game_state']
        # if 'katago_game_state' in timestep.obs.keys():
        #     del timestep.obs['katago_game_state']
        # Note: used in _foward_collect  in alphazero_collector now

        return {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'root_sampled_actions': model_output['root_sampled_actions'],
            'probs': model_output['probs'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass
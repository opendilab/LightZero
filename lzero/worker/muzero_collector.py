import time
from collections import deque, namedtuple
from typing import Optional, Any, List, Dict, Set

import numpy as np
import torch
import wandb
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, get_rank, get_world_size, \
    allreduce_data
from ding.worker.collector.base_serial_collector import ISerialCollector
from torch.nn import L1Loss
import torch.distributed as dist

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation


@SERIAL_COLLECTOR_REGISTRY.register('episode_muzero')
class MuZeroCollector(ISerialCollector):
    """
    Overview:
        The episode-based collector for MCTS-based reinforcement learning algorithms, 
        including MuZero, EfficientZero, Sampled EfficientZero, and Gumbel MuZero.
        It orchestrates the data collection process in a serial manner, managing interactions 
        between the policy and the environment to generate game segments for training.
    Interfaces:
        ``__init__``, ``reset``, ``reset_env``, ``reset_policy``, ``_reset_stat``, ``collect``, 
        ``_compute_priorities``, ``pad_and_save_last_trajectory``, ``_output_log``, ``close``, ``__del__``.
    Properties:
        ``envstep``.
    """

    # Default configuration for the collector. To be compatible with ISerialCollector.
    config = dict()

    def __init__(
            self,
            collect_print_freq: int = 100,
            env: Optional[BaseEnvManager] = None,
            policy: Optional[namedtuple] = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: str = 'default_experiment',
            instance_name: str = 'collector',
            policy_config: 'policy_config' = None,  # noqa
            task_id: Optional[int] = None,
    ) -> None:
        """
        Overview:
            Initializes the MuZeroCollector with the given configuration.
        Arguments:
            - collect_print_freq (:obj:`int`): The frequency (in training iterations) at which to print collection statistics.
            - env (:obj:`Optional[BaseEnvManager]`): An instance of a vectorized environment manager.
            - policy (:obj:`Optional[namedtuple]`): A namedtuple containing the policy's forward pass and other methods.
            - tb_logger (:obj:`Optional[SummaryWriter]`): A TensorBoard logger instance for logging metrics.
            - exp_name (:obj:`str`): The name of the experiment, used for organizing logs.
            - instance_name (:obj:`str`): A unique name for this collector instance.
            - policy_config (:obj:`'policy_config'`): The configuration object for the policy.
            - task_id (:obj:`Optional[int]`): The identifier for the current task in a multi-task setting. If None, operates in single-task mode.
        """
        self.task_id = task_id
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = collect_print_freq
        self._timer = EasyTimer()
        self._end_flag = False

        # Get distributed training info
        self._rank = get_rank()
        self._world_size = get_world_size()

        # Logger setup: only rank 0 creates the main logger and TensorBoard logger.
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    path=f'./{self._exp_name}/log/{self._instance_name}',
                    name=self._instance_name,
                    need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    path=f'./{self._exp_name}/log/{self._instance_name}', name=self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                path=f'./{self._exp_name}/log/{self._instance_name}', name=self._instance_name, need_tb=False
            )
            self._tb_logger = None

        self.policy_config = policy_config
        self.collect_with_pure_policy = self.policy_config.collect_with_pure_policy

        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Resets or replaces the environment managed by the collector.
            If `_env` is None, it resets the existing environment. Otherwise, it replaces the old
            environment with the new one and launches it.
        Arguments:
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment to be used. If None, resets the current environment.
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Resets or replaces the policy used by the collector.
            If `_policy` is None, it resets the existing policy. Otherwise, it replaces the old
            policy with the new one.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to be used.
        """
        assert hasattr(self, '_env'), "Please set env first before resetting policy."
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').get('n_episode', None)
            self._logger.debug(
                f"Set default n_episode mode(n_episode({self._default_n_episode}), env_num({self._env_num}))"
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Resets the collector, including the environment and policy. Also re-initializes
            internal state variables for tracking collection progress.
        Arguments:
            - _policy (:obj:`Optional[namedtuple]`): The new policy to use.
            - _env (:obj:`Optional[BaseEnvManager]`): The new environment to use.
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        # Initialize per-environment tracking info
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        # Reset overall statistics
        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False

        # A pool to store completed game segments, implemented using a deque.
        self.game_segment_pool = deque(maxlen=int(1e6))
        self.unroll_plus_td_steps = self.policy_config.num_unroll_steps + self.policy_config.td_steps

    def _reset_stat(self, env_id: int) -> None:
        """
        Overview:
            Resets the statistics for a specific environment, identified by `env_id`.
            This is typically called when an episode in that environment ends.
        Arguments:
            - env_id (:obj:`int`): The ID of the environment to reset statistics for.
        """
        self._env_info[env_id] = {'time': 0., 'step': 0}

    @property
    def envstep(self) -> int:
        """
        Overview:
            Returns the total number of environment steps collected since the last reset.
        Returns:
            - envstep (:obj:`int`): The total environment step count.
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Closes the collector, including the environment and any loggers.
            Ensures that all resources are properly released.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Destructor for the collector instance, ensuring that `close` is called
            to clean up resources.
        """
        self.close()

    # ==============================================================
    # MCTS+RL Core Collection Logic
    # ==============================================================
    def _compute_priorities(self, i: int, pred_values_lst: List[float], search_values_lst: List[float]) -> Optional[np.ndarray]:
        """
        Overview:
            Computes priorities for experience replay based on the discrepancy between
            predicted values and MCTS search values.
        Arguments:
            - i (:obj:`int`): The index of the environment's data in the lists.
            - pred_values_lst (:obj:`List[float]`): A list containing lists of predicted values for each environment.
            - search_values_lst (:obj:`List[float]`): A list containing lists of search values from MCTS for each environment.
        Returns:
            - priorities (:obj:`Optional[np.ndarray]`): An array of priorities for the transitions. Returns None if priority is not used.
        """
        if self.policy_config.use_priority:
            # Calculate priorities as the L1 loss between predicted values and search values.
            # 'reduction=none' ensures the loss is calculated for each element individually.
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.policy_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.policy_config.device).float().view(-1)
            
            # A small epsilon is added to avoid zero priorities.
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + 1e-6
        else:
            # If priority is not used, return None. The replay buffer will use max priority for new data.
            priorities = None

        return priorities

    def pad_and_save_last_trajectory(
            self, i: int, last_game_segments: List[Optional[GameSegment]],
            last_game_priorities: List[Optional[np.ndarray]],
            game_segments: List[GameSegment], done: np.ndarray
    ) -> None:
        """
        Overview:
            Pads the end of the `last_game_segment` with data from the start of the current `game_segment`.
            This is necessary to compute target values for the final transitions of a segment. After padding,
            the completed segment is stored in the `game_segment_pool`.
        Arguments:
            - i (:obj:`int`): The index of the environment being processed.
            - last_game_segments (:obj:`List[Optional[GameSegment]]`): List of game segments from the previous collection chunk.
            - last_game_priorities (:obj:`List[Optional[np.ndarray]]`): List of priorities corresponding to the last game segments.
            - game_segments (:obj:`List[GameSegment]`): List of game segments from the current collection chunk.
            - done (:obj:`np.ndarray`): Array indicating if the episode has terminated for each environment.
        Note:
            An implicit assumption is that the start of the new segment's observation history overlaps with the
            end of the last segment's, e.g., `(last_game_segments[i].obs_segment[-4:][j] == game_segments[i].obs_segment[:4][j]).all()` is True.
        """
        # --- Prepare padding data from the current game segment ---
        # Observations for padding are taken from the start of the new segment.
        beg_index_obs = self.policy_config.model.frame_stack_num
        end_index_obs = beg_index_obs + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        pad_obs_lst = game_segments[i].obs_segment[beg_index_obs:end_index_obs]

        # Actions for padding.
        beg_index_ac = 0
        end_index_ac = beg_index_ac + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        pad_action_lst = game_segments[i].action_segment[beg_index_ac:end_index_ac]

        # Child visits for padding.
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self.policy_config.num_unroll_steps + self.policy_config.td_steps]

        # Rewards for padding.
        beg_index_rew = 0
        end_index_rew = beg_index_rew + self.unroll_plus_td_steps - 1
        pad_reward_lst = game_segments[i].reward_segment[beg_index_rew:end_index_rew]
        
        # Root values for padding.
        beg_index_val = 0
        end_index_val = beg_index_val + self.unroll_plus_td_steps
        pad_root_values_lst = game_segments[i].root_value_segment[beg_index_val:end_index_val]

        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_lst = game_segments[i].chance_segment[beg_index_rew:end_index_rew]
        
        if self.policy_config.gumbel_algo:
            pad_improved_policy_prob = game_segments[i].improved_policy_probs[beg_index_val:end_index_val]

        # --- Pad the last game segment and save it ---
        if self.policy_config.gumbel_algo:
            last_game_segments[i].pad_over(
                pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst,
                pad_child_visits_lst, next_segment_improved_policy=pad_improved_policy_prob
            )
        else:
            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst,
                    pad_child_visits_lst, next_chances=chance_lst
                )
            else:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst
                )
        
        # Convert the segment's lists to NumPy arrays for efficient storage.
        last_game_segments[i].game_segment_to_array()

        # Add the completed game segment and its associated data to the pool.
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # Reset the placeholder for the last game segment.
        last_game_segments[i] = None
        last_game_priorities[i] = None

    def collect(
            self,
            n_episode: Optional[int] = None,
            train_iter: int = 0,
            policy_kwargs: Optional[Dict] = None,
            collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        Overview:
            Collects `n_episode` episodes of data. It manages the entire lifecycle of an episode,
            from getting actions from the policy, stepping the environment, storing transitions,
            and saving completed game segments.
        Arguments:
            - n_episode (:obj:`Optional[int]`): The number of episodes to collect. If None, uses the default from the policy config.
            - train_iter (:obj:`int`): The current training iteration, used for logging.
            - policy_kwargs (:obj:`Optional[Dict]`): Additional keyword arguments to pass to the policy's forward method, like temperature for exploration.
            - collect_with_pure_policy (:obj:`bool`): If True, collects data using a pure policy (e.g., greedy action) without MCTS.
        Returns:
            - return_data (:obj:`List[Any]`): A list containing the collected game segments and metadata.
        """
        # TODO(author): Consider implementing `collect_with_pure_policy` as a separate, more streamlined collector for clarity and modularity.
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify `n_episode` for collection.")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, f"Please ensure n_episode ({n_episode}) >= env_num ({self._env_num})."
        
        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)

        # --- Initializations ---
        collected_episode = 0
        env_nums = self._env_num
        retry_waiting_time = 0.05

        # Wait for all environments to be ready and get initial observations.
        init_obs = self._env.ready_obs
        while len(init_obs.keys()) != self._env_num:
            self._logger.warning(f"Waiting for all environments to reset. Ready envs: {list(init_obs.keys())}")
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        # Prepare initial state dictionaries from observations.
        action_mask_dict = {i: to_ndarray(init_obs[i]['action_mask']) for i in range(env_nums)}
        to_play_dict = {i: to_ndarray(init_obs[i]['to_play']) for i in range(env_nums)}
        timestep_dict = {i: to_ndarray(init_obs[i].get('timestep', -1)) for i in range(env_nums)}
        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_dict = {i: to_ndarray(init_obs[i]['chance']) for i in range(env_nums)}

        # Initialize game segments and observation stacks for each environment.
        game_segments = [GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config) for _ in range(env_nums)]
        observation_window_stack = [deque(maxlen=self.policy_config.model.frame_stack_num) for _ in range(env_nums)]
        for env_id in range(env_nums):
            for _ in range(self.policy_config.model.frame_stack_num):
                observation_window_stack[env_id].append(to_ndarray(init_obs[env_id]['observation']))
            game_segments[env_id].reset(observation_window_stack[env_id])

        # State tracking variables for the collection loop.
        dones = np.array([False for _ in range(env_nums)])
        last_game_segments: List[Optional[GameSegment]] = [None for _ in range(env_nums)]
        last_game_priorities: List[Optional[np.ndarray]] = [None for _ in range(env_nums)]
        
        # Buffers for priority calculation.
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]
        if self.policy_config.gumbel_algo:
            improved_policy_lst = [[] for _ in range(env_nums)]

        # Logging variables.
        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)
        if self.policy_config.gumbel_algo:
            completed_value_lst = np.zeros(env_nums)

        ready_env_id: Set[int] = set()
        remain_episode = n_episode
        if collect_with_pure_policy:
            # Dummy visit counts for pure policy collection.
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]

        # --- Main Collection Loop ---
        while True:
            with self._timer:
                # Get observations from ready environments.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id.update(list(new_available_env_id)[:remain_episode])
                remain_episode -= min(len(new_available_env_id), remain_episode)
                
                # Prepare policy inputs.
                stack_obs_list = [game_segments[env_id].get_obs() for env_id in ready_env_id]
                action_mask = [action_mask_dict[env_id] for env_id in ready_env_id]
                to_play = [to_play_dict[env_id] for env_id in ready_env_id]
                timestep = [timestep_dict[env_id] for env_id in ready_env_id]
                
                stack_obs_array = to_ndarray(stack_obs_list)
                stack_obs_tensor = prepare_observation(stack_obs_array, self.policy_config.model.model_type)
                stack_obs_tensor = torch.from_numpy(stack_obs_tensor).to(self.policy_config.device)

                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_input = {
                    'x': stack_obs_tensor,
                    'action_mask': action_mask,
                    'temperature': temperature,
                    'to_play': to_play,
                    'epsilon': epsilon,
                    'ready_env_id': ready_env_id,
                    'timestep': timestep
                }
                if self.task_id is not None:
                    policy_input['task_id'] = self.task_id
                
                policy_output = self._policy.forward(**policy_input)
                
                # --- Unpack policy outputs ---
                actions, value_dict, pred_value_dict = {}, {}, {}
                distributions_dict, visit_entropy_dict = {}, {}
                if self.policy_config.sampled_algo:
                    root_sampled_actions_dict = {}
                if self.policy_config.gumbel_algo:
                    improved_policy_dict, completed_value_dict = {}, {}

                for env_id in ready_env_id:
                    output = policy_output[env_id]
                    actions[env_id] = output['action']
                    value_dict[env_id] = output['searched_value']
                    pred_value_dict[env_id] = output['predicted_value']
                    
                    if not collect_with_pure_policy:
                        distributions_dict[env_id] = output['visit_count_distributions']
                        visit_entropy_dict[env_id] = output['visit_count_distribution_entropy']
                        if self.policy_config.sampled_algo:
                            root_sampled_actions_dict[env_id] = output['root_sampled_actions']
                        if self.policy_config.gumbel_algo:
                            improved_policy_dict[env_id] = output['improved_policy_probs']
                            completed_value_dict[env_id] = output['roots_completed_value']

                # ==============================================================
                # Environment Interaction
                # ==============================================================
                timesteps = self._env.step(actions)

            interaction_duration = self._timer.value / len(timesteps) if timesteps else 0

            for env_id, episode_timestep in timesteps.items():
                with self._timer:
                    # Handle abnormal timesteps by resetting the environment and policy state.
                    if episode_timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info(f"Environment {env_id} returned an abnormal step, info: {episode_timestep.info}")
                        continue

                    obs, reward, done, info = episode_timestep.obs, episode_timestep.reward, episode_timestep.done, episode_timestep.info

                    # Store MCTS search statistics.
                    if collect_with_pure_policy:
                        game_segments[env_id].store_search_stats(temp_visit_list, 0)
                    else:
                        if self.policy_config.sampled_algo:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id], root_sampled_actions_dict[env_id])
                        elif self.policy_config.gumbel_algo:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id], improved_policy=improved_policy_dict[env_id])
                        else:
                            game_segments[env_id].store_search_stats(distributions_dict[env_id], value_dict[env_id])

                    # Append the current transition to the game segment.
                    append_args = (actions[env_id], to_ndarray(obs['observation']), reward, action_mask_dict[env_id], to_play_dict[env_id])
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        append_args += (chance_dict[env_id],)
                    append_args += (timestep_dict[env_id],)
                    game_segments[env_id].append(*append_args)

                    # Update state dictionaries for the next step.
                    action_mask_dict[env_id] = to_ndarray(obs['action_mask'])
                    to_play_dict[env_id] = to_ndarray(obs['to_play'])
                    timestep_dict[env_id] = to_ndarray(obs.get('timestep', -1))
                    if self.policy_config.use_ture_chance_label_in_chance_encoder:
                        chance_dict[env_id] = to_ndarray(obs['chance'])

                    dones[env_id] = done if not self.policy_config.ignore_done else False
                    
                    # Update logging and priority data.
                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict[env_id]
                        if self.policy_config.gumbel_algo:
                            completed_value_lst[env_id] += np.mean(np.array(completed_value_dict[env_id]))
                    
                    eps_steps_lst[env_id] += 1
                    if self.policy_config.use_priority:
                        pred_values_lst[env_id].append(pred_value_dict[env_id])
                        search_values_lst[env_id].append(value_dict[env_id])

                    # Update the observation window with the new observation.
                    observation_window_stack[env_id].append(to_ndarray(obs['observation']))

                    # ==============================================================
                    # Game Segment Saving Logic
                    # ==============================================================
                    # If a segment is full, pad and save the previous segment.
                    if game_segments[env_id].is_full():
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(env_id, last_game_segments, last_game_priorities, game_segments, dones)

                        # Calculate priorities for the now-completed `last_game_segment`.
                        priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id], search_values_lst[env_id] = [], []

                        # The current segment becomes the `last_game_segment`.
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities

                        # Start a new game segment.
                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        game_segments[env_id].reset(observation_window_stack[env_id])

                    self._env_info[env_id]['step'] += 1
                    collected_step += 1

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration
                
                # --- Episode Termination Handling ---
                if done:
                    collected_episode += 1
                    reward = info['eval_episode_return']
                    log_info = {'reward': reward, 'time': self._env_info[env_id]['time'], 'step': self._env_info[env_id]['step']}
                    if not collect_with_pure_policy:
                        log_info['visit_entropy'] = visit_entropies_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                        if self.policy_config.gumbel_algo:
                            log_info['completed_value'] = completed_value_lst[env_id] / eps_steps_lst[env_id] if eps_steps_lst[env_id] > 0 else 0
                    self._episode_info.append(log_info)

                    # Pad and save the segment before the final one.
                    if last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(env_id, last_game_segments, last_game_priorities, game_segments, dones)
                    
                    # Process and save the final segment of the episode.
                    priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                    game_segments[env_id].game_segment_to_array()
                    if len(game_segments[env_id].reward_segment) > 0:
                        self.game_segment_pool.append((game_segments[env_id], priorities, dones[env_id]))

                    # Reset environment-specific states for a new episode.
                    if n_episode > self._env_num:
                        # Re-initialize the state for this env_id.
                        init_obs = self._env.ready_obs
                        while env_id not in init_obs:
                            self._logger.warning(f"Waiting for env {env_id} to reset...")
                            time.sleep(retry_waiting_time)
                            init_obs = self._env.ready_obs
                        
                        action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                        to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                        timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))
                        if self.policy_config.use_ture_chance_label_in_chance_encoder:
                           chance_dict[env_id] = to_ndarray(init_obs[env_id]['chance'])

                        # Reset game segment and observation stack.
                        game_segments[env_id] = GameSegment(self._env.action_space, game_segment_length=self.policy_config.game_segment_length, config=self.policy_config)
                        observation_window_stack[env_id].clear()
                        for _ in range(self.policy_config.model.frame_stack_num):
                            observation_window_stack[env_id].append(init_obs[env_id]['observation'])
                        game_segments[env_id].reset(observation_window_stack[env_id])
                        last_game_segments[env_id] = None
                        last_game_priorities[env_id] = None

                    # Reset tracking and logging variables.
                    pred_values_lst[env_id], search_values_lst[env_id] = [], []
                    eps_steps_lst[env_id], visit_entropies_lst[env_id] = 0, 0
                    if self.policy_config.gumbel_algo:
                        completed_value_lst[env_id] = 0

                    # Reset policy and collector stats for the finished environment.
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)

            # --- Check for Collection Completion ---
            if collected_episode >= n_episode:
                # Prepare data for returning.
                return_data = [
                    [item[0] for item in self.game_segment_pool],
                    [{
                        'priorities': item[1],
                        'done': item[2],
                        'unroll_plus_td_steps': self.unroll_plus_td_steps
                    } for item in self.game_segment_pool]
                ]
                self.game_segment_pool.clear()
                break
        
        # --- Finalize and Log ---
        collected_duration = sum([d['time'] for d in self._episode_info])

        # NOTE: Only for usual DDP not for unizero_multitask pipeline.
        # In DDP, aggregate statistics across all processes.
        # if self._world_size > 1:
        #     collected_step = allreduce_data(collected_step, 'sum')
        #     collected_episode = allreduce_data(collected_episode, 'sum')
        #     collected_duration = allreduce_data(collected_duration, 'sum')

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        self._output_log(train_iter)
        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Aggregates and logs collection statistics to the console, TensorBoard, and WandB.
            This method is only executed by the rank 0 process in a distributed setup.
        Arguments:
            - train_iter (:obj:`int`): The current training iteration number, used as the logging step.
        """
        if self._rank != 0:
            return
        
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration if duration > 0 else 0,
                'avg_episode_per_sec': episode_count / duration if duration > 0 else 0,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
            }
            
            if not self.collect_with_pure_policy:
                visit_entropy = [d['visit_entropy'] for d in self._episode_info]
                info['visit_entropy_mean'] = np.mean(visit_entropy)
            if self.policy_config.gumbel_algo:
                completed_value = [d['completed_value'] for d in self._episode_info]
                info['completed_value_mean'] = np.mean(completed_value)

            self._episode_info.clear()
            
            # Log to console
            self._logger.info("Collector Training Summary:\n{}".format('\n'.join([f'  {k}: {v}' for k, v in info.items()])))
            
            # Log to TensorBoard and WandB
            for k, v in info.items():
                if self.task_id is None:
                    tb_prefix_iter = f'{self._instance_name}_iter/'
                    tb_prefix_step = f'{self._instance_name}_step/'
                else:
                    tb_prefix_iter = f'{self._instance_name}_iter_task{self.task_id}/'
                    tb_prefix_step = f'{self._instance_name}_step_task{self.task_id}/'
                
                self._tb_logger.add_scalar(tb_prefix_iter + k, v, train_iter)
                self._tb_logger.add_scalar(tb_prefix_step + k, v, self._total_envstep_count)
            
            if self.policy_config.use_wandb:
                wandb_log_data = {tb_prefix_step + k: v for k, v in info.items()}
                wandb.log(wandb_log_data, step=self._total_envstep_count)
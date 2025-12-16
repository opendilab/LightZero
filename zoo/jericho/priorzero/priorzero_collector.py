import asyncio
import logging
import sys
import time
import cProfile
from contextlib import contextmanager
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY, allreduce_data
from vllm import SamplingParams
import os

# Import from local LightZero
from lzero.worker.muzero_segment_collector import MuZeroSegmentCollector as OriginalCollector
from lzero.mcts.utils import prepare_observation
from game_segment_priorzero import GameSegment
from priorzero_policy import build_llm_prompt

# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_raw_obs_text(obs_dict: Dict[str, Any]) -> str:
    """
    Extract text observation from environment observation dictionary.

    Args:
        obs_dict: Observation dictionary from environment

    Returns:
        text_obs: Text observation string
    """
    # [PRIORZERO-FIX] Try to get 'raw_obs_text' field first (Jericho env adds this)
    if 'raw_obs_text' in obs_dict:
        return str(obs_dict['raw_obs_text'])

    # Try to get 'raw_obs' field (alternative naming)
    if 'raw_obs' in obs_dict:
        return str(obs_dict['raw_obs'])

    # Try to get 'text' field
    if 'text' in obs_dict:
        return str(obs_dict['text'])

    # Try to get 'observation_str' field (Jericho env provides this in save_replay mode)
    if 'observation_str' in obs_dict:
        return str(obs_dict['observation_str'])

    # Try to get 'observation' and check if it's text
    if 'observation' in obs_dict:
        obs = obs_dict['observation']
        if isinstance(obs, str):
            return obs
        elif isinstance(obs, (list, np.ndarray)):
            # If observation is already processed (e.g., embeddings), cannot extract text
            # Return a placeholder
            return f"[Observation vector of shape {np.array(obs).shape}]"

    # Fallback: return str representation
    return str(obs_dict)


# ==============================================================================
# PriorZero Collector Class
# ==============================================================================

@SERIAL_COLLECTOR_REGISTRY.register('priorzero_segment', force_overwrite=True)
class PriorZeroCollector(OriginalCollector):
    """
    [PRIORZERO-MODIFIED]

    Features:
    - History buffer for each environment (sliding window)
    - Robust error handling with retries
    - Detailed logging of LLM prior statistics
    """

    def __init__(
        self,
        llm_prior_generator,
        policy_config: Dict,
        **kwargs
    ):
        """
        Initialize PriorZeroCollector.

        Args:
            vllm_engine
            policy_config: Policy configuration (contains llm_policy_cfg)
            **kwargs: Additional arguments for parent class
        """
        # [FIX] Set policy_config in kwargs before calling super().__init__
        # because parent class needs it
        kwargs['policy_config'] = policy_config

        super().__init__(**kwargs)

        self.llm_prior_generator = llm_prior_generator
        self.llm_policy_cfg = policy_config.llm_policy_cfg

        # [PRIORZERO-NEW] History buffer for each environment
        # Format: {env_id: deque([(obs_text, action_text, reward), ...])}
        self.history_buffers = defaultdict(
            lambda: deque(maxlen=self.llm_policy_cfg.history_length)
        )
        self.prompt_log_interval = getattr(self.llm_policy_cfg, 'prompt_log_interval', 0)

        self.profile_cfg = getattr(self.policy_config, 'profile_cfg', {})
        self._profile_enabled = bool(self.profile_cfg.get('enable_cprofile', False))
        self._profile_log_interval = int(self.profile_cfg.get('log_interval', 50))
        self._profile_dir = f"./{self._exp_name}/log/profile"
        self._profile_stats = { 'collect_get_llm_prior_profile': {'count': 0, 'total': 0.0, 'max': 0.0}, 
                                'collect_step_profile': {'count': 0, 'total': 0.0, 'max': 0.0},
                                'collect_forward_profile': {'count': 0, 'total': 0.0, 'max': 0.0}
                            }
        self._profile_stats_file = f'{self._profile_dir}/collector_time.log'
        if self._profile_enabled:
            os.makedirs(self._profile_dir, exist_ok=True)

        # Where to persist sampled LLM outputs during collect
        self._llm_output_log_path = f"./{self._exp_name}/log/collector/llm_output.log"
        self._llm_call_count = 0
        self._llm_prior_req_counter = 0

        self._logger.info("✓ PriorZeroCollector initialized with vLLM engine")
        self._logger.info(f"  - History length: {self.llm_policy_cfg.history_length}")
        self._logger.info(f"  - Generate max length: {self.llm_policy_cfg.generate_max_len}")

    def pad_and_save_last_trajectory(
            self, i: int, last_game_segments: List[GameSegment], last_game_priorities: List[np.ndarray],
            game_segments: List[GameSegment], done: np.ndarray
    ) -> None:
        beg_index = self.policy_config.model.frame_stack_num
        end_index = beg_index + self.policy_config.num_unroll_steps + self.policy_config.td_steps
        
        pad_obs_lst = game_segments[i].obs_segment[beg_index:end_index]
        pad_raw_obs_lst = game_segments[i].raw_obs_segment[beg_index:end_index]
        pad_history_obs_lst = game_segments[i].history_obs_segment[beg_index:end_index]
        pad_action_logprob_lst = game_segments[i].action_logprob_segment[beg_index:end_index]

        # NOTE: Specific padding logic for UniZero.
        pad_action_lst = game_segments[i].action_segment[:self.policy_config.num_unroll_steps + self.policy_config.td_steps]
        pad_child_visits_lst = game_segments[i].child_visit_segment[:self.policy_config.num_unroll_steps + self.policy_config.td_steps]

        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps - 1
        pad_reward_lst = game_segments[i].reward_segment[beg_index:end_index]

        if self.policy_config.use_ture_chance_label_in_chance_encoder:
            chance_lst = game_segments[i].chance_segment[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.unroll_plus_td_steps
        pad_root_values_lst = game_segments[i].root_value_segment[beg_index:end_index]

        if self.policy_config.gumbel_algo:
            pad_improved_policy_prob = game_segments[i].improved_policy_probs[beg_index:end_index]

        # Pad and finalize the last game segment.
        if self.policy_config.gumbel_algo:
            last_game_segments[i].pad_over(
                pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                next_segment_improved_policy=pad_improved_policy_prob
            )
        else:
            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                    next_chances=chance_lst, next_segment_raw_obs=pad_raw_obs_lst,
                    next_segment_history_obs=pad_history_obs_lst, next_segment_action_logprob=pad_action_logprob_lst
                )
            else:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst, 
                    next_segment_raw_obs=pad_raw_obs_lst, next_segment_history_obs=pad_history_obs_lst,
                    next_segment_action_logprob=pad_action_logprob_lst
                )

        last_game_segments[i].game_segment_to_array()

        # Add the completed game segment to the pool.
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # Reset placeholders for the next collection cycle.
        last_game_segments[i] = None
        last_game_priorities[i] = None
    
    def _get_llm_prior(
        self,
        states: List[str],
        valid_actions_list: List[List[str]], 
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
    ) -> List[Any]:
        """
        [PRIORZERO-SEQUENCE-SCORING]
        Ensures every action has a logprob by retrying and falling back if needed.
        """

        assert self.llm_prior_generator is not None, "llm_prior_generator is None."
        all_prompts = []
        all_labels = []
        for i, actions in enumerate(valid_actions_list):  
            actions.append('go')   # 确保环境使用的动作都在valid actions里有对应的logprob
            state = states[i]
            history = histories[i]
            prompt = build_llm_prompt(current_obs=state, history=history, use_cot=self.llm_policy_cfg.use_cot)
            for action in actions:
                all_prompts.append(prompt)
                all_labels.append(action)
                
        all_prior_scores = self.llm_prior_generator._generate_vllm(all_prompts, all_labels, reduction='mean')
        llm_prior, idx = [], 0
        for env_id in range(len(states)):
            tmp_dict = {}
            for action in valid_actions_list[env_id]:
                tmp_dict[action] = all_prior_scores[idx]
                idx = idx + 1
            llm_prior.append(tmp_dict)
        return llm_prior

    @contextmanager
    def _profile_block(self, name: str):
        if not self._profile_enabled:
            yield None
            return
        profiler = cProfile.Profile()
        start_time = time.perf_counter()
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - start_time
            self._record_profile_time(name, elapsed)

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        log_every = max(1, self._profile_log_interval)
        self._profile_stats[name]['count'] += 1
        self._profile_stats[name]['total'] += elapsed
        self._profile_stats[name]['max'] = max(self._profile_stats[name]['max'], elapsed)
        if self._profile_stats[name]['count'] % log_every == 0:
            avg = self._profile_stats[name]['total'] / self._profile_stats[name]['count']
            with open(self._profile_stats_file, mode='a', encoding='utf-8') as f:
                f.write(
                    f"{time.time():.3f}\tname={name}\tcount={self._profile_stats[name]['count']}\t"
                    f"total_s={self._profile_stats[name]['total']:.4f}\tavg_s={avg:.4f}\tmax_s={self._profile_stats[name]['max']:.4f}\n"
                )
                
    def collect(
        self,
        num_segments: Optional[int] = None,
        train_iter: int = 0,
        policy_kwargs: Optional[dict] = None,
        collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        [PRIORZERO-MODIFIED]
        Collect game segments with LLM-guided MCTS.

        Main changes from parent:
        1. Extract text observations from environment
        2. Pass LLM priors to policy forward pass
        3. Update history buffers after each step

        Args:
            num_segments: Number of segments to collect
            train_iter: Current training iteration
            policy_kwargs: Additional kwargs for policy
            collect_with_pure_policy: Whether to use pure policy without MCTS

        Returns:
            return_data: List containing [game_segments, metadata]
        """
        if num_segments is None:
            if self._default_num_segments is None:
                raise RuntimeError("Please specify num_segments for collection.")
            else:
                num_segments = self._default_num_segments

        assert num_segments == self._env_num, \
            f"num_segments({num_segments}) must equal env_num({self._env_num})"

        if policy_kwargs is None:
            policy_kwargs = {}

        temperature = policy_kwargs.get('temperature', 1.0)
        epsilon = policy_kwargs.get('epsilon', 0.0)

        collected_episode = 0
        collected_step = 0
        llm_prior_entropy = [[] for _ in range(self._env_num)]
        env_nums = self._env_num
        init_obs = self._env.ready_obs

        retry_waiting_time = 0.05
        while len(init_obs.keys()) != env_nums:
            self._logger.info(f'Waiting for all environments to reset. Ready: {list(init_obs.keys())}')
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        for env_id in range(env_nums):
            if env_id in init_obs:
                self.action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                self.to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                self.timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))

        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]
        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self.policy_config.game_segment_length,
                config=self.policy_config,
                task_id=self.task_id
            ) for _ in range(env_nums)
        ]

        observation_window_stack = [
            deque(maxlen=self.policy_config.model.frame_stack_num)
            for _ in range(env_nums)
        ]
        for env_id in range(env_nums):
            initial_frames = [
                to_ndarray(init_obs[env_id]['observation'])
                for _ in range(self.policy_config.model.frame_stack_num)
            ]
            observation_window_stack[env_id].extend(initial_frames)
            game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(init_obs[env_id]), 
                                        init_history_obs=list(self.history_buffers[env_id]), init_action_logprob=None)

        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)

        if collect_with_pure_policy:
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]

        # ==================================================================
        # Main Collection Loop
        # ==================================================================
        while True:
            with self._timer:
                obs = self._env.ready_obs
                ready_env_id = set(obs.keys())

                if len(ready_env_id) < self._env_num:
                    self._logger.debug(f'Only {len(ready_env_id)}/{self._env_num} envs ready')

                stack_obs_dict = {
                    env_id: game_segments[env_id].get_obs()
                    for env_id in ready_env_id
                }
                stack_obs_list = [stack_obs_dict[env_id] for env_id in sorted(list(ready_env_id))]

                action_mask = [self.action_mask_dict[env_id] for env_id in sorted(list(ready_env_id))]
                to_play = [self.to_play_dict[env_id] for env_id in sorted(list(ready_env_id))]
                timestep = [self.timestep_dict[env_id] for env_id in sorted(list(ready_env_id))]

                # Convert to tensors
                stack_obs_array = to_ndarray(stack_obs_list)
                stack_obs_tensor = prepare_observation(
                    stack_obs_array,
                    self.policy_config.model.model_type
                )
                stack_obs_tensor = torch.from_numpy(stack_obs_tensor).to(self.policy_config.device)

                # ==============================================================
                # [PRIORZERO-NEW] Get LLM Priors
                # ==============================================================
                if collect_with_pure_policy:
                    continue
                else:
                    # Extract text observations and valid actions
                    raw_obs_list = []
                    histories_list = []
                    valid_actions_list = [] 
                    for env_id in sorted(list(ready_env_id)):
                        raw_obs_text = extract_raw_obs_text(obs[env_id])
                        raw_obs_list.append(raw_obs_text)

                        history = list(self.history_buffers[env_id])
                        histories_list.append(history)

                        valid_actions = obs[env_id].get('valid_actions', [])
                        valid_actions_list.append(valid_actions)

                    if self.policy_config.llm_policy_cfg.enable_llm:
                        with self._profile_block(name='collect_get_llm_prior_profile'):
                            llm_prior_logprob = self._get_llm_prior(
                                states=raw_obs_list,
                                valid_actions_list=valid_actions_list,  # [PRIORZERO] Pass valid actions
                                histories=histories_list
                            )
                    else:
                        llm_prior_logprob = [None for i in range(len(valid_actions_list))]

                policy_kwargs_forward = {
                    'llm_prior_logprob': llm_prior_logprob,
                    'valid_actions_list': valid_actions_list,
                }

                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id
                with self._profile_block(name='collect_forward_profile'):
                    policy_output = self._policy.forward(data=stack_obs_tensor, action_mask=action_mask,
                                                        temperature=temperature, to_play=to_play, epsilon=epsilon,
                                                        ready_env_id=sorted(list(ready_env_id)), timestep=timestep,
                                                        **policy_kwargs_forward)

                # Extract outputs
                actions_with_env_id = {k: v['action'] for k, v in policy_output.items()}
                value_dict_with_env_id = {k: v['searched_value'] for k, v in policy_output.items()}
                pred_value_dict_with_env_id = {k: v['predicted_value'] for k, v in policy_output.items()}

                if not collect_with_pure_policy:
                    distributions_dict_with_env_id = {
                        k: v['visit_count_distributions'] for k, v in policy_output.items()
                    }
                    visit_entropy_dict_with_env_id = {
                        k: v['visit_count_distribution_entropy'] for k, v in policy_output.items()
                    }

                actions: Dict[int, Any] = {
                    env_id: actions_with_env_id.pop(env_id)
                    for env_id in ready_env_id
                }

                # ==============================================================
                # Step Environments
                # ==============================================================
                with self._profile_block(name='collect_step_profile'):
                    timesteps = self._env.step(actions)

            interaction_duration = self._timer.value / len(timesteps)

            # ==================================================================
            # Process Environment Responses
            # ==================================================================
            for env_id, episode_timestep in timesteps.items():
                with self._timer:
                    # Handle abnormal timesteps
                    if episode_timestep.info.get('abnormal', False):
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])
                        self._reset_stat(env_id)
                        self._logger.info(f'⚠ Env {env_id} had abnormal step: {episode_timestep.info}')
                        continue

                    obs_new, reward, done, info = (
                        episode_timestep.obs,
                        episode_timestep.reward,
                        episode_timestep.done,
                        episode_timestep.info
                    )
                    game_segments[env_id].store_search_stats(
                        distributions_dict_with_env_id[env_id],
                        value_dict_with_env_id[env_id])
                    # ===========================================================
                    # [PRIORZERO-NEW] Update History Buffer
                    # ===========================================================
                    raw_obs_text = extract_raw_obs_text(obs[env_id])
                    if env_id < len(valid_actions_list) and actions[env_id] < len(valid_actions_list[env_id]):  
                        action = valid_actions_list[env_id][actions[env_id]]
                    else:
                        action = info.get('action_str', "go")
                        
                    self.history_buffers[env_id].append((raw_obs_text, action, float(reward)))
                    
                    # Append transition to game segment
                    game_segments[env_id].append(
                        actions[env_id],
                        to_ndarray(obs_new['observation']),
                        reward,
                        self.action_mask_dict[env_id],
                        self.to_play_dict[env_id],
                        timestep=to_ndarray(obs_new.get('timestep', -1)),
                        raw_obs_text=extract_raw_obs_text(obs_new),
                        history_obs=list(self.history_buffers[env_id]),
                        action_logprob=llm_prior_logprob[env_id] # 是一个字典对 {'open': -151; "down": -231}
                    )

                    # Update state
                    self.action_mask_dict[env_id] = to_ndarray(obs_new['action_mask'])
                    self.to_play_dict[env_id] = to_ndarray(obs_new['to_play'])
                    self.timestep_dict[env_id] = to_ndarray(obs_new.get('timestep', -1))
                    self.dones[env_id] = False if self.policy_config.ignore_done else done

                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict_with_env_id[env_id]

                    eps_steps_lst[env_id] += 1

                    # Reset policy if needed (for UniZero)
                    if self._policy.get_attribute('cfg').type in ['unizero', 'sampled_unizero', 'priorzero']:
                        self._policy.reset(
                            env_id=env_id,
                            current_steps=eps_steps_lst[env_id],
                            reset_init_data=False
                        )

                    # Store values for priority calculation
                    if self.policy_config.use_priority:
                        pred_values_lst[env_id].append(pred_value_dict_with_env_id[env_id])
                        search_values_lst[env_id].append(value_dict_with_env_id[env_id])

                    # Update observation window
                    observation_window_stack[env_id].append(to_ndarray(obs_new['observation']))

                    # ===========================================================
                    # Save Full Game Segment
                    # ===========================================================
                    if game_segments[env_id].is_full():
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(env_id, last_game_segments, last_game_priorities,
                                                               game_segments, self.dones)

                        # Calculate priorities
                        priorities = self._compute_priorities(env_id, pred_values_lst, search_values_lst)
                        pred_values_lst[env_id], search_values_lst[env_id] = [], []

                        # Save segment
                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = priorities

                        # Create new segment
                        game_segments[env_id] = GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config,
                            task_id=self.task_id
                        )
                        game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(obs_new), init_history_obs=list(self.history_buffers[env_id]), init_action_logprob=None)

                    self._env_info[env_id]['step'] += 1
                    if llm_prior_logprob[env_id] is not None:
                        llm_prior_tensor = torch.tensor([logit for k, logit in llm_prior_logprob[env_id].items()]) 
                        llm_prior_prob = torch.softmax(llm_prior_tensor, dim=-1)
                        llm_prior_entropy[env_id].append(-torch.sum(llm_prior_prob * torch.log(llm_prior_prob + 1e-9), dim=-1))
                    else:
                        llm_prior_entropy[env_id].append(0.0)
                    collected_step += 1

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # ==============================================================
                # Episode Done
                # ==============================================================
                if episode_timestep.done:
                    self._logger.info(f'======== Env {env_id} episode finished! ========')
                    self._total_episode_count += 1
                    # Logging
                    info_log = {
                        'reward': episode_timestep.info['eval_episode_return'],
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                        'llm_prior_entropy': sum(llm_prior_entropy[env_id])/len(llm_prior_entropy[env_id])}
                    if not collect_with_pure_policy:
                        info_log['visit_entropy'] = (
                            visit_entropies_lst[env_id] / eps_steps_lst[env_id]
                            if eps_steps_lst[env_id] > 0 else 0
                        )

                    collected_episode += 1
                    self._episode_info.append(info_log)
                    # Save remaining segments
                    if last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory( env_id, last_game_segments, last_game_priorities, game_segments, self.dones)

                    priorities = self._compute_priorities( env_id, pred_values_lst, search_values_lst)
                    game_segments[env_id].game_segment_to_array()
                    if len(game_segments[env_id].reward_segment) > 0:
                        self.game_segment_pool.append((
                            game_segments[env_id],
                            priorities,
                            self.dones[env_id]
                        ))
                    # Reset
                    pred_values_lst[env_id], search_values_lst[env_id] = [], []
                    eps_steps_lst[env_id], visit_entropies_lst[env_id] = 0, 0

                    self._policy.reset([env_id], task_id=self.task_id)
                    self._reset_stat(env_id)

                    # Clear history buffer for this environment
                    self.history_buffers[env_id].clear()
                    # Re-initialize game segment
                    init_obs = self._env.ready_obs
                    observation_window_stack[env_id] = deque(
                            [init_obs[env_id]['observation'] for _ in range(self.policy_config.model.frame_stack_num)],
                            maxlen=self.policy_config.model.frame_stack_num
                        )
                    
                    game_segments[env_id] = GameSegment(
                        self._env.action_space,
                        game_segment_length=self.policy_config.game_segment_length,
                        config=self.policy_config,
                        task_id=self.task_id
                    )
                    game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(init_obs[env_id]), init_history_obs=list(self.history_buffers[env_id]), init_action_logprob=None)
                    last_game_segments[env_id] = None
                    last_game_priorities[env_id] = None

            # ==================================================================
            # Check if Enough Segments Collected
            # ==================================================================
            if len(self.game_segment_pool) >= self._default_num_segments:
                self._logger.info(
                    f'✓ Collected {len(self.game_segment_pool)} segments '
                    f'(target: {self._default_num_segments})'
                )

                # Format return data
                return_data = [
                    [self.game_segment_pool[i][0] for i in range(len(self.game_segment_pool))],
                    [
                        {
                            'priorities': self.game_segment_pool[i][1],
                            'done': self.game_segment_pool[i][2],
                            'unroll_plus_td_steps': self.unroll_plus_td_steps
                        }
                        for i in range(len(self.game_segment_pool))
                    ]
                ]
                self.game_segment_pool.clear()
                break

        # ==================================================================
        # Final Logging
        # ==================================================================
        collected_duration = sum([d['time'] for d in self._episode_info])

        if self._world_size > 1:
            # Before allreduce
            self._logger.info(f"Rank {self._rank} before allreduce: collected_step={collected_step}, collected_episode={collected_episode}")
            collected_step = allreduce_data(collected_step, 'sum')
            collected_episode = allreduce_data(collected_episode, 'sum')
            collected_duration = allreduce_data(collected_duration, 'sum')
            # After allreduce
            self._logger.info(f"Rank {self._rank} after allreduce: collected_step={collected_step}, collected_episode={collected_episode}")

        
        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        self._output_log(train_iter)

        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        [INHERITED]
        Log collection statistics (inherited from parent).
        """
        if self._rank != 0:
            return
        
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            episode_llm_prior_entropy = [d['llm_prior_entropy'] for d in self._episode_info]
            
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
                'llm_prior_entropy_mean': np.mean(episode_llm_prior_entropy),
                'llm_prior_entropy_max': np.max(episode_llm_prior_entropy),
                'llm_prior_entropy_min': np.min(episode_llm_prior_entropy)
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
            
    

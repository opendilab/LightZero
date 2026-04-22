"""
Unified PriorZero Collector supporting both LLM and VL priors

This collector uses a unified prior_generator interface to support:
- Text input with LLM prior (Jericho games)
- Image input with VL prior (Atari games)
"""
import asyncio
import logging
import sys
import time

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


# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_raw_obs_text(obs_dict: Dict[str, Any]) -> str:
    """Extract text observation from environment observation dictionary."""
    if 'raw_obs_text' in obs_dict:
        return str(obs_dict['raw_obs_text'])
    if 'raw_obs' in obs_dict:
        return str(obs_dict['raw_obs'])
    if 'text' in obs_dict:
        return str(obs_dict['text'])
    if 'observation_str' in obs_dict:
        return str(obs_dict['observation_str'])
    if 'observation' in obs_dict:
        obs = obs_dict['observation']
        if isinstance(obs, str):
            return obs
        elif isinstance(obs, (list, np.ndarray)):
            return f"[Observation vector of shape {np.array(obs).shape}]"
    return str(obs_dict)


def extract_raw_obs_image(obs_dict: Dict[str, Any]) -> np.ndarray:
    """Extract image observation from environment observation dictionary."""
    if 'observation' in obs_dict:
        obs = obs_dict['observation']
        if isinstance(obs, np.ndarray):
            # Assume image format (H, W, C) or (C, H, W)
            return obs
    raise ValueError(f"Cannot extract image from observation: {obs_dict.keys()}")


# ==============================================================================
# Unified PriorZero Collector Class
# ==============================================================================

@SERIAL_COLLECTOR_REGISTRY.register('priorzero_segment', force_overwrite=True)
class PriorZeroCollector(OriginalCollector):
    """
    Unified PriorZero Collector supporting both LLM and VL priors.

    Features:
    - Unified prior_generator interface (supports LLM and VL)
    - History buffer for each environment
    - Automatic detection of observation type (text vs image)
    - Backward compatible with existing LLM-based implementation
    """

    def __init__(
        self,
        policy_config: Dict,
        llm_config: Dict,  # Can be LLM or VL config
        data_processor=None,  # Backward compatibility
        prior_generator=None,  # NEW: Unified prior generator
        prof=None,
        obs_type: str = 'text',  # NEW: 'text' or 'image'
        env_id: str = None,  # NEW: Environment ID for action mapping
        **kwargs
    ):
        """
        Initialize Unified PriorZeroCollector.

        Args:
            policy_config: Policy configuration
            llm_config: LLM/VL configuration
            data_processor: DataProcessor (for backward compatibility)
            prior_generator: Unified PriorGenerator instance (NEW)
            prof: Profiler
            obs_type: Observation type ('text' or 'image')
            env_id: Environment ID (e.g., 'PongNoFrameskip-v4')
            **kwargs: Additional arguments for parent class
        """
        kwargs['policy_config'] = policy_config

        super().__init__(**kwargs)

        self.data_processor = data_processor
        self.prior_generator = prior_generator  # NEW: Unified interface
        self.prof = prof
        self.llm_cfg = llm_config
        self.obs_type = obs_type  # NEW: Track observation type
        self.env_id = env_id or 'PongNoFrameskip-v4'  # NEW: Store env_id

        # History buffers
        history_length = getattr(llm_config, 'history_length', 5)
        self.history_buffers = defaultdict(lambda: deque(maxlen=history_length))
        self.llm_prior_temperature = getattr(llm_config, 'llm_prior_temperature', 1.0)

        # Logging
        prior_type = "VL" if obs_type == 'image' else "LLM"
        self._logger.info(f"✓ PriorZeroCollector initialized with {prior_type} prior")
        self._logger.info(f"  - Observation type: {obs_type}")
        if obs_type == 'image':
            self._logger.info(f"  - Environment: {self.env_id}")
        self._logger.info(f"  - History length: {history_length}")
        self._logger.info(f"  - Prior generator: {type(prior_generator).__name__ if prior_generator else 'None'}")

        # First-call validation flag
        self._first_collect_logged = False

    def _get_prior_from_generator(
        self,
        observations: List[Any],
        valid_actions_list: List[List[str]],
        histories_list: List[List],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        """
        Get action priors using the unified prior_generator interface.

        Args:
            observations: List of observations (text strings or image arrays)
            valid_actions_list: List of valid action lists
            histories_list: List of history buffers

        Returns:
            Tuple of (prior_per_seq, prior_per_tok, cot_prefixes)
        """
        if self.prior_generator is None:
            # Fallback: uniform prior
            num_envs = len(observations)
            prior_per_seq = []
            for actions in valid_actions_list:
                uniform_prior = np.ones(len(actions)) / len(actions)
                prior_per_seq.append(uniform_prior)
            prior_per_tok = [None] * num_envs
            cot_prefixes = [None] * num_envs
            return prior_per_seq, prior_per_tok, cot_prefixes

        # Use unified prior generator
        prior_results = self.prior_generator.batch_generate_prior(
            observations=observations,
            action_candidates_list=valid_actions_list,
            histories=histories_list,
            temperature=self.llm_prior_temperature,
        )

        # Extract results
        prior_per_seq = [result['action_probs'] for result in prior_results]
        # VL path: action_logits is np.ndarray of shape (num_actions,) with per-action log-probs.
        # The VL datafactory (_make_vl_train_samples) expects this numpy format and spreads
        # the chosen action's logprob uniformly across target tokens for PPO.
        # NOTE: Do NOT convert to dict here — the game_buffer's _is_llm_text_mode guard
        # uses isinstance(..., dict) to distinguish LLM text mode from VL image mode.
        prior_per_tok = [result.get('action_logits', None) for result in prior_results]
        cot_prefixes = [result.get('raw_output', None) for result in prior_results]

        return prior_per_seq, prior_per_tok, cot_prefixes

    def _get_prior_legacy(
        self,
        raw_obs_list: List[str],
        valid_actions_list: List[List[str]],
        histories_list: List[List],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        """
        Legacy method using data_processor (for backward compatibility).

        This is the original implementation for LLM-based priors.
        """
        if self.data_processor is None:
            raise ValueError("data_processor is None. Cannot use legacy prior generation.")

        llm_prior_per_seq, llm_prior_per_tok, cot_prefixes = self.data_processor.get_llm_prior(
            states=raw_obs_list,
            valid_actions_list=valid_actions_list,
            histories=histories_list,
            return_cot=True
        )

        return llm_prior_per_seq, llm_prior_per_tok, cot_prefixes

    def collect(
        self,
        num_segments: Optional[int] = None,
        train_iter: int = 0,
        policy_kwargs: Optional[dict] = None,
        collect_with_pure_policy: bool = False
    ) -> List[Any]:
        """
        Collect game segments with prior-guided MCTS.

        Supports both LLM (text) and VL (image) priors through unified interface.

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

            # Extract initial raw observation (text or image)
            if self.obs_type == 'text':
                init_raw_obs = extract_raw_obs_text(init_obs[env_id])
            else:
                init_raw_obs = extract_raw_obs_image(init_obs[env_id])

            game_segments[env_id].reset(
                observation_window_stack[env_id],
                init_raw_obs=init_raw_obs,
                init_history_obs=list(self.history_buffers[env_id])
            )

        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)

        if collect_with_pure_policy:
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]

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
                stack_obs_tensor = torch.from_numpy(stack_obs_tensor).to(self.policy_config.device).float()

                if collect_with_pure_policy:
                    continue
                else:
                    # ===========================================================
                    # [UNIFIED] Extract observations and get priors
                    # ===========================================================
                    observations_list = []
                    histories_list = []
                    valid_actions_list = []

                    for env_id in sorted(list(ready_env_id)):
                        # Extract observation based on type
                        if self.obs_type == 'text':
                            raw_obs = extract_raw_obs_text(obs[env_id])
                        else:  # image
                            raw_obs = extract_raw_obs_image(obs[env_id])

                        observations_list.append(raw_obs)
                        histories_list.append(list(self.history_buffers[env_id]))

                        # Get valid actions
                        # For text games: use valid_actions from obs
                        # For Atari: convert integer indices to semantic action names
                        valid_actions = obs[env_id].get('valid_actions', [])
                        if len(valid_actions) == 0 and self.obs_type == 'image':
                            # Atari: convert integer action indices to semantic names
                            from zoo.jericho.priorzero.atari_action_meanings import get_action_meanings
                            action_space_size = self.policy_config.model.action_space_size
                            action_meanings = get_action_meanings(self.env_id, action_space_size)
                            # Use semantic names instead of integers
                            valid_actions = [action_meanings[i] for i in range(action_space_size)]
                        valid_actions_list.append(valid_actions)

                    # First-call validation logging for image data flow
                    if not self._first_collect_logged and self.obs_type == 'image' and len(observations_list) > 0:
                        self._first_collect_logged = True
                        obs_sample = observations_list[0]
                        if isinstance(obs_sample, np.ndarray):
                            self._logger.info(
                                f"[Collector Validation] === FIRST COLLECT IMAGE CHECK ===\n"
                                f"  Image shape: {obs_sample.shape}, dtype: {obs_sample.dtype}, "
                                f"min: {obs_sample.min()}, max: {obs_sample.max()}\n"
                                f"  Num envs: {len(observations_list)}\n"
                                f"  Actions: {valid_actions_list[0]}\n"
                                f"[Collector Validation] === END CHECK ==="
                            )

                    # Get priors using unified interface
                    with self.prof.block("collect_step_get_prior", rank=self._rank):
                        if self.prior_generator is not None:
                            # NEW: Use unified prior generator
                            llm_prior_per_seq, llm_prior_per_tok, cot_prefixes = self._get_prior_from_generator(
                                observations=observations_list,
                                valid_actions_list=valid_actions_list,
                                histories_list=histories_list,
                            )
                        elif self.data_processor is not None:
                            # LEGACY: Use data_processor (backward compatibility)
                            llm_prior_per_seq, llm_prior_per_tok, cot_prefixes = self._get_prior_legacy(
                                raw_obs_list=observations_list,
                                valid_actions_list=valid_actions_list,
                                histories_list=histories_list,
                            )
                        else:
                            # Fallback: uniform prior
                            llm_prior_per_seq = [
                                np.ones(len(actions)) / len(actions)
                                for actions in valid_actions_list
                            ]
                            llm_prior_per_tok = [None] * len(observations_list)
                            cot_prefixes = [None] * len(observations_list)

                        # Apply temperature scaling
                        for env_id, llm_prior in enumerate(llm_prior_per_seq):
                            scaled_llm_prior = self.apply_temperature_scaling(llm_prior, return_logprobs=True)
                            llm_prior_per_seq[env_id] = scaled_llm_prior

                policy_kwargs_forward = {
                    'llm_prior_logprob': llm_prior_per_seq,
                    'valid_actions_list': valid_actions_list,
                }

                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id

                with self.prof.block("collect_step_forward", rank=self._rank):
                    policy_output = self._policy.forward(
                        data=stack_obs_tensor,
                        action_mask=action_mask,
                        temperature=temperature,
                        to_play=to_play,
                        epsilon=epsilon,
                        ready_env_id=sorted(list(ready_env_id)),
                        timestep=timestep,
                        **policy_kwargs_forward
                    )

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

                with self.prof.block("collect_step", rank=self._rank):
                    timesteps = self._env.step(actions)

            interaction_duration = self._timer.value / len(timesteps)

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
                        value_dict_with_env_id[env_id]
                    )

                    # ===========================================================
                    # [UNIFIED] Update History Buffer
                    # ===========================================================
                    if self.obs_type == 'text':
                        raw_obs = extract_raw_obs_text(obs[env_id])
                    else:
                        raw_obs = extract_raw_obs_image(obs[env_id])

                    # Get action string
                    # For Atari: convert integer action index to semantic name
                    if self.obs_type == 'image':
                        from zoo.jericho.priorzero.atari_action_meanings import action_index_to_name
                        action_space_size = self.policy_config.model.action_space_size
                        action_str = action_index_to_name(self.env_id, actions[env_id], action_space_size)
                    elif env_id < len(valid_actions_list) and actions[env_id] < len(valid_actions_list[env_id]):
                        # Text games: use action name from valid_actions_list
                        action_str = valid_actions_list[env_id][actions[env_id]]
                    else:
                        # Fallback
                        action_str = info.get('action_str', str(actions[env_id]))

                    # Use absolute timestep from environment, not relative episode step counter
                    abs_timestep = int(self.timestep_dict[env_id]) if int(self.timestep_dict[env_id]) >= 0 else int(eps_steps_lst[env_id])
                    self.history_buffers[env_id].append((raw_obs, action_str, float(reward), abs_timestep))

                    # Append transition to game segment
                    game_segments[env_id].append(
                        actions[env_id],
                        to_ndarray(obs_new['observation']),
                        reward,
                        self.action_mask_dict[env_id],
                        self.to_play_dict[env_id],
                        raw_obs_text=raw_obs,
                        history_obs=list(self.history_buffers[env_id]),
                        llm_prior_per_tok=llm_prior_per_tok[env_id] if env_id < len(llm_prior_per_tok) else None,
                        cot_prefix=cot_prefixes[env_id] if env_id < len(cot_prefixes) else None,
                        llm_action=action_str
                    )

                    # Update statistics
                    self.action_mask_dict[env_id] = to_ndarray(obs_new['action_mask'])
                    self.to_play_dict[env_id] = to_ndarray(obs_new['to_play'])
                    self.timestep_dict[env_id] = to_ndarray(obs_new.get('timestep', -1))

                    observation_window_stack[env_id].append(to_ndarray(obs_new['observation']))

                    search_values_lst[env_id].append(value_dict_with_env_id[env_id])
                    pred_values_lst[env_id].append(pred_value_dict_with_env_id[env_id])

                    if not collect_with_pure_policy:
                        visit_entropies_lst[env_id] += visit_entropy_dict_with_env_id[env_id]

                    eps_steps_lst[env_id] += 1
                    collected_step += 1

                    # Check if segment is complete
                    if game_segments[env_id].is_full():
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(
                                env_id, last_game_segments, last_game_priorities,
                                game_segments, done
                            )

                        last_game_segments[env_id] = game_segments[env_id]
                        last_game_priorities[env_id] = self._compute_priorities(game_segments[env_id])

                        # Create new segment
                        game_segments[env_id] = GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config,
                            task_id=self.task_id
                        )

                        if self.obs_type == 'text':
                            current_raw_obs = extract_raw_obs_text(obs_new)
                        else:
                            current_raw_obs = extract_raw_obs_image(obs_new)

                        game_segments[env_id].reset(
                            observation_window_stack[env_id],
                            init_raw_obs=current_raw_obs,
                            init_history_obs=list(self.history_buffers[env_id])
                        )

                    # Handle episode end
                    if done:
                        self._env.reset({env_id: None})
                        self._policy.reset([env_id])

                        # Save second-to-last segment (if exists)
                        if last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(
                                env_id, last_game_segments, last_game_priorities,
                                game_segments, done
                            )

                        # Save the final segment of the episode
                        game_segments[env_id].game_segment_to_array()
                        if len(game_segments[env_id].reward_segment) > 0:
                            priorities = self._compute_priorities(game_segments[env_id])
                            self.game_segment_pool.append((game_segments[env_id], priorities, done))

                        # Log episode statistics
                        collected_episode += 1
                        episode_return = info.get('eval_episode_return', info.get('score', reward))
                        self._logger.info(
                            f"Episode {collected_episode} | Env {env_id} | "
                            f"Steps: {eps_steps_lst[env_id]} | "
                            f"Reward: {episode_return:.2f}"
                        )

                        # Populate _episode_info for parent's _output_log() and TB logging
                        ep_info = {
                            'reward': episode_return,
                            'time': interaction_duration * eps_steps_lst[env_id],
                            'step': int(eps_steps_lst[env_id]),
                            'visit_entropy': visit_entropies_lst[env_id] / max(eps_steps_lst[env_id], 1),
                        }
                        self._episode_info.append(ep_info)

                        # TB logging for episode metrics
                        if hasattr(self, '_tb_logger') and self._tb_logger is not None:
                            self._tb_logger.add_scalar('collect/episode_reward', episode_return, self._total_envstep_count + collected_step)
                            self._tb_logger.add_scalar('collect/episode_length', eps_steps_lst[env_id], self._total_envstep_count + collected_step)

                        # Reset for next episode
                        eps_steps_lst[env_id] = 0
                        visit_entropies_lst[env_id] = 0
                        search_values_lst[env_id] = []
                        pred_values_lst[env_id] = []
                        self.history_buffers[env_id].clear()

                        # Re-initialize game segment for next episode
                        init_obs = self._env.ready_obs
                        if env_id in init_obs:
                            game_segments[env_id] = GameSegment(
                                self._env.action_space,
                                game_segment_length=self.policy_config.game_segment_length,
                                config=self.policy_config,
                                task_id=self.task_id
                            )
                            observation_window_stack[env_id] = deque(maxlen=self.policy_config.model.frame_stack_num)
                            initial_frames = [
                                to_ndarray(init_obs[env_id]['observation'])
                                for _ in range(self.policy_config.model.frame_stack_num)
                            ]
                            observation_window_stack[env_id].extend(initial_frames)

                            if self.obs_type == 'text':
                                init_raw_obs = extract_raw_obs_text(init_obs[env_id])
                            else:
                                init_raw_obs = extract_raw_obs_image(init_obs[env_id])

                            game_segments[env_id].reset(
                                observation_window_stack[env_id],
                                init_raw_obs=init_raw_obs,
                                init_history_obs=list(self.history_buffers[env_id])
                            )

                            self.action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                            self.to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])

                            last_game_segments[env_id] = None
                            last_game_priorities[env_id] = None

            # Check if collection is complete
            if collected_episode >= num_segments:
                break

        # Update statistics that the parent class normally maintains
        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode

        # Call parent's _output_log to write standard TB metrics (collector_step/xxx)
        # Only call when tb_logger is available (rank 0); parent _output_log has no None guard.
        if self._tb_logger is not None:
            self._output_log(train_iter)

        # Return collected data in the format expected by push_game_segments:
        # [list_of_game_segments, list_of_meta_dicts]
        return_data = [
            [seg for seg, _, _ in self.game_segment_pool],
            [
                {
                    'priorities': priorities,
                    'done': done,
                    'unroll_plus_td_steps': self.unroll_plus_td_steps,
                }
                for _, priorities, done in self.game_segment_pool
            ]
        ]
        self.game_segment_pool = []

        return return_data

    def pad_and_save_last_trajectory(
        self, i: int, last_game_segments: List[GameSegment], last_game_priorities: List[np.ndarray],
        game_segments: List[GameSegment], done: bool
    ) -> None:
        """Pad and save the last trajectory (same as original)."""
        beg_index = self.policy_config.model.frame_stack_num
        end_index = beg_index + self.policy_config.num_unroll_steps + self.policy_config.td_steps

        pad_obs_lst = game_segments[i].obs_segment[beg_index:end_index]
        pad_raw_obs_lst = game_segments[i].raw_obs_segment[beg_index:end_index]
        pad_history_obs_lst = game_segments[i].history_obs_segment[beg_index:end_index]
        pad_llm_prior_per_tok_lst = game_segments[i].llm_prior_per_tok_segment[beg_index:end_index]
        pad_cot_prefix_lst = game_segments[i].cot_prefix_segment[beg_index:end_index]
        pad_llm_action_lst = game_segments[i].llm_action_segment[beg_index:end_index]

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

        # Pad and finalize
        if self.policy_config.gumbel_algo:
            last_game_segments[i].pad_over(
                pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                next_segment_improved_policy=pad_improved_policy_prob,
                next_segment_cot_prefix=pad_cot_prefix_lst,
                next_segment_llm_action=pad_llm_action_lst
            )
        else:
            if self.policy_config.use_ture_chance_label_in_chance_encoder:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                    next_chances=chance_lst, next_segment_raw_obs=pad_raw_obs_lst,
                    next_segment_history_obs=pad_history_obs_lst, next_segment_llm_prior_per_tok=pad_llm_prior_per_tok_lst,
                    next_segment_cot_prefix=pad_cot_prefix_lst,
                    next_segment_llm_action=pad_llm_action_lst
                )
            else:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst,
                    next_segment_raw_obs=pad_raw_obs_lst, next_segment_history_obs=pad_history_obs_lst,
                    next_segment_llm_prior_per_tok=pad_llm_prior_per_tok_lst,
                    next_segment_cot_prefix=pad_cot_prefix_lst,
                    next_segment_llm_action=pad_llm_action_lst
                )

        last_game_segments[i].game_segment_to_array()
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done))

        last_game_segments[i] = None
        last_game_priorities[i] = None

    def _compute_priorities(self, game_segment: GameSegment) -> np.ndarray:
        """Compute priorities for the game segment."""
        # Simple priority: uniform for now
        return np.ones(len(game_segment.reward_segment))

    def apply_temperature_scaling(self, prior: np.ndarray, return_logprobs: bool = False) -> np.ndarray:
        """Apply temperature scaling to prior distribution."""
        if return_logprobs:
            # Convert to log probabilities
            log_probs = np.log(prior + 1e-10)
            return log_probs
        else:
            return prior

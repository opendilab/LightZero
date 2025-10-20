# priorzero_collector.py
"""
[PRIORZERO] PriorZero Collector Implementation

This module implements async data collection with LLM prior integration.

Key Features:
- Async LLM inference using vLLM for efficient batch generation
- History buffer management for context-aware prompting
- Error handling and retry logic for robust LLM calls
- Full alignment with UniZero collector architecture

Author: PriorZero Team
Date: 2025-01-20
"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from typing import Optional, Any, List, Dict, Tuple

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from vllm import AsyncLLMEngine, SamplingParams

# Import from LightZero
from lzero.worker.muzero_segment_collector import MuZeroSegmentCollector as OriginalCollector
from lzero.mcts.utils import prepare_observation
from game_segment_priorzero import GameSegment


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
    # Try to get 'raw_obs' field first (Jericho-style)
    if 'raw_obs' in obs_dict:
        return str(obs_dict['raw_obs'])

    # Try to get 'text' field
    if 'text' in obs_dict:
        return str(obs_dict['text'])

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

@SERIAL_COLLECTOR_REGISTRY.register('priorzero_segment')
class PriorZeroCollector(OriginalCollector):
    """
    [PRIORZERO-MODIFIED]
    Async collector that integrates LLM priors into MCTS-based data collection.

    Features:
    - Async LLM inference with vLLM engine
    - History buffer for each environment (sliding window)
    - Robust error handling with retries
    - Detailed logging of LLM prior statistics
    """

    def __init__(
        self,
        vllm_engine: AsyncLLMEngine,
        policy_config: Dict,
        **kwargs
    ):
        """
        Initialize PriorZeroCollector.

        Args:
            vllm_engine: vLLM async engine for LLM inference
            policy_config: Policy configuration (contains llm_policy_cfg)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)

        self.vllm_engine = vllm_engine
        self.policy_config = policy_config
        self.llm_policy_cfg = policy_config.llm_policy_cfg

        # [PRIORZERO-NEW] History buffer for each environment
        # Format: {env_id: deque([(obs_text, action_text, reward), ...])}
        self.history_buffers = defaultdict(
            lambda: deque(maxlen=self.llm_policy_cfg.history_length)
        )

        # [PRIORZERO-NEW] Statistics for logging
        self.llm_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_count': 0,
            'total_latency': 0.0,
            'llm_prior_top1_match_count': 0,  # How often LLM top-1 matches MCTS choice
        }

        self._logger.info("✓ PriorZeroCollector initialized with vLLM engine")
        self._logger.info(f"  - History length: {self.llm_policy_cfg.history_length}")
        self._logger.info(f"  - Generate max length: {self.llm_policy_cfg.generate_max_len}")

        # [PRIORZERO-NEW] Use custom GameSegment
        self.GameSegment = GameSegment

    async def _async_get_llm_prior(
        self,
        states: List[str],
        request_ids: List[str],
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> List[Any]:
        """
        [PRIORZERO-NEW]
        Async call to LLM to get action ranking priors.

        Args:
            states: List of current observation texts
            request_ids: List of unique request IDs for tracking
            histories: Optional list of history tuples for each state
            max_retries: Maximum number of retries on failure
            timeout: Timeout in seconds for each request

        Returns:
            llm_outputs: List of vLLM output objects
        """
        from priorzero_policy import build_llm_prompt

        # Build prompts
        prompts = []
        for i, state in enumerate(states):
            history = histories[i] if histories is not None else None

            # Build instruction using the helper function from policy
            instruction = build_llm_prompt(
                current_obs=state,
                history=history,
                use_cot=self.llm_policy_cfg.use_cot
            )

            # Apply chat template if policy has tokenizer
            if hasattr(self._policy, 'llm_tokenizer'):
                prompt = self._policy.llm_tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = instruction

            prompts.append(prompt)

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.llm_policy_cfg.generate_max_len,
            skip_special_tokens=False,
        )

        # Retry logic
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Async generation
                results_generator = self.vllm_engine.generate(
                    prompts,
                    sampling_params,
                    request_ids
                )

                # Collect results
                llm_outputs = [None] * len(prompts)

                try:
                    async for result in asyncio.wait_for(
                        results_generator,
                        timeout=timeout
                    ):
                        # Parse request_id to get original index
                        # Format: "collect_{train_iter}_{env_idx}"
                        original_index = int(result.request_id.split('_')[-1])
                        llm_outputs[original_index] = result

                except asyncio.TimeoutError:
                    self._logger.warning(f"⚠ LLM generation timeout after {timeout}s (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        self.llm_stats['retry_count'] += 1
                        continue
                    else:
                        # On final timeout, return None for all
                        self.llm_stats['failed_calls'] += len(prompts)
                        return [None] * len(prompts)

                # Check if all outputs were received
                if None in llm_outputs:
                    missing_count = llm_outputs.count(None)
                    self._logger.warning(f"⚠ {missing_count}/{len(prompts)} LLM outputs missing (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        self.llm_stats['retry_count'] += 1
                        continue

                # Success
                elapsed = time.time() - start_time
                self.llm_stats['total_calls'] += len(prompts)
                self.llm_stats['successful_calls'] += len([o for o in llm_outputs if o is not None])
                self.llm_stats['failed_calls'] += len([o for o in llm_outputs if o is None])
                self.llm_stats['total_latency'] += elapsed

                self._logger.debug(f"✓ LLM generation completed in {elapsed:.2f}s ({len(prompts)} prompts)")

                return llm_outputs

            except Exception as e:
                self._logger.error(f"✗ LLM generation error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.llm_stats['retry_count'] += 1
                    await asyncio.sleep(0.5)  # Brief pause before retry
                else:
                    # Final failure
                    self.llm_stats['failed_calls'] += len(prompts)
                    return [None] * len(prompts)

        return [None] * len(prompts)

    async def collect(
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
        2. Async call to LLM to get action priors
        3. Pass LLM priors to policy forward pass
        4. Update history buffers after each step

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

        # ==================================================================
        # Initialization
        # ==================================================================
        collected_episode = 0
        collected_step = 0
        env_nums = self._env_num
        init_obs = self._env.ready_obs

        # Wait for all environments to be ready
        retry_waiting_time = 0.05
        while len(init_obs.keys()) != env_nums:
            self._logger.info(f'Waiting for all environments to reset. Ready: {list(init_obs.keys())}')
            time.sleep(retry_waiting_time)
            init_obs = self._env.ready_obs

        # Initialize state tracking
        for env_id in range(env_nums):
            if env_id in init_obs:
                self.action_mask_dict[env_id] = to_ndarray(init_obs[env_id]['action_mask'])
                self.to_play_dict[env_id] = to_ndarray(init_obs[env_id]['to_play'])
                self.timestep_dict[env_id] = to_ndarray(init_obs[env_id].get('timestep', -1))

        # Initialize game segments
        game_segments = [
            GameSegment(
                self._env.action_space,
                game_segment_length=self.policy_config.game_segment_length,
                config=self.policy_config,
                task_id=self.task_id
            ) for _ in range(env_nums)
        ]

        # Initialize observation stacks
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
            game_segments[env_id].reset(observation_window_stack[env_id])

        # Priority calculation lists
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        # Logging variables
        eps_steps_lst = np.zeros(env_nums)
        visit_entropies_lst = np.zeros(env_nums)

        if collect_with_pure_policy:
            temp_visit_list = [0.0 for _ in range(self._env.action_space.n)]

        # ==================================================================
        # Main Collection Loop
        # ==================================================================
        while True:
            with self._timer:
                # Get ready environments
                obs = self._env.ready_obs
                ready_env_id = set(obs.keys())

                if len(ready_env_id) < self._env_num:
                    self._logger.debug(f'Only {len(ready_env_id)}/{self._env_num} envs ready')

                # Prepare stacked observations for world model
                stack_obs_dict = {
                    env_id: game_segments[env_id].get_obs()
                    for env_id in ready_env_id
                }
                stack_obs_list = [stack_obs_dict[env_id] for env_id in sorted(list(ready_env_id))]

                # Prepare action masks and other info
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
                if not collect_with_pure_policy:
                    # Extract text observations
                    raw_obs_list = []
                    histories_list = []
                    for env_id in sorted(list(ready_env_id)):
                        # Extract raw text
                        raw_obs_text = extract_raw_obs_text(obs[env_id])
                        raw_obs_list.append(raw_obs_text)

                        # Get history for this environment
                        history = list(self.history_buffers[env_id])
                        histories_list.append(history)

                    # Generate request IDs
                    request_ids = [
                        f"collect_{train_iter}_{i}"
                        for i in range(len(raw_obs_list))
                    ]

                    # Async call to LLM
                    llm_outputs = await self._async_get_llm_prior(
                        raw_obs_list,
                        request_ids,
                        histories_list
                    )

                    # Add to policy kwargs
                    policy_kwargs['llm_prior_outputs'] = llm_outputs
                else:
                    policy_kwargs['llm_prior_outputs'] = None

                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_args = (stack_obs_tensor, action_mask, temperature, to_play, epsilon)
                policy_kwargs_forward = {
                    'ready_env_id': sorted(list(ready_env_id)),
                    'timestep': timestep,
                    'llm_prior_outputs': policy_kwargs.get('llm_prior_outputs')
                }

                if self.task_id is not None:
                    policy_kwargs_forward['task_id'] = self.task_id

                policy_output = self._policy.forward(*policy_args, **policy_kwargs_forward)

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

                    # Store search statistics
                    if collect_with_pure_policy:
                        game_segments[env_id].store_search_stats(temp_visit_list, 0)
                    else:
                        game_segments[env_id].store_search_stats(
                            distributions_dict_with_env_id[env_id],
                            value_dict_with_env_id[env_id]
                        )

                    # Append transition to game segment
                    game_segments[env_id].append(
                        actions[env_id],
                        to_ndarray(obs_new['observation']),
                        reward,
                        self.action_mask_dict[env_id],
                        self.to_play_dict[env_id],
                        timestep=to_ndarray(obs_new.get('timestep', -1))
                    )

                    # ===========================================================
                    # [PRIORZERO-NEW] Update History Buffer
                    # ===========================================================
                    raw_obs_text = extract_raw_obs_text(obs[env_id])
                    action_text = getattr(self._policy, 'action_inv_map', {}).get(
                        actions[env_id],
                        f"action_{actions[env_id]}"
                    )
                    self.history_buffers[env_id].append((raw_obs_text, action_text, float(reward)))

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
                        if self.last_game_segments[env_id] is not None:
                            self.pad_and_save_last_trajectory(
                                env_id,
                                self.last_game_segments,
                                self.last_game_priorities,
                                game_segments,
                                self.dones
                            )

                        # Calculate priorities
                        priorities = self._compute_priorities(
                            env_id,
                            pred_values_lst,
                            search_values_lst
                        )
                        pred_values_lst[env_id], search_values_lst[env_id] = [], []

                        # Save segment
                        self.last_game_segments[env_id] = game_segments[env_id]
                        self.last_game_priorities[env_id] = priorities

                        # Create new segment
                        game_segments[env_id] = GameSegment(
                            self._env.action_space,
                            game_segment_length=self.policy_config.game_segment_length,
                            config=self.policy_config,
                            task_id=self.task_id
                        )
                        game_segments[env_id].reset(observation_window_stack[env_id])

                    self._env_info[env_id]['step'] += 1
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
                    }
                    if not collect_with_pure_policy:
                        info_log['visit_entropy'] = (
                            visit_entropies_lst[env_id] / eps_steps_lst[env_id]
                            if eps_steps_lst[env_id] > 0 else 0
                        )

                    collected_episode += 1
                    self._episode_info.append(info_log)

                    # Save remaining segments
                    if self.last_game_segments[env_id] is not None:
                        self.pad_and_save_last_trajectory(
                            env_id,
                            self.last_game_segments,
                            self.last_game_priorities,
                            game_segments,
                            self.dones
                        )

                    priorities = self._compute_priorities(
                        env_id,
                        pred_values_lst,
                        search_values_lst
                    )

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
                    game_segments[env_id] = GameSegment(
                        self._env.action_space,
                        game_segment_length=self.policy_config.game_segment_length,
                        config=self.policy_config,
                        task_id=self.task_id
                    )
                    game_segments[env_id].reset(observation_window_stack[env_id])

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

        self._total_envstep_count += collected_step
        self._total_episode_count += collected_episode
        self._total_duration += collected_duration

        self._output_log(train_iter)

        # [PRIORZERO-NEW] Log LLM statistics
        if self.llm_stats['total_calls'] > 0:
            avg_latency = self.llm_stats['total_latency'] / self.llm_stats['total_calls']
            success_rate = self.llm_stats['successful_calls'] / self.llm_stats['total_calls']

            self._logger.info(
                f"📊 LLM Prior Statistics:\n"
                f"  - Total calls: {self.llm_stats['total_calls']}\n"
                f"  - Success rate: {success_rate*100:.1f}%\n"
                f"  - Avg latency: {avg_latency:.3f}s\n"
                f"  - Retry count: {self.llm_stats['retry_count']}"
            )

        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        [INHERITED]
        Log collection statistics (inherited from parent).
        """
        super()._output_log(train_iter)

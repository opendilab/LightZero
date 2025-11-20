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
import sys
import time
import cProfile
from contextlib import contextmanager
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple

# [CRITICAL] Ensure local LightZero is used
from ensure_local_lightzero import ensure_local_lightzero
ensure_local_lightzero()

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from vllm import AsyncLLMEngine, SamplingParams

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
        # [FIX] Set policy_config in kwargs before calling super().__init__
        # because parent class needs it
        kwargs['policy_config'] = policy_config

        # Extract debug_mode before passing to parent (parent doesn't accept this parameter)
        self.debug_mode = kwargs.pop('debug_mode', False)

        super().__init__(**kwargs)

        self.vllm_engine = vllm_engine
        self._vllm_tokenizer = None
        # self.policy_config already set by parent class from kwargs
        self.llm_policy_cfg = policy_config.llm_policy_cfg

        # [PRIORZERO-NEW] History buffer for each environment
        # Format: {env_id: deque([(obs_text, action_text, reward), ...])}
        self.history_buffers = defaultdict(
            lambda: deque(maxlen=self.llm_policy_cfg.history_length)
        )
        self.prompt_log_interval = getattr(self.llm_policy_cfg, 'prompt_log_interval', 0)
        self._last_prompt_log_step = 0

        self.profile_cfg = getattr(self.policy_config, 'profile_cfg', {})
        self._profile_enabled = bool(self.profile_cfg.get('enable_cprofile', False))
        self._profile_log_interval = int(self.profile_cfg.get('log_interval', 50))
        self._profile_dir = Path(self.profile_cfg.get('output_dir', f"./{self._exp_name}/log/profile"))
        
        self._profile_stats: Dict[str, Dict[str, float]] = {}
        self._profile_stats_file = self._profile_dir / "collector_time.log"
        if self._profile_enabled:
            self._profile_dir.mkdir(parents=True, exist_ok=True)

        # Where to persist sampled LLM outputs during collect
        self._llm_output_log_path = Path(f"./{self._exp_name}/log/collector/llm_output.log")
        self._llm_output_log_path.parent.mkdir(parents=True, exist_ok=True)

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
                    next_chances=chance_lst
                )
            else:
                last_game_segments[i].pad_over(
                    pad_obs_lst, pad_reward_lst, pad_action_lst, pad_root_values_lst, pad_child_visits_lst, 
                    next_segment_raw_obs=pad_raw_obs_lst, next_segment_history_obs=pad_history_obs_lst
                )

        last_game_segments[i].game_segment_to_array()

        # Add the completed game segment to the pool.
        self.game_segment_pool.append((last_game_segments[i], last_game_priorities[i], done[i]))

        # Reset placeholders for the next collection cycle.
        last_game_segments[i] = None
        last_game_priorities[i] = None
    
    async def _get_tokenizer(self):
        """
        从 vLLM 引擎获取已加载的 tokenizer 引用。
        只在第一次调用时会有极小的 async 开销，之后直接返回内存引用。
        """
        if self._vllm_tokenizer is None:
            self._vllm_tokenizer = await self.vllm_engine.get_tokenizer()
        return self._vllm_tokenizer
    
    async def _async_get_llm_prior(
        self,
        states: List[str],
        request_ids: List[str],
        valid_actions_list: List[List[str]], 
        histories: Optional[List[List[Tuple[str, str, float]]]] = None,
        timeout: float = 30.0
    ) -> List[Any]:
        """
        [PRIORZERO-SEQUENCE-SCORING]
        Async call to calculate the log-probability of full action sequences.
        
        Method:
        Constructs "Context + Action" for every valid action, feeds it to vLLM with 
        prompt_logprobs=1, and sums the log-probs of the action tokens.
        
        Args:
            states: List of observation texts.
            request_ids: IDs for the request batch.
            valid_actions_list: List of valid actions for each env.
            
        Returns:
            prior_results: List of dicts {action_str: total_logprob}.
        """
        
        

        # 1. Check Engine Availability & Get Tokenizer
        assert self.vllm_engine is not None, "vLLM engine is not initialized."
        tokenizer = await self._get_tokenizer()
        
        # 2. Prepare Flattened Prompt Data (Env x Actions)
        all_prompts_data = []
        for i, state in enumerate(states):
            history = histories[i]
            instruction = build_llm_prompt(
                current_obs=state,
                history=history,
                use_cot=self.llm_policy_cfg.use_cot
            )
            context_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            context_tokens = tokenizer.encode(context_text)
            context_len = len(context_tokens)
            
            actions = valid_actions_list[i]
            
            for act_idx, action in enumerate(actions):
                # 我们构造成模型应该生成的完整格式: "<answer>Turn Left</answer>"
                formatted_action = f"<answer>{action}</answer>"
                
                # 拼接 Full Text
                # Context: "... Assistant:" # Target:  "<answer>Turn Left</answer>" # Result:  "... Assistant:<answer>Turn Left</answer>"
                full_text = context_text + formatted_action
                unique_req_id = f"{request_ids[i]}_act_{act_idx}"
                all_prompts_data.append({
                    "idx": i,
                    "action_str": action, 
                    "full_text": full_text,
                    "context_len": context_len,
                    "req_id": unique_req_id
                })
            
        # 3. Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            prompt_logprobs=1, 
        )
        
        # 4. 定义单个请求的处理函数 (逻辑解耦)
        async def get_sequence_score(item):
            # vLLM 的 generate 返回一个 async iterator
            results_generator = self.vllm_engine.generate(item["full_text"], sampling_params, item["req_id"])
            final_output = None
            # 使用 asyncio.wait_for 自动处理超时
            async for request_output in results_generator:
                final_output = request_output

            # 5. Extract & Sum Logprobs: 从 Context 结束的位置开始，提取后面所有 Token (即 <answer>...</answer>) 的分数
            action_logprobs_list = final_output.prompt_logprobs[item["context_len"]:]
            total_score, valid_tokens = 0.0, 0
            for token_dict in action_logprobs_list:
                if token_dict:
                    for lp_obj in token_dict.values():
                        total_score += lp_obj.logprob
                        valid_tokens += 1
                        break 
            return item["idx"], item["action_str"], total_score
            

        # 6. 并发执行所有请求
        # 使用 wait_for 在最外层控制整体超时，避免死等
        try:
            tasks = [get_sequence_score(item) for item in all_prompts_data]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except Exception as e:
            self._logger.error(f"Batch LLM critical error: {e}")
            return [{}] * len(states)
            
        final_priors = [{} for _ in range(len(states))]
        for i, action_str, score in results:
            final_priors[i][action_str] = score
        return final_priors

    @contextmanager
    def _profile_block(self, name: str):
        if not self._profile_enabled:
            yield None
            return
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        start_time = time.perf_counter()
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            elapsed = time.perf_counter() - start_time
            self._record_profile_time(name, elapsed)
            # No per-iteration .prof dumps; we only aggregate to the log file.

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        log_every = max(1, self._profile_log_interval)
        stats = self._profile_stats.setdefault(name, {'count': 0, 'total': 0.0, 'max': 0.0})
        stats['count'] += 1
        stats['total'] += elapsed
        stats['max'] = max(stats['max'], elapsed)
        if stats['count'] % log_every == 0:
            avg = stats['total'] / stats['count']
            self._profile_dir.mkdir(parents=True, exist_ok=True)
            with self._profile_stats_file.open("a") as f:
                f.write(
                    f"{time.time():.3f}\t{name}\tcount={stats['count']}\t"
                    f"total_s={stats['total']:.4f}\tavg_s={avg:.6f}\tmax_s={stats['max']:.6f}\n"
                )
            self._logger.info(
                f"[cprofile][agg] {name}: count={stats['count']} total={stats['total']:.2f}s "
                f"avg={avg:.4f}s max={stats['max']:.4f}s"
            )

    async def _log_llm_response(
        self,
        raw_obs_text: str,
        history: List[Tuple[str, str, float]],
        valid_actions: List[str],
        train_iter: int,
        collected_step: int,
    ) -> None:
        """
        Periodically log LLM output for a debug prompt and current valid actions.
        """
        if self.prompt_log_interval <= 0:
            return
        if (collected_step - self._last_prompt_log_step) < self.prompt_log_interval:
            return
        self._last_prompt_log_step = collected_step
        tokenizer = await self._get_tokenizer()
        instruction = build_llm_prompt(
            current_obs=raw_obs_text,
            history=history,
            use_cot=self.llm_policy_cfg.use_cot,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.llm_policy_cfg.generate_max_len,
            top_p=1.0,
        )
        request_id = f"debug_prompt_{train_iter}_{collected_step}"
        try:
            result_gen = self.vllm_engine.generate(
                prompt_text,
                sampling_params,
                request_id=request_id,
            )
            async for request_output in result_gen:
                if request_output.finished:
                    llm_output_text = request_output.outputs[0].text or ""
                    break
        except Exception as e:
            llm_output_text = f"[LLM logging error: {repr(e)}]"         
        llm_output_text = llm_output_text.strip()
        # Truncate for logging
        with self._llm_output_log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"iter={train_iter}\tstep={collected_step}\t"
                f"valid_actions={valid_actions}\n"
                f"llm_input_output={llm_output_text}\n"
                "----\n"
            )

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

        last_game_segments = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]
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
            game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(init_obs[env_id]), init_history_obs=list(self.history_buffers[env_id]))

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
                if collect_with_pure_policy:
                    continue
                else:
                    # Extract text observations and valid actions
                    raw_obs_list = []
                    histories_list = []
                    valid_actions_list = []  # [PRIORZERO] Store valid actions for each env
                    for env_id in sorted(list(ready_env_id)):
                        # Extract raw text
                        raw_obs_text = extract_raw_obs_text(obs[env_id])
                        raw_obs_list.append(raw_obs_text)

                        # Get history for this environment
                        history = list(self.history_buffers[env_id])
                        histories_list.append(history)

                        # [PRIORZERO] Extract valid actions from observation
                        valid_actions = obs[env_id].get('valid_actions', [])
                        valid_actions_list.append(valid_actions)

                    # Generate request IDs
                    request_ids = [
                        f"collect_{train_iter}_{i}"
                        for i in range(len(raw_obs_list))
                    ]

                    # Async call to LLM debug
                    profile_name = f"collect_llm_prior_iter{train_iter}_step{collected_step}"
                    with self._profile_block(profile_name):
                        llm_prior_logprob = await self._async_get_llm_prior(
                            states=raw_obs_list,
                            request_ids=request_ids,
                            valid_actions_list=valid_actions_list,  # [PRIORZERO] Pass valid actions
                            histories=histories_list
                        )
                    if raw_obs_list:
                        await self._log_llm_response(
                            raw_obs_text=raw_obs_list[0],
                            history=histories_list[0],
                            valid_actions=valid_actions_list[0],
                            train_iter=train_iter,
                            collected_step=collected_step,
                        )
                    # llm_prior_logprob = []
                    # for i, actions in enumerate(valid_actions_list):
                    #     tmp_dict = {}
                    #     for action in actions:
                    #         tmp_dict[action] = -3  # Placeholder zero logprob
                    #     llm_prior_logprob.append(tmp_dict)
                    

                # ==============================================================
                # Policy Forward Pass
                # ==============================================================
                policy_args = (stack_obs_tensor, action_mask, temperature, to_play, epsilon)
                policy_kwargs_forward = {
                    'ready_env_id': sorted(list(ready_env_id)),
                    'timestep': timestep,
                    'llm_prior_logprob': llm_prior_logprob,
                    'valid_actions_list': valid_actions_list
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
                with self._profile_block(f"collect_env_step_iter{train_iter}_envstep{self._total_envstep_count}"):
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
                    action = valid_actions_list[env_id][actions[env_id]]
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
                        history_obs=list(self.history_buffers[env_id])
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
                        game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(obs_new), init_history_obs=list(self.history_buffers[env_id]))

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
                        'step': self._env_info[env_id]['step']}
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
                    game_segments[env_id].reset(observation_window_stack[env_id], init_raw_obs=extract_raw_obs_text(init_obs[env_id]), init_history_obs=list(self.history_buffers[env_id]))
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
        super()._output_log(train_iter)
    
